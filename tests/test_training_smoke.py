"""The two-step closed loop: Stage 1, Stage 2, Stage 3 and the whole protocol.

The loop here is the smallest one that exercises data reading, a forward pass, the
loss, a backward pass and a parameter update, so a failure is a statement about the
training path rather than about a metric. The end-to-end test runs the short-budget
protocol from ``protocols/_smoke.yaml``, which is labelled for unit-test smoke only
and is never used for a reported value.

Ref: Eq. (3)-(5), Algorithm 1, Sec. 2.6.
"""

from __future__ import annotations

import math

import pytest
import torch

from oaer.catalogue.concepts import build_catalogue
from oaer.catalogue.graph import build_knowledge_graph
from oaer.catalogue.sampling import corrupt_anchors, sample_masked_edges
from oaer.cohort.builder import build_cohort
from oaer.encoder.relational import RelationalEncoder, masked_edge_loss
from oaer.fitting.checkpoint import (
    CheckpointPayload,
    load_state,
    save_state,
    torch_state_equal,
)
from oaer.fitting.loop import (
    Trainer,
    TrainingConfig,
    iterate_batches,
    loss_decreased,
    parameter_delta,
    snapshot,
)
from oaer.fitting.optim import (
    ScheduleConfig,
    WarmupCosine,
    build_optimiser,
    clip_gradients,
    learning_rate_at,
)
from oaer.fitting.stages import fit_graph_stages, graph_tensors
from oaer.study import (
    assemble_study,
    attach_patients,
    run_study,
    training_config,
)
from oaer.support.seeding import set_seed


def _small_graph(seed: int = 7):  # type: ignore[no-untyped-def]
    catalogue = build_catalogue(size=8, seed=seed)
    return build_knowledge_graph(catalogue, seed=seed)


def _encoder(graph, *, width: int = 16) -> RelationalEncoder:  # type: ignore[no-untyped-def]
    from oaer.catalogue.graph import RELATION_INDEX

    return RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=len(RELATION_INDEX),
        embedding_dim=width,
        hidden_dim=width,
        layers=1,
        num_basis=2,
        dropout=0.0,
    )


class TestSchedule:
    def test_the_rate_warms_up_then_decays(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimiser = build_optimiser([parameter], 0.1, 0.0)
        schedule = WarmupCosine(optimiser, 0.1, ScheduleConfig(warmup_steps=2, total_steps=10))
        rates = [schedule.current()] + [schedule.advance() for _ in range(9)]
        assert rates[0] < rates[1]
        assert rates[1] == pytest.approx(0.1)
        assert rates[-1] < rates[3]

    def test_the_rate_never_goes_negative(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimiser = build_optimiser([parameter], 0.1, 0.0)
        schedule = WarmupCosine(optimiser, 0.1, ScheduleConfig(warmup_steps=1, total_steps=200))
        assert all(schedule.advance() >= 0.0 for _ in range(200))

    def test_the_rate_decays_to_the_floor_and_stops(self) -> None:
        config = ScheduleConfig(warmup_steps=1, total_steps=10, floor=0.02)
        assert learning_rate_at(10, 0.1, config) == pytest.approx(0.002)
        assert learning_rate_at(400, 0.1, config) == pytest.approx(0.002)

    def test_a_negative_step_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            learning_rate_at(-1, 0.1, ScheduleConfig())

    def test_a_negative_warm_up_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScheduleConfig(warmup_steps=-1, total_steps=10)

    def test_clipping_returns_the_norm_before_clipping(self) -> None:
        model = torch.nn.Linear(2, 1)
        loss = model(torch.ones(4, 2)).sum()
        loss.backward()
        norm = clip_gradients(model, 1e-6)
        assert norm >= 0.0

    def test_clipping_without_a_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            clip_gradients(torch.nn.Linear(1, 1), 0.0)


class TestBatching:
    def test_every_row_is_visited_once_in_an_epoch(self) -> None:
        seen = [int(index) for batch in iterate_batches(10, 4, seed=1, epoch=0) for index in batch]
        assert sorted(seen) == list(range(10))

    def test_a_different_epoch_permutes_differently(self) -> None:
        first = [
            int(index) for batch in iterate_batches(50, 10, seed=1, epoch=0) for index in batch
        ]
        second = [
            int(index) for batch in iterate_batches(50, 10, seed=1, epoch=1) for index in batch
        ]
        assert first != second
        assert sorted(first) == sorted(second)


class TestTwoStepLoop:
    def test_two_steps_update_the_parameters_and_lower_the_loss(self) -> None:
        set_seed(7)
        graph = _small_graph()
        encoder = _encoder(graph)
        tensors = graph_tensors(graph)
        batch = sample_masked_edges(graph, 16, negatives=4, seed=7)
        head = torch.as_tensor(batch.head, dtype=torch.long)
        tail = torch.as_tensor(batch.tail, dtype=torch.long)
        negative = torch.as_tensor(batch.negative, dtype=torch.long)

        def step(model: torch.nn.Module, index: torch.Tensor) -> torch.Tensor:
            state = model(
                tensors["entity"],
                tensors["head"],
                tensors["relation"],
                tensors["tail"],
                tensors["degree"],
            )
            positive = (state[head[index]] * state[tail[index]]).sum(dim=-1) / (
                state.shape[1] ** 0.5
            )
            negatives = torch.einsum("bw,bkw->bk", state[head[index]], state[negative[index]]) / (
                state.shape[1] ** 0.5
            )
            return masked_edge_loss(positive, negatives)

        before = snapshot(encoder)
        config = TrainingConfig(
            epochs=2, batch_size=8, learning_rate=5e-2, warmup_steps=1, total_steps=4, seed=7
        )
        run = Trainer(encoder, config, step, batch_size=8).fit(int(head.shape[0]))
        # Two epochs over sixteen rows at a batch of eight is four steps.
        assert len(run.records) == 4
        assert parameter_delta(before, snapshot(encoder)) > 0.0
        assert loss_decreased(run.records)

    def test_the_loss_is_finite_from_the_first_step(self) -> None:
        set_seed(11)
        graph = _small_graph(11)
        encoder = _encoder(graph)
        tensors = graph_tensors(graph)
        batch = sample_masked_edges(graph, 16, negatives=4, seed=11)
        head = torch.as_tensor(batch.head, dtype=torch.long)
        tail = torch.as_tensor(batch.tail, dtype=torch.long)
        negative = torch.as_tensor(batch.negative, dtype=torch.long)

        def step(model: torch.nn.Module, index: torch.Tensor) -> torch.Tensor:
            state = model(
                tensors["entity"],
                tensors["head"],
                tensors["relation"],
                tensors["tail"],
                tensors["degree"],
            )
            positive = (state[head[index]] * state[tail[index]]).sum(dim=-1)
            negatives = torch.einsum("bw,bkw->bk", state[head[index]], state[negative[index]])
            return masked_edge_loss(positive, negatives)

        config = TrainingConfig(
            epochs=1, batch_size=16, learning_rate=1e-3, warmup_steps=1, total_steps=1, seed=11
        )
        run = Trainer(encoder, config, step, batch_size=16).fit(int(head.shape[0]))
        assert math.isfinite(run.losses[0])


class TestGraphPretrainingStages:
    def test_stage_one_and_two_report_their_own_loss(self) -> None:
        set_seed(13)
        graph = _small_graph(13)
        names = [entry.label for entry in build_catalogue(size=8, seed=13)]
        batch, _ = build_cohort(seed=13, candidates=names)
        patients = attach_patients(graph, batch, limit=16)
        records = corrupt_anchors(graph, patients, negatives=3, seed=13)
        masked = sample_masked_edges(graph, 16, negatives=4, seed=13)
        config = TrainingConfig(
            epochs=2, batch_size=16, learning_rate=1e-2, warmup_steps=1, total_steps=8, seed=13
        )
        bundle = fit_graph_stages(
            graph,
            masked,
            records,
            embedding_dim=16,
            hidden_dim=16,
            layers=1,
            num_basis=2,
            dropout=0.0,
            training=config,
        )
        summary = bundle.summary()
        assert float(summary["stage1"]["steps"]) > 0.0
        assert math.isfinite(float(summary["stage1"]["trailing_mean"]))
        assert math.isfinite(float(summary["stage2"]["trailing_mean"]))

    def test_the_training_config_matches_the_protocol_block(self, smoke_config) -> None:  # type: ignore[no-untyped-def]
        config = training_config(smoke_config, stage="stage3")
        assert config.epochs == smoke_config.fitting.stage3_epochs
        assert config.batch_size == smoke_config.fitting.batch_size
        assert config.learning_rate == pytest.approx(smoke_config.fitting.learning_rate)


class TestCheckpointRoundTrip:
    def test_a_saved_state_restores_identical_parameters_and_its_seed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        set_seed(17)
        graph = _small_graph(17)
        encoder = _encoder(graph, width=12)
        reference = _encoder(graph, width=12)
        path = tmp_path / "checkpoint.pt"
        save_state(
            path,
            CheckpointPayload(stage="stage2", epoch=3, step=12, seed=17, metrics={"loss": 0.5}),
            model=encoder,
        )
        payload = load_state(path, model=reference, restore_rng=False)
        assert torch_state_equal(encoder, reference)
        assert payload.step == 12
        assert payload.seed == 17

    def test_the_checkpoint_is_world_readable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        set_seed(19)
        encoder = _encoder(_small_graph(19), width=8)
        path = tmp_path / "checkpoint.pt"
        save_state(
            path,
            CheckpointPayload(stage="stage1", epoch=0, step=1, seed=19, metrics={}),
            model=encoder,
        )
        mode = path.stat().st_mode & 0o777
        assert mode & 0o044 == 0o044


class TestProtocolEndToEnd:
    def test_the_smoke_protocol_holds_its_marginals_and_trains(self, smoke_config) -> None:  # type: ignore[no-untyped-def]
        result = run_study(smoke_config)
        assert result.marginal_match
        assert all(result.marginal_match.values())
        assert result.training_records["steps"] > 0.0
        assert result.policy_value["decisions"] > 0.0

    def test_the_assembled_smoke_study_partitions_its_records(self, smoke_config) -> None:  # type: ignore[no-untyped-def]
        inputs = assemble_study(smoke_config)
        assert len(inputs.development) + len(inputs.held_out) == len(inputs.retrospective)
        folds = set(inputs.folds.assignment.values())
        assert folds and folds <= set(range(smoke_config.emulation.folds))

    def test_every_held_out_record_carries_the_held_out_site(self, smoke_config) -> None:  # type: ignore[no-untyped-def]
        inputs = assemble_study(smoke_config)
        assert {decision.site for decision in inputs.held_out.decisions} == {
            smoke_config.site_holdout
        }
