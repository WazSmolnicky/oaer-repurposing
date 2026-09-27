"""The three pretraining stages of Algorithm 1.

Stage 1 fits the relational encoder on masked knowledge-graph edges with no
patient data reachable. Stage 2 removes the masked anchor edges from the graph and
reconstructs them against non-anchors of the same type, which is the only stage at
which the graph sees a patient. Stage 3 is where outcomes enter: the ranking head
is fitted by weighted least squares on the out-of-fold pseudo-outcomes, with early
stopping taken on an inner split of the training folds and never on the reported
out-of-fold set. Nothing from Stage 3 reaches the encoder.

Ref: Sec. 2.6, Eq. (3)-(5), Algorithm 1 lines 1-9.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from oaer.catalogue.graph import ANCHOR_RELATION, AnchorType, KnowledgeGraph
from oaer.catalogue.sampling import AnchorCorruption, MaskedEdgeBatch
from oaer.encoder.anchoring import (
    AnchorReconstruction,
    PatientAnchorHead,
    anchor_reconstruction_loss,
    anchor_type_index,
)
from oaer.encoder.propagation import degree_tensor
from oaer.encoder.relational import RelationalEncoder, masked_edge_loss
from oaer.fitting.loop import Trainer, TrainingConfig, TrainingRun
from oaer.support.seeding import set_seed


@dataclass(frozen=True)
class StageReport:
    """What one stage produced, for the run's own record."""

    stage: str
    steps: int
    first_loss: float
    last_loss: float
    trailing_mean: float
    leading_mean: float
    parameters: int

    def as_dict(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "steps": self.steps,
            "first_loss": round(self.first_loss, 8),
            "last_loss": round(self.last_loss, 8),
            "leading_mean": round(self.leading_mean, 8),
            "trailing_mean": round(self.trailing_mean, 8),
            "parameters": self.parameters,
        }


@dataclass
class StageBundle:
    """The fitted objects the inference path needs."""

    encoder: RelationalEncoder
    anchor_head: PatientAnchorHead
    anchor_readout: AnchorReconstruction
    reports: list[StageReport] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float | str]]:
        return {report.stage: report.as_dict() for report in self.reports}


def graph_tensors(graph: KnowledgeGraph, device: str = "cpu") -> dict[str, Tensor]:
    """Index arrays and degrees for one forward pass over the graph."""

    head, relation, tail = graph.arrays()
    num_entities = len(graph.nodes)
    return {
        "entity": torch.arange(num_entities, dtype=torch.long, device=device),
        "head": torch.as_tensor(head, dtype=torch.long, device=device),
        "relation": torch.as_tensor(relation, dtype=torch.long, device=device),
        "tail": torch.as_tensor(tail, dtype=torch.long, device=device),
        "degree": degree_tensor(
            num_entities,
            torch.as_tensor(head, dtype=torch.long),
            torch.as_tensor(tail, dtype=torch.long),
        ).to(device),
    }


def masked_graph_tensors(
    graph: KnowledgeGraph,
    anchor_type: AnchorType,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """The graph with one anchor type's patient edges removed.

    Stage 2 has to reconstruct an attachment it cannot see, so the forward pass
    that produces the representation runs on the graph with those edges dropped.
    """

    from oaer.catalogue.graph import RELATION_INDEX

    relation = ANCHOR_RELATION[anchor_type]
    retained = [
        edge
        for edge in graph.edges
        if not (edge.relation == relation and graph.nodes[edge.head].kind.value == "patient")
    ]
    lookup = graph.index()
    head_index = [lookup[edge.head] for edge in retained]
    tail_index = [lookup[edge.tail] for edge in retained]
    return {
        "entity": torch.arange(len(graph.nodes), dtype=torch.long, device=device),
        "head": torch.as_tensor(head_index, dtype=torch.long, device=device),
        "relation": torch.as_tensor(
            [RELATION_INDEX[edge.relation] for edge in retained],
            dtype=torch.long,
            device=device,
        ),
        "tail": torch.as_tensor(tail_index, dtype=torch.long, device=device),
        "degree": degree_tensor(
            len(graph.nodes),
            torch.as_tensor(head_index, dtype=torch.long),
            torch.as_tensor(tail_index, dtype=torch.long),
        ).to(device),
    }


def _report(stage: str, run: TrainingRun, parameters: int) -> StageReport:
    summary = run.summary()
    if summary.get("steps", 0.0) == 0.0:
        return StageReport(
            stage, 0, float("nan"), float("nan"), float("nan"), float("nan"), parameters
        )
    return StageReport(
        stage=stage,
        steps=int(summary["steps"]),
        first_loss=summary["first_loss"],
        last_loss=summary["last_loss"],
        trailing_mean=summary["trailing_mean"],
        leading_mean=summary["leading_mean"],
        parameters=parameters,
    )


def build_encoder(
    graph: KnowledgeGraph,
    *,
    embedding_dim: int,
    hidden_dim: int,
    layers: int,
    num_basis: int,
    dropout: float,
    device: str = "cpu",
) -> RelationalEncoder:
    """One encoder over the graph's node and relation vocabularies."""

    relations = {edge.relation for edge in graph.edges}
    from oaer.catalogue.graph import RELATION_TYPES

    return RelationalEncoder(
        num_entities=len(graph.nodes),
        num_relations=max(len(relations), len(RELATION_TYPES)),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        num_basis=num_basis,
        dropout=dropout,
    ).to(device)


def stage_one(
    graph: KnowledgeGraph,
    batch: MaskedEdgeBatch,
    encoder: RelationalEncoder,
    config: TrainingConfig,
    *,
    device: str = "cpu",
) -> StageReport:
    """Eq. (3) over the masked edges, with the negative tails as alternatives."""

    set_seed(config.seed)
    tensors = graph_tensors(graph, device)
    head = torch.as_tensor(batch.head, dtype=torch.long, device=device)
    tail = torch.as_tensor(batch.tail, dtype=torch.long, device=device)
    negative = torch.as_tensor(batch.negative, dtype=torch.long, device=device)
    total = int(head.shape[0])

    def step(model: nn.Module, index: Tensor) -> Tensor:
        state = model(
            tensors["entity"],
            tensors["head"],
            tensors["relation"],
            tensors["tail"],
            tensors["degree"],
        )
        rows = index
        positive = (state[head[rows]] * state[tail[rows]]).sum(dim=-1) / (state.shape[1] ** 0.5)
        negatives = torch.einsum("bw,bkw->bk", state[head[rows]], state[negative[rows]])
        negatives = negatives / (state.shape[1] ** 0.5)
        return masked_edge_loss(positive, negatives)

    trainer = Trainer(encoder, config, step, batch_size=config.batch_size)
    run = trainer.fit(total)
    return _report("stage1", run, sum(parameter.numel() for parameter in encoder.parameters()))


@dataclass(frozen=True)
class AnchorBatch:
    """A rectangular batch of anchor reconstructions, grouped by anchor type."""

    anchor_type: AnchorType
    patient: Tensor
    positive: Tensor
    negative: Tensor

    @property
    def rows(self) -> int:
        return int(self.patient.shape[0])


def anchor_batches(
    graph: KnowledgeGraph,
    corruptions: tuple[AnchorCorruption, ...],
    *,
    device: str = "cpu",
) -> tuple[AnchorBatch, ...]:
    """Group the corrupted anchors into one rectangular batch per anchor type."""

    lookup = graph.index()
    grouped: dict[AnchorType, list[tuple[int, int, tuple[int, ...]]]] = {}
    for record in corruptions:
        patient = f"PT:{record.patient}"
        if patient not in lookup:
            continue
        for anchor in record.positive:
            if anchor not in lookup:
                continue
            negatives = tuple(lookup[node] for node in record.negative if node in lookup)
            if not negatives:
                continue
            grouped.setdefault(record.anchor_type, []).append(
                (lookup[patient], lookup[anchor], negatives)
            )
    batches: list[AnchorBatch] = []
    for anchor_type, rows in sorted(grouped.items(), key=lambda item: item[0].value):
        width = min(len(row[2]) for row in rows)
        batches.append(
            AnchorBatch(
                anchor_type=anchor_type,
                patient=torch.as_tensor([row[0] for row in rows], dtype=torch.long, device=device),
                positive=torch.as_tensor([row[1] for row in rows], dtype=torch.long, device=device),
                negative=torch.as_tensor(
                    [row[2][:width] for row in rows], dtype=torch.long, device=device
                ),
            )
        )
    return tuple(batches)


def stage_two(
    graph: KnowledgeGraph,
    corruptions: tuple[AnchorCorruption, ...],
    encoder: RelationalEncoder,
    head: PatientAnchorHead,
    readout: AnchorReconstruction,
    config: TrainingConfig,
    *,
    device: str = "cpu",
) -> StageReport:
    """Eq. (4): reconstruct masked patient anchors against their non-anchors.

    For every anchor type the forward pass runs on the graph with that type's
    patient edges removed, so the representation the readout scores against cannot
    have seen the edge it is asked to reconstruct.
    """

    set_seed(config.seed)
    batches = anchor_batches(graph, corruptions, device=device)
    if not batches:
        return StageReport("stage2", 0, float("nan"), float("nan"), float("nan"), float("nan"), 0)
    masked = {
        batch.anchor_type: masked_graph_tensors(graph, batch.anchor_type, device)
        for batch in batches
    }
    module = nn.ModuleList([encoder, head, readout])
    total = sum(batch.rows for batch in batches)

    def step(model: nn.Module, index: Tensor) -> Tensor:
        del model
        losses: list[Tensor] = []
        cursor = 0
        order = index.tolist()
        for batch in batches:
            rows = set(range(cursor, cursor + batch.rows))
            cursor += batch.rows
            selected = [row - (cursor - batch.rows) for row in order if row in rows]
            if not selected:
                continue
            tensors = masked[batch.anchor_type]
            state = encoder(
                tensors["entity"],
                tensors["head"],
                tensors["relation"],
                tensors["tail"],
                tensors["degree"],
            )
            pick = torch.as_tensor(selected, dtype=torch.long, device=device)
            patients = state[batch.patient[pick]]
            types = torch.full(
                (patients.shape[0],),
                anchor_type_index(batch.anchor_type),
                dtype=torch.long,
                device=device,
            )
            context = head.pool_rows(patients, types).unsqueeze(1)
            positive = readout(context, state[batch.positive[pick]].unsqueeze(1)).squeeze(1)
            broadcast = context.expand(-1, batch.negative.shape[1], -1)
            negative = readout(broadcast, state[batch.negative[pick]]).reshape(-1)
            losses.append(anchor_reconstruction_loss(positive, negative))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        return torch.stack(losses).mean()

    trainer = Trainer(module, config, step, batch_size=config.batch_size)
    run = trainer.fit(total)
    parameters = sum(parameter.numel() for parameter in head.parameters()) + sum(
        parameter.numel() for parameter in readout.parameters()
    )
    return _report("stage2", run, parameters)


def stage_three(
    prediction: Callable[[Tensor], Tensor],
    target: Tensor,
    weights: Tensor,
    module: nn.Module,
    config: TrainingConfig,
    *,
    penalty: float = 1.0e-4,
) -> StageReport:
    """Eq. (5): weighted least squares on the out-of-fold pseudo-outcomes."""

    set_seed(config.seed)
    total = int(target.shape[0])

    def step(model: nn.Module, index: Tensor) -> Tensor:
        output = prediction(index)
        residual = output - target[index]
        mass = weights[index]
        data = (mass * residual * residual).sum() / mass.sum().clamp(min=1.0)
        regulariser = sum(
            parameter.pow(2).sum() for parameter in model.parameters() if parameter.requires_grad
        )
        return data + penalty * regulariser

    trainer = Trainer(module, config, step, batch_size=config.batch_size)
    run = trainer.fit(total)
    return _report("stage3", run, sum(parameter.numel() for parameter in module.parameters()))


def fit_graph_stages(
    graph: KnowledgeGraph,
    masked_edges: MaskedEdgeBatch,
    corruptions: tuple[AnchorCorruption, ...],
    *,
    embedding_dim: int,
    hidden_dim: int,
    layers: int,
    num_basis: int,
    dropout: float,
    training: TrainingConfig,
    device: str = "cpu",
) -> StageBundle:
    """Run Stage 1 and Stage 2 and return the fitted bundle with its reports."""

    encoder = build_encoder(
        graph,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        num_basis=num_basis,
        dropout=dropout,
        device=device,
    )
    anchor_head = PatientAnchorHead(encoder.hidden_dim).to(device)
    anchor_readout = AnchorReconstruction(encoder.hidden_dim).to(device)
    report_one = stage_one(graph, masked_edges, encoder, training, device=device)
    report_two = stage_two(
        graph, corruptions, encoder, anchor_head, anchor_readout, training, device=device
    )
    return StageBundle(
        encoder=encoder,
        anchor_head=anchor_head,
        anchor_readout=anchor_readout,
        reports=[report_one, report_two],
    )
