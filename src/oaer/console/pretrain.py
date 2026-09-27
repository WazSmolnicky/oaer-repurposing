"""Stage 1 and Stage 2 on the typed knowledge graph.

Ref: Eq. (3)-(4), Sec. 2.6 (the two graph-side stages, neither of which reads an
outcome).
"""

from __future__ import annotations

import json

from oaer.console.common import configure_and_parse, load_config, report_path
from oaer.study import assemble_study, attach_patients, fit_graph, training_config
from oaer.support.io import write_text
from oaer.support.logging import get_logger
from oaer.validation.manifest import manifest_root

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    invocation = configure_and_parse("Pretrain the relational encoder on the graph", argv)
    config = load_config(invocation)
    root = manifest_root(invocation.protocol)
    inputs = assemble_study(config)
    patients = attach_patients(
        inputs.graph, inputs.retrospective, limit=config.catalogue.anchor_types * 50
    )
    LOGGER.info(
        "pretraining on %d nodes, %d edges and %d patient anchors",
        len(inputs.graph.nodes),
        len(inputs.graph.edges),
        len(patients),
    )
    bundle = fit_graph(config, inputs.graph, patients, seed=config.seed)
    report = {
        "protocol": config.name,
        "seed": config.seed,
        "graph_census": inputs.graph.census(),
        "patients_attached": len(patients),
        "schedule": {
            "batch_size": training_config(config, stage="stage1").batch_size,
            "learning_rate": training_config(config, stage="stage1").learning_rate,
            "weight_decay": training_config(config, stage="stage1").weight_decay,
            "grad_clip": training_config(config, stage="stage1").grad_clip,
            "precision": config.fitting.precision,
            "device": config.fitting.device,
        },
        "stages": {name: dict(payload) for name, payload in bundle.summary().items()},
    }
    lines = [
        "Outcome-anchored effect ranking: graph pretraining",
        "=" * 78,
        json.dumps(report, indent=2, sort_keys=True),
        "",
    ]
    target = report_path(invocation, f"pretrain_{config.name}.txt", root)
    write_text("\n".join(lines), target)
    if invocation.emit:
        LOGGER.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
