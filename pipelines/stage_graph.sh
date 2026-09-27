#!/usr/bin/env bash
# Build the catalogue and the typed graph, then run the two graph-side stages.
#
#   pipelines/stage_graph.sh protocols/main.yaml
set -euo pipefail

PROTOCOL="${1:-protocols/main.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.pretrain --protocol "${PROTOCOL}"
