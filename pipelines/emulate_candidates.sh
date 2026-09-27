#!/usr/bin/env bash
# Cross-fit the per-candidate target-trial emulation and write its table.
#
#   pipelines/emulate_candidates.sh protocols/main.yaml
set -euo pipefail

PROTOCOL="${1:-protocols/main.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.emulate --protocol "${PROTOCOL}"
