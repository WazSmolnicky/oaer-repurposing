#!/usr/bin/env bash
# Re-fit the head with each construct removed, then recompute the interactions.
#
#   pipelines/ablate_components.sh protocols/main.yaml
set -euo pipefail

PROTOCOL="${1:-protocols/main.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.ablate --protocol "${PROTOCOL}"
