#!/usr/bin/env bash
# Run one protocol end to end and write its plain-text report under reports/.
#
#   pipelines/run_protocol.sh protocols/main.yaml
set -euo pipefail

PROTOCOL="${1:-protocols/main.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.rank --protocol "${PROTOCOL}" --print
