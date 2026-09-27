#!/usr/bin/env bash
# Render the article's reported tables as plain text, for side-by-side reading.
#
#   pipelines/render_ledger.sh protocols/main.yaml
set -euo pipefail

PROTOCOL="${1:-protocols/main.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.ledger --protocol "${PROTOCOL}"
