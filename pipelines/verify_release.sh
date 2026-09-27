#!/usr/bin/env bash
# Run the verification battery. The manifest is written last, inside the driver,
# so it digests the finished tree; this script must not write anything after it.
#
#   pipelines/verify_release.sh [protocols/main.yaml] [--live] [--no-study]
set -euo pipefail

PROTOCOL="protocols/main.yaml"
if [[ $# -gt 0 && $1 != -* ]]; then
  PROTOCOL="$1"
  shift
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m oaer.console.verify --protocol "${PROTOCOL}" "$@"
