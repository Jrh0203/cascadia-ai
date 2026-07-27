#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
exec "${ROOT}/.venv/bin/python" \
  "${ROOT}/tools/all_wildlife_fixed_count_watchdog.py" \
  --config \
  "${ROOT}/cascadiav3/fleet/all_wildlife_fixed_count_pipeline_20260726.json"
