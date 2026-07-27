#!/usr/bin/env bash
set -euo pipefail

SHALLOW_TAG="${SHALLOW_TAG:?set SHALLOW_TAG}"
PRODUCTION_TAG="${PRODUCTION_TAG:?set PRODUCTION_TAG}"
BEST_TAG="${BEST_TAG:?set BEST_TAG}"
REMOTE_HOSTS="${REMOTE_HOSTS:-john2 john3 john4}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"

ROOT="${HOME}/cascadia"
SYNC="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_sync.sh"
BEST_DIRECTORY="${ROOT}/cascadiav3/fleet_outputs/${BEST_TAG}"

cd "$ROOT"
test -x "$SYNC"
for stage_tag in "$SHALLOW_TAG" "$PRODUCTION_TAG"; do
  FLEET_TAG="$stage_tag" REMOTE_HOSTS="$REMOTE_HOSTS" \
    SYNC_INTERVAL="$SYNC_INTERVAL" CHUNK_SIZE="$CHUNK_SIZE" \
    /bin/bash "$SYNC"
done

rm -f "${BEST_DIRECTORY}/delivery_summary.json"
PYTHONDONTWRITEBYTECODE=1 "${ROOT}/.venv/bin/python" \
  -m tools.all_wildlife_fixed_count_merge \
  --stage shallow "${ROOT}/cascadiav3/fleet_outputs/${SHALLOW_TAG}" \
  --stage production "${ROOT}/cascadiav3/fleet_outputs/${PRODUCTION_TAG}" \
  --output-directory "${ROOT}/cascadiav3/fleet_outputs/${BEST_TAG}" \
  --chunk-size "$CHUNK_SIZE" \
  --deep

PYTHONDONTWRITEBYTECODE=1 "${ROOT}/.venv/bin/python" \
  -m tools.all_wildlife_fixed_count_report \
  "$BEST_DIRECTORY" \
  --output-json "${BEST_DIRECTORY}/ruleset_catalog.json" \
  --output-markdown "${BEST_DIRECTORY}/ruleset_catalog.md" \
  --atlas-output "${BEST_DIRECTORY}/wildlife-atlas.json" \
  --atlas-output "${ROOT}/apps/web/public/wildlife-atlas.json" \
  --delivery-summary "${BEST_DIRECTORY}/delivery_summary.json" \
  --chunk-size "$CHUNK_SIZE"
