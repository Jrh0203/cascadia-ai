#!/usr/bin/env bash
set -euo pipefail

SHALLOW_TAG="${SHALLOW_TAG:?set SHALLOW_TAG}"
PRODUCTION_TAG="${PRODUCTION_TAG:?set PRODUCTION_TAG}"
REMOTE_HOSTS="${REMOTE_HOSTS:-john2 john3 john4}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"

ROOT="${HOME}/cascadia"
SYNC="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_sync.sh"

test -x "$SYNC"
for stage_tag in "$SHALLOW_TAG" "$PRODUCTION_TAG"; do
  FLEET_TAG="$stage_tag" REMOTE_HOSTS="$REMOTE_HOSTS" \
    SYNC_INTERVAL="$SYNC_INTERVAL" CHUNK_SIZE="$CHUNK_SIZE" \
    /bin/bash "$SYNC"
done
