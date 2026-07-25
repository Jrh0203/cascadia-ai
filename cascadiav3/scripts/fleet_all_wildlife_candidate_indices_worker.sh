#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
THREADS="${THREADS:-8}"
RESTARTS="${RESTARTS:-96}"
ITERATIONS="${ITERATIONS:-500000}"
BASE_SEED="${BASE_SEED:-2026072302}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$INDICES:$THREADS:$RESTARTS:$ITERATIONS:$BASE_SEED" in
  *[!0-9,:]*)
    echo "indices and candidate settings contain invalid characters" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
BINARY="${ROOT}/target/release/all_wildlife_candidates"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$BINARY"

IFS=',' read -r -a index_array <<< "$INDICES"
test "${#index_array[@]}" -gt 0
seen_indices=","
for index in "${index_array[@]}"; do
  [ "$index" -ge 0 ] && [ "$index" -lt 1024 ] || {
    echo "ruleset index out of range: $index" >&2
    exit 64
  }
  case "$seen_indices" in
    *",$index,"*)
      echo "duplicate ruleset index: $index" >&2
      exit 64
      ;;
  esac
  seen_indices="${seen_indices}${index},"
done

for index in "${index_array[@]}"; do
  output="${OUTPUT_DIR}/ruleset_${index}.json"
  printf '%s host=%s index=%s threads=%s restarts=%s iterations=%s seed=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" \
    "$index" "$THREADS" "$RESTARTS" "$ITERATIONS" "$BASE_SEED"
  "$BINARY" "$output" "$index" "$((index + 1))" "$THREADS" \
    "$RESTARTS" "$ITERATIONS" "$BASE_SEED"
done
