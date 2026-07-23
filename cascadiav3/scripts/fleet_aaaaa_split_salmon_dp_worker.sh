#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
CASE_INDEX="${CASE_INDEX:?set CASE_INDEX}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
RUNNER_SHA256="${RUNNER_SHA256:?set RUNNER_SHA256}"
DP_SHA256="${DP_SHA256:?set DP_SHA256}"
GAP_SOURCE_SHA256="${GAP_SOURCE_SHA256:?set GAP_SOURCE_SHA256}"
ZERO_SOURCE_SHA256="${ZERO_SOURCE_SHA256:?set ZERO_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"
MOTIF_SOURCE_SHA256="${MOTIF_SOURCE_SHA256:?set MOTIF_SOURCE_SHA256}"
WORKER_SHA256="${WORKER_SHA256:?set WORKER_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$CASE_INDEX" in
  0|1|2|3) ;;
  *)
    echo "CASE_INDEX must be 0, 1, 2, or 3" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
OUTPUT="${OUTPUT_DIR}/case_${CASE_INDEX}_${SHARD_HOST}.json"
LOG_DIR="${ROOT}/cascadiav3/logs"
PREFIX="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}"
HEARTBEAT="${PREFIX}.heartbeat"
EXIT_FILE="${PREFIX}.exit"
CHILD_PID_FILE="${PREFIX}.solver.pid"
WRAPPER_PID_FILE="${PREFIX}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test ! -e "$OUTPUT"
test ! -e "$EXIT_FILE"
test ! -e "$WRAPPER_PID_FILE"
[ "$(shasum -a 256 tools/aaaaa_wildlife_split_salmon_dp_screen.py | awk '{print $1}')" = "$RUNNER_SHA256" ]
[ "$(shasum -a 256 tools/aaaaa_wildlife_split_salmon_dp.py | awk '{print $1}')" = "$DP_SHA256" ]
[ "$(shasum -a 256 tools/aaaaa_wildlife_gap_two_salmon_pair_bound.py | awk '{print $1}')" = "$GAP_SOURCE_SHA256" ]
[ "$(shasum -a 256 tools/aaaaa_wildlife_zero_hawk_bound.py | awk '{print $1}')" = "$ZERO_SOURCE_SHA256" ]
[ "$(shasum -a 256 tools/aaaaa_wildlife_exact.py | awk '{print $1}')" = "$EXACT_SOURCE_SHA256" ]
[ "$(shasum -a 256 tools/aaaaa_wildlife_motif_certificate.py | awk '{print $1}')" = "$MOTIF_SOURCE_SHA256" ]
[ "$(shasum -a 256 "$0" | awk '{print $1}')" = "$WORKER_SHA256" ]

printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"
printf '%s source=%s host=%s case=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" "$CASE_INDEX"

set +e
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.aaaaa_wildlife_split_salmon_dp_screen \
  --case-index "$CASE_INDEX" --output "$OUTPUT" &
solver_pid=$!
printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
while kill -0 "$solver_pid" 2>/dev/null; do
  printf '%s solver_pid=%s case=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$CASE_INDEX" \
    > "${HEARTBEAT}.tmp"
  mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
  sleep 10
done
wait "$solver_pid"
status=$?
set -e

printf '%s complete case=%s exit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CASE_INDEX" "$status" \
  > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
exit "$status"
