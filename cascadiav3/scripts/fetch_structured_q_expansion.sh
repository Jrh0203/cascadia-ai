#!/usr/bin/env bash
set -euo pipefail

# Fetch the quarantined three-host structured-Q expansion and prove that it is
# internally valid, contract-identical, and seed-disjoint from the locked
# fit/selection/verdict pilot. This script never copies data to john0 and never
# adds a shard to a training command.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
DEST="${DEST:-$ROOT/cascadiav3/reports/structured_q_v4_expansion_20260709}"
LOCKED="$ROOT/cascadiav3/reports/structured_q_v4_20260709"
SOURCE_REVISION="6e89d9555f6126bdc29f65657d8431cab3d2c024"
TEACHER_MANIFEST_SHA256="b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d"
TEACHER_WEIGHTS_SHA256="33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/cascadiav3/src"

command -v ssh >/dev/null
command -v rsync >/dev/null
command -v shasum >/dev/null
test -x "$PYTHON"
mkdir -p "$DEST"

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

fetch_one() {
  local host="$1"
  local label="$2"
  local remote_root="cascadiav3/reports/structured_q_v4_expansion_20260709"
  local producer_pid="cascadiav3/logs/structured_q_v4_${label}.pid"
  local validator_pid="cascadiav3/logs/structured_q_v4_${label}_validator.pid"
  local producer_log="cascadiav3/logs/structured_q_v4_${label}.log"
  local validator_log="cascadiav3/logs/structured_q_v4_${label}_validator.log"
  local remote_npz_sha
  local remote_manifest_sha

  echo "[structured-q-fetch] preflight $host/$label"
  ssh "$host" /bin/bash -s -- \
    "$label" "$remote_root" "$producer_pid" "$validator_pid" \
    "$producer_log" "$validator_log" <<'REMOTE'
set -euo pipefail
label=$1
remote_root=$2
producer_pid_file=$3
validator_pid_file=$4
producer_log=$5
validator_log=$6
cd "$HOME/cascadia"
producer_pid=$(cat "$producer_pid_file")
validator_pid=$(cat "$validator_pid_file")
if kill -0 "$producer_pid" 2>/dev/null; then
  echo "producer $producer_pid is still active for $label" >&2
  exit 2
fi
if kill -0 "$validator_pid" 2>/dev/null; then
  echo "validator $validator_pid is still active for $label" >&2
  exit 2
fi
for path in \
  "$remote_root/$label.npz" \
  "$remote_root/$label.manifest.json" \
  "$remote_root/$label.summary.json" \
  "$remote_root/$label.invariants.json" \
  "$producer_log" \
  "$validator_log"; do
  test -s "$path"
done
python3 - "$remote_root/$label.summary.json" \
  "$remote_root/$label.invariants.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    report = json.load(open(path, encoding="utf-8"))
    if report.get("status") != "pass":
        raise SystemExit(f"non-passing validation report: {path}")
PY
REMOTE

  remote_npz_sha="$(
    ssh "$host" "cd \"\$HOME/cascadia\" && shasum -a 256 '$remote_root/$label.npz'" \
      | awk '{print $1}'
  )"
  remote_manifest_sha="$(
    ssh "$host" \
      "cd \"\$HOME/cascadia\" && shasum -a 256 '$remote_root/$label.manifest.json'" \
      | awk '{print $1}'
  )"

  for suffix in npz manifest.json summary.json invariants.json; do
    rsync -a --partial "$host:~/cascadia/$remote_root/$label.$suffix" "$DEST/"
  done
  rsync -a "$host:~/cascadia/$producer_log" "$DEST/$label.producer.log"
  rsync -a "$host:~/cascadia/$validator_log" "$DEST/$label.validator.log"

  test "$(sha256 "$DEST/$label.npz")" = "$remote_npz_sha"
  test "$(sha256 "$DEST/$label.manifest.json")" = "$remote_manifest_sha"
  echo "[structured-q-fetch] verified $host/$label npz=$remote_npz_sha"
}

fetch_one john2 expansion_a
fetch_one john3 expansion_b
fetch_one john4 expansion_c

for locked in train_a train_b val; do
  test -s "$LOCKED/$locked.npz"
  test -s "$LOCKED/$locked.manifest.json"
done

"$PYTHON" -m cascadiav3.audit_structured_q_shards \
  --shard "expansion_a=$DEST/expansion_a.npz" \
  --shard "expansion_b=$DEST/expansion_b.npz" \
  --shard "expansion_c=$DEST/expansion_c.npz" \
  --exclude-shard "locked_fit=$LOCKED/train_a.npz" \
  --exclude-shard "locked_selection=$LOCKED/train_b.npz" \
  --exclude-shard "locked_verdict=$LOCKED/val.npz" \
  --expected-source-revision "$SOURCE_REVISION" \
  --expected-teacher-manifest-sha256 "$TEACHER_MANIFEST_SHA256" \
  --expected-teacher-weights-sha256 "$TEACHER_WEIGHTS_SHA256" \
  --out "$DEST/expansion_cross_shard_audit.json"

audit_sha="$(sha256 "$DEST/expansion_cross_shard_audit.json")"
echo "[structured-q-fetch] complete audit_sha256=$audit_sha"
echo "[structured-q-fetch] data remains quarantined; no john0/training copy performed"
