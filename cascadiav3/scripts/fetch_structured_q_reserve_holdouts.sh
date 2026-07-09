#!/usr/bin/env bash
set -euo pipefail

# Fetch the three preregistered structured-Q holdout roles only after their
# chained generation and validation finish. Prove exact seed roles, one data
# contract, and disjointness from both the locked pilot and fit expansion.
# This script never copies data to john0 or adds a shard to training.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
DEST="${DEST:-$ROOT/cascadiav3/reports/structured_q_v4_reserve_20260709}"
LOCKED="$ROOT/cascadiav3/reports/structured_q_v4_20260709"
EXPANSION="$ROOT/cascadiav3/reports/structured_q_v4_expansion_20260709"
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
  local first_seed="$3"
  local remote_root="cascadiav3/reports/structured_q_v4_reserve_20260709"
  local chain_pid="cascadiav3/logs/structured_q_v4_${label}_chain.pid"
  local chain_log="cascadiav3/logs/structured_q_v4_${label}_chain.log"
  local remote_npz_sha
  local remote_manifest_sha

  echo "[structured-q-reserve-fetch] preflight $host/$label"
  ssh "$host" /bin/bash -s -- \
    "$label" "$first_seed" "$remote_root" "$chain_pid" "$chain_log" <<'REMOTE'
set -euo pipefail
label=$1
first_seed=$2
remote_root=$3
chain_pid_file=$4
chain_log=$5
cd "$HOME/cascadia"
chain_pid=$(cat "$chain_pid_file")
if kill -0 "$chain_pid" 2>/dev/null; then
  echo "chain $chain_pid is still active for $label" >&2
  exit 2
fi
for path in \
  "$remote_root/$label.npz" \
  "$remote_root/$label.manifest.json" \
  "$remote_root/$label.summary.json" \
  "$remote_root/$label.invariants.json" \
  "$chain_log"; do
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
grep -F \
  "[structured-q-reserve] complete label=$label first_seed=$first_seed seed_count=20" \
  "$chain_log" >/dev/null
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
  rsync -a "$host:~/cascadia/$chain_log" "$DEST/$label.chain.log"

  test "$(sha256 "$DEST/$label.npz")" = "$remote_npz_sha"
  test "$(sha256 "$DEST/$label.manifest.json")" = "$remote_manifest_sha"
  echo "[structured-q-reserve-fetch] verified $host/$label npz=$remote_npz_sha"
}

fetch_one john2 reserve_selection 2027073750
fetch_one john3 reserve_verdict 2027073770
fetch_one john4 reserve_replication 2027073790

for locked in train_a train_b val; do
  test -s "$LOCKED/$locked.npz"
  test -s "$LOCKED/$locked.manifest.json"
done
for expansion in expansion_a expansion_b expansion_c; do
  test -s "$EXPANSION/$expansion.npz"
  test -s "$EXPANSION/$expansion.manifest.json"
done

"$PYTHON" -m cascadiav3.audit_structured_q_shards \
  --shard "reserve_selection=$DEST/reserve_selection.npz" \
  --shard "reserve_verdict=$DEST/reserve_verdict.npz" \
  --shard "reserve_replication=$DEST/reserve_replication.npz" \
  --expected-seed-domain \
  "reserve_selection=first_seed=2027073750,seed_count=20,plies_per_seed=80,mode=gumbel_selfplay_tensor_corpus" \
  --expected-seed-domain \
  "reserve_verdict=first_seed=2027073770,seed_count=20,plies_per_seed=80,mode=gumbel_selfplay_tensor_corpus" \
  --expected-seed-domain \
  "reserve_replication=first_seed=2027073790,seed_count=20,plies_per_seed=80,mode=gumbel_selfplay_tensor_corpus" \
  --exclude-shard "locked_fit=$LOCKED/train_a.npz" \
  --exclude-shard "locked_selection=$LOCKED/train_b.npz" \
  --exclude-shard "locked_verdict=$LOCKED/val.npz" \
  --exclude-shard "fit_expansion_a=$EXPANSION/expansion_a.npz" \
  --exclude-shard "fit_expansion_b=$EXPANSION/expansion_b.npz" \
  --exclude-shard "fit_expansion_c=$EXPANSION/expansion_c.npz" \
  --expected-source-revision "$SOURCE_REVISION" \
  --expected-teacher-manifest-sha256 "$TEACHER_MANIFEST_SHA256" \
  --expected-teacher-weights-sha256 "$TEACHER_WEIGHTS_SHA256" \
  --out "$DEST/reserve_cross_shard_audit.json"

audit_sha="$(sha256 "$DEST/reserve_cross_shard_audit.json")"
echo "[structured-q-reserve-fetch] complete audit_sha256=$audit_sha"
echo "[structured-q-reserve-fetch] holdouts remain quarantined; no john0/training copy performed"
