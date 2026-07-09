#!/usr/bin/env bash
set -euo pipefail

# Keep the data-only fleet productive after the 50-seed expansion completes.
# Roles and seeds are fixed before any structured-Q candidate exists. Each
# remote chain waits for its current producer and validator, requires their
# reports to pass, generates one reserve raw-v4 shard, and validates it. No
# artifact is fetched, admitted to training, or sent to john0.

SOURCE_REVISION="6e89d9555f6126bdc29f65657d8431cab3d2c024"
TEACHER_MANIFEST_SHA256="b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d"
TEACHER_WEIGHTS_SHA256="33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354"
SEED_COUNT=20

REMOTE_RUNNER=""
read -r -d '' REMOTE_RUNNER <<'RUNNER' || true
set -euo pipefail
root=$1
current_producer_pid=$2
current_validator_pid=$3
current_label=$4
reserve_label=$5
first_seed=$6
seed_count=$7
source_revision=$8
teacher_manifest_sha=$9
teacher_weights_sha=${10}
out=${11}
cd "$root"

while kill -0 "$current_producer_pid" 2>/dev/null; do sleep 30; done
while kill -0 "$current_validator_pid" 2>/dev/null; do sleep 30; done
python3 - \
  "cascadiav3/reports/structured_q_v4_expansion_20260709/${current_label}.summary.json" \
  "cascadiav3/reports/structured_q_v4_expansion_20260709/${current_label}.invariants.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    if json.load(open(path, encoding="utf-8")).get("status") != "pass":
        raise SystemExit(f"current expansion validation failed: {path}")
PY

manifest=cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json
weights=cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.weights.pt
model_service="venv/bin/python -m cascadiav3.torch_inference_bridge "
model_service+="--manifest $manifest --device mps"
test "$(cat cascadiav3/logs/structured_q_source_revision.txt)" = "$source_revision"
test "$(shasum -a 256 "$manifest" | awk '{print $1}')" = "$teacher_manifest_sha"
test "$(shasum -a 256 "$weights" | awk '{print $1}')" = "$teacher_weights_sha"

env CASCADIA_CGAB_FUSED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --gumbel-selfplay-tensor-corpus \
  --model-service "$model_service" \
  --model-manifest "$manifest" \
  --source-revision "$source_revision" \
  --model-timeout-ms 120000 \
  --first-seed "$first_seed" \
  --seed-count "$seed_count" \
  --plies-per-seed 80 \
  --max-actions 8 \
  --rollouts-per-action 1 \
  --rollout-top-k 4 \
  --gumbel-n-simulations 8 \
  --gumbel-top-m 4 \
  --gumbel-depth-rounds 1 \
  --gumbel-determinizations 1 \
  --gumbel-market-decision-samples 8 \
  --gumbel-exact-endgame-turns 1 \
  --gumbel-blend-weight 0.5 \
  --k-interior 8 \
  --model-sessions 2 \
  --shared-model-session \
  --rayon-threads 8 \
  --tensor-compression stored \
  --out "$out"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src venv/bin/python \
  -m cascadiav3.expert_tensor_shards \
  --summarize-shard "$out" \
  --report "${out%.npz}.summary.json"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src venv/bin/python \
  -m cascadiav3.validate_expert_tensor_invariants \
  --shard "$out" \
  --require-q-equals-afterstate-plus-score-to-go \
  --report "${out%.npz}.invariants.json"
shasum -a 256 \
  "$out" \
  "${out%.npz}.manifest.json" \
  "${out%.npz}.summary.json" \
  "${out%.npz}.invariants.json"
printf '[structured-q-reserve] complete label=%s first_seed=%s seed_count=%s\n' \
  "$reserve_label" "$first_seed" "$seed_count"
RUNNER
REMOTE_RUNNER_B64="$(
  printf '%s' "$REMOTE_RUNNER" \
    | python3 -c 'import base64,sys; print(base64.b64encode(sys.stdin.buffer.read()).decode())'
)"

arm_one() {
  local host="$1"
  local current_label="$2"
  local reserve_label="$3"
  local first_seed="$4"

  ssh "$host" /bin/bash -s -- \
    "$current_label" "$reserve_label" "$first_seed" "$SEED_COUNT" \
    "$SOURCE_REVISION" "$TEACHER_MANIFEST_SHA256" "$TEACHER_WEIGHTS_SHA256" \
    "$REMOTE_RUNNER_B64" <<'REMOTE'
set -euo pipefail
current_label=$1
reserve_label=$2
first_seed=$3
seed_count=$4
source_revision=$5
teacher_manifest_sha=$6
teacher_weights_sha=$7
runner_b64=$8
cd "$HOME/cascadia"

test "$(cat cascadiav3/logs/structured_q_source_revision.txt)" = "$source_revision"
test "$(shasum -a 256 \
  cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json \
  | awk '{print $1}')" = "$teacher_manifest_sha"
test "$(shasum -a 256 \
  cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.weights.pt \
  | awk '{print $1}')" = "$teacher_weights_sha"

current_producer_pid=$(
  cat "cascadiav3/logs/structured_q_v4_${current_label}.pid"
)
current_validator_pid=$(
  cat "cascadiav3/logs/structured_q_v4_${current_label}_validator.pid"
)
out="cascadiav3/reports/structured_q_v4_reserve_20260709/${reserve_label}.npz"
chain_log="cascadiav3/logs/structured_q_v4_${reserve_label}_chain.log"
chain_pid_file="cascadiav3/logs/structured_q_v4_${reserve_label}_chain.pid"
test ! -e "$out"
test ! -e "${out%.npz}.manifest.json"
test ! -e "$chain_pid_file"
mkdir -p "$(dirname "$out")"

runner="$(
  python3 -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode())' \
    "$runner_b64"
)"
nohup /bin/bash -c "$runner" chain \
  "$PWD" "$current_producer_pid" "$current_validator_pid" \
  "$current_label" "$reserve_label" "$first_seed" "$seed_count" \
  "$source_revision" "$teacher_manifest_sha" "$teacher_weights_sha" \
  "$out" >"$chain_log" 2>&1 &
chain_pid=$!
printf '%s\n' "$chain_pid" > "$chain_pid_file"
sleep 1
kill -0 "$chain_pid"
printf '%s %s %s %s\n' "$chain_pid" "$reserve_label" "$first_seed" "$seed_count"
REMOTE
}

arm_one john2 expansion_a reserve_selection 2027073750
arm_one john3 expansion_b reserve_verdict 2027073770
arm_one john4 expansion_c reserve_replication 2027073790

echo "[structured-q-reserve] armed; data only, no fetch or training action"
