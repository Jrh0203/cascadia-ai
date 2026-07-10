#!/usr/bin/env bash
# One-seed raw-game category replay under the exact d20 contract
# (recovery for scalar seed 2027070908 and distq seed 2027070962).
#
# Run FROM THE ORCHESTRATOR. Subcommands:
#   setup                      — push a d20 source snapshot to john0:~/cascadia-d20
#   generate <experiment> <seed> — build + run the single-seed batch on john0
#   fetch-validate <experiment> <seed> — fetch, validate against the pinned
#                                ledger/report, and install on pass
#
# <experiment> is cycle4 or distq. Generation refuses while any scientific
# job is live on john0 (one-job rule). Validation is fail-closed
# (cascadiav3.validate_seed_replay); nothing is installed on any mismatch.
set -euo pipefail

D20_REV=d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135
D20_ROOT='~/cascadia-d20'
MAIN_ROOT='~/cascadia'
STAGING="cascadiav3/reports/seed_replay_staging"

experiment_paths() {
  case "$1" in
    cycle4)
      MANIFEST="$HOME_REMOTE/cascadia/cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json"
      TAG=rules_20260709_cycle4_n1024_d16 ;;
    distq)
      MANIFEST="$HOME_REMOTE/cascadia/cascadiav3/checkpoints/full_v3_distq_k8/best_locked_val.manifest.json"
      TAG=rules_20260709_distq_k8_n1024_d16 ;;
    *) echo "unknown experiment: $1 (cycle4|distq)" >&2; exit 2 ;;
  esac
}

cmd="${1:-}"
case "$cmd" in
setup)
  git rev-parse --verify "$D20_REV^{commit}" >/dev/null
  ssh john0 "mkdir -p $D20_ROOT"
  git archive "$D20_REV" | ssh john0 "tar -x -C $D20_ROOT"
  ssh john0 "printf '%s\n' $D20_REV > $D20_ROOT/DEPLOYED_REVISION.txt && ls $D20_ROOT/cascadiav3/real-root-exporter/Cargo.toml && echo 'd20 snapshot staged'"
  ;;
generate)
  exp="${2:?experiment}"; seed="${3:?seed}"
  HOME_REMOTE='$HOME'
  experiment_paths "$exp"
  ssh john0 /bin/bash -s <<REMOTE
set -euo pipefail
cd \$HOME/cascadia-d20
[ "\$(cat DEPLOYED_REVISION.txt)" = "$D20_REV" ] || { echo "d20 marker missing/mismatched"; exit 3; }
# One-job rule: refuse while any scientific job runs.
if ps aux | grep -E 'gumbel-benchmark-batch|run_exact_k1|run_structured_q|run_model_throughput|run_market_samples|run_cuda_concurrency|torch_train' | grep -v grep | grep -q .; then
  echo "REFUSED: a scientific job is live on john0"; exit 4
fi
export PATH="\$HOME/.cargo/bin:\$PATH" BLAKE3_NO_ASM=1 \
  CC=\$HOME/.local/bin/zig-cc \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=\$HOME/.local/bin/zig-cc
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
mkdir -p $STAGING
# Battery execution contract: TF32 off, fused CGAB on.
unset CASCADIA_BRIDGE_TF32
./cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --gumbel-benchmark-batch \
  --first-seed "$seed" --seed-count 1 \
  --model-service "env CASCADIA_CGAB_FUSED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src /home/john0/venvs/torch/bin/python3 -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device cuda" \
  --model-manifest "$MANIFEST" --model-timeout-ms 300000 \
  --gumbel-n-simulations 1024 --gumbel-top-m 16 --gumbel-depth-rounds 1 \
  --gumbel-determinizations 16 --gumbel-market-decision-samples 8 \
  --gumbel-blend-weight 0.5 --gumbel-exploration off --k-interior 16 \
  --max-actions 64 --rollout-top-k 4 \
  --model-sessions 1 --shared-model-session \
  --output-dir "$STAGING"
echo "replayed file:"; ls -la "$STAGING/gumbel_game_seed_${seed}.jsonl"
REMOTE
  ;;
fetch-validate)
  exp="${2:?experiment}"; seed="${3:?seed}"
  HOME_REMOTE="/home/john0"
  experiment_paths "$exp"
  work="cascadiav3/reports/seed_replay_${exp}_${seed}"
  mkdir -p "$work"
  scp -q "john0:cascadia-d20/$STAGING/gumbel_game_seed_${seed}.jsonl" "$work/"
  scp -q "john0:cascadia/cascadiav3/reports/${TAG}_decisions.jsonl" "$work/"
  scp -q "john0:cascadia/cascadiav3/reports/${TAG}.json" "$work/"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src ./venv/bin/python \
    -m cascadiav3.validate_seed_replay \
    --replayed-file "$work/gumbel_game_seed_${seed}.jsonl" \
    --decisions-ledger "$work/${TAG}_decisions.jsonl" \
    --report "$work/${TAG}.json" \
    --seed "$seed" \
    --raw-games-dir /nonexistent-remote-validated-below
  echo "local validation passed; installing on john0 (guarded)"
  ssh john0 /bin/bash -s <<REMOTE
set -euo pipefail
raw=\$HOME/cascadia/cascadiav3/reports/${TAG}_raw_games
[ ! -e "\$raw/gumbel_game_seed_${seed}.jsonl" ] || { echo "already installed"; exit 5; }
n=\$(find "\$raw" -maxdepth 1 -name 'gumbel_game_seed_*.jsonl' | wc -l)
[ "\$n" -eq 99 ] || { echo "raw dir has \$n files, expected 99"; exit 6; }
cp \$HOME/cascadia-d20/$STAGING/gumbel_game_seed_${seed}.jsonl "\$raw/"
n=\$(find "\$raw" -maxdepth 1 -name 'gumbel_game_seed_*.jsonl' | wc -l)
[ "\$n" -eq 100 ] || { echo "post-install count \$n != 100"; exit 7; }
echo "installed; ledger complete at 100"
REMOTE
  ;;
*)
  echo "usage: $0 setup | generate <cycle4|distq> <seed> | fetch-validate <cycle4|distq> <seed>" >&2
  exit 2
  ;;
esac
