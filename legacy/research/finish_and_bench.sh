#!/bin/bash
# Wait for v9 Modal training and GNN iter training to both finish, then
# kick off end-of-run benchmarks:
#   - v9: MCE(750) on Modal (100 games, 10 workers)
#   - GNN: gameplay bench on local via bench_gnn.py (200 games)
#
# Prints results to stdout and also to bench_results_final.log.

set -u
LOG=bench_results_final.log

echo "=== finish_and_bench.sh ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo | tee -a "$LOG"

poll_pid() {
  local pid=$1
  local label=$2
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date +%H:%M:%S)] $label (PID $pid) finished" | tee -a "$LOG"
}

# Wait for processes to finish
wait_for_pids() {
  local pids=("$@")
  for pid in "${pids[@]}"; do
    if [ -n "$pid" ]; then
      poll_pid "$pid" "process"
    fi
  done
}

# Best-effort detection of running processes
V9_PID=$(ps aux | grep "run_v9_modal.sh" | grep -v grep | awk '{print $2}' | head -1)
GNN_PID=$(ps aux | grep "train_gnn_iter.py" | grep -v grep | awk '{print $2}' | head -1)

echo "Waiting for: v9=${V9_PID:-none}, gnn=${GNN_PID:-none}" | tee -a "$LOG"

if [ -n "$V9_PID" ]; then poll_pid "$V9_PID" "v9 wrapper"; fi
if [ -n "$GNN_PID" ]; then poll_pid "$GNN_PID" "GNN orchestrator"; fi

echo | tee -a "$LOG"
echo "Both runs done. Starting benches." | tee -a "$LOG"
echo | tee -a "$LOG"

# Find final v9 weights (highest iter N)
V9_WEIGHTS=""
for i in $(seq 15 -1 1); do
  if [ -f "nnue_weights_v9_iter${i}.bin" ]; then
    V9_WEIGHTS="nnue_weights_v9_iter${i}.bin"
    break
  fi
done

if [ -n "$V9_WEIGHTS" ]; then
  echo "═══ v9 MCE(750) bench — weights: $V9_WEIGHTS ═══" | tee -a "$LOG"
  python3 -m modal run modal_collect.py::benchmark \
    --num-workers 10 --games-per-worker 10 \
    --strategy mce --rollouts 750 \
    --weights "$V9_WEIGHTS" 2>&1 | tee -a "$LOG"
else
  echo "No v9 weights found — skipping v9 bench" | tee -a "$LOG"
fi

echo | tee -a "$LOG"

# Use the best known GNN checkpoint (iter 1 greedy-trained was peak at 84.60;
# self-play iters 2-6 regressed to 81-82). Override via GNN_WEIGHTS env var.
GNN_WEIGHTS="${GNN_WEIGHTS:-gnn_v2_50k.pt}"
if [ ! -f "$GNN_WEIGHTS" ]; then
  GNN_WEIGHTS=""
  for i in $(seq 15 -1 1); do
    if [ -f "gnn_sp_iter${i}.pt" ]; then
      GNN_WEIGHTS="gnn_sp_iter${i}.pt"
      break
    fi
  done
fi

if [ -n "$GNN_WEIGHTS" ]; then
  echo "═══ GNN direct gameplay bench — weights: $GNN_WEIGHTS (100 games) ═══" | tee -a "$LOG"
  python3 bench_gnn.py \
    --checkpoint "$GNN_WEIGHTS" \
    --games 100 \
    --strategy gnn \
    --random-seed 2>&1 | tee -a "$LOG"

  echo | tee -a "$LOG"
  echo "═══ GNN MCE(300) bench — weights: $GNN_WEIGHTS (50 games, ~1h) ═══" | tee -a "$LOG"
  python3 bench_gnn_mce.py \
    --checkpoint "$GNN_WEIGHTS" \
    --games 50 \
    --rollouts 300 \
    --depth 6 \
    --random-seed 2>&1 | tee -a "$LOG"
else
  echo "No GNN weights found — skipping GNN bench" | tee -a "$LOG"
fi

echo | tee -a "$LOG"
echo "=== finish_and_bench.sh done: $(date) ===" | tee -a "$LOG"
