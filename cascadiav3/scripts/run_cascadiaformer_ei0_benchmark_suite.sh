#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

JOB_SLUG="${JOB_SLUG:-cascadiaformer_ei0_benchmark_suite}"
REMOTE_LOG_DIR="$REMOTE_ROOT/cascadiav3/logs"
REMOTE_JOB="$REMOTE_LOG_DIR/${JOB_SLUG}_job.sh"
REMOTE_LOG="$REMOTE_LOG_DIR/${JOB_SLUG}_job.log"
REMOTE_PID="$REMOTE_LOG_DIR/${JOB_SLUG}_job.pid"

BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/best_locked_val.manifest.json}"
NO_SEARCH_GAMES="${NO_SEARCH_GAMES:-100}"
NO_SEARCH_FIRST_SEED="${NO_SEARCH_FIRST_SEED:-2026990000}"
NO_SEARCH_MAX_ACTIONS="${NO_SEARCH_MAX_ACTIONS:-32}"
NO_SEARCH_SELECTION_HEADS="${NO_SEARCH_SELECTION_HEADS:-policy,q}"
NO_SEARCH_TREATMENT_WORKERS="${NO_SEARCH_TREATMENT_WORKERS:-1}"
NO_SEARCH_BASELINE_WORKERS="${NO_SEARCH_BASELINE_WORKERS:-8}"
SEARCH_GAMES="${SEARCH_GAMES:-20}"
SEARCH_FIRST_SEED="${SEARCH_FIRST_SEED:-2026995000}"
SEARCH_SELECTION_HEAD="${SEARCH_SELECTION_HEAD:-q}"
SEARCH_RETAIN_K="${SEARCH_RETAIN_K:-32}"
SEARCH_MAX_ACTIONS="${SEARCH_MAX_ACTIONS:-64}"
SEARCH_ROLLOUTS_PER_ACTION="${SEARCH_ROLLOUTS_PER_ACTION:-16}"
SEARCH_ROLLOUT_TOP_K="${SEARCH_ROLLOUT_TOP_K:-4}"
# Search-integrated gameplay is CPU rollout-bound. john0 has 16 physical cores
# and 32 threads, so default to enough game workers to avoid underfilled waves;
# individual experiments can still override candidate/control workers directly.
SEARCH_CPU_WORKERS="${SEARCH_CPU_WORKERS:-16}"
SEARCH_CANDIDATE_WORKERS="${SEARCH_CANDIDATE_WORKERS:-$SEARCH_CPU_WORKERS}"
SEARCH_BASELINE_WORKERS="${SEARCH_BASELINE_WORKERS:-$SEARCH_CPU_WORKERS}"
SEARCH_SHADOW_FULL_SEARCH="${SEARCH_SHADOW_FULL_SEARCH:-1}"
SEARCH_INCLUDE_FULL_SEARCH_BASELINE="${SEARCH_INCLUDE_FULL_SEARCH_BASELINE:-1}"
MAX_TREATMENT_CONTROL_TIME_RATIO="${MAX_TREATMENT_CONTROL_TIME_RATIO:-1.20}"
RUN_NO_SEARCH="${RUN_NO_SEARCH:-1}"
RUN_SEARCH="${RUN_SEARCH:-1}"

NO_SEARCH_REPORT="${NO_SEARCH_REPORT:-cascadiav3/reports/cascadiaformer_ei0_no_search_game100.json}"
NO_SEARCH_DECISIONS="${NO_SEARCH_DECISIONS:-cascadiav3/reports/cascadiaformer_ei0_no_search_game100_decisions.jsonl}"
NO_SEARCH_GAME_RESULTS="${NO_SEARCH_GAME_RESULTS:-cascadiav3/reports/cascadiaformer_ei0_no_search_game100_games.jsonl}"
NO_SEARCH_SUMMARY="${NO_SEARCH_SUMMARY:-cascadiav3/reports/cascadiaformer_ei0_no_search_game100_summary.md}"
NO_SEARCH_EXPERIMENT_ID="${NO_SEARCH_EXPERIMENT_ID:-cascadiaformer-ei0-no-search-game100}"
SEARCH_REPORT="${SEARCH_REPORT:-cascadiav3/reports/cascadiaformer_ei0_search_game20.json}"
SEARCH_DECISIONS="${SEARCH_DECISIONS:-cascadiav3/reports/cascadiaformer_ei0_search_game20_decisions.jsonl}"
SEARCH_GAME_RESULTS="${SEARCH_GAME_RESULTS:-cascadiav3/reports/cascadiaformer_ei0_search_game20_games.jsonl}"
SEARCH_SUMMARY="${SEARCH_SUMMARY:-cascadiav3/reports/cascadiaformer_ei0_search_game20_summary.md}"
SEARCH_EXPERIMENT_ID="${SEARCH_EXPERIMENT_ID:-cascadiaformer-ei0-search-game20}"

sync_benchmark_sources() {
  cd "$LOCAL_ROOT"
  rsync -az -e "$RSYNC_SSH" \
    cascadiav3/src/cascadiav3/torch_cascadiaformer_game_benchmark.py \
    "$REMOTE:$REMOTE_ROOT/cascadiav3/src/cascadiav3/torch_cascadiaformer_game_benchmark.py"
  rsync -az -e "$RSYNC_SSH" \
    cascadiav3/src/cascadiav3/torch_cascadiaformer_search_benchmark.py \
    "$REMOTE:$REMOTE_ROOT/cascadiav3/src/cascadiav3/torch_cascadiaformer_search_benchmark.py"
  rsync -az -e "$RSYNC_SSH" \
    cascadiav3/src/cascadiav3/validate_runbook_performance.py \
    "$REMOTE:$REMOTE_ROOT/cascadiav3/src/cascadiav3/validate_runbook_performance.py"
}

write_remote_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
  ssh -p "$SSH_PORT" "$REMOTE" "cat > '$REMOTE_JOB'" <<REMOTE_JOB
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=cascadiav3/src
trap 'status=\$?; if [ "\$status" -ne 0 ]; then echo "[ei0-bench] failed exit_code=\$status \$(date -Is)" >&2; fi' EXIT

echo "[ei0-bench] started \$(date -Is)"
echo "[ei0-bench] manifest=$MANIFEST"
test -s '$MANIFEST'
test -x '$BINARY'

python -m py_compile \
  cascadiav3/src/cascadiav3/torch_cascadiaformer_game_benchmark.py \
  cascadiav3/src/cascadiav3/torch_cascadiaformer_search_benchmark.py

if [ '$RUN_NO_SEARCH' = '1' ]; then
  echo "[ei0-bench] running no-search benchmark games=$NO_SEARCH_GAMES treatment_workers=$NO_SEARCH_TREATMENT_WORKERS baseline_workers=$NO_SEARCH_BASELINE_WORKERS"
  python -m cascadiav3.torch_cascadiaformer_game_benchmark \
    --binary '$BINARY' \
    --manifest '$MANIFEST' \
    --selection-heads '$NO_SEARCH_SELECTION_HEADS' \
    --games '$NO_SEARCH_GAMES' \
    --first-seed '$NO_SEARCH_FIRST_SEED' \
    --max-actions '$NO_SEARCH_MAX_ACTIONS' \
    --baseline-workers '$NO_SEARCH_BASELINE_WORKERS' \
    --treatment-workers '$NO_SEARCH_TREATMENT_WORKERS' \
    --device cuda \
    --experiment-id '$NO_SEARCH_EXPERIMENT_ID' \
    --out '$NO_SEARCH_REPORT' \
    --decisions-out '$NO_SEARCH_DECISIONS' \
    --game-results-out '$NO_SEARCH_GAME_RESULTS' \
    --summary-out '$NO_SEARCH_SUMMARY'
else
  echo "[ei0-bench] skipping no-search benchmark"
fi

if [ '$RUN_SEARCH' = '1' ]; then
  search_extra_flags=()
  if [ '$SEARCH_SHADOW_FULL_SEARCH' = '1' ]; then
    search_extra_flags+=(--shadow-full-search)
  fi
  if [ '$SEARCH_INCLUDE_FULL_SEARCH_BASELINE' = '1' ]; then
    search_extra_flags+=(--include-full-search-baseline)
  fi
  echo "[ei0-bench] running search-integrated benchmark games=$SEARCH_GAMES candidate_workers=$SEARCH_CANDIDATE_WORKERS shadow=$SEARCH_SHADOW_FULL_SEARCH full_baseline=$SEARCH_INCLUDE_FULL_SEARCH_BASELINE"
  python -m cascadiav3.torch_cascadiaformer_search_benchmark \
    --binary '$BINARY' \
    --manifest '$MANIFEST' \
    --selection-head '$SEARCH_SELECTION_HEAD' \
    --games '$SEARCH_GAMES' \
    --first-seed '$SEARCH_FIRST_SEED' \
    --retain-k '$SEARCH_RETAIN_K' \
    --max-actions '$SEARCH_MAX_ACTIONS' \
    --rollouts-per-action '$SEARCH_ROLLOUTS_PER_ACTION' \
    --rollout-top-k '$SEARCH_ROLLOUT_TOP_K' \
    --candidate-workers '$SEARCH_CANDIDATE_WORKERS' \
    --full-baseline-workers '$SEARCH_BASELINE_WORKERS' \
    "\${search_extra_flags[@]}" \
    --device cuda \
    --experiment-id '$SEARCH_EXPERIMENT_ID' \
    --out '$SEARCH_REPORT' \
    --decisions-out '$SEARCH_DECISIONS' \
    --game-results-out '$SEARCH_GAME_RESULTS' \
    --summary-out '$SEARCH_SUMMARY'

  if [ '$SEARCH_INCLUDE_FULL_SEARCH_BASELINE' = '1' ]; then
    python -m cascadiav3.validate_runbook_performance \
      --benchmark '$SEARCH_REPORT' \
      --max-treatment-control-time-ratio '$MAX_TREATMENT_CONTROL_TIME_RATIO'
  else
    echo "[ei0-bench] skipping treatment/control ratio validation because full baseline is disabled"
  fi
else
  echo "[ei0-bench] skipping search-integrated benchmark"
fi

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
echo "[ei0-bench] completed \$(date -Is)"
REMOTE_JOB
  ssh -p "$SSH_PORT" "$REMOTE" "chmod 700 '$REMOTE_JOB'"
}

launch_job() {
  sync_benchmark_sources
  write_remote_job
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ] && kill -0 \"\$(cat '$REMOTE_PID')\" 2>/dev/null; then
  echo \"already running pid \$(cat '$REMOTE_PID')\"
  echo '$REMOTE_LOG'
  exit 0
fi
nohup setsid '$REMOTE_JOB' > '$REMOTE_LOG' 2>&1 < /dev/null &
echo \$! > '$REMOTE_PID'
echo \"launched pid \$(cat '$REMOTE_PID')\"
echo '$REMOTE_LOG'"
}

status_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ] && kill -0 \"\$(cat '$REMOTE_PID')\" 2>/dev/null; then
  echo \"running pid \$(cat '$REMOTE_PID')\"
else
  echo 'not running'
fi
ps -eo pid=,ppid=,psr=,nlwp=,pcpu=,pmem=,etime=,args= | awk '
  /torch_cascadiaformer_(game|search)_benchmark/ && !/awk/ {print \"matching benchmark pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" mem \" \$6 \" elapsed \" \$7}
  /cascadiav3-real-root-exporter/ && /interactive-policy-game/ && !/awk/ {print \"matching interactive simulator pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" mem \" \$6 \" elapsed \" \$7}
'
for f in '$NO_SEARCH_REPORT' '$NO_SEARCH_SUMMARY' '$SEARCH_REPORT' '$SEARCH_SUMMARY' '$NO_SEARCH_DECISIONS' '$SEARCH_DECISIONS' '$NO_SEARCH_GAME_RESULTS' '$SEARCH_GAME_RESULTS'; do
  [ -e \"\$f\" ] && ls -lh \"\$f\"
done
tail -n 120 '$REMOTE_LOG' 2>/dev/null || true"
}

fetch_artifacts() {
  cd "$LOCAL_ROOT"
  mkdir -p cascadiav3/reports cascadiav3/logs
  for rel in \
    "$NO_SEARCH_REPORT" "$NO_SEARCH_DECISIONS" "$NO_SEARCH_GAME_RESULTS" "$NO_SEARCH_SUMMARY" \
    "$SEARCH_REPORT" "$SEARCH_DECISIONS" "$SEARCH_GAME_RESULTS" "$SEARCH_SUMMARY" \
    "cascadiav3/logs/${JOB_SLUG}_job.log"
  do
    if ssh -p "$SSH_PORT" "$REMOTE" "[ -e '$REMOTE_ROOT/$rel' ]"; then
      mkdir -p "$(dirname "$rel")"
      rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$rel" "$(dirname "$rel")/"
    fi
  done
}

stop_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ]; then
  pid=\"\$(cat '$REMOTE_PID')\"
  if kill -0 \"\$pid\" 2>/dev/null; then
    kill -TERM -\"\$pid\" 2>/dev/null || kill -TERM \"\$pid\" 2>/dev/null || true
  fi
fi
rm -f '$REMOTE_PID'
echo stopped"
}

case "$ACTION" in
  launch)
    launch_job
    ;;
  status)
    status_job
    ;;
  fetch)
    fetch_artifacts
    ;;
  stop)
    stop_job
    ;;
  *)
    echo "usage: $0 {launch|status|fetch|stop}" >&2
    exit 2
    ;;
esac
