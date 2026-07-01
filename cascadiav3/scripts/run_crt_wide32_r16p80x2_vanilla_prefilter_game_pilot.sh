#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

CHECKPOINTS="${CHECKPOINTS:-cascadiav3/checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_pilot.pt,cascadiav3/checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_pilot.pt,cascadiav3/checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_pilot.pt}"
FIRST_SEED="${FIRST_SEED:-2026170000}"
GAMES="${GAMES:-4}"
SEEDS="${SEEDS:-}"
RETAIN_K="${RETAIN_K:-16}"
MAX_ACTIONS="${MAX_ACTIONS:-32}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
SHADOW_FULL_SEARCH="${SHADOW_FULL_SEARCH:-1}"
INCLUDE_FULL_SEARCH_BASELINE="${INCLUDE_FULL_SEARCH_BASELINE:-1}"
FULL_BASELINE_WORKERS="${FULL_BASELINE_WORKERS:-1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p80x2-vanilla-prefilter-game-pilot-v1}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.json}"
DECISIONS_OUT="${DECISIONS_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_decisions.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_summary.md}"
BINARY="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
JOB_SLUG="${JOB_SLUG:-r16p80x2_vanilla_prefilter_game_pilot}"

REMOTE_LOG_DIR="$REMOTE_ROOT/cascadiav3/logs"
REMOTE_JOB="$REMOTE_LOG_DIR/${JOB_SLUG}_job.sh"
REMOTE_LOG="$REMOTE_LOG_DIR/${JOB_SLUG}_job.log"
REMOTE_PID="$REMOTE_LOG_DIR/${JOB_SLUG}_job.pid"

sync_sources() {
  cd "$LOCAL_ROOT"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/crates' '$REMOTE_LOG_DIR'"
  rsync -az -e "$RSYNC_SSH" Cargo.toml Cargo.lock "$REMOTE:$REMOTE_ROOT/"
  rsync -az -e "$RSYNC_SSH" --delete \
    --exclude 'target/' \
    crates/cascadia-game/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-game/"
  rsync -az -e "$RSYNC_SSH" --delete \
    --exclude 'target/' \
    crates/cascadia-sim/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-sim/"
  rsync -az -e "$RSYNC_SSH" --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'logs/' \
    --exclude 'target/' \
    cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
}

write_remote_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
  ssh -p "$SSH_PORT" "$REMOTE" "cat > '$REMOTE_JOB'" <<REMOTE_JOB
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
. ~/.cargo/env 2>/dev/null || true
export BLAKE3_NO_ASM=1
if [ -x /home/john0/.local/bin/zig-cc ]; then
  export CC=/home/john0/.local/bin/zig-cc
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
fi

echo "[game-pilot] started \$(date -Is)"
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v

extra_args=()
if [ '$SHADOW_FULL_SEARCH' = '1' ]; then
  extra_args+=(--shadow-full-search)
fi
if [ '$INCLUDE_FULL_SEARCH_BASELINE' = '1' ]; then
  extra_args+=(--include-full-search-baseline)
fi
if [ -n '$SEEDS' ]; then
  extra_args+=(--seeds '$SEEDS')
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_game_pilot \
  --binary '$BINARY' \
  --checkpoints '$CHECKPOINTS' \
  --first-seed '$FIRST_SEED' \
  --games '$GAMES' \
  --retain-k '$RETAIN_K' \
  --max-actions '$MAX_ACTIONS' \
  --rollouts-per-action '$ROLLOUTS_PER_ACTION' \
  --rollout-top-k '$ROLLOUT_TOP_K' \
  --device cuda \
  --full-baseline-workers '$FULL_BASELINE_WORKERS' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --decisions-out '$DECISIONS_OUT' \
  --summary-out '$SUMMARY_OUT' \
  "\${extra_args[@]}"

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
echo "[game-pilot] completed \$(date -Is)"
REMOTE_JOB
  ssh -p "$SSH_PORT" "$REMOTE" "chmod 700 '$REMOTE_JOB'"
}

launch_job() {
  sync_sources
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
ps -eo pid=,ppid=,args= | awk '
  /torch_prefilter_game_pilot/ && !/awk/ {print \"matching python pid \" \$1 \" ppid \" \$2}
  /cascadiav3-real-root-exporter/ && /interactive-policy-game/ && !/awk/ {print \"matching simulator pid \" \$1 \" ppid \" \$2}
'
ls -lh '$REPORT' '$DECISIONS_OUT' '$SUMMARY_OUT' 2>/dev/null || true
tail -n 120 '$REMOTE_LOG' 2>/dev/null || true"
}

stop_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ]; then
  pid=\"\$(cat '$REMOTE_PID')\"
  if kill -0 \"\$pid\" 2>/dev/null; then
    kill -TERM -\"\$pid\" 2>/dev/null || kill -TERM \"\$pid\" 2>/dev/null || true
    sleep 2
    kill -KILL -\"\$pid\" 2>/dev/null || kill -KILL \"\$pid\" 2>/dev/null || true
  fi
fi
pids=\"\$(ps -eo pid=,args= | awk '
  /torch_prefilter_game_pilot/ && !/awk/ {print \$1}
  /cascadiav3-real-root-exporter/ && /interactive-policy-game/ && !/awk/ {print \$1}
' | sort -u | tr '\n' ' ')\"
if [ -n \"\$pids\" ]; then
  kill -TERM \$pids 2>/dev/null || true
  sleep 2
  kill -KILL \$pids 2>/dev/null || true
fi
rm -f '$REMOTE_PID'
echo stopped"
}

fetch_artifacts() {
  cd "$LOCAL_ROOT"
  mkdir -p cascadiav3/reports cascadiav3/logs
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/logs/" cascadiav3/logs/
}

case "$ACTION" in
  launch)
    launch_job
    ;;
  status)
    status_job
    ;;
  stop)
    stop_job
    ;;
  fetch)
    fetch_artifacts
    ;;
  *)
    echo "usage: $0 {launch|status|stop|fetch}" >&2
    exit 2
    ;;
esac
