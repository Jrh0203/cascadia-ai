#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

JOB_SLUG="${JOB_SLUG:-greedy_policy_pretrain}"
RAYON_THREADS="${RAYON_THREADS:-32}"

CORPUS_FORMAT="${CORPUS_FORMAT:-npz}"
COMPACT_DTYPE="${COMPACT_DTYPE:-float16}"
TENSOR_EXPORTER="${TENSOR_EXPORTER:-rust}"
TENSOR_COMPRESSION="${TENSOR_COMPRESSION:-deflate}"
KEEP_JSONL="${KEEP_JSONL:-0}"
FETCH_FIXTURES="${FETCH_FIXTURES:-0}"
TRAIN_SHARD_SEED_COUNT="${TRAIN_SHARD_SEED_COUNT:-0}"
VAL_SHARD_SEED_COUNT="${VAL_SHARD_SEED_COUNT:-0}"

TRAIN_JSONL="${TRAIN_JSONL:-cascadiav3/fixtures/greedy_policy_pretrain_train.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/greedy_policy_pretrain_train_manifest.json}"
VAL_JSONL="${VAL_JSONL:-cascadiav3/fixtures/greedy_policy_pretrain_val.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/greedy_policy_pretrain_val_manifest.json}"
TRAIN_NPZ="${TRAIN_NPZ:-cascadiav3/fixtures/greedy_policy_pretrain_train.${COMPACT_DTYPE}.npz}"
VAL_NPZ="${VAL_NPZ:-cascadiav3/fixtures/greedy_policy_pretrain_val.${COMPACT_DTYPE}.npz}"
TRAIN_TENSOR_REPORT="${TRAIN_TENSOR_REPORT:-cascadiav3/reports/greedy_policy_pretrain_train_tensor_shard.json}"
VAL_TENSOR_REPORT="${VAL_TENSOR_REPORT:-cascadiav3/reports/greedy_policy_pretrain_val_tensor_shard.json}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026190000}"
TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-1024}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026290000}"
VAL_SEED_COUNT="${VAL_SEED_COUNT:-128}"
PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
MAX_ACTIONS="${MAX_ACTIONS:-32}"
REGENERATE_CORPUS="${REGENERATE_CORPUS:-0}"

STEPS="${STEPS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.00030}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
MLP_DIM="${MLP_DIM:-512}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-16384}"
MAX_VAL_RECORDS="${MAX_VAL_RECORDS:-20000}"
SEED="${SEED:-20260660}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-greedy-policy-pretrain-v1}"
REPORT="${REPORT:-cascadiav3/reports/greedy_policy_pretrain.json}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/greedy_policy_pretrain.pt}"
RUN_GAME_BENCHMARK="${RUN_GAME_BENCHMARK:-0}"
BENCHMARK_FIRST_SEED="${BENCHMARK_FIRST_SEED:-2026990000}"
BENCHMARK_GAMES="${BENCHMARK_GAMES:-100}"
BENCHMARK_BASELINE_WORKERS="${BENCHMARK_BASELINE_WORKERS:-8}"
BENCHMARK_EXPERIMENT_ID="${BENCHMARK_EXPERIMENT_ID:-greedy-policy-complete-game-benchmark-v1}"
BENCHMARK_REPORT="${BENCHMARK_REPORT:-cascadiav3/reports/greedy_policy_game_benchmark.json}"
BENCHMARK_DECISIONS="${BENCHMARK_DECISIONS:-cascadiav3/reports/greedy_policy_game_benchmark_decisions.jsonl}"
BENCHMARK_SUMMARY="${BENCHMARK_SUMMARY:-cascadiav3/reports/greedy_policy_game_benchmark_summary.md}"

shard_path() {
  local path="$1"
  local shard_index="$2"
  local prefix suffix
  if [[ "$path" == *.npz ]]; then
    prefix="${path%.npz}"
    suffix="npz"
  elif [[ "$path" == *.json ]]; then
    prefix="${path%.json}"
    suffix="json"
  else
    prefix="$path"
    suffix=""
  fi
  if [ -n "$suffix" ]; then
    printf "%s.shard%04d.%s" "$prefix" "$shard_index" "$suffix"
  else
    printf "%s.shard%04d" "$prefix" "$shard_index"
  fi
}

npz_corpus_paths() {
  local npz_path="$1"
  local seed_count="$2"
  local shard_seed_count="$3"
  if [ "$shard_seed_count" -le 0 ] || [ "$seed_count" -le "$shard_seed_count" ]; then
    printf "%s" "$npz_path"
    return
  fi
  local remaining="$seed_count"
  local shard_index=0
  local sep=""
  while [ "$remaining" -gt 0 ]; do
    printf "%s%s" "$sep" "$(shard_path "$npz_path" "$shard_index")"
    sep=","
    if [ "$remaining" -le "$shard_seed_count" ]; then
      remaining=0
    else
      remaining=$((remaining - shard_seed_count))
    fi
    shard_index=$((shard_index + 1))
  done
}

case "$CORPUS_FORMAT" in
  jsonl)
    TRAIN_CORPUS="$TRAIN_JSONL"
    VAL_CORPUS="$VAL_JSONL"
    ;;
  npz)
    if [ "$TENSOR_EXPORTER" = "rust" ]; then
      TRAIN_CORPUS="$(npz_corpus_paths "$TRAIN_NPZ" "$TRAIN_SEED_COUNT" "$TRAIN_SHARD_SEED_COUNT")"
      VAL_CORPUS="$(npz_corpus_paths "$VAL_NPZ" "$VAL_SEED_COUNT" "$VAL_SHARD_SEED_COUNT")"
    else
      TRAIN_CORPUS="$TRAIN_NPZ"
      VAL_CORPUS="$VAL_NPZ"
    fi
    ;;
  *)
    echo "CORPUS_FORMAT must be jsonl or npz, got '$CORPUS_FORMAT'" >&2
    exit 2
    ;;
esac

if [ "$TENSOR_EXPORTER" = "rust" ] && [ "$COMPACT_DTYPE" != "float16" ]; then
  echo "TENSOR_EXPORTER=rust currently writes float16 shards; got COMPACT_DTYPE='$COMPACT_DTYPE'" >&2
  exit 2
fi

BINARY="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
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
    --exclude 'fixtures/' \
    --exclude 'reports/' \
    --exclude 'checkpoints/' \
    --exclude 'logs/' \
    --exclude 'target/' \
    cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR' '$REMOTE_ROOT/cascadiav3/fixtures' '$REMOTE_ROOT/cascadiav3/reports' '$REMOTE_ROOT/cascadiav3/checkpoints'"
}

write_remote_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
  ssh -p "$SSH_PORT" "$REMOTE" "cat > '$REMOTE_JOB'" <<REMOTE_JOB
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
. ~/.cargo/env 2>/dev/null || true
export BLAKE3_NO_ASM=1
export RAYON_NUM_THREADS='$RAYON_THREADS'
export OMP_NUM_THREADS='$RAYON_THREADS'
export MKL_NUM_THREADS='$RAYON_THREADS'
if [ -x /home/john0/.local/bin/zig-cc ]; then
  export CC=/home/john0/.local/bin/zig-cc
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
fi

echo "[greedy-pretrain] started \$(date -Is)"
echo "[greedy-pretrain] RAYON_NUM_THREADS=\$RAYON_NUM_THREADS OMP_NUM_THREADS=\$OMP_NUM_THREADS"
echo "[greedy-pretrain] train_seed_count=$TRAIN_SEED_COUNT val_seed_count=$VAL_SEED_COUNT plies=$PLIES_PER_SEED corpus_format=$CORPUS_FORMAT dtype=$COMPACT_DTYPE tensor_exporter=$TENSOR_EXPORTER tensor_compression=$TENSOR_COMPRESSION keep_jsonl=$KEEP_JSONL"
echo "[greedy-pretrain] train_shard_seed_count=$TRAIN_SHARD_SEED_COUNT val_shard_seed_count=$VAL_SHARD_SEED_COUNT"
echo "[greedy-pretrain] steps=$STEPS batch_size=$BATCH_SIZE run_game_benchmark=$RUN_GAME_BENCHMARK benchmark_games=$BENCHMARK_GAMES"
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}

generate_corpus_jsonl() {
  local first_seed="\$1"
  local seed_count="\$2"
  local jsonl_path="\$3"
  local manifest_path="\$4"
  '$BINARY' \
    --greedy-policy-corpus \
    --first-seed "\$first_seed" \
    --seed-count "\$seed_count" \
    --plies-per-seed '$PLIES_PER_SEED' \
    --max-actions '$MAX_ACTIONS' \
    --rayon-threads '$RAYON_THREADS' \
    --out "\$jsonl_path" \
    --manifest "\$manifest_path"
}

generate_corpus_npz_rust() {
  local first_seed="\$1"
  local seed_count="\$2"
  local npz_path="\$3"
  local manifest_path="\$4"
  local tensor_report="\$5"
  '$BINARY' \
    --greedy-policy-tensor-corpus \
    --first-seed "\$first_seed" \
    --seed-count "\$seed_count" \
    --plies-per-seed '$PLIES_PER_SEED' \
    --max-actions '$MAX_ACTIONS' \
    --rayon-threads '$RAYON_THREADS' \
    --tensor-compression '$TENSOR_COMPRESSION' \
    --out "\$npz_path" \
    --manifest "\$manifest_path"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.greedy_tensor_shards \
    --summarize-shard "\$npz_path" \
    --report "\$tensor_report" \
    > /dev/null
}

shard_path() {
  local path="\$1"
  local shard_index="\$2"
  local prefix suffix
  if [[ "\$path" == *.npz ]]; then
    prefix="\${path%.npz}"
    suffix="npz"
  elif [[ "\$path" == *.json ]]; then
    prefix="\${path%.json}"
    suffix="json"
  else
    prefix="\$path"
    suffix=""
  fi
  if [ -n "\$suffix" ]; then
    printf "%s.shard%04d.%s" "\$prefix" "\$shard_index" "\$suffix"
  else
    printf "%s.shard%04d" "\$prefix" "\$shard_index"
  fi
}

generate_corpus_npz_rust_sharded() {
  local label="\$1"
  local first_seed="\$2"
  local seed_count="\$3"
  local npz_path="\$4"
  local manifest_path="\$5"
  local tensor_report="\$6"
  local shard_seed_count="\$7"
  if [ "\$shard_seed_count" -le 0 ] || [ "\$seed_count" -le "\$shard_seed_count" ]; then
    generate_corpus_npz_rust "\$first_seed" "\$seed_count" "\$npz_path" "\$manifest_path" "\$tensor_report"
    return
  fi
  local remaining="\$seed_count"
  local offset=0
  local shard_index=0
  while [ "\$remaining" -gt 0 ]; do
    local this_count="\$shard_seed_count"
    if [ "\$remaining" -lt "\$this_count" ]; then
      this_count="\$remaining"
    fi
    local shard_first=\$((first_seed + offset))
    local shard_npz shard_manifest shard_report
    shard_npz="\$(shard_path "\$npz_path" "\$shard_index")"
    shard_manifest="\$(shard_path "\$manifest_path" "\$shard_index")"
    shard_report="\$(shard_path "\$tensor_report" "\$shard_index")"
    if [ '$REGENERATE_CORPUS' = '1' ] || [ ! -s "\$shard_npz" ]; then
      echo "[greedy-pretrain] generating \$label shard \$shard_index first_seed=\$shard_first seed_count=\$this_count"
      generate_corpus_npz_rust "\$shard_first" "\$this_count" "\$shard_npz" "\$shard_manifest" "\$shard_report"
    fi
    remaining=\$((remaining - this_count))
    offset=\$((offset + this_count))
    shard_index=\$((shard_index + 1))
  done
}

ensure_corpus() {
  local label="\$1"
  local first_seed="\$2"
  local seed_count="\$3"
  local jsonl_path="\$4"
  local manifest_path="\$5"
  local npz_path="\$6"
  local tensor_report="\$7"
  local shard_seed_count="\$8"
  local target_path="\$jsonl_path"
  if [ '$CORPUS_FORMAT' = 'npz' ]; then
    target_path="\$npz_path"
  fi
  if [ '$CORPUS_FORMAT' = 'npz' ] && [ '$TENSOR_EXPORTER' = 'rust' ] && [ "\$shard_seed_count" -gt 0 ] && [ "\$seed_count" -gt "\$shard_seed_count" ]; then
    generate_corpus_npz_rust_sharded "\$label" "\$first_seed" "\$seed_count" "\$npz_path" "\$manifest_path" "\$tensor_report" "\$shard_seed_count"
    return
  fi
  if [ '$REGENERATE_CORPUS' = '1' ] || [ ! -s "\$target_path" ]; then
    if [ '$CORPUS_FORMAT' = 'npz' ]; then
      if [ '$TENSOR_EXPORTER' = 'rust' ]; then
        echo "[greedy-pretrain] generating \$label Rust-native tensor shard"
        generate_corpus_npz_rust "\$first_seed" "\$seed_count" "\$npz_path" "\$manifest_path" "\$tensor_report"
      elif [ '$TENSOR_EXPORTER' = 'python-stream' ] && [ '$KEEP_JSONL' = '0' ]; then
        echo "[greedy-pretrain] streaming \$label corpus directly to tensor shard"
        '$BINARY' \
          --greedy-policy-corpus \
          --first-seed "\$first_seed" \
          --seed-count "\$seed_count" \
          --plies-per-seed '$PLIES_PER_SEED' \
          --max-actions '$MAX_ACTIONS' \
          --rayon-threads '$RAYON_THREADS' \
          --out - \
          --manifest "\$manifest_path" \
          | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.greedy_tensor_shards \
              --jsonl - \
              --out "\$npz_path" \
              --dtype '$COMPACT_DTYPE' \
              --report "\$tensor_report"
      else
        if [ '$TENSOR_EXPORTER' != 'python-file' ] && [ '$TENSOR_EXPORTER' != 'python-stream' ]; then
          echo "TENSOR_EXPORTER must be rust, python-stream, or python-file; got '$TENSOR_EXPORTER'" >&2
          exit 2
        fi
        if [ '$REGENERATE_CORPUS' = '1' ] || [ ! -s "\$jsonl_path" ]; then
          echo "[greedy-pretrain] generating \$label JSONL corpus"
          generate_corpus_jsonl "\$first_seed" "\$seed_count" "\$jsonl_path" "\$manifest_path"
        fi
        echo "[greedy-pretrain] compacting \$label corpus to tensor shard"
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.greedy_tensor_shards \
          --jsonl "\$jsonl_path" \
          --out "\$npz_path" \
          --dtype '$COMPACT_DTYPE' \
          --report "\$tensor_report"
      fi
    elif [ '$REGENERATE_CORPUS' = '1' ] || [ ! -s "\$jsonl_path" ]; then
      echo "[greedy-pretrain] generating \$label JSONL corpus"
      generate_corpus_jsonl "\$first_seed" "\$seed_count" "\$jsonl_path" "\$manifest_path"
    fi
  fi
}

ensure_corpus train '$TRAIN_FIRST_SEED' '$TRAIN_SEED_COUNT' '$TRAIN_JSONL' '$TRAIN_MANIFEST' '$TRAIN_NPZ' '$TRAIN_TENSOR_REPORT' '$TRAIN_SHARD_SEED_COUNT'
ensure_corpus val '$VAL_FIRST_SEED' '$VAL_SEED_COUNT' '$VAL_JSONL' '$VAL_MANIFEST' '$VAL_NPZ' '$VAL_TENSOR_REPORT' '$VAL_SHARD_SEED_COUNT'

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_greedy_policy_pretrain \
  --train '$TRAIN_CORPUS' \
  --val '$VAL_CORPUS' \
  --train-format '$CORPUS_FORMAT' \
  --val-format '$CORPUS_FORMAT' \
  --steps '$STEPS' \
  --batch-size '$BATCH_SIZE' \
  --lr '$LR' \
  --weight-decay '$WEIGHT_DECAY' \
  --hidden-dim '$HIDDEN_DIM' \
  --layers '$LAYERS' \
  --heads '$HEADS' \
  --mlp-dim '$MLP_DIM' \
  --grad-clip '$GRAD_CLIP' \
  --shuffle-buffer '$SHUFFLE_BUFFER' \
  --max-val-records '$MAX_VAL_RECORDS' \
  --seed '$SEED' \
  --device cuda \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --checkpoint '$CHECKPOINT'

if [ '$RUN_GAME_BENCHMARK' = '1' ]; then
  echo "[greedy-pretrain] running complete-game learned-policy vs greedy benchmark"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_greedy_policy_game_benchmark \
    --binary '$BINARY' \
    --checkpoint '$CHECKPOINT' \
    --first-seed '$BENCHMARK_FIRST_SEED' \
    --games '$BENCHMARK_GAMES' \
    --max-actions '$MAX_ACTIONS' \
    --baseline-workers '$BENCHMARK_BASELINE_WORKERS' \
    --device cuda \
    --experiment-id '$BENCHMARK_EXPERIMENT_ID' \
    --out '$BENCHMARK_REPORT' \
    --decisions-out '$BENCHMARK_DECISIONS' \
    --summary-out '$BENCHMARK_SUMMARY'
fi

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
echo "[greedy-pretrain] completed \$(date -Is)"
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
ps -eo pid=,ppid=,psr=,nlwp=,pcpu=,pmem=,etime=,args= | awk '
  /greedy_policy_pretrain/ && !/awk/ {print \"matching trainer pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" elapsed \" \$7}
  /torch_greedy_policy_game_benchmark/ && !/awk/ {print \"matching benchmark pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" elapsed \" \$7}
  /cascadiav3-real-root-exporter/ && /greedy-policy-corpus/ && !/awk/ {print \"matching exporter pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" elapsed \" \$7}
  /cascadiav3-real-root-exporter/ && /greedy-policy-tensor-corpus/ && !/awk/ {print \"matching tensor exporter pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" elapsed \" \$7}
  /cascadiav3-real-root-exporter/ && /interactive-policy-game/ && !/awk/ {print \"matching simulator pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" elapsed \" \$7}
'
for f in '$TRAIN_JSONL' '$VAL_JSONL' '$TRAIN_NPZ' '$VAL_NPZ' '$REPORT' '$CHECKPOINT' '$TRAIN_MANIFEST' '$VAL_MANIFEST' '$TRAIN_TENSOR_REPORT' '$VAL_TENSOR_REPORT' '$BENCHMARK_REPORT' '$BENCHMARK_DECISIONS' '$BENCHMARK_SUMMARY'; do
  [ -e \"\$f\" ] && ls -lh \"\$f\"
done
printf '%s\n' '$TRAIN_CORPUS,$VAL_CORPUS' | tr ',' '\n' | while read -r f; do
  [ -n \"\$f\" ] && [ -e \"\$f\" ] && ls -lh \"\$f\"
done || true
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
  /greedy_policy_pretrain/ && !/awk/ {print \$1}
  /torch_greedy_policy_game_benchmark/ && !/awk/ {print \$1}
  /cascadiav3-real-root-exporter/ && /greedy-policy-corpus/ && !/awk/ {print \$1}
  /cascadiav3-real-root-exporter/ && /greedy-policy-tensor-corpus/ && !/awk/ {print \$1}
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
  mkdir -p cascadiav3/reports cascadiav3/logs cascadiav3/checkpoints cascadiav3/fixtures
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/logs/" cascadiav3/logs/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/checkpoints/" cascadiav3/checkpoints/
  if [ "$FETCH_FIXTURES" = "1" ]; then
    rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/fixtures/" cascadiav3/fixtures/
  else
    rsync -az -e "$RSYNC_SSH" --include='*/' --include='*_manifest.json' --exclude='*' \
      "$REMOTE:$REMOTE_ROOT/cascadiav3/fixtures/" cascadiav3/fixtures/
  fi
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
