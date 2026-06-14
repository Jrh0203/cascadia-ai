#!/usr/bin/env bash
#
# Overnight Phase 2 runner — Option A from the plan.
#
# 50 self-play iterations with a 32→256 sim ramp. Each iter does:
#   1. Collect self-play games (all 4 seats use latest weights)
#   2. Train MLX over sliding window (bootstrap + last 8 iter shards)
#   3. Bench vs greedy opponents (20 games, AAAAA with-bonus)
#   4. Append metrics line + update state.json
#
# Designed for unattended overnight execution. Crash-resumable via
# state.json — restart picks up at last_completed_iter + 1. Generates a
# final morning_report.md with strength curve + final H2H vs NNUE champion.

# NOTE: deliberately NOT using `set -o pipefail`.
# Parsing pipelines like `grep | grep | sort | head -1` legitimately produce
# SIGPIPE on the upstream commands (head closes its stdin early). With
# pipefail+set -e the script aborts mid-iter — root cause of the
# overnight-v1 crashes at iters 24 and 30.
set -eu

# ── Configuration ──────────────────────────────────────────────────────

RUN_DIR="${RUN_DIR:-alphazero_v2_run}"
N_ITERS="${OVERNIGHT_N_ITERS:-80}"
SLIDING_WINDOW="${OVERNIGHT_SLIDING_WINDOW:-15}"
# v2 hyperparam pass (post-v1 overnight diagnosis):
#   v1: LR 3e-4 → 5e-5, epochs=4, K=8, games 30/18/10/6 → val_top1 saturated 0.99
#   v2: LR 1e-4 → 1e-5, epochs=2, K=15, games 50/30/18/12 → combat overfit
LR_START="${OVERNIGHT_LR_START:-1e-4}"
LR_END="${OVERNIGHT_LR_END:-1e-5}"
# SMOKE=1 shrinks games/epochs for a fast end-to-end verification.
SMOKE="${SMOKE:-0}"
PY="/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
CLI="./target/release/cascadia-cli"
CHAMPION_WEIGHTS="nnue_weights_v4opp_modal_iter3.bin"

# Sim / game / epoch schedule by iteration (1-indexed).
#
# v2 pass schedule (80 iters): bigger early-game data volume to delay
# overfit, fewer epochs per iter (combats memorization). Sim ramp delayed
# slightly so the network has more data at low sims before the cost of
# search dominates collect time.
sim_for_iter() {
    local i=$1
    if [[ "$SMOKE" == "1" ]]; then echo 8; return; fi
    if   (( i <= 30 )); then echo 32
    elif (( i <= 55 )); then echo 64
    elif (( i <= 75 )); then echo 128
    else echo 256
    fi
}
games_for_iter() {
    local i=$1
    if [[ "$SMOKE" == "1" ]]; then echo 4; return; fi
    # ~1.7× more games/iter than v1 to provide fresher data, dilute the
    # MCTS-memorization overfit, and grow the sliding window faster.
    if   (( i <= 30 )); then echo 50
    elif (( i <= 55 )); then echo 30
    elif (( i <= 75 )); then echo 18
    else echo 12
    fi
}
epochs_for_iter() {
    local i=$1
    if [[ "$SMOKE" == "1" ]]; then echo 1; return; fi
    # v2: 2 epochs / iter instead of 4. The sliding window grows over time,
    # so the cumulative epoch-equivalents on any given sample stays high
    # while per-iter memorization is reduced.
    echo 2
}

# Cosine LR schedule from LR_START to LR_END over N_ITERS.
# v2 default: 1e-4 → 1e-5 (one-third the LR of v1, to combat overfit).
lr_for_iter() {
    local i=$1
    "$PY" -c "
import math
i = $i
n = $N_ITERS
lr_start = float('$LR_START')
lr_end   = float('$LR_END')
frac = (i - 1) / max(1, n - 1)
print(f'{lr_end + (lr_start - lr_end) * 0.5 * (1 + math.cos(math.pi * frac)):.6e}')
"
}

# ── Setup ───────────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR" "$RUN_DIR/logs"
STATE_FILE="$RUN_DIR/state.json"
METRICS="$RUN_DIR/metrics.jsonl"
RUNLOG="$RUN_DIR/runner.log"

# Logging helper — every line gets a timestamp, written ONLY to runner.log.
# No tee → no SIGPIPE risk if the parent's stdout pipe closes.
log() {
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf '[%s] %s\n' "$ts" "$*" >> "$RUNLOG"
}

# Redirect script's own stdout/stderr to the runner log too, so any
# accidental println from sub-tools lands in the same place (and isn't
# tied to whatever parent pipe spawned us).
exec >> "$RUNLOG" 2>&1

# Atomic state.json write.
write_state() {
    local last_iter=$1
    cat > "${STATE_FILE}.tmp" <<EOF
{
  "schema": 1,
  "run_dir": "$RUN_DIR",
  "n_iters": $N_ITERS,
  "last_completed_iter": $last_iter,
  "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

# Determine starting iteration: resume from state.json if it exists.
START_ITER=1
if [[ -f "$STATE_FILE" ]]; then
    LAST=$(grep '"last_completed_iter"' "$STATE_FILE" | grep -oE '[0-9]+' || echo 0)
    START_ITER=$((LAST + 1))
    log "Resuming from iter $START_ITER (state.json last_completed_iter=$LAST)"
fi

# Sanity-check Phase 1 artifacts.
BOOTSTRAP_AZD="$RUN_DIR/bootstrap.azd"
ITER0_AZR="$RUN_DIR/az_iter000.azr"
if [[ ! -f "$BOOTSTRAP_AZD" ]]; then
    log "ERROR: $BOOTSTRAP_AZD not found. Run Phase 1 first."
    exit 1
fi
if [[ ! -f "$ITER0_AZR" ]]; then
    log "ERROR: $ITER0_AZR not found. Run Phase 1 first."
    exit 1
fi
if [[ ! -x "$CLI" ]]; then
    log "ERROR: $CLI not found / not executable. Build with --features v4-opp,v5-feat,czero-feat,az-v2,accelerate."
    exit 1
fi

# Initialize state.json if first run.
if [[ ! -f "$STATE_FILE" ]]; then
    write_state 0
fi

# Re-initialize metrics.jsonl if starting fresh.
if [[ "$START_ITER" -eq 1 && ! -f "$METRICS" ]]; then
    log "Fresh start at iter 1 — initializing metrics.jsonl"
    : > "$METRICS"
fi

log "═══ Overnight Phase 2 runner ═══"
log "RUN_DIR=$RUN_DIR  N_ITERS=$N_ITERS  start=$START_ITER  cli=$CLI"

OVERALL_START_S=$(date +%s)

# Health-gate state: consecutive low-val_top1 iters.
LOW_VAL_STREAK=0

# ── Main loop ──────────────────────────────────────────────────────────

for (( iter=START_ITER; iter<=N_ITERS; iter++ )); do
    iter_padded=$(printf "%03d" "$iter")
    prev_iter_padded=$(printf "%03d" "$((iter - 1))")
    sims=$(sim_for_iter "$iter")
    games=$(games_for_iter "$iter")
    epochs=$(epochs_for_iter "$iter")
    lr=$(lr_for_iter "$iter")

    log ""
    log "─── Iter $iter / $N_ITERS  (sims=$sims games=$games epochs=$epochs lr=$lr) ───"

    PREV_AZR="$RUN_DIR/az_iter${prev_iter_padded}.azr"
    NEW_AZR="$RUN_DIR/az_iter${iter_padded}.azr"
    SHARD_AZD="$RUN_DIR/iter${iter_padded}.azd"

    # ── 1) Collect self-play data ──
    iter_start_s=$(date +%s)
    log "  collect: $games games × $sims sims"
    CASCADIA_SCORING_CARDS=A,A,A,A,A "$CLI" "$games" \
        --az-collect --az-arch v2 \
        --az-weights "$PREV_AZR" \
        --az-sims "$sims" --temperature 1.0 \
        --az-parallel --az-threads 4 --az-min-sims-per-thread 8 \
        --score-target with-bonus \
        --random-seed \
        --out "$SHARD_AZD" \
        > "$RUN_DIR/logs/iter${iter_padded}_collect.log" 2>&1
    collect_end_s=$(date +%s)
    collect_s=$((collect_end_s - iter_start_s))
    samples=$(grep -oE "samples=[0-9]+" "$RUN_DIR/logs/iter${iter_padded}_collect.log" | head -1 | cut -d= -f2)
    log "  collect done: ${collect_s}s samples=$samples"

    # ── 2) Train MLX over sliding window ──
    train_start_s=$(date +%s)
    # Sliding window: bootstrap + last SLIDING_WINDOW iter shards.
    window_files=("$BOOTSTRAP_AZD")
    win_start=$(( iter - SLIDING_WINDOW + 1 ))
    (( win_start < 1 )) && win_start=1
    for (( w=win_start; w<=iter; w++ )); do
        w_padded=$(printf "%03d" "$w")
        window_files+=("$RUN_DIR/iter${w_padded}.azd")
    done
    log "  train: window=${#window_files[@]} files (last shards: $(basename "${window_files[${#window_files[@]}-1]}"))"

    "$PY" train_alphazero_mlx_v2.py \
        --samples "${window_files[@]}" \
        --init "$PREV_AZR" \
        --out "$NEW_AZR" \
        --channels 96 --blocks 6 --entity-dim 64 --sab 2 --heads 4 --hidden 128 \
        --epochs "$epochs" --batch-size 128 --lr "$lr" --warmup-frac 0.05 \
        --value-weight 1.0 --aux-weight 0.3 \
        --val-fraction 0.05 \
        > "$RUN_DIR/logs/iter${iter_padded}_train.log" 2>&1
    train_end_s=$(date +%s)
    train_s=$((train_end_s - train_start_s))
    # Pull the highest val_top1 across epochs (best checkpoint).
    val_top1=$(grep -oE "val_top1=[0-9.]+" "$RUN_DIR/logs/iter${iter_padded}_train.log" | \
               grep -oE "[0-9.]+$" | sort -rn | head -1)
    val_top1=${val_top1:-0.0}
    log "  train done: ${train_s}s val_top1=$val_top1"

    # Halt gate: 3 consecutive iters with val_top1 < 0.30 → abort.
    if "$PY" -c "import sys; sys.exit(0 if float('$val_top1') < 0.30 else 1)"; then
        LOW_VAL_STREAK=$((LOW_VAL_STREAK + 1))
        log "  WARNING: val_top1 < 0.30 (streak=$LOW_VAL_STREAK)"
        if (( LOW_VAL_STREAK >= 3 )); then
            log "  ABORTING: val_top1 < 0.30 for 3 consecutive iters."
            write_state "$((iter - 1))"
            exit 2
        fi
    else
        LOW_VAL_STREAK=0
    fi

    # ── 3) Bench: 10 games vs greedy at the iter's sim count (trend monitor) ──
    bench_start_s=$(date +%s)
    bench_log="$RUN_DIR/logs/iter${iter_padded}_bench.log"
    bench_games=10
    [[ "$SMOKE" == "1" ]] && bench_games=4
    CASCADIA_SCORING_CARDS=A,A,A,A,A "$CLI" "$bench_games" \
        --az --weights "$NEW_AZR" \
        --az-sims "$sims" \
        --az-parallel --az-threads 4 --az-min-sims-per-thread 8 \
        --score-target with-bonus \
        > "$bench_log" 2>&1 || true
    bench_end_s=$(date +%s)
    bench_s=$((bench_end_s - bench_start_s))
    # Extract base + with-bonus mean.
    bonus_mean=$(grep -A1 "With Habitat Bonus:" "$bench_log" | grep "Mean:" | head -1 | \
                 awk '{print $2}')
    base_mean=$(grep -B1 "With Habitat Bonus:" "$bench_log" | head -1 | \
                grep -A6 "Base Score" "$bench_log" | grep "Mean:" | head -1 | awk '{print $2}')
    # Fallback parsing in case the above is fragile.
    if [[ -z "${bonus_mean:-}" ]]; then
        bonus_mean=$(grep -oE "Mean: +[0-9.]+ \(\+" "$bench_log" | tail -1 | awk '{print $2}')
    fi
    bonus_mean=${bonus_mean:-0.0}
    base_mean=${base_mean:-0.0}
    log "  bench done: ${bench_s}s base=$base_mean bonus=$bonus_mean"

    # ── 4) Persist metrics + state ──
    iter_elapsed_s=$(( bench_end_s - iter_start_s ))
    metrics_line=$(cat <<EOF
{"iter":$iter,"sims":$sims,"games":$games,"samples":${samples:-0},"epochs":$epochs,"lr":"$lr","collect_s":$collect_s,"train_s":$train_s,"bench_s":$bench_s,"iter_s":$iter_elapsed_s,"val_top1":$val_top1,"base_mean":$base_mean,"bonus_mean":$bonus_mean,"weights":"az_iter${iter_padded}.azr"}
EOF
    )
    echo "$metrics_line" >> "$METRICS"
    write_state "$iter"

    overall_elapsed=$(( bench_end_s - OVERALL_START_S ))
    log "  iter $iter complete (${iter_elapsed_s}s); overall ${overall_elapsed}s"
done

OVERALL_END_S=$(date +%s)
TOTAL_S=$(( OVERALL_END_S - OVERALL_START_S ))
log ""
log "═══ Phase 2 main loop complete: $((N_ITERS - START_ITER + 1)) iters in ${TOTAL_S}s ═══"

# ── Final bench (vs greedy opponents, AAAAA with-bonus) ──
# 60 games × 128 sims gives ~25-30 min wall-clock; uses the BEST iter weights
# (highest bonus_mean across iters), not just the last iter — overfit
# late-iter checkpoints sometimes lose to mid-iter ones.
BEST_AZR=$("$PY" - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
mp = run / "metrics.jsonl"
if not mp.exists():
    print("")
    sys.exit(0)
ms = [json.loads(l) for l in open(mp) if l.strip()]
if not ms:
    print("")
    sys.exit(0)
best = max(ms, key=lambda m: m.get("bonus_mean", 0.0))
print(run / best["weights"])
PYEOF
)
if [[ -n "$BEST_AZR" && -f "$BEST_AZR" ]]; then
    log "Final bench with BEST iter ($BEST_AZR) @ 128 sims, 60 games..."
    bench_final_games=60
    [[ "$SMOKE" == "1" ]] && bench_final_games=4
    CASCADIA_SCORING_CARDS=A,A,A,A,A "$CLI" "$bench_final_games" \
        --az --weights "$BEST_AZR" \
        --az-sims 128 \
        --az-parallel --az-threads 4 --az-min-sims-per-thread 16 \
        --score-target with-bonus \
        > "$RUN_DIR/logs/final_bench.log" 2>&1 || true
    log "Final bench saved to logs/final_bench.log"
fi

# ── Morning report ─────────────────────────────────────────────────────
# Use a *quoted* heredoc (`'PYEOF'`) so backticks, $vars, and quotes in the
# Python source aren't expanded by the shell. Config is passed via argv.
"$PY" - "$RUN_DIR" "$N_ITERS" "$TOTAL_S" <<'PYEOF' > "$RUN_DIR/morning_report.md"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
n_iters_target = int(sys.argv[2])
total_s = int(sys.argv[3])

metrics = []
mp = run_dir / "metrics.jsonl"
if mp.exists():
    with open(mp) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

print("# Overnight Phase 2 Morning Report")
print()
print(f"- Run dir: `{run_dir}`")
print(f"- Iterations completed: **{len(metrics)} / {n_iters_target}**")
print(f"- Total wall-clock: **{total_s}s** ({total_s//3600}h {(total_s%3600)//60}m)")
print()

if not metrics:
    print("(No metrics — runner died before iter 1 finished. Check runner.log.)")
    raise SystemExit

best = max(metrics, key=lambda m: m.get("bonus_mean", 0.0))
final = metrics[-1]
print("## Headline")
print()
print(f"- Best iter: **#{best['iter']}** — bonus_mean **{best['bonus_mean']:.2f}** "
      f"base_mean {best['base_mean']:.2f} val_top1 {best['val_top1']:.3f} "
      f"(weights: `{best['weights']}`)")
print(f"- Final iter: **#{final['iter']}** — bonus_mean {final['bonus_mean']:.2f} "
      f"base_mean {final['base_mean']:.2f} val_top1 {final['val_top1']:.3f}")
print("- Phase 1 baseline (greedy bootstrap): bonus_mean 79.3 base_mean 75.2")
print("- NNUE+MCE champion at AAAAA: bonus_mean **95.94** "
      "(`nnue_weights_v4opp_modal_iter3.bin` + `mce_wide_v1`)")
print()
delta_vs_phase1 = best["bonus_mean"] - 79.3
delta_vs_champ = best["bonus_mean"] - 95.94
print(f"- **vs Phase 1 baseline: {delta_vs_phase1:+.1f} pts**")
print(f"- **vs NNUE champion: {delta_vs_champ:+.1f} pts**")
print()

print("## Strength curve (per iter)")
print()
print("| iter | sims | games | samples | val_top1 | base_mean | bonus_mean | iter_s |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in metrics:
    print(f"| {m['iter']} | {m['sims']} | {m['games']} | {m['samples']} | "
          f"{m['val_top1']:.3f} | {m['base_mean']:.2f} | {m['bonus_mean']:.2f} | "
          f"{m['iter_s']} |")
print()

print("## Time breakdown")
print()
total_collect = sum(m["collect_s"] for m in metrics)
total_train = sum(m["train_s"] for m in metrics)
total_bench = sum(m["bench_s"] for m in metrics)
total = total_collect + total_train + total_bench
print(f"- Collect: **{total_collect}s** ({100*total_collect/max(1,total):.1f}%)")
print(f"- Train: **{total_train}s** ({100*total_train/max(1,total):.1f}%)")
print(f"- Bench: **{total_bench}s** ({100*total_bench/max(1,total):.1f}%)")
print()

final_bench_log = run_dir / "logs" / "final_bench.log"
if final_bench_log.exists():
    print("## Final 100-game bench @ 256 sims")
    print()
    text = final_bench_log.read_text()
    lines = text.splitlines()
    # Capture the block from "Results (" through "Score Distribution".
    start = None
    end = None
    for i, ln in enumerate(lines):
        if start is None and ln.lstrip().startswith("Results ("):
            start = i
        if start is not None and "Score Distribution" in ln:
            end = i
            break
    if start is not None:
        slice_end = end if end is not None else min(len(lines), start + 60)
        print("```")
        for ln in lines[start:slice_end]:
            print(ln)
        print("```")
    print()

print("## Notes")
print()
print("- All checkpoints saved as `az_iterNNN.azr` in the run dir.")
print("- Self-play shards saved as `iterNNN.azd`.")
print("- Per-iter logs in `logs/iterNNN_{collect,train,bench}.log`.")
print("- Runner log: `runner.log`.")
PYEOF

log "Morning report written to $RUN_DIR/morning_report.md"
log "═══ All done ═══"
