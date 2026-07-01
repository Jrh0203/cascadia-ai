#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_SLUG="${JOB_SLUG:-r16p80x2_vanilla_prefilter_game_k24_nonshadow20}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p80x2-vanilla-prefilter-game-k24-nonshadow20-v1}"
export FIRST_SEED="${FIRST_SEED:-2026171000}"
export GAMES="${GAMES:-20}"
export RETAIN_K="${RETAIN_K:-24}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export SHADOW_FULL_SEARCH="${SHADOW_FULL_SEARCH:-0}"
export INCLUDE_FULL_SEARCH_BASELINE="${INCLUDE_FULL_SEARCH_BASELINE:-0}"
export FULL_BASELINE_WORKERS="${FULL_BASELINE_WORKERS:-1}"
export REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_k24_nonshadow20.json}"
export DECISIONS_OUT="${DECISIONS_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_k24_nonshadow20_decisions.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_k24_nonshadow20_summary.md}"

exec "$SCRIPT_DIR/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh" "$@"
