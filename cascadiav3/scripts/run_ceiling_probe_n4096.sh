#!/usr/bin/env bash
set -euo pipefail

# R3.6 mega-budget ceiling probe (preregistered 2026-07-11, EXPERIMENT_LOG
# 11:50): cycle4 scalar at n4096/top16/d16, K1 OFF (matching the stored
# baseline arm exactly), 25 games on seeds 2027070900..24 — the rebaseline
# battery block, added-arm pattern — paired per-seed against the stored
# champion n1024/d16 scores. Informative probe, NOT promotion evidence.
# Preregistered bands on the paired mean: >= +0.45 scaling lane OPEN;
# +0.15..+0.45 decelerating; <= +0.15 scaling lane effectively closed.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027070900}"
GAMES="${GAMES:-25}"
JOBS="${JOBS:-12}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
BASELINE_REPORT="${BASELINE_REPORT:-cascadiav3/reports/rules_20260709_cycle4_n1024_d16.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="ceiling_probe_n4096_20260711"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
preflight() {
  local label="$1"
  shift
  if ! "$@"; then
    echo "[ceiling-probe] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "baseline report missing: $BASELINE_REPORT" test -s "$BASELINE_REPORT"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[ceiling-probe] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[ceiling-probe] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

REPORT="$REPORT_DIR/${TAG}.json"
report_matches() {
  [ -s "$REPORT" ] && "$PYTHON" - "$REPORT" "$RULESET_ID" "$SOURCE_REVISION" \
    "$FIRST_SEED" "$GAMES" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected_seeds = list(range(int(sys.argv[4]), int(sys.argv[4]) + int(sys.argv[5])))
search = report.get("search", {})
raise SystemExit(
    0
    if report.get("status") == "pass"
    and report.get("ruleset_id") == sys.argv[2]
    and report.get("source_revision") == sys.argv[3]
    and report.get("seeds") == expected_seeds
    and report.get("control", {}).get("kind") == "none"
    and search.get("n_simulations") == 4096
    and search.get("top_m") == 16
    and search.get("depth_rounds") == 1
    and search.get("determinizations") == 16
    and search.get("market_decision_samples") == 8
    and search.get("blend_weight") == 0.5
    and search.get("k_interior") == 16
    and search.get("exact_endgame_turns") == 0
    else 1
)
PY
}

if report_matches; then
  echo "[ceiling-probe] reuse $REPORT"
else
  echo "[ceiling-probe] $(date '+%F %T') mega arm n4096/d16 (seeds ${FIRST_SEED}x${GAMES}, jobs $JOBS, K1 off)"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations 4096 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 16 \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 600000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "$TAG" \
    --out "$REPORT" \
    --decisions-out "$REPORT_DIR/${TAG}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${TAG}_games.jsonl" \
    --summary-out "$REPORT_DIR/${TAG}.md"
fi

# Paired verdict against the stored champion n1024/d16 per-seed scores.
# compare_search_shape would refuse the (bit-identical-by-test) revision
# mismatch, so the pairing is computed here, pinned by the preregistration.
"$PYTHON" - "$REPORT" "$BASELINE_REPORT" "$FIRST_SEED" "$GAMES" \
  "$REPORT_DIR/${TAG}_verdict.json" "$REPORT_DIR/${TAG}_verdict.md" <<'PY'
import json
import sys

sys.path.insert(0, "cascadiav3/src")
from cascadiav3.torch_benchmark_stats import paired_delta_stats

mega = json.load(open(sys.argv[1], encoding="utf-8"))
baseline = json.load(open(sys.argv[2], encoding="utf-8"))
seeds = list(range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4])))
mega_scores = {int(r["seed"]): float(r["mean_score_per_seat"]) for r in mega["candidate_per_seed"]}
base_scores = {int(r["seed"]): float(r["mean_score_per_seat"]) for r in baseline["candidate_per_seed"]}
missing = [s for s in seeds if s not in mega_scores or s not in base_scores]
if missing:
    raise SystemExit(f"per-seed coverage missing for {missing}")
deltas = [mega_scores[s] - base_scores[s] for s in seeds]
stats = paired_delta_stats(deltas)
mean = stats["mean"]
band = (
    "scaling_lane_open" if mean >= 0.45
    else "decelerating" if mean > 0.15
    else "scaling_lane_effectively_closed"
)
verdict = {
    "status": "pass",
    "scientific_eligibility": "informative_ceiling_probe_only",
    "ruleset_id": mega["ruleset_id"],
    "mega_source_revision": mega["source_revision"],
    "baseline_source_revision": baseline["source_revision"],
    "revision_note": "default-path serving bit-identical between revisions (regression-pinned)",
    "seeds": seeds,
    "baseline_mean_seat_score": sum(base_scores[s] for s in seeds) / len(seeds),
    "mega_mean_seat_score": sum(mega_scores[s] for s in seeds) / len(seeds),
    "paired_delta_stats": stats,
    "log_linear_prediction": 0.615,
    "preregistered_band": band,
    "mega_mean_decision_seconds": float(
        mega["strategies"]["gumbel-search"]["mean_total_decision_seconds"]
    ),
    "mega_wall_seconds": mega["candidate_wall_seconds"],
}
with open(sys.argv[5], "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)
    handle.write("\n")
lines = [
    "# Mega-Budget Ceiling Probe (R3.6, n4096/d16 vs n1024/d16)",
    "",
    f"Seeds: `{len(seeds)}` paired (rebaseline battery block, added arm)",
    f"Baseline (stored champion): `{verdict['baseline_mean_seat_score']:.4f}`",
    f"Mega arm: `{verdict['mega_mean_seat_score']:.4f}`",
    f"Paired delta: `{mean:+.4f}` (95% t-CI `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`)",
    f"Log-linear prediction for this 4x step: `+0.615`",
    f"Preregistered band: `{band}`",
    f"Mean decision seconds: `{verdict['mega_mean_decision_seconds']:.2f}`",
    "",
    "Informative probe only — never promotion evidence.",
]
with open(sys.argv[6], "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(json.dumps({"paired_delta": mean, "band": band}))
PY

echo "[ceiling-probe] complete: $REPORT_DIR/${TAG}_verdict.md"
