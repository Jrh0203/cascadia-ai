#!/usr/bin/env bash
set -euo pipefail

# R0.1 sigma-calibration sweep (preregistered 2026-07-10, EXPERIMENT_LOG
# 21:55): cycle4 scalar, corrected rules, n256/top16/d4, K1 on, samples 8,
# jobs12. 8 arms = c_scale {0.05, 0.1, 0.25, 1.0} x sigma-norm
# {minmax, topk:8} on 25 paired seeds 2027072100..24 (selection block). The
# (c_scale=1.0, minmax) arm IS the incumbent champion config and doubles as
# the paired baseline. Preregistered screen rule: the best candidate arm by
# paired mean delta proceeds to a 100-seed confirm on 2027072200..99
# (touched once) iff its screen mean >= +0.25; the confirm verdict is the
# R0.1 kill test (CI+ -> n1024 confirmation; else close R0.1).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027072100}"
GAMES="${GAMES:-25}"
CONFIRM_FIRST_SEED="${CONFIRM_FIRST_SEED:-2027072200}"
CONFIRM_GAMES="${CONFIRM_GAMES:-100}"
SCREEN_FLOOR="${SCREEN_FLOOR:-0.25}"
JOBS="${JOBS:-12}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="sigma_sweep_20260710_n256"
CONFIRM_TAG="sigma_confirm_20260710_n256"

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
    echo "[sigma-sweep] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --gumbel-sigma-norm (stale build?)" \
  bash -c "\"$BINARY\" --gumbel-sigma-norm bogus --help 2>&1 | grep -q 'invalid --gumbel-sigma-norm'"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[sigma-sweep] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[sigma-sweep] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

hold_gate() {
  while [ -e "$LOG_DIR/HOLD_sigma_sweep" ]; do
    echo "[sigma-sweep] $(date '+%F %T') holding on HOLD_sigma_sweep"
    sleep 60
  done
}

report_matches() {
  local report="$1"
  local c_scale="$2"
  local norm="$3"
  local first_seed="$4"
  local games="$5"
  [ -s "$report" ] && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" \
    "$first_seed" "$games" "$c_scale" "$norm" <<'PY'
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
    and search.get("n_simulations") == 256
    and search.get("top_m") == 16
    and search.get("depth_rounds") == 1
    and search.get("determinizations") == 4
    and search.get("market_decision_samples") == 8
    and search.get("blend_weight") == 0.5
    and search.get("k_interior") == 16
    and search.get("exact_endgame_turns") == 1
    and search.get("c_visit") == 50.0
    and search.get("c_scale") == float(sys.argv[6])
    and search.get("sigma_norm") == sys.argv[7]
    and search.get("paired_rollouts") is False
    else 1
)
PY
}

run_arm() {
  local c_scale="$1"
  local norm="$2"
  local arm_tag="$3"
  local first_seed="$4"
  local games="$5"
  local report="$REPORT_DIR/${arm_tag}.json"
  if report_matches "$report" "$c_scale" "$norm" "$first_seed" "$games"; then
    echo "[sigma-sweep] reuse $report"
    return
  fi
  hold_gate
  echo "[sigma-sweep] $(date '+%F %T') arm $arm_tag (c_scale=$c_scale norm=$norm seeds=${first_seed}x${games})"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$first_seed" \
    --games "$games" \
    --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations 256 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 1 \
    --gumbel-blend-weight 0.5 \
    --gumbel-c-scale "$c_scale" \
    --gumbel-sigma-norm "$norm" \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "$arm_tag" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${arm_tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${arm_tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/${arm_tag}.md"
}

verdict_for() {
  local arm_tag="$1"
  local baseline_report="$2"
  local out="$REPORT_DIR/${arm_tag}_verdict.json"
  if [ -s "$out" ]; then
    echo "[sigma-sweep] reuse $out"
    return
  fi
  "$PYTHON" -m cascadiav3.compare_search_shape \
    --baseline "$baseline_report" \
    --candidate "$REPORT_DIR/${arm_tag}.json" \
    --source-revision "$SOURCE_REVISION" \
    --varied-key c_scale \
    --varied-key sigma_norm \
    --out "$out" \
    --summary-out "$REPORT_DIR/${arm_tag}_verdict.md" >/dev/null
}

echo "[sigma-sweep] source=$SOURCE_REVISION screen=${FIRST_SEED}x${GAMES} confirm=${CONFIRM_FIRST_SEED}x${CONFIRM_GAMES} floor=$SCREEN_FLOOR"

# Fail-fast smoke on the most exotic arm: 1 game, throwaway seeds are NOT
# used — reuse the screen block's first seed (screen arms overwrite nothing:
# the smoke writes its own _smoke report).
run_arm 0.05 "topk:8" "${TAG}_smoke_cs005_tk8" "$FIRST_SEED" 1

# Screen arms (25 paired seeds each). Incumbent first.
run_arm 1.0  "minmax" "${TAG}_cs10_mm"   "$FIRST_SEED" "$GAMES"
run_arm 0.25 "minmax" "${TAG}_cs025_mm"  "$FIRST_SEED" "$GAMES"
run_arm 0.1  "minmax" "${TAG}_cs01_mm"   "$FIRST_SEED" "$GAMES"
run_arm 0.05 "minmax" "${TAG}_cs005_mm"  "$FIRST_SEED" "$GAMES"
run_arm 1.0  "topk:8" "${TAG}_cs10_tk8"  "$FIRST_SEED" "$GAMES"
run_arm 0.25 "topk:8" "${TAG}_cs025_tk8" "$FIRST_SEED" "$GAMES"
run_arm 0.1  "topk:8" "${TAG}_cs01_tk8"  "$FIRST_SEED" "$GAMES"
run_arm 0.05 "topk:8" "${TAG}_cs005_tk8" "$FIRST_SEED" "$GAMES"

INCUMBENT_REPORT="$REPORT_DIR/${TAG}_cs10_mm.json"
for slug in cs025_mm cs01_mm cs005_mm cs10_tk8 cs025_tk8 cs01_tk8 cs005_tk8; do
  verdict_for "${TAG}_${slug}" "$INCUMBENT_REPORT"
done

# Preregistered selection: best candidate arm by paired mean delta; proceed
# to the confirm block iff mean >= SCREEN_FLOOR.
SELECTION="$REPORT_DIR/${TAG}_selection.json"
"$PYTHON" - "$REPORT_DIR" "$TAG" "$SCREEN_FLOOR" "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path

report_dir, tag, floor, out_path = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
arms = {
    "cs025_mm": (0.25, "minmax"),
    "cs01_mm": (0.1, "minmax"),
    "cs005_mm": (0.05, "minmax"),
    "cs10_tk8": (1.0, "topk:8"),
    "cs025_tk8": (0.25, "topk:8"),
    "cs01_tk8": (0.1, "topk:8"),
    "cs005_tk8": (0.05, "topk:8"),
}
rows = []
for slug, (c_scale, norm) in arms.items():
    verdict = json.loads((report_dir / f"{tag}_{slug}_verdict.json").read_text(encoding="utf-8"))
    stats = verdict["paired_delta_stats"]
    rows.append(
        {
            "slug": slug,
            "c_scale": c_scale,
            "sigma_norm": norm,
            "mean": stats["mean"],
            "t_ci_low": stats["t_ci_low"],
            "t_ci_high": stats["t_ci_high"],
            "verdict": verdict["score_verdict"],
        }
    )
rows.sort(key=lambda row: row["mean"], reverse=True)
best = rows[0]
selection = {
    "screen_floor": floor,
    "arms_ranked": rows,
    "best": best,
    "proceed_to_confirm": best["mean"] >= floor,
}
Path(out_path).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(selection["best"] | {"proceed": selection["proceed_to_confirm"]}))
PY

PROCEED="$("$PYTHON" -c "import json,sys; s=json.load(open('$SELECTION')); print('yes' if s['proceed_to_confirm'] else 'no')")"
if [ "$PROCEED" = "yes" ]; then
  WINNER_SLUG="$("$PYTHON" -c "import json; print(json.load(open('$SELECTION'))['best']['slug'])")"
  WINNER_CSCALE="$("$PYTHON" -c "import json; print(json.load(open('$SELECTION'))['best']['c_scale'])")"
  WINNER_NORM="$("$PYTHON" -c "import json; print(json.load(open('$SELECTION'))['best']['sigma_norm'])")"
  echo "[sigma-sweep] $(date '+%F %T') screen winner $WINNER_SLUG (mean floor passed) -> 100-seed confirm"
  run_arm 1.0 "minmax" "${CONFIRM_TAG}_incumbent" "$CONFIRM_FIRST_SEED" "$CONFIRM_GAMES"
  run_arm "$WINNER_CSCALE" "$WINNER_NORM" "${CONFIRM_TAG}_${WINNER_SLUG}" "$CONFIRM_FIRST_SEED" "$CONFIRM_GAMES"
  "$PYTHON" -m cascadiav3.compare_search_shape \
    --baseline "$REPORT_DIR/${CONFIRM_TAG}_incumbent.json" \
    --candidate "$REPORT_DIR/${CONFIRM_TAG}_${WINNER_SLUG}.json" \
    --source-revision "$SOURCE_REVISION" \
    --varied-key c_scale \
    --varied-key sigma_norm \
    --out "$REPORT_DIR/${CONFIRM_TAG}_verdict.json" \
    --summary-out "$REPORT_DIR/${CONFIRM_TAG}_verdict.md" >/dev/null
  echo "[sigma-sweep] confirm verdict written: $REPORT_DIR/${CONFIRM_TAG}_verdict.json"
else
  echo "[sigma-sweep] no arm met the screen floor; R0.1 closes per preregistration"
fi

"$PYTHON" - "$LOG_DIR/${TAG}_complete.json" "$SELECTION" "$REPORT_DIR/${CONFIRM_TAG}_verdict.json" <<'PY'
import json
import os
import sys

selection = json.load(open(sys.argv[2], encoding="utf-8"))
confirm = None
if os.path.exists(sys.argv[3]):
    confirm_report = json.load(open(sys.argv[3], encoding="utf-8"))
    confirm = {
        "verdict": confirm_report["score_verdict"],
        "mean": confirm_report["paired_delta_stats"]["mean"],
        "t_ci_low": confirm_report["paired_delta_stats"]["t_ci_low"],
        "t_ci_high": confirm_report["paired_delta_stats"]["t_ci_high"],
    }
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "complete",
            "screen_best": selection["best"],
            "proceed_to_confirm": selection["proceed_to_confirm"],
            "confirm": confirm,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[sigma-sweep] complete"
