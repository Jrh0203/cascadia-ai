#!/usr/bin/env bash
set -u

# One-command digest of every experiment artifact from the current wave:
# concatenates the newest screen analyses, gate verdicts, coverage audit,
# and acceptance files into a single markdown report. Read-only; safe to
# run any time, anywhere the reports directory exists.
#
#   ssh john0 'bash /home/john0/cascadia/cascadiav3/scripts/morning_report.sh'

ROOT="${ROOT:-/home/john0/cascadia}"
REPORT_DIR="$ROOT/cascadiav3/reports"
OUT="${OUT:-$REPORT_DIR/morning_report_$(date '+%Y%m%d').md}"

{
  echo "# Morning Report — $(date '+%F %T')"
  echo
  echo "## Chain heartbeats (last line each)"
  echo
  for log in "$ROOT"/cascadiav3/logs/{screen_wave_20260712,refresh_gate_20260712,ghost_gate_20260712}.log; do
    if [ -s "$log" ]; then
      echo "- \`$(basename "$log")\`: $(tail -1 "$log")"
    fi
  done
  echo
  for md in \
    "$REPORT_DIR"/puzzle_screen_*_analysis.md \
    "$REPORT_DIR"/menu_coverage_*_analysis.md \
    "$REPORT_DIR"/gate_*_verdict.md; do
    if [ -s "$md" ]; then
      echo "---"
      echo
      echo "## $(basename "$md")"
      echo
      cat "$md"
      echo
    fi
  done
  echo "---"
  echo
  echo "Decision rules for every artifact above: EXPERIMENT_LOG entries"
  echo "2026-07-12 01:02 (screen wave), 01:20 (refresh gate), 01:25 (ghost"
  echo "gate). Screens rank; gates decide; nothing adopts without a CI+"
  echo "verdict on a registered block."
} > "$OUT"
echo "wrote $OUT"
cat "$OUT"
