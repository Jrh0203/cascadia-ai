#!/usr/bin/env bash
set -euo pipefail

# Fetch the complete corrected-rules n1024 artifacts from john0, verify every
# transfer, and reconcile the paired category attribution with the canonical
# total-score verdict. This is read-only on john0.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
DEST="${DEST:-$ROOT/cascadiav3/reports/rules_20260709_rebaseline_complete}"
REMOTE_ROOT="/home/john0/cascadia"
REPORT_DIR="cascadiav3/reports"
LOG_DIR="cascadiav3/logs"
SOURCE_REVISION="d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135"
CYCLE4="rules_20260709_cycle4_n1024_d16"
DISTQ="rules_20260709_distq_k8_n1024_d16"
TOTAL_VERDICT="rules_20260709_rebaseline_verdict"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/cascadiav3/src"

command -v ssh >/dev/null
command -v rsync >/dev/null
command -v shasum >/dev/null
test -x "$PYTHON"
mkdir -p "$DEST"

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

ssh john0 /bin/bash -s -- \
  "$REPORT_DIR" "$LOG_DIR" "$CYCLE4" "$DISTQ" "$TOTAL_VERDICT" <<'REMOTE'
set -euo pipefail
report_dir=$1
log_dir=$2
cycle4=$3
distq=$4
total_verdict=$5
cd /home/john0/cascadia
main_pid=$(cat "$log_dir/rules_20260709_rebaseline.pid")
raw_watcher_pid=$(cat "$log_dir/rules_20260709_remaining_raw_watcher.pid")
if kill -0 "$main_pid" 2>/dev/null; then
  echo "corrected-rules rebaseline is still active: $main_pid" >&2
  exit 2
fi
if kill -0 "$raw_watcher_pid" 2>/dev/null; then
  echo "n1024 raw-ledger watcher is still active: $raw_watcher_pid" >&2
  exit 2
fi
for tag in "$cycle4" "$distq"; do
  for suffix in .json .md _decisions.jsonl _games.jsonl _category_summary.json; do
    test -s "$report_dir/$tag$suffix"
  done
done
test -s "$report_dir/$total_verdict.json"
test -s "$report_dir/$total_verdict.md"
python3 - "$report_dir" "$cycle4" "$distq" "$total_verdict" <<'PY'
import json
import sys
from pathlib import Path

report_dir = Path(sys.argv[1])
cycle4, distq, total_verdict = sys.argv[2:]
for tag in (cycle4, distq):
    report = json.load(open(report_dir / f"{tag}.json", encoding="utf-8"))
    category = json.load(
        open(report_dir / f"{tag}_category_summary.json", encoding="utf-8")
    )
    games = [
        json.loads(line)
        for line in (report_dir / f"{tag}_games.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if report.get("status") != "pass" or category.get("status") != "complete":
        raise SystemExit(f"non-passing n1024 artifact: {tag}")
    if len(games) != 100 or any(row.get("type") != "gumbel_game_done" for row in games):
        raise SystemExit(f"incomplete n1024 game ledger: {tag}")
verdict = json.load(open(report_dir / f"{total_verdict}.json", encoding="utf-8"))
if verdict.get("status") != "pass":
    raise SystemExit("canonical rebaseline verdict is not passing")
PY
REMOTE

files=(
  "$CYCLE4.json"
  "$CYCLE4.md"
  "${CYCLE4}_decisions.jsonl"
  "${CYCLE4}_games.jsonl"
  "${CYCLE4}_category_summary.json"
  "$DISTQ.json"
  "$DISTQ.md"
  "${DISTQ}_decisions.jsonl"
  "${DISTQ}_games.jsonl"
  "${DISTQ}_category_summary.json"
  "$TOTAL_VERDICT.json"
  "$TOTAL_VERDICT.md"
)

for file in "${files[@]}"; do
  remote_sha="$(
    ssh john0 "shasum -a 256 '$REMOTE_ROOT/$REPORT_DIR/$file'" | awk '{print $1}'
  )"
  rsync -a --partial "john0:$REMOTE_ROOT/$REPORT_DIR/$file" "$DEST/"
  local_sha="$(sha256 "$DEST/$file")"
  if [ "$local_sha" != "$remote_sha" ]; then
    echo "hash mismatch after fetching $file" >&2
    exit 1
  fi
done

"$PYTHON" -m cascadiav3.compare_game_categories \
  --left-report "$DEST/$DISTQ.json" \
  --left-games "$DEST/${DISTQ}_games.jsonl" \
  --right-report "$DEST/$CYCLE4.json" \
  --right-games "$DEST/${CYCLE4}_games.jsonl" \
  --source-revision "$SOURCE_REVISION" \
  --label "distq_k8 - cycle4 at corrected-rules n1024/d16" \
  --total-verdict "$DEST/$TOTAL_VERDICT.json" \
  --out "$DEST/rules_20260709_n1024_category_verdict.json" \
  --summary-out "$DEST/rules_20260709_n1024_category_verdict.md"

verdict_sha="$(sha256 "$DEST/rules_20260709_n1024_category_verdict.json")"
echo "[n1024-verdict-fetch] complete category_verdict_sha256=$verdict_sha"
