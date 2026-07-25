#!/usr/bin/env bash
set -u

# Concise live status. Process state is authoritative; historical markers,
# receipts, hashes, HOLD files, and ledgers are intentionally ignored.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 1

echo "== local =="
hostname
git status --short --branch 2>/dev/null | sed -n '1p' || true
change_count="$(git status --short 2>/dev/null | wc -l | tr -d ' ')"
echo "worktree changes: ${change_count:-unknown}"
ps aux \
  | grep -E 'real-root-exporter|torch_inference_bridge|torch_.*(train|benchmark)|wildlife_(solver|exact)|cascadia-api' \
  | grep -v -E 'grep|campaign_status|tail -F' \
  || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "== gpu =="
  nvidia-smi
fi

echo
echo "== fleet =="
for host in john0 john2 john3 john4; do
  (
    snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" \
      "hostname; ps aux \
        | grep -E 'real-root-exporter|torch_inference_bridge|torch_.*(train|benchmark)|wildlife_(solver|exact)|cascadia-api' \
        | grep -v -E 'grep|campaign_status|tail -F|bacalhau' \
        | head -8" \
      2>/dev/null)" || snapshot="unreachable"
    printf '%s:\n%s\n' "$host" "$snapshot"
  ) &
done
wait
