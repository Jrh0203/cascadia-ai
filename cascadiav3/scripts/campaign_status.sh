#!/usr/bin/env bash
# One-command, read-only live status snapshot of the whole campaign
# (AGENTS.md: verify live state yourself; a busy pid is not liveness).
#
# Reports: local git sync + worktree, web UI health, john0 job pids with
# heartbeat freshness, HOLD files, raw-games ledger progress, GPU state, and
# mini fleet reachability. Never mutates anything.
set -u

section() { printf '\n== %s\n' "$*"; }

section "local ($(hostname -s)) $(date '+%F %T')"
if git fetch -q origin main 2>/dev/null; then
  head_rev=$(git rev-parse --short HEAD)
  origin_rev=$(git rev-parse --short origin/main)
  if [ "$head_rev" = "$origin_rev" ]; then
    echo "  git: HEAD == origin/main ($head_rev)"
  else
    echo "  git: HEAD $head_rev != origin/main $origin_rev  <-- DIVERGED"
  fi
else
  echo "  git: fetch failed (offline?); HEAD $(git rev-parse --short HEAD)"
fi
dirty=$(git status --short | wc -l | tr -d ' ')
[ "$dirty" = "0" ] && echo "  worktree: clean" || echo "  worktree: $dirty entries  <-- NOT CLEAN"
echo "  disk: $(df -h / | awk 'NR==2{print $4" free"}')"
if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8787/; then
  echo "  web ui: http://127.0.0.1:8787 OK"
else
  echo "  web ui: DOWN"
fi

section "john0"
ssh -o ConnectTimeout=5 john0 /bin/bash -s <<'REMOTE' 2>/dev/null || echo "  UNREACHABLE"
cd "$HOME/cascadia" || exit 1
echo "  disk: $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')"
smi=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
"$smi" --query-gpu=utilization.gpu,power.draw,memory.used \
  --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /'
now=$(date +%s)
for f in cascadiav3/logs/*.pid; do
  [ -e "$f" ] || continue
  pid=$(head -1 "$f" 2>/dev/null | tr -dc '0-9')
  [ -n "$pid" ] || continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "  pid ALIVE  $(basename "$f" .pid) ($pid)"
  else
    # Dead pid files older than 3 days are stale noise; skip them.
    age=$(( now - $(stat -c %Y "$f") ))
    [ "$age" -lt 259200 ] && echo "  pid dead   $(basename "$f" .pid) ($pid)"
  fi
done
for h in cascadiav3/logs/HOLD_*; do
  [ -e "$h" ] && echo "  HOLD: $h"
done
for l in cascadiav3/logs/*mirror*.log cascadiav3/logs/*watcher*.log cascadiav3/logs/*waiter*.log; do
  [ -e "$l" ] || continue
  age=$(( now - $(stat -c %Y "$l") ))
  echo "  heartbeat: $(basename "$l") updated ${age}s ago"
done
for d in cascadiav3/reports/*_raw_games; do
  [ -d "$d" ] || continue
  n=$(find "$d" -maxdepth 1 -name 'gumbel_game_seed_*.jsonl' | wc -l)
  echo "  raw ledger: $(basename "$d") = $n seed files"
done
REMOTE

for host in john2 john3 john4; do
  section "$host"
  ssh -o ConnectTimeout=5 "$host" '
    echo "  disk: $(df -h / | awk '"'"'NR==2{print $4" free"}'"'"')"
    n=$(ps aux | grep -E "real-root-exporter|torch_inference_bridge" | grep -v grep | wc -l | tr -d " ")
    echo "  cascadia processes: $n"
  ' 2>/dev/null || echo "  UNREACHABLE"
done
echo
