#!/usr/bin/env bash
set -euo pipefail

# Collect every output that currently exists. Partial collection is normal;
# missing shards can be relaunched on any host.

FLEET_TAG="${1:?usage: fleet_wildlife_exact_collect.sh <tag> [status|collect]}"
ACTION="${2:-status}"
HOSTS="${HOSTS:-john1 john2 john3 john4}"
REMOTE_ROOT="${REMOTE_ROOT:-~/cascadia}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/cascadiav3/fleet/collected_${FLEET_TAG}"
mkdir -p "$DEST"
read -r -a host_list <<< "$HOSTS"

is_local_host() {
  case "$1" in
    john1|localhost|127.0.0.1|"$(hostname -s)") return 0 ;;
    *) return 1 ;;
  esac
}

collected=()
for host in "${host_list[@]}"; do
  output="cascadiav3/fleet_outputs/$FLEET_TAG/shard_${host}.json"
  if is_local_host "$host"; then
    if [ -s "$ROOT/$output" ]; then
      echo "$host: complete"
      if [ "$ACTION" = "collect" ]; then
        cp "$ROOT/$output" "$DEST/shard_${host}.json"
        collected+=("$DEST/shard_${host}.json")
      fi
    else
      pid_file="$ROOT/cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.pid"
      if [ -s "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$host: running pid $(cat "$pid_file")"
      else
        echo "$host: no output"
      fi
    fi
  else
    if ssh "$host" "test -s $REMOTE_ROOT/$output"; then
      echo "$host: complete"
      if [ "$ACTION" = "collect" ]; then
        rsync -a "$host:$REMOTE_ROOT/$output" "$DEST/shard_${host}.json"
        collected+=("$DEST/shard_${host}.json")
      fi
    else
      echo "$host: no output yet"
    fi
  fi
done

if [ "$ACTION" = "collect" ] && [ "${#collected[@]}" -gt 0 ]; then
  PYTHONPATH="$ROOT" python3 -m tools.merge_wildlife_catalogs \
    --output "$DEST/catalog.json" "${collected[@]}"
  echo "merged ${#collected[@]} available shard(s) into $DEST/catalog.json"
fi
