#!/usr/bin/env bash
set -euo pipefail

# CBDDB fleet collector — runs on the Mac. Reads the fleet ledger,
# reports per-shard progress, and once every shard's manifest exists,
# pulls shards to local staging and pushes them to john0's fixtures.
# The trainer consumes shards natively via comma-separated --train
# paths (ExpertTensorCorpus is multi-shard); no merge step is needed.
#
# Usage:
#   fleet_cbddb_collect.sh <cycle_tag> status   — progress only
#   fleet_cbddb_collect.sh <cycle_tag> collect  — pull + push to john0
#
# Optional env: JOHN0="john0", JOHN0_ROOT="/home/john0/cascadia".

CYCLE_TAG="${1:?usage: fleet_cbddb_collect.sh <cycle_tag> status|collect}"
ACTION="${2:?usage: fleet_cbddb_collect.sh <cycle_tag> status|collect}"
JOHN0="${JOHN0:-john0}"
JOHN0_ROOT="${JOHN0_ROOT:-/home/john0/cascadia}"

FLEET_DIR="$(cd "$(dirname "$0")/.." && pwd)/fleet"
LEDGER="$FLEET_DIR/cbddb_${CYCLE_TAG}_fleet.json"
test -s "$LEDGER"
HOSTS=$(python3 -c "import json;print(' '.join(s['host'] for s in json.load(open('$LEDGER'))['shards']))")

STAGING="$FLEET_DIR/staging_${CYCLE_TAG}"
mkdir -p "$STAGING"

all_done=1
for h in $HOSTS; do
  TAG="cbddb_${CYCLE_TAG}_shard_${h}"
  line=$(ssh "$h" "tail -2 ~/cascadia/cascadiav3/logs/${TAG}.log 2>/dev/null | tr '\n' ' '" || echo unreachable)
  done_flag=$(ssh "$h" "[ -s ~/cascadia/cascadiav3/fixtures/${TAG}_manifest.json ] && echo yes || echo no" || echo no)
  echo "[$h] manifest_ready=$done_flag :: $line"
  [ "$done_flag" = yes ] || all_done=0
done

[ "$ACTION" = status ] && exit 0
if [ "$all_done" -ne 1 ]; then
  echo "[fleet] not all shards ready — refusing to collect partial fleet" >&2
  exit 1
fi

for h in $HOSTS; do
  TAG="cbddb_${CYCLE_TAG}_shard_${h}"
  rsync -a "$h:~/cascadia/cascadiav3/fixtures/${TAG}_tensor.npz" \
           "$h:~/cascadia/cascadiav3/fixtures/${TAG}_manifest.json" \
           "$h:~/cascadia/cascadiav3/fixtures/${TAG}_decisions.jsonl" \
           "$STAGING/"
  # Integrity: recorded seed range and record count must match the ledger.
  python3 - "$STAGING/${TAG}_manifest.json" "$LEDGER" "$h" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); ledger = json.load(open(sys.argv[2]))
shard = next(s for s in ledger["shards"] if s["host"] == sys.argv[3])
first = m.get("first_seed"); count = m.get("seed_count")
assert first == shard["first_seed"] and count == shard["seed_count"], \
    f"{sys.argv[3]}: manifest {first}x{count} != ledger {shard['first_seed']}x{shard['seed_count']}"
print(f"[{sys.argv[3]}] verified {first}x{count}, records={m.get('record_count', m.get('records', '?'))}")
PY
done

rsync -a "$STAGING/" "$JOHN0:$JOHN0_ROOT/cascadiav3/fixtures/"
echo "[fleet] all shards staged and pushed to $JOHN0:$JOHN0_ROOT/cascadiav3/fixtures/"
echo "[fleet] trainer usage: --train <john0_shard.npz>,$(for h in $HOSTS; do printf 'cascadiav3/fixtures/cbddb_%s_shard_%s_tensor.npz,' "$CYCLE_TAG" "$h"; done | sed 's/,$//')"
