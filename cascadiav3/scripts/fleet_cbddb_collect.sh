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
  # Integrity: seed range + ruleset from metadata.seed_domain must match
  # the ledger; npz sha256 must match the manifest checksum; no skipped
  # seeds tolerated.
  python3 - "$STAGING/${TAG}_manifest.json" "$STAGING/${TAG}_tensor.npz" "$LEDGER" "$h" <<'PY'
import hashlib, json, re, sys
m = json.load(open(sys.argv[1])); md = m["metadata"]
ledger = json.load(open(sys.argv[3]))
shard = next(s for s in ledger["shards"] if s["host"] == sys.argv[4])
dom = md["seed_domain"]
first = int(re.search(r"first_seed=(\d+)", dom).group(1))
count = int(re.search(r"seed_count=(\d+)", dom).group(1))
assert (first, count) == (shard["first_seed"], shard["seed_count"]), \
    f"{sys.argv[4]}: manifest {first}x{count} != ledger {shard['first_seed']}x{shard['seed_count']}"
assert md["ruleset_id"] == "cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19", md["ruleset_id"]
assert md.get("generation_skipped_seeds") in (None, []), f"skipped seeds: {md['generation_skipped_seeds']}"
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert digest == m["checksum"], f"{sys.argv[4]}: npz sha256 mismatch"
print(f"[{sys.argv[4]}] verified {first}x{count}, records={md['record_count']}, sha256 ok")
PY
done

rsync -a "$STAGING/" "$JOHN0:$JOHN0_ROOT/cascadiav3/fixtures/"
echo "[fleet] all shards staged and pushed to $JOHN0:$JOHN0_ROOT/cascadiav3/fixtures/"
echo "[fleet] trainer usage: --train <john0_shard.npz>,$(for h in $HOSTS; do printf 'cascadiav3/fixtures/cbddb_%s_shard_%s_tensor.npz,' "$CYCLE_TAG" "$h"; done | sed 's/,$//')"
