#!/usr/bin/env bash
set -euo pipefail

# Status and fail-closed collection for fleet_wildlife_exact_launch.sh.
# Usage: fleet_wildlife_exact_collect.sh <tag> status|collect

FLEET_TAG="${1:?usage: fleet_wildlife_exact_collect.sh <tag> status|collect}"
ACTION="${2:?usage: fleet_wildlife_exact_collect.sh <tag> status|collect}"
case "$ACTION" in
  status|collect) ;;
  *) echo "action must be status or collect" >&2; exit 64 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLEET_DIR="$ROOT/cascadiav3/fleet"
LEDGER="$FLEET_DIR/wildlife_${FLEET_TAG}_fleet.json"
INPUT_DIR="$FLEET_DIR/inputs_${FLEET_TAG}"
STAGING="$FLEET_DIR/staging_${FLEET_TAG}"
test -s "$LEDGER"
test -s "$INPUT_DIR/candidates.json"
test -s "$INPUT_DIR/counts.json"

local_host="$(
  .venv/bin/python - "$LEDGER" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload.get("configuration", {}).get("local_host", "john1"))
PY
)"

fleet_exec() {
  local host="$1"
  local command="$2"
  if [ "$host" = "$local_host" ]; then
    bash -c "$command"
  else
    ssh "$host" "$command"
  fi
}

hosts="$(
  python3 - "$LEDGER" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("schema") != "wildlife-exact-fleet-v1":
    raise SystemExit("unsupported fleet ledger")
print(" ".join(row["host"] for row in payload["shards"]))
PY
)"

all_terminal=1
for host in $hosts; do
  fleet_exec "$host" "cd ~/cascadia && python3 - '$FLEET_TAG' '$host' <<'PY'
import json, os, signal, sys, time
from pathlib import Path

tag, host = sys.argv[1:]
logs = Path('cascadiav3/logs')
base = logs / f'wildlife_{tag}_shard_{host}'
pid_path = base.with_suffix('.pid')
heartbeat = base.with_suffix('.heartbeat')
exit_path = base.with_suffix('.exit')
output = Path('cascadiav3/fleet_outputs') / tag / f'shard_{host}.json'
pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
alive = False
if pid is not None:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
age = int(time.time() - heartbeat.stat().st_mtime) if heartbeat.exists() else None
exit_code = exit_path.read_text().strip() if exit_path.exists() else None
summary = None
if output.exists():
    try:
        payload = json.loads(output.read_text())
        summary = {
            'completed': payload.get('completed_count'),
            'rows': len(payload.get('results', [])),
            'proof_complete': payload.get('proof_complete'),
        }
    except Exception as error:
        summary = {'invalid_json': str(error)}
print(json.dumps({
    'host': host, 'worker_pid': pid, 'alive': alive,
    'heartbeat_age_seconds': age, 'exit_code': exit_code,
    'output': summary,
}, sort_keys=True))
PY"
  terminal="$(
    fleet_exec "$host" \
      "test -s ~/cascadia/cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.exit && echo yes || echo no"
  )"
  [ "$terminal" = yes ] || all_terminal=0
done

[ "$ACTION" = status ] && exit 0
if [ "$all_terminal" -ne 1 ]; then
  echo "[fleet] not all exact shards are terminal; refusing partial collection" >&2
  exit 65
fi

mkdir -p "$STAGING"
for host in $hosts; do
  remote_base="wildlife_${FLEET_TAG}_shard_${host}"
  if [ "$host" = "$local_host" ]; then
    rsync -a "$ROOT/cascadiav3/fleet_outputs/${FLEET_TAG}/shard_${host}.json" \
      "$STAGING/"
    rsync -a "$ROOT/cascadiav3/logs/${remote_base}.log" \
      "$ROOT/cascadiav3/logs/${remote_base}.heartbeat" \
      "$ROOT/cascadiav3/logs/${remote_base}.exit" \
      "$STAGING/"
  else
    rsync -a "$host:~/cascadia/cascadiav3/fleet_outputs/${FLEET_TAG}/shard_${host}.json" \
      "$STAGING/"
    rsync -a "$host:~/cascadia/cascadiav3/logs/${remote_base}.log" \
      "$host:~/cascadia/cascadiav3/logs/${remote_base}.heartbeat" \
      "$host:~/cascadia/cascadiav3/logs/${remote_base}.exit" \
      "$STAGING/"
  fi
done

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - \
  "$LEDGER" "$INPUT_DIR" "$STAGING" <<'PY'
import hashlib, json, os, sys, tempfile
from collections import Counter
from pathlib import Path

ledger_path, input_text, staging_text = map(Path, sys.argv[1:])
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
inputs = ledger["inputs"]
staging = staging_text

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

candidates_path = input_text / "candidates.json"
counts_path = input_text / "counts.json"
assert sha(candidates_path) == inputs["candidates_sha256"]
assert sha(counts_path) == inputs["counts_sha256"]

ruleset = ledger["ruleset"]
if ruleset == "aaaaa":
    from tools import aaaaa_wildlife_catalog as catalog
    from tools.aaaaa_wildlife_exact import count_vectors
elif ruleset == "cbddb":
    from tools import cbddb_wildlife_catalog as catalog
    from tools.cbddb_wildlife_exact import count_vectors
else:
    raise SystemExit(f"unsupported ruleset {ruleset}")
from tools.wildlife_catalog_sharding import load_taskset

catalog.load_candidates(candidates_path)
canonical = [counts for counts, _ in count_vectors()]
requested, _ = load_taskset(
    counts_path,
    scoring_cards=ledger["scoring_cards"],
    canonical_counts=canonical,
)

seen_requested = Counter()
verified_files = []
requested_results = []
for shard in ledger["shards"]:
    host = shard["host"]
    path = staging / f"shard_{host}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != catalog.SCHEMA:
        raise SystemExit(f"{host}: catalog schema mismatch")
    if payload.get("candidates_sha256") != inputs["candidates_sha256"]:
        raise SystemExit(f"{host}: candidate hash mismatch")
    if payload.get("catalog_source_sha256") != ledger["sources"]["catalog_sha256"]:
        raise SystemExit(f"{host}: catalog source hash mismatch")
    if payload.get("exact_model_source_sha256") != ledger["sources"]["exact_sha256"]:
        raise SystemExit(f"{host}: exact source hash mismatch")
    config = payload.get("configuration", {})
    taskset = config.get("taskset") or {}
    if taskset.get("sha256") != inputs["counts_sha256"]:
        raise SystemExit(f"{host}: taskset hash mismatch")
    if (config.get("shard_index"), config.get("shard_count")) != (
        shard["shard_index"],
        shard["shard_count"],
    ):
        raise SystemExit(f"{host}: shard configuration mismatch")
    exit_code = int((staging / f"wildlife_{ledger['tag']}_shard_{host}.exit").read_text())
    if exit_code not in (0, 2):
        raise SystemExit(f"{host}: unexpected worker exit {exit_code}")
    for result in payload.get("results", []):
        counts = tuple(int(value) for value in result["counts"])
        tokens, breakdown = catalog.validate_witness(counts, result["tokens"])
        if sum(breakdown) != int(result["optimum"]):
            raise SystemExit(f"{host}: witness mismatch for {counts}")
        if counts in requested:
            seen_requested[counts] += 1
            requested_results.append(result)
    verified_files.append({"host": host, "path": str(path), "sha256": sha(path)})

missing = requested - set(seen_requested)
duplicates = {counts: count for counts, count in seen_requested.items() if count != 1}
if missing or duplicates:
    raise SystemExit(
        f"fleet task coverage mismatch: missing={sorted(missing)} duplicates={duplicates}"
    )

proof_methods = Counter(
    row["proof_method"] for row in requested_results if row.get("proof_complete")
)
summary = {
    "schema": "wildlife-exact-fleet-collection-v1",
    "tag": ledger["tag"],
    "ruleset": ruleset,
    "source_revision": ledger["source_revision"],
    "requested_count": len(requested),
    "returned_count": len(requested_results),
    "exact_count": sum(bool(row.get("proof_complete")) for row in requested_results),
    "incomplete_count": sum(not bool(row.get("proof_complete")) for row in requested_results),
    "proof_methods": dict(sorted(proof_methods.items())),
    "highest_returned_score": max(int(row["optimum"]) for row in requested_results),
    "files": verified_files,
}
output = staging / "collection_manifest.json"
with tempfile.NamedTemporaryFile("w", dir=staging, delete=False) as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
print(json.dumps(summary, sort_keys=True))
PY

echo "[fleet] exact shards verified in $STAGING"
