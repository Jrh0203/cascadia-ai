#!/usr/bin/env bash
set -euo pipefail

# Launch disjoint exact wildlife-catalog shards on idle Mac minis.
#
# Required:
#   RULESET          aaaaa or cbddb
#   FLEET_TAG        unique durable tag
#   CANDIDATES       local full 826-row candidate JSON
#   COUNTS_FILE      local wildlife-catalog-taskset-v1 JSON
#   SOURCE_REVISION  committed revision being deployed
# Optional:
#   IMPORT_LEDGER    local catalog ledger whose complete proofs are imported
#   HOSTS            default "john2 john3 john4"
#   JOBS             default 2
#   SOLVER_WORKERS   default 4
#   RELAXATION_TIME_LIMIT / CONNECTED_TIME_LIMIT / BASE_SEED

RULESET="${RULESET:?set RULESET (aaaaa or cbddb)}"
FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
CANDIDATES="${CANDIDATES:?set CANDIDATES}"
COUNTS_FILE="${COUNTS_FILE:?set COUNTS_FILE}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
IMPORT_LEDGER="${IMPORT_LEDGER:-}"
HOSTS="${HOSTS:-john2 john3 john4}"
JOBS="${JOBS:-2}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
RELAXATION_TIME_LIMIT="${RELAXATION_TIME_LIMIT:-60}"
CONNECTED_TIME_LIMIT="${CONNECTED_TIME_LIMIT:-120}"
BASE_SEED="${BASE_SEED:-20260725}"
ORTOOLS_VERSION="${ORTOOLS_VERSION:-9.15.6755}"
WILDLIFE_PYTHON_VERSION="${WILDLIFE_PYTHON_VERSION:-3.12.13}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"

case "$WILDLIFE_VENV" in
  ""|/*|*".."*|*[!A-Za-z0-9._/-]*)
    echo "WILDLIFE_VENV must be a safe relative path under ~/cascadia" >&2
    exit 64
    ;;
esac

case "$RULESET" in
  aaaaa)
    SCORING_CARDS="AAAAA"
    CATALOG_SOURCE="tools/aaaaa_wildlife_catalog.py"
    EXACT_SOURCE="tools/aaaaa_wildlife_exact.py"
    CANDIDATE_SOURCE="crates/cascadia-game/src/bin/aaaaa_wildlife_solver.rs"
    ;;
  cbddb)
    SCORING_CARDS="CBDDB"
    CATALOG_SOURCE="tools/cbddb_wildlife_catalog.py"
    EXACT_SOURCE="tools/cbddb_wildlife_exact.py"
    CANDIDATE_SOURCE="crates/cascadia-game/src/bin/cbddb_wildlife_solver.rs"
    ;;
  *)
    echo "unsupported RULESET=$RULESET" >&2
    exit 64
    ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
git cat-file -e "${SOURCE_REVISION}^{commit}"
[ "$(git rev-parse HEAD)" = "$SOURCE_REVISION" ] || {
  echo "SOURCE_REVISION must equal current HEAD" >&2
  exit 65
}
test -s "$CANDIDATES"
test -s "$COUNTS_FILE"
if [ -n "$IMPORT_LEDGER" ]; then
  test -s "$IMPORT_LEDGER"
fi

read -r -a host_array <<< "$HOSTS"
shard_count="${#host_array[@]}"
[ "$shard_count" -gt 0 ] || {
  echo "HOSTS is empty" >&2
  exit 66
}

FLEET_DIR="$ROOT/cascadiav3/fleet"
INPUT_DIR="$FLEET_DIR/inputs_${FLEET_TAG}"
LEDGER="$FLEET_DIR/wildlife_${FLEET_TAG}_fleet.json"
if [ -e "$LEDGER" ] || [ -e "$INPUT_DIR" ]; then
  echo "fleet tag $FLEET_TAG already has durable local state; refusing double-launch" >&2
  exit 67
fi

mkdir -p "$INPUT_DIR"
rsync -a "$CANDIDATES" "$INPUT_DIR/candidates.json"
rsync -a "$COUNTS_FILE" "$INPUT_DIR/counts.json"
if [ -n "$IMPORT_LEDGER" ]; then
  rsync -a "$IMPORT_LEDGER" "$INPUT_DIR/import_ledger.json"
fi

catalog_sha="$(shasum -a 256 "$CATALOG_SOURCE" | awk '{print $1}')"
exact_sha="$(shasum -a 256 "$EXACT_SOURCE" | awk '{print $1}')"
candidate_sha="$(shasum -a 256 "$INPUT_DIR/candidates.json" | awk '{print $1}')"
counts_sha="$(shasum -a 256 "$INPUT_DIR/counts.json" | awk '{print $1}')"
import_sha=""
if [ -s "$INPUT_DIR/import_ledger.json" ]; then
  import_sha="$(shasum -a 256 "$INPUT_DIR/import_ledger.json" | awk '{print $1}')"
fi

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - \
  "$RULESET" "$INPUT_DIR/candidates.json" "$INPUT_DIR/counts.json" \
  "${INPUT_DIR}/import_ledger.json" <<'PY'
import json, sys
from pathlib import Path

ruleset, candidates_path, counts_path, import_path = sys.argv[1:]
cards = ruleset.upper()
if ruleset == "aaaaa":
    from tools.aaaaa_wildlife_catalog import load_candidates
elif ruleset == "cbddb":
    from tools.cbddb_wildlife_catalog import load_candidates
else:
    raise SystemExit(f"unsupported ruleset {ruleset}")
from tools.wildlife_catalog_sharding import load_taskset
if ruleset == "aaaaa":
    from tools.aaaaa_wildlife_exact import count_vectors
else:
    from tools.cbddb_wildlife_exact import count_vectors

load_candidates(Path(candidates_path))
canonical = [counts for counts, _ in count_vectors()]
selected, _ = load_taskset(
    Path(counts_path), scoring_cards=cards, canonical_counts=canonical
)
if not selected:
    raise SystemExit("taskset is empty")
path = Path(import_path)
if path.exists():
    ledger = json.loads(path.read_text(encoding="utf-8"))
    completed = {
        tuple(int(value) for value in row["counts"])
        for row in ledger.get("results", [])
        if row.get("proof_complete")
    }
    overlap = selected & completed
    if overlap:
        raise SystemExit(f"taskset overlaps imported complete proofs: {sorted(overlap)}")
print(f"validated candidates=826 taskset={len(selected)} imported={path.exists()}")
PY

# Preflight every host before writing the launch ledger. No process is stopped
# or replaced; any collision or missing dependency fails the whole launch.
for host in "${host_array[@]}"; do
  ssh "$host" \
    "cd ~/cascadia && \
     test -x '$WILDLIFE_VENV/bin/python' && \
     test \"\$('$WILDLIFE_VENV/bin/python' -c 'import platform; print(platform.python_version())')\" = '$WILDLIFE_PYTHON_VERSION' && \
     test \"\$('$WILDLIFE_VENV/bin/python' -c 'import ortools; print(ortools.__version__)')\" = '$ORTOOLS_VERSION' && \
     test ! -e cascadiav3/fleet_outputs/$FLEET_TAG && \
     test ! -e cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.pid && \
     ! pgrep -f 'fleet_wildlife_exact_worker.sh.*$FLEET_TAG' >/dev/null"
done

.venv/bin/python - "$LEDGER" "$RULESET" "$SOURCE_REVISION" "$candidate_sha" "$counts_sha" \
  "$import_sha" "$catalog_sha" "$exact_sha" "$ORTOOLS_VERSION" \
  "$WILDLIFE_PYTHON_VERSION" "$JOBS" \
  "$SOLVER_WORKERS" "$RELAXATION_TIME_LIMIT" "$CONNECTED_TIME_LIMIT" \
  "$BASE_SEED" "$WILDLIFE_VENV" "$HOSTS" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    ledger_path, ruleset, revision, candidate_sha, counts_sha, import_sha,
    catalog_sha, exact_sha, ortools, python_version, jobs, workers, relaxation, connected,
    base_seed, wildlife_venv, hosts_text,
) = sys.argv[1:]
hosts = hosts_text.split()
payload = {
    "schema": "wildlife-exact-fleet-v1",
    "state": "planned",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "tag": Path(ledger_path).stem.removeprefix("wildlife_").removesuffix("_fleet"),
    "ruleset": ruleset,
    "scoring_cards": ruleset.upper(),
    "source_revision": revision,
    "inputs": {
        "candidates_sha256": candidate_sha,
        "counts_sha256": counts_sha,
        "import_ledger_sha256": import_sha or None,
    },
    "sources": {
        "catalog_sha256": catalog_sha,
        "exact_sha256": exact_sha,
        "ortools_version": ortools,
        "python_version": python_version,
    },
    "configuration": {
        "jobs": int(jobs),
        "solver_workers": int(workers),
        "relaxation_time_limit_seconds": float(relaxation),
        "connected_time_limit_seconds": float(connected),
        "base_seed": int(base_seed),
        "wildlife_venv": wildlife_venv,
    },
    "shards": [
        {"host": host, "shard_index": index, "shard_count": len(hosts)}
        for index, host in enumerate(hosts)
    ],
}
path = Path(ledger_path)
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY

# Deploy and hash-pin every host before starting any worker. This prevents a
# source-layout failure on a later host from leaving an avoidable partial run.
for index in "${!host_array[@]}"; do
  host="${host_array[$index]}"
  remote_input="cascadiav3/fleet_inputs/${FLEET_TAG}"
  candidate_parent="$(dirname "$CANDIDATE_SOURCE")"
  ssh "$host" "cd ~/cascadia && \
    mkdir -p '$remote_input' cascadiav3/fleet_outputs/${FLEET_TAG} \
      '$candidate_parent'"
  rsync -a --exclude __pycache__ "$ROOT/tools/" "$host:~/cascadia/tools/"
  rsync -a "$ROOT/cascadiav3/scripts/fleet_wildlife_exact_worker.sh" \
    "$host:~/cascadia/cascadiav3/scripts/"
  rsync -a "$ROOT/$CANDIDATE_SOURCE" \
    "$host:~/cascadia/$CANDIDATE_SOURCE"
  if [ "$RULESET" = cbddb ]; then
    rsync -a "$ROOT/crates/cascadia-game/src/bin/wildlife_solver_support/" \
      "$host:~/cascadia/crates/cascadia-game/src/bin/wildlife_solver_support/"
  fi
  rsync -a "$INPUT_DIR/" "$host:~/cascadia/$remote_input/"
done

declare -a launched_pids=()
for index in "${!host_array[@]}"; do
  host="${host_array[$index]}"
  log="cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.log"
  pid_file="cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.pid"
  pid="$(
    ssh "$host" "cd ~/cascadia && \
      nohup env RULESET='$RULESET' FLEET_TAG='$FLEET_TAG' SHARD_HOST='$host' \
      SHARD_INDEX='$index' SHARD_COUNT='$shard_count' \
      SOURCE_REVISION='$SOURCE_REVISION' JOBS='$JOBS' \
      SOLVER_WORKERS='$SOLVER_WORKERS' RELAXATION_TIME_LIMIT='$RELAXATION_TIME_LIMIT' \
      CONNECTED_TIME_LIMIT='$CONNECTED_TIME_LIMIT' BASE_SEED='$BASE_SEED' \
      ORTOOLS_VERSION='$ORTOOLS_VERSION' \
      WILDLIFE_PYTHON_VERSION='$WILDLIFE_PYTHON_VERSION' \
      WILDLIFE_VENV='$WILDLIFE_VENV' \
      CATALOG_SOURCE_SHA256='$catalog_sha' \
      EXACT_SOURCE_SHA256='$exact_sha' \
      bash cascadiav3/scripts/fleet_wildlife_exact_worker.sh \
      > '$log' 2>&1 < /dev/null & \
      echo \$! > '$pid_file'; cat '$pid_file'"
  )"
  launched_pids+=("$pid")
  echo "[fleet] $host shard=$index/$shard_count pid=$pid"
done

.venv/bin/python - "$LEDGER" "${launched_pids[*]}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
pids = [int(value) for value in sys.argv[2].split()]
payload = json.loads(path.read_text(encoding="utf-8"))
if len(pids) != len(payload["shards"]):
    raise SystemExit("launched PID count mismatch")
for shard, pid in zip(payload["shards"], pids, strict=True):
    shard["worker_pid"] = pid
payload["state"] = "launched"
payload["launched_utc"] = datetime.now(timezone.utc).isoformat()
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY

echo "[fleet] launched $shard_count exact shards; ledger: $LEDGER"
