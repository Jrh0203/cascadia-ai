#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LOCAL_ROOT"

OUT="${OUT:-cascadiav3/fixtures/real_roots.jsonl}"
MANIFEST="${MANIFEST:-cascadiav3/fixtures/real_roots_manifest.json}"
FIRST_SEED="${FIRST_SEED:-2026062900}"
SEED_COUNT="${SEED_COUNT:-2}"
PLIES_PER_SEED="${PLIES_PER_SEED:-2}"
MAX_ACTIONS="${MAX_ACTIONS:-8}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-1}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-1}"

cargo run --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml -- \
  --out "$OUT" \
  --manifest "$MANIFEST" \
  --first-seed "$FIRST_SEED" \
  --seed-count "$SEED_COUNT" \
  --plies-per-seed "$PLIES_PER_SEED" \
  --max-actions "$MAX_ACTIONS" \
  --rollouts-per-action "$ROLLOUTS_PER_ACTION" \
  --rollout-top-k "$ROLLOUT_TOP_K"

OUT="$OUT" MANIFEST="$MANIFEST" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 - <<'PY'
import json
import os
from pathlib import Path

from cascadiav3.replay import read_replay_jsonl
from cascadiav3.schema import validate_replay_manifest

jsonl = Path(os.environ["OUT"])
manifest_path = Path(os.environ["MANIFEST"])
records = read_replay_jsonl(jsonl)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
validate_replay_manifest(manifest)
if manifest["record_count"] != len(records):
    raise SystemExit(
        f"manifest count {manifest['record_count']} does not match JSONL count {len(records)}"
    )
print(
    json.dumps(
        {
            "status": "pass",
            "records": len(records),
            "action_counts": [len(record["legal_actions"]) for record in records],
            "manifest": str(manifest_path),
            "jsonl": str(jsonl),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
