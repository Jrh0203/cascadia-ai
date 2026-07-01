#!/usr/bin/env python3
"""Bind measured V3 replay density, collection rate, and checkpoint footprint."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def measure(
    direct: dict[str, Any], corpus: dict[str, Any], checkpoint_directory: Path
) -> dict[str, object]:
    compact = direct.get("compact_shard")
    if not isinstance(compact, dict):
        raise ValueError("direct smoke has no compact replay measurement")
    if (
        direct.get("scientific_eligible") is not False
        or corpus.get("scientific_eligible") is not False
        or int(compact.get("games", 0)) <= 0
        or int(corpus.get("games", 0)) <= 0
    ):
        raise ValueError("capacity inputs must be non-scientific, positive smoke data")
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint_directory.rglob("*") if path.is_file()
    )
    return {
        "schema_id": "cascadia-v3-part1-capacity-measurement-v1",
        "scientific_eligible": False,
        "bytes": int(compact["bytes"]),
        "games": int(compact["games"]),
        "bytes_per_game": math.ceil(int(compact["bytes"]) / int(compact["games"])),
        "collection_seconds_per_game": float(corpus["elapsed_seconds"])
        / int(corpus["games"]),
        "checkpoint_bytes": checkpoint_bytes,
        "sources": {
            "replay_density": direct["schema_id"],
            "collection_rate": corpus["schema_id"],
            "checkpoint_directory": str(checkpoint_directory),
        },
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-smoke", type=Path, required=True)
    parser.add_argument("--engineering-corpus", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(
        _read(args.direct_smoke),
        _read(args.engineering_corpus),
        args.checkpoint_directory,
    )
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
