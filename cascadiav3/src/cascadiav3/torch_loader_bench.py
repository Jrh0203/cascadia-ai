"""Benchmark expert tensor/JSONL loader throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .replay import read_replay_jsonl


def _paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = _paths(args.train)
    started = time.perf_counter()
    records = []
    for path in paths:
        if path.suffix == ".jsonl":
            records.extend(read_replay_jsonl(path))
        else:
            report = {
                "status": "skipped",
                "reason": "expert tensor shard reader requires numpy/torch runtime; JSONL path is available now",
                "train": [str(path) for path in paths],
            }
            out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
    loaded_seconds = time.perf_counter() - started
    examples = min(len(records), args.batch_size * args.steps)
    report = {
        "status": "pass",
        "format": "jsonl",
        "paths": [str(path) for path in paths],
        "records_loaded": len(records),
        "examples_considered": examples,
        "load_seconds": loaded_seconds,
        "records_per_second": len(records) / max(loaded_seconds, 1.0e-9),
        "batch_size": args.batch_size,
        "steps": args.steps,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
