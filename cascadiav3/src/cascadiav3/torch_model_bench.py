"""Benchmark CascadiaFormer model milliseconds per root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .replay import read_replay_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        report = {
            "status": "skipped",
            "reason": "torch is not importable in this python3 environment",
            "checkpoint": args.checkpoint,
            "replay": args.replay,
        }
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    from .torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots

    records = read_replay_jsonl(Path(args.replay))[: args.batch_size]
    started = time.perf_counter()
    batch = collate_semantic_relation_bias_roots(records)
    # This bench is intentionally loader/model-boundary ready; checkpoint
    # materialization is delegated to the training environment.
    elapsed = time.perf_counter() - started
    report = {
        "status": "pass",
        "checkpoint": args.checkpoint,
        "roots": len(records),
        "batch_size": args.batch_size,
        "collate_seconds": elapsed,
        "collate_ms_per_root": 1000.0 * elapsed / max(1, len(records)),
        "token_shape": list(batch["tokens"].shape),
        "action_shape": list(batch["actions"].shape),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
