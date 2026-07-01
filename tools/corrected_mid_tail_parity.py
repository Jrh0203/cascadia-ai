#!/usr/bin/env python3
"""Run or aggregate the frozen F5 corrected-mid-tail parity campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.corrected_mid_tail_parity import (  # noqa: E402
    DEFAULT_CORPUS_ROOT,
    DEFAULT_CORRECTED_CHECKPOINT,
    DEFAULT_HISTORICAL_CHECKPOINT,
    ParityCampaignError,
    aggregate_reports,
    run_shard,
)


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard", help="evaluate one frozen corpus shard")
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    shard.add_argument(
        "--historical-checkpoint",
        type=Path,
        default=DEFAULT_HISTORICAL_CHECKPOINT,
    )
    shard.add_argument(
        "--corrected-checkpoint",
        type=Path,
        default=DEFAULT_CORRECTED_CHECKPOINT,
    )
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--batch-rows", type=int, default=512)
    shard.add_argument("--row-limit", type=int)
    shard.add_argument("--expected-implementation-blake3")

    aggregate = subparsers.add_parser(
        "aggregate",
        help="require and aggregate all ten complete shard reports",
    )
    aggregate.add_argument("--report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "shard":
            report = run_shard(
                corpus_root=_repository_path(args.corpus_root),
                shard_index=args.shard_index,
                historical_checkpoint=_repository_path(args.historical_checkpoint),
                corrected_checkpoint=_repository_path(args.corrected_checkpoint),
                output=_repository_path(args.output),
                batch_rows=args.batch_rows,
                row_limit=args.row_limit,
                expected_implementation_blake3=args.expected_implementation_blake3,
            )
        else:
            report = aggregate_reports(
                [_repository_path(path) for path in args.report],
                output=_repository_path(args.output),
            )
    except (OSError, ParityCampaignError) as error:
        print(f"corrected mid-tail parity error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
