#!/usr/bin/env python3
"""Build and inspect compact R2-MAP replay locally on the john2 storage host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_contracts import (  # noqa: E402
    CAMPAIGN_ROOT,
    canonical_campaign_path,
    require_local_storage_authority,
)
from cascadia_mlx.r2_map_dataset import (  # noqa: E402
    R2MapCompactDatasetAdapter,
    build_compact_index,
    compact_storage_projection,
    validate_compact_index,
)


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def command_build(arguments: argparse.Namespace) -> None:
    _print(
        build_compact_index(
            arguments.shard,
            exporter=arguments.exporter,
            output=arguments.output,
            scratch=arguments.scratch,
            maximum_window_bytes=arguments.maximum_window_bytes,
        )
    )


def command_validate(arguments: argparse.Namespace) -> None:
    value = validate_compact_index(arguments.index, shard_root=arguments.shard_root)
    _print(
        {
            "valid": True,
            "index_blake3": value["index_blake3"],
            "dataset_blake3": value["dataset_manifest"]["dataset_blake3"],
            "games": len(value["games"]),
            "examples": value["dataset_manifest"]["example_count"],
        }
    )


def command_project(arguments: argparse.Namespace) -> None:
    projection = compact_storage_projection(
        arguments.index,
        target_games=arguments.target_games,
        maximum_window_bytes=arguments.maximum_window_bytes,
        maximum_prefetch_windows=arguments.maximum_prefetch_windows,
        expanded_bytes_per_game=arguments.expanded_bytes_per_game,
    )
    _print(projection.to_dict())
    if not projection.compact_fits_run_budget or projection.expanded_fits_run_budget:
        raise SystemExit(2)


def command_smoke(arguments: argparse.Namespace) -> None:
    options = dict(
        index=arguments.index,
        shard_root=arguments.shard_root,
        exporter=arguments.exporter,
        window_root=arguments.window_root,
        group_batch_size=arguments.group_batch_size,
        maximum_window_bytes=arguments.maximum_window_bytes,
        maximum_prefetch_windows=arguments.maximum_prefetch_windows,
        fixed_panel_games=arguments.fixed_panel_games,
        require_ssd=arguments.require_ssd,
    )
    with R2MapCompactDatasetAdapter(**options) as adapter:
        cursor, sampler = adapter.initial_state(arguments.seed)
        first = adapter.training_batch(cursor, sampler)
        repeated = adapter.training_batch(cursor, sampler)
        expected_next = adapter.training_batch(
            first.next_cursor, first.next_sampler_state
        ).batch.batch_identity
        validation = next(iter(adapter.validation_batches()))
        panel = adapter.fixed_prediction_batch(arguments.panel_id)
        result = {
            "schema_version": 1,
            "schema_id": "r2-map-compact-physical-smoke-v1",
            "dataset_blake3": adapter.dataset_blake3,
            "first_batch_identity": first.batch.batch_identity,
            "deterministic_repeat": first.batch.batch_identity == repeated.batch.batch_identity,
            "next_cursor": first.next_cursor,
            "expected_next_batch_identity": expected_next,
            "validation_batch_identity": validation.batch_identity,
            "panel_groups": int(panel.candidate_mask.shape[0]),
            "panel_candidates": int(panel.candidate_mask.shape[1]),
        }
    with R2MapCompactDatasetAdapter(**options) as reopened:
        actual_next = reopened.training_batch(
            first.next_cursor, first.next_sampler_state
        ).batch.batch_identity
    result.update(
        {
            "reopen_next_batch_identity": actual_next,
            "exact_resume_next_batch": actual_next == expected_next,
            "remaining_window_files_after_close": len(tuple(arguments.window_root.glob("*.r2map"))),
        }
    )
    _print(result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-index")
    build.add_argument("--shard", type=Path, action="append", required=True)
    build.add_argument("--exporter", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--scratch", type=Path, required=True)
    build.add_argument("--maximum-window-bytes", type=int, default=1 << 30)
    build.set_defaults(function=command_build)

    validate = commands.add_parser("validate")
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--shard-root", type=Path, required=True)
    validate.set_defaults(function=command_validate)

    project = commands.add_parser("project")
    project.add_argument("--index", type=Path, required=True)
    project.add_argument("--target-games", type=int, default=100_000)
    project.add_argument("--maximum-window-bytes", type=int, default=1 << 30)
    project.add_argument("--maximum-prefetch-windows", type=int, choices=(0, 1), default=1)
    project.add_argument("--expanded-bytes-per-game", type=int, default=2_000_000)
    project.set_defaults(function=command_project)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--index", type=Path, required=True)
    smoke.add_argument("--shard-root", type=Path, required=True)
    smoke.add_argument("--exporter", type=Path, required=True)
    smoke.add_argument("--window-root", type=Path, required=True)
    smoke.add_argument("--group-batch-size", type=int, default=2)
    smoke.add_argument("--maximum-window-bytes", type=int, default=1 << 30)
    smoke.add_argument("--maximum-prefetch-windows", type=int, choices=(0, 1), default=1)
    smoke.add_argument("--fixed-panel-games", type=int, default=1)
    smoke.add_argument("--seed", type=int, default=20260618)
    smoke.add_argument("--panel-id", default="r2-map-fixed-panel-v1")
    smoke.add_argument("--require-ssd", action="store_true")
    smoke.set_defaults(function=command_smoke)
    return result


def main() -> int:
    arguments = parser().parse_args()
    require_local_storage_authority()
    paths: list[Path] = []
    for name in (
        "exporter",
        "output",
        "scratch",
        "index",
        "shard_root",
        "window_root",
    ):
        value = getattr(arguments, name, None)
        if value is not None:
            paths.append(value)
    paths.extend(getattr(arguments, "shard", []) or [])
    for path in paths:
        canonical_campaign_path(path, root=CAMPAIGN_ROOT, label="compact dataset path")
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
