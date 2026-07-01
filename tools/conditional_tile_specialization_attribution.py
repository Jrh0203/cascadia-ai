#!/usr/bin/env python3
"""Run ADR 0122 tile specialization block attribution."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGE_WIDTHS,
    HierarchicalFactorCache,
    build_stage_model,
    score_stage_shard,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

EXPERIMENT_ID = "conditional-tile-specialization-attribution-v1"
STAGE = "tile"
BLOCKS = {
    "tile_factor": (0, 8),
    "local_geometry": (8, 188),
    "descendant_summary": (188, 249),
}
SOURCE_WEIGHTS_BLAKE3 = (
    "5c13fe87d7b4ac0a8ff9f647f57c69b8d9ab583b3ce2e85e41ee0f3d97e8f514"
)
EXTENDED_WEIGHTS_BLAKE3 = (
    "7acd245b20bf5a35bb3bcab848f3b4b3014d763058fa803b0b4ae3b17c80205d"
)


def _host() -> str:
    return socket.gethostname().split(".")[0]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def permute_block(
    arrays: dict[str, np.ndarray],
    *,
    left_column: int,
    right_column: int,
    query_base: int,
) -> dict[str, np.ndarray]:
    """Cyclically permute one item block within every query."""
    changed = dict(arrays)
    items = arrays["tile_item_features"].copy()
    offsets = arrays["tile_query_offsets"]
    for local_query, (left, right) in enumerate(pairwise(offsets)):
        left = int(left)
        right = int(right)
        width = right - left
        if width <= 1:
            continue
        global_query = query_base + local_query
        shift = 1 + global_query % (width - 1)
        items[left:right, left_column:right_column] = np.roll(
            items[left:right, left_column:right_column],
            shift=shift,
            axis=0,
        )
    changed["tile_item_features"] = items
    return changed


class _Metrics:
    def __init__(self) -> None:
        self.queries = 0
        self.items = 0
        self.targets = 0
        self.hits = 0
        self.exact = 0
        self.finite = True

    def add(
        self,
        scores: np.ndarray,
        offsets: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.finite &= bool(np.all(np.isfinite(scores)))
        self.items += len(scores)
        for left, right in pairwise(offsets):
            left = int(left)
            right = int(right)
            width = min(STAGE_WIDTHS[STAGE], right - left)
            selected = sorted(
                range(left, right),
                key=lambda index: (-float(scores[index]), index),
            )[:width]
            quota = int(np.sum(targets[left:right]))
            hits = int(np.sum(targets[selected]))
            self.targets += quota
            self.hits += hits
            self.exact += int(hits == quota)
            self.queries += 1

    def report(self, cache: HierarchicalFactorCache) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "items": self.items,
            "target_factors": self.targets,
            "target_hits": self.hits,
            "target_factor_recall": self.hits / max(self.targets, 1),
            "exact_query_fraction": self.exact / max(self.queries, 1),
            "all_scores_finite": self.finite,
            "all_queries_scored_once": (
                self.queries == int(cache.manifest["queries"][STAGE])
            ),
            "all_items_scored_once": (
                self.items == int(cache.manifest["items"][STAGE])
            ),
        }


def evaluate_block(
    model: Any,
    cache: HierarchicalFactorCache,
    *,
    block: str,
) -> dict[str, Any]:
    """Evaluate original and block-permuted tile retrieval."""
    left_column, right_column = BLOCKS[block]
    baseline = _Metrics()
    permuted = _Metrics()
    query_base = 0
    for arrays in cache.iter_shards():
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        baseline.add(
            score_stage_shard(model, arrays, STAGE),
            offsets,
            targets,
        )
        changed = permute_block(
            arrays,
            left_column=left_column,
            right_column=right_column,
            query_base=query_base,
        )
        permuted.add(
            score_stage_shard(model, changed, STAGE),
            offsets,
            targets,
        )
        query_base += len(offsets) - 1
    baseline_report = baseline.report(cache)
    permuted_report = permuted.report(cache)
    return {
        "baseline": baseline_report,
        "permuted": permuted_report,
        "recall_drop": (
            float(baseline_report["target_factor_recall"])
            - float(permuted_report["target_factor_recall"])
        ),
        "exact_drop": (
            float(baseline_report["exact_query_fraction"])
            - float(permuted_report["exact_query_fraction"])
        ),
    }


def run_arm(
    *,
    block: str,
    train_cache_root: Path,
    validation_cache_root: Path,
    source_weights: Path,
    extended_weights: Path,
) -> dict[str, Any]:
    """Run one frozen feature-block permutation arm."""
    if block not in BLOCKS:
        raise ValueError("unknown specialization block")
    started = time.perf_counter()
    configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    source_hash = checksum(source_weights)
    extended_hash = checksum(extended_weights)
    checkpoints = {}
    for name, weights in (
        ("source_20_epoch", source_weights),
        ("extended_200_epoch", extended_weights),
    ):
        model = build_stage_model(STAGE)
        model.load_weights(str(weights))
        mx.eval(model.parameters())
        checkpoints[name] = {
            "train": evaluate_block(model, train_cache, block=block),
            "validation": evaluate_block(
                model,
                validation_cache,
                block=block,
            ),
        }
        del model
        mx.clear_cache()
    source = checkpoints["source_20_epoch"]
    extended = checkpoints["extended_200_epoch"]
    source_train_drop = float(source["train"]["recall_drop"])
    source_validation_drop = float(source["validation"]["recall_drop"])
    extended_train_drop = float(extended["train"]["recall_drop"])
    extended_validation_drop = float(extended["validation"]["recall_drop"])
    contribution = (extended_train_drop - source_train_drop) - (
        extended_validation_drop - source_validation_drop
    )
    pipeline = {
        "source_weights_identity": source_hash == SOURCE_WEIGHTS_BLAKE3,
        "extended_weights_identity": extended_hash == EXTENDED_WEIGHTS_BLAKE3,
        "cache_payloads_distinct": (
            train_cache.manifest["payload_blake3"]
            != validation_cache.manifest["payload_blake3"]
        ),
        "all_scores_finite": all(
            bool(report[split][condition]["all_scores_finite"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "permuted")
        ),
        "all_queries_scored_once": all(
            bool(report[split][condition]["all_queries_scored_once"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "permuted")
        ),
        "all_items_scored_once": all(
            bool(report[split][condition]["all_items_scored_once"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "permuted")
        ),
    }
    scientific = {
        "block": block,
        "columns": list(BLOCKS[block]),
        "source_weights_blake3": source_hash,
        "extended_weights_blake3": extended_hash,
        "checkpoints": checkpoints,
        "specialization_contribution": contribution,
        "components": {
            "source_train_recall_drop": source_train_drop,
            "source_validation_recall_drop": source_validation_drop,
            "extended_train_recall_drop": extended_train_drop,
            "extended_validation_recall_drop": extended_validation_drop,
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": (
            validation_cache.manifest["payload_blake3"]
        ),
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": block,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


def combine(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a targeted or distributed structural-regularization result."""
    if {arm.get("arm") for arm in arms} != set(BLOCKS):
        raise ValueError("specialization attribution arm coverage is incomplete")
    by_block = {str(arm["arm"]): arm for arm in arms}
    pipeline = {
        block: (
            arm.get("experiment_id") == EXPERIMENT_ID
            and bool(arm["scientific"]["pipeline_passed"])
        )
        for block, arm in by_block.items()
    }
    ranking = sorted(
        (
            (
                float(arm["scientific"]["specialization_contribution"]),
                block,
            )
            for block, arm in by_block.items()
        ),
        reverse=True,
    )
    largest, selected = ranking[0]
    second = ranking[1][0]
    identified = all(pipeline.values()) and largest >= 0.05 and largest - second >= 0.02
    classification = (
        "specialization_block_identified"
        if identified
        else (
            "specialization_distributed_across_blocks"
            if all(pipeline.values())
            else "specialization_attribution_pipeline_invalid"
        )
    )
    scientific = {
        "classification": classification,
        "selected_block": selected if identified else None,
        "largest_contribution": largest,
        "second_largest_contribution": second,
        "contribution_gap": largest - second,
        "contributions": {
            block: float(arm["scientific"]["specialization_contribution"])
            for block, arm in by_block.items()
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "arms": by_block,
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
    }


def render_markdown(report: dict[str, Any]) -> str:
    scientific = report["scientific"]
    rows = "\n".join(
        f"| {block} | {value:+.4f} |"
        for block, value in sorted(scientific["contributions"].items())
    )
    return f"""# Conditional Tile Specialization Attribution V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

| Feature block | Specialization contribution |
|---|---:|
{rows}

Selected block: `{scientific["selected_block"]}`.
Largest contribution: `{scientific["largest_contribution"]:+.4f}`.
Runner-up gap: `{scientific["contribution_gap"]:+.4f}`.

Combined scientific BLAKE3:
`{report["scientific_blake3"]}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--block", choices=sorted(BLOCKS), required=True)
    arm.add_argument("--train-cache", type=Path, required=True)
    arm.add_argument("--validation-cache", type=Path, required=True)
    arm.add_argument("--source-weights", type=Path, required=True)
    arm.add_argument("--extended-weights", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--arm", type=Path, action="append", required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "arm":
        report = run_arm(
            block=args.block,
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            source_weights=args.source_weights,
            extended_weights=args.extended_weights,
        )
    else:
        report = combine(
            [json.loads(path.read_text()) for path in args.arm]
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report))
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "arm": report.get("arm", "combined"),
                "scientific_blake3": report["scientific_blake3"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
