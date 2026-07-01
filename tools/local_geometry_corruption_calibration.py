#!/usr/bin/env python3
"""Run ADR 0123 local-geometry corruption calibration."""

from __future__ import annotations

import argparse
import json
import math
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

EXPERIMENT_ID = "local-geometry-corruption-calibration-v1"
STAGE = "tile"
LOCAL_LEFT = 8
LOCAL_RIGHT = 188
RATES = (0.10, 0.25, 0.50)
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


def corrupt_local_geometry(
    arrays: dict[str, np.ndarray],
    *,
    rate: float,
) -> dict[str, np.ndarray]:
    """Corrupt a stable hash-selected item fraction within each query."""
    if rate not in RATES:
        raise ValueError("corruption rate is not frozen")
    changed = dict(arrays)
    items = arrays["tile_item_features"].copy()
    hashes = arrays["tile_item_hash"]
    for left, right in pairwise(arrays["tile_query_offsets"]):
        left = int(left)
        right = int(right)
        width = right - left
        count = min(width, math.ceil(rate * width))
        if count < 2:
            continue
        selected = sorted(
            range(left, right),
            key=lambda index: bytes(hashes[index]),
        )[:count]
        rotated = np.roll(
            items[selected, LOCAL_LEFT:LOCAL_RIGHT],
            shift=1,
            axis=0,
        )
        items[selected, LOCAL_LEFT:LOCAL_RIGHT] = rotated
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


def evaluate(
    model: Any,
    cache: HierarchicalFactorCache,
    *,
    rate: float,
) -> dict[str, Any]:
    baseline = _Metrics()
    corrupted = _Metrics()
    for arrays in cache.iter_shards():
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        baseline.add(
            score_stage_shard(model, arrays, STAGE),
            offsets,
            targets,
        )
        changed = corrupt_local_geometry(arrays, rate=rate)
        corrupted.add(
            score_stage_shard(model, changed, STAGE),
            offsets,
            targets,
        )
    baseline_report = baseline.report(cache)
    corrupted_report = corrupted.report(cache)
    return {
        "baseline": baseline_report,
        "corrupted": corrupted_report,
        "recall_damage": (
            float(baseline_report["target_factor_recall"])
            - float(corrupted_report["target_factor_recall"])
        ),
        "exact_damage": (
            float(baseline_report["exact_query_fraction"])
            - float(corrupted_report["exact_query_fraction"])
        ),
    }


def run_arm(
    *,
    rate: float,
    train_cache_root: Path,
    validation_cache_root: Path,
    source_weights: Path,
    extended_weights: Path,
) -> dict[str, Any]:
    """Run one frozen corruption-rate arm."""
    if rate not in RATES:
        raise ValueError("corruption rate is not frozen")
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
            "train": evaluate(model, train_cache, rate=rate),
            "validation": evaluate(model, validation_cache, rate=rate),
        }
        del model
        mx.clear_cache()
    source = checkpoints["source_20_epoch"]
    extended = checkpoints["extended_200_epoch"]
    baseline_gap = (
        float(extended["train"]["baseline"]["target_factor_recall"])
        - float(extended["validation"]["baseline"]["target_factor_recall"])
    )
    corrupted_gap = (
        float(extended["train"]["corrupted"]["target_factor_recall"])
        - float(extended["validation"]["corrupted"]["target_factor_recall"])
    )
    gap_reduction = baseline_gap - corrupted_gap
    gap_reduction_fraction = gap_reduction / max(baseline_gap, 1e-12)
    source_validation_damage = float(source["validation"]["recall_damage"])
    extended_validation_damage = float(extended["validation"]["recall_damage"])
    feasible = (
        gap_reduction_fraction >= 0.25
        and source_validation_damage <= 0.02
        and extended_validation_damage <= 0.02
    )
    pipeline = {
        "source_weights_identity": source_hash == SOURCE_WEIGHTS_BLAKE3,
        "extended_weights_identity": extended_hash == EXTENDED_WEIGHTS_BLAKE3,
        "all_scores_finite": all(
            bool(report[split][condition]["all_scores_finite"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "corrupted")
        ),
        "all_queries_scored_once": all(
            bool(report[split][condition]["all_queries_scored_once"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "corrupted")
        ),
        "all_items_scored_once": all(
            bool(report[split][condition]["all_items_scored_once"])
            for report in checkpoints.values()
            for split in ("train", "validation")
            for condition in ("baseline", "corrupted")
        ),
    }
    scientific = {
        "rate": rate,
        "checkpoints": checkpoints,
        "baseline_extended_train_validation_gap": baseline_gap,
        "corrupted_extended_train_validation_gap": corrupted_gap,
        "gap_reduction": gap_reduction,
        "gap_reduction_fraction": gap_reduction_fraction,
        "source_validation_recall_damage": source_validation_damage,
        "extended_validation_recall_damage": extended_validation_damage,
        "feasible": feasible,
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
        "arm": f"rate-{rate:.2f}",
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


def combine(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the smallest feasible frozen corruption rate."""
    rates = {float(arm["scientific"]["rate"]) for arm in arms}
    if rates != set(RATES):
        raise ValueError("corruption calibration arm coverage is incomplete")
    by_rate = {
        float(arm["scientific"]["rate"]): arm
        for arm in arms
    }
    pipeline = {
        f"{rate:.2f}": (
            arm.get("experiment_id") == EXPERIMENT_ID
            and bool(arm["scientific"]["pipeline_passed"])
        )
        for rate, arm in by_rate.items()
    }
    feasible = sorted(
        rate
        for rate, arm in by_rate.items()
        if bool(arm["scientific"]["feasible"])
    )
    selected = feasible[0] if all(pipeline.values()) and feasible else None
    classification = (
        "local_geometry_corruption_calibrated"
        if selected is not None
        else (
            "local_geometry_corruption_not_calibrated"
            if all(pipeline.values())
            else "local_geometry_corruption_pipeline_invalid"
        )
    )
    scientific = {
        "classification": classification,
        "selected_rate": selected,
        "feasible_rates": feasible,
        "arms": {
            f"{rate:.2f}": arm
            for rate, arm in sorted(by_rate.items())
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
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
    rows = []
    for rate, arm in sorted(scientific["arms"].items()):
        value = arm["scientific"]
        rows.append(
            f"| {rate} | {value['gap_reduction_fraction']:.2%} | "
            f"{value['source_validation_recall_damage']:+.2%} | "
            f"{value['extended_validation_recall_damage']:+.2%} | "
            f"{value['feasible']} |"
        )
    return f"""# Local Geometry Corruption Calibration V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

Classification: **`{scientific["classification"]}`**

| Rate | Gap reduction | Source validation damage | Extended validation damage | Feasible |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Selected rate: `{scientific["selected_rate"]}`.

Combined scientific BLAKE3:
`{report["scientific_blake3"]}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--rate", type=float, choices=RATES, required=True)
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
            rate=args.rate,
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
