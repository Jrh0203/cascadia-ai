#!/usr/bin/env python3
"""Audit public-prior tile retrieval baselines for ADR 0115."""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    DESCENDANT_VALUE_DIM,
    EXPERIMENT_ID,
    STAGE_WIDTHS,
    TILE_FACTOR_DIM,
    TILE_LOCAL_DIM,
    HierarchicalFactorCache,
)

STATS_OFFSET = TILE_FACTOR_DIM + TILE_LOCAL_DIM
PRIOR_INDEX = {
    "model_immediate_score": 0,
    "model_remaining_value": 1,
    "screen_value": 2,
    "screen_rank_scaled": 3,
    "screen_inverse_rank": 4,
    "uniform_market_survival_proxy": 5,
}
BASELINES = {
    "maximum_model_immediate_score": ("maximum", "model_immediate_score", 1),
    "maximum_model_remaining_value": ("maximum", "model_remaining_value", 1),
    "maximum_screen_value": ("maximum", "screen_value", 1),
    "minimum_screen_rank": ("minimum", "screen_rank_scaled", -1),
    "maximum_screen_inverse_rank": ("maximum", "screen_inverse_rank", 1),
    "maximum_market_survival": (
        "maximum",
        "uniform_market_survival_proxy",
        1,
    ),
}


def _feature_column(
    statistic: str,
    feature: str,
) -> int:
    statistic_offset = {
        "minimum": 0,
        "mean": DESCENDANT_VALUE_DIM,
        "maximum": DESCENDANT_VALUE_DIM * 2,
    }[statistic]
    return STATS_OFFSET + statistic_offset + PRIOR_INDEX[feature]


def audit_split(cache: HierarchicalFactorCache) -> dict[str, Any]:
    accumulators = {
        name: {
            "queries": 0,
            "target_factors": 0,
            "target_hits": 0,
            "exact_queries": 0,
        }
        for name in BASELINES
    }
    for arrays in cache.iter_shards():
        offsets = arrays["tile_query_offsets"]
        features = arrays["tile_item_features"]
        targets = arrays["tile_item_target"]
        for left, right in pairwise(offsets):
            left = int(left)
            right = int(right)
            quota = int(np.sum(targets[left:right]))
            for name, (statistic, feature, direction) in BASELINES.items():
                column = _feature_column(statistic, feature)
                scores = features[left:right, column] * direction
                ranking = sorted(
                    range(right - left),
                    key=lambda index: (-float(scores[index]), index),
                )
                selected = np.asarray(
                    ranking[: min(STAGE_WIDTHS["tile"], right - left)],
                    dtype=np.int32,
                )
                hits = int(np.sum(targets[left:right][selected]))
                accumulator = accumulators[name]
                accumulator["queries"] += 1
                accumulator["target_factors"] += quota
                accumulator["target_hits"] += hits
                accumulator["exact_queries"] += int(hits == quota)
    return {
        name: {
            **values,
            "target_factor_recall": values["target_hits"]
            / max(values["target_factors"], 1),
            "exact_query_fraction": values["exact_queries"]
            / max(values["queries"], 1),
        }
        for name, values in accumulators.items()
    }


def run(
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "analysis": "conditional-tile-public-prior-baselines-v1",
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest[
            "payload_blake3"
        ],
        "train": audit_split(train),
        "validation": audit_split(validation),
        "training_used": False,
        "gradients_used": False,
        "optimizer_updates_used": False,
        "test_split_opened": False,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.train_cache, args.validation_cache)
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
