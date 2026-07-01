#!/usr/bin/env python3
"""Measure learned-tile versus screen-prior retrieval complementarity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    DESCENDANT_VALUE_DIM,
    EXPERIMENT_ID,
    STAGE_WIDTHS,
    TILE_FACTOR_DIM,
    TILE_LOCAL_DIM,
    HierarchicalFactorCache,
    load_stage_model,
    score_stage_shard,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

ANALYSIS_ID = "conditional-tile-screen-complementarity-audit-v1"
MAXIMUM_SCREEN_VALUE_COLUMN = TILE_FACTOR_DIM + TILE_LOCAL_DIM + 2 * DESCENDANT_VALUE_DIM + 2


@dataclass
class _Accumulator:
    queries: int = 0
    target_factors: int = 0
    learned_hits: int = 0
    prior_hits: int = 0
    union_hits: int = 0
    learned_exact: int = 0
    prior_exact: int = 0
    union_exact: int = 0
    selected_slots: int = 0
    selected_overlap: int = 0
    shared_target_hits: int = 0
    learned_only_target_hits: int = 0
    prior_only_target_hits: int = 0

    def add(
        self,
        target: np.ndarray,
        learned: set[int],
        prior: set[int],
    ) -> None:
        quota = int(np.sum(target))
        union = learned | prior
        learned_hits = sum(bool(target[index]) for index in learned)
        prior_hits = sum(bool(target[index]) for index in prior)
        union_hits = sum(bool(target[index]) for index in union)
        self.queries += 1
        self.target_factors += quota
        self.learned_hits += learned_hits
        self.prior_hits += prior_hits
        self.union_hits += union_hits
        self.learned_exact += int(learned_hits == quota)
        self.prior_exact += int(prior_hits == quota)
        self.union_exact += int(union_hits == quota)
        self.selected_slots += len(learned)
        self.selected_overlap += len(learned & prior)
        self.shared_target_hits += sum(bool(target[index]) for index in learned & prior)
        self.learned_only_target_hits += sum(bool(target[index]) for index in learned - prior)
        self.prior_only_target_hits += sum(bool(target[index]) for index in prior - learned)

    def report(self) -> dict[str, float | int]:
        return {
            "queries": self.queries,
            "target_factors": self.target_factors,
            "learned_target_hits": self.learned_hits,
            "prior_target_hits": self.prior_hits,
            "union_target_hits": self.union_hits,
            "learned_target_recall": self.learned_hits / max(self.target_factors, 1),
            "prior_target_recall": self.prior_hits / max(self.target_factors, 1),
            "union_oracle_rerank_target_recall": self.union_hits / max(self.target_factors, 1),
            "learned_exact_query_fraction": self.learned_exact / max(self.queries, 1),
            "prior_exact_query_fraction": self.prior_exact / max(self.queries, 1),
            "union_oracle_rerank_exact_query_fraction": self.union_exact / max(self.queries, 1),
            "mean_selected_overlap_fraction": self.selected_overlap / max(self.selected_slots, 1),
            "shared_target_hits": self.shared_target_hits,
            "learned_only_target_hits": self.learned_only_target_hits,
            "prior_only_target_hits": self.prior_only_target_hits,
        }


def audit_cache(
    model: object,
    cache: HierarchicalFactorCache,
) -> dict[str, float | int | bool]:
    accumulator = _Accumulator()
    items = 0
    for arrays in cache.iter_shards():
        learned_scores = score_stage_shard(model, arrays, "tile")
        prior_scores = arrays["tile_item_features"][:, MAXIMUM_SCREEN_VALUE_COLUMN]
        targets = arrays["tile_item_target"]
        for left, right in pairwise(arrays["tile_query_offsets"]):
            left = int(left)
            right = int(right)
            width = min(STAGE_WIDTHS["tile"], right - left)
            learned = set(
                index - left
                for index in sorted(
                    range(left, right),
                    key=lambda index: (-float(learned_scores[index]), index),
                )[:width]
            )
            prior = set(
                index - left
                for index in sorted(
                    range(left, right),
                    key=lambda index: (-float(prior_scores[index]), index),
                )[:width]
            )
            accumulator.add(targets[left:right], learned, prior)
            items += right - left
    return {
        **accumulator.report(),
        "all_queries_covered": accumulator.queries == int(cache.manifest["queries"]["tile"]),
        "all_items_covered": items == int(cache.manifest["items"]["tile"]),
    }


def run(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    weights: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    model = load_stage_model("tile", weights)
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    scientific = {
        "stage": "tile",
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest["payload_blake3"],
        "train": audit_cache(model, train),
        "validation": audit_cache(model, validation),
        "union_interpretation": (
            "oracle reranking within the union of learned top32 and "
            "maximum-screen-value top32; diagnostic only"
        ),
        "test_split_opened": False,
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "analysis": ANALYSIS_ID,
        "host": socket.gethostname(),
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak,
            "process_swaps": int(usage.ru_nswap),
            "mlx_allocator": allocator,
        },
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
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
        weights=args.weights,
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
