#!/usr/bin/env python3
"""Audit ADR 0115 tile objective gradients on the widest train queries."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    EXPERIMENT_ID,
    STAGE_ITEM_DIMS,
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
    HierarchicalFactorCache,
    HierarchicalFactorRanker,
    load_stage_model,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from mlx.utils import tree_flatten

ANALYSIS_ID = "conditional-tile-objective-gradient-audit-v1"
QUERY_COUNT = 8
CONFLICT_COSINE = -0.25
DOMINATION_RATIO = 0.5


def classify_objective_gradient(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify boundary pressure against the two auxiliary terms."""
    boundary = float(metrics["mean_gradient_norms"]["boundary"])
    auxiliary = float(metrics["mean_combined_auxiliary_gradient_norm"])
    cosine = float(metrics["mean_boundary_auxiliary_gradient_cosine"])
    conflict = cosine <= CONFLICT_COSINE and auxiliary >= DOMINATION_RATIO * boundary
    dominated = boundary < DOMINATION_RATIO * auxiliary
    if conflict:
        primary = "objective_gradient_conflict"
    elif dominated:
        primary = "target_boundary_gradient_dominated"
    else:
        primary = "objective_gradient_pressure_not_primary"
    return {
        "primary": primary,
        "objective_gradient_conflict": conflict,
        "target_boundary_gradient_dominated": dominated,
        "conflict_cosine_gate": CONFLICT_COSINE,
        "domination_ratio_gate": DOMINATION_RATIO,
    }


def _regression_loss(
    model: HierarchicalFactorRanker,
    state: mx.array,
    context: mx.array,
    items: mx.array,
    item_mask: mx.array,
    expected_rank: mx.array,
    expected_rank_mask: mx.array,
    _target: mx.array,
) -> mx.array:
    scores = model(state, context, items, item_mask)
    target = -mx.log1p(expected_rank)
    delta = mx.abs(scores - target)
    smooth_l1 = mx.where(delta < 1.0, 0.5 * delta * delta, delta - 0.5)
    per_query = mx.sum(
        mx.where(expected_rank_mask, smooth_l1, 0.0),
        axis=-1,
    ) / mx.maximum(mx.sum(expected_rank_mask, axis=-1), 1)
    valid = mx.any(expected_rank_mask, axis=-1)
    return mx.sum(mx.where(valid, per_query, 0.0)) / mx.maximum(
        mx.sum(valid),
        1,
    )


def _listwise_loss(
    model: HierarchicalFactorRanker,
    state: mx.array,
    context: mx.array,
    items: mx.array,
    item_mask: mx.array,
    expected_rank: mx.array,
    expected_rank_mask: mx.array,
    _target: mx.array,
) -> mx.array:
    scores = model(state, context, items, item_mask)
    target_logits = mx.where(
        expected_rank_mask,
        -(expected_rank - 1.0) / TARGET_SCALE,
        -1e9,
    )
    target_probability = mx.softmax(target_logits, axis=-1)
    student_logits = mx.where(
        expected_rank_mask,
        scores / STUDENT_TEMPERATURE,
        -1e9,
    )
    log_probability = student_logits - mx.logsumexp(
        student_logits,
        axis=-1,
        keepdims=True,
    )
    return mx.mean(
        -mx.sum(
            mx.where(
                expected_rank_mask,
                target_probability * log_probability,
                0.0,
            ),
            axis=-1,
        )
    )


def _boundary_loss(
    model: HierarchicalFactorRanker,
    state: mx.array,
    context: mx.array,
    items: mx.array,
    item_mask: mx.array,
    _expected_rank: mx.array,
    _expected_rank_mask: mx.array,
    target: mx.array,
) -> mx.array:
    scores = model(state, context, items, item_mask)
    negative = item_mask & ~target
    positive_count = mx.sum(target, axis=-1)
    negative_count = mx.sum(negative, axis=-1)
    positive_loss = mx.sum(
        mx.where(target, nn.softplus(-scores), 0.0),
        axis=-1,
    ) / mx.maximum(positive_count, 1)
    negative_loss = mx.sum(
        mx.where(negative, nn.softplus(scores), 0.0),
        axis=-1,
    ) / mx.maximum(negative_count, 1)
    valid = (positive_count > 0) & (negative_count > 0)
    return mx.sum(mx.where(valid, positive_loss + negative_loss, 0.0)) / mx.maximum(
        mx.sum(valid), 1
    )


def _gradient_vector(gradient: dict[str, Any]) -> np.ndarray:
    values = [
        np.asarray(value, dtype=np.float32).reshape(-1) for _name, value in tree_flatten(gradient)
    ]
    vector = np.concatenate(values)
    if not np.all(np.isfinite(vector)):
        raise ValueError("tile objective gradient contains nonfinite values")
    return vector


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-30))


def _widest_queries(
    cache: HierarchicalFactorCache,
    count: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for shard_index, arrays in enumerate(cache.iter_shards()):
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        rank_masks = arrays["tile_item_rank_mask"]
        for query_index in range(len(offsets) - 1):
            left = int(offsets[query_index])
            right = int(offsets[query_index + 1])
            target_count = int(np.sum(targets[left:right]))
            if (
                target_count == 0
                or target_count == right - left
                or not np.any(rank_masks[left:right])
            ):
                continue
            candidates.append((-(right - left), shard_index, query_index))
    return [
        (-negative_width, shard_index, query_index)
        for negative_width, shard_index, query_index in sorted(candidates)[:count]
    ]


def _query_values(
    arrays: dict[str, np.ndarray],
    query_index: int,
) -> tuple[np.ndarray, ...]:
    left = int(arrays["tile_query_offsets"][query_index])
    right = int(arrays["tile_query_offsets"][query_index + 1])
    width = right - left
    group = int(arrays["tile_query_group"][query_index])
    items = np.zeros((1, width, STAGE_ITEM_DIMS["tile"]), dtype=np.float32)
    items[0] = arrays["tile_item_features"][left:right]
    return (
        arrays["group_state"][group : group + 1],
        arrays["tile_query_context"][query_index : query_index + 1],
        items,
        np.ones((1, width), dtype=np.bool_),
        arrays["tile_item_rank"][None, left:right],
        arrays["tile_item_rank_mask"][None, left:right],
        arrays["tile_item_target"][None, left:right],
    )


def run(
    *,
    cache_root: Path,
    weights: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    cache = HierarchicalFactorCache(cache_root)
    if cache.split != "train":
        raise ValueError("objective audit requires the open train cache")
    model = load_stage_model("tile", weights)
    selected = _widest_queries(cache, QUERY_COUNT)
    selected_by_shard: dict[int, list[tuple[int, int]]] = {}
    for width, shard_index, query_index in selected:
        selected_by_shard.setdefault(shard_index, []).append((width, query_index))

    functions = {
        "regression": nn.value_and_grad(model, _regression_loss),
        "listwise": nn.value_and_grad(model, _listwise_loss),
        "boundary": nn.value_and_grad(model, _boundary_loss),
    }
    rows = []
    for shard_index, arrays in enumerate(cache.iter_shards()):
        for width, query_index in selected_by_shard.get(shard_index, []):
            values = tuple(
                mx.array(value)
                for value in _query_values(
                    arrays,
                    query_index,
                )
            )
            losses: dict[str, float] = {}
            vectors: dict[str, np.ndarray] = {}
            for name, function in functions.items():
                loss, gradient = function(model, *values)
                mx.eval(loss, gradient)
                losses[name] = float(loss.item())
                vectors[name] = _gradient_vector(gradient)
            auxiliary = vectors["regression"] + vectors["listwise"]
            rows.append(
                {
                    "width": width,
                    "shard_index": shard_index,
                    "query_index": query_index,
                    "losses": losses,
                    "gradient_norms": {
                        name: float(np.linalg.norm(vector)) for name, vector in vectors.items()
                    },
                    "combined_auxiliary_gradient_norm": float(np.linalg.norm(auxiliary)),
                    "boundary_regression_gradient_cosine": _cosine(
                        vectors["boundary"],
                        vectors["regression"],
                    ),
                    "boundary_listwise_gradient_cosine": _cosine(
                        vectors["boundary"],
                        vectors["listwise"],
                    ),
                    "boundary_auxiliary_gradient_cosine": _cosine(
                        vectors["boundary"],
                        auxiliary,
                    ),
                }
            )
            del vectors, auxiliary, values
            mx.clear_cache()
    rows.sort(
        key=lambda row: (
            -int(row["width"]),
            int(row["shard_index"]),
            int(row["query_index"]),
        )
    )
    mean_norms = {
        name: float(np.mean([row["gradient_norms"][name] for row in rows])) for name in functions
    }
    metrics = {
        "queries": len(rows),
        "selection": "eight widest tile queries in the open train cache",
        "query_widths": [int(row["width"]) for row in rows],
        "mean_losses": {
            name: float(np.mean([row["losses"][name] for row in rows])) for name in functions
        },
        "mean_gradient_norms": mean_norms,
        "mean_combined_auxiliary_gradient_norm": float(
            np.mean([row["combined_auxiliary_gradient_norm"] for row in rows])
        ),
        "mean_boundary_regression_gradient_cosine": float(
            np.mean([row["boundary_regression_gradient_cosine"] for row in rows])
        ),
        "mean_boundary_listwise_gradient_cosine": float(
            np.mean([row["boundary_listwise_gradient_cosine"] for row in rows])
        ),
        "mean_boundary_auxiliary_gradient_cosine": float(
            np.mean([row["boundary_auxiliary_gradient_cosine"] for row in rows])
        ),
        "all_values_finite": all(
            math.isfinite(float(value))
            for row in rows
            for values in (
                row["losses"].values(),
                row["gradient_norms"].values(),
                (
                    row["combined_auxiliary_gradient_norm"],
                    row["boundary_regression_gradient_cosine"],
                    row["boundary_listwise_gradient_cosine"],
                    row["boundary_auxiliary_gradient_cosine"],
                ),
            )
            for value in values
        ),
        "rows": rows,
    }
    scientific = {
        "stage": "tile",
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": cache.manifest["payload_blake3"],
        "metrics": metrics,
        "classification": classify_objective_gradient(metrics),
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(cache_root=args.cache, weights=args.weights)
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
