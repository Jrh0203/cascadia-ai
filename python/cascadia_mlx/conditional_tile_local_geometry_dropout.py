"""Contingent local-geometry dropout treatment for ADR 0124."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.conditional_tile_optimizer_schedule import (
    EPOCHS,
    SCHEDULE_ID,
    late_cosine_learning_rate,
)
from cascadia_mlx.conditional_tile_target_only import (
    OBJECTIVE_ID,
    target_only_tile_loss,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGE_ITEM_DIMS,
    HierarchicalFactorCache,
    StageTrainingConfig,
    _append_json,
    _host,
    _resource_usage,
    _scientific_blake3,
    _tree_finite,
    _write_json_atomic,
    build_stage_model,
    evaluate_integrated,
    evaluate_mixed_stage_ceiling,
    evaluate_stage,
    membership_stage_selection_key,
    replay_stage,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

EXPERIMENT_ID = "conditional-tile-local-geometry-dropout-v1"
STAGE = "tile"
DROPOUT_RATE = 0.50
DROPOUT_SEED = 2026061650
LOCAL_LEFT = 8
LOCAL_RIGHT = 188
CORRUPTION_ID = "epoch-hash-half-query-local-geometry-rotation-v1"

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_MIX_A = np.uint64(0x9E3779B97F4A7C15)
_MIX_B = np.uint64(0xBF58476D1CE4E5B9)
_MIX_C = np.uint64(0x94D049BB133111EB)


def frozen_config() -> StageTrainingConfig:
    """Return the ADR 0120 config; only training inputs are corrupted."""
    from cascadia_mlx.conditional_tile_optimizer_schedule import frozen_config

    return frozen_config()


def _mix_u64(values: np.ndarray | np.uint64) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = (values + _MIX_A) & _MASK64
        values = ((values ^ (values >> np.uint64(30))) * _MIX_B) & _MASK64
        values = ((values ^ (values >> np.uint64(27))) * _MIX_C) & _MASK64
    return values ^ (values >> np.uint64(31))


def dropout_count(width: int) -> int:
    """Return the exact calibrated item count for one query."""
    if width <= 0:
        raise ValueError("query width must be positive")
    if width == 1:
        return 1
    return min(width, max(2, math.ceil(DROPOUT_RATE * width)))


def selected_item_indices(
    hashes: np.ndarray,
    *,
    epoch: int,
    shard_index: int,
    query_index: int,
) -> np.ndarray:
    """Select an exact epoch-varying half-query by immutable item hash."""
    if hashes.ndim != 2 or hashes.shape[1] != 16:
        raise ValueError("tile item hashes must have shape [items, 16]")
    if not 1 <= epoch <= EPOCHS:
        raise ValueError("dropout epoch is outside the frozen contract")
    if shard_index < 0 or query_index < 0:
        raise ValueError("dropout shard and query indices must be nonnegative")
    prefix = np.ascontiguousarray(hashes[:, :8]).view("<u8").reshape(-1)
    salt_input = np.uint64(DROPOUT_SEED)
    with np.errstate(over="ignore"):
        salt_input ^= np.uint64(epoch) * _MIX_A
        salt_input ^= np.uint64(shard_index + 1) * _MIX_B
        salt_input ^= np.uint64(query_index + 1) * _MIX_C
    salt = _mix_u64(salt_input).reshape(())
    keys = _mix_u64(prefix ^ salt)
    positions = np.arange(len(hashes), dtype=np.int64)
    count = dropout_count(len(hashes))
    if count == len(hashes):
        selected = positions
    else:
        cutoff = np.partition(keys, count - 1)[count - 1]
        lower = positions[keys < cutoff]
        equal = positions[keys == cutoff]
        needed = count - len(lower)
        selected = np.concatenate((lower, np.sort(equal)[:needed]))
    return selected[np.lexsort((selected, keys[selected]))]


def corrupt_query_local_geometry(
    items: np.ndarray,
    hashes: np.ndarray,
    *,
    epoch: int,
    shard_index: int,
    query_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate local geometry among the selected half of one query."""
    if items.ndim != 2 or items.shape[1] != STAGE_ITEM_DIMS[STAGE]:
        raise ValueError("tile item feature shape drifted")
    if len(items) != len(hashes):
        raise ValueError("tile item and hash counts differ")
    selected = selected_item_indices(
        hashes,
        epoch=epoch,
        shard_index=shard_index,
        query_index=query_index,
    )
    changed = items.copy()
    if len(selected) >= 2:
        changed[selected, LOCAL_LEFT:LOCAL_RIGHT] = np.roll(
            items[selected, LOCAL_LEFT:LOCAL_RIGHT],
            shift=1,
            axis=0,
        )
    return changed, selected


def _query_batches_with_dropout(
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    epoch: int,
    shard_index: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[tuple[np.ndarray, ...], int, int]]:
    offsets = arrays["tile_query_offsets"]
    query_count = len(offsets) - 1
    order = np.arange(query_count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, query_count, batch_size):
        selected_queries = order[start : start + batch_size]
        widths = offsets[selected_queries + 1] - offsets[selected_queries]
        maximum = int(np.max(widths))
        items = np.zeros(
            (len(selected_queries), maximum, STAGE_ITEM_DIMS[STAGE]),
            dtype=np.float32,
        )
        item_mask = np.zeros(
            (len(selected_queries), maximum),
            dtype=np.bool_,
        )
        ranks = np.zeros_like(item_mask, dtype=np.float32)
        rank_mask = np.zeros_like(item_mask)
        target = np.zeros_like(item_mask)
        dropped = 0
        total = 0
        for row, query_index in enumerate(selected_queries):
            left = int(offsets[query_index])
            right = int(offsets[query_index + 1])
            width = right - left
            source = arrays["tile_item_features"][left:right]
            hashes = arrays["tile_item_hash"][left:right]
            selected = selected_item_indices(
                hashes,
                epoch=epoch,
                shard_index=shard_index,
                query_index=int(query_index),
            )
            items[row, :width] = source
            if len(selected) >= 2:
                items[row, selected, LOCAL_LEFT:LOCAL_RIGHT] = np.roll(
                    source[selected, LOCAL_LEFT:LOCAL_RIGHT],
                    shift=1,
                    axis=0,
                )
            item_mask[row, :width] = True
            ranks[row, :width] = arrays["tile_item_rank"][left:right]
            rank_mask[row, :width] = arrays["tile_item_rank_mask"][left:right]
            target[row, :width] = arrays["tile_item_target"][left:right]
            dropped += len(selected)
            total += width
        groups = arrays["tile_query_group"][selected_queries]
        yield (
            (
                arrays["group_state"][groups],
                arrays["tile_query_context"][selected_queries],
                items,
                item_mask,
                ranks,
                rank_mask,
                target,
            ),
            dropped,
            total,
        )


def train(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Train the sole contingent dropout origin."""
    config = frozen_config()
    if output_root.exists():
        raise ValueError("hierarchical stage output already exists")
    allocator = configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("hierarchical stage cache split mismatch")
    mx.random.seed(config.seed)
    model = build_stage_model(STAGE)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, target_only_tile_loss)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    finite_training = True
    for epoch in range(1, config.epochs + 1):
        learning_rate = late_cosine_learning_rate(epoch, config.epochs)
        optimizer.learning_rate = learning_rate
        model.train()
        epoch_loss = 0.0
        batches = 0
        dropped = 0
        total = 0
        for shard_index, arrays in enumerate(train_cache.iter_shards()):
            iterator = _query_batches_with_dropout(
                arrays,
                batch_size=config.batch_size,
                epoch=epoch,
                shard_index=shard_index,
                shuffle=True,
                seed=config.seed + epoch * 1000 + shard_index,
            )
            for values, batch_dropped, batch_total in iterator:
                loss, gradients = loss_and_grad(
                    model,
                    *(mx.array(value) for value in values),
                )
                optimizer.update(model, gradients)
                mx.eval(model.parameters(), optimizer.state, loss)
                loss_value = float(loss.item())
                finite_training &= (
                    math.isfinite(loss_value)
                    and _tree_finite(model.parameters())
                    and _tree_finite(optimizer.state)
                )
                if not finite_training:
                    raise RuntimeError("local-geometry dropout training became nonfinite")
                epoch_loss += loss_value
                batches += 1
                dropped += batch_dropped
                total += batch_total
        train_metrics = evaluate_stage(model, train_cache, STAGE)
        event = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": float(optimizer.learning_rate.item()),
            "dropout_items": dropped,
            "dropout_eligible_items": total,
            "dropout_fraction": dropped / max(total, 1),
            "train": train_metrics,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        key = membership_stage_selection_key(train_metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            mx.save_safetensors(
                str(output_root / "best.safetensors"),
                dict(tree_flatten(model.parameters())),
            )
            _write_json_atomic(output_root / "best.json", event)
        mx.clear_cache()
    model.load_weights(str(output_root / "best.safetensors"))
    mx.eval(model.parameters())
    usage = _resource_usage()
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "parameter_count": sum(
            int(value.size) for _name, value in tree_flatten(model.parameters())
        ),
        "weights_blake3": checksum(output_root / "best.safetensors"),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation_cache.manifest["payload_blake3"],
        "train": evaluate_stage(model, train_cache, STAGE),
        "validation": evaluate_stage(model, validation_cache, STAGE),
        "finite_training": finite_training,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **usage,
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
        "objective_id": OBJECTIVE_ID,
        "source_experiment_id": ("conditional-tile-optimizer-schedule-v1"),
        "source_epoch_budget": EPOCHS,
        "treatment_epoch_budget": EPOCHS,
        "schedule_id": SCHEDULE_ID,
        "corruption_id": CORRUPTION_ID,
        "dropout_rate": DROPOUT_RATE,
        "dropout_seed": DROPOUT_SEED,
        "local_feature_columns": [LOCAL_LEFT, LOCAL_RIGHT],
        "validation_corruption_used": False,
        "inference_corruption_used": False,
        "rank_regression_used": False,
        "listwise_loss_used": False,
        "warm_start_used": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report["scientific_blake3"] = _scientific_blake3(report)
    _write_json_atomic(output_root / "report.json", report)
    return report


def _retag(report: dict[str, Any]) -> dict[str, Any]:
    report["experiment_id"] = EXPERIMENT_ID
    report["source_pipeline_experiment_id"] = "full-legal-hierarchical-factor-retrieval-pilot-v1"
    report["source_treatment_experiment_id"] = "conditional-tile-optimizer-schedule-v1"
    report["schedule_id"] = SCHEDULE_ID
    report["corruption_id"] = CORRUPTION_ID
    return report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-cache", type=Path, required=True)
    train_parser.add_argument("--validation-cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--train-cache", type=Path, required=True)
    replay_parser.add_argument("--validation-cache", type=Path, required=True)
    replay_parser.add_argument("--weights", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)

    mixed_parser = subparsers.add_parser("mixed-ceiling")
    mixed_parser.add_argument("--train-cache", type=Path, required=True)
    mixed_parser.add_argument("--validation-cache", type=Path, required=True)
    mixed_parser.add_argument("--weights", type=Path, required=True)
    mixed_parser.add_argument("--output", type=Path, required=True)

    integration_parser = subparsers.add_parser("integrated")
    integration_parser.add_argument("--train-cache", type=Path, required=True)
    integration_parser.add_argument("--validation-cache", type=Path, required=True)
    integration_parser.add_argument("--draft-weights", type=Path, required=True)
    integration_parser.add_argument("--tile-weights", type=Path, required=True)
    integration_parser.add_argument("--wildlife-weights", type=Path, required=True)
    integration_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "train":
        report = train(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            output_root=args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "replay":
        report = _retag(
            replay_stage(
                stage=STAGE,
                weights=args.weights,
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
            )
        )
    elif args.command == "mixed-ceiling":
        report = _retag(
            evaluate_mixed_stage_ceiling(
                stage=STAGE,
                weights=args.weights,
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
            )
        )
    else:
        report = _retag(
            evaluate_integrated(
                train_cache_root=args.train_cache,
                validation_cache_root=args.validation_cache,
                weights={
                    "draft": args.draft_weights,
                    "tile": args.tile_weights,
                    "wildlife": args.wildlife_weights,
                },
            )
        )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
