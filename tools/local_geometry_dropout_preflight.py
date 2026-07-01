#!/usr/bin/env python3
"""Run the contingent ADR 0124 implementation preflights."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.conditional_tile_local_geometry_dropout import (
    DROPOUT_RATE,
    EPOCHS,
    EXPERIMENT_ID,
    LOCAL_LEFT,
    LOCAL_RIGHT,
    STAGE,
    _query_batches_with_dropout,
    corrupt_query_local_geometry,
    dropout_count,
    selected_item_indices,
)
from cascadia_mlx.conditional_tile_target_only import (
    BATCH_SIZE,
    SEED,
    target_only_tile_loss,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    HierarchicalFactorCache,
    _query_batches,
    build_stage_model,
)
from mlx.utils import tree_flatten

PREFLIGHT_ID = "conditional-tile-local-geometry-dropout-preflight-repair-v1"
ARMS = ("contract", "coverage", "gradient", "resource")
FROZEN_SELECTION_BLAKE3 = "87a234b381161f78eeefc63199dac85ba342492ed79cee060204a8f36516ed4e"


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


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if peak < 1024 * 1024:
        peak *= 1024
    return {
        "peak_process_rss_bytes": peak,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
    }


def _selection_digest_update(
    digest: Any,
    *,
    epoch: int,
    shard_index: int,
    query_index: int,
    hashes: np.ndarray,
    selected: np.ndarray,
) -> None:
    digest.update(epoch.to_bytes(2, "little"))
    digest.update(shard_index.to_bytes(4, "little"))
    digest.update(query_index.to_bytes(4, "little"))
    digest.update(len(selected).to_bytes(4, "little"))
    digest.update(np.ascontiguousarray(hashes[selected]).tobytes())


def contract_preflight(cache_root: Path) -> dict[str, Any]:
    """Prove that epoch-one corruption changes only the frozen block."""
    started = time.perf_counter()
    cache = HierarchicalFactorCache(cache_root)
    digest = blake3.blake3()
    queries = 0
    items = 0
    selected_total = 0
    selected_count_exact = True
    nonlocal_exact = True
    unselected_local_exact = True
    selected_rotation_exact = True
    source_unchanged = True
    for shard_index, arrays in enumerate(cache.iter_shards()):
        source = arrays["tile_item_features"]
        source_snapshot = source.copy()
        offsets = arrays["tile_query_offsets"]
        hashes = arrays["tile_item_hash"]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            original = source[left:right]
            query_hashes = hashes[left:right]
            changed, selected = corrupt_query_local_geometry(
                original,
                query_hashes,
                epoch=1,
                shard_index=shard_index,
                query_index=query_index,
            )
            unselected = np.setdiff1d(
                np.arange(len(original)),
                selected,
                assume_unique=True,
            )
            selected_count_exact &= len(selected) == dropout_count(len(original))
            nonlocal_exact &= np.array_equal(
                changed[:, :LOCAL_LEFT],
                original[:, :LOCAL_LEFT],
            ) and np.array_equal(
                changed[:, LOCAL_RIGHT:],
                original[:, LOCAL_RIGHT:],
            )
            unselected_local_exact &= np.array_equal(
                changed[unselected, LOCAL_LEFT:LOCAL_RIGHT],
                original[unselected, LOCAL_LEFT:LOCAL_RIGHT],
            )
            if len(selected) >= 2:
                selected_rotation_exact &= np.array_equal(
                    changed[selected, LOCAL_LEFT:LOCAL_RIGHT],
                    np.roll(
                        original[selected, LOCAL_LEFT:LOCAL_RIGHT],
                        shift=1,
                        axis=0,
                    ),
                )
            _selection_digest_update(
                digest,
                epoch=1,
                shard_index=shard_index,
                query_index=query_index,
                hashes=query_hashes,
                selected=selected,
            )
            queries += 1
            items += len(original)
            selected_total += len(selected)
        source_unchanged &= np.array_equal(source, source_snapshot)
    gates = {
        "train_cache_only": cache.split == "train",
        "selected_count_exact": selected_count_exact,
        "nonlocal_columns_byte_exact": nonlocal_exact,
        "unselected_local_columns_byte_exact": unselected_local_exact,
        "selected_local_rotation_exact": selected_rotation_exact,
        "source_arrays_unchanged": source_unchanged,
        "query_coverage_exact": queries == int(cache.manifest["queries"][STAGE]),
        "item_coverage_exact": items == int(cache.manifest["items"][STAGE]),
    }
    scientific = {
        "arm": "contract",
        "queries": queries,
        "items": items,
        "selected_items": selected_total,
        "selected_fraction": selected_total / max(items, 1),
        "epoch_one_selection_blake3": digest.hexdigest(),
        "gates": gates,
        "passed": all(gates.values()),
        "train_cache_payload_blake3": cache.manifest["payload_blake3"],
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _arm_report(scientific, started)


def coverage_preflight(cache_root: Path) -> dict[str, Any]:
    """Measure deterministic selection coverage over all 200 epochs."""
    started = time.perf_counter()
    cache = HierarchicalFactorCache(cache_root)
    digest = blake3.blake3()
    all_rates = []
    exact_counts = True
    selected_total = 0
    exposure_total = 0
    queries = 0
    items = 0
    for shard_index, arrays in enumerate(cache.iter_shards()):
        hashes = arrays["tile_item_hash"]
        offsets = arrays["tile_query_offsets"]
        counts = np.zeros(len(hashes), dtype=np.uint16)
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            query_hashes = hashes[left:right]
            expected = dropout_count(len(query_hashes))
            for epoch in range(1, EPOCHS + 1):
                selected = selected_item_indices(
                    query_hashes,
                    epoch=epoch,
                    shard_index=shard_index,
                    query_index=query_index,
                )
                exact_counts &= len(selected) == expected
                counts[left + selected] += 1
                selected_total += len(selected)
                exposure_total += len(query_hashes)
                if epoch == 1:
                    _selection_digest_update(
                        digest,
                        epoch=epoch,
                        shard_index=shard_index,
                        query_index=query_index,
                        hashes=query_hashes,
                        selected=selected,
                    )
            queries += 1
            items += len(query_hashes)
        all_rates.append(counts.astype(np.float64) / EPOCHS)
    rates = np.concatenate(all_rates)
    expected_fraction = selected_total / max(exposure_total, 1)
    gates = {
        "train_cache_only": cache.split == "train",
        "exact_selected_count_every_query_epoch": exact_counts,
        "query_coverage_exact": queries == int(cache.manifest["queries"][STAGE]),
        "item_coverage_exact": items == int(cache.manifest["items"][STAGE]),
        "no_item_never_selected": bool(np.all(rates > 0.0)),
        "no_item_always_selected": bool(np.all(rates < 1.0)),
        "minimum_selection_rate_at_least_0_30": (float(np.min(rates)) >= 0.30),
        "maximum_selection_rate_at_most_0_70": (float(np.max(rates)) <= 0.70),
        "mean_selection_rate_exact": math.isclose(
            float(np.mean(rates)),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    scientific = {
        "arm": "coverage",
        "epochs": EPOCHS,
        "queries": queries,
        "items": items,
        "selection_rate": {
            "minimum": float(np.min(rates)),
            "p01": float(np.quantile(rates, 0.01)),
            "median": float(np.median(rates)),
            "mean": float(np.mean(rates)),
            "p99": float(np.quantile(rates, 0.99)),
            "maximum": float(np.max(rates)),
            "expected": expected_fraction,
        },
        "epoch_one_selection_blake3": digest.hexdigest(),
        "gates": gates,
        "passed": all(gates.values()),
        "train_cache_payload_blake3": cache.manifest["payload_blake3"],
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _arm_report(scientific, started)


def _flat_norm(tree: object) -> float:
    arrays = [np.asarray(value) for _name, value in tree_flatten(tree)]
    return math.sqrt(sum(float(np.sum(value * value)) for value in arrays))


def _flat_delta_norm(left: object, right: object) -> float:
    left_flat = tree_flatten(left)
    right_flat = tree_flatten(right)
    if [name for name, _value in left_flat] != [name for name, _value in right_flat]:
        raise ValueError("gradient tree structure drifted")
    return math.sqrt(
        sum(
            float(np.sum((np.asarray(left_value) - np.asarray(right_value)) ** 2))
            for (_name, left_value), (_other, right_value) in zip(
                left_flat,
                right_flat,
                strict=True,
            )
        )
    )


def gradient_preflight(cache_root: Path) -> dict[str, Any]:
    """Prove that the treatment changes finite optimization signal."""
    started = time.perf_counter()
    cache = HierarchicalFactorCache(cache_root)
    arrays = next(cache.iter_shards())
    baseline_values = next(
        _query_batches(
            arrays,
            stage=STAGE,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEED,
        )
    )
    dropout_values, dropped, total = next(
        _query_batches_with_dropout(
            arrays,
            batch_size=BATCH_SIZE,
            epoch=1,
            shard_index=0,
            shuffle=False,
            seed=SEED,
        )
    )
    mx.random.seed(SEED)
    model = build_stage_model(STAGE)
    mx.eval(model.parameters())
    loss_and_grad = nn.value_and_grad(model, target_only_tile_loss)
    baseline_mx = tuple(mx.array(value) for value in baseline_values)
    dropout_mx = tuple(mx.array(value) for value in dropout_values)
    baseline_loss, baseline_gradients = loss_and_grad(
        model,
        *baseline_mx,
    )
    dropout_loss, dropout_gradients = loss_and_grad(
        model,
        *dropout_mx,
    )

    def input_loss(item_values: mx.array) -> mx.array:
        values = list(baseline_mx)
        values[2] = item_values
        return target_only_tile_loss(model, *values)

    baseline_input_gradient = mx.grad(input_loss)(baseline_mx[2])

    def dropout_input_loss(item_values: mx.array) -> mx.array:
        values = list(dropout_mx)
        values[2] = item_values
        return target_only_tile_loss(model, *values)

    dropout_input_gradient = mx.grad(dropout_input_loss)(dropout_mx[2])
    mx.eval(
        baseline_loss,
        dropout_loss,
        baseline_gradients,
        dropout_gradients,
        baseline_input_gradient,
        dropout_input_gradient,
    )
    baseline_input = np.asarray(baseline_input_gradient)
    dropout_input = np.asarray(dropout_input_gradient)
    baseline_parameter_norm = _flat_norm(baseline_gradients)
    dropout_parameter_norm = _flat_norm(dropout_gradients)
    parameter_delta_norm = _flat_delta_norm(
        baseline_gradients,
        dropout_gradients,
    )
    values = {
        "baseline_loss": float(baseline_loss.item()),
        "dropout_loss": float(dropout_loss.item()),
        "baseline_parameter_gradient_norm": baseline_parameter_norm,
        "dropout_parameter_gradient_norm": dropout_parameter_norm,
        "parameter_gradient_delta_norm": parameter_delta_norm,
        "baseline_local_input_gradient_l1": float(
            np.sum(np.abs(baseline_input[..., LOCAL_LEFT:LOCAL_RIGHT]))
        ),
        "dropout_local_input_gradient_l1": float(
            np.sum(np.abs(dropout_input[..., LOCAL_LEFT:LOCAL_RIGHT]))
        ),
        "dropout_nonlocal_input_gradient_l1": float(
            np.sum(np.abs(dropout_input[..., :LOCAL_LEFT]))
            + np.sum(np.abs(dropout_input[..., LOCAL_RIGHT:]))
        ),
    }
    gates = {
        "train_cache_only": cache.split == "train",
        "dropout_fraction_exact": (
            dropped
            == sum(
                dropout_count(int(width))
                for width in (
                    arrays["tile_query_offsets"][1 : BATCH_SIZE + 1]
                    - arrays["tile_query_offsets"][:BATCH_SIZE]
                )
            )
            and total == int(arrays["tile_query_offsets"][BATCH_SIZE])
        ),
        "all_values_finite": all(math.isfinite(value) for value in values.values()),
        "baseline_parameter_gradient_nonzero": baseline_parameter_norm > 0.0,
        "dropout_parameter_gradient_nonzero": dropout_parameter_norm > 0.0,
        "treatment_changes_loss": not math.isclose(
            values["baseline_loss"],
            values["dropout_loss"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "treatment_changes_parameter_gradient": parameter_delta_norm > 1e-9,
        "dropout_local_input_gradient_nonzero": (values["dropout_local_input_gradient_l1"] > 0.0),
        "dropout_nonlocal_input_gradient_nonzero": (
            values["dropout_nonlocal_input_gradient_l1"] > 0.0
        ),
    }
    scientific = {
        "arm": "gradient",
        "batch_queries": BATCH_SIZE,
        "batch_items": total,
        "dropout_items": dropped,
        "values": values,
        "gates": gates,
        "passed": all(gates.values()),
        "train_cache_payload_blake3": cache.manifest["payload_blake3"],
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _arm_report(scientific, started)


def _consume_baseline(cache: HierarchicalFactorCache) -> int:
    total = 0
    for shard_index, arrays in enumerate(cache.iter_shards()):
        for values in _query_batches(
            arrays,
            stage=STAGE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED + 1000 + shard_index,
        ):
            total += int(np.sum(values[3]))
    return total


def _consume_dropout(cache: HierarchicalFactorCache) -> int:
    total = 0
    for shard_index, arrays in enumerate(cache.iter_shards()):
        for values, _dropped, _eligible in _query_batches_with_dropout(
            arrays,
            batch_size=BATCH_SIZE,
            epoch=1,
            shard_index=shard_index,
            shuffle=True,
            seed=SEED + 1000 + shard_index,
        ):
            total += int(np.sum(values[3]))
    return total


def resource_preflight(cache_root: Path) -> dict[str, Any]:
    """Measure host-local CPU preparation overhead without training."""
    started = time.perf_counter()
    cache = HierarchicalFactorCache(cache_root)
    baseline_seconds = []
    dropout_seconds = []
    baseline_items = []
    dropout_items = []
    for _repeat in range(3):
        before = time.perf_counter()
        baseline_items.append(_consume_baseline(cache))
        baseline_seconds.append(time.perf_counter() - before)
        before = time.perf_counter()
        dropout_items.append(_consume_dropout(cache))
        dropout_seconds.append(time.perf_counter() - before)
    baseline_median = float(np.median(baseline_seconds))
    dropout_median = float(np.median(dropout_seconds))
    overhead_fraction = dropout_median / baseline_median - 1.0
    usage = _resource_usage()
    gates = {
        "train_cache_only": cache.split == "train",
        "baseline_coverage_exact": all(
            value == int(cache.manifest["items"][STAGE]) for value in baseline_items
        ),
        "dropout_coverage_exact": all(
            value == int(cache.manifest["items"][STAGE]) for value in dropout_items
        ),
        "timings_finite_positive": all(
            math.isfinite(value) and value > 0 for value in baseline_seconds + dropout_seconds
        ),
        "preparation_overhead_at_most_0_50": overhead_fraction <= 0.50,
        "peak_rss_below_4_gib": (usage["peak_process_rss_bytes"] < 4 * 1024**3),
        "zero_process_swaps": usage["process_swaps"] == 0,
    }
    scientific = {
        "arm": "resource",
        "repeats": 3,
        "baseline_seconds": baseline_seconds,
        "dropout_seconds": dropout_seconds,
        "baseline_median_seconds": baseline_median,
        "dropout_median_seconds": dropout_median,
        "preparation_overhead_fraction": overhead_fraction,
        "resource_usage": usage,
        "gates": gates,
        "passed": all(gates.values()),
        "train_cache_payload_blake3": cache.manifest["payload_blake3"],
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _arm_report(scientific, started)


def _arm_report(scientific: dict[str, Any], started: float) -> dict[str, Any]:
    scientific["dropout_rate"] = DROPOUT_RATE
    return {
        "schema_version": 1,
        "experiment_id": PREFLIGHT_ID,
        "treatment_experiment_id": EXPERIMENT_ID,
        "arm": scientific["arm"],
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


def classify_preflight(arms: list[dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    """Classify all implementation evidence before contingent training."""
    by_arm = {arm["arm"]: arm for arm in arms}
    if set(by_arm) != set(ARMS):
        raise ValueError("dropout preflight arm coverage is incomplete")
    payloads = {arm["scientific"]["train_cache_payload_blake3"] for arm in arms}
    gates = {
        "all_arm_identities": all(
            arm.get("experiment_id") == PREFLIGHT_ID
            and arm.get("treatment_experiment_id") == EXPERIMENT_ID
            for arm in arms
        ),
        "all_arms_passed": all(bool(arm["scientific"]["passed"]) for arm in arms),
        "cache_identity_shared": len(payloads) == 1,
        "cross_host_epoch_one_selection_exact": (
            by_arm["contract"]["scientific"]["epoch_one_selection_blake3"]
            == by_arm["coverage"]["scientific"]["epoch_one_selection_blake3"]
        ),
        "original_epoch_one_selection_preserved": (
            by_arm["contract"]["scientific"]["epoch_one_selection_blake3"]
            == FROZEN_SELECTION_BLAKE3
        ),
        "all_closed_domains_preserved": all(
            not bool(arm["scientific"][field])
            for arm in arms
            for field in (
                "validation_opened",
                "test_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        ),
    }
    classification = (
        "local_geometry_dropout_preflight_passed"
        if all(gates.values())
        else "local_geometry_dropout_preflight_invalid"
    )
    return classification, gates


def combine(arms: list[dict[str, Any]]) -> dict[str, Any]:
    classification, gates = classify_preflight(arms)
    scientific = {
        "classification": classification,
        "gates": gates,
        "arms": {arm["arm"]: arm for arm in sorted(arms, key=lambda x: x["arm"])},
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": PREFLIGHT_ID,
        "treatment_experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _blake3(scientific),
    }


def render_markdown(report: dict[str, Any]) -> str:
    scientific = report["scientific"]
    rows = "\n".join(
        f"| {name} | {arm['host']} | {arm['scientific']['passed']} | {arm['elapsed_seconds']:.2f} |"
        for name, arm in scientific["arms"].items()
    )
    failed = [name for name, passed in scientific["gates"].items() if not passed]
    failed_text = ", ".join(failed) if failed else "None"
    return f"""# Conditional Tile Local-Geometry Dropout Preflight

Date: 2026-06-16

Classification: **`{scientific["classification"]}`**

| Arm | Host | Passed | Seconds |
|---|---|---:|---:|
{rows}

Failed combined gates: {failed_text}.

Combined scientific BLAKE3:
`{report["scientific_blake3"]}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for arm in ARMS:
        arm_parser = subparsers.add_parser(arm)
        arm_parser.add_argument("--train-cache", type=Path, required=True)
        arm_parser.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--arm", type=Path, action="append", required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "contract":
        report = contract_preflight(args.train_cache)
    elif args.command == "coverage":
        report = coverage_preflight(args.train_cache)
    elif args.command == "gradient":
        report = gradient_preflight(args.train_cache)
    elif args.command == "resource":
        report = resource_preflight(args.train_cache)
    else:
        report = combine([json.loads(path.read_text()) for path in args.arm])
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
