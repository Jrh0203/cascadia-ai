"""Quality and serving evidence for the ADR 0161 relational tournament."""

from __future__ import annotations

import platform
import resource
import time
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_metrics import (
    CANDIDATE_CHUNK,
    _system_swap_used_bytes,
    evaluate_r3_action_edit,
)
from cascadia_mlx.relational_substrate_mlx_cache import (
    OPPORTUNITY_NAMES,
    S5_ARM,
    RelationalSubstrateMlxDataset,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateRanker,
)


def evaluate_relational_substrate(
    model: object,
    dataset: RelationalSubstrateMlxDataset,
    *,
    arm: str,
    rows: np.ndarray | None = None,
    candidate_chunk: int = CANDIDATE_CHUNK,
    prediction_panel_size: int = 64,
) -> dict[str, Any]:
    """Evaluate quality once and retain the preregistered strategic slices."""
    metrics = evaluate_r3_action_edit(
        model,
        dataset,
        arm=arm,
        rows=rows,
        candidate_chunk=candidate_chunk,
        prediction_panel_size=prediction_panel_size,
        row_subsets={
            f"{name}_opportunity": subset
            for name, subset in dataset.opportunity_rows.items()
        },
    )
    strategic = [
        float(
            metrics["subsets"][f"{name}_opportunity"][
                "top64_r4800_winner_recall"
            ]
        )
        for name in OPPORTUNITY_NAMES[:3]
    ]
    metrics["strategic_opportunity_recall"] = {
        "elk": strategic[0],
        "salmon": strategic[1],
        "hawk": strategic[2],
        "bear_diagnostic": float(
            metrics["subsets"]["bear_opportunity"][
                "top64_r4800_winner_recall"
            ]
        ),
        "primary_mean": float(np.mean(strategic)),
    }
    if rows is None:
        metrics["parent_tokens"] = dataset.parent_token_statistics(arm)
        metrics["derivative_features"] = dataset.derivative_statistics(arm)
    else:
        metrics["parent_tokens"] = _selected_parent_statistics(
            dataset,
            arm,
            np.asarray(rows, dtype=np.int64),
        )
        metrics["derivative_features"] = _selected_derivative_statistics(
            dataset,
            arm,
            np.asarray(rows, dtype=np.int64),
        )
    return metrics


def benchmark_relational_substrate(
    model: RelationalSubstrateRanker,
    dataset: RelationalSubstrateMlxDataset,
    *,
    arm: str,
    row: int = 0,
    candidate_chunk: int = CANDIDATE_CHUNK,
    warmup_iterations: int = 5,
    steady_iterations: int = 30,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure model-only and materialized complete-decision serving cost."""
    if candidate_chunk <= 0 or warmup_iterations <= 0 or steady_iterations <= 0:
        raise ValueError("relational benchmark dimensions must be positive")
    batch = dataset.batch([row], arm=arm, transform_ids=[0])
    count = int(np.asarray(batch.base.candidate_mask)[0].sum())
    width = min(candidate_chunk, count)
    inputs = _batch_inputs(batch)
    parent_inputs = inputs[:12]

    def encode_parent(*values: mx.array) -> mx.array:
        return model.parent_encoder(_parent_batch(values))

    def predict(*values: mx.array) -> mx.array:
        materialized = _model_batch(values)
        parent = model.encode_parent(materialized)
        return model.predict(
            materialized,
            candidate_slice=slice(0, width),
            parent_state=parent,
        ).scores

    compiled_parent = mx.compile(encode_parent, inputs=model.parent_encoder.state)
    compiled = mx.compile(predict, inputs=model.state)
    mx.clear_cache()
    mx.reset_peak_memory()

    parent_compile_started = time.perf_counter()
    parent_output = compiled_parent(*parent_inputs)
    mx.eval(parent_output)
    parent_compile_seconds = time.perf_counter() - parent_compile_started
    for _ in range(warmup_iterations):
        parent_output = compiled_parent(*parent_inputs)
        mx.eval(parent_output)
    parent_latencies = np.empty(steady_iterations, dtype=np.float64)
    for iteration in range(steady_iterations):
        started = time.perf_counter()
        parent_output = compiled_parent(*parent_inputs)
        mx.eval(parent_output)
        parent_latencies[iteration] = time.perf_counter() - started

    compile_started = time.perf_counter()
    output = compiled(*inputs)
    mx.eval(output)
    compile_seconds = time.perf_counter() - compile_started
    warmup_started = time.perf_counter()
    for _ in range(warmup_iterations):
        output = compiled(*inputs)
        mx.eval(output)
    warmup_seconds = time.perf_counter() - warmup_started
    chunk_latencies = np.empty(steady_iterations, dtype=np.float64)
    for iteration in range(steady_iterations):
        started = time.perf_counter()
        output = compiled(*inputs)
        mx.eval(output)
        chunk_latencies[iteration] = time.perf_counter() - started

    rows = (
        np.arange(min(dataset.group_count, 20), dtype=np.int64)
        if decision_rows is None
        else np.asarray(decision_rows, dtype=np.int64)
    )
    materialization_latencies: list[float] = []
    model_latencies: list[float] = []
    combined_latencies: list[float] = []
    decision_actions = 0
    swap_before = _system_swap_used_bytes()
    for selected_row in rows:
        combined_started = time.perf_counter()
        materialization_started = time.perf_counter()
        decision_batch = dataset.batch(
            [int(selected_row)],
            arm=arm,
            transform_ids=[0],
        )
        materialization_latencies.append(
            time.perf_counter() - materialization_started
        )
        action_count = int(
            np.asarray(decision_batch.base.candidate_mask)[0].sum()
        )
        model_started = time.perf_counter()
        parent = model.encode_parent(decision_batch)
        outputs = []
        for start in range(0, action_count, candidate_chunk):
            outputs.append(
                model.predict(
                    decision_batch,
                    candidate_slice=slice(
                        start,
                        min(start + candidate_chunk, action_count),
                    ),
                    parent_state=parent,
                ).scores
            )
        mx.eval(parent, *outputs)
        model_latencies.append(time.perf_counter() - model_started)
        combined_latencies.append(time.perf_counter() - combined_started)
        decision_actions += action_count
    swap_after = _system_swap_used_bytes()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    materialization_array = np.asarray(
        materialization_latencies,
        dtype=np.float64,
    )
    model_array = np.asarray(model_latencies, dtype=np.float64)
    combined_array = np.asarray(combined_latencies, dtype=np.float64)
    combined_seconds = float(combined_array.sum())
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else swap_after - swap_before
    )
    return {
        "parent_encode": {
            "compile_seconds": parent_compile_seconds,
            "warmup_iterations": warmup_iterations,
            "steady_iterations": steady_iterations,
            "latency_milliseconds": _latencies(parent_latencies),
            "row": row,
            "r2_tokens": int(np.asarray(batch.parent.r2_token_mask).sum()),
            "relational_tokens": int(
                np.asarray(batch.parent.relational_mask).sum()
            ),
        },
        "fixed_chunk": {
            "actions": width,
            "compile_seconds": compile_seconds,
            "warmup_iterations": warmup_iterations,
            "warmup_seconds": warmup_seconds,
            "steady_iterations": steady_iterations,
            "steady_seconds": float(chunk_latencies.sum()),
            "action_scores_per_second": (
                width * steady_iterations
                / max(float(chunk_latencies.sum()), 1e-12)
            ),
            "latency_milliseconds": _latencies(chunk_latencies),
        },
        "materialization": {
            "groups": len(rows),
            "latency_milliseconds": _latencies(materialization_array),
            "latency_samples_milliseconds": (
                materialization_array * 1000
            ).tolist(),
        },
        "model_complete_decisions": {
            "groups": len(rows),
            "actions": decision_actions,
            "elapsed_seconds": float(model_array.sum()),
            "action_scores_per_second": (
                decision_actions / max(float(model_array.sum()), 1e-12)
            ),
            "latency_milliseconds": _latencies(model_array),
            "latency_samples_milliseconds": (
                model_array * 1000
            ).tolist(),
        },
        "combined_complete_decisions": {
            "groups": len(rows),
            "actions": decision_actions,
            "elapsed_seconds": combined_seconds,
            "action_scores_per_second": (
                decision_actions / max(combined_seconds, 1e-12)
            ),
            "latency_milliseconds": _latencies(combined_array),
            "latency_samples_milliseconds": (
                combined_array * 1000
            ).tolist(),
        },
        "memory": {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_active_bytes": int(mx.get_peak_memory()),
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": swap_delta,
        },
    }


def _selected_parent_statistics(
    dataset: RelationalSubstrateMlxDataset,
    arm: str,
    rows: np.ndarray,
) -> dict[str, Any]:
    r2_counts: list[int] = []
    relational_counts: list[int] = []
    class_counts = np.zeros(12, dtype=np.int64)
    for row in rows:
        parent = dataset.parent_batch([int(row)], arm=arm, transform_ids=[0])
        r2_mask = np.asarray(parent.r2_token_mask, dtype=np.bool_)
        r2_types = np.asarray(parent.r2_token_types, dtype=np.int64)
        relational_mask = np.asarray(
            parent.relational_mask,
            dtype=np.bool_,
        )
        relational_classes = np.asarray(
            parent.relational_classes,
            dtype=np.int64,
        )
        r2_counts.extend(r2_mask.sum(axis=2).reshape(-1).tolist())
        relational_counts.extend(
            relational_mask.sum(axis=2).reshape(-1).tolist()
        )
        for token_class in range(1, 9):
            class_counts[token_class - 1] += int(
                np.sum(
                    relational_mask
                    & (relational_classes == token_class)
                )
            )
        for token_type in range(1, 5):
            class_counts[token_type + 7] += int(
                np.sum(r2_mask & (r2_types == token_type))
            )
    return {
        "groups": len(rows),
        "r2_per_board": _distribution(np.asarray(r2_counts)),
        "relational_per_board": _distribution(
            np.asarray(relational_counts)
        ),
        "class_tokens": class_counts.tolist(),
    }


def _selected_derivative_statistics(
    dataset: RelationalSubstrateMlxDataset,
    arm: str,
    rows: np.ndarray,
) -> dict[str, Any]:
    candidates = 0
    nonzero = 0
    for row in rows:
        batch = dataset.batch([int(row)], arm=arm, transform_ids=[0])
        mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)
        values = np.asarray(batch.derivative_features)
        candidates += int(mask.sum())
        nonzero += int(np.count_nonzero(values[mask]))
    return {
        "enabled": arm == S5_ARM,
        "features": int(
            dataset.cache.splits[dataset.split].tensors["s5_values"].shape[1]
        ),
        "candidates": candidates,
        "nonzero_values": nonzero,
    }


def _batch_inputs(batch: object) -> tuple[mx.array, ...]:
    base = batch.base
    parent = batch.parent
    return (
        parent.r2_token_features,
        parent.r2_token_types,
        parent.r2_token_mask,
        parent.relational_values,
        parent.relational_classes,
        parent.relational_mask,
        parent.market_features,
        parent.market_mask,
        parent.player_features,
        parent.player_mask,
        parent.global_features,
        parent.transform_ids,
        batch.candidate_token_features,
        batch.candidate_token_mask,
        base.action_features,
        base.prior_features,
        base.staged_market_entities,
        base.staged_market_mask,
        base.candidate_mask,
        base.screen_value,
        batch.supply_vector,
        batch.staged_supply_vector,
        batch.selected_archetype,
        batch.frontier_features,
        batch.derivative_features,
    )


def _parent_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        r2_token_features=values[0],
        r2_token_types=values[1],
        r2_token_mask=values[2],
        relational_values=values[3],
        relational_classes=values[4],
        relational_mask=values[5],
        market_features=values[6],
        market_mask=values[7],
        player_features=values[8],
        player_mask=values[9],
        global_features=values[10],
        transform_ids=values[11],
    )


def _model_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    base = SimpleNamespace(
        action_features=values[14],
        prior_features=values[15],
        staged_market_entities=values[16],
        staged_market_mask=values[17],
        candidate_mask=values[18],
        screen_value=values[19],
    )
    return SimpleNamespace(
        parent=_parent_batch(values[:12]),
        base=base,
        candidate_token_features=values[12],
        candidate_token_mask=values[13],
        supply_vector=values[20],
        staged_supply_vector=values[21],
        selected_archetype=values[22],
        frontier_features=values[23],
        derivative_features=values[24],
    )


def _latencies(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50) * 1000),
        "p95": float(np.quantile(values, 0.95) * 1000),
        "p99": float(np.quantile(values, 0.99) * 1000),
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "count": len(numeric),
        "minimum": int(numeric.min()),
        "mean": float(numeric.mean()),
        "p50": float(np.quantile(numeric, 0.50)),
        "p90": float(np.quantile(numeric, 0.90)),
        "p99": float(np.quantile(numeric, 0.99)),
        "maximum": int(numeric.max()),
    }


__all__ = [
    "CANDIDATE_CHUNK",
    "benchmark_relational_substrate",
    "evaluate_relational_substrate",
]
