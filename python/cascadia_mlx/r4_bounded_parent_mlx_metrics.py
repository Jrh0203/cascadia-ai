"""Quality and serving evidence for the ADR 0156 parent comparison."""

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
from cascadia_mlx.r4_bounded_parent_mlx_cache import R4BoundedParentMlxDataset
from cascadia_mlx.r4_bounded_parent_mlx_model import R4BoundedParentRanker


def evaluate_r4_bounded_parent(
    model: object,
    dataset: R4BoundedParentMlxDataset,
    *,
    arm: str,
    rows: np.ndarray | None = None,
    candidate_chunk: int = CANDIDATE_CHUNK,
    prediction_panel_size: int = 64,
) -> dict[str, Any]:
    metrics = evaluate_r3_action_edit(
        model,
        dataset,
        arm=arm,
        rows=rows,
        candidate_chunk=candidate_chunk,
        prediction_panel_size=prediction_panel_size,
    )
    if rows is None:
        metrics["parent_tokens"] = dataset.parent_token_statistics(arm)
    else:
        counts = []
        class_counts = np.zeros(9, dtype=np.int64)
        for row in np.asarray(rows, dtype=np.int64):
            batch = dataset.batch([int(row)], arm=arm, transform_ids=[0])
            mask = np.asarray(batch.parent.token_mask, dtype=np.bool_)
            classes = np.asarray(batch.parent.token_classes, dtype=np.int64)
            counts.extend(mask.sum(axis=2).reshape(-1).tolist())
            for token_class in range(1, 10):
                class_counts[token_class - 1] += int(np.sum(mask & (classes == token_class)))
        metrics["parent_tokens"] = {
            "groups": len(rows),
            "per_board_tokens": _distribution(np.asarray(counts)),
            "class_tokens": class_counts.tolist(),
        }
    return metrics


def benchmark_r4_bounded_parent(
    model: R4BoundedParentRanker,
    dataset: R4BoundedParentMlxDataset,
    *,
    arm: str,
    row: int = 0,
    candidate_chunk: int = CANDIDATE_CHUNK,
    warmup_iterations: int = 5,
    steady_iterations: int = 30,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure parent-only, fixed-chunk, and complete-decision serving cost."""
    if candidate_chunk <= 0 or warmup_iterations <= 0 or steady_iterations <= 0:
        raise ValueError("R4 parent benchmark dimensions must be positive")
    batch = dataset.batch([row], arm=arm, transform_ids=[0])
    count = int(np.asarray(batch.base.candidate_mask)[0].sum())
    width = min(candidate_chunk, count)
    inputs = _batch_inputs(batch)
    parent_inputs = inputs[:8]

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
    fixed_parent_latencies = np.empty(steady_iterations, dtype=np.float64)
    for iteration in range(steady_iterations):
        started = time.perf_counter()
        parent_output = compiled_parent(*parent_inputs)
        mx.eval(parent_output)
        fixed_parent_latencies[iteration] = time.perf_counter() - started

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
    steady_seconds = float(chunk_latencies.sum())

    rows = (
        np.arange(min(dataset.group_count, 20), dtype=np.int64)
        if decision_rows is None
        else np.asarray(decision_rows, dtype=np.int64)
    )
    decision_latencies: list[float] = []
    decision_actions = 0
    parent_encodes = 0
    swap_before = _system_swap_used_bytes()
    for selected_row in rows:
        decision_batch = dataset.batch(
            [int(selected_row)],
            arm=arm,
            transform_ids=[0],
        )
        action_count = int(np.asarray(decision_batch.base.candidate_mask)[0].sum())
        started = time.perf_counter()
        parent = model.encode_parent(decision_batch)
        parent_encodes += 1
        outputs = []
        for chunk_start in range(0, action_count, candidate_chunk):
            prediction = model.predict(
                decision_batch,
                candidate_slice=slice(
                    chunk_start,
                    min(chunk_start + candidate_chunk, action_count),
                ),
                parent_state=parent,
            )
            outputs.append(prediction.scores)
        mx.eval(parent, *outputs)
        decision_latencies.append(time.perf_counter() - started)
        decision_actions += action_count

    representative_parent_latencies: list[float] = []
    representative_parent_tokens: list[int] = []
    for selected_row in rows:
        parent_batch = dataset.parent_batch(
            [int(selected_row)],
            arm=arm,
            transform_ids=[0],
        )
        representative_parent_tokens.append(int(np.asarray(parent_batch.token_mask).sum()))
        started = time.perf_counter()
        parent = model.parent_encoder(parent_batch)
        mx.eval(parent)
        representative_parent_latencies.append(time.perf_counter() - started)
    swap_after = _system_swap_used_bytes()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    decision_array = np.asarray(decision_latencies, dtype=np.float64)
    parent_array = np.asarray(representative_parent_latencies, dtype=np.float64)
    decision_seconds = float(decision_array.sum())
    swap_delta = None if swap_before is None or swap_after is None else swap_after - swap_before
    return {
        "parent_encode": {
            "groups": len(rows),
            "latency_milliseconds": _latencies(parent_array),
            "tokens": _distribution(np.asarray(representative_parent_tokens)),
            "measurement": "steady-per-decision-after-complete-pass",
        },
        "fixed_parent_encode": {
            "compile_seconds": parent_compile_seconds,
            "warmup_iterations": warmup_iterations,
            "steady_iterations": steady_iterations,
            "latency_milliseconds": _latencies(fixed_parent_latencies),
            "tokens": int(np.asarray(batch.parent.token_mask).sum()),
            "row": row,
        },
        "fixed_chunk": {
            "actions": width,
            "compile_seconds": compile_seconds,
            "warmup_iterations": warmup_iterations,
            "warmup_seconds": warmup_seconds,
            "steady_iterations": steady_iterations,
            "steady_seconds": steady_seconds,
            "action_scores_per_second": width * steady_iterations / max(steady_seconds, 1e-12),
            "latency_milliseconds": _latencies(chunk_latencies),
        },
        "complete_decisions": {
            "groups": len(rows),
            "actions": decision_actions,
            "parent_encodes": parent_encodes,
            "parent_encode_count_exact": parent_encodes == len(rows),
            "elapsed_seconds": decision_seconds,
            "action_scores_per_second": decision_actions / max(decision_seconds, 1e-12),
            "latency_milliseconds": _latencies(decision_array),
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


def _batch_inputs(batch: object) -> tuple[mx.array, ...]:
    base = batch.base
    parent = batch.parent
    return (
        parent.token_values,
        parent.token_classes,
        parent.token_mask,
        parent.market_features,
        parent.market_mask,
        parent.player_features,
        parent.player_mask,
        parent.global_features,
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
    )


def _parent_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        token_values=values[0],
        token_classes=values[1],
        token_mask=values[2],
        market_features=values[3],
        market_mask=values[4],
        player_features=values[5],
        player_mask=values[6],
        global_features=values[7],
    )


def _model_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    base = SimpleNamespace(
        action_features=values[10],
        prior_features=values[11],
        staged_market_entities=values[12],
        staged_market_mask=values[13],
        candidate_mask=values[14],
        screen_value=values[15],
    )
    return SimpleNamespace(
        parent=_parent_batch(values),
        base=base,
        candidate_token_features=values[8],
        candidate_token_mask=values[9],
        supply_vector=values[16],
        staged_supply_vector=values[17],
        selected_archetype=values[18],
        frontier_features=values[19],
    )


def _latencies(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50) * 1000),
        "p95": float(np.quantile(values, 0.95) * 1000),
        "p99": float(np.quantile(values, 0.99) * 1000),
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": int(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": int(values.max()),
    }
