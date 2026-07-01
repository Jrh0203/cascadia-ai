"""Shared MLX serving benchmark for complete Cascadia action decisions."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_metrics import (
    _system_swap_used_bytes,
)


@dataclass(frozen=True)
class CompleteDecisionBatchAdapter:
    """Make one cached batch reconstructible inside an MLX compiled graph."""

    inputs: Callable[[object], tuple[mx.array, ...]]
    parent_input_count: int
    parent_batch: Callable[[tuple[mx.array, ...]], object]
    model_batch: Callable[[tuple[mx.array, ...]], object]

    def validate(self) -> None:
        if self.parent_input_count <= 0:
            raise ValueError("parent input count must be positive")


def benchmark_complete_decisions(
    model: object,
    dataset: object,
    *,
    arm: str,
    adapter: CompleteDecisionBatchAdapter,
    row: int = 0,
    candidate_chunk: int,
    warmup_iterations: int = 5,
    steady_iterations: int = 30,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure compiled chunks and materialized complete decisions."""
    adapter.validate()
    if candidate_chunk <= 0 or warmup_iterations <= 0 or steady_iterations <= 0:
        raise ValueError("complete-decision benchmark dimensions must be positive")
    batch = dataset.batch([row], arm=arm, transform_ids=[0])
    count = int(np.asarray(batch.base.candidate_mask)[0].sum())
    width = min(candidate_chunk, count)
    inputs = adapter.inputs(batch)
    if len(inputs) <= adapter.parent_input_count:
        raise ValueError("compiled model inputs omit candidate tensors")
    parent_inputs = inputs[: adapter.parent_input_count]

    def encode_parent(*values: mx.array) -> mx.array:
        return model.parent_encoder(adapter.parent_batch(values))

    def predict(*values: mx.array) -> mx.array:
        materialized = adapter.model_batch(values)
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


def run_exact_r6_replay(
    binary: Path,
    *,
    dataset: Path,
    relational_cache: Path,
    rows: np.ndarray,
    expected_experiment_id: str,
    expected_protocol_id: str,
) -> dict[str, Any]:
    """Run and validate the exact sparse apply/undo replay sidecar."""
    if not binary.is_file():
        raise ValueError("R6 replay binary is absent")
    with tempfile.TemporaryDirectory(prefix="cascadia-r6-replay-") as raw:
        output = Path(raw) / "r6.json"
        completed = subprocess.run(
            [
                str(binary),
                "--dataset",
                str(dataset),
                "--relational-cache",
                str(relational_cache),
                "--rows",
                ",".join(str(int(row)) for row in rows),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"R6 replay failed: {detail}")
        try:
            report = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("R6 replay result is unreadable") from error
    identity = report.get("scientific_identity")
    if (
        report.get("experiment_id") != expected_experiment_id
        or report.get("protocol_id") != expected_protocol_id
        or not isinstance(identity, dict)
        or _canonical_blake3(identity) != report.get("report_id")
        or identity.get("rows") != [int(row) for row in rows]
        or identity.get("groups") != len(rows)
        or identity.get("exact_parity_pass") is not True
        or identity.get("apply_failures") != 0
        or identity.get("undo_failures") != 0
    ):
        raise ValueError("R6 replay result is malformed or inexact")
    return identity


def combine_complete_decisions_with_r6(
    performance: dict[str, Any],
    r6: dict[str, Any],
) -> dict[str, Any]:
    """Add aligned R6 latency samples to complete model decisions."""
    base = performance["combined_complete_decisions"]
    base_samples = np.asarray(
        base["latency_samples_milliseconds"],
        dtype=np.float64,
    )
    r6_samples = r6.get("samples")
    if (
        not isinstance(r6_samples, list)
        or len(r6_samples) != len(base_samples)
        or any(
            not isinstance(sample, dict)
            or sample.get("row") is None
            or sample.get("nanoseconds") is None
            for sample in r6_samples
        )
    ):
        raise ValueError(
            "R6 replay samples do not align with model decisions"
        )
    r6_milliseconds = np.asarray(
        [float(sample["nanoseconds"]) / 1_000_000 for sample in r6_samples],
        dtype=np.float64,
    )
    combined = base_samples + r6_milliseconds
    elapsed_seconds = float(combined.sum() / 1000)
    actions = int(base["actions"])
    return {
        "groups": len(combined),
        "actions": actions,
        "elapsed_seconds": elapsed_seconds,
        "action_scores_per_second": actions / max(elapsed_seconds, 1e-12),
        "latency_milliseconds": {
            "p50": float(np.quantile(combined, 0.50)),
            "p95": float(np.quantile(combined, 0.95)),
            "p99": float(np.quantile(combined, 0.99)),
        },
        "latency_samples_milliseconds": combined.tolist(),
        "r6_exact_parity_pass": True,
    }


def _latencies(values: np.ndarray) -> dict[str, float]:
    milliseconds = values * 1000
    return {
        "mean": float(np.mean(milliseconds)),
        "p50": float(np.quantile(milliseconds, 0.50)),
        "p95": float(np.quantile(milliseconds, 0.95)),
        "p99": float(np.quantile(milliseconds, 0.99)),
        "maximum": float(np.max(milliseconds)),
    }


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


__all__ = [
    "CompleteDecisionBatchAdapter",
    "benchmark_complete_decisions",
    "combine_complete_decisions_with_r6",
    "run_exact_r6_replay",
]
