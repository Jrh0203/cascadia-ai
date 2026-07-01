"""Quality and serving measurements for S4 candidate-context models."""

from __future__ import annotations

import platform
import re
import resource
import subprocess
import time
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_metrics import evaluate_r3_action_edit
from cascadia_mlx.s4_candidate_set_mlx_data import S4CandidateSetDataset
from cascadia_mlx.s4_candidate_set_mlx_model import S4CandidateSetRanker

CANDIDATE_CHUNK = 256
_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


def evaluate_s4_candidate_set(
    model: S4CandidateSetRanker,
    dataset: S4CandidateSetDataset,
    *,
    arm: str,
    rows: np.ndarray | None = None,
    candidate_chunk: int = CANDIDATE_CHUNK,
    prediction_panel_size: int = 64,
) -> dict[str, Any]:
    """Evaluate every requested complete action through shared S4 context."""
    return evaluate_r3_action_edit(
        model,
        dataset,
        arm=arm,
        rows=rows,
        candidate_chunk=candidate_chunk,
        prediction_panel_size=prediction_panel_size,
    )


def benchmark_s4_candidate_set(
    model: S4CandidateSetRanker,
    dataset: S4CandidateSetDataset,
    *,
    arm: str,
    decision_rows: np.ndarray,
    candidate_chunk: int = CANDIDATE_CHUNK,
    warmup_iterations: int = 5,
    steady_iterations: int = 10,
) -> dict[str, Any]:
    """Measure anchor preparation and complete-decision MLX serving latency."""
    rows = np.asarray(decision_rows, dtype=np.int64)
    if (
        candidate_chunk <= 0
        or warmup_iterations <= 0
        or steady_iterations <= 0
        or rows.ndim != 1
        or not len(rows)
        or len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= dataset.group_count)
    ):
        raise ValueError("S4 serving benchmark dimensions are invalid")

    model.eval()
    mx.clear_cache()
    mx.reset_peak_memory()
    compile_batch = dataset.batch(
        [int(rows[0])],
        arm=arm,
        transform_ids=[0],
    )
    compile_count = int(
        np.asarray(compile_batch.base.candidate_mask, dtype=np.bool_).sum()
    )
    compile_started = time.perf_counter()
    compile_prepared = model.prepare_context(
        compile_batch.r3,
        compile_batch.context,
    )
    compile_prediction = model.predict(
        compile_batch.r3,
        compile_batch.context,
        candidate_slice=slice(0, min(candidate_chunk, compile_count)),
        prepared_context=compile_prepared,
    )
    mx.eval(
        compile_prepared.parent_state,
        compile_prepared.anchor_hidden,
        compile_prepared.inducing_latents,
        compile_prediction.scores,
    )
    compile_seconds = time.perf_counter() - compile_started

    for _ in range(warmup_iterations):
        prediction = model.predict(
            compile_batch.r3,
            compile_batch.context,
            candidate_slice=slice(0, min(candidate_chunk, compile_count)),
            prepared_context=compile_prepared,
        )
        mx.eval(prediction.scores, prediction.standard_errors)

    fixed_latencies = np.empty(steady_iterations, dtype=np.float64)
    for iteration in range(steady_iterations):
        started = time.perf_counter()
        prediction = model.predict(
            compile_batch.r3,
            compile_batch.context,
            candidate_slice=slice(0, min(candidate_chunk, compile_count)),
            prepared_context=compile_prepared,
        )
        mx.eval(prediction.scores, prediction.standard_errors)
        fixed_latencies[iteration] = time.perf_counter() - started

    preparation_latencies: list[float] = []
    scoring_latencies: list[float] = []
    decision_latencies: list[float] = []
    decision_actions = 0
    swap_before = _system_swap_used_bytes()
    for row in rows:
        batch = dataset.batch(
            [int(row)],
            arm=arm,
            transform_ids=[0],
        )
        count = int(
            np.asarray(batch.base.candidate_mask, dtype=np.bool_).sum()
        )
        decision_started = time.perf_counter()
        prepare_started = time.perf_counter()
        prepared = model.prepare_context(batch.r3, batch.context)
        mx.eval(
            prepared.parent_state,
            prepared.anchor_hidden,
            prepared.inducing_latents,
        )
        preparation_latencies.append(time.perf_counter() - prepare_started)
        scoring_started = time.perf_counter()
        outputs = []
        for start in range(0, count, candidate_chunk):
            prediction = model.predict(
                batch.r3,
                batch.context,
                candidate_slice=slice(
                    start,
                    min(start + candidate_chunk, count),
                ),
                prepared_context=prepared,
            )
            outputs.extend((prediction.scores, prediction.standard_errors))
        mx.eval(*outputs)
        scoring_latencies.append(time.perf_counter() - scoring_started)
        decision_latencies.append(time.perf_counter() - decision_started)
        decision_actions += count
    swap_after = _system_swap_used_bytes()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    fixed_actions = min(candidate_chunk, compile_count)
    fixed_seconds = float(fixed_latencies.sum())
    decision = np.asarray(decision_latencies, dtype=np.float64)
    preparation = np.asarray(preparation_latencies, dtype=np.float64)
    scoring = np.asarray(scoring_latencies, dtype=np.float64)
    decision_seconds = float(decision.sum())
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else swap_after - swap_before
    )
    return {
        "fixed_chunk": {
            "actions": fixed_actions,
            "compile_seconds": compile_seconds,
            "warmup_iterations": warmup_iterations,
            "steady_iterations": steady_iterations,
            "steady_seconds": fixed_seconds,
            "action_scores_per_second": (
                fixed_actions * steady_iterations / max(fixed_seconds, 1e-12)
            ),
            "latency_milliseconds": _latency_report(fixed_latencies),
        },
        "complete_decisions": {
            "groups": len(rows),
            "actions": decision_actions,
            "parent_encodes": len(rows),
            "anchor_encodes": len(rows),
            "parent_encode_count_exact": True,
            "anchor_encode_count_exact": True,
            "elapsed_seconds": decision_seconds,
            "action_scores_per_second": (
                decision_actions / max(decision_seconds, 1e-12)
            ),
            "latency_milliseconds": _latency_report(decision),
            "prepare_latency_milliseconds": _latency_report(preparation),
            "score_latency_milliseconds": _latency_report(scoring),
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


def _latency_report(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50) * 1000),
        "p95": float(np.quantile(values, 0.95) * 1000),
        "p99": float(np.quantile(values, 0.99) * 1000),
    }


def _system_swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = _SWAP_USED_RE.search(output)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)
