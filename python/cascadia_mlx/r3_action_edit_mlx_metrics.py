"""Quality and performance evidence for the ADR 0150 matched comparison."""

from __future__ import annotations

import math
import platform
import re
import resource
import subprocess
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_model import GRADED_ORACLE_UNCERTAINTY_FLOOR
from cascadia_mlx.r3_action_edit_mlx_cache import (
    LOW_SUPPLY_MAX_UNSEEN,
    R3ActionEditMlxDataset,
)
from cascadia_mlx.r3_action_edit_mlx_model import R3ActionEditRanker

RECALL_WIDTHS = (1, 8, 32, 64)
NORMAL_95 = 1.959963984540054
CANDIDATE_CHUNK = 256
_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


@dataclass
class _Slice:
    groups: int = 0
    recall: int = 0
    confidence: int = 0
    regret: float = 0.0

    def add(self, *, recalled: bool, confidence: bool, regret: float) -> None:
        self.groups += 1
        self.recall += int(recalled)
        self.confidence += int(confidence)
        self.regret += regret

    def report(self) -> dict[str, float | int]:
        denominator = max(self.groups, 1)
        return {
            "groups": self.groups,
            "top64_r4800_winner_recall": self.recall / denominator,
            "top64_confidence_set_coverage_95": self.confidence / denominator,
            "mean_top64_retained_r4800_regret": self.regret / denominator,
        }


def evaluate_r3_action_edit(
    model: object,
    dataset: object,
    *,
    arm: str,
    rows: np.ndarray | None = None,
    candidate_chunk: int = CANDIDATE_CHUNK,
    prediction_panel_size: int = 64,
    row_subsets: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Score each selected decision and candidate exactly once in stable chunks."""
    if candidate_chunk <= 0 or prediction_panel_size <= 0:
        raise ValueError("R3 evaluation chunk and prediction panel must be positive")
    selected_rows = (
        np.arange(dataset.group_count, dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64)
    )
    if (
        selected_rows.ndim != 1
        or not len(selected_rows)
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= dataset.group_count)
        or len(np.unique(selected_rows)) != len(selected_rows)
    ):
        raise ValueError("R3 evaluation rows must be unique, nonempty, and in range")

    model.eval()
    groups = 0
    candidates = 0
    parent_encodes = 0
    nonfinite_scores = 0
    nonfinite_uncertainties = 0
    recall = {width: 0 for width in RECALL_WIDTHS}
    regret = {width: 0.0 for width in RECALL_WIDTHS}
    confidence64 = 0
    r4800_predictions: list[np.ndarray] = []
    r4800_targets: list[np.ndarray] = []
    token_counts: list[np.ndarray] = []
    padding_counts: list[int] = []
    panel_hashes: list[np.ndarray] = []
    panel_scores: list[np.ndarray] = []
    panel_uncertainties: list[np.ndarray] = []
    slices = {
        "early": _Slice(),
        "middle": _Slice(),
        "late": _Slice(),
        "low_supply": _Slice(),
        "independent_draft_winner": _Slice(),
    }
    subset_membership: dict[str, set[int]] = {}
    for name, subset_rows in (row_subsets or {}).items():
        values = np.asarray(subset_rows, dtype=np.int64)
        if (
            not name
            or name in slices
            or values.ndim != 1
            or np.any(values < 0)
            or np.any(values >= dataset.group_count)
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError("R3 evaluation row subset is malformed or duplicated")
        slices[name] = _Slice()
        subset_membership[name] = set(int(value) for value in values)

    for row in selected_rows:
        batch = dataset.batch(
            [int(row)],
            arm=arm,
            transform_ids=[0],
        )
        mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)[0]
        count = int(mask.sum())
        if count <= 0:
            raise ValueError("R3 validation group has no candidates")
        prepared = _prepare_prediction_context(model, batch)
        _eval_prediction_context(prepared)
        parent_encodes += 1
        score_chunks: list[np.ndarray] = []
        uncertainty_chunks: list[np.ndarray] = []
        for start in range(0, count, candidate_chunk):
            end = min(start + candidate_chunk, count)
            prediction = _predict_chunk(
                model,
                batch,
                slice(start, end),
                prepared,
            )
            mx.eval(prediction.scores, prediction.standard_errors)
            score_chunks.append(np.asarray(prediction.scores)[0])
            uncertainty_chunks.append(np.asarray(prediction.standard_errors)[0])
        scores = np.concatenate(score_chunks)
        uncertainties = np.concatenate(uncertainty_chunks)
        hashes = np.asarray(batch.base.action_hash)[0, :count]
        winner = int(np.asarray(batch.base.selected_index)[0])
        r4800 = np.asarray(batch.base.r4800_mean)[0, :count]
        r4800_stddev = np.asarray(batch.base.r4800_stddev)[0, :count]
        r4800_samples = np.asarray(batch.base.r4800_samples)[0, :count]
        r4800_mask = np.asarray(batch.base.r4800_mask)[0, :count]
        draft_kind = np.asarray(batch.base.draft_kind)[0, :count]
        turn = int(np.asarray(batch.base.turn)[0])
        counts = np.asarray(batch.candidate_token_counts)[0, :count]
        token_counts.append(counts)
        padding_counts.append(int(count * counts.max() - counts.sum()))

        if winner >= count or not r4800_mask[winner]:
            raise ValueError("R3 validation winner lacks an R4800 label")
        ranking = _stable_ranking(scores, hashes)
        confidence = _confidence_set(
            r4800,
            r4800_stddev,
            r4800_samples,
            r4800_mask,
            winner,
        )
        retained64 = ranking[: min(64, count)]
        recalled64 = bool(np.any(retained64 == winner))
        covered64 = bool(np.any(confidence[retained64]))
        regret64 = _retained_regret(retained64, r4800, r4800_mask)
        confidence64 += int(covered64)
        for width in RECALL_WIDTHS:
            retained = ranking[: min(width, count)]
            recall[width] += int(np.any(retained == winner))
            regret[width] += _retained_regret(
                retained,
                r4800,
                r4800_mask,
            )
        if np.any(r4800_mask):
            r4800_predictions.append(scores[r4800_mask])
            r4800_targets.append(r4800[r4800_mask])

        phase = "early" if turn < 27 else "middle" if turn < 54 else "late"
        slices[phase].add(
            recalled=recalled64,
            confidence=covered64,
            regret=regret64,
        )
        if 81 - turn <= LOW_SUPPLY_MAX_UNSEEN:
            slices["low_supply"].add(
                recalled=recalled64,
                confidence=covered64,
                regret=regret64,
            )
        if int(draft_kind[winner]) == 1:
            slices["independent_draft_winner"].add(
                recalled=recalled64,
                confidence=covered64,
                regret=regret64,
            )
        for name, members in subset_membership.items():
            if int(row) in members:
                slices[name].add(
                    recalled=recalled64,
                    confidence=covered64,
                    regret=regret64,
                )

        remaining_panel = prediction_panel_size - sum(len(values) for values in panel_scores)
        if remaining_panel > 0:
            take = min(remaining_panel, count)
            panel_hashes.append(hashes[:take].copy())
            panel_scores.append(scores[:take].astype("<f4"))
            panel_uncertainties.append(uncertainties[:take].astype("<f4"))
        nonfinite_scores += int(np.sum(~np.isfinite(scores)))
        nonfinite_uncertainties += int(np.sum(~np.isfinite(uncertainties) | (uncertainties <= 0)))
        groups += 1
        candidates += count

    predicted = np.concatenate(r4800_predictions).astype(np.float64)
    target = np.concatenate(r4800_targets).astype(np.float64)
    errors = predicted - target
    all_token_counts = np.concatenate(token_counts)
    panel_action_hashes = np.concatenate(panel_hashes, axis=0)
    panel_score_values = np.concatenate(panel_scores)
    panel_uncertainty_values = np.concatenate(panel_uncertainties)
    panel_digest = blake3.blake3()
    panel_digest.update(panel_action_hashes.tobytes(order="C"))
    panel_digest.update(panel_score_values.tobytes(order="C"))
    panel_digest.update(panel_uncertainty_values.tobytes(order="C"))
    complete = rows is None
    metrics: dict[str, Any] = {
        "groups": groups,
        "candidates": candidates,
        "expected_groups": (dataset.group_count if complete else len(selected_rows)),
        "expected_candidates": (dataset.candidate_count if complete else candidates),
        "all_groups_scored_once": groups
        == (dataset.group_count if complete else len(selected_rows)),
        "all_candidates_scored_once": (candidates == dataset.candidate_count if complete else True),
        "parent_encodes": parent_encodes,
        "parent_encode_count_exact": parent_encodes == groups,
        "nonfinite_scores": nonfinite_scores,
        "nonfinite_uncertainties": nonfinite_uncertainties,
        "all_scores_and_uncertainties_finite": (
            nonfinite_scores == 0 and nonfinite_uncertainties == 0
        ),
        "r4800_value": {
            "count": len(errors),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "bias": float(np.mean(errors)),
            "correlation": _correlation(predicted, target),
            **_calibration(predicted, target),
        },
        "top64_confidence_set_coverage_95": confidence64 / groups,
        "subsets": {name: values.report() for name, values in slices.items()},
        "candidate_tokens": {
            "count": len(all_token_counts),
            "minimum": int(np.min(all_token_counts)),
            "mean": float(np.mean(all_token_counts)),
            "p50": float(np.quantile(all_token_counts, 0.50)),
            "p90": float(np.quantile(all_token_counts, 0.90)),
            "p99": float(np.quantile(all_token_counts, 0.99)),
            "maximum": int(np.max(all_token_counts)),
            "padding_tokens": int(np.sum(padding_counts)),
        },
        "prediction_panel": {
            "count": len(panel_score_values),
            "action_hashes": [bytes(value).hex() for value in panel_action_hashes],
            "scores": panel_score_values.tolist(),
            "standard_errors": panel_uncertainty_values.tolist(),
            "panel_blake3": panel_digest.hexdigest(),
        },
    }
    for width in RECALL_WIDTHS:
        metrics[f"top{width}_r4800_winner_recall"] = recall[width] / groups
        metrics[f"mean_top{width}_retained_r4800_regret"] = regret[width] / groups
    return metrics


def _prepare_prediction_context(model: object, batch: object) -> object:
    if hasattr(batch, "context") and hasattr(batch, "r3"):
        return model.prepare_context(batch.r3, batch.context)
    return model.encode_parent(batch)


def _eval_prediction_context(prepared: object) -> None:
    if hasattr(prepared, "parent_state"):
        mx.eval(
            prepared.parent_state,
            prepared.anchor_hidden,
            prepared.anchor_mask,
            prepared.inducing_latents,
        )
    else:
        mx.eval(prepared)


def _predict_chunk(
    model: object,
    batch: object,
    selected: slice,
    prepared: object,
) -> object:
    if hasattr(batch, "context") and hasattr(batch, "r3"):
        return model.predict(
            batch.r3,
            batch.context,
            candidate_slice=selected,
            prepared_context=prepared,
        )
    return model.predict(
        batch,
        candidate_slice=selected,
        parent_state=prepared,
    )


def benchmark_r3_action_edit(
    model: R3ActionEditRanker,
    dataset: R3ActionEditMlxDataset,
    *,
    arm: str,
    row: int = 0,
    candidate_chunk: int = CANDIDATE_CHUNK,
    warmup_iterations: int = 5,
    steady_iterations: int = 30,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure compiled fixed-chunk throughput and complete-decision latency."""
    if candidate_chunk <= 0 or warmup_iterations <= 0 or steady_iterations <= 0:
        raise ValueError("R3 benchmark dimensions must be positive")
    batch = dataset.batch([row], arm=arm, transform_ids=[0])
    count = int(np.asarray(batch.base.candidate_mask)[0].sum())
    width = min(candidate_chunk, count)
    inputs = _batch_inputs(batch)

    def predict(*values: mx.array) -> mx.array:
        materialized = _model_batch(values)
        parent = model.encode_parent(materialized)
        return model.predict(
            materialized,
            candidate_slice=slice(0, width),
            parent_state=parent,
        ).scores

    compiled = mx.compile(predict, inputs=model.state)
    mx.clear_cache()
    mx.reset_peak_memory()
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
    swap_after = _system_swap_used_bytes()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    decision_array = np.asarray(decision_latencies, dtype=np.float64)
    decision_seconds = float(decision_array.sum())
    swap_delta = None if swap_before is None or swap_after is None else swap_after - swap_before
    return {
        "fixed_chunk": {
            "actions": width,
            "compile_seconds": compile_seconds,
            "warmup_iterations": warmup_iterations,
            "warmup_seconds": warmup_seconds,
            "steady_iterations": steady_iterations,
            "steady_seconds": steady_seconds,
            "action_scores_per_second": width * steady_iterations / max(steady_seconds, 1e-12),
            "latency_milliseconds": {
                "p50": float(np.quantile(chunk_latencies, 0.50) * 1000),
                "p95": float(np.quantile(chunk_latencies, 0.95) * 1000),
                "p99": float(np.quantile(chunk_latencies, 0.99) * 1000),
            },
        },
        "complete_decisions": {
            "groups": len(rows),
            "actions": decision_actions,
            "parent_encodes": parent_encodes,
            "parent_encode_count_exact": parent_encodes == len(rows),
            "elapsed_seconds": decision_seconds,
            "action_scores_per_second": decision_actions / max(decision_seconds, 1e-12),
            "latency_milliseconds": {
                "p50": float(np.quantile(decision_array, 0.50) * 1000),
                "p95": float(np.quantile(decision_array, 0.95) * 1000),
                "p99": float(np.quantile(decision_array, 0.99) * 1000),
            },
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
        parent.token_features,
        parent.token_types,
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


def _model_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    parent = SimpleNamespace(
        token_features=values[0],
        token_types=values[1],
        token_mask=values[2],
        market_features=values[3],
        market_mask=values[4],
        player_features=values[5],
        player_mask=values[6],
        global_features=values[7],
    )
    base = SimpleNamespace(
        action_features=values[10],
        prior_features=values[11],
        staged_market_entities=values[12],
        staged_market_mask=values[13],
        candidate_mask=values[14],
        screen_value=values[15],
    )
    return SimpleNamespace(
        parent=parent,
        base=base,
        candidate_token_features=values[8],
        candidate_token_mask=values[9],
        supply_vector=values[16],
        staged_supply_vector=values[17],
        selected_archetype=values[18],
        frontier_features=values[19],
    )


def _confidence_set(
    means: np.ndarray,
    stddev: np.ndarray,
    samples: np.ndarray,
    mask: np.ndarray,
    winner: int,
) -> np.ndarray:
    standard_error = np.sqrt(
        np.square(stddev) / np.maximum(samples, 1.0) + GRADED_ORACLE_UNCERTAINTY_FLOOR**2
    )
    confidence = np.zeros(len(means), dtype=np.bool_)
    pairwise = np.sqrt(np.square(standard_error[winner]) + np.square(standard_error))
    confidence[mask] = means[winner] - means[mask] <= NORMAL_95 * pairwise[mask]
    return confidence


def _stable_ranking(scores: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), bytes(hashes[index])),
        ),
        dtype=np.int32,
    )


def _retained_regret(
    retained: np.ndarray,
    teacher: np.ndarray,
    mask: np.ndarray,
) -> float:
    labeled = teacher[mask]
    if not len(labeled):
        raise ValueError("R3 graded group has no R4800 labels")
    retained_labeled = retained[mask[retained]]
    if not len(retained_labeled):
        return float(np.max(labeled) - np.min(labeled))
    return float(np.max(labeled) - np.max(teacher[retained_labeled]))


def _calibration(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    if len(predicted) < 2 or float(np.var(predicted)) == 0.0:
        return {
            "calibration_slope": 0.0,
            "calibration_intercept": float(np.mean(target)),
        }
    slope = float(np.cov(predicted, target, ddof=0)[0, 1] / np.var(predicted))
    return {
        "calibration_slope": slope,
        "calibration_intercept": float(np.mean(target) - slope * np.mean(predicted)),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


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
