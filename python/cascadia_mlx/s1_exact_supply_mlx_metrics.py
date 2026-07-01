"""Offline quality, refill, and performance measurements for ADR 0147."""

from __future__ import annotations

import math
import platform
import re
import resource
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_model import GRADED_ORACLE_UNCERTAINTY_FLOOR
from cascadia_mlx.s1_exact_supply_mlx_model import (
    S1ExactSupplyRanker,
    s1_exact_supply_loss,
)

RECALL_WIDTHS = (1, 8, 32, 64)
NORMAL_95 = 1.959963984540054
LOW_SUPPLY_MAX_UNSEEN = 20
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


def evaluate_s1_exact_supply(
    model: S1ExactSupplyRanker,
    dataset: object,
    group_batch_size: int,
) -> dict[str, Any]:
    """Evaluate every validation action once and every refill law once."""
    model.eval()
    groups = 0
    candidates = 0
    total_loss = 0.0
    nonfinite_scores = 0
    recall = {width: 0 for width in RECALL_WIDTHS}
    regret = {width: 0.0 for width in RECALL_WIDTHS}
    confidence64 = 0
    r4800_predictions: list[np.ndarray] = []
    r4800_targets: list[np.ndarray] = []
    refill_total_variation: list[np.ndarray] = []
    refill_cross_entropy: list[np.ndarray] = []
    refill_probability_error: list[np.ndarray] = []
    refill_top1 = 0
    slices = {
        "low_supply": _Slice(),
        "independent_draft_winner": _Slice(),
    }

    for batch in dataset.batches(group_batch_size):
        prediction = model(batch)
        loss = s1_exact_supply_loss(model, batch)
        mx.eval(
            prediction.scores,
            prediction.standard_errors,
            prediction.refill_probabilities,
            loss,
        )
        scores = np.asarray(prediction.scores)
        refill = np.asarray(prediction.refill_probabilities)
        refill_target = np.asarray(batch.refill_target)
        masks = np.asarray(batch.candidate_mask)
        hashes = np.asarray(batch.action_hash)
        selected = np.asarray(batch.selected_index)
        r4800 = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        draft_kind = np.asarray(batch.draft_kind)
        turns = np.asarray(batch.turn)
        total_loss += float(loss.item()) * len(scores)

        tv = 0.5 * np.abs(refill - refill_target).sum(axis=1)
        refill_total_variation.append(tv)
        refill_cross_entropy.append(
            -np.sum(refill_target * np.log(np.maximum(refill, 1e-12)), axis=1)
        )
        refill_probability_error.append(np.abs(refill - refill_target).reshape(-1))
        refill_top1 += int(
            np.sum(np.argmax(refill, axis=1) == np.argmax(refill_target, axis=1))
        )

        for row, mask in enumerate(masks):
            count = int(mask.sum())
            group_scores = scores[row, :count]
            group_hashes = hashes[row, :count]
            group_r4800 = r4800[row, :count]
            group_r4800_mask = r4800_mask[row, :count]
            group_stddev = r4800_stddev[row, :count]
            group_samples = r4800_samples[row, :count]
            winner = int(selected[row])
            if winner >= count or not group_r4800_mask[winner]:
                raise ValueError("S1 validation winner lacks an R4800 label")
            ranking = _stable_ranking(group_scores, group_hashes)
            nonfinite_scores += int(np.sum(~np.isfinite(group_scores)))
            group_confidence = _confidence_set(
                group_r4800,
                group_stddev,
                group_samples,
                group_r4800_mask,
                winner,
            )
            retained64 = ranking[: min(64, count)]
            recalled64 = bool(np.any(retained64 == winner))
            covered64 = bool(np.any(group_confidence[retained64]))
            regret64 = _retained_regret(
                retained64,
                group_r4800,
                group_r4800_mask,
            )
            confidence64 += int(covered64)
            for width in RECALL_WIDTHS:
                retained = ranking[: min(width, count)]
                recall[width] += int(np.any(retained == winner))
                regret[width] += _retained_regret(
                    retained,
                    group_r4800,
                    group_r4800_mask,
                )
            if np.any(group_r4800_mask):
                r4800_predictions.append(group_scores[group_r4800_mask])
                r4800_targets.append(group_r4800[group_r4800_mask])
            unseen = 81 - int(turns[row])
            if unseen <= LOW_SUPPLY_MAX_UNSEEN:
                slices["low_supply"].add(
                    recalled=recalled64,
                    confidence=covered64,
                    regret=regret64,
                )
            if int(draft_kind[row, winner]) == 1:
                slices["independent_draft_winner"].add(
                    recalled=recalled64,
                    confidence=covered64,
                    regret=regret64,
                )
            groups += 1
            candidates += count

    if groups == 0:
        raise ValueError("S1 evaluation dataset is empty")
    predicted = _concatenate(r4800_predictions)
    target = _concatenate(r4800_targets)
    errors = predicted - target
    calibration = _calibration(predicted, target)
    refill_tv = _concatenate(refill_total_variation)
    refill_ce = _concatenate(refill_cross_entropy)
    refill_error = _concatenate(refill_probability_error)
    metrics: dict[str, Any] = {
        "groups": groups,
        "candidates": candidates,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_scored_once": groups == dataset.group_count,
        "all_candidates_scored_once": candidates == dataset.candidate_count,
        "training_objective": total_loss / groups,
        "nonfinite_scores": nonfinite_scores,
        "all_scores_finite": nonfinite_scores == 0,
        "r4800_value": {
            "count": len(errors),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "bias": float(np.mean(errors)),
            "correlation": _correlation(predicted, target),
            **calibration,
        },
        "top64_confidence_set_coverage_95": confidence64 / groups,
        "refill": {
            "groups": groups,
            "mean_total_variation": float(np.mean(refill_tv)),
            "p99_total_variation": float(np.quantile(refill_tv, 0.99)),
            "mean_cross_entropy": float(np.mean(refill_ce)),
            "mean_probability_mae": float(np.mean(refill_error)),
            "top1_mode_accuracy": refill_top1 / groups,
            "mean_fidelity": 1.0 - float(np.mean(refill_tv)),
            "all_probabilities_finite": bool(np.all(np.isfinite(refill_error))),
        },
        "subsets": {name: values.report() for name, values in slices.items()},
    }
    metrics["r4800_value_mae"] = metrics["r4800_value"]["mae"]
    metrics["refill_mean_total_variation"] = metrics["refill"]["mean_total_variation"]
    for width in RECALL_WIDTHS:
        metrics[f"top{width}_r4800_winner_recall"] = recall[width] / groups
        metrics[f"mean_top{width}_retained_r4800_regret"] = regret[width] / groups
    return metrics


def benchmark_s1_exact_supply(
    model: S1ExactSupplyRanker,
    dataset: object,
    *,
    maximum_groups: int | None = None,
    warmup_groups: int = 3,
) -> dict[str, Any]:
    """Measure model-only complete-action throughput and allocator memory."""
    model.eval()
    iterator = dataset.batches(1)
    warmed: list[object] = []
    for batch in iterator:
        prediction = model(batch)
        mx.eval(prediction.scores, prediction.refill_probabilities)
        warmed.append(batch)
        if len(warmed) >= warmup_groups:
            break
    if not warmed:
        raise ValueError("S1 performance dataset is empty")

    mx.clear_cache()
    mx.reset_peak_memory()
    swap_before = _system_swap_used_bytes()
    latencies: list[float] = []
    actions = 0
    measured = 0
    for batch in _chain(warmed, iterator):
        if maximum_groups is not None and measured >= maximum_groups:
            break
        started = time.perf_counter()
        prediction = model(batch)
        mx.eval(prediction.scores, prediction.refill_probabilities)
        latencies.append(time.perf_counter() - started)
        actions += int(np.asarray(batch.candidate_mask)[0].sum())
        measured += 1
    swap_after = _system_swap_used_bytes()
    elapsed = float(np.sum(latencies))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else swap_after - swap_before
    )
    report = {
        "groups": measured,
        "actions": actions,
        "elapsed_seconds": elapsed,
        "action_scores_per_second": actions / max(elapsed, 1e-12),
        "mean_decision_milliseconds": 1000.0 * elapsed / measured,
        "p99_decision_milliseconds": 1000.0 * float(np.quantile(latencies, 0.99)),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_delta_bytes": swap_delta,
    }
    report["absolute_gates"] = {
        "action_scores_per_second_at_least_20000": (
            report["action_scores_per_second"] >= 20_000.0
        ),
        "p99_decision_milliseconds_at_most_250": (
            report["p99_decision_milliseconds"] <= 250.0
        ),
        "peak_process_rss_at_most_4_gib": peak_rss <= 4 * 1024**3,
        "process_swaps_zero": report["process_swaps"] == 0,
        "system_swap_not_consumed": swap_delta is not None and swap_delta <= 0,
    }
    return report


def _confidence_set(
    means: np.ndarray,
    stddev: np.ndarray,
    samples: np.ndarray,
    mask: np.ndarray,
    winner: int,
) -> np.ndarray:
    standard_error = np.sqrt(
        np.square(stddev) / np.maximum(samples, 1.0)
        + GRADED_ORACLE_UNCERTAINTY_FLOOR**2
    )
    confidence = np.zeros(len(means), dtype=np.bool_)
    pairwise = np.sqrt(np.square(standard_error[winner]) + np.square(standard_error))
    confidence[mask] = (
        means[winner] - means[mask]
        <= NORMAL_95 * pairwise[mask]
    )
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
    if len(labeled) == 0:
        raise ValueError("S1 graded group has no R4800 labels")
    retained_labeled = retained[mask[retained]]
    if len(retained_labeled) == 0:
        return float(np.max(labeled) - np.min(labeled))
    return float(np.max(labeled) - np.max(teacher[retained_labeled]))


def _calibration(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(predicted) < 2 or float(np.var(predicted)) == 0.0:
        return {"calibration_slope": 0.0, "calibration_intercept": float(np.mean(target))}
    slope = float(np.cov(predicted, target, ddof=0)[0, 1] / np.var(predicted))
    intercept = float(np.mean(target) - slope * np.mean(predicted))
    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values).astype(np.float64) if values else np.zeros(0)


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


def _chain(first: list[object], second: Iterator[object]) -> Iterator[object]:
    yield from first
    yield from second
