"""Frozen offline metrics and performance gates for ADR 0081."""

from __future__ import annotations

import platform
import re
import resource
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_UNCERTAINTY_FLOOR,
    GradedOracleRanker,
    graded_oracle_loss,
    predict_graded_oracle_batch,
)

_RECALL_WIDTHS = (1, 8, 32, 64)
_PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}
_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


@dataclass
class _SliceAccumulator:
    groups: int = 0
    top64: int = 0
    regret: float = 0.0

    def add(self, recalled: bool, regret: float) -> None:
        self.groups += 1
        self.top64 += int(recalled)
        self.regret += regret

    def report(self) -> dict[str, float | int]:
        return {
            "groups": self.groups,
            "top64_r4800_winner_recall": self.top64 / max(self.groups, 1),
            "mean_top64_retained_r4800_regret": self.regret / max(self.groups, 1),
        }


def evaluate_graded_oracle(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    group_batch_size: int,
    *,
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT,
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
) -> dict[str, Any]:
    """Evaluate every legal action with stable score/hash ordering."""
    model.eval()
    groups = 0
    candidates = 0
    nonfinite_scores = 0
    total_loss = 0.0
    selected_rank_sum = 0.0
    recall = {width: 0 for width in _RECALL_WIDTHS}
    regret = {width: 0.0 for width in _RECALL_WIDTHS}
    baseline_recall64 = 0
    baseline_regret64 = 0.0
    r1200_prediction: list[np.ndarray] = []
    r1200_target: list[np.ndarray] = []
    r4800_prediction: list[np.ndarray] = []
    r4800_target: list[np.ndarray] = []
    predicted_standard_error: list[np.ndarray] = []
    teacher_standard_error: list[np.ndarray] = []
    scored_error: list[np.ndarray] = []
    screen_only_residual: list[np.ndarray] = []
    phases = {name: _SliceAccumulator() for name in _PHASE_NAMES.values()}
    subsets = {
        "nature_token_available": _SliceAccumulator(),
        "independent_draft_winner": _SliceAccumulator(),
    }
    families = {
        "paired": _SliceAccumulator(),
        "independent": _SliceAccumulator(),
        "same_slot_independent": _SliceAccumulator(),
        "free_refresh": _SliceAccumulator(),
        "paid_wipe": _SliceAccumulator(),
    }

    for batch in dataset.batches(
        group_batch_size,
        maximum_actions_per_batch=maximum_actions_per_batch,
        maximum_group_actions=maximum_group_actions,
    ):
        prediction = predict_graded_oracle_batch(model, batch)
        loss = graded_oracle_loss(model, batch)
        mx.eval(
            prediction.scores,
            prediction.residuals,
            prediction.standard_errors,
            loss,
        )
        score_values = np.asarray(prediction.scores)
        residual_values = np.asarray(prediction.residuals)
        standard_error_values = np.asarray(prediction.standard_errors)
        masks = np.asarray(batch.candidate_mask)
        screen_values = np.asarray(batch.screen_value)
        selected_values = np.asarray(batch.selected_index)
        action_hashes = np.asarray(batch.action_hash)
        r600_values = np.asarray(batch.r600_mean)
        r600_stddev = np.asarray(batch.r600_stddev)
        r600_samples = np.asarray(batch.r600_samples)
        r600_masks = np.asarray(batch.r600_mask)
        r1200_values = np.asarray(batch.r1200_mean)
        r1200_stddev = np.asarray(batch.r1200_stddev)
        r1200_samples = np.asarray(batch.r1200_samples)
        r1200_masks = np.asarray(batch.r1200_mask)
        r4800_values = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_masks = np.asarray(batch.r4800_mask)
        phase_values = np.asarray(batch.phase)
        active_tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        same_slot_independent = np.asarray(batch.same_slot_independent)
        free_refresh = np.asarray(batch.replace_three_of_a_kind)
        wipe_count = np.asarray(batch.wipe_count)
        total_loss += float(loss.item()) * len(score_values)

        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            scores = score_values[group_index, :count]
            residuals = residual_values[group_index, :count]
            standard_errors = standard_error_values[group_index, :count]
            screen = screen_values[group_index, :count]
            hashes = action_hashes[group_index, :count]
            selected = int(selected_values[group_index])
            r600 = r600_values[group_index, :count]
            r600_seeds = r600_samples[group_index, :count]
            r600_mask = r600_masks[group_index, :count]
            r1200 = r1200_values[group_index, :count]
            r1200_seeds = r1200_samples[group_index, :count]
            r1200_mask = r1200_masks[group_index, :count]
            r4800 = r4800_values[group_index, :count]
            r4800_seeds = r4800_samples[group_index, :count]
            r4800_mask = r4800_masks[group_index, :count]

            finite = np.isfinite(scores)
            nonfinite_scores += int(np.sum(~finite))
            ranking = _stable_ranking(scores, hashes)
            baseline_ranking = _stable_ranking(screen, hashes)
            selected_rank = int(np.flatnonzero(ranking == selected)[0]) + 1
            selected_rank_sum += selected_rank
            group_regret64 = 0.0
            for width in _RECALL_WIDTHS:
                retained = ranking[: min(width, count)]
                recall[width] += int(np.any(retained == selected))
                retained_regret = _retained_regret(retained, r4800, r4800_mask)
                regret[width] += retained_regret
                if width == 64:
                    group_regret64 = retained_regret
            baseline_retained = baseline_ranking[: min(64, count)]
            baseline_recall64 += int(np.any(baseline_retained == selected))
            baseline_regret64 += _retained_regret(
                baseline_retained,
                r4800,
                r4800_mask,
            )

            if np.any(r1200_mask):
                r1200_prediction.append(scores[r1200_mask])
                r1200_target.append(r1200[r1200_mask])
            if np.any(r4800_mask):
                r4800_prediction.append(scores[r4800_mask])
                r4800_target.append(r4800[r4800_mask])

            highest_mean = np.where(
                r4800_mask,
                r4800,
                np.where(r1200_mask, r1200, r600),
            )
            highest_stddev = np.where(
                r4800_mask,
                r4800_stddev[group_index, :count],
                np.where(
                    r1200_mask,
                    r1200_stddev[group_index, :count],
                    r600_stddev[group_index, :count],
                ),
            )
            highest_samples = np.where(
                r4800_mask,
                r4800_seeds,
                np.where(r1200_mask, r1200_seeds, r600_seeds),
            )
            scored_mask = r4800_mask | r1200_mask | r600_mask
            if np.any(scored_mask):
                predicted_standard_error.append(standard_errors[scored_mask])
                teacher_standard_error.append(
                    np.sqrt(
                        highest_stddev[scored_mask] ** 2
                        / np.maximum(highest_samples[scored_mask], 1.0)
                        + GRADED_ORACLE_UNCERTAINTY_FLOOR**2
                    )
                )
                scored_error.append(scores[scored_mask] - highest_mean[scored_mask])
            screen_only_mask = ~scored_mask
            if np.any(screen_only_mask):
                screen_only_residual.append(residuals[screen_only_mask])

            recalled64 = selected_rank <= min(64, count)
            phase_name = _PHASE_NAMES.get(int(phase_values[group_index]))
            if phase_name is None:
                raise ValueError("graded-oracle group has an invalid phase")
            phases[phase_name].add(recalled64, group_regret64)
            if int(active_tokens[group_index]) > 0:
                subsets["nature_token_available"].add(recalled64, group_regret64)

            winner_draft = int(draft_kind[group_index, selected])
            winner_same_slot = int(same_slot_independent[group_index, selected]) != 0
            winner_refresh = int(free_refresh[group_index, selected]) != 0
            winner_wipes = int(wipe_count[group_index, selected]) > 0
            if winner_draft == 1:
                subsets["independent_draft_winner"].add(recalled64, group_regret64)
                families["independent"].add(recalled64, group_regret64)
            else:
                families["paired"].add(recalled64, group_regret64)
            if winner_same_slot:
                families["same_slot_independent"].add(recalled64, group_regret64)
            if winner_refresh:
                families["free_refresh"].add(recalled64, group_regret64)
            if winner_wipes:
                families["paid_wipe"].add(recalled64, group_regret64)

            groups += 1
            candidates += count

    if groups == 0:
        raise ValueError("graded-oracle evaluation dataset is empty")
    r1200_pred = _concatenate(r1200_prediction)
    r1200_true = _concatenate(r1200_target)
    r4800_pred = _concatenate(r4800_prediction)
    r4800_true = _concatenate(r4800_target)
    predicted_se = _concatenate(predicted_standard_error)
    teacher_se = _concatenate(teacher_standard_error)
    score_error = _concatenate(scored_error)
    screen_residual = _concatenate(screen_only_residual)

    metrics: dict[str, Any] = {
        "groups": groups,
        "candidates": candidates,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_scored_once": groups == dataset.group_count,
        "all_candidates_scored_once": candidates == dataset.candidate_count,
        "nonfinite_scores": nonfinite_scores,
        "all_scores_finite": nonfinite_scores == 0,
        "training_objective": total_loss / groups,
        "mean_selected_rank": selected_rank_sum / groups,
        "screen_top64_r4800_winner_recall": baseline_recall64 / groups,
        "screen_mean_top64_retained_r4800_regret": baseline_regret64 / groups,
        "r1200_residual_mae": _mae(r1200_pred, r1200_true),
        "r1200_correlation": _correlation(r1200_pred, r1200_true),
        "r4800_residual_mae": _mae(r4800_pred, r4800_true),
        "r4800_correlation": _correlation(r4800_pred, r4800_true),
        "uncertainty": _uncertainty_report(predicted_se, teacher_se, score_error),
        "screen_only_residual": _distribution_report(screen_residual),
        "phase": {name: values.report() for name, values in phases.items()},
        "subsets": {name: values.report() for name, values in subsets.items()},
        "action_family": {name: values.report() for name, values in families.items()},
    }
    for width in _RECALL_WIDTHS:
        metrics[f"top{width}_r4800_winner_recall"] = recall[width] / groups
        metrics[f"mean_top{width}_retained_r4800_regret"] = regret[width] / groups
    return metrics


def benchmark_graded_oracle(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    *,
    maximum_groups: int | None = None,
    warmup_groups: int = 3,
) -> dict[str, Any]:
    """Measure warmed model-only complete-decision scoring on one Mac."""
    model.eval()
    batches = dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    )
    warmed: list[object] = []
    for batch in batches:
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        warmed.append(batch)
        if len(warmed) >= warmup_groups:
            break
    if not warmed:
        raise ValueError("graded-oracle benchmark dataset is empty")

    swap_before = _system_swap_used_bytes()
    latencies: list[float] = []
    scored_actions = 0
    nonfinite_scores = 0
    measured = 0
    for batch in _chain(warmed, batches):
        if maximum_groups is not None and measured >= maximum_groups:
            break
        started = time.perf_counter()
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        elapsed = time.perf_counter() - started
        mask = np.asarray(batch.candidate_mask)[0]
        scores = np.asarray(prediction.scores)[0][mask]
        latencies.append(elapsed)
        scored_actions += len(scores)
        nonfinite_scores += int(np.sum(~np.isfinite(scores)))
        measured += 1
    swap_after = _system_swap_used_bytes()
    total_seconds = float(np.sum(latencies))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    process_swaps = int(getattr(usage, "ru_nswap", 0))
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else swap_after - swap_before
    )
    p99 = float(np.quantile(latencies, 0.99))
    report = {
        "groups": measured,
        "actions": scored_actions,
        "elapsed_seconds": total_seconds,
        "action_scores_per_second": scored_actions / max(total_seconds, 1e-9),
        "mean_decision_milliseconds": 1000.0 * total_seconds / measured,
        "p99_decision_milliseconds": 1000.0 * p99,
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": process_swaps,
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": swap_delta,
        "nonfinite_scores": nonfinite_scores,
    }
    report["gates"] = {
        "action_scores_per_second_at_least_20000": (
            report["action_scores_per_second"] >= 20_000.0
        ),
        "p99_decision_milliseconds_at_most_250": (
            report["p99_decision_milliseconds"] <= 250.0
        ),
        "peak_process_rss_at_most_4_gib": peak_rss <= 4 * 1024**3,
        "process_swaps_zero": process_swaps == 0,
        "system_swap_not_consumed": swap_delta is not None and swap_delta <= 0,
        "all_scores_finite": nonfinite_scores == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report


def graded_oracle_validation_gates(
    metrics: dict[str, Any],
    performance_by_host: dict[str, dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Apply every ADR 0081 validation threshold without relaxing strict bounds."""
    gates = {
        "top64_r4800_winner_recall_strictly_greater_than_0_98": (
            float(metrics["top64_r4800_winner_recall"]) > 0.98
        ),
        "mean_top64_retained_r4800_regret_strictly_less_than_0_15": (
            float(metrics["mean_top64_retained_r4800_regret"]) < 0.15
        ),
        "all_groups_scored_once": bool(metrics["all_groups_scored_once"]),
        "all_candidates_scored_once": bool(metrics["all_candidates_scored_once"]),
        "all_scores_finite": bool(metrics["all_scores_finite"]),
    }
    for name, values in metrics["phase"].items():
        gates[f"{name}_top64_recall_at_least_0_97"] = (
            float(values["top64_r4800_winner_recall"]) >= 0.97
        )
        gates[f"{name}_retained_regret_strictly_less_than_0_20"] = (
            float(values["mean_top64_retained_r4800_regret"]) < 0.20
        )
    for name in ("nature_token_available", "independent_draft_winner"):
        values = metrics["subsets"][name]
        if int(values["groups"]) >= 20:
            gates[f"{name}_top64_recall_at_least_0_95"] = (
                float(values["top64_r4800_winner_recall"]) >= 0.95
            )
            gates[f"{name}_retained_regret_strictly_less_than_0_25"] = (
                float(values["mean_top64_retained_r4800_regret"]) < 0.25
            )
    if performance_by_host is not None:
        for host, performance in performance_by_host.items():
            gates[f"{host}_performance_passed"] = bool(performance["passed"])
    return gates


def _stable_ranking(scores: np.ndarray, action_hashes: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), bytes(action_hashes[index])),
        ),
        dtype=np.int32,
    )


def _retained_regret(
    retained: np.ndarray,
    teacher_values: np.ndarray,
    teacher_mask: np.ndarray,
) -> float:
    labeled = teacher_values[teacher_mask]
    if len(labeled) == 0:
        raise ValueError("graded-oracle group has no R4800 labels")
    retained_labeled = retained[teacher_mask[retained]]
    if len(retained_labeled) == 0:
        return float(np.max(labeled) - np.min(labeled))
    return float(np.max(labeled) - np.max(teacher_values[retained_labeled]))


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values).astype(np.float64) if values else np.zeros(0)


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target))) if len(prediction) else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _distribution_report(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0, "mean": 0.0, "stddev": 0.0, "p95_abs": 0.0, "p99_abs": 0.0}
    absolute = np.abs(values)
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "stddev": float(np.std(values)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
    }


def _uncertainty_report(
    predicted: np.ndarray,
    teacher: np.ndarray,
    error: np.ndarray,
) -> dict[str, float | int]:
    if len(predicted) == 0:
        return {
            "count": 0,
            "mean_predicted_standard_error": 0.0,
            "mean_teacher_standard_error": 0.0,
            "standard_error_mae": 0.0,
            "standard_error_correlation": 0.0,
            "one_sigma_empirical_coverage": 0.0,
            "gaussian_cross_entropy": 0.0,
        }
    variance = np.maximum(predicted**2, 1e-8)
    gaussian = np.log(np.maximum(predicted, 1e-4)) + (
        teacher**2 + error**2
    ) / (2.0 * variance)
    return {
        "count": len(predicted),
        "mean_predicted_standard_error": float(np.mean(predicted)),
        "mean_teacher_standard_error": float(np.mean(teacher)),
        "standard_error_mae": float(np.mean(np.abs(predicted - teacher))),
        "standard_error_correlation": _correlation(predicted, teacher),
        "one_sigma_empirical_coverage": float(np.mean(np.abs(error) <= predicted)),
        "gaussian_cross_entropy": float(np.mean(gaussian)),
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


def _chain(first: Iterable[object], second: Iterable[object]) -> Iterable[object]:
    yield from first
    yield from second
