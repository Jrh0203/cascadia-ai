"""Free-residual objective and optimizer diagnostics for ADR 0103."""

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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankBatch,
    build_expected_rank_target_mask,
    expected_rank_loss_from_scores,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
    Scale16ExpectedRankDataset,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    ERROR_CHECKPOINTS,
    LEARNING_RATE,
    SELECTED_MODEL_BLAKE3,
    WEIGHT_DECAY,
    _aggregate_metrics,
    _input_identity,
    _new_selected_model,
    _selected_checkpoint_spec,
    fit_model,
    load_cohort_batches,
    select_audit_cohort,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-frontier-free-residual-audit-v1"
SEED = 2026061631
ANALYTIC_GROUPS = 64
FREE_GROUPS = 24
NEURAL_GROUPS = 4
FREE_UPDATES = 1200
NEURAL_EXPOSURES = 1200
FREE_CHECKPOINTS = (0, 6, 24, 60, 120, 300, 600, 1200)
NEURAL_CHECKPOINTS = (120, 300, 600, 1200)
PROJECTED_INITIAL_STEP = 8.0
PROJECTED_KKT_TOLERANCE = 1e-9
PROJECTED_MAX_ITERATIONS = 10_000
ANALYTIC_KKT_GATE = 1e-8
OBJECTIVE_MATCH_GATE = 1e-7


def _closed_domains() -> dict[str, bool]:
    return {
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def _report(
    scientific: dict[str, Any],
    started: float,
    swap_before: int | None,
) -> dict[str, Any]:
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "telemetry": {
            "host": socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": (
                None
                if swap_before is None or swap_after is None
                else swap_after - swap_before
            ),
        },
    }


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / np.sum(weights)


def _target_probabilities(
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros(expected_rank.shape, dtype=np.float64)
    logits = -(expected_rank[expected_rank_mask].astype(np.float64) - 1.0) / TARGET_SCALE
    probabilities[expected_rank_mask] = _softmax(logits)
    return probabilities


def expected_rank_objective(
    scores: np.ndarray,
    probabilities: np.ndarray,
    eligible: np.ndarray,
) -> float:
    """Return the exact single-group deployed cross entropy."""
    logits = scores[eligible].astype(np.float64) / STUDENT_TEMPERATURE
    return _logsumexp(logits) - float(
        np.dot(probabilities[eligible], logits)
    )


def expected_rank_gradient(
    scores: np.ndarray,
    probabilities: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Return the exact score gradient for one group."""
    gradient = np.zeros(scores.shape, dtype=np.float64)
    gradient[eligible] = (
        _softmax(scores[eligible] / STUDENT_TEMPERATURE)
        - probabilities[eligible]
    ) / STUDENT_TEMPERATURE
    return gradient


def projected_kkt_violation(
    scores: np.ndarray,
    probabilities: np.ndarray,
    eligible: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> float:
    """Measure box-constrained first-order KKT violation in score space."""
    gradient = expected_rank_gradient(scores, probabilities, eligible)
    violations: list[np.ndarray] = []
    at_lower = eligible & (scores <= lower + tolerance)
    at_upper = eligible & (scores >= upper - tolerance)
    interior = eligible & ~at_lower & ~at_upper
    if np.any(at_lower):
        violations.append(np.maximum(0.0, -gradient[at_lower]))
    if np.any(at_upper):
        violations.append(np.maximum(0.0, gradient[at_upper]))
    if np.any(interior):
        violations.append(np.abs(gradient[interior]))
    return max(
        (float(np.max(values)) for values in violations if values.size),
        default=0.0,
    )


def solve_box_constrained_expected_rank(
    screen: np.ndarray,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    """Solve the exact convex expected-rank objective under residual bounds."""
    if not np.any(expected_rank_mask):
        raise ValueError("expected-rank group has no positive target mass")
    if np.any(expected_rank_mask & ~eligible):
        raise ValueError("expected-rank mass includes an ineligible action")
    residual_range = GRADED_ORACLE_RESIDUAL_RANGE
    lower = screen.astype(np.float64) - residual_range
    upper = screen.astype(np.float64) + residual_range
    probabilities = _target_probabilities(expected_rank, expected_rank_mask)
    base = np.zeros(screen.shape, dtype=np.float64)
    base[expected_rank_mask] = (
        STUDENT_TEMPERATURE * np.log(probabilities[expected_rank_mask])
    )

    def scores_for(offset: float) -> np.ndarray:
        scores = lower.copy()
        scores[expected_rank_mask] = np.clip(
            base[expected_rank_mask] + offset,
            lower[expected_rank_mask],
            upper[expected_rank_mask],
        )
        return scores

    def normalization_residual(offset: float) -> float:
        scores = scores_for(offset)
        return (
            _logsumexp(scores[eligible] / STUDENT_TEMPERATURE)
            - offset / STUDENT_TEMPERATURE
        )

    low = float(np.min(lower[expected_rank_mask] - base[expected_rank_mask])) - 64.0
    high = float(np.max(upper[expected_rank_mask] - base[expected_rank_mask])) + 64.0
    while normalization_residual(low) <= 0.0:
        low -= 64.0
    while normalization_residual(high) >= 0.0:
        high += 64.0
    for _ in range(256):
        midpoint = (low + high) / 2.0
        if normalization_residual(midpoint) > 0.0:
            low = midpoint
        else:
            high = midpoint
    offset = (low + high) / 2.0
    scores = scores_for(offset)
    kkt = projected_kkt_violation(
        scores,
        probabilities,
        eligible,
        lower,
        upper,
    )
    return {
        "scores": scores,
        "objective": expected_rank_objective(scores, probabilities, eligible),
        "kkt_violation": kkt,
        "normalization_offset": offset,
        "active_lower": int(np.sum(eligible & (scores <= lower + 1e-9))),
        "active_upper": int(np.sum(eligible & (scores >= upper - 1e-9))),
        "active_interior": int(
            np.sum(
                eligible
                & (scores > lower + 1e-9)
                & (scores < upper - 1e-9)
            )
        ),
    }


def projected_optimize_expected_rank(
    initial_scores: np.ndarray,
    screen: np.ndarray,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    eligible: np.ndarray,
    *,
    initial_step: float = PROJECTED_INITIAL_STEP,
    tolerance: float = PROJECTED_KKT_TOLERANCE,
    maximum_iterations: int = PROJECTED_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Independently solve the convex box problem by projected descent."""
    if initial_step <= 0.0 or tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("projected optimizer parameters must be positive")
    lower = screen.astype(np.float64) - GRADED_ORACLE_RESIDUAL_RANGE
    upper = screen.astype(np.float64) + GRADED_ORACLE_RESIDUAL_RANGE
    probabilities = _target_probabilities(expected_rank, expected_rank_mask)
    scores = np.clip(initial_scores.astype(np.float64), lower, upper)
    extrapolated = scores.copy()
    momentum = 1.0
    objective = expected_rank_objective(scores, probabilities, eligible)
    trajectory: list[dict[str, float | int]] = []
    next_record = 1
    converged = False
    iteration = 0
    for iteration in range(maximum_iterations + 1):
        violation = projected_kkt_violation(
            scores,
            probabilities,
            eligible,
            lower,
            upper,
        )
        if iteration == 0 or iteration == next_record or violation <= tolerance:
            trajectory.append(
                {
                    "iteration": iteration,
                    "objective": objective,
                    "kkt_violation": violation,
                }
            )
            next_record *= 2
        if violation <= tolerance:
            converged = True
            break
        gradient = expected_rank_gradient(
            extrapolated,
            probabilities,
            eligible,
        )
        step = initial_step
        accepted = False
        while step >= 2.0**-40:
            proposal = extrapolated.copy()
            proposal[eligible] = np.clip(
                extrapolated[eligible] - step * gradient[eligible],
                lower[eligible],
                upper[eligible],
            )
            proposal_objective = expected_rank_objective(
                proposal,
                probabilities,
                eligible,
            )
            if proposal_objective <= objective + 1e-15:
                previous = scores
                scores = proposal
                objective = proposal_objective
                next_momentum = (
                    1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
                ) / 2.0
                candidate_extrapolated = scores + (
                    (momentum - 1.0) / next_momentum
                ) * (scores - previous)
                candidate_extrapolated = np.clip(
                    candidate_extrapolated,
                    lower,
                    upper,
                )
                if (
                    expected_rank_objective(
                        candidate_extrapolated,
                        probabilities,
                        eligible,
                    )
                    <= objective + 1e-15
                ):
                    extrapolated = candidate_extrapolated
                    momentum = next_momentum
                else:
                    extrapolated = scores.copy()
                    momentum = 1.0
                accepted = True
                break
            step /= 2.0
        if not accepted:
            break
    final_violation = projected_kkt_violation(
        scores,
        probabilities,
        eligible,
        lower,
        upper,
    )
    if not trajectory or trajectory[-1]["iteration"] != iteration:
        trajectory.append(
            {
                "iteration": iteration,
                "objective": objective,
                "kkt_violation": final_violation,
            }
        )
    return {
        "scores": scores,
        "objective": objective,
        "kkt_violation": final_violation,
        "iterations": iteration,
        "converged": converged,
        "trajectory": trajectory,
    }


def _batch_arrays(
    batch: ExpectedRankBatch,
    *,
    model: nn.Module | None = None,
) -> dict[str, np.ndarray]:
    count = int(np.sum(np.asarray(batch.candidate_mask)[0]))
    screen = np.asarray(batch.screen_value)[0, :count].astype(np.float64)
    if model is None:
        residual = np.zeros(count, dtype=np.float64)
    else:
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.residuals)
        residual = np.asarray(prediction.residuals)[0, :count].astype(np.float64)
    flags = np.asarray(batch.source_flags)[0, :count]
    hashes = np.asarray(batch.action_hash)[0, :count]
    ranks = np.asarray(batch.expected_rank)[0, :count].astype(np.float64)
    rank_mask = np.asarray(batch.expected_rank_mask)[0, :count].astype(np.bool_)
    eligible = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    target = build_expected_rank_target_mask(
        expected_rank=ranks[None, :],
        expected_rank_mask=rank_mask[None, :],
        source_flags=flags[None, :],
        candidate_mask=np.ones((1, count), dtype=np.bool_),
        action_hashes=hashes[None, :],
    )[0]
    return {
        "screen": screen,
        "selected_residual": residual,
        "flags": flags,
        "hashes": hashes,
        "ranks": ranks,
        "rank_mask": rank_mask,
        "eligible": eligible,
        "target": target,
    }


def _score_metrics(
    batch: ExpectedRankBatch,
    scores: np.ndarray,
    *,
    objective: float,
) -> dict[str, Any]:
    arrays = _batch_arrays(batch)
    retained = frontier_anchored_retained_indices(
        scores=scores,
        source_flags=arrays["flags"],
        action_hashes=arrays["hashes"],
    )
    nonfrontier = retained[
        (arrays["flags"][retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    ]
    target_slots = int(np.sum(arrays["target"]))
    target_hits = int(np.sum(arrays["target"][nonfrontier]))
    winner = int(np.asarray(batch.selected_index)[0])
    return {
        "group_id": int(np.asarray(batch.group_id)[0]) & ((1 << 64) - 1),
        "phase": int(np.asarray(batch.phase)[0]),
        "candidate_count": len(scores),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact": target_hits == target_slots,
        "r4800_winner_retained": winner in retained,
        "objective": float(objective),
        "finite_scores": bool(np.all(np.isfinite(scores))),
        "finite_residuals": True,
    }


def _aggregate_single_group_reports(
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    if not groups:
        raise ValueError("cannot aggregate empty single-group reports")
    target_slots = sum(int(group["target_slots"]) for group in groups)
    target_hits = sum(int(group["target_hits"]) for group in groups)
    return {
        "groups": len(groups),
        "candidates": sum(int(group["candidates"]) for group in groups),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact_fraction": sum(
            int(float(group["target_set_exact_fraction"]) == 1.0)
            for group in groups
        )
        / len(groups),
        "r4800_winner_retention": sum(
            float(group["r4800_winner_retention"]) for group in groups
        )
        / len(groups),
        "mean_objective": sum(float(group["mean_objective"]) for group in groups)
        / len(groups),
        "all_scores_finite": all(
            bool(group["all_scores_finite"]) for group in groups
        ),
    }


def _load_inputs(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> tuple[
    Scale16ExpectedRankDataset,
    list[Any],
    list[ExpectedRankBatch],
    dict[str, Any],
    Any,
    Path,
]:
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    cohort = select_audit_cohort(dataset, ANALYTIC_GROUPS)
    batches = load_cohort_batches(dataset, cohort)
    identity = _input_identity(dataset, cache_root, cohort)
    config, model_path = _selected_checkpoint_spec(selected_run)
    return dataset, cohort, batches, identity, config, model_path


def run_analytic_optimum(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> dict[str, Any]:
    """Run Arm A's exact optimum and selector ceiling."""
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_before = _system_swap_used_bytes()
    _dataset, _cohort, batches, identity, _config, _model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    groups: list[dict[str, Any]] = []
    ceiling_groups: list[dict[str, Any]] = []
    max_kkt = 0.0
    for batch in batches:
        arrays = _batch_arrays(batch)
        solution = solve_box_constrained_expected_rank(
            arrays["screen"],
            arrays["ranks"],
            arrays["rank_mask"],
            arrays["eligible"],
        )
        metrics = _score_metrics(
            batch,
            solution["scores"],
            objective=float(solution["objective"]),
        )
        metrics.update(
            {
                key: solution[key]
                for key in (
                    "kkt_violation",
                    "normalization_offset",
                    "active_lower",
                    "active_upper",
                    "active_interior",
                )
            }
        )
        groups.append(metrics)
        max_kkt = max(max_kkt, float(solution["kkt_violation"]))

        ceiling = arrays["screen"].copy()
        ceiling[arrays["eligible"] & arrays["target"]] += (
            GRADED_ORACLE_RESIDUAL_RANGE
        )
        ceiling[arrays["eligible"] & ~arrays["target"]] -= (
            GRADED_ORACLE_RESIDUAL_RANGE
        )
        probabilities = _target_probabilities(
            arrays["ranks"],
            arrays["rank_mask"],
        )
        ceiling_groups.append(
            _score_metrics(
                batch,
                ceiling,
                objective=expected_rank_objective(
                    ceiling,
                    probabilities,
                    arrays["eligible"],
                ),
            )
        )
    aggregate = _aggregate_metrics(groups)
    ceiling = _aggregate_metrics(ceiling_groups)
    analytic_pass = bool(
        aggregate["target_positive_recall"] >= 0.95
        and aggregate["target_set_exact_fraction"] >= 0.75
        and max_kkt <= ANALYTIC_KKT_GATE
    )
    scientific = {
        "arm": "analytic-optimum",
        "input_identity": identity,
        "full_cohort_digest_blake3": identity["cohort_digest_blake3"],
        "groups": groups,
        "aggregate": aggregate,
        "selector_ceiling": ceiling,
        "maximum_kkt_violation": max_kkt,
        "gates": {
            "analytic_optimum_passed": analytic_pass,
            "selector_ceiling_passed": bool(
                ceiling["target_positive_recall"] == 1.0
                and ceiling["target_set_exact_fraction"] == 1.0
            ),
            "all_analytic_groups_completed": len(groups) == ANALYTIC_GROUPS,
            "all_analytic_scores_finite": bool(
                aggregate["all_scores_finite"]
                and ceiling["all_scores_finite"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


class _FreeResidualModel(nn.Module):
    def __init__(self, initial_raw: np.ndarray):
        super().__init__()
        self.raw = mx.array(initial_raw.astype(np.float32))

    def scores(self, screen: mx.array) -> mx.array:
        return screen + GRADED_ORACLE_RESIDUAL_RANGE * mx.tanh(self.raw)


def _free_loss(
    model: _FreeResidualModel,
    screen: mx.array,
    ranks: mx.array,
    rank_mask: mx.array,
    eligible: mx.array,
) -> mx.array:
    return expected_rank_loss_from_scores(
        model.scores(screen)[None, :],
        ranks[None, :],
        rank_mask[None, :],
        eligible[None, :],
        target_scale=TARGET_SCALE,
        student_temperature=STUDENT_TEMPERATURE,
    )


def _free_metric_event(
    model: _FreeResidualModel,
    batch: ExpectedRankBatch,
    screen: mx.array,
    ranks: mx.array,
    rank_mask: mx.array,
    eligible: mx.array,
    update: int,
) -> dict[str, Any]:
    scores = model.scores(screen)
    loss = _free_loss(model, screen, ranks, rank_mask, eligible)
    mx.eval(scores, loss)
    return {
        "updates": update,
        "metrics": _score_metrics(
            batch,
            np.asarray(scores).astype(np.float64),
            objective=float(loss.item()),
        ),
    }


def run_free_adam(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> dict[str, Any]:
    """Run Arm B's frozen AdamW free-residual trajectories."""
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_before = _system_swap_used_bytes()
    _dataset, _cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    selected = _new_selected_model(config, model_path)
    group_reports: list[dict[str, Any]] = []
    final_groups: list[dict[str, Any]] = []
    update_checkpoints = set(FREE_CHECKPOINTS)
    for batch in batches[:FREE_GROUPS]:
        arrays = _batch_arrays(batch, model=selected)
        ratio = np.clip(
            arrays["selected_residual"] / GRADED_ORACLE_RESIDUAL_RANGE,
            -0.999999,
            0.999999,
        )
        model = _FreeResidualModel(np.arctanh(ratio))
        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        loss_and_grad = nn.value_and_grad(model, _free_loss)
        screen = mx.array(arrays["screen"].astype(np.float32))
        ranks = mx.array(arrays["ranks"].astype(np.float32))
        rank_mask = mx.array(arrays["rank_mask"])
        eligible = mx.array(arrays["eligible"])
        trajectory: list[dict[str, Any]] = []
        trajectory.append(
            _free_metric_event(
                model,
                batch,
                screen,
                ranks,
                rank_mask,
                eligible,
                0,
            )
        )
        for update in range(1, FREE_UPDATES + 1):
            loss, gradients = loss_and_grad(
                model,
                screen,
                ranks,
                rank_mask,
                eligible,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            if update in update_checkpoints:
                event = _free_metric_event(
                    model,
                    batch,
                    screen,
                    ranks,
                    rank_mask,
                    eligible,
                    update,
                )
                trajectory.append(event)
                final = event["metrics"]
        group_reports.append(
            {
                **final,
                "trajectory": trajectory,
            }
        )
        final_groups.append(final)
    aggregate = _aggregate_metrics(final_groups)
    at_120 = _aggregate_metrics(
        [
            next(
                event["metrics"]
                for event in group["trajectory"]
                if event["updates"] == 120
            )
            for group in group_reports
        ]
    )
    scientific = {
        "arm": "free-adam",
        "input_identity": identity,
        "full_cohort_digest_blake3": identity["cohort_digest_blake3"],
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "updates": FREE_UPDATES,
        "groups": group_reports,
        "aggregate_at_120": at_120,
        "aggregate": aggregate,
        "gates": {
            "free_adam_passed": bool(
                aggregate["target_positive_recall"] >= 0.95
                and aggregate["target_set_exact_fraction"] >= 0.75
            ),
            "free_adam_passed_at_120": bool(
                at_120["target_positive_recall"] >= 0.95
                and at_120["target_set_exact_fraction"] >= 0.75
            ),
            "all_free_adam_groups_completed": len(group_reports) == FREE_GROUPS,
            "all_free_adam_scores_finite": bool(
                aggregate["all_scores_finite"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def run_projected_control(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> dict[str, Any]:
    """Run Arm C's independent projected convex control."""
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_before = _system_swap_used_bytes()
    _dataset, _cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    selected = _new_selected_model(config, model_path)
    groups: list[dict[str, Any]] = []
    for batch in batches[:FREE_GROUPS]:
        arrays = _batch_arrays(batch, model=selected)
        analytic = solve_box_constrained_expected_rank(
            arrays["screen"],
            arrays["ranks"],
            arrays["rank_mask"],
            arrays["eligible"],
        )
        projected = projected_optimize_expected_rank(
            arrays["screen"] + arrays["selected_residual"],
            arrays["screen"],
            arrays["ranks"],
            arrays["rank_mask"],
            arrays["eligible"],
        )
        metrics = _score_metrics(
            batch,
            projected["scores"],
            objective=float(projected["objective"]),
        )
        analytic_metrics = _score_metrics(
            batch,
            analytic["scores"],
            objective=float(analytic["objective"]),
        )
        groups.append(
            {
                **metrics,
                "converged": projected["converged"],
                "iterations": projected["iterations"],
                "kkt_violation": projected["kkt_violation"],
                "objective_gap_from_analytic": (
                    float(projected["objective"])
                    - float(analytic["objective"])
                ),
                "selection_matches_analytic": bool(
                    metrics["target_hits"] == analytic_metrics["target_hits"]
                    and metrics["target_set_exact"]
                    == analytic_metrics["target_set_exact"]
                ),
                "trajectory": projected["trajectory"],
            }
        )
    aggregate = _aggregate_metrics(groups)
    maximum_kkt = max(float(group["kkt_violation"]) for group in groups)
    maximum_gap = max(
        abs(float(group["objective_gap_from_analytic"])) for group in groups
    )
    scientific = {
        "arm": "projected-control",
        "input_identity": identity,
        "full_cohort_digest_blake3": identity["cohort_digest_blake3"],
        "groups": groups,
        "aggregate": aggregate,
        "maximum_kkt_violation": maximum_kkt,
        "maximum_objective_gap_from_analytic": maximum_gap,
        "gates": {
            "projected_control_passed": bool(
                all(bool(group["converged"]) for group in groups)
                and all(bool(group["selection_matches_analytic"]) for group in groups)
                and maximum_kkt <= ANALYTIC_KKT_GATE
                and maximum_gap <= OBJECTIVE_MATCH_GATE
            ),
            "all_projected_groups_completed": len(groups) == FREE_GROUPS,
            "all_projected_scores_finite": bool(
                aggregate["all_scores_finite"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def run_neural_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
) -> dict[str, Any]:
    """Run one disjoint Arm D long-horizon neural continuation shard."""
    if not 0 <= group_index < NEURAL_GROUPS:
        raise ValueError("neural group index is outside the frozen shard range")
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    model = _new_selected_model(config, model_path)
    trajectory = fit_model(
        model,
        [batches[group_index]],
        exposures_per_group=NEURAL_EXPOSURES,
        checkpoints=NEURAL_CHECKPOINTS,
        arm_identity=f"adr0103-neural-{group_index}",
    )
    scientific = {
        "arm": "neural-continuation-shard",
        "group_index": group_index,
        "group_id": cohort[group_index].group_id,
        "input_identity": identity,
        "full_cohort_digest_blake3": identity["cohort_digest_blake3"],
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "trajectory": trajectory,
        "final": trajectory[-1]["metrics"],
        "gates": {
            "all_neural_shard_exposures_completed": (
                trajectory[-1]["exposures_per_group"] == NEURAL_EXPOSURES
            ),
            "all_neural_shard_scores_finite": bool(
                trajectory[-1]["metrics"]["all_scores_finite"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def _arm_pipeline_passed(report: dict[str, Any]) -> bool:
    scientific = report["scientific"]
    telemetry = report["telemetry"]
    completion = all(
        bool(value)
        for name, value in scientific.get("gates", {}).items()
        if name.startswith("all_")
    )
    return bool(
        scientific.get("test_split_opened") is False
        and scientific.get("gameplay_opened") is False
        and scientific.get("new_teacher_compute_used") is False
        and scientific.get("external_compute_used") is False
        and completion
        and int(telemetry["peak_process_rss_bytes"]) <= 4 * 1024**3
        and int(telemetry["process_swaps"]) == 0
        and telemetry["system_swap_delta_bytes"] is not None
        and int(telemetry["system_swap_delta_bytes"]) <= 0
    )


def combine_reports(paths: list[Path]) -> dict[str, Any]:
    """Combine the three fixed arms and four disjoint neural shards."""
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import _system_swap_used_bytes

    swap_before = _system_swap_used_bytes()
    fixed: dict[str, dict[str, Any]] = {}
    neural: list[dict[str, Any]] = []
    cohort_digests: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected ADR 0103 report: {path}")
        scientific = report["scientific"]
        cohort_digests.add(str(scientific["full_cohort_digest_blake3"]))
        arm = str(scientific["arm"])
        if arm == "neural-continuation-shard":
            neural.append(report)
        elif arm in fixed:
            raise ValueError(f"duplicate ADR 0103 arm: {arm}")
        else:
            fixed[arm] = report
    if set(fixed) != {"analytic-optimum", "free-adam", "projected-control"}:
        raise ValueError("ADR 0103 fixed arm set is incomplete")
    if len(neural) != NEURAL_GROUPS or len(cohort_digests) != 1:
        raise ValueError("ADR 0103 neural shards or cohort identity are incomplete")
    neural.sort(key=lambda report: int(report["scientific"]["group_index"]))
    if [int(report["scientific"]["group_index"]) for report in neural] != list(
        range(NEURAL_GROUPS)
    ):
        raise ValueError("ADR 0103 neural shard indices differ")
    if len({int(report["scientific"]["group_id"]) for report in neural}) != NEURAL_GROUPS:
        raise ValueError("ADR 0103 neural shards duplicate a group")

    neural_final_groups = [report["scientific"]["final"] for report in neural]
    neural_at_120_groups = [
        next(
            event["metrics"]
            for event in report["scientific"]["trajectory"]
            if event["exposures_per_group"] == ERROR_CHECKPOINTS[-1]
        )
        for report in neural
    ]
    neural_final = _aggregate_single_group_reports(neural_final_groups)
    neural_at_120 = _aggregate_single_group_reports(neural_at_120_groups)
    analytic = fixed["analytic-optimum"]["scientific"]
    free = fixed["free-adam"]["scientific"]
    projected = fixed["projected-control"]["scientific"]
    pipeline = all(_arm_pipeline_passed(report) for report in [*fixed.values(), *neural])
    selector_ceiling = bool(analytic["gates"]["selector_ceiling_passed"])
    analytic_pass = bool(analytic["gates"]["analytic_optimum_passed"])
    projected_pass = bool(projected["gates"]["projected_control_passed"])
    free_pass = bool(free["gates"]["free_adam_passed"])
    neural_120_pass = bool(
        neural_at_120["target_positive_recall"] >= 0.90
        and neural_at_120["target_set_exact_fraction"] >= 0.75
    )
    neural_final_pass = bool(
        neural_final["target_positive_recall"] >= 0.90
        and neural_final["target_set_exact_fraction"] >= 0.75
    )
    if not pipeline or not selector_ceiling or not projected_pass:
        classification = "free_residual_pipeline_invalid"
    elif not analytic_pass:
        classification = "scale16_objective_box_misaligned"
    elif not free_pass:
        classification = "frozen_optimizer_hyperparameters_insufficient"
    elif not neural_120_pass and neural_final_pass:
        classification = "full_model_local_budget_insufficient"
    elif not neural_final_pass:
        classification = "public_observable_representation_insufficient"
    elif neural_120_pass:
        classification = "local_failure_not_reproduced"
    else:
        classification = "local_mechanism_unresolved"
    scientific = {
        "arm": "combined",
        "classification": classification,
        "full_cohort_digest_blake3": next(iter(cohort_digests)),
        "gates": {
            "pipeline_passed": pipeline,
            "selector_ceiling_passed": selector_ceiling,
            "analytic_optimum_passed": analytic_pass,
            "projected_control_passed": projected_pass,
            "free_adam_passed": free_pass,
            "neural_at_120_passed": neural_120_pass,
            "neural_at_1200_passed": neural_final_pass,
        },
        "neural_at_120": neural_at_120,
        "neural_at_1200": neural_final,
        "arm_telemetry": {
            name: report["telemetry"] for name, report in sorted(fixed.items())
        },
        "neural_telemetry": [
            {
                "group_index": report["scientific"]["group_index"],
                "group_id": report["scientific"]["group_id"],
                **report["telemetry"],
            }
            for report in neural
        ],
        "duplicate_discovery_fraction": 0.0,
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("analytic-optimum", "free-adam", "projected-control"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--selected-run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    neural = subparsers.add_parser("neural-continuation")
    neural.add_argument("--dataset", type=Path, required=True)
    neural.add_argument("--cache", type=Path, required=True)
    neural.add_argument("--selected-run", type=Path, required=True)
    neural.add_argument("--group-index", type=int, required=True)
    neural.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--arm", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "analytic-optimum":
        report = run_analytic_optimum(args.dataset, args.cache, args.selected_run)
    elif args.command == "free-adam":
        report = run_free_adam(args.dataset, args.cache, args.selected_run)
    elif args.command == "projected-control":
        report = run_projected_control(args.dataset, args.cache, args.selected_run)
    elif args.command == "neural-continuation":
        report = run_neural_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.group_index,
        )
    else:
        report = combine_reports(args.arm)
    _write_json(args.output, report)


if __name__ == "__main__":
    main()
