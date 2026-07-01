"""Calibrated monotone AdamW local-fit controls for ADR 0107."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from cascadia_mlx.graded_oracle_frontier_anchor import (
    _system_swap_used_bytes,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    frontier_expected_rank_scale16_loss,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    SELECTED_MODEL_BLAKE3,
    WEIGHT_DECAY,
    _aggregate_metrics,
    _new_selected_model,
    evaluate_model,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    FREE_CHECKPOINTS,
    FREE_GROUPS,
    FREE_UPDATES,
    NEURAL_CHECKPOINTS,
    NEURAL_EXPOSURES,
    NEURAL_GROUPS,
    _batch_arrays,
    _closed_domains,
    _free_loss,
    _free_metric_event,
    _FreeResidualModel,
    _load_inputs,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
)

EXPERIMENT_ID = "complete-action-frontier-calibrated-monotone-adamw-v1"
MAXIMUM_LEARNING_RATE = 2.0 * math.atanh(0.999) / 1200.0
MINIMUM_LEARNING_RATE = 1e-8
BACKTRACK_FACTOR = 0.5
RATE_REGROWTH = 2.0
MAXIMUM_TRIALS = 16
LOSS_TOLERANCE = 1e-12
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
MEMORY_GATE_BYTES = 4 * 1024**3


class NumericalConvergence(RuntimeError):
    """Raised when finite backtracking proves float32 numerical saturation."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        super().__init__("monotone AdamW numerically converged")
        self.diagnostics = diagnostics


def _tree_all_finite(tree: Any) -> bool:
    for _name, value in tree_flatten(tree):
        mx.eval(value)
        if not bool(np.all(np.isfinite(np.asarray(value)))):
            return False
    return True


class MonotoneAdamW:
    """AdamW with an analytically capped rate and same-batch backtracking."""

    def __init__(self) -> None:
        self.first_moment: Any | None = None
        self.second_moment: Any | None = None
        self.next_rate = MAXIMUM_LEARNING_RATE
        self.accepted_rates: list[float] = []
        self.backtracks: list[int] = []
        self.nonfinite_rejections = 0
        self.monotone = True
        self.last_exhaustion_diagnostics: dict[str, Any] | None = None

    def step(
        self,
        model: nn.Module,
        gradients: Any,
        current_loss: mx.array,
        loss_function: Callable[..., mx.array],
        *loss_args: Any,
        allow_numerical_convergence: bool = False,
        convergence_improvement_domain: str = "all",
    ) -> float:
        if convergence_improvement_domain not in {"all", "eligible"}:
            raise ValueError(
                "convergence improvement domain must be all or eligible"
            )
        mx.eval(current_loss, gradients)
        current_value = float(current_loss.item())
        if not math.isfinite(current_value):
            raise RuntimeError("pre-update loss is non-finite")
        parameters = model.trainable_parameters()
        if self.first_moment is None:
            self.first_moment = tree_map(mx.zeros_like, gradients)
            self.second_moment = tree_map(mx.zeros_like, gradients)
        next_first = tree_map(
            lambda moment, gradient: (
                BETA1 * moment + (1.0 - BETA1) * gradient
            ),
            self.first_moment,
            gradients,
        )
        next_second = tree_map(
            lambda moment, gradient: (
                BETA2 * moment
                + (1.0 - BETA2) * mx.square(gradient)
            ),
            self.second_moment,
            gradients,
        )
        direction = tree_map(
            lambda first, second, parameter: (
                first / (mx.sqrt(second) + EPSILON)
                + WEIGHT_DECAY * parameter
            ),
            next_first,
            next_second,
            parameters,
        )
        current_state_finite = bool(
            _tree_all_finite(parameters)
            and _tree_all_finite(next_first)
            and _tree_all_finite(next_second)
            and _tree_all_finite(direction)
        )
        rate = min(MAXIMUM_LEARNING_RATE, self.next_rate)
        attempted_rates: list[float] = []
        candidate_values: list[float] = []
        proposal_finite: list[bool] = []
        for trial in range(MAXIMUM_TRIALS):
            attempted_rates.append(rate)
            proposal = tree_map(
                lambda parameter, update, step=rate: (
                    parameter - step * update
                ),
                parameters,
                direction,
            )
            model.update(proposal)
            candidate_loss = loss_function(model, *loss_args)
            mx.eval(candidate_loss, model.parameters())
            candidate_value = float(candidate_loss.item())
            finite = bool(
                math.isfinite(candidate_value)
                and _tree_all_finite(model.trainable_parameters())
            )
            candidate_values.append(candidate_value)
            proposal_finite.append(finite)
            rate_is_eligible = rate >= MINIMUM_LEARNING_RATE
            if (
                rate_is_eligible
                and finite
                and candidate_value <= current_value + LOSS_TOLERANCE
            ):
                mx.eval(next_first, next_second)
                self.first_moment = next_first
                self.second_moment = next_second
                self.accepted_rates.append(rate)
                self.backtracks.append(trial)
                self.last_exhaustion_diagnostics = None
                self.next_rate = min(
                    MAXIMUM_LEARNING_RATE,
                    rate * RATE_REGROWTH,
                )
                self.monotone = self.monotone and (
                    candidate_value <= current_value + LOSS_TOLERANCE
                )
                return candidate_value
            if not finite:
                self.nonfinite_rejections += 1
            model.update(parameters)
            rate *= BACKTRACK_FACTOR
            if (
                not allow_numerical_convergence
                and rate < MINIMUM_LEARNING_RATE
            ):
                break
        model.update(parameters)
        finite_candidates = [
            value
            for value, finite in zip(
                candidate_values,
                proposal_finite,
                strict=True,
            )
            if finite
        ]
        eligible_finite_candidates = [
            value
            for value, finite, attempted_rate in zip(
                candidate_values,
                proposal_finite,
                attempted_rates,
                strict=True,
            )
            if finite and attempted_rate >= MINIMUM_LEARNING_RATE
        ]
        maximum_all_candidate_improvement = (
            current_value - min(finite_candidates)
            if finite_candidates
            else math.inf
        )
        maximum_eligible_candidate_improvement = (
            current_value - min(eligible_finite_candidates)
            if eligible_finite_candidates
            else math.inf
        )
        convergence_improvement = (
            maximum_all_candidate_improvement
            if convergence_improvement_domain == "all"
            else maximum_eligible_candidate_improvement
        )
        diagnostics = {
            "proposals_attempted": len(attempted_rates),
            "all_proposals_finite": bool(
                len(proposal_finite) == MAXIMUM_TRIALS
                and all(proposal_finite)
            ),
            "current_state_finite": current_state_finite,
            "smallest_attempted_rate": (
                min(attempted_rates) if attempted_rates else None
            ),
            "maximum_candidate_improvement": convergence_improvement,
            "maximum_all_candidate_improvement": (
                maximum_all_candidate_improvement
            ),
            "maximum_eligible_candidate_improvement": (
                maximum_eligible_candidate_improvement
            ),
            "convergence_improvement_domain": (
                convergence_improvement_domain
            ),
            "prior_accepted_updates": len(self.accepted_rates),
        }
        self.last_exhaustion_diagnostics = diagnostics
        if (
            allow_numerical_convergence
            and diagnostics["all_proposals_finite"]
            and diagnostics["current_state_finite"]
            and diagnostics["smallest_attempted_rate"] is not None
            and diagnostics["smallest_attempted_rate"] < 1e-7
            and diagnostics["maximum_candidate_improvement"]
            <= LOSS_TOLERANCE
            and diagnostics["prior_accepted_updates"] > 0
        ):
            raise NumericalConvergence(diagnostics)
        raise RuntimeError("monotone AdamW could not accept an update")

    def summary(self) -> dict[str, Any]:
        rates = np.asarray(self.accepted_rates, dtype=np.float64)
        backs = np.asarray(self.backtracks, dtype=np.int64)
        moments_finite = bool(
            self.first_moment is not None
            and self.second_moment is not None
            and _tree_all_finite(self.first_moment)
            and _tree_all_finite(self.second_moment)
        )
        return {
            "accepted_updates": len(self.accepted_rates),
            "maximum_learning_rate": MAXIMUM_LEARNING_RATE,
            "minimum_accepted_rate": (
                float(np.min(rates)) if len(rates) else None
            ),
            "maximum_accepted_rate": (
                float(np.max(rates)) if len(rates) else None
            ),
            "mean_accepted_rate": (
                float(np.mean(rates)) if len(rates) else None
            ),
            "total_backtracks": int(np.sum(backs)) if len(backs) else 0,
            "maximum_backtracks": int(np.max(backs)) if len(backs) else 0,
            "nonfinite_rejections": self.nonfinite_rejections,
            "loss_monotone": self.monotone,
            "moments_finite": moments_finite,
            "last_exhaustion_diagnostics": (
                self.last_exhaustion_diagnostics
            ),
        }


def _input_identity_summary(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_manifest_blake3": identity["train_manifest_blake3"],
        "cache_manifest_blake3": identity["cache_manifest_blake3"],
        "cache_ordered_group_action_identity_blake3": (
            identity["cache_ordered_group_action_identity_blake3"]
        ),
        "cohort_digest_blake3": identity["cohort_digest_blake3"],
    }


def _report(
    scientific: dict[str, Any],
    started: float,
    swap_before: int | None,
    *,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
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


def run_free_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
    *,
    experiment_id: str = EXPERIMENT_ID,
    arm: str = "calibrated-free-residual-group",
    allow_numerical_convergence: bool = False,
) -> dict[str, Any]:
    """Run one independently scheduled calibrated free-residual group."""
    if not 0 <= group_index < FREE_GROUPS:
        raise ValueError("free-residual group index is outside 0-23")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    selected = _new_selected_model(config, model_path)
    batch = batches[group_index]
    arrays = _batch_arrays(batch, model=selected)
    ratio = np.clip(
        arrays["selected_residual"] / GRADED_ORACLE_RESIDUAL_RANGE,
        -0.999999,
        0.999999,
    )
    model = _FreeResidualModel(np.arctanh(ratio))
    optimizer = MonotoneAdamW()
    loss_and_grad = nn.value_and_grad(model, _free_loss)
    screen = mx.array(arrays["screen"].astype(np.float32))
    ranks = mx.array(arrays["ranks"].astype(np.float32))
    rank_mask = mx.array(arrays["rank_mask"])
    eligible = mx.array(arrays["eligible"])
    trajectory = [
        _free_metric_event(
            model,
            batch,
            screen,
            ranks,
            rank_mask,
            eligible,
            0,
        )
    ]
    checkpoints = set(FREE_CHECKPOINTS)
    failure: str | None = None
    numerical_convergence: dict[str, Any] | None = None
    for update in range(1, FREE_UPDATES + 1):
        loss, gradients = loss_and_grad(
            model,
            screen,
            ranks,
            rank_mask,
            eligible,
        )
        try:
            optimizer.step(
                model,
                gradients,
                loss,
                _free_loss,
                screen,
                ranks,
                rank_mask,
                eligible,
                allow_numerical_convergence=(
                    allow_numerical_convergence
                ),
            )
        except NumericalConvergence as convergence:
            numerical_convergence = convergence.diagnostics
            if trajectory[-1]["updates"] != (
                optimizer.summary()["accepted_updates"]
            ):
                trajectory.append(
                    _free_metric_event(
                        model,
                        batch,
                        screen,
                        ranks,
                        rank_mask,
                        eligible,
                        optimizer.summary()["accepted_updates"],
                    )
                )
            break
        except RuntimeError as error:
            failure = str(error)
            break
        if update in checkpoints:
            trajectory.append(
                _free_metric_event(
                    model,
                    batch,
                    screen,
                    ranks,
                    rank_mask,
                    eligible,
                    update,
                )
            )
    final = trajectory[-1]["metrics"]
    optimizer_summary = optimizer.summary()
    completed = bool(
        optimizer_summary["accepted_updates"] == FREE_UPDATES
        or numerical_convergence is not None
    )
    scientific = {
        "arm": arm,
        "group_index": group_index,
        "group_id": int(cohort[group_index].group_id),
        "input_identity": _input_identity_summary(identity),
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "updates": FREE_UPDATES,
        "trajectory": trajectory,
        "final": final,
        "optimizer": optimizer_summary,
        "failure": failure,
        "numerical_convergence": numerical_convergence,
        "gates": {
            "all_updates_completed_or_numerically_converged": completed,
            "all_scores_finite": bool(final["finite_scores"]),
            "all_optimizer_values_finite": bool(
                optimizer_summary["moments_finite"]
                and optimizer_summary["minimum_accepted_rate"] is not None
                and math.isfinite(
                    optimizer_summary["minimum_accepted_rate"]
                )
            ),
            "all_updates_monotone": bool(
                optimizer_summary["loss_monotone"]
            ),
        },
        **_closed_domains(),
    }
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=experiment_id,
    )


def run_neural_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
    *,
    experiment_id: str = EXPERIMENT_ID,
    arm: str = "calibrated-neural-group",
    allow_numerical_convergence: bool = False,
) -> dict[str, Any]:
    """Run one calibrated full-model local continuation group."""
    if not 0 <= group_index < NEURAL_GROUPS:
        raise ValueError("neural group index is outside 0-3")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    model = _new_selected_model(config, model_path)
    batch = batches[group_index]
    optimizer = MonotoneAdamW()
    loss_and_grad = nn.value_and_grad(
        model,
        frontier_expected_rank_scale16_loss,
    )
    trajectory = [
        {
            "exposures_per_group": 0,
            "optimizer_steps": 0,
            "metrics": evaluate_model(model, [batch]),
        }
    ]
    checkpoints = set(NEURAL_CHECKPOINTS)
    failure: str | None = None
    numerical_convergence: dict[str, Any] | None = None
    for exposure in range(1, NEURAL_EXPOSURES + 1):
        rotation = (exposure - 1) % 6
        rotated = rotate_expected_rank_batch(batch, rotation)
        loss, gradients = loss_and_grad(model, rotated)
        try:
            optimizer.step(
                model,
                gradients,
                loss,
                frontier_expected_rank_scale16_loss,
                rotated,
                allow_numerical_convergence=(
                    allow_numerical_convergence
                ),
            )
        except NumericalConvergence as convergence:
            numerical_convergence = convergence.diagnostics
            accepted = optimizer.summary()["accepted_updates"]
            if trajectory[-1]["exposures_per_group"] != accepted:
                trajectory.append(
                    {
                        "exposures_per_group": accepted,
                        "optimizer_steps": accepted,
                        "metrics": evaluate_model(model, [batch]),
                    }
                )
            break
        except RuntimeError as error:
            failure = str(error)
            break
        if exposure in checkpoints:
            trajectory.append(
                {
                    "exposures_per_group": exposure,
                    "optimizer_steps": exposure,
                    "metrics": evaluate_model(model, [batch]),
                }
            )
    final = trajectory[-1]["metrics"]
    optimizer_summary = optimizer.summary()
    completed = bool(
        optimizer_summary["accepted_updates"] == NEURAL_EXPOSURES
        or numerical_convergence is not None
    )
    scientific = {
        "arm": arm,
        "group_index": group_index,
        "group_id": int(cohort[group_index].group_id),
        "input_identity": _input_identity_summary(identity),
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "exposures": NEURAL_EXPOSURES,
        "trajectory": trajectory,
        "final": final,
        "optimizer": optimizer_summary,
        "failure": failure,
        "numerical_convergence": numerical_convergence,
        "gates": {
            "all_exposures_completed_or_numerically_converged": completed,
            "all_scores_finite": bool(final["all_scores_finite"]),
            "all_optimizer_values_finite": bool(
                optimizer_summary["moments_finite"]
                and optimizer_summary["minimum_accepted_rate"] is not None
                and math.isfinite(
                    optimizer_summary["minimum_accepted_rate"]
                )
            ),
            "all_updates_monotone": bool(
                optimizer_summary["loss_monotone"]
            ),
        },
        **_closed_domains(),
    }
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=experiment_id,
    )


def _resource_passed(telemetry: dict[str, Any]) -> bool:
    return bool(
        int(telemetry["peak_process_rss_bytes"]) <= MEMORY_GATE_BYTES
        and int(telemetry["process_swaps"]) == 0
        and telemetry["system_swap_delta_bytes"] is not None
        and int(telemetry["system_swap_delta_bytes"]) <= 0
    )


def _validate_reports(
    paths: list[Path],
    comparisons: list[Path],
    *,
    arm: str,
    groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    reports_by_group: dict[int, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected ADR 0107 report: {path}")
        scientific = report["scientific"]
        if scientific.get("arm") != arm:
            raise ValueError(f"unexpected ADR 0107 arm: {path}")
        index = int(scientific["group_index"])
        if index in reports_by_group:
            raise ValueError(f"duplicate ADR 0107 group: {index}")
        reports_by_group[index] = report
    if set(reports_by_group) != set(range(groups)):
        raise ValueError("ADR 0107 group set is incomplete")
    comparisons_by_group: dict[int, dict[str, Any]] = {}
    for path in comparisons:
        comparison = json.loads(path.read_text())
        if comparison.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected ADR 0107 replay: {path}")
        index = int(comparison["group_index"])
        if index in comparisons_by_group:
            raise ValueError(f"duplicate ADR 0107 replay: {index}")
        comparisons_by_group[index] = comparison
    if set(comparisons_by_group) != set(range(groups)):
        raise ValueError("ADR 0107 replay set is incomplete")
    ordered = [reports_by_group[index] for index in range(groups)]
    ordered_comparisons = [
        comparisons_by_group[index] for index in range(groups)
    ]
    pipeline = all(
        all(bool(value) for value in report["scientific"]["gates"].values())
        and _resource_passed(report["telemetry"])
        and _resource_passed(comparison["replay_telemetry"])
        and comparison["scientific_payload_identical"] is True
        and report["scientific"]["test_split_opened"] is False
        and report["scientific"]["gameplay_opened"] is False
        and report["scientific"]["new_teacher_compute_used"] is False
        and report["scientific"]["external_compute_used"] is False
        for report, comparison in zip(
            ordered,
            ordered_comparisons,
            strict=True,
        )
    )
    return ordered, ordered_comparisons, pipeline


def combine_free(
    paths: list[Path],
    comparisons: list[Path],
) -> dict[str, Any]:
    """Combine and gate the 24 calibrated free-residual groups."""
    reports, replay_reports, pipeline = _validate_reports(
        paths,
        comparisons,
        arm="calibrated-free-residual-group",
        groups=FREE_GROUPS,
    )
    finals = [report["scientific"]["final"] for report in reports]
    at_120 = [
        next(
            event["metrics"]
            for event in report["scientific"]["trajectory"]
            if event["updates"] == 120
        )
        for report in reports
    ]
    aggregate = _aggregate_metrics(finals)
    aggregate_at_120 = _aggregate_metrics(at_120)
    strength_gate = bool(
        aggregate["target_positive_recall"] >= 0.95
        and aggregate["target_set_exact_fraction"] >= 0.75
    )
    scientific = {
        "arm": "calibrated-free-combined",
        "classification": (
            "free_stage_passed"
            if pipeline and strength_gate
            else (
                "calibrated_optimizer_mechanism_insufficient"
                if pipeline
                else "calibrated_optimizer_pipeline_invalid"
            )
        ),
        "groups": [report["scientific"] for report in reports],
        "aggregate_at_120": aggregate_at_120,
        "aggregate": aggregate,
        "gates": {
            "free_pipeline_passed": pipeline,
            "free_strength_gate_passed": strength_gate,
            "all_24_replays_identical": all(
                comparison["scientific_payload_identical"] is True
                for comparison in replay_reports
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, time.perf_counter(), _system_swap_used_bytes())


def combine_final(
    free_path: Path,
    neural_paths: list[Path],
    comparisons: list[Path],
) -> dict[str, Any]:
    """Combine the authorized neural stage and classify ADR 0107."""
    free_report = json.loads(free_path.read_text())
    free = free_report["scientific"]
    if (
        free_report.get("experiment_id") != EXPERIMENT_ID
        or free.get("classification") != "free_stage_passed"
    ):
        raise ValueError("ADR 0107 neural stage lacks a passing free gate")
    reports, replay_reports, neural_pipeline = _validate_reports(
        neural_paths,
        comparisons,
        arm="calibrated-neural-group",
        groups=NEURAL_GROUPS,
    )
    finals = [report["scientific"]["final"] for report in reports]
    at_120 = [
        next(
            event["metrics"]
            for event in report["scientific"]["trajectory"]
            if event["exposures_per_group"] == 120
        )
        for report in reports
    ]
    aggregate = _aggregate_metrics(finals)
    aggregate_at_120 = _aggregate_metrics(at_120)
    passes_1200 = bool(
        aggregate["target_positive_recall"] >= 0.90
        and aggregate["target_set_exact_fraction"] >= 0.75
    )
    passes_120 = bool(
        aggregate_at_120["target_positive_recall"] >= 0.90
        and aggregate_at_120["target_set_exact_fraction"] >= 0.75
    )
    if not neural_pipeline:
        classification = "calibrated_optimizer_pipeline_invalid"
    elif not passes_1200:
        classification = "public_observable_representation_insufficient"
    elif not passes_120:
        classification = "full_model_local_budget_insufficient"
    elif passes_120:
        classification = "local_failure_not_reproduced"
    else:
        classification = "local_optimizer_mechanism_confirmed"
    scientific = {
        "arm": "combined",
        "classification": classification,
        "free_stage": free,
        "neural_groups": [report["scientific"] for report in reports],
        "neural_at_120": aggregate_at_120,
        "neural_at_1200": aggregate,
        "gates": {
            "free_stage_passed": True,
            "neural_pipeline_passed": neural_pipeline,
            "neural_strength_gate_at_120": passes_120,
            "neural_strength_gate_at_1200": passes_1200,
            "all_four_neural_replays_identical": all(
                comparison["scientific_payload_identical"] is True
                for comparison in replay_reports
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, time.perf_counter(), _system_swap_used_bytes())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("free-group", "neural-group"):
        group = subparsers.add_parser(command)
        group.add_argument("--dataset", type=Path, required=True)
        group.add_argument("--cache", type=Path, required=True)
        group.add_argument("--selected-run", type=Path, required=True)
        group.add_argument("--analytic", type=Path)
        group.add_argument("--group-index", type=int, required=True)
        group.add_argument("--output", type=Path, required=True)
    combine_free_parser = subparsers.add_parser("combine-free")
    combine_free_parser.add_argument(
        "--group",
        type=Path,
        action="append",
        required=True,
    )
    combine_free_parser.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine_free_parser.add_argument("--output", type=Path, required=True)
    combine_final_parser = subparsers.add_parser("combine-final")
    combine_final_parser.add_argument("--free", type=Path, required=True)
    combine_final_parser.add_argument(
        "--neural",
        type=Path,
        action="append",
        required=True,
    )
    combine_final_parser.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine_final_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "free-group":
        report = run_free_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.group_index,
        )
    elif args.command == "neural-group":
        report = run_neural_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.group_index,
        )
    elif args.command == "combine-free":
        report = combine_free(args.group, args.replay_comparison)
    else:
        report = combine_final(
            args.free,
            args.neural,
            args.replay_comparison,
        )
    _write_json(args.output, report)
    if args.command in {"free-group", "neural-group"}:
        telemetry = report["telemetry"]
        print(
            json.dumps(
                {
                    "group_index": report["scientific"]["group_index"],
                    "resource_qualification_passed": _resource_passed(
                        telemetry
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
