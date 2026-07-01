"""Independent arbitrary-precision frontier control for ADR 0105."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_frontier_anchor import (
    FRONTIER_ANCHORED_WIDTH,
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    _batch_arrays,
    _closed_domains,
    _load_inputs,
)

EXPERIMENT_ID = "complete-action-frontier-arbitrary-precision-control-v1"
SOURCE_EXPERIMENT_ID = "complete-action-frontier-free-residual-audit-v1"
GROUPS = 24
DECIMAL_PRECISION = 96
TARGET_SCALE = Decimal(16)
STUDENT_TEMPERATURE = Decimal(2)
RESIDUAL_RANGE = Decimal(12)
NORMALIZATION_GATE = Decimal("1e-60")
KKT_GATE = Decimal("1e-60")
FLOAT_MATCH_GATE = Decimal("1e-12")
MEMORY_GATE_BYTES = 4 * 1024**3


def _decimal_float(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite Decimal value")
    return str(value)


def _target_probabilities_decimal(
    ranks: np.ndarray,
    rank_mask: np.ndarray,
    *,
    exact_float_ranks: bool = False,
) -> dict[int, Decimal]:
    weights: dict[int, Decimal] = {}
    for index in np.flatnonzero(rank_mask):
        rank = (
            _decimal_float(float(ranks[index]))
            if exact_float_ranks
            else Decimal(int(ranks[index]))
        )
        weights[int(index)] = (
            -(rank - Decimal(1)) / TARGET_SCALE
        ).exp()
    if not weights:
        raise ValueError("expected-rank group has no positive target mass")
    total = sum(weights.values(), Decimal(0))
    return {index: weight / total for index, weight in weights.items()}


def _active_set_offset(
    *,
    lower: list[Decimal],
    upper: list[Decimal],
    eligible: np.ndarray,
    probabilities: dict[int, Decimal],
) -> Decimal:
    base = {
        index: STUDENT_TEMPERATURE * probability.ln()
        for index, probability in probabilities.items()
    }
    bound_sum = sum(
        (
            (lower[index] / STUDENT_TEMPERATURE).exp()
            for index in np.flatnonzero(eligible)
        ),
        Decimal(0),
    )
    interior_mass = Decimal(0)
    events: dict[Decimal, list[tuple[str, int]]] = {}
    for index in probabilities:
        lower_break = lower[index] - base[index]
        upper_break = upper[index] - base[index]
        events.setdefault(lower_break, []).append(("lower", index))
        events.setdefault(upper_break, []).append(("upper", index))

    def candidate() -> tuple[str, Decimal | None]:
        remaining = Decimal(1) - interior_mass
        if (
            abs(bound_sum) <= Decimal("1e-80")
            and abs(remaining) <= Decimal("1e-80")
        ):
            return "flat", None
        if bound_sum <= 0 or remaining <= 0:
            return "none", None
        return (
            "value",
            STUDENT_TEMPERATURE * (bound_sum / remaining).ln(),
        )

    lower_boundary: Decimal | None = None
    for upper_boundary in sorted(events):
        state, value = candidate()
        if state == "flat":
            return (
                upper_boundary
                if lower_boundary is None
                else (lower_boundary + upper_boundary) / Decimal(2)
            )
        if value is not None and (
            (lower_boundary is None or value >= lower_boundary)
            and value <= upper_boundary
        ):
            return value
        for kind, index in sorted(events[upper_boundary]):
            if kind == "lower":
                bound_sum -= (
                    lower[index] / STUDENT_TEMPERATURE
                ).exp()
                interior_mass += probabilities[index]
            else:
                interior_mass -= probabilities[index]
                bound_sum += (
                    upper[index] / STUDENT_TEMPERATURE
                ).exp()
        lower_boundary = upper_boundary
    state, value = candidate()
    if state == "flat":
        if lower_boundary is None:
            raise ValueError("translation-invariant interval is unbounded")
        return lower_boundary
    if value is None:
        raise ValueError("active-set sweep has no normalization root")
    if lower_boundary is not None and value < lower_boundary:
        raise ValueError("active-set sweep failed to locate the offset")
    return value


def solve_decimal_box_expected_rank(
    screen: np.ndarray,
    ranks: np.ndarray,
    rank_mask: np.ndarray,
    eligible: np.ndarray,
    *,
    exact_float_ranks: bool = False,
) -> dict[str, Any]:
    """Solve one expected-rank box problem with independent Decimal math."""
    if np.any(rank_mask & ~eligible):
        raise ValueError("expected-rank mass includes an ineligible action")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        screen_decimal = [_decimal_float(value) for value in screen]
        lower = [value - RESIDUAL_RANGE for value in screen_decimal]
        upper = [value + RESIDUAL_RANGE for value in screen_decimal]
        probabilities = _target_probabilities_decimal(
            ranks,
            rank_mask,
            exact_float_ranks=exact_float_ranks,
        )
        base = {
            index: STUDENT_TEMPERATURE * probability.ln()
            for index, probability in probabilities.items()
        }
        offset = _active_set_offset(
            lower=lower,
            upper=upper,
            eligible=eligible,
            probabilities=probabilities,
        )
        scores = list(lower)
        for index, value in base.items():
            score = value + offset
            scores[index] = min(max(score, lower[index]), upper[index])

        logits = [
            scores[index] / STUDENT_TEMPERATURE
            for index in np.flatnonzero(eligible)
        ]
        maximum_logit = max(logits)
        log_sum_exp = maximum_logit + sum(
            ((value - maximum_logit).exp() for value in logits),
            Decimal(0),
        ).ln()
        objective = log_sum_exp - sum(
            (
                probability
                * scores[index]
                / STUDENT_TEMPERATURE
                for index, probability in probabilities.items()
            ),
            Decimal(0),
        )
        model_probabilities = {
            int(index): (
                (scores[int(index)] - offset)
                / STUDENT_TEMPERATURE
            ).exp()
            for index in np.flatnonzero(eligible)
        }
        normalization_residual = abs(
            sum(model_probabilities.values(), Decimal(0)) - Decimal(1)
        )
        kkt_violation = Decimal(0)
        for index in np.flatnonzero(eligible):
            integer_index = int(index)
            gradient = (
                model_probabilities[integer_index]
                - probabilities.get(integer_index, Decimal(0))
            ) / STUDENT_TEMPERATURE
            if scores[integer_index] == lower[integer_index]:
                violation = max(Decimal(0), -gradient)
            elif scores[integer_index] == upper[integer_index]:
                violation = max(Decimal(0), gradient)
            else:
                violation = abs(gradient)
            kkt_violation = max(kkt_violation, violation)
        active_lower = sum(
            int(eligible[index] and scores[index] == lower[index])
            for index in range(len(scores))
        )
        active_upper = sum(
            int(eligible[index] and scores[index] == upper[index])
            for index in range(len(scores))
        )
        active_interior = int(np.sum(eligible)) - active_lower - active_upper
        return {
            "scores": scores,
            "objective": objective,
            "normalization_offset": offset,
            "normalization_residual": normalization_residual,
            "kkt_violation": kkt_violation,
            "active_lower": active_lower,
            "active_upper": active_upper,
            "active_interior": active_interior,
            "all_values_finite": bool(
                objective.is_finite()
                and offset.is_finite()
                and normalization_residual.is_finite()
                and kkt_violation.is_finite()
                and all(value.is_finite() for value in scores)
            ),
        }


def decimal_frontier_retained_indices(
    *,
    scores: list[Decimal],
    source_flags: np.ndarray,
    action_hashes: np.ndarray,
    width: int = FRONTIER_ANCHORED_WIDTH,
) -> list[int]:
    """Independently apply the anchored selector in Decimal score space."""
    count = len(scores)
    if count == 0 or len(source_flags) != count or len(action_hashes) != count:
        raise ValueError("Decimal selector arrays have inconsistent lengths")

    def ranking(indices: list[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: (-scores[index], bytes(action_hashes[index])),
        )

    frontier = ranking(
        [
            index
            for index in range(count)
            if int(source_flags[index]) & GRADED_SOURCE_CHAMPION_FRONTIER
        ]
    )
    if len(frontier) > min(width, count):
        raise ValueError("champion frontier exceeds the anchored proposal width")
    nonfrontier = ranking(
        [
            index
            for index in range(count)
            if not (
                int(source_flags[index])
                & GRADED_SOURCE_CHAMPION_FRONTIER
            )
        ]
    )
    quota = min(width, count) - len(frontier)
    return frontier + nonfrontier[:quota]


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_decimal_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    analytic_report_path: Path,
    group_index: int,
    *,
    experiment_id: str = EXPERIMENT_ID,
    scientific_arm: str = "arbitrary-precision-control-group",
    exact_float_ranks: bool = False,
) -> dict[str, Any]:
    """Run one independently schedulable Decimal control group."""
    if not 0 <= group_index < GROUPS:
        raise ValueError("Decimal control group index is outside 0-23")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, _config, _model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    analytic_report = json.loads(analytic_report_path.read_text())
    analytic_scientific = analytic_report["scientific"]
    if analytic_scientific.get("arm") != "analytic-optimum":
        raise ValueError("frozen analytic report has the wrong arm")
    batch = batches[group_index]
    arrays = _batch_arrays(batch)
    solution = solve_decimal_box_expected_rank(
        arrays["screen"],
        arrays["ranks"],
        arrays["rank_mask"],
        arrays["eligible"],
        exact_float_ranks=exact_float_ranks,
    )
    retained = decimal_frontier_retained_indices(
        scores=solution["scores"],
        source_flags=arrays["flags"],
        action_hashes=arrays["hashes"],
    )
    retained_nonfrontier = [
        index
        for index in retained
        if not (
            int(arrays["flags"][index])
            & GRADED_SOURCE_CHAMPION_FRONTIER
        )
    ]
    target_slots = int(np.sum(arrays["target"]))
    target_hits = sum(
        int(bool(arrays["target"][index]))
        for index in retained_nonfrontier
    )
    winner = int(np.asarray(batch.selected_index)[0])
    selected_hash_payload = b"".join(
        bytes(arrays["hashes"][index]) for index in retained
    )
    analytic = analytic_scientific["groups"][group_index]
    objective_difference = abs(
        solution["objective"] - Decimal(str(analytic["objective"]))
    )
    offset_difference = abs(
        solution["normalization_offset"]
        - Decimal(str(analytic["normalization_offset"]))
    )
    active_counts_match = bool(
        solution["active_lower"] == int(analytic["active_lower"])
        and solution["active_interior"] == int(analytic["active_interior"])
        and solution["active_upper"] == int(analytic["active_upper"])
    )
    identity_matches = bool(
        int(cohort[group_index].group_id) == int(analytic["group_id"])
        and len(arrays["screen"]) == int(analytic["candidate_count"])
        and target_slots == int(analytic["target_slots"])
    )
    selector_matches = bool(
        target_hits == int(analytic["target_hits"])
        and (target_hits == target_slots)
        == bool(analytic["target_set_exact"])
        and (winner in retained)
        == bool(analytic["r4800_winner_retained"])
    )
    group_passed = bool(
        solution["normalization_residual"] <= NORMALIZATION_GATE
        and solution["kkt_violation"] <= KKT_GATE
        and objective_difference <= FLOAT_MATCH_GATE
        and offset_difference <= FLOAT_MATCH_GATE
        and active_counts_match
        and identity_matches
        and selector_matches
        and target_hits == target_slots
        and solution["all_values_finite"]
    )
    scientific = {
        "arm": scientific_arm,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "group_index": group_index,
        "group_id": int(cohort[group_index].group_id),
        "phase": int(analytic["phase"]),
        "candidate_count": len(arrays["screen"]),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": (
            target_hits / max(target_slots, 1)
        ),
        "target_set_exact": target_hits == target_slots,
        "r4800_winner_retained": winner in retained,
        "selected_hash_blake3": blake3.blake3(
            selected_hash_payload
        ).hexdigest(),
        "decimal_precision": DECIMAL_PRECISION,
        "rank_conversion": (
            "Decimal.from_float"
            if exact_float_ranks
            else "integer"
        ),
        "objective": _decimal_text(solution["objective"]),
        "normalization_offset": _decimal_text(
            solution["normalization_offset"]
        ),
        "normalization_residual": _decimal_text(
            solution["normalization_residual"]
        ),
        "kkt_violation": _decimal_text(solution["kkt_violation"]),
        "active_lower": int(solution["active_lower"]),
        "active_interior": int(solution["active_interior"]),
        "active_upper": int(solution["active_upper"]),
        "frozen_analytic": {
            "objective": str(analytic["objective"]),
            "normalization_offset": str(analytic["normalization_offset"]),
            "active_lower": int(analytic["active_lower"]),
            "active_interior": int(analytic["active_interior"]),
            "active_upper": int(analytic["active_upper"]),
            "target_hits": int(analytic["target_hits"]),
            "target_set_exact": bool(analytic["target_set_exact"]),
            "r4800_winner_retained": bool(
                analytic["r4800_winner_retained"]
            ),
        },
        "objective_difference": _decimal_text(objective_difference),
        "offset_difference": _decimal_text(offset_difference),
        "input_identity": {
            "train_manifest_blake3": identity["train_manifest_blake3"],
            "cache_manifest_blake3": identity["cache_manifest_blake3"],
            "cache_ordered_group_action_identity_blake3": (
                identity["cache_ordered_group_action_identity_blake3"]
            ),
            "cohort_digest_blake3": identity["cohort_digest_blake3"],
        },
        "gates": {
            "group_passed": group_passed,
            "normalization_gate_passed": (
                solution["normalization_residual"]
                <= NORMALIZATION_GATE
            ),
            "kkt_gate_passed": solution["kkt_violation"] <= KKT_GATE,
            "objective_match_gate_passed": (
                objective_difference <= FLOAT_MATCH_GATE
            ),
            "offset_match_gate_passed": (
                offset_difference <= FLOAT_MATCH_GATE
            ),
            "active_counts_match": active_counts_match,
            "identity_matches": identity_matches,
            "selector_matches": selector_matches,
            "all_values_finite": bool(solution["all_values_finite"]),
        },
        **_closed_domains(),
    }
    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "scientific": scientific,
        "telemetry": {
            "host": socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": _peak_rss_bytes(),
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


def _resource_report_passed(telemetry: dict[str, Any]) -> bool:
    return bool(
        int(telemetry["peak_process_rss_bytes"]) <= MEMORY_GATE_BYTES
        and int(telemetry["process_swaps"]) == 0
        and telemetry["system_swap_delta_bytes"] is not None
        and int(telemetry["system_swap_delta_bytes"]) <= 0
    )


def combine_group_reports(
    group_paths: list[Path],
    replay_comparison_paths: list[Path],
    *,
    experiment_id: str = EXPERIMENT_ID,
    scientific_arm: str = "arbitrary-precision-control-group",
    invalid_classification: str = "arbitrary_precision_control_invalid",
) -> dict[str, Any]:
    """Combine all Decimal groups and replay evidence."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    groups: dict[int, dict[str, Any]] = {}
    for path in group_paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != experiment_id:
            raise ValueError(f"unexpected ADR 0105 group report: {path}")
        scientific = report["scientific"]
        if scientific.get("arm") != scientific_arm:
            raise ValueError(f"unexpected control arm in group report: {path}")
        group_index = int(scientific["group_index"])
        if group_index in groups:
            raise ValueError(f"duplicate ADR 0105 group: {group_index}")
        groups[group_index] = report
    if set(groups) != set(range(GROUPS)):
        raise ValueError("ADR 0105 group set is incomplete")

    comparisons: dict[int, dict[str, Any]] = {}
    for path in replay_comparison_paths:
        comparison = json.loads(path.read_text())
        if comparison.get("experiment_id") != experiment_id:
            raise ValueError(f"unexpected ADR 0105 replay report: {path}")
        group_index = int(comparison["group_index"])
        if group_index in comparisons:
            raise ValueError(f"duplicate ADR 0105 replay: {group_index}")
        comparisons[group_index] = comparison
    if set(comparisons) != set(range(GROUPS)):
        raise ValueError("ADR 0105 replay set is incomplete")

    scientific_groups = [
        groups[index]["scientific"] for index in range(GROUPS)
    ]
    target_slots = sum(int(group["target_slots"]) for group in scientific_groups)
    target_hits = sum(int(group["target_hits"]) for group in scientific_groups)
    all_replays_identical = all(
        bool(comparisons[index]["scientific_payload_identical"])
        for index in range(GROUPS)
    )
    all_resources_passed = all(
        _resource_report_passed(groups[index]["telemetry"])
        and _resource_report_passed(
            comparisons[index]["replay_telemetry"]
        )
        for index in range(GROUPS)
    )
    all_group_gates_passed = all(
        bool(group["gates"]["group_passed"])
        for group in scientific_groups
    )
    passed = bool(
        all_group_gates_passed
        and all_replays_identical
        and all_resources_passed
    )
    classification = (
        "frozen_optimizer_hyperparameters_insufficient"
        if passed
        else invalid_classification
    )
    scientific = {
        "arm": "combined",
        "classification": classification,
        "groups": scientific_groups,
        "aggregate": {
            "groups": GROUPS,
            "candidates": sum(
                int(group["candidate_count"])
                for group in scientific_groups
            ),
            "target_slots": target_slots,
            "target_hits": target_hits,
            "target_positive_recall": (
                target_hits / max(target_slots, 1)
            ),
            "target_set_exact_fraction": sum(
                int(bool(group["target_set_exact"]))
                for group in scientific_groups
            )
            / GROUPS,
            "r4800_winner_retention": sum(
                int(bool(group["r4800_winner_retained"]))
                for group in scientific_groups
            )
            / GROUPS,
        },
        "maximum_normalization_residual": max(
            (
                group["normalization_residual"]
                for group in scientific_groups
            ),
            key=Decimal,
        ),
        "maximum_kkt_violation": max(
            (group["kkt_violation"] for group in scientific_groups),
            key=Decimal,
        ),
        "maximum_objective_difference": max(
            (group["objective_difference"] for group in scientific_groups),
            key=Decimal,
        ),
        "maximum_offset_difference": max(
            (group["offset_difference"] for group in scientific_groups),
            key=Decimal,
        ),
        "gates": {
            "control_pipeline_passed": passed,
            "all_24_group_gates_passed": all_group_gates_passed,
            "all_24_replays_identical": all_replays_identical,
            "all_origin_and_replay_resources_passed": all_resources_passed,
            "all_24_exact_target_sets": all(
                bool(group["target_set_exact"])
                for group in scientific_groups
            ),
        },
        "origin_telemetry": [
            {
                "group_index": index,
                **groups[index]["telemetry"],
            }
            for index in range(GROUPS)
        ],
        "replay_telemetry": [
            {
                "group_index": index,
                **comparisons[index]["replay_telemetry"],
            }
            for index in range(GROUPS)
        ],
        "duplicate_discovery_fraction": 0.0,
        **_closed_domains(),
    }
    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "scientific": scientific,
        "telemetry": {
            "host": socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": _peak_rss_bytes(),
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    group = subparsers.add_parser("group")
    group.add_argument("--dataset", type=Path, required=True)
    group.add_argument("--cache", type=Path, required=True)
    group.add_argument("--selected-run", type=Path, required=True)
    group.add_argument("--analytic", type=Path, required=True)
    group.add_argument("--group-index", type=int, required=True)
    group.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--group", type=Path, action="append", required=True)
    combine.add_argument(
        "--replay-comparison",
        type=Path,
        action="append",
        required=True,
    )
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "group":
        report = run_decimal_group(
            args.dataset,
            args.cache,
            args.selected_run,
            args.analytic,
            args.group_index,
        )
    else:
        report = combine_group_reports(
            args.group,
            args.replay_comparison,
        )
    _write_json(args.output, report)
    if args.command == "group":
        telemetry = report["telemetry"]
        print(
            json.dumps(
                {
                    "group_index": report["scientific"]["group_index"],
                    "group_passed": report["scientific"]["gates"][
                        "group_passed"
                    ],
                    "resource_qualification_passed": bool(
                        int(telemetry["peak_process_rss_bytes"])
                        <= MEMORY_GATE_BYTES
                        and int(telemetry["process_swaps"]) == 0
                        and telemetry["system_swap_delta_bytes"] is not None
                        and int(telemetry["system_swap_delta_bytes"]) <= 0
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
