"""Higher-precision projected-control repair for ADR 0104."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import resource
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    ANALYTIC_KKT_GATE,
    OBJECTIVE_MATCH_GATE,
    PROJECTED_INITIAL_STEP,
    PROJECTED_KKT_TOLERANCE,
    _batch_arrays,
    _closed_domains,
    _load_inputs,
    _score_metrics,
    projected_optimize_expected_rank,
    solve_box_constrained_expected_rank,
)

EXPERIMENT_ID = "complete-action-frontier-projected-control-repair-v1"
SOURCE_EXPERIMENT_ID = "complete-action-frontier-free-residual-audit-v1"
GROUPS = 24
SHARDS = 4
GROUPS_PER_SHARD = 6
MAXIMUM_WORKERS = 6
MAXIMUM_ITERATIONS = 100_000


def shard_group_indices(shard_index: int) -> tuple[int, ...]:
    """Return one frozen contiguous six-group repair shard."""
    if not 0 <= shard_index < SHARDS:
        raise ValueError("repair shard index is outside 0-3")
    start = shard_index * GROUPS_PER_SHARD
    return tuple(range(start, start + GROUPS_PER_SHARD))


def _worker_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _optimize_group(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    projected = projected_optimize_expected_rank(
        payload["selected_scores"],
        payload["screen"],
        payload["ranks"],
        payload["rank_mask"],
        payload["eligible"],
        initial_step=PROJECTED_INITIAL_STEP,
        tolerance=PROJECTED_KKT_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
    )
    analytic = solve_box_constrained_expected_rank(
        payload["screen"],
        payload["ranks"],
        payload["rank_mask"],
        payload["eligible"],
    )
    retained = frontier_anchored_retained_indices(
        scores=projected["scores"],
        source_flags=payload["flags"],
        action_hashes=payload["hashes"],
    )
    retained_nonfrontier = retained[
        (payload["flags"][retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    ]
    analytic_retained = frontier_anchored_retained_indices(
        scores=analytic["scores"],
        source_flags=payload["flags"],
        action_hashes=payload["hashes"],
    )
    analytic_nonfrontier = analytic_retained[
        (
            payload["flags"][analytic_retained]
            & GRADED_SOURCE_CHAMPION_FRONTIER
        )
        == 0
    ]
    target_slots = int(np.sum(payload["target"]))
    target_hits = int(np.sum(payload["target"][retained_nonfrontier]))
    analytic_hits = int(np.sum(payload["target"][analytic_nonfrontier]))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "group_index": int(payload["group_index"]),
        "group_id": int(payload["group_id"]),
        "phase": int(payload["phase"]),
        "candidate_count": len(payload["screen"]),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact": target_hits == target_slots,
        "r4800_winner_retained": int(payload["winner"]) in retained,
        "objective": float(projected["objective"]),
        "analytic_objective": float(analytic["objective"]),
        "objective_gap_from_analytic": (
            float(projected["objective"]) - float(analytic["objective"])
        ),
        "analytic_target_hits": analytic_hits,
        "selection_matches_analytic": bool(
            target_hits == analytic_hits
            and (target_hits == target_slots) == (analytic_hits == target_slots)
        ),
        "converged": bool(projected["converged"]),
        "iterations": int(projected["iterations"]),
        "kkt_violation": float(projected["kkt_violation"]),
        "trajectory": projected["trajectory"],
        "finite_scores": bool(np.all(np.isfinite(projected["scores"]))),
        "elapsed_seconds": time.perf_counter() - started,
        "worker_peak_rss_bytes": _worker_peak_rss_bytes(),
        "worker_process_swaps": int(getattr(usage, "ru_nswap", 0)),
    }


def _aggregate_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    target_slots = sum(int(group["target_slots"]) for group in groups)
    target_hits = sum(int(group["target_hits"]) for group in groups)
    return {
        "groups": len(groups),
        "candidates": sum(int(group["candidate_count"]) for group in groups),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact_fraction": sum(
            int(bool(group["target_set_exact"])) for group in groups
        )
        / len(groups),
        "r4800_winner_retention": sum(
            int(bool(group["r4800_winner_retained"])) for group in groups
        )
        / len(groups),
        "mean_objective": sum(float(group["objective"]) for group in groups)
        / len(groups),
        "all_scores_finite": all(bool(group["finite_scores"]) for group in groups),
    }


def _report(
    scientific: dict[str, Any],
    started: float,
    swap_before: int | None,
    worker_count: int,
    worker_resources: list[dict[str, Any]],
) -> dict[str, Any]:
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
            "worker_count": worker_count,
            "peak_process_rss_bytes": peak_rss,
            "maximum_worker_rss_bytes": max(
                int(worker["peak_rss_bytes"]) for worker in worker_resources
            ),
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "worker_process_swaps": sum(
                int(worker["process_swaps"]) for worker in worker_resources
            ),
            "workers": worker_resources,
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": (
                None
                if swap_before is None or swap_after is None
                else swap_after - swap_before
            ),
        },
    }


def run_repair_shard(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    shard_index: int,
) -> dict[str, Any]:
    """Run one six-group parallel precision-repair shard."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    from cascadia_mlx.graded_oracle_frontier_fit_interference import (
        _new_selected_model,
    )

    selected = _new_selected_model(config, model_path)
    payloads = []
    for group_index in shard_group_indices(shard_index):
        batch = batches[group_index]
        arrays = _batch_arrays(batch, model=selected)
        baseline = _score_metrics(
            batch,
            arrays["screen"] + arrays["selected_residual"],
            objective=0.0,
        )
        payloads.append(
            {
                "group_index": group_index,
                "group_id": cohort[group_index].group_id,
                "phase": baseline["phase"],
                "winner": int(np.asarray(batch.selected_index)[0]),
                "selected_scores": arrays["screen"]
                + arrays["selected_residual"],
                **arrays,
            }
        )
    worker_count = min(MAXIMUM_WORKERS, len(payloads))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        raw_groups = list(executor.map(_optimize_group, payloads))
    raw_groups.sort(key=lambda group: int(group["group_index"]))
    worker_resources = [
        {
            "group_index": int(group["group_index"]),
            "elapsed_seconds": float(group["elapsed_seconds"]),
            "peak_rss_bytes": int(group["worker_peak_rss_bytes"]),
            "process_swaps": int(group["worker_process_swaps"]),
        }
        for group in raw_groups
    ]
    groups = [
        {
            key: value
            for key, value in group.items()
            if key
            not in {
                "elapsed_seconds",
                "worker_peak_rss_bytes",
                "worker_process_swaps",
            }
        }
        for group in raw_groups
    ]
    aggregate = _aggregate_groups(groups)
    maximum_kkt = max(float(group["kkt_violation"]) for group in groups)
    maximum_gap = max(
        abs(float(group["objective_gap_from_analytic"])) for group in groups
    )
    scientific = {
        "arm": "projected-control-repair-shard",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "shard_index": shard_index,
        "group_indices": list(shard_group_indices(shard_index)),
        "input_identity": identity,
        "full_cohort_digest_blake3": identity["cohort_digest_blake3"],
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "groups": groups,
        "aggregate": aggregate,
        "maximum_kkt_violation": maximum_kkt,
        "maximum_objective_gap_from_analytic": maximum_gap,
        "gates": {
            "repair_shard_passed": bool(
                all(bool(group["converged"]) for group in groups)
                and all(bool(group["selection_matches_analytic"]) for group in groups)
                and maximum_kkt <= ANALYTIC_KKT_GATE
                and maximum_gap <= OBJECTIVE_MATCH_GATE
            ),
            "all_repair_groups_completed": len(groups) == GROUPS_PER_SHARD,
            "all_repair_scores_finite": bool(aggregate["all_scores_finite"]),
            "all_workers_under_memory_limit": all(
                int(worker["peak_rss_bytes"]) <= 4 * 1024**3
                for worker in worker_resources
            ),
            "all_workers_zero_swaps": all(
                int(worker["process_swaps"]) == 0 for worker in worker_resources
            ),
        },
        **_closed_domains(),
    }
    return _report(
        scientific,
        started,
        swap_before,
        worker_count,
        worker_resources,
    )


def _shard_pipeline_passed(report: dict[str, Any]) -> bool:
    scientific = report["scientific"]
    telemetry = report["telemetry"]
    return bool(
        scientific["gates"]["repair_shard_passed"]
        and all(
            bool(value)
            for name, value in scientific["gates"].items()
            if name.startswith("all_")
        )
        and scientific["test_split_opened"] is False
        and scientific["gameplay_opened"] is False
        and scientific["new_teacher_compute_used"] is False
        and scientific["external_compute_used"] is False
        and int(telemetry["peak_process_rss_bytes"]) <= 4 * 1024**3
        and int(telemetry["maximum_worker_rss_bytes"]) <= 4 * 1024**3
        and int(telemetry["process_swaps"]) == 0
        and int(telemetry["worker_process_swaps"]) == 0
        and telemetry["system_swap_delta_bytes"] is not None
        and int(telemetry["system_swap_delta_bytes"]) <= 0
    )


def combine_reports(paths: list[Path]) -> dict[str, Any]:
    """Combine four disjoint repair shards and classify the repair."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    shards: dict[int, dict[str, Any]] = {}
    cohort_digests: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected ADR 0104 report: {path}")
        scientific = report["scientific"]
        shard_index = int(scientific["shard_index"])
        if shard_index in shards:
            raise ValueError(f"duplicate ADR 0104 shard: {shard_index}")
        shards[shard_index] = report
        cohort_digests.add(str(scientific["full_cohort_digest_blake3"]))
    if set(shards) != set(range(SHARDS)) or len(cohort_digests) != 1:
        raise ValueError("ADR 0104 shard set or cohort identity is incomplete")
    groups = [
        group
        for shard_index in range(SHARDS)
        for group in shards[shard_index]["scientific"]["groups"]
    ]
    if [int(group["group_index"]) for group in groups] != list(range(GROUPS)):
        raise ValueError("ADR 0104 groups are not complete and ordered")
    aggregate = _aggregate_groups(groups)
    maximum_kkt = max(float(group["kkt_violation"]) for group in groups)
    maximum_gap = max(
        abs(float(group["objective_gap_from_analytic"])) for group in groups
    )
    passed = all(_shard_pipeline_passed(report) for report in shards.values())
    classification = (
        "frozen_optimizer_hyperparameters_insufficient"
        if passed
        else "projected_control_repair_invalid"
    )
    scientific = {
        "arm": "combined",
        "classification": classification,
        "full_cohort_digest_blake3": next(iter(cohort_digests)),
        "aggregate": aggregate,
        "maximum_kkt_violation": maximum_kkt,
        "maximum_objective_gap_from_analytic": maximum_gap,
        "gates": {
            "repair_pipeline_passed": passed,
            "all_24_groups_converged": all(
                bool(group["converged"]) for group in groups
            ),
            "all_24_selections_match_analytic": all(
                bool(group["selection_matches_analytic"]) for group in groups
            ),
            "kkt_gate_passed": maximum_kkt <= ANALYTIC_KKT_GATE,
            "objective_gap_gate_passed": maximum_gap <= OBJECTIVE_MATCH_GATE,
        },
        "shard_telemetry": [
            {
                "shard_index": shard_index,
                **shards[shard_index]["telemetry"],
            }
            for shard_index in range(SHARDS)
        ],
        "duplicate_discovery_fraction": 0.0,
        **_closed_domains(),
    }
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--dataset", type=Path, required=True)
    shard.add_argument("--cache", type=Path, required=True)
    shard.add_argument("--selected-run", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--shard", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        report = run_repair_shard(
            args.dataset,
            args.cache,
            args.selected_run,
            args.shard_index,
        )
    else:
        report = combine_reports(args.shard)
    _write_json(args.output, report)


if __name__ == "__main__":
    main()
