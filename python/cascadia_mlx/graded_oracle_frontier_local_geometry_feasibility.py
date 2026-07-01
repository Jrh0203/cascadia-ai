"""ADR 0112 static bounded-range and observable-alias forensic."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _input_identity_summary,
    _report,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    SELECTED_MODEL_BLAKE3,
    _new_selected_model,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    NEURAL_GROUPS,
    _batch_arrays,
    _closed_domains,
    _load_inputs,
    _score_metrics,
)
from cascadia_mlx.graded_oracle_frontier_local_geometry_adapter import (
    build_adapter_batch,
)
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_CORRECTION_RANGE,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
)

EXPERIMENT_ID = (
    "complete-action-frontier-local-geometry-feasibility-forensic-v1"
)
ARM = "local-geometry-feasibility-group"
FROZEN_ADR0111_BLAKE3 = (
    "6e21675a7c05ac815368f1cc02c2b6769bfddbd57571d21efaa695fa2752e15f"
)
BOUND_TOLERANCE = 1e-6


def _file_blake3(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _equivalence_summary(
    features: np.ndarray,
    targets: np.ndarray,
    base_scores: np.ndarray,
) -> dict[str, Any]:
    classes: dict[bytes, list[int]] = {}
    for index, row in enumerate(features):
        payload = np.ascontiguousarray(row).tobytes()
        classes.setdefault(payload, []).append(index)
    duplicate = [
        indices for indices in classes.values() if len(indices) > 1
    ]
    mixed = [
        indices
        for indices in duplicate
        if bool(np.any(targets[indices]))
        and bool(np.any(~targets[indices]))
    ]
    exact_tie_conflicts = [
        indices
        for indices in mixed
        if len({float(base_scores[index]) for index in indices}) == 1
    ]
    return {
        "classes": len(classes),
        "duplicate_classes": len(duplicate),
        "duplicate_candidates": sum(len(indices) for indices in duplicate),
        "maximum_class_size": max(
            (len(indices) for indices in classes.values()),
            default=0,
        ),
        "mixed_target_classes": len(mixed),
        "mixed_target_candidates": sum(len(indices) for indices in mixed),
        "exact_score_tie_conflict_classes": len(exact_tie_conflicts),
        "row_identity_blake3": blake3.blake3(
            b"".join(
                len(payload).to_bytes(8, "little") + payload
                for payload in sorted(classes)
            )
        ).hexdigest(),
    }


def run_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    evidence_path: Path,
    group_index: int,
) -> dict[str, Any]:
    """Audit one frozen group without gradients or parameter updates."""
    if not 0 <= group_index < NEURAL_GROUPS:
        raise ValueError("feasibility group index is outside 0-3")
    if (
        _file_blake3(evidence_path) != FROZEN_ADR0111_BLAKE3
        or json.loads(evidence_path.read_text())["scientific"][
            "classification"
        ]
        != "calibrated_local_geometry_insufficient"
    ):
        raise ValueError("ADR 0111 frozen evidence differs")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    batch = batches[group_index]
    base_model = _new_selected_model(config, model_path)
    adapter_batch = build_adapter_batch(base_model, batch)
    count = int(np.sum(np.asarray(batch.candidate_mask)[0]))
    features = np.asarray(adapter_batch.local_features)[0, :count]
    base_residuals = np.asarray(
        adapter_batch.base_residuals
    )[0, :count].astype(np.float64)
    screen = np.asarray(batch.screen_value)[0, :count].astype(np.float64)
    base_scores = screen + base_residuals
    arrays = _batch_arrays(batch)
    targets = arrays["target"].astype(np.bool_)
    flags = arrays["flags"]
    hashes = arrays["hashes"]

    lower_residuals = np.clip(
        base_residuals - LOCAL_GEOMETRY_CORRECTION_RANGE,
        -GRADED_ORACLE_RESIDUAL_RANGE,
        GRADED_ORACLE_RESIDUAL_RANGE,
    )
    upper_residuals = np.clip(
        base_residuals + LOCAL_GEOMETRY_CORRECTION_RANGE,
        -GRADED_ORACLE_RESIDUAL_RANGE,
        GRADED_ORACLE_RESIDUAL_RANGE,
    )
    lower_scores = screen + lower_residuals
    upper_scores = screen + upper_residuals
    interval_scores = np.where(targets, upper_scores, lower_scores)
    selected_retained = frontier_anchored_retained_indices(
        scores=base_scores,
        source_flags=flags,
        action_hashes=hashes,
    )
    selected_nonfrontier = selected_retained[
        (
            flags[selected_retained]
            & GRADED_SOURCE_CHAMPION_FRONTIER
        )
        == 0
    ]
    missed_targets = targets.copy()
    missed_targets[selected_nonfrontier] = False

    base_metrics = _score_metrics(
        batch,
        base_scores,
        objective=0.0,
    )
    interval_metrics = _score_metrics(
        batch,
        interval_scores,
        objective=0.0,
    )
    equivalence = _equivalence_summary(
        features,
        targets,
        base_scores,
    )
    finite = bool(
        np.all(np.isfinite(features))
        and np.all(np.isfinite(base_residuals))
        and np.all(np.isfinite(lower_scores))
        and np.all(np.isfinite(upper_scores))
        and np.all(np.isfinite(interval_scores))
    )
    ordered_intervals = bool(
        np.all(lower_scores <= upper_scores + BOUND_TOLERANCE)
    )
    scientific = {
        "arm": ARM,
        "group_index": group_index,
        "group_id": int(cohort[group_index].group_id),
        "input_identity": _input_identity_summary(identity),
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "adr0111_evidence_blake3": FROZEN_ADR0111_BLAKE3,
        "candidate_count": count,
        "target_count": int(np.sum(targets)),
        "selected_base": base_metrics,
        "candidate_independent_interval_ceiling": interval_metrics,
        "equivalence_classes": equivalence,
        "residual_saturation": {
            "at_lower_bound": int(
                np.sum(
                    base_residuals
                    <= -GRADED_ORACLE_RESIDUAL_RANGE + BOUND_TOLERANCE
                )
            ),
            "at_upper_bound": int(
                np.sum(
                    base_residuals
                    >= GRADED_ORACLE_RESIDUAL_RANGE - BOUND_TOLERANCE
                )
            ),
        },
        "missed_target_margin": {
            "count": int(np.sum(missed_targets)),
            "mean_upward_reach": (
                float(
                    np.mean(
                        upper_scores[missed_targets]
                        - base_scores[missed_targets]
                    )
                )
                if np.any(missed_targets)
                else None
            ),
            "minimum_upward_reach": (
                float(
                    np.min(
                        upper_scores[missed_targets]
                        - base_scores[missed_targets]
                    )
                )
                if np.any(missed_targets)
                else None
            ),
        },
        "training_used": False,
        "gradients_used": False,
        "optimizer_updates_used": False,
        "gates": {
            "all_values_finite": finite,
            "all_intervals_ordered": ordered_intervals,
            "selector_accounting_exact": bool(
                base_metrics["target_slots"] == int(np.sum(targets))
                and interval_metrics["target_slots"]
                == int(np.sum(targets))
            ),
            "frozen_evidence_matched": True,
            "training_gradients_optimizer_unused": True,
        },
        **_closed_domains(),
    }
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("group",))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_group(
        args.dataset,
        args.cache,
        args.selected_run,
        args.analytic,
        args.group_index,
    )
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "group_index": args.group_index,
                "interval_recall": report["scientific"][
                    "candidate_independent_interval_ceiling"
                ]["target_positive_recall"],
                "resource_qualification_passed": bool(
                    all(report["scientific"]["gates"].values())
                    and report["telemetry"]["peak_process_rss_bytes"]
                    <= 4 * 1024**3
                    and report["telemetry"]["process_swaps"] == 0
                    and report["telemetry"][
                        "system_swap_delta_bytes"
                    ]
                    is not None
                    and report["telemetry"][
                        "system_swap_delta_bytes"
                    ]
                    <= 0
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
