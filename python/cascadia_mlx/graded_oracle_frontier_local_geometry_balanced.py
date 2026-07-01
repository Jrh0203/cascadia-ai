"""ADR 0113 balanced target-membership control."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
)
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    MonotoneAdamW,
    NumericalConvergence,
    _input_identity_summary,
    _report,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankBatch,
    rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    SELECTED_MODEL_BLAKE3,
    _new_selected_model,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    NEURAL_CHECKPOINTS,
    NEURAL_EXPOSURES,
    NEURAL_GROUPS,
    _batch_arrays,
    _closed_domains,
    _load_inputs,
    _score_metrics,
)
from cascadia_mlx.graded_oracle_frontier_local_geometry_adapter import (
    ADAPTER_ARCHITECTURE,
    ADAPTER_HIDDEN_DIM,
    ADAPTER_SEED,
    LocalGeometryAdapterBatch,
    LocalGeometryResidualAdapter,
    _zero_initialized_equality,
    build_adapter_batch,
)

EXPERIMENT_ID = (
    "complete-action-frontier-local-geometry-balanced-target-control-v1"
)
ARM = "local-geometry-balanced-target-group"
FROZEN_ADR0112_BLAKE3 = (
    "0802dcbd57b2273134670c547e069483951303165a00f8d4f5cb5c6ecd4bc12a"
)


@dataclass(frozen=True)
class BalancedTargetBatch:
    """Adapter inputs plus frozen target and eligible masks."""

    adapter: LocalGeometryAdapterBatch
    target_mask: mx.array
    eligible_mask: mx.array


def build_balanced_batch(
    base_model: nn.Module,
    batch: ExpectedRankBatch,
) -> BalancedTargetBatch:
    """Build one frozen rotation and its invariant target labels."""
    adapter = build_adapter_batch(base_model, batch)
    arrays = _batch_arrays(batch)
    count = len(arrays["target"])
    target = np.zeros(np.asarray(batch.candidate_mask).shape, dtype=np.bool_)
    target[0, :count] = arrays["target"]
    frontier = (
        np.asarray(batch.source_flags).astype(np.uint8)
        & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = np.asarray(batch.candidate_mask) & ~frontier
    if not np.any(target & eligible) or not np.any(~target & eligible):
        raise ValueError("balanced target control requires both classes")
    return BalancedTargetBatch(
        adapter=adapter,
        target_mask=mx.array(target),
        eligible_mask=mx.array(eligible),
    )


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(mx.where(mask, values, 0.0)) / mx.maximum(
        mx.sum(weights),
        1.0,
    )


def balanced_target_loss(
    model: LocalGeometryResidualAdapter,
    batch: BalancedTargetBatch,
) -> mx.array:
    """Balanced BCE on target membership using pre-tanh adapter logits."""
    logits = model.correction_logits(
        batch.adapter.local_features,
        batch.adapter.candidate_mask,
    )
    positives = batch.target_mask & batch.eligible_mask
    negatives = ~batch.target_mask & batch.eligible_mask
    positive_loss = _masked_mean(nn.softplus(-logits), positives)
    negative_loss = _masked_mean(nn.softplus(logits), negatives)
    return 0.5 * positive_loss + 0.5 * negative_loss


def evaluate_control(
    model: LocalGeometryResidualAdapter,
    original: ExpectedRankBatch,
    batch: BalancedTargetBatch,
) -> dict[str, Any]:
    """Evaluate the frozen selector and balanced objective."""
    model.eval()
    scores, residuals = model(
        batch.adapter.local_features,
        batch.adapter.candidate_mask,
        batch.adapter.base_residuals,
        batch.adapter.screen_value,
    )
    loss = balanced_target_loss(model, batch)
    mx.eval(scores, residuals, loss)
    count = int(np.sum(np.asarray(batch.adapter.candidate_mask)[0]))
    metrics = _score_metrics(
        original,
        np.asarray(scores)[0, :count].astype(np.float64),
        objective=float(loss.item()),
    )
    metrics["finite_residuals"] = bool(
        np.all(np.isfinite(np.asarray(residuals)[0, :count]))
    )
    metrics["all_scores_finite"] = bool(
        metrics["finite_scores"] and metrics["finite_residuals"]
    )
    metrics["target_set_exact_fraction"] = float(
        bool(metrics.pop("target_set_exact"))
    )
    metrics["r4800_winner_retention"] = float(
        bool(metrics.pop("r4800_winner_retained"))
    )
    metrics["groups"] = 1
    metrics["candidates"] = int(metrics.pop("candidate_count"))
    metrics["mean_objective"] = float(metrics.pop("objective"))
    return metrics


def _file_blake3(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def run_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    evidence_path: Path,
    group_index: int,
) -> dict[str, Any]:
    """Run one balanced-target local-fit control group."""
    if not 0 <= group_index < NEURAL_GROUPS:
        raise ValueError("balanced target group index is outside 0-3")
    evidence = json.loads(evidence_path.read_text())
    if (
        _file_blake3(evidence_path) != FROZEN_ADR0112_BLAKE3
        or evidence["scientific"]["classification"]
        != "parameterized_fit_or_optimizer_insufficient"
    ):
        raise ValueError("ADR 0112 frozen evidence differs")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    _dataset, cohort, batches, identity, config, model_path = _load_inputs(
        dataset_root,
        cache_root,
        selected_run,
    )
    original = batches[group_index]
    base_model = _new_selected_model(config, model_path)
    rotated_originals = [
        rotate_expected_rank_batch(original, rotation)
        for rotation in range(6)
    ]
    rotated = [
        build_balanced_batch(base_model, batch)
        for batch in rotated_originals
    ]
    del base_model
    mx.random.seed(ADAPTER_SEED)
    model = LocalGeometryResidualAdapter()
    mx.eval(model.parameters())
    zero_equality = all(
        _zero_initialized_equality(model, batch.adapter)
        for batch in rotated
    )
    optimizer = MonotoneAdamW()
    loss_and_grad = nn.value_and_grad(model, balanced_target_loss)
    trajectory = [
        {
            "exposures_per_group": 0,
            "optimizer_steps": 0,
            "metrics": evaluate_control(model, original, rotated[0]),
        }
    ]
    checkpoints = set(NEURAL_CHECKPOINTS)
    failure: str | None = None
    numerical_convergence: dict[str, Any] | None = None
    for exposure in range(1, NEURAL_EXPOSURES + 1):
        batch = rotated[(exposure - 1) % 6]
        model.train()
        loss, gradients = loss_and_grad(model, batch)
        try:
            optimizer.step(
                model,
                gradients,
                loss,
                balanced_target_loss,
                batch,
                allow_numerical_convergence=True,
                convergence_improvement_domain="eligible",
            )
        except NumericalConvergence as convergence:
            numerical_convergence = convergence.diagnostics
            accepted = optimizer.summary()["accepted_updates"]
            if trajectory[-1]["exposures_per_group"] != accepted:
                trajectory.append(
                    {
                        "exposures_per_group": accepted,
                        "optimizer_steps": accepted,
                        "metrics": evaluate_control(
                            model,
                            original,
                            rotated[0],
                        ),
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
                    "metrics": evaluate_control(
                        model,
                        original,
                        rotated[0],
                    ),
                }
            )
    optimizer_summary = optimizer.summary()
    accepted = int(optimizer_summary["accepted_updates"])
    if trajectory[-1]["exposures_per_group"] != accepted:
        trajectory.append(
            {
                "exposures_per_group": accepted,
                "optimizer_steps": accepted,
                "metrics": evaluate_control(model, original, rotated[0]),
            }
        )
    final = trajectory[-1]["metrics"]
    completed = bool(
        accepted == NEURAL_EXPOSURES
        or numerical_convergence is not None
    )
    scientific = {
        "arm": ARM,
        "architecture": ADAPTER_ARCHITECTURE,
        "group_index": group_index,
        "group_id": int(cohort[group_index].group_id),
        "input_identity": _input_identity_summary(identity),
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "adr0112_evidence_blake3": FROZEN_ADR0112_BLAKE3,
        "base_model_frozen": True,
        "adapter_seed": ADAPTER_SEED,
        "adapter_hidden_dim": ADAPTER_HIDDEN_DIM,
        "adapter_parameter_count": sum(
            int(value.size)
            for _name, value in tree_flatten(
                model.trainable_parameters()
            )
        ),
        "objective": "balanced-target-membership-bce",
        "exposures": NEURAL_EXPOSURES,
        "trajectory": trajectory,
        "final": final,
        "optimizer": optimizer_summary,
        "failure": failure,
        "numerical_convergence": numerical_convergence,
        "zero_initialized_base_equality": zero_equality,
        "gates": {
            "all_exposures_completed_or_numerically_converged": completed,
            "zero_initialized_base_equality": zero_equality,
            "both_target_classes_present": True,
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
                "accepted_updates": report["scientific"]["optimizer"][
                    "accepted_updates"
                ],
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
