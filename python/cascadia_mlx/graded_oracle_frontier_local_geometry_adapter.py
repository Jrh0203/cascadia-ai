"""ADR 0111 calibrated frozen-base local-geometry adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.dataset import GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import GRADED_ORACLE_PRIOR_DIM
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    MonotoneAdamW,
    NumericalConvergence,
    _input_identity_summary,
    _report,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankBatch,
    expected_rank_loss_from_scores,
    rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
)
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    SELECTED_MODEL_BLAKE3,
    _new_selected_model,
)
from cascadia_mlx.graded_oracle_frontier_free_residual import (
    NEURAL_CHECKPOINTS,
    NEURAL_EXPOSURES,
    NEURAL_GROUPS,
    _closed_domains,
    _load_inputs,
    _score_metrics,
)
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_CANONICAL_ACTION_DIM,
    LOCAL_GEOMETRY_CONTEXT_DIM,
    LOCAL_GEOMETRY_CORRECTION_RANGE,
    candidate_local_geometry,
    canonical_local_action_features,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    predict_graded_oracle_batch,
)
from cascadia_mlx.model import _masked_pool

EXPERIMENT_ID = (
    "complete-action-frontier-calibrated-local-geometry-adapter-v1"
)
ARM = "calibrated-local-geometry-adapter-group"
ADAPTER_ARCHITECTURE = "frozen-selected-local-geometry-adapter-v1"
ADAPTER_SEED = 2026061642
ADAPTER_HIDDEN_DIM = 192
LOCAL_INPUT_DIM = (
    LOCAL_GEOMETRY_CONTEXT_DIM
    + LOCAL_GEOMETRY_CANONICAL_ACTION_DIM
    + GRADED_ORACLE_PRIOR_DIM
    + GLOBAL_DIM
)


@dataclass(frozen=True)
class LocalGeometryAdapterBatch:
    """Precomputed public observables and frozen-base prediction."""

    local_features: mx.array
    candidate_mask: mx.array
    base_residuals: mx.array
    screen_value: mx.array
    expected_rank: mx.array
    expected_rank_mask: mx.array
    source_flags: mx.array


class LocalGeometryResidualAdapter(nn.Module):
    """Trainable ADR 0088 local path over immutable selected-model scores."""

    def __init__(self, hidden_dim: int = ADAPTER_HIDDEN_DIM):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("adapter hidden dimension must be positive")
        self.hidden_dim = hidden_dim
        self.local_projection = nn.Sequential(
            nn.Linear(LOCAL_INPUT_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.local_output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.local_residual_head = nn.Linear(hidden_dim, 1)
        self.local_residual_head.weight = mx.zeros_like(
            self.local_residual_head.weight
        )
        self.local_residual_head.bias = mx.zeros_like(
            self.local_residual_head.bias
        )

    def __call__(
        self,
        local_features: mx.array,
        candidate_mask: mx.array,
        base_residuals: mx.array,
        screen_value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        logits = self.correction_logits(
            local_features,
            candidate_mask,
        )
        correction = (
            LOCAL_GEOMETRY_CORRECTION_RANGE
            * mx.tanh(logits)
            * candidate_mask
        )
        residuals = mx.clip(
            base_residuals + correction,
            -GRADED_ORACLE_RESIDUAL_RANGE,
            GRADED_ORACLE_RESIDUAL_RANGE,
        )
        residuals = residuals * candidate_mask
        return screen_value + residuals, residuals

    def correction_logits(
        self,
        local_features: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        """Return the pre-tanh scalar correction logits."""
        local = self.local_projection(local_features)
        local = local * candidate_mask[..., None]
        pooled = _masked_pool(local, candidate_mask)
        hidden = self.hidden_dim
        mean = mx.broadcast_to(pooled[:, None, :hidden], local.shape)
        maximum = mx.broadcast_to(pooled[:, None, hidden:], local.shape)
        output = self.local_output(
            mx.concatenate([local, mean, maximum, local - mean], axis=-1)
        )
        return (
            self.local_residual_head(output).reshape(
                candidate_mask.shape
            )
            * candidate_mask
        )


def build_adapter_batch(
    base_model: nn.Module,
    batch: ExpectedRankBatch,
) -> LocalGeometryAdapterBatch:
    """Precompute one rotation without putting the base in the gradient tree."""
    base_model.eval()
    base = predict_graded_oracle_batch(base_model, batch)
    groups, candidates = batch.candidate_mask.shape
    repeated_global = mx.broadcast_to(
        batch.global_features[:, None, :],
        (groups, candidates, GLOBAL_DIM),
    )
    local_features = mx.concatenate(
        [
            candidate_local_geometry(
                batch.board_entities,
                batch.board_mask,
                batch.action_features,
                batch.candidate_mask,
            ),
            canonical_local_action_features(batch.action_features),
            batch.prior_features,
            repeated_global,
        ],
        axis=-1,
    )
    local_features = local_features * batch.candidate_mask[..., None]
    mx.eval(local_features, base.residuals)
    if local_features.shape[-1] != LOCAL_INPUT_DIM:
        raise AssertionError("local-geometry adapter input dimension drifted")
    return LocalGeometryAdapterBatch(
        local_features=local_features,
        candidate_mask=batch.candidate_mask,
        base_residuals=base.residuals,
        screen_value=batch.screen_value,
        expected_rank=batch.expected_rank,
        expected_rank_mask=batch.expected_rank_mask,
        source_flags=batch.source_flags,
    )


def local_geometry_adapter_loss(
    model: LocalGeometryResidualAdapter,
    batch: LocalGeometryAdapterBatch,
) -> mx.array:
    """Apply the frozen scale-16 expected-rank objective to adapter scores."""
    scores, _residuals = model(
        batch.local_features,
        batch.candidate_mask,
        batch.base_residuals,
        batch.screen_value,
    )
    frontier = (
        batch.source_flags.astype(mx.int32)
        & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = batch.candidate_mask & ~frontier
    return expected_rank_loss_from_scores(
        scores,
        batch.expected_rank,
        batch.expected_rank_mask,
        eligible,
        target_scale=TARGET_SCALE,
        student_temperature=STUDENT_TEMPERATURE,
    )


def evaluate_adapter(
    model: LocalGeometryResidualAdapter,
    original_batch: ExpectedRankBatch,
    adapter_batch: LocalGeometryAdapterBatch,
) -> dict[str, Any]:
    """Evaluate one group with the deployment selector and frozen objective."""
    model.eval()
    scores, residuals = model(
        adapter_batch.local_features,
        adapter_batch.candidate_mask,
        adapter_batch.base_residuals,
        adapter_batch.screen_value,
    )
    objective = local_geometry_adapter_loss(model, adapter_batch)
    mx.eval(scores, residuals, objective)
    count = int(np.sum(np.asarray(adapter_batch.candidate_mask)[0]))
    score_values = np.asarray(scores)[0, :count].astype(np.float64)
    residual_values = np.asarray(residuals)[0, :count]
    metrics = _score_metrics(
        original_batch,
        score_values,
        objective=float(objective.item()),
    )
    metrics["finite_residuals"] = bool(
        np.all(np.isfinite(residual_values))
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


def _zero_initialized_equality(
    model: LocalGeometryResidualAdapter,
    batch: LocalGeometryAdapterBatch,
) -> bool:
    scores, residuals = model(
        batch.local_features,
        batch.candidate_mask,
        batch.base_residuals,
        batch.screen_value,
    )
    expected_scores = batch.screen_value + batch.base_residuals
    mx.eval(scores, residuals, expected_scores)
    return bool(
        np.array_equal(np.asarray(scores), np.asarray(expected_scores))
        and np.array_equal(
            np.asarray(residuals),
            np.asarray(batch.base_residuals),
        )
    )


def run_group(
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
    group_index: int,
) -> dict[str, Any]:
    """Fit one independently scheduled frozen-base adapter group."""
    if not 0 <= group_index < NEURAL_GROUPS:
        raise ValueError("adapter group index is outside 0-3")
    started = time.perf_counter()
    from cascadia_mlx.graded_oracle_frontier_anchor import (
        _system_swap_used_bytes,
    )

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
        build_adapter_batch(base_model, batch)
        for batch in rotated_originals
    ]
    del base_model
    mx.random.seed(ADAPTER_SEED)
    model = LocalGeometryResidualAdapter()
    mx.eval(model.parameters())
    zero_equality = all(
        _zero_initialized_equality(model, batch)
        for batch in rotated
    )
    optimizer = MonotoneAdamW()
    loss_and_grad = nn.value_and_grad(model, local_geometry_adapter_loss)
    trajectory = [
        {
            "exposures_per_group": 0,
            "optimizer_steps": 0,
            "metrics": evaluate_adapter(model, original, rotated[0]),
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
                local_geometry_adapter_loss,
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
                        "metrics": evaluate_adapter(
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
                    "metrics": evaluate_adapter(
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
                "metrics": evaluate_adapter(model, original, rotated[0]),
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
        "base_model_frozen": True,
        "adapter_seed": ADAPTER_SEED,
        "adapter_hidden_dim": ADAPTER_HIDDEN_DIM,
        "adapter_parameter_count": sum(
            int(value.size)
            for _name, value in tree_flatten(
                model.trainable_parameters()
            )
        ),
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
    parser.add_argument("--analytic", type=Path)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_group(
        args.dataset,
        args.cache,
        args.selected_run,
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
                    report["scientific"]["gates"][
                        "all_scores_finite"
                    ]
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
