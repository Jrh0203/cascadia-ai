#!/usr/bin/env python3
"""Independent maximum-width and score-space audits for ADR 0093."""

from __future__ import annotations

import argparse
import json
import math
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
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_frontier_rank_boundary_train import (
    EXPERIMENT_ID,
    RANK_BOUNDARY_LEARNING_RATE,
    RANK_BOUNDARY_MARGIN,
    RANK_BOUNDARY_SEED,
    RANK_BOUNDARY_TEMPERATURE,
    RANK_BOUNDARY_WEIGHT_DECAY,
    frontier_rank_boundary_loss,
    rank_matched_boundary_loss_from_scores,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import (
    checksum,
    load_frontier_warm_start,
)
from cascadia_mlx.graded_oracle_model import predict_graded_oracle_batch
from frontier_boundary_support import (
    RESIDUAL_LIMIT,
    SCORE_SPACE_GROUPS,
    SCORE_SPACE_LEARNING_RATE,
    SCORE_SPACE_STEPS,
    all_finite,
    optimize_scores,
    selected_target_count,
    target_and_eligible,
    widest_refs,
    write_json_atomic,
)
from graded_oracle_max_width_smoke import (
    system_swap_used_bytes,
    widest_unsealed_group,
)


def run_gradient_audit(
    dataset_roots: list[Path],
    *,
    init_model_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Verify full rank-pair gradient coverage on the widest open group."""
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    batch = decode_graded_oracle_groups(
        dataset.shards[shard_index].bytes(),
        (ref,),
    )
    target, eligible = target_and_eligible(batch)
    non_target = eligible & ~target
    target_count = int(np.sum(target))
    model = load_frontier_warm_start(init_model_dir)
    prediction = predict_graded_oracle_batch(model, batch)
    mx.eval(prediction.scores)
    initial_scores = np.asarray(prediction.scores)
    score_gradient = mx.grad(
        lambda values: rank_matched_boundary_loss_from_scores(
            values,
            mx.array(target),
            mx.array(eligible),
        )
    )(prediction.scores)
    mx.eval(score_gradient)
    gradient_values = np.asarray(score_gradient)
    positive_non_targets = non_target & (gradient_values > 0.0)
    zero_non_targets = non_target & (gradient_values == 0.0)

    mx.random.seed(seed)
    optimizer = optim.AdamW(
        learning_rate=RANK_BOUNDARY_LEARNING_RATE,
        weight_decay=RANK_BOUNDARY_WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, frontier_rank_boundary_loss)
    swap_before = system_swap_used_bytes()
    started = time.perf_counter()
    loss, gradients = loss_and_grad(model, batch)
    gradients_finite = all_finite(gradients)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    optimizer_step_seconds = time.perf_counter() - started
    swap_after = system_swap_used_bytes()
    updated = predict_graded_oracle_batch(model, batch)
    mx.eval(updated.scores)
    updated_scores = np.asarray(updated.scores)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": "maximum-width-rank-boundary-gradient-and-update",
        "host": socket.gethostname(),
        "seed": seed,
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        **identity,
        "candidate_count": ref.candidate_count,
        "target_positive_count": target_count,
        "non_target_count": int(np.sum(non_target)),
        "positive_non_target_gradient_count": int(
            np.sum(positive_non_targets)
        ),
        "zero_non_target_gradient_count": int(np.sum(zero_non_targets)),
        "singleton_overflow_exercised": (
            ref.candidate_count > GRADED_ORACLE_PACKED_ACTION_LIMIT
        ),
        "temperature": RANK_BOUNDARY_TEMPERATURE,
        "margin": RANK_BOUNDARY_MARGIN,
        "target_score_gradients_strictly_negative": bool(
            np.all(gradient_values[target] < 0.0)
        ),
        "matching_hard_negative_count": bool(
            np.sum(positive_non_targets) == target_count
        ),
        "remaining_non_target_gradients_zero": bool(
            np.sum(zero_non_targets) == np.sum(non_target) - target_count
        ),
        "excluded_score_gradients_zero": bool(
            np.all(gradient_values[~eligible] == 0.0)
        ),
        "model_gradients_finite": gradients_finite,
        "initial_scores_finite": bool(np.all(np.isfinite(initial_scores))),
        "updated_scores_finite": bool(np.all(np.isfinite(updated_scores))),
        "scores_changed": bool(not np.array_equal(initial_scores, updated_scores)),
        "loss": float(loss.item()),
        "loss_finite": bool(math.isfinite(float(loss.item()))),
        "optimizer_step_seconds": optimizer_step_seconds,
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": (
            None
            if swap_before is None or swap_after is None
            else swap_after - swap_before
        ),
        "test_split_opened": False,
    }
    report["passed"] = all(
        [
            report["singleton_overflow_exercised"],
            report["target_score_gradients_strictly_negative"],
            report["matching_hard_negative_count"],
            report["remaining_non_target_gradients_zero"],
            report["excluded_score_gradients_zero"],
            report["model_gradients_finite"],
            report["initial_scores_finite"],
            report["updated_scores_finite"],
            report["scores_changed"],
            report["loss_finite"],
            report["peak_process_rss_bytes"] <= 4 * 1024**3,
            report["process_swaps"] == 0,
            report["system_swap_delta_bytes"] is not None,
            report["system_swap_delta_bytes"] <= 0,
        ]
    )
    return report


def run_score_space_convergence(
    dataset_root: Path,
    *,
    init_model_dir: Path,
    maximum_groups: int = SCORE_SPACE_GROUPS,
    steps: int = SCORE_SPACE_STEPS,
    learning_rate: float = SCORE_SPACE_LEARNING_RATE,
) -> dict[str, Any]:
    """Test rank-matched convergence on the widest validation decisions."""
    dataset = GradedOracleDataset(dataset_root, verify_checksums=True)
    if dataset.split != "validation":
        raise ValueError("score-space convergence requires validation")
    model = load_frontier_warm_start(init_model_dir)
    initial_hits = 0
    final_hits = 0
    targets = 0
    initial_exact = 0
    final_exact = 0
    groups: list[dict[str, Any]] = []
    started = time.perf_counter()
    for shard_index, ref in widest_refs(dataset, maximum_groups):
        batch = decode_graded_oracle_groups(
            dataset.shards[shard_index].bytes(),
            (ref,),
        )
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        count = ref.candidate_count
        initial_scores = np.asarray(prediction.scores)[0, :count]
        screen = np.asarray(batch.screen_value)[0, :count]
        flags = np.asarray(batch.source_flags)[0, :count]
        hashes = np.asarray(batch.action_hash)[0, :count]
        target, eligible = target_and_eligible(batch)
        target = target[0, :count]
        eligible = eligible[0, :count]
        quota = int(np.sum(target))
        before = selected_target_count(initial_scores, flags, hashes, target)
        optimized = optimize_scores(
            initial_scores[np.newaxis, :],
            screen[np.newaxis, :],
            target[np.newaxis, :],
            eligible[np.newaxis, :],
            steps=steps,
            learning_rate=learning_rate,
            loss_function=rank_matched_boundary_loss_from_scores,
        )[0]
        after = selected_target_count(optimized, flags, hashes, target)
        initial_hits += before
        final_hits += after
        targets += quota
        initial_exact += int(before == quota)
        final_exact += int(after == quota)
        groups.append(
            {
                "candidate_count": count,
                "target_positive_count": quota,
                "initial_target_hits": before,
                "final_target_hits": after,
                "maximum_absolute_residual": float(
                    np.max(np.abs(optimized - screen))
                ),
            }
        )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    group_count = len(groups)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": "bounded-score-space-rank-boundary-convergence",
        "host": socket.gethostname(),
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "groups": groups,
        "group_count": group_count,
        "steps": steps,
        "learning_rate": learning_rate,
        "residual_limit": RESIDUAL_LIMIT,
        "initial_target_positive_recall": initial_hits / targets,
        "final_target_positive_recall": final_hits / targets,
        "initial_target_set_exact_fraction": initial_exact / group_count,
        "final_target_set_exact_fraction": final_exact / group_count,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "test_split_opened": False,
    }
    report["passed"] = all(
        [
            report["final_target_positive_recall"] >= 0.99,
            report["final_target_set_exact_fraction"] >= 0.90,
            all(
                group["maximum_absolute_residual"] <= RESIDUAL_LIMIT + 1e-5
                for group in groups
            ),
            report["peak_process_rss_bytes"] <= 4 * 1024**3,
            report["process_swaps"] == 0,
        ]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gradient = subparsers.add_parser("gradient-audit")
    gradient.add_argument("--dataset", type=Path, action="append", required=True)
    gradient.add_argument("--init-model-dir", type=Path, required=True)
    gradient.add_argument("--seed", type=int, choices=[RANK_BOUNDARY_SEED], required=True)
    gradient.add_argument("--output", type=Path, required=True)
    convergence = subparsers.add_parser("score-space-convergence")
    convergence.add_argument("--dataset", type=Path, required=True)
    convergence.add_argument("--init-model-dir", type=Path, required=True)
    convergence.add_argument("--maximum-groups", type=int, default=SCORE_SPACE_GROUPS)
    convergence.add_argument("--steps", type=int, default=SCORE_SPACE_STEPS)
    convergence.add_argument(
        "--learning-rate",
        type=float,
        default=SCORE_SPACE_LEARNING_RATE,
    )
    convergence.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gradient-audit":
        report = run_gradient_audit(
            args.dataset,
            init_model_dir=args.init_model_dir,
            seed=args.seed,
        )
    else:
        report = run_score_space_convergence(
            args.dataset,
            init_model_dir=args.init_model_dir,
            maximum_groups=args.maximum_groups,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
