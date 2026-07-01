"""Cache, optimization, and baseline diagnostics for ADR 0100."""

from __future__ import annotations

import argparse
import json
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
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import decode_graded_oracle_groups
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    EXPERIMENT_ID,
    ExpectedRankBatch,
    ExpectedRankDataset,
    build_expected_rank_cache,
    build_expected_rank_target_mask,
    compare_expected_rank_caches,
    evaluate_expected_rank_screen_baseline,
    frontier_expected_rank_loss,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_train import (
    EXPECTED_RANK_LEARNING_RATE,
    EXPECTED_RANK_SEED,
    EXPECTED_RANK_WEIGHT_DECAY,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

GRADIENT_AUDIT_STEPS = 32


def run_cache_build(
    *,
    dataset: Path,
    cache: Path,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Build one cache and wrap deterministic science with host telemetry."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    manifest = build_expected_rank_cache(
        dataset,
        cache,
        workers=workers,
        overwrite=overwrite,
    )
    return _report(
        {
            "kind": "target-cache-build",
            "cache_manifest": manifest,
            "workers": workers,
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
        },
        started,
        swap_before,
    )


def run_cache_compare(
    *,
    left_train: Path,
    right_train: Path,
    left_validation: Path,
    right_validation: Path,
) -> dict[str, Any]:
    """Compare independently generated train and validation cache pairs."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    scientific = {
        "kind": "target-cache-comparison",
        "train": compare_expected_rank_caches(left_train, right_train),
        "validation": compare_expected_rank_caches(
            left_validation,
            right_validation,
        ),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    scientific["passed"] = all(
        values["scientific_payload_identical"]
        and values["all_file_bytes_identical"]
        for values in (scientific["train"], scientific["validation"])
    )
    return _report(scientific, started, swap_before)


def run_gradient_audit(
    *,
    dataset_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Prove finite nonzero gradients and bounded single-group optimization."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = ExpectedRankDataset(dataset_root, cache_root)
    if dataset.split != "train":
        raise ValueError("expected-rank gradient audit requires train split")
    batch = _widest_expected_rank_batch(dataset)
    mx.random.seed(EXPECTED_RANK_SEED)
    model = GradedOracleRanker()
    optimizer = optim.AdamW(
        learning_rate=EXPECTED_RANK_LEARNING_RATE,
        weight_decay=EXPECTED_RANK_WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, frontier_expected_rank_loss)

    initial_loss, initial_gradient = loss_and_grad(model, batch)
    mx.eval(initial_loss, initial_gradient)
    initial_metrics = _batch_metrics(model, batch)
    whole_gradient_norm = _gradient_norm(initial_gradient)
    residual_gradient_norm = _gradient_norm(
        {
            name: value
            for name, value in tree_flatten(initial_gradient)
            if "residual_head" in name
        }
    )

    losses: list[float] = []
    after_one_loss: float | None = None
    for step in range(GRADIENT_AUDIT_STEPS):
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if step == 0:
            observed = frontier_expected_rank_loss(model, batch)
            mx.eval(observed)
            after_one_loss = float(observed.item())
    final_loss = frontier_expected_rank_loss(model, batch)
    mx.eval(final_loss)
    final_metrics = _batch_metrics(model, batch)
    scientific = {
        "kind": "optimization-gradient-audit",
        "seed": EXPECTED_RANK_SEED,
        "steps": GRADIENT_AUDIT_STEPS,
        "group_id": int(np.asarray(batch.group_id)[0]),
        "candidate_count": int(np.sum(np.asarray(batch.candidate_mask)[0])),
        "target_candidates": int(
            np.sum(np.asarray(batch.expected_rank_mask)[0])
        ),
        "initial_loss": float(initial_loss.item()),
        "after_one_step_loss": after_one_loss,
        "final_loss": float(final_loss.item()),
        "minimum_training_step_loss": min(losses),
        "whole_model_gradient_norm": whole_gradient_norm,
        "residual_head_gradient_norm": residual_gradient_norm,
        "initial": initial_metrics,
        "final": final_metrics,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report = _report(scientific, started, swap_before)
    telemetry = report["telemetry"]
    report["scientific"]["gates"] = {
        "initial_loss_finite": bool(np.isfinite(scientific["initial_loss"])),
        "after_one_step_loss_finite": bool(
            after_one_loss is not None and np.isfinite(after_one_loss)
        ),
        "final_loss_finite": bool(np.isfinite(scientific["final_loss"])),
        "whole_gradient_finite_and_nonzero": (
            np.isfinite(whole_gradient_norm) and whole_gradient_norm > 0.0
        ),
        "residual_gradient_finite_and_nonzero": (
            np.isfinite(residual_gradient_norm) and residual_gradient_norm > 0.0
        ),
        "loss_decreased_strictly": (
            scientific["final_loss"] < scientific["initial_loss"]
        ),
        "peak_rss_at_most_4_gib": (
            int(telemetry["peak_process_rss_bytes"]) <= 4 * 1024**3
        ),
        "process_swaps_zero": int(telemetry["process_swaps"]) == 0,
        "system_swap_not_consumed": (
            telemetry["system_swap_delta_bytes"] is not None
            and int(telemetry["system_swap_delta_bytes"]) <= 0
        ),
    }
    report["scientific"]["passed"] = all(
        report["scientific"]["gates"].values()
    )
    return report


def run_baseline_audit(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    train_cache: Path,
    validation_cache: Path,
) -> dict[str, Any]:
    """Measure expected-rank target novelty and screen error anatomy."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    train = ExpectedRankDataset(train_dataset, train_cache)
    validation = ExpectedRankDataset(validation_dataset, validation_cache)
    scientific = {
        "kind": "baseline-error-anatomy",
        "train": evaluate_expected_rank_screen_baseline(train),
        "validation": evaluate_expected_rank_screen_baseline(validation),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _report(scientific, started, swap_before)


def _widest_expected_rank_batch(
    dataset: ExpectedRankDataset,
) -> ExpectedRankBatch:
    selected: tuple[int, int, Any] | None = None
    for shard_index, shard in enumerate(dataset.shards):
        for ref in shard.groups:
            candidate = (ref.candidate_count, -shard_index, ref)
            if selected is None or candidate[:2] > selected[:2]:
                selected = candidate
    if selected is None:
        raise ValueError("expected-rank train dataset is empty")
    _count, negative_shard_index, ref = selected
    shard = dataset.shards[-negative_shard_index]
    base = decode_graded_oracle_groups(shard.bytes(), (ref,))
    ranks, mask = dataset.cache.ranks_for_batch(base)
    return ExpectedRankBatch(
        base,
        ranks,
        mask,
        dataset.target_scale,
        dataset.student_temperature,
    )


def _batch_metrics(
    model: GradedOracleRanker,
    batch: ExpectedRankBatch,
) -> dict[str, Any]:
    prediction = predict_graded_oracle_batch(model, batch)
    mx.eval(prediction.scores, prediction.residuals)
    scores = np.asarray(prediction.scores)
    residuals = np.asarray(prediction.residuals)
    masks = np.asarray(batch.candidate_mask)
    flags = np.asarray(batch.source_flags)
    hashes = np.asarray(batch.action_hash)
    ranks = np.asarray(batch.expected_rank)
    rank_masks = np.asarray(batch.expected_rank_mask)
    targets = build_expected_rank_target_mask(
        expected_rank=ranks,
        expected_rank_mask=rank_masks,
        source_flags=flags,
        candidate_mask=masks,
        action_hashes=hashes,
    )
    count = int(np.sum(masks[0]))
    retained = frontier_anchored_retained_indices(
        scores=scores[0, :count],
        source_flags=flags[0, :count],
        action_hashes=hashes[0, :count],
    )
    retained_nonfrontier = retained[
        (
            flags[0, retained] & GRADED_SOURCE_CHAMPION_FRONTIER
        )
        == 0
    ]
    target = targets[0, :count]
    recalled = int(np.sum(target[retained_nonfrontier]))
    target_count = int(np.sum(target))
    finite_residuals = residuals[0, :count]
    return {
        "target_positive_recall": recalled / max(target_count, 1),
        "target_set_exact": recalled == target_count,
        "all_scores_finite": bool(np.all(np.isfinite(scores[0, :count]))),
        "residual": {
            "mean": float(np.mean(finite_residuals)),
            "minimum": float(np.min(finite_residuals)),
            "maximum": float(np.max(finite_residuals)),
            "standard_deviation": float(np.std(finite_residuals)),
        },
    }


def _gradient_norm(gradient: Any) -> float:
    values = (
        tree_flatten(gradient)
        if not isinstance(gradient, dict)
        or any(isinstance(value, dict) for value in gradient.values())
        else gradient.items()
    )
    total = 0.0
    count = 0
    for _name, value in values:
        array = np.asarray(value, dtype=np.float64)
        total += float(np.sum(array * array))
        count += array.size
    return float(np.sqrt(total)) if count else 0.0


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
            "host": socket.gethostname(),
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

    cache = subparsers.add_parser("cache")
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--cache", type=Path, required=True)
    cache.add_argument("--workers", type=int, choices=[8], required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--left-train", type=Path, required=True)
    compare.add_argument("--right-train", type=Path, required=True)
    compare.add_argument("--left-validation", type=Path, required=True)
    compare.add_argument("--right-validation", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    gradient = subparsers.add_parser("gradient")
    gradient.add_argument("--dataset", type=Path, required=True)
    gradient.add_argument("--cache", type=Path, required=True)
    gradient.add_argument("--output", type=Path, required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--train-dataset", type=Path, required=True)
    baseline.add_argument("--validation-dataset", type=Path, required=True)
    baseline.add_argument("--train-cache", type=Path, required=True)
    baseline.add_argument("--validation-cache", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "cache":
        report = run_cache_build(
            dataset=args.dataset,
            cache=args.cache,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    elif args.command == "compare":
        report = run_cache_compare(
            left_train=args.left_train,
            right_train=args.right_train,
            left_validation=args.left_validation,
            right_validation=args.right_validation,
        )
    elif args.command == "gradient":
        report = run_gradient_audit(
            dataset_root=args.dataset,
            cache_root=args.cache,
        )
    else:
        report = run_baseline_audit(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            train_cache=args.train_cache,
            validation_cache=args.validation_cache,
        )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
