"""Independent cache, alignment, optimization, and baseline audits for ADR 0101."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankBatch,
    build_expected_rank_target_mask,
    compare_expected_rank_array_payloads,
    compare_expected_rank_caches,
    evaluate_expected_rank_screen_baseline,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_diagnostics import (
    _batch_metrics,
    _report,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    EXPERIMENT_ID,
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
    Scale16ExpectedRankDataset,
    build_scale16_expected_rank_cache,
    frontier_expected_rank_scale16_loss,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_train import (
    EXPECTED_RANK_LEARNING_RATE,
    EXPECTED_RANK_SEED,
    EXPECTED_RANK_WEIGHT_DECAY,
)
from cascadia_mlx.graded_oracle_model import GradedOracleRanker

GRADIENT_GROUPS = 12
GRADIENT_STEPS = 32
RESIDUAL_RANGES = (0.0, 3.0, 6.0, 12.0)


def run_cache_build(
    *,
    dataset: Path,
    cache: Path,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Build one scale-16 cache with bounded host telemetry."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    manifest = build_scale16_expected_rank_cache(
        dataset,
        cache,
        workers=workers,
        overwrite=overwrite,
    )
    return _report(
        {
            "kind": "scale16-target-cache-build",
            "cache_manifest": manifest,
            "workers": workers,
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
        },
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def run_cache_compare(
    *,
    left_train: Path,
    right_train: Path,
    left_validation: Path,
    right_validation: Path,
    source_train: Path,
    source_validation: Path,
) -> dict[str, Any]:
    """Verify independent scale-16 caches and unchanged ADR 0100 rank bytes."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    scientific = {
        "kind": "scale16-target-cache-comparison",
        "scale16_train": compare_expected_rank_caches(left_train, right_train),
        "scale16_validation": compare_expected_rank_caches(
            left_validation,
            right_validation,
        ),
        "adr0100_train_rank_bytes": compare_expected_rank_array_payloads(
            source_train,
            right_train,
        ),
        "adr0100_validation_rank_bytes": compare_expected_rank_array_payloads(
            source_validation,
            right_validation,
        ),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    scientific["passed"] = bool(
        scientific["scale16_train"]["scientific_payload_identical"]
        and scientific["scale16_train"]["all_file_bytes_identical"]
        and scientific["scale16_validation"]["scientific_payload_identical"]
        and scientific["scale16_validation"]["all_file_bytes_identical"]
        and scientific["adr0100_train_rank_bytes"]["all_file_bytes_identical"]
        and scientific["adr0100_train_rank_bytes"][
            "ordered_group_action_identity_identical"
        ]
        and scientific["adr0100_validation_rank_bytes"][
            "all_file_bytes_identical"
        ]
        and scientific["adr0100_validation_rank_bytes"][
            "ordered_group_action_identity_identical"
        ]
    )
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def run_alignment_audit(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    train_cache: Path,
    validation_cache: Path,
) -> dict[str, Any]:
    """Measure split-wide target concentration and uniform-start gradients."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    train = _alignment_summary(
        Scale16ExpectedRankDataset(train_dataset, train_cache)
    )
    validation = _alignment_summary(
        Scale16ExpectedRankDataset(validation_dataset, validation_cache)
    )
    scientific = {
        "kind": "scale16-target-gradient-alignment",
        "train": train,
        "validation": validation,
        "gates": {
            "train_deployed_target_mass_above_0_90": (
                train["probability_mass_in_deployed_target"]["mean"] > 0.90
            ),
            "validation_deployed_target_mass_above_0_90": (
                validation["probability_mass_in_deployed_target"]["mean"] > 0.90
            ),
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    scientific["passed"] = all(scientific["gates"].values())
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def run_multi_group_gradient_audit(
    *,
    dataset_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Optimize the 12 widest groups independently for 32 frozen steps."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    if dataset.split != "train":
        raise ValueError("scale-16 gradient audit requires train split")
    batches = _widest_batches(dataset, GRADIENT_GROUPS)
    groups: list[dict[str, Any]] = []
    for batch in batches:
        mx.random.seed(EXPECTED_RANK_SEED)
        model = GradedOracleRanker()
        optimizer = optim.AdamW(
            learning_rate=EXPECTED_RANK_LEARNING_RATE,
            weight_decay=EXPECTED_RANK_WEIGHT_DECAY,
        )
        loss_and_grad = nn.value_and_grad(
            model,
            frontier_expected_rank_scale16_loss,
        )
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
        after_one_loss: float | None = None
        minimum_step_loss = float(initial_loss.item())
        for step in range(GRADIENT_STEPS):
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            minimum_step_loss = min(minimum_step_loss, float(loss.item()))
            if step == 0:
                observed = frontier_expected_rank_scale16_loss(model, batch)
                mx.eval(observed)
                after_one_loss = float(observed.item())
        final_loss = frontier_expected_rank_scale16_loss(model, batch)
        mx.eval(final_loss)
        groups.append(
            {
                "group_id": int(np.asarray(batch.group_id)[0])
                & ((1 << 64) - 1),
                "candidate_count": int(
                    np.sum(np.asarray(batch.candidate_mask)[0])
                ),
                "target_candidates": int(
                    np.sum(np.asarray(batch.expected_rank_mask)[0])
                ),
                "initial_loss": float(initial_loss.item()),
                "after_one_step_loss": after_one_loss,
                "final_loss": float(final_loss.item()),
                "minimum_training_step_loss": minimum_step_loss,
                "whole_model_gradient_norm": whole_gradient_norm,
                "residual_head_gradient_norm": residual_gradient_norm,
                "initial": initial_metrics,
                "final": _batch_metrics(model, batch),
            }
        )
        mx.clear_cache()
    scientific = {
        "kind": "scale16-multi-group-optimization-audit",
        "seed": EXPECTED_RANK_SEED,
        "steps_per_group": GRADIENT_STEPS,
        "groups": groups,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report = _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )
    telemetry = report["telemetry"]
    report["scientific"]["gates"] = {
        "audited_exactly_12_groups": len(groups) == GRADIENT_GROUPS,
        "all_losses_finite": all(
            np.isfinite(group["initial_loss"])
            and group["after_one_step_loss"] is not None
            and np.isfinite(group["after_one_step_loss"])
            and np.isfinite(group["final_loss"])
            for group in groups
        ),
        "all_gradients_finite_and_nonzero": all(
            np.isfinite(group["whole_model_gradient_norm"])
            and group["whole_model_gradient_norm"] > 0.0
            and np.isfinite(group["residual_head_gradient_norm"])
            and group["residual_head_gradient_norm"] > 0.0
            for group in groups
        ),
        "every_final_loss_decreased_strictly": all(
            group["final_loss"] < group["initial_loss"] for group in groups
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


def run_baseline_reachability_audit(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    train_cache: Path,
    validation_cache: Path,
) -> dict[str, Any]:
    """Measure the screen baseline and bounded target-set reachability."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    train = Scale16ExpectedRankDataset(train_dataset, train_cache)
    validation = Scale16ExpectedRankDataset(
        validation_dataset,
        validation_cache,
    )
    scientific = {
        "kind": "scale16-baseline-reachability-anatomy",
        "train": {
            "baseline": evaluate_expected_rank_screen_baseline(train),
            "reachability": _reachability_summary(train),
        },
        "validation": {
            "baseline": evaluate_expected_rank_screen_baseline(validation),
            "reachability": _reachability_summary(validation),
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    scientific["gates"] = {
        "train_exact_reachability_at_6": (
            scientific["train"]["reachability"]["ceilings"]["6.0"][
                "target_set_exact_fraction"
            ]
            == 1.0
        ),
        "validation_exact_reachability_at_6": (
            scientific["validation"]["reachability"]["ceilings"]["6.0"][
                "target_set_exact_fraction"
            ]
            == 1.0
        ),
    }
    scientific["passed"] = all(scientific["gates"].values())
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def _alignment_summary(dataset: Scale16ExpectedRankDataset) -> dict[str, Any]:
    target_mass: list[float] = []
    target_gradient: list[float] = []
    entropy: list[float] = []
    effective_support: list[float] = []
    groups = 0
    candidates = 0
    for batch in dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        count = int(np.sum(np.asarray(batch.candidate_mask)[0]))
        ranks = np.asarray(batch.expected_rank)[0, :count]
        rank_mask = np.asarray(batch.expected_rank_mask)[0, :count]
        flags = np.asarray(batch.source_flags)[0, :count]
        hashes = np.asarray(batch.action_hash)[0, :count]
        mask = np.ones((1, count), dtype=np.bool_)
        target = build_expected_rank_target_mask(
            expected_rank=ranks[None, :],
            expected_rank_mask=rank_mask[None, :],
            source_flags=flags[None, :],
            candidate_mask=mask,
            action_hashes=hashes[None, :],
        )[0]
        labeled = np.flatnonzero(rank_mask)
        logits = -(ranks[labeled].astype(np.float64) - 1.0) / TARGET_SCALE
        probability = _softmax(logits)
        full_probability = np.zeros(count, dtype=np.float64)
        full_probability[labeled] = probability
        frontier = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
        eligible = ~frontier
        student = np.zeros(count, dtype=np.float64)
        student[eligible] = 1.0 / int(np.sum(eligible))
        absolute_gradient = np.abs(
            (student - full_probability) / STUDENT_TEMPERATURE
        )
        target_mass.append(
            float(np.clip(np.sum(full_probability[target]), 0.0, 1.0))
        )
        target_gradient.append(
            float(np.sum(absolute_gradient[target]))
            / float(np.sum(absolute_gradient))
        )
        positive = probability[probability > 0.0]
        group_entropy = -float(np.sum(positive * np.log2(positive)))
        entropy.append(group_entropy)
        effective_support.append(2.0**group_entropy)
        groups += 1
        candidates += count
    return {
        "split": dataset.split,
        "groups": groups,
        "candidates": candidates,
        "probability_mass_in_deployed_target": _distribution(target_mass),
        "uniform_student_absolute_gradient_fraction_in_deployed_target": (
            _distribution(target_gradient)
        ),
        "target_entropy_bits": _distribution(entropy),
        "target_effective_support": _distribution(effective_support),
    }


def _widest_batches(
    dataset: Scale16ExpectedRankDataset,
    count: int,
) -> list[ExpectedRankBatch]:
    refs: list[tuple[int, int, int, Any]] = []
    for shard_index, shard in enumerate(dataset.shards):
        for ref_index, ref in enumerate(shard.groups):
            refs.append(
                (
                    ref.candidate_count,
                    -shard_index,
                    -ref_index,
                    ref,
                )
            )
    selected = sorted(refs, reverse=True)[:count]
    if len(selected) != count:
        raise ValueError("scale-16 gradient audit dataset is too small")
    batches: list[ExpectedRankBatch] = []
    for _width, negative_shard, _negative_ref, ref in selected:
        shard = dataset.shards[-negative_shard]
        base = decode_graded_oracle_groups(shard.bytes(), (ref,))
        ranks, mask = dataset.cache.ranks_for_batch(base)
        batches.append(
            ExpectedRankBatch(
                base,
                ranks,
                mask,
                TARGET_SCALE,
                STUDENT_TEMPERATURE,
            )
        )
    return batches


def _reachability_summary(
    dataset: Scale16ExpectedRankDataset,
) -> dict[str, Any]:
    hits = {value: 0 for value in RESIDUAL_RANGES}
    exact = {value: 0 for value in RESIDUAL_RANGES}
    required: list[float] = []
    groups = 0
    targets = 0
    for batch in dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        count = int(np.sum(np.asarray(batch.candidate_mask)[0]))
        ranks = np.asarray(batch.expected_rank)[0, :count]
        rank_mask = np.asarray(batch.expected_rank_mask)[0, :count]
        flags = np.asarray(batch.source_flags)[0, :count]
        hashes = np.asarray(batch.action_hash)[0, :count]
        screen = np.asarray(batch.screen_value)[0, :count]
        target = build_expected_rank_target_mask(
            expected_rank=ranks[None, :],
            expected_rank_mask=rank_mask[None, :],
            source_flags=flags[None, :],
            candidate_mask=np.ones((1, count), dtype=np.bool_),
            action_hashes=hashes[None, :],
        )[0]
        eligible = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        non_target = eligible & ~target
        target_count = int(np.sum(target))
        required.append(
            max(
                0.0,
                float(np.max(screen[non_target]) - np.min(screen[target]))
                / 2.0,
            )
        )
        eligible_indices = np.flatnonzero(eligible)
        for residual_range in RESIDUAL_RANGES:
            optimistic = screen.copy()
            optimistic[target] += residual_range
            optimistic[non_target] -= residual_range
            ranked = np.asarray(
                sorted(
                    (int(index) for index in eligible_indices),
                    key=lambda index: (
                        -float(optimistic[index]),
                        bytes(hashes[index]),
                    ),
                ),
                dtype=np.int32,
            )[:target_count]
            recalled = int(np.sum(target[ranked]))
            hits[residual_range] += recalled
            exact[residual_range] += int(recalled == target_count)
        groups += 1
        targets += target_count
    return {
        "split": dataset.split,
        "groups": groups,
        "target_positives": targets,
        "required_symmetric_residual_range": _distribution(required),
        "ceilings": {
            str(value): {
                "target_positive_recall": hits[value] / targets,
                "target_set_exact_fraction": exact[value] / groups,
            }
            for value in RESIDUAL_RANGES
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


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("scale-16 diagnostic distribution is invalid")
    return {
        "count": len(array),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


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
    compare.add_argument("--source-train", type=Path, required=True)
    compare.add_argument("--source-validation", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    alignment = subparsers.add_parser("alignment")
    alignment.add_argument("--train-dataset", type=Path, required=True)
    alignment.add_argument("--validation-dataset", type=Path, required=True)
    alignment.add_argument("--train-cache", type=Path, required=True)
    alignment.add_argument("--validation-cache", type=Path, required=True)
    alignment.add_argument("--output", type=Path, required=True)

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
            source_train=args.source_train,
            source_validation=args.source_validation,
        )
    elif args.command == "alignment":
        report = run_alignment_audit(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            train_cache=args.train_cache,
            validation_cache=args.validation_cache,
        )
    elif args.command == "gradient":
        report = run_multi_group_gradient_audit(
            dataset_root=args.dataset,
            cache_root=args.cache,
        )
    else:
        report = run_baseline_reachability_audit(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            train_cache=args.train_cache,
            validation_cache=args.validation_cache,
        )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
