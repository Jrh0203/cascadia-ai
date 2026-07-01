"""Fit-scaling and cross-group-interference diagnostics for ADR 0102."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import socket
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.frontier_supervision_identifiability import (
    iter_supervision_groups,
)
from cascadia_mlx.graded_oracle_dataset import decode_graded_oracle_groups
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    ExpectedRankBatch,
    build_expected_rank_target_mask,
    rotate_expected_rank_batch,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
    Scale16ExpectedRankDataset,
    frontier_expected_rank_scale16_loss,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-frontier-fit-interference-audit-v1"
SEED = 2026061630
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
COHORT_SIZE = 64
NESTED_SIZES = (1, 4, 16, 64)
CAPACITY_WIDTHS = (96, 192, 288)
CAPACITY_GROUPS = 32
GRADIENT_GROUPS = 32
ERROR_GROUPS = 24
FIT_EXPOSURES = 60
ERROR_EXPOSURES = 120
FIT_CHECKPOINTS = (6, 12, 24, 36, 48, 60)
ERROR_CHECKPOINTS = (6, 24, 60, 120)
SELECTED_CHECKPOINT = "step-000004514-epoch-0010-batch-000000"
SELECTED_MODEL_BLAKE3 = (
    "5b50a1db5f1f415ad6a10a7588d9521d6c11a9408be2e67d5691e85f60c04869"
)
EXPECTED_TRAIN_MANIFEST_BLAKE3 = (
    "7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99"
)


@dataclass(frozen=True)
class CohortGroup:
    """Stable pointer and metadata for one train decision group."""

    group_id: int
    phase: int
    candidate_count: int
    width_bucket: str
    shard_index: int
    ref_index: int


def width_bucket(candidate_count: int) -> str:
    """Return the frozen ADR 0102 width bucket."""
    if candidate_count <= 2048:
        return "at_most_2048"
    if candidate_count <= 4096:
        return "2049_to_4096"
    return "above_4096"


def select_audit_cohort(
    dataset: Scale16ExpectedRankDataset,
    count: int = COHORT_SIZE,
) -> list[CohortGroup]:
    """Select the deterministic phase-by-width interleaved audit cohort."""
    refs = [
        (shard_index, ref_index, ref)
        for shard_index, shard in enumerate(dataset.shards)
        for ref_index, ref in enumerate(shard.groups)
    ]
    supervision = list(iter_supervision_groups(dataset.base))
    if len(refs) != len(supervision):
        raise ValueError("dataset references and supervision groups differ")

    cells: dict[str, list[tuple[bytes, CohortGroup]]] = {}
    for (shard_index, ref_index, ref), group in zip(
        refs,
        supervision,
        strict=True,
    ):
        if int(ref.candidate_count) != int(group.candidate_count):
            raise ValueError("cohort candidate count drifted")
        digest = blake3.blake3()
        digest.update(EXPERIMENT_ID.encode("ascii"))
        digest.update(int(group.group_id).to_bytes(8, "little", signed=False))
        selected = CohortGroup(
            group_id=int(group.group_id),
            phase=int(group.phase),
            candidate_count=int(ref.candidate_count),
            width_bucket=width_bucket(int(ref.candidate_count)),
            shard_index=shard_index,
            ref_index=ref_index,
        )
        cell = f"{selected.phase}:{selected.width_bucket}"
        cells.setdefault(cell, []).append((digest.digest(), selected))

    ordered_cells = {
        name: [value for _digest, value in sorted(values, key=lambda item: item[0])]
        for name, values in sorted(cells.items())
    }
    cohort: list[CohortGroup] = []
    offset = 0
    while len(cohort) < count:
        progressed = False
        for values in ordered_cells.values():
            if offset < len(values):
                cohort.append(values[offset])
                progressed = True
                if len(cohort) == count:
                    break
        if not progressed:
            break
        offset += 1
    if len(cohort) != count:
        raise ValueError("train dataset is too small for the audit cohort")
    if len({group.group_id for group in cohort}) != count:
        raise ValueError("audit cohort contains duplicate group IDs")
    return cohort


def cohort_digest(groups: Iterable[CohortGroup]) -> str:
    """Hash the exact ordered cohort identity and stratification metadata."""
    digest = blake3.blake3()
    digest.update(EXPERIMENT_ID.encode("ascii"))
    for group in groups:
        digest.update(group.group_id.to_bytes(8, "little", signed=False))
        digest.update(group.phase.to_bytes(1, "little", signed=False))
        digest.update(group.candidate_count.to_bytes(4, "little", signed=False))
        digest.update(group.width_bucket.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_cohort_batches(
    dataset: Scale16ExpectedRankDataset,
    groups: Iterable[CohortGroup],
) -> list[ExpectedRankBatch]:
    """Decode exact one-group batches for a selected cohort."""
    batches: list[ExpectedRankBatch] = []
    for group in groups:
        shard = dataset.shards[group.shard_index]
        ref = shard.groups[group.ref_index]
        base = decode_graded_oracle_groups(shard.bytes(), (ref,))
        observed = int(np.asarray(base.group_id)[0]) & ((1 << 64) - 1)
        if observed != group.group_id:
            raise ValueError("decoded cohort group identity drifted")
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


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_identity(
    dataset: Scale16ExpectedRankDataset,
    cache_root: Path,
    cohort: list[CohortGroup],
) -> dict[str, Any]:
    manifest_digest = _checksum(dataset.root / "dataset.json")
    if manifest_digest != EXPECTED_TRAIN_MANIFEST_BLAKE3:
        raise ValueError("ADR 0102 train dataset manifest drifted")
    cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    if float(cache_manifest["target"]["target_scale"]) != TARGET_SCALE:
        raise ValueError("ADR 0102 cache target scale drifted")
    if (
        float(cache_manifest["target"]["student_temperature"])
        != STUDENT_TEMPERATURE
    ):
        raise ValueError("ADR 0102 cache student temperature drifted")
    if (
        cache_manifest.get("experiment_id")
        != "complete-action-frontier-expected-rank-scale16-v1"
    ):
        raise ValueError("ADR 0102 cache experiment identity drifted")
    if int(cache_manifest["dataset"]["groups"]) != dataset.group_count:
        raise ValueError("ADR 0102 cache group coverage drifted")
    if int(cache_manifest["dataset"]["candidates"]) != dataset.candidate_count:
        raise ValueError("ADR 0102 cache candidate coverage drifted")
    return {
        "train_manifest_blake3": manifest_digest,
        "cache_manifest_blake3": _checksum(cache_root / "manifest.json"),
        "cache_ordered_group_action_identity_blake3": cache_manifest[
            "ordered_group_action_identity_blake3"
        ],
        "cohort_digest_blake3": cohort_digest(cohort),
        "cohort": [asdict(group) for group in cohort],
    }


def _new_model(config: GradedOracleModelConfig | None = None) -> GradedOracleRanker:
    mx.random.seed(SEED)
    model = GradedOracleRanker(config or GradedOracleModelConfig())
    mx.eval(model.parameters())
    return model


def _selected_checkpoint_spec(run_dir: Path) -> tuple[GradedOracleModelConfig, Path]:
    best = json.loads((run_dir / "best.json").read_text())
    if best.get("checkpoint") != SELECTED_CHECKPOINT:
        raise ValueError("ADR 0101 selected checkpoint pointer drifted")
    checkpoint = run_dir / "checkpoints" / SELECTED_CHECKPOINT
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    model_path = checkpoint / "model.safetensors"
    metadata = manifest["files"]["model.safetensors"]
    if (
        model_path.stat().st_size != int(metadata["bytes"])
        or _checksum(model_path) != SELECTED_MODEL_BLAKE3
        or metadata["blake3"] != SELECTED_MODEL_BLAKE3
    ):
        raise ValueError("ADR 0101 selected model identity drifted")
    return GradedOracleModelConfig.from_dict(manifest["model_config"]), model_path


def _new_selected_model(
    config: GradedOracleModelConfig,
    model_path: Path,
) -> GradedOracleRanker:
    model = GradedOracleRanker(config)
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    return model


def _parameter_count(model: GradedOracleRanker) -> int:
    return sum(int(value.size) for _name, value in tree_flatten(model.parameters()))


def _group_metrics(
    model: GradedOracleRanker,
    batch: ExpectedRankBatch,
) -> dict[str, Any]:
    prediction = predict_graded_oracle_batch(model, batch)
    loss = frontier_expected_rank_scale16_loss(model, batch)
    mx.eval(prediction.scores, prediction.residuals, loss)
    scores = np.asarray(prediction.scores)[0]
    residuals = np.asarray(prediction.residuals)[0]
    candidate_mask = np.asarray(batch.candidate_mask)
    count = int(np.sum(candidate_mask[0]))
    flags = np.asarray(batch.source_flags)[0, :count]
    hashes = np.asarray(batch.action_hash)[0, :count]
    targets = build_expected_rank_target_mask(
        expected_rank=np.asarray(batch.expected_rank),
        expected_rank_mask=np.asarray(batch.expected_rank_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=candidate_mask,
        action_hashes=np.asarray(batch.action_hash),
    )[0, :count]
    retained = frontier_anchored_retained_indices(
        scores=scores[:count],
        source_flags=flags,
        action_hashes=hashes,
    )
    retained_nonfrontier = retained[
        (flags[retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    ]
    target_slots = int(np.sum(targets))
    target_hits = int(np.sum(targets[retained_nonfrontier]))
    winner = int(np.asarray(batch.selected_index)[0])
    return {
        "group_id": int(np.asarray(batch.group_id)[0]) & ((1 << 64) - 1),
        "phase": int(np.asarray(batch.phase)[0]),
        "candidate_count": count,
        "width_bucket": width_bucket(count),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact": target_hits == target_slots,
        "r4800_winner_retained": winner in retained,
        "objective": float(loss.item()),
        "finite_scores": bool(np.all(np.isfinite(scores[:count]))),
        "finite_residuals": bool(np.all(np.isfinite(residuals[:count]))),
    }


def _aggregate_metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        raise ValueError("cannot aggregate an empty metric set")
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
        "all_scores_finite": all(
            bool(group["finite_scores"]) and bool(group["finite_residuals"])
            for group in groups
        ),
    }


def evaluate_model(
    model: GradedOracleRanker,
    batches: list[ExpectedRankBatch],
    *,
    include_groups: bool = False,
) -> dict[str, Any]:
    """Evaluate exact deployment recovery on one decoded cohort."""
    model.eval()
    groups = [_group_metrics(model, batch) for batch in batches]
    report = _aggregate_metrics(groups)
    if include_groups:
        report["group_metrics"] = groups
    return report


def _arm_seed(arm_identity: str, block: int) -> int:
    digest = blake3.blake3()
    digest.update(SEED.to_bytes(8, "little", signed=False))
    digest.update(arm_identity.encode("ascii"))
    digest.update(block.to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(length=8), "little", signed=False)


def fit_model(
    model: GradedOracleRanker,
    batches: list[ExpectedRankBatch],
    *,
    exposures_per_group: int,
    checkpoints: tuple[int, ...],
    arm_identity: str,
) -> list[dict[str, Any]]:
    """Train with exact equal exposure and uniform rotation per group."""
    if exposures_per_group <= 0 or exposures_per_group % 6:
        raise ValueError("fit exposures must be a positive multiple of six")
    if not checkpoints or checkpoints[-1] != exposures_per_group:
        raise ValueError("fit checkpoints must end at the exposure budget")
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, frontier_expected_rank_scale16_loss)
    trajectory: list[dict[str, Any]] = [
        {
            "exposures_per_group": 0,
            "optimizer_steps": 0,
            "metrics": evaluate_model(model, batches),
        }
    ]
    optimizer_steps = 0
    started = time.perf_counter()
    for exposure in range(1, exposures_per_group + 1):
        block = (exposure - 1) // 6
        rotation = (exposure - 1) % 6
        order = np.arange(len(batches))
        np.random.default_rng(_arm_seed(arm_identity, block)).shuffle(order)
        model.train()
        for batch_index in order:
            batch = rotate_expected_rank_batch(batches[int(batch_index)], rotation)
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            optimizer_steps += 1
        if exposure in checkpoints:
            metrics = evaluate_model(model, batches)
            event = {
                "arm": arm_identity,
                "exposures_per_group": exposure,
                "optimizer_steps": optimizer_steps,
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": metrics,
            }
            trajectory.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    return trajectory


def run_nested_subset(
    *,
    dataset_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Run frozen nested 1/4/16/64-group fit scaling on john1."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    if dataset.split != "train":
        raise ValueError("ADR 0102 accepts only the open train split")
    cohort = select_audit_cohort(dataset)
    batches = load_cohort_batches(dataset, cohort)
    variants: dict[str, Any] = {}
    for size in NESTED_SIZES:
        model = _new_model()
        trajectory = fit_model(
            model,
            batches[:size],
            exposures_per_group=FIT_EXPOSURES,
            checkpoints=FIT_CHECKPOINTS,
            arm_identity=f"nested-{size}",
        )
        variants[str(size)] = {
            "group_count": size,
            "parameter_count": _parameter_count(model),
            "trajectory": trajectory,
            "final": trajectory[-1]["metrics"],
        }
        del model
        mx.clear_cache()
        gc.collect()
    size1 = variants["1"]["final"]
    size4 = variants["4"]["final"]
    size64 = variants["64"]["final"]
    local_size1 = bool(
        size1["target_positive_recall"] >= 0.95
        and size1["target_set_exact_fraction"] == 1.0
    )
    local_size4 = bool(
        size4["target_positive_recall"] >= 0.90
        and size4["target_set_exact_fraction"] >= 0.75
    )
    scaling_collapse = bool(
        (
            size4["target_positive_recall"]
            - size64["target_positive_recall"]
            >= 0.15
            or size4["target_set_exact_fraction"]
            - size64["target_set_exact_fraction"]
            >= 0.25
        )
        and size64["target_positive_recall"] < 0.80
    )
    scientific = {
        "arm": "nested-subset",
        "input_identity": _input_identity(dataset, cache_root, cohort),
        "seed": SEED,
        "exposures_per_group": FIT_EXPOSURES,
        "checkpoints": list(FIT_CHECKPOINTS),
        "variants": variants,
        "gates": {
            "size1_local_fit": local_size1,
            "size4_local_fit": local_size4,
            "scaling_collapse_material": scaling_collapse,
            "all_nested_sizes_completed": set(variants)
            == {str(size) for size in NESTED_SIZES},
            "all_exposure_checkpoints_completed": all(
                [
                    int(point["exposures_per_group"])
                    for point in value["trajectory"][1:]
                ]
                == list(FIT_CHECKPOINTS)
                for value in variants.values()
            ),
            "all_variants_finite": all(
                bool(value["final"]["all_scores_finite"])
                for value in variants.values()
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def run_capacity_scaling(
    *,
    dataset_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Compare 96/192/288 hidden widths on one identical 32-group cohort."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    cohort = select_audit_cohort(dataset)
    selected = cohort[:CAPACITY_GROUPS]
    batches = load_cohort_batches(dataset, selected)
    variants: dict[str, Any] = {}
    for hidden in CAPACITY_WIDTHS:
        config = GradedOracleModelConfig(hidden_dim=hidden, attention_heads=6)
        model = _new_model(config)
        trajectory = fit_model(
            model,
            batches,
            exposures_per_group=FIT_EXPOSURES,
            checkpoints=FIT_CHECKPOINTS,
            arm_identity=f"capacity-{hidden}",
        )
        variants[str(hidden)] = {
            "config": config.to_dict(),
            "parameter_count": _parameter_count(model),
            "trajectory": trajectory,
            "final": trajectory[-1]["metrics"],
        }
        del model
        mx.clear_cache()
        gc.collect()
    small = variants["96"]["final"]
    baseline = variants["192"]["final"]
    large = variants["288"]["final"]
    monotonic = bool(
        baseline["target_positive_recall"]
        >= small["target_positive_recall"] - 0.02
        and large["target_positive_recall"]
        >= baseline["target_positive_recall"] - 0.02
    )
    capacity_material = bool(
        monotonic
        and large["target_positive_recall"]
        - baseline["target_positive_recall"]
        >= 0.08
        and large["target_set_exact_fraction"]
        - baseline["target_set_exact_fraction"]
        >= 0.10
    )
    scientific = {
        "arm": "capacity-scaling",
        "input_identity": _input_identity(dataset, cache_root, selected),
        "full_cohort_digest_blake3": cohort_digest(cohort),
        "seed": SEED,
        "group_count": CAPACITY_GROUPS,
        "exposures_per_group": FIT_EXPOSURES,
        "variants": variants,
        "gates": {
            "recall_monotonic_with_tolerance": monotonic,
            "capacity_material": capacity_material,
            "all_capacity_widths_completed": set(variants)
            == {str(width) for width in CAPACITY_WIDTHS},
            "all_exposure_checkpoints_completed": all(
                [
                    int(point["exposures_per_group"])
                    for point in value["trajectory"][1:]
                ]
                == list(FIT_CHECKPOINTS)
                for value in variants.values()
            ),
            "all_variants_finite": all(
                bool(value["final"]["all_scores_finite"])
                for value in variants.values()
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def _gradient_vector(
    model: GradedOracleRanker,
    batch: ExpectedRankBatch,
) -> tuple[np.ndarray, list[tuple[str, int, int]], float]:
    loss_and_grad = nn.value_and_grad(model, frontier_expected_rank_scale16_loss)
    loss, gradients = loss_and_grad(model, batch)
    mx.eval(loss, gradients)
    flattened = tree_flatten(gradients)
    total = sum(int(value.size) for _name, value in flattened)
    vector = np.empty(total, dtype=np.float32)
    ranges: list[tuple[str, int, int]] = []
    offset = 0
    for name, value in flattened:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        stop = offset + len(array)
        vector[offset:stop] = array
        ranges.append((name, offset, stop))
        offset = stop
    if not np.all(np.isfinite(vector)):
        raise ValueError("gradient vector contains nonfinite values")
    return vector, ranges, float(loss.item())


def _scope_ranges(
    ranges: list[tuple[str, int, int]],
) -> dict[str, list[tuple[int, int]]]:
    scopes = {
        "full_model": [],
        "residual_head": [],
        "output_trunk": [],
        "candidate_projection": [],
        "representation_rest": [],
    }
    for name, start, stop in ranges:
        scopes["full_model"].append((start, stop))
        matched = False
        for scope in ("residual_head", "output_trunk", "candidate_projection"):
            if name.startswith(scope):
                scopes[scope].append((start, stop))
                matched = True
                break
        if not matched:
            scopes["representation_rest"].append((start, stop))
    return scopes


def _cosine_report(
    gradients: np.ndarray,
    ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    count = gradients.shape[0]
    gram = np.zeros((count, count), dtype=np.float64)
    for start, stop in ranges:
        block = gradients[:, start:stop]
        gram += np.asarray(block @ block.T, dtype=np.float64)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norms, norms)
    cosine = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )
    np.fill_diagonal(cosine, 1.0)
    off_diagonal = cosine[~np.eye(count, dtype=np.bool_)]
    total_gram = float(np.sum(gram))
    row_sum = np.sum(gram, axis=1)
    diagonal = np.diag(gram)
    other_dot = row_sum - diagonal
    other_norm_squared = np.maximum(
        total_gram - 2.0 * row_sum + diagonal,
        0.0,
    )
    other_denominator = norms * np.sqrt(other_norm_squared)
    other_cosine = np.divide(
        other_dot,
        other_denominator,
        out=np.zeros_like(other_dot),
        where=other_denominator > 0.0,
    )
    return {
        "cosine_matrix": cosine.tolist(),
        "gradient_norms": norms.tolist(),
        "off_diagonal": _distribution(off_diagonal),
        "off_diagonal_negative_fraction": float(np.mean(off_diagonal < 0.0)),
        "off_diagonal_at_most_negative_0_10_fraction": float(
            np.mean(off_diagonal <= -0.10)
        ),
        "cosine_to_other_gradient_sum": {
            "values": other_cosine.tolist(),
            "distribution": _distribution(other_cosine),
            "negative_fraction": float(np.mean(other_cosine < 0.0)),
        },
    }


def _gradient_state_report(
    model: GradedOracleRanker,
    batches: list[ExpectedRankBatch],
    *,
    state_name: str,
) -> dict[str, Any]:
    gradients: np.ndarray | None = None
    parameter_ranges: list[tuple[str, int, int]] | None = None
    losses: list[float] = []
    for row, batch in enumerate(batches):
        vector, observed_ranges, loss = _gradient_vector(model, batch)
        if gradients is None:
            gradients = np.empty(
                (len(batches), len(vector)),
                dtype=np.float32,
            )
            parameter_ranges = observed_ranges
        elif observed_ranges != parameter_ranges:
            raise ValueError("gradient parameter layout drifted")
        gradients[row] = vector
        losses.append(loss)
        print(
            json.dumps(
                {
                    "arm": "gradient-conflict",
                    "state": state_name,
                    "completed_groups": row + 1,
                    "total_groups": len(batches),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        mx.clear_cache()
    if gradients is None or parameter_ranges is None:
        raise ValueError("gradient cohort is empty")
    scopes = _scope_ranges(parameter_ranges)
    report = {
        "parameter_count": gradients.shape[1],
        "loss": _distribution(np.asarray(losses, dtype=np.float64)),
        "scopes": {
            name: _cosine_report(gradients, ranges)
            for name, ranges in scopes.items()
        },
    }
    del gradients
    gc.collect()
    return report


def run_gradient_conflict(
    *,
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> dict[str, Any]:
    """Measure exact gradient geometry at initialization and selected model."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    cohort = select_audit_cohort(dataset)
    selected = cohort[:GRADIENT_GROUPS]
    batches = load_cohort_batches(dataset, selected)

    initial_model = _new_model()
    initial = _gradient_state_report(
        initial_model,
        batches,
        state_name="initial",
    )
    del initial_model
    mx.clear_cache()
    gc.collect()

    config, model_path = _selected_checkpoint_spec(selected_run)
    selected_model = _new_selected_model(config, model_path)
    trained = _gradient_state_report(
        selected_model,
        batches,
        state_name="selected",
    )
    del selected_model
    mx.clear_cache()
    gc.collect()

    full = trained["scopes"]["full_model"]
    other = full["cosine_to_other_gradient_sum"]
    material = bool(
        other["negative_fraction"] >= 0.30
        and other["distribution"]["median"] <= -0.02
        and full["off_diagonal_at_most_negative_0_10_fraction"] >= 0.20
    )
    scientific = {
        "arm": "gradient-conflict",
        "input_identity": _input_identity(dataset, cache_root, selected),
        "full_cohort_digest_blake3": cohort_digest(cohort),
        "selected_checkpoint": SELECTED_CHECKPOINT,
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "group_ids": [group.group_id for group in selected],
        "initial": initial,
        "selected": trained,
        "gates": {
            "gradient_interference_material": material,
            "all_gradient_groups_completed": (
                len(initial["scopes"]["full_model"]["gradient_norms"])
                == GRADIENT_GROUPS
                and len(trained["scopes"]["full_model"]["gradient_norms"])
                == GRADIENT_GROUPS
            ),
            "all_gradient_norms_positive": all(
                value > 0.0
                for state in (initial, trained)
                for value in state["scopes"]["full_model"]["gradient_norms"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def _independent_adaptation(
    batches: list[ExpectedRankBatch],
    config: GradedOracleModelConfig,
    model_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        model = _new_selected_model(config, model_path)
        trajectory = fit_model(
            model,
            [batch],
            exposures_per_group=ERROR_EXPOSURES,
            checkpoints=ERROR_CHECKPOINTS,
            arm_identity=f"error-independent-{index}",
        )
        final = _group_metrics(model, batch)
        final["trajectory"] = trajectory
        groups.append(final)
        del model
        mx.clear_cache()
        gc.collect()
    return groups, _aggregate_metrics(groups)


def _slice_recovery(groups: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        slices.setdefault(f"phase_{group['phase']}", []).append(group)
        slices.setdefault(str(group["width_bucket"]), []).append(group)
    return {
        name: _aggregate_metrics(values)
        for name, values in sorted(slices.items())
    }


def run_error_anatomy(
    *,
    dataset_root: Path,
    cache_root: Path,
    selected_run: Path,
) -> dict[str, Any]:
    """Compare independent and shared adaptation from the selected model."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    cohort = select_audit_cohort(dataset)
    selected = cohort[:ERROR_GROUPS]
    batches = load_cohort_batches(dataset, selected)
    config, model_path = _selected_checkpoint_spec(selected_run)

    baseline_model = _new_selected_model(config, model_path)
    initial = evaluate_model(baseline_model, batches, include_groups=True)
    del baseline_model
    mx.clear_cache()
    gc.collect()

    independent_groups, independent = _independent_adaptation(
        batches,
        config,
        model_path,
    )

    shared_model = _new_selected_model(config, model_path)
    shared_trajectory = fit_model(
        shared_model,
        batches,
        exposures_per_group=ERROR_EXPOSURES,
        checkpoints=ERROR_CHECKPOINTS,
        arm_identity="error-shared",
    )
    shared = evaluate_model(shared_model, batches, include_groups=True)
    del shared_model
    mx.clear_cache()
    gc.collect()

    local_recovery = bool(
        independent["target_positive_recall"] >= 0.90
        and independent["target_set_exact_fraction"] >= 0.75
    )
    empirical_interference = bool(
        independent["target_positive_recall"]
        - shared["target_positive_recall"]
        >= 0.15
        and independent["target_set_exact_fraction"]
        - shared["target_set_exact_fraction"]
        >= 0.25
    )
    scientific = {
        "arm": "error-anatomy",
        "input_identity": _input_identity(dataset, cache_root, selected),
        "full_cohort_digest_blake3": cohort_digest(cohort),
        "selected_checkpoint": SELECTED_CHECKPOINT,
        "selected_model_blake3": SELECTED_MODEL_BLAKE3,
        "exposures_per_group": ERROR_EXPOSURES,
        "initial": initial,
        "independent": {
            "aggregate": independent,
            "slices": _slice_recovery(independent_groups),
            "groups": independent_groups,
        },
        "shared": {
            "aggregate": shared,
            "trajectory": shared_trajectory,
            "slices": _slice_recovery(shared["group_metrics"]),
            "groups": shared["group_metrics"],
        },
        "gates": {
            "independent_local_recovery": local_recovery,
            "empirical_interference_material": empirical_interference,
            "all_error_groups_completed": (
                len(independent_groups) == ERROR_GROUPS
                and int(shared["groups"]) == ERROR_GROUPS
            ),
            "all_error_scores_finite": bool(
                independent["all_scores_finite"]
                and shared["all_scores_finite"]
                and initial["all_scores_finite"]
            ),
        },
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def classify_fit_interference(
    arms: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    """Apply the frozen ADR 0102 classification precedence."""
    required = {
        "nested-subset",
        "capacity-scaling",
        "gradient-conflict",
        "error-anatomy",
    }
    if set(arms) != required:
        raise ValueError("ADR 0102 arm set is incomplete")
    pipeline = all(_arm_pipeline_passed(report) for report in arms.values())
    nested = arms["nested-subset"]["scientific"]["gates"]
    capacity = arms["capacity-scaling"]["scientific"]["gates"]
    gradient = arms["gradient-conflict"]["scientific"]["gates"]
    error = arms["error-anatomy"]["scientific"]["gates"]
    gates = {
        "pipeline_passed": pipeline,
        "local_recovery_passed": bool(
            nested["size1_local_fit"]
            and nested["size4_local_fit"]
            and error["independent_local_recovery"]
        ),
        "scaling_collapse_material": bool(
            nested["scaling_collapse_material"]
        ),
        "capacity_material": bool(capacity["capacity_material"]),
        "gradient_interference_material": bool(
            gradient["gradient_interference_material"]
        ),
        "empirical_interference_material": bool(
            error["empirical_interference_material"]
        ),
    }
    if not gates["pipeline_passed"]:
        classification = "fit_interference_pipeline_invalid"
    elif not gates["local_recovery_passed"]:
        classification = "local_optimization_or_representation_insufficient"
    else:
        interference = bool(
            gates["gradient_interference_material"]
            and gates["empirical_interference_material"]
        )
        if gates["scaling_collapse_material"]:
            if gates["capacity_material"] and interference:
                classification = "mixed_capacity_and_interference"
            elif gates["capacity_material"] and not interference:
                classification = "shared_capacity_bottleneck"
            elif interference and not gates["capacity_material"]:
                classification = "cross_group_gradient_interference"
            else:
                classification = "shared_model_scaling_failure_unresolved"
        else:
            classification = "no_material_fit_scaling_failure"
    return classification, gates


def combine_reports(paths: list[Path]) -> dict[str, Any]:
    """Validate four arm reports and produce the frozen classification."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    arms: dict[str, dict[str, Any]] = {}
    full_cohort_digests: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"unexpected experiment report: {path}")
        arm = str(report["scientific"]["arm"])
        if arm in arms:
            raise ValueError(f"duplicate ADR 0102 arm: {arm}")
        arms[arm] = report
        identity = report["scientific"]
        full_cohort_digests.add(
            str(
                identity.get(
                    "full_cohort_digest_blake3",
                    identity["input_identity"]["cohort_digest_blake3"],
                )
            )
        )
    if len(full_cohort_digests) != 1:
        raise ValueError("ADR 0102 arm cohorts do not share one frozen root")
    classification, gates = classify_fit_interference(arms)
    scientific = {
        "arm": "combined",
        "classification": classification,
        "gates": gates,
        "full_cohort_digest_blake3": next(iter(full_cohort_digests)),
        "arm_scientific_blake3": {
            name: _canonical_digest(report["scientific"])
            for name, report in sorted(arms.items())
        },
        "arm_telemetry": {
            name: report["telemetry"]
            for name, report in sorted(arms.items())
        },
        "duplicate_training_fraction": 0.0,
        **_closed_domains(),
    }
    return _report(scientific, started, swap_before)


def _arm_pipeline_passed(report: dict[str, Any]) -> bool:
    scientific = report["scientific"]
    telemetry = report["telemetry"]
    completion_gate = all(
        bool(value)
        for name, value in scientific.get("gates", {}).items()
        if name.startswith("all_")
    )
    finite_gate = all(
        bool(value)
        for name, value in scientific.get("gates", {}).items()
        if "finite" in name or "gradient_norms_positive" in name
    )
    return bool(
        scientific.get("test_split_opened") is False
        and scientific.get("gameplay_opened") is False
        and scientific.get("new_teacher_compute_used") is False
        and scientific.get("external_compute_used") is False
        and completion_gate
        and finite_gate
        and int(telemetry["peak_process_rss_bytes"]) <= 4 * 1024**3
        and int(telemetry["process_swaps"]) == 0
        and telemetry["system_swap_delta_bytes"] is not None
        and int(telemetry["system_swap_delta_bytes"]) <= 0
    )


def _closed_domains() -> dict[str, bool]:
    return {
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake3.blake3(payload).hexdigest()


def _report(
    scientific: dict[str, Any],
    started: float,
    swap_before: int | None,
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

    for name in ("nested-subset", "capacity-scaling"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)

    for name in ("gradient-conflict", "error-anatomy"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--selected-run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--arm", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "nested-subset":
        report = run_nested_subset(
            dataset_root=args.dataset,
            cache_root=args.cache,
        )
    elif args.command == "capacity-scaling":
        report = run_capacity_scaling(
            dataset_root=args.dataset,
            cache_root=args.cache,
        )
    elif args.command == "gradient-conflict":
        report = run_gradient_conflict(
            dataset_root=args.dataset,
            cache_root=args.cache,
            selected_run=args.selected_run,
        )
    elif args.command == "error-anatomy":
        report = run_error_anatomy(
            dataset_root=args.dataset,
            cache_root=args.cache,
            selected_run=args.selected_run,
        )
    else:
        report = combine_reports(args.arm)
    _write_json(args.output, report)


if __name__ == "__main__":
    main()
