"""Frozen-cache capacity and query-representation audit for ADR 0117."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.conditional_tile_target_only import target_only_tile_loss
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    HIDDEN_DIM,
    PARENT_STATE_DIM,
    STAGE_BATCH_SIZES,
    STAGE_CONTEXT_DIMS,
    STAGE_ITEM_DIMS,
    STAGE_WIDTHS,
    TILE_FACTOR_DIM,
    TILE_LOCAL_DIM,
    HierarchicalFactorCache,
    HierarchicalFactorRanker,
    build_stage_model,
    load_stage_model,
    score_stage_shard,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.model import SetAttentionBlock

EXPERIMENT_ID = "conditional-tile-capacity-query-audit-v1"
SEED = 2026061649
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
COHORT_CONFIGS = {
    16: {"batch_size": 16, "maximum_steps": 2000, "evaluation_interval": 100},
    256: {"batch_size": 16, "maximum_steps": 4000, "evaluation_interval": 100},
}
MODEL_KINDS = ("baseline", "attention")
WIDTH_BINS = (
    ("within_budget", 0, 32),
    ("width_33_64", 33, 64),
    ("width_65_96", 65, 96),
    ("width_97_128", 97, 128),
    ("width_129_plus", 129, 1 << 30),
)


@dataclass(frozen=True)
class QueryReference:
    """Stable location and selection key for one hard tile query."""

    shard_index: int
    query_index: int
    width: int
    phase: int
    selection_key: str


@dataclass(frozen=True)
class QuerySample:
    """Materialized model-visible values for one tile query."""

    reference: QueryReference
    state: np.ndarray
    context: np.ndarray
    items: np.ndarray
    target: np.ndarray


class AttentionTileRanker(nn.Module):
    """ADR 0115 ranker with explicit candidate-to-candidate self-attention."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(PARENT_STATE_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(STAGE_CONTEXT_DIMS["tile"], hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(STAGE_ITEM_DIMS["tile"], hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attention = [
            SetAttentionBlock(hidden_dim, 8, 4),
            SetAttentionBlock(hidden_dim, 8, 4),
        ]
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(
        self,
        state: mx.array,
        context: mx.array,
        items: mx.array,
        item_mask: mx.array,
    ) -> mx.array:
        state_value = self.state_encoder(state)
        context_value = self.context_encoder(context)
        item_value = self.item_encoder(items)
        state_items = mx.broadcast_to(state_value[:, None, :], item_value.shape)
        context_items = mx.broadcast_to(context_value[:, None, :], item_value.shape)
        hidden = self.interaction(
            mx.concatenate(
                [
                    item_value,
                    state_items,
                    context_items,
                    item_value * state_items,
                    item_value * context_items,
                    mx.abs(item_value - state_items),
                    mx.abs(item_value - context_items),
                ],
                axis=-1,
            )
        )
        hidden = hidden * item_mask[..., None]
        for block in self.attention:
            hidden = block(hidden, item_mask)
        denominator = mx.maximum(mx.sum(item_mask, axis=1, keepdims=True), 1)
        mean = mx.sum(hidden, axis=1) / denominator
        maximum = mx.max(
            mx.where(item_mask[..., None], hidden, -1e9),
            axis=1,
        )
        scores = self.output(
            mx.concatenate(
                [
                    hidden,
                    mx.broadcast_to(mean[:, None, :], hidden.shape),
                    mx.broadcast_to(maximum[:, None, :], hidden.shape),
                    hidden - mean[:, None, :],
                ],
                axis=-1,
            )
        ).reshape(item_mask.shape)
        return mx.where(item_mask, scores, -1e9)


def build_audit_model(model_kind: str) -> nn.Module:
    """Build exactly one frozen audit architecture."""
    if model_kind == "baseline":
        return build_stage_model("tile")
    if model_kind == "attention":
        return AttentionTileRanker()
    raise ValueError("unsupported ADR 0117 model kind")


def select_query_references(
    cache: HierarchicalFactorCache,
    *,
    cohort_size: int,
) -> list[QueryReference]:
    """Select one nested, deterministic sample of nontrivial train queries."""
    if cache.split != "train":
        raise ValueError("capacity cohorts require the open train cache")
    if cohort_size <= 0:
        raise ValueError("cohort size must be positive")
    candidates: list[QueryReference] = []
    payload = str(cache.manifest["payload_blake3"])
    for shard_index, arrays in enumerate(cache.iter_shards()):
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        groups = arrays["tile_query_group"]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            width = right - left
            positives = int(np.sum(targets[left:right]))
            if width <= STAGE_WIDTHS["tile"] or positives <= 0 or positives >= width:
                continue
            group = int(groups[query_index])
            phase = int(arrays["phase"][group])
            key = blake3.blake3(
                f"{payload}:{SEED}:{shard_index}:{query_index}".encode()
            ).hexdigest()
            candidates.append(
                QueryReference(
                    shard_index=shard_index,
                    query_index=query_index,
                    width=width,
                    phase=phase,
                    selection_key=key,
                )
            )
    candidates.sort(
        key=lambda value: (
            value.selection_key,
            value.shard_index,
            value.query_index,
        )
    )
    if len(candidates) < cohort_size:
        raise ValueError("requested cohort exceeds eligible hard queries")
    return candidates[:cohort_size]


def materialize_queries(
    cache: HierarchicalFactorCache,
    references: list[QueryReference],
) -> list[QuerySample]:
    """Copy only selected query values from the immutable cache."""
    by_shard: dict[int, list[QueryReference]] = {}
    for reference in references:
        by_shard.setdefault(reference.shard_index, []).append(reference)
    samples: list[QuerySample] = []
    for shard_index, arrays in enumerate(cache.iter_shards()):
        for reference in by_shard.get(shard_index, []):
            query_index = reference.query_index
            left = int(arrays["tile_query_offsets"][query_index])
            right = int(arrays["tile_query_offsets"][query_index + 1])
            group = int(arrays["tile_query_group"][query_index])
            samples.append(
                QuerySample(
                    reference=reference,
                    state=np.array(arrays["group_state"][group], copy=True),
                    context=np.array(
                        arrays["tile_query_context"][query_index],
                        copy=True,
                    ),
                    items=np.array(
                        arrays["tile_item_features"][left:right],
                        copy=True,
                    ),
                    target=np.array(
                        arrays["tile_item_target"][left:right],
                        copy=True,
                    ),
                )
            )
    samples.sort(key=lambda value: references.index(value.reference))
    if len(samples) != len(references):
        raise AssertionError("selected cohort materialization is incomplete")
    return samples


def _batch(
    samples: list[QuerySample],
    indices: np.ndarray,
) -> tuple[np.ndarray, ...]:
    selected = [samples[int(index)] for index in indices]
    maximum = max(len(sample.items) for sample in selected)
    items = np.zeros(
        (len(selected), maximum, STAGE_ITEM_DIMS["tile"]),
        dtype=np.float32,
    )
    item_mask = np.zeros((len(selected), maximum), dtype=np.bool_)
    target = np.zeros((len(selected), maximum), dtype=np.bool_)
    for row, sample in enumerate(selected):
        width = len(sample.items)
        items[row, :width] = sample.items
        item_mask[row, :width] = True
        target[row, :width] = sample.target
    return (
        np.stack([sample.state for sample in selected]),
        np.stack([sample.context for sample in selected]),
        items,
        item_mask,
        np.zeros((len(selected), maximum), dtype=np.float32),
        np.zeros((len(selected), maximum), dtype=np.bool_),
        target,
    )


def evaluate_cohort(model: nn.Module, samples: list[QuerySample]) -> dict[str, Any]:
    """Measure exact top-32 membership recovery on a materialized cohort."""
    target_total = 0
    target_hits = 0
    exact = 0
    finite = True
    losses: list[float] = []
    model.eval()
    for start in range(0, len(samples), 16):
        indices = np.arange(start, min(start + 16, len(samples)))
        values = tuple(mx.array(value) for value in _batch(samples, indices))
        scores = model(values[0], values[1], values[2], values[3])
        loss = target_only_tile_loss(model, *values)
        mx.eval(scores, loss)
        output = np.asarray(scores)
        losses.append(float(loss.item()))
        finite &= bool(np.all(np.isfinite(output))) and math.isfinite(losses[-1])
        for row, sample in enumerate(samples[start : start + len(indices)]):
            width = len(sample.items)
            selected = sorted(
                range(width),
                key=lambda index: (-float(output[row, index]), index),
            )[: min(STAGE_WIDTHS["tile"], width)]
            quota = int(np.sum(sample.target))
            hits = int(np.sum(sample.target[selected]))
            target_total += quota
            target_hits += hits
            exact += int(hits == quota)
    return {
        "queries": len(samples),
        "items": sum(len(sample.items) for sample in samples),
        "target_factors": target_total,
        "target_hits": target_hits,
        "target_factor_recall": target_hits / max(target_total, 1),
        "exact_query_fraction": exact / max(len(samples), 1),
        "mean_balanced_membership_loss": float(np.mean(losses)),
        "all_scores_finite": finite,
    }


def train_cohort(
    *,
    cache_root: Path,
    cohort_size: int,
    model_kind: str,
    output_root: Path,
    reference_weights: Path,
) -> dict[str, Any]:
    """Run one frozen memorization arm without touching validation."""
    if cohort_size not in COHORT_CONFIGS:
        raise ValueError("unsupported ADR 0117 cohort size")
    if model_kind not in MODEL_KINDS:
        raise ValueError("unsupported ADR 0117 model kind")
    if model_kind == "attention" and cohort_size != 256:
        raise ValueError("attention control is frozen to the 256-query cohort")
    if output_root.exists():
        raise ValueError("ADR 0117 cohort output already exists")
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    cache = HierarchicalFactorCache(cache_root)
    references = select_query_references(cache, cohort_size=cohort_size)
    samples = materialize_queries(cache, references)
    reference = load_stage_model("tile", reference_weights)
    reference_metrics = evaluate_cohort(reference, samples)
    del reference
    mx.clear_cache()

    config = COHORT_CONFIGS[cohort_size]
    mx.random.seed(SEED)
    model = build_audit_model(model_kind)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, target_only_tile_loss)
    output_root.mkdir(parents=True)
    trajectory_path = output_root / "metrics.jsonl"
    initial = evaluate_cohort(model, samples)
    best = initial
    best_step = 0
    stable_exact_checks = 0
    step = 0
    epoch = 0
    while step < int(config["maximum_steps"]):
        order = np.arange(len(samples))
        np.random.default_rng(SEED + epoch).shuffle(order)
        epoch += 1
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            values = tuple(mx.array(value) for value in _batch(samples, indices))
            loss, gradients = loss_and_grad(model, *values)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            if not math.isfinite(float(loss.item())):
                raise RuntimeError("ADR 0117 cohort training became nonfinite")
            step += 1
            should_evaluate = (
                step == 1
                or step % int(config["evaluation_interval"]) == 0
                or step == int(config["maximum_steps"])
            )
            if should_evaluate:
                metrics = evaluate_cohort(model, samples)
                event = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics": metrics,
                }
                _append_json(trajectory_path, event)
                print(json.dumps(event, sort_keys=True), flush=True)
                key = (
                    float(metrics["target_factor_recall"]),
                    float(metrics["exact_query_fraction"]),
                    -float(metrics["mean_balanced_membership_loss"]),
                )
                best_key = (
                    float(best["target_factor_recall"]),
                    float(best["exact_query_fraction"]),
                    -float(best["mean_balanced_membership_loss"]),
                )
                if key > best_key:
                    best = metrics
                    best_step = step
                    mx.save_safetensors(
                        str(output_root / "best.safetensors"),
                        dict(tree_flatten(model.parameters())),
                    )
                stable_exact_checks = (
                    stable_exact_checks + 1 if metrics["exact_query_fraction"] == 1.0 else 0
                )
                mx.clear_cache()
                if stable_exact_checks >= 3:
                    break
            if step >= int(config["maximum_steps"]):
                break
        if stable_exact_checks >= 3:
            break
    if best_step == 0:
        mx.save_safetensors(
            str(output_root / "best.safetensors"),
            dict(tree_flatten(model.parameters())),
        )
    usage = _resource_usage()
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname(),
        "arm": {
            "model_kind": model_kind,
            "cohort_size": cohort_size,
            "selection": "nested BLAKE3 sample of train tile queries wider than 32",
            "seed": SEED,
            "batch_size": int(config["batch_size"]),
            "maximum_steps": int(config["maximum_steps"]),
            "evaluation_interval": int(config["evaluation_interval"]),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "query_references": [
            {
                "shard_index": value.shard_index,
                "query_index": value.query_index,
                "width": value.width,
                "phase": value.phase,
                "selection_key": value.selection_key,
            }
            for value in references
        ],
        "cache_payload_blake3": cache.manifest["payload_blake3"],
        "reference_weights_blake3": checksum(reference_weights),
        "reference": reference_metrics,
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "steps_completed": step,
        "early_stopped_exact": stable_exact_checks >= 3,
        "parameter_count": sum(
            int(value.size) for _name, value in tree_flatten(model.parameters())
        ),
        "selected_weights_blake3": checksum(output_root / "best.safetensors"),
        "all_values_finite": bool(best["all_scores_finite"]),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **usage,
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
        "test_split_opened": False,
        "validation_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    report["scientific_blake3"] = _scientific_blake3(report)
    _write_json(output_root / "report.json", report)
    return report


def _bucket_name(width: int) -> str:
    for name, minimum, maximum in WIDTH_BINS:
        if minimum <= width <= maximum:
            return name
    raise AssertionError("tile width fell outside frozen bins")


def _score_variant(
    model: HierarchicalFactorRanker,
    arrays: dict[str, np.ndarray],
    variant: str,
) -> np.ndarray:
    scores = np.empty(len(arrays["tile_item_features"]), dtype=np.float32)
    offsets = arrays["tile_query_offsets"]
    query_count = len(offsets) - 1
    permutation = np.roll(np.arange(query_count), 1)
    cursor = 0
    model.eval()
    for start in range(0, query_count, STAGE_BATCH_SIZES["tile"]):
        selected = np.arange(start, min(start + STAGE_BATCH_SIZES["tile"], query_count))
        widths = offsets[selected + 1] - offsets[selected]
        maximum = int(np.max(widths))
        items = np.zeros(
            (len(selected), maximum, STAGE_ITEM_DIMS["tile"]),
            dtype=np.float32,
        )
        mask = np.zeros((len(selected), maximum), dtype=np.bool_)
        for row, query_index in enumerate(selected):
            left = int(offsets[query_index])
            right = int(offsets[query_index + 1])
            width = right - left
            items[row, :width] = arrays["tile_item_features"][left:right]
            mask[row, :width] = True
        groups = arrays["tile_query_group"][selected]
        state = arrays["group_state"][groups]
        context = arrays["tile_query_context"][selected]
        if variant == "permuted_context":
            context = arrays["tile_query_context"][permutation[selected]]
        elif variant == "permuted_state":
            permuted_groups = arrays["tile_query_group"][permutation[selected]]
            state = arrays["group_state"][permuted_groups]
        elif variant == "zero_context":
            context = np.zeros_like(context)
        elif variant == "zero_tile_factor":
            items[..., :TILE_FACTOR_DIM] = 0.0
        elif variant == "zero_tile_local":
            items[..., TILE_FACTOR_DIM : TILE_FACTOR_DIM + TILE_LOCAL_DIM] = 0.0
        elif variant == "zero_descendant":
            items[..., TILE_FACTOR_DIM + TILE_LOCAL_DIM :] = 0.0
        elif variant != "baseline":
            raise ValueError("unsupported ADR 0117 anatomy variant")
        output = model(
            mx.array(state),
            mx.array(context),
            mx.array(items),
            mx.array(mask),
        )
        mx.eval(output)
        values = np.asarray(output)
        for row, width in enumerate(widths):
            width = int(width)
            scores[cursor : cursor + width] = values[row, :width]
            cursor += width
    if cursor != len(scores):
        raise AssertionError("ADR 0117 variant score coverage drifted")
    return scores


def _anatomy_metrics(
    cache: HierarchicalFactorCache,
    model: HierarchicalFactorRanker,
    *,
    variant: str,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, float | int]] = {}
    for arrays in cache.iter_shards():
        scores = (
            score_stage_shard(model, arrays, "tile")
            if variant == "baseline"
            else _score_variant(model, arrays, variant)
        )
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        groups = arrays["tile_query_group"]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            width = right - left
            selected = sorted(
                range(left, right),
                key=lambda index: (-float(scores[index]), index),
            )[: min(STAGE_WIDTHS["tile"], width)]
            quota = int(np.sum(targets[left:right]))
            hits = int(np.sum(targets[selected]))
            phase = int(arrays["phase"][int(groups[query_index])])
            target_scores = scores[left:right][targets[left:right]]
            negative_scores = scores[left:right][~targets[left:right]]
            margin = (
                float(np.min(target_scores) - np.max(negative_scores))
                if len(target_scores) and len(negative_scores)
                else 0.0
            )
            for name in ("overall", _bucket_name(width), f"phase_{phase}"):
                bucket = buckets.setdefault(
                    name,
                    {
                        "queries": 0,
                        "target_factors": 0,
                        "target_hits": 0,
                        "exact_queries": 0,
                        "margin_sum": 0.0,
                        "positive_margin_queries": 0,
                    },
                )
                bucket["queries"] += 1
                bucket["target_factors"] += quota
                bucket["target_hits"] += hits
                bucket["exact_queries"] += int(hits == quota)
                bucket["margin_sum"] += margin
                bucket["positive_margin_queries"] += int(margin > 0.0)
    return {
        name: {
            **values,
            "target_factor_recall": int(values["target_hits"])
            / max(int(values["target_factors"]), 1),
            "exact_query_fraction": int(values["exact_queries"]) / max(int(values["queries"]), 1),
            "mean_target_nontarget_margin": float(values["margin_sum"])
            / max(int(values["queries"]), 1),
            "positive_margin_fraction": int(values["positive_margin_queries"])
            / max(int(values["queries"]), 1),
        }
        for name, values in sorted(buckets.items())
    }


def run_anatomy(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    calibrated_weights: Path,
    target_only_weights: Path,
) -> dict[str, Any]:
    """Compare ADR 0115/0116 errors and perturb target-only inputs."""
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    train = HierarchicalFactorCache(train_cache_root)
    validation = HierarchicalFactorCache(validation_cache_root)
    calibrated = load_stage_model("tile", calibrated_weights)
    target_only = load_stage_model("tile", target_only_weights)
    models = {
        "adr0115_calibrated": calibrated,
        "adr0116_target_only": target_only,
    }
    splits = {"train": train, "validation": validation}
    baseline = {
        model_name: {
            split: _anatomy_metrics(cache, model, variant="baseline")
            for split, cache in splits.items()
        }
        for model_name, model in models.items()
    }
    variants = (
        "permuted_context",
        "permuted_state",
        "zero_context",
        "zero_tile_factor",
        "zero_tile_local",
        "zero_descendant",
    )
    sensitivity = {
        variant: {
            split: _anatomy_metrics(cache, target_only, variant=variant)["overall"]
            for split, cache in splits.items()
        }
        for variant in variants
    }
    target_baseline = {split: baseline["adr0116_target_only"][split]["overall"] for split in splits}
    sensitivity_delta = {
        variant: {
            split: (
                float(target_baseline[split]["target_factor_recall"])
                - float(sensitivity[variant][split]["target_factor_recall"])
            )
            for split in splits
        }
        for variant in variants
    }
    scientific = {
        "calibrated_weights_blake3": checksum(calibrated_weights),
        "target_only_weights_blake3": checksum(target_only_weights),
        "train_cache_payload_blake3": train.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation.manifest["payload_blake3"],
        "baseline": baseline,
        "target_only_input_sensitivity": sensitivity,
        "target_only_recall_drop": sensitivity_delta,
        "query_context_material_at_1_point": (
            sensitivity_delta["permuted_context"]["validation"] >= 0.01
        ),
        "all_scores_finite": all(
            math.isfinite(float(value))
            for model_values in baseline.values()
            for split_values in model_values.values()
            for bucket in split_values.values()
            for value in (
                bucket["target_factor_recall"],
                bucket["exact_query_fraction"],
                bucket["mean_target_nontarget_margin"],
            )
        ),
        "test_split_opened": False,
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname(),
        "analysis": "error-anatomy-and-input-sensitivity",
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **_resource_usage(),
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
        "training_used": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return report


def classify_capacity_audit(
    baseline_16: dict[str, Any],
    baseline_256: dict[str, Any],
    attention_256: dict[str, Any],
) -> str:
    """Apply the frozen ADR 0117 capacity fork."""
    small = baseline_16["best"]
    medium = baseline_256["best"]
    attention = attention_256["best"]
    if float(small["target_factor_recall"]) < 0.995 or float(small["exact_query_fraction"]) < 0.95:
        return "local_baseline_fit_insufficient"
    if (
        float(medium["target_factor_recall"]) >= 0.98
        and float(medium["exact_query_fraction"]) >= 0.90
    ):
        return "full_data_scale_or_optimization_insufficient"
    attention_passed = (
        float(attention["target_factor_recall"]) >= 0.98
        and float(attention["exact_query_fraction"]) >= 0.90
    )
    attention_lift = (
        float(attention["target_factor_recall"]) - float(medium["target_factor_recall"]) >= 0.05
        and float(attention["exact_query_fraction"]) - float(medium["exact_query_fraction"]) >= 0.10
    )
    if attention_passed and attention_lift:
        return "query_relational_representation_insufficient"
    return "shared_capacity_or_optimization_insufficient"


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "peak_process_rss_bytes": peak,
        "process_swaps": int(usage.ru_nswap),
    }


def _scientific_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cohort = subparsers.add_parser("cohort")
    cohort.add_argument("--cache", type=Path, required=True)
    cohort.add_argument("--cohort-size", type=int, choices=COHORT_CONFIGS, required=True)
    cohort.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    cohort.add_argument("--reference-weights", type=Path, required=True)
    cohort.add_argument("--output", type=Path, required=True)

    anatomy = subparsers.add_parser("anatomy")
    anatomy.add_argument("--train-cache", type=Path, required=True)
    anatomy.add_argument("--validation-cache", type=Path, required=True)
    anatomy.add_argument("--calibrated-weights", type=Path, required=True)
    anatomy.add_argument("--target-only-weights", type=Path, required=True)
    anatomy.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "cohort":
        report = train_cohort(
            cache_root=args.cache,
            cohort_size=args.cohort_size,
            model_kind=args.model_kind,
            output_root=args.output,
            reference_weights=args.reference_weights,
        )
    else:
        report = run_anatomy(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            calibrated_weights=args.calibrated_weights,
            target_only_weights=args.target_only_weights,
        )
        _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
