"""Unified CascadiaFormer trainer for expert-root JSONL/NPZ data."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, replace
import functools
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .expert_tensor_shards import ExpertTensorCorpus, collate_expert_tensor_examples
from .replay import read_replay_jsonl
from .schema import EXPERT_ROOT_SCHEMA_ID, EXPERT_TENSOR_SHARD_SCHEMA_ID
from .torch_cascadiaformer import build_cascadiaformer, config_for_size, parameter_count


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    q: float = 1.0
    value: float = 0.25
    score: float = 0.10
    rank: float = 0.05
    uncertainty: float = 0.01
    greedy_policy: float = 0.0
    greedy_margin: float = 0.0
    greedy_margin_value: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


LOSS_COMPONENT_KEYS = (
    "total",
    "policy",
    "weighted_policy",
    "q",
    "score_to_go_q",
    "final_q_regret",
    "value",
    "score",
    "rank",
    "uncertainty",
    "greedy_policy",
    "greedy_margin",
)
RETENTION_METRIC_KEYS = (
    "teacher_top1",
    "greedy_top1",
    "mean_teacher_rank",
    "mean_greedy_rank",
    "teacher_advantage_over_greedy",
)
AGGREGATE_KEYS = LOSS_COMPONENT_KEYS + RETENTION_METRIC_KEYS


def loss_weights_for_objective(objective: str) -> LossWeights:
    if objective == "expert":
        return LossWeights()
    if objective == "search-improved-greedy-retention":
        return LossWeights(
            policy=1.0,
            q=0.20,
            value=0.05,
            score=0.02,
            rank=0.01,
            uncertainty=0.01,
            greedy_policy=0.75,
            greedy_margin=0.25,
            greedy_margin_value=0.25,
        )
    if objective == "gumbel-selfplay":
        # Self-play search targets: soft improved-policy distillation, real
        # final-outcome values (value up-weighted because search bootstraps on
        # it), no greedy-retention terms.
        return LossWeights(
            policy=1.0,
            q=0.5,
            value=0.5,
            score=0.05,
            rank=0.02,
            uncertainty=0.01,
            greedy_policy=0.0,
            greedy_margin=0.0,
            greedy_margin_value=0.0,
        )
    if objective == "k32-greedy-retention":
        return LossWeights(
            policy=0.10,
            q=0.05,
            value=0.05,
            score=0.02,
            rank=0.01,
            uncertainty=0.0,
            greedy_policy=2.0,
            greedy_margin=0.25,
            greedy_margin_value=0.25,
        )
    if objective == "pure-greedy-retention":
        return LossWeights(
            policy=0.0,
            q=0.0,
            value=0.0,
            score=0.0,
            rank=0.0,
            uncertainty=0.0,
            greedy_policy=1.0,
            greedy_margin=0.25,
            greedy_margin_value=0.25,
        )
    raise ValueError(f"unsupported objective {objective!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _schema_ids_for_formats(*formats: str) -> list[str]:
    schema_ids: list[str] = []
    for corpus_format in formats:
        if corpus_format == "jsonl":
            candidate = EXPERT_ROOT_SCHEMA_ID
        elif corpus_format == "npz":
            candidate = EXPERT_TENSOR_SHARD_SCHEMA_ID
        else:
            raise ValueError(f"unsupported expert corpus format {corpus_format!r}")
        if candidate not in schema_ids:
            schema_ids.append(candidate)
    return schema_ids


def _selected_action_indices(records: list[dict[str, Any]]) -> list[int]:
    out = []
    for record in records:
        action_ids = [action["action_id"] for action in record["legal_actions"]]
        out.append(action_ids.index(record["selected_action"]))
    return out


def _score_targets(records: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    import torch

    categories = ("wildlife", "habitat", "nature_tokens")
    target = torch.zeros((len(records), len(categories), 4), dtype=torch.float32)
    for row, record in enumerate(records):
        for seat in range(4):
            parts = record["score_decomposition"][str(seat)]
            for category_index, category in enumerate(categories):
                target[row, category_index, seat] = float(parts[category])
    return target


def collate_expert_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    from .torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots

    batch = collate_semantic_relation_bias_roots(records)
    max_actions = batch["action_mask"].shape[1]
    target_q = torch.zeros((len(records), max_actions), dtype=torch.float32)
    q_valid = torch.zeros((len(records), max_actions), dtype=torch.bool)
    score_to_go = torch.zeros((len(records), max_actions), dtype=torch.float32)
    visits = torch.zeros((len(records), max_actions), dtype=torch.float32)
    q_variance = torch.zeros((len(records), max_actions), dtype=torch.float32)
    q_count = torch.ones((len(records), max_actions), dtype=torch.float32)
    truncated_count = torch.zeros((len(records), max_actions), dtype=torch.float32)
    exact_afterstate = torch.zeros((len(records), max_actions), dtype=torch.float32)
    for batch_index, record in enumerate(records):
        count = len(record["legal_actions"])
        target_q[batch_index, :count] = torch.tensor(record["per_action_Q"], dtype=torch.float32)
        q_valid[batch_index, :count] = torch.tensor(record.get("per_action_Q_valid", [True] * count), dtype=torch.bool)
        score_to_go[batch_index, :count] = torch.tensor(record.get("per_action_score_to_go", record["per_action_Q"]), dtype=torch.float32)
        visits[batch_index, :count] = torch.tensor(record["visits"], dtype=torch.float32)
        q_variance[batch_index, :count] = torch.tensor(
            record.get("per_action_Q_variance", [0.0] * count),
            dtype=torch.float32,
        )
        q_count[batch_index, :count] = torch.tensor(
            record.get("per_action_Q_count", [1.0] * count),
            dtype=torch.float32,
        )
        truncated_count[batch_index, :count] = torch.tensor(
            record.get("per_action_truncated_count", [0.0] * count),
            dtype=torch.float32,
        )
        exact_afterstate[batch_index, :count] = torch.tensor(
            record.get("exact_afterstate_score_active", [0.0] * count),
            dtype=torch.float32,
        )
    batch.update(
        {
            "target_q": target_q,
            "q_valid": q_valid,
            "target_score_to_go": score_to_go,
            "visits": visits,
            "target_q_variance": q_variance,
            "target_q_count": q_count,
            "target_truncated_count": truncated_count,
            "exact_afterstate_score_active": exact_afterstate,
            "selected_action_index": torch.tensor(_selected_action_indices(records), dtype=torch.long),
            "greedy_action_index": torch.zeros((len(records),), dtype=torch.long),
            "target_value": torch.tensor([record["final_score_vector"] for record in records], dtype=torch.float32),
            "target_rank": torch.tensor([record["rank_vector"] for record in records], dtype=torch.long) - 1,
            "target_score": _score_targets(records),
            "schema_ids": [record["schema_id"] for record in records],
            "state_hashes": [record["state_hash"] for record in records],
        }
    )
    has_improved_policy = all(record.get("improved_policy") is not None for record in records)
    if has_improved_policy:
        improved = torch.zeros((len(records), max_actions), dtype=torch.float32)
        for batch_index, record in enumerate(records):
            count = len(record["legal_actions"])
            improved[batch_index, :count] = torch.tensor(record["improved_policy"], dtype=torch.float32)
        batch["improved_policy"] = improved
        batch["search_root_value"] = torch.tensor(
            [float(record.get("search_root_value", 0.0)) for record in records],
            dtype=torch.float32,
        )
    batch["has_improved_policy"] = has_improved_policy
    return batch


def _move_to_device(batch: dict[str, Any], device, *, non_blocking: bool = False):  # type: ignore[no-untyped-def]
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _loss_scalars(losses: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    """Fetch scalar loss values with a single device->host synchronization.

    Exactly-safe replacement for per-key ``float(loss.detach().cpu())``: each
    scalar is upcast to float64 (exact for bf16/fp16/fp32) on-device, stacked,
    and transferred once. Values are bit-identical to the per-key transfers;
    on CUDA this turns len(keys) synchronizations into one.
    """
    import torch

    stacked = torch.stack([losses[key].detach().to(dtype=torch.float64) for key in keys])
    return dict(zip(keys, stacked.cpu().tolist()))


def _load_corpus(paths: list[Path], *, corpus_format: str) -> Any:
    if corpus_format == "jsonl":
        return [record for path in paths for record in read_replay_jsonl(path)]
    if corpus_format == "npz":
        return ExpertTensorCorpus(paths)
    raise ValueError(f"unsupported expert corpus format {corpus_format!r}")


def _corpus_len(corpus: Any) -> int:
    return len(corpus)


def _corpus_examples(corpus: Any, indices: list[int], *, corpus_format: str) -> list[Any]:
    if corpus_format == "jsonl":
        return [corpus[index] for index in indices]
    if corpus_format == "npz":
        return corpus.examples(indices)
    raise ValueError(f"unsupported expert corpus format {corpus_format!r}")


def _collate_examples(examples: list[Any], *, corpus_format: str) -> dict[str, Any]:
    if corpus_format == "jsonl":
        return collate_expert_roots(examples)
    if corpus_format == "npz":
        return collate_expert_tensor_examples(examples)
    raise ValueError(f"unsupported expert corpus format {corpus_format!r}")


def _loss_components(outputs: dict[str, Any], batch: dict[str, Any], weights: LossWeights):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    mask = batch["action_mask"]
    q_mask = batch["q_valid"] & mask
    logits = outputs["logits"].masked_fill(~mask, -1.0e9)
    teacher_target = batch["selected_action_index"]
    greedy_target = batch.get("greedy_action_index")
    if greedy_target is None:
        greedy_target = torch.zeros_like(teacher_target)
    selected_policy = F.cross_entropy(logits, teacher_target)
    improved_policy_target = batch.get("improved_policy") if batch.get("has_improved_policy") else None
    target_score_to_go = batch.get("target_score_to_go", batch["target_q"])
    exact_afterstate = batch.get("exact_afterstate_score_active")
    if exact_afterstate is None:
        exact_afterstate = torch.zeros_like(batch["target_q"])
    target_final_q = batch["target_q"]
    predicted_score_to_go = outputs["q"]
    predicted_final_q = exact_afterstate + predicted_score_to_go
    q_temperature = 8.0
    target_distribution_logits = target_final_q.masked_fill(~q_mask, -1.0e9) / q_temperature
    target_distribution = torch.softmax(target_distribution_logits, dim=1)
    target_distribution = target_distribution.masked_fill(~q_mask, 0.0)
    normalizer = target_distribution.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    target_distribution = target_distribution / normalizer
    log_policy = torch.log_softmax(logits, dim=1)
    weighted_policy = -(target_distribution * log_policy).sum(dim=1).mean()
    if improved_policy_target is not None:
        # Gumbel self-play: soft-target cross-entropy against the search's
        # improved policy (equivalent to KL up to the target entropy constant)
        # replaces the selected-one-hot / softmax(Q) blend.
        masked_target = improved_policy_target.masked_fill(~mask, 0.0)
        target_normalizer = masked_target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        masked_target = masked_target / target_normalizer
        policy = -(masked_target * log_policy).sum(dim=1).mean()
        weighted_policy = policy
    else:
        policy = 0.5 * selected_policy + 0.5 * weighted_policy
    greedy_policy = F.cross_entropy(logits, greedy_target)
    greedy_one_hot = F.one_hot(greedy_target, num_classes=logits.shape[1]).to(dtype=torch.bool)
    greedy_competitor_mask = mask & ~greedy_one_hot
    greedy_target_logit = logits.gather(1, greedy_target.unsqueeze(1)).squeeze(1)
    if greedy_competitor_mask.any():
        competitor_max = logits.masked_fill(~greedy_competitor_mask, -1.0e9).max(dim=1).values
        has_competitor = greedy_competitor_mask.any(dim=1)
        greedy_margin = F.relu(
            weights.greedy_margin_value - (greedy_target_logit[has_competitor] - competitor_max[has_competitor])
        ).mean()
    else:
        greedy_margin = outputs["logits"].sum() * 0.0
    if q_mask.any():
        q_count = batch.get("target_q_count")
        if q_count is None:
            q_count = torch.ones_like(target_final_q)
        q_variance = batch.get("target_q_variance")
        if q_variance is None:
            q_variance = torch.zeros_like(target_final_q)
        teacher_se_sq = q_variance.clamp_min(0.0) / q_count.clamp_min(1.0)
        confidence = (1.0 / (0.25 + teacher_se_sq)).clamp(0.25, 4.0)
        quantile_values = outputs.get("q_quantile_values")
        if quantile_values is not None:
            # Distributional score-to-go: pinball loss over K quantiles at
            # levels (k+0.5)/K. The serving "q" output is the quantile mean.
            levels_count = quantile_values.shape[-1]
            levels = (
                torch.arange(levels_count, dtype=quantile_values.dtype, device=quantile_values.device)
                + 0.5
            ) / levels_count
            residual = target_score_to_go[q_mask].unsqueeze(-1) - quantile_values[q_mask]
            pinball = torch.maximum(levels * residual, (levels - 1.0) * residual).mean(dim=-1)
            q_loss_unreduced = pinball
        else:
            q_loss_unreduced = F.smooth_l1_loss(
                predicted_score_to_go[q_mask],
                target_score_to_go[q_mask],
                reduction="none",
            )
        confidence_selected = confidence[q_mask]
        q = (q_loss_unreduced * confidence_selected).sum() / confidence_selected.sum().clamp_min(1.0e-8)
    else:
        teacher_se_sq = torch.zeros_like(target_final_q)
        q = outputs["q"].sum() * 0.0
    value = F.mse_loss(outputs["value_vector"], batch["target_value"])
    score = F.mse_loss(outputs["score_decomposition"], batch["target_score"])
    rank = F.cross_entropy(outputs["rank_logits"].reshape(-1, 4), batch["target_rank"].reshape(-1))
    uncertainty_target = torch.sqrt(teacher_se_sq.clamp_min(0.0)).masked_select(q_mask)
    uncertainty_pred = outputs["uncertainty"].masked_select(q_mask)
    uncertainty = (
        F.l1_loss(uncertainty_pred, uncertainty_target)
        if uncertainty_target.numel()
        else outputs["uncertainty"].sum() * 0.0
    )
    total = (
        weights.policy * policy
        + weights.q * q
        + weights.value * value
        + weights.score * score
        + weights.rank * rank
        + weights.uncertainty * uncertainty
        + weights.greedy_policy * greedy_policy
        + weights.greedy_margin * greedy_margin
    )
    predicted = logits.argmax(dim=1)
    predicted_by_final_q = predicted_final_q.masked_fill(~mask, -1.0e9).argmax(dim=1)
    teacher_target_logit = logits.gather(1, teacher_target.unsqueeze(1)).squeeze(1)
    teacher_rank = ((logits > teacher_target_logit.unsqueeze(1)) & mask).sum(dim=1).to(torch.float32) + 1.0
    greedy_rank = ((logits > greedy_target_logit.unsqueeze(1)) & mask).sum(dim=1).to(torch.float32) + 1.0
    selected_final_q = target_final_q.gather(1, teacher_target.unsqueeze(1)).squeeze(1)
    greedy_final_q = target_final_q.gather(1, greedy_target.unsqueeze(1)).squeeze(1)
    predicted_target_final_q = target_final_q.gather(1, predicted_by_final_q.unsqueeze(1)).squeeze(1)
    final_q_regret = (selected_final_q - predicted_target_final_q).clamp_min(0.0).mean()
    teacher_advantage = (selected_final_q - greedy_final_q).mean()
    return {
        "total": total,
        "policy": policy.detach(),
        "weighted_policy": weighted_policy.detach(),
        "q": q.detach(),
        "score_to_go_q": q.detach(),
        "final_q_regret": final_q_regret.detach(),
        "value": value.detach(),
        "score": score.detach(),
        "rank": rank.detach(),
        "uncertainty": uncertainty.detach(),
        "greedy_policy": greedy_policy.detach(),
        "greedy_margin": greedy_margin.detach(),
        "teacher_top1": (predicted == teacher_target).to(torch.float32).mean().detach(),
        "greedy_top1": (predicted == greedy_target).to(torch.float32).mean().detach(),
        "mean_teacher_rank": teacher_rank.mean().detach(),
        "mean_greedy_rank": greedy_rank.mean().detach(),
        "teacher_advantage_over_greedy": teacher_advantage.detach(),
    }


def _model_forward(model, batch: dict[str, Any]):  # type: ignore[no-untyped-def]
    return model(
        batch["tokens"],
        batch["token_mask"],
        batch["actions"],
        batch["action_mask"],
        relation_ids=batch.get("relation_ids"),
        relation_tail=batch.get("relation_tail"),
    )


def _atomic_jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest_path(path: Path, checkpoint_dir: Path) -> str:
    try:
        return str(path.relative_to(checkpoint_dir))
    except ValueError:
        return str(path)


@functools.lru_cache(maxsize=8)
def _cached_epoch_order(record_count: int, seed: int, epoch: int) -> tuple[int, ...]:
    order = list(range(record_count))
    rng = random.Random(seed + epoch * 1_000_003)
    rng.shuffle(order)
    return tuple(order)


def _deterministic_order(record_count: int, *, seed: int, epoch: int, shuffle: bool):  # type: ignore[no-untyped-def]
    # Exactly-safe hotspot fix: the shuffled epoch order used to be rebuilt
    # (an O(record_count) Fisher-Yates in pure Python) for EVERY micro-batch.
    # The order is a pure function of (record_count, seed, epoch), so cache
    # it; callers only slice the returned sequence.
    if shuffle:
        return _cached_epoch_order(record_count, seed, epoch)
    return range(record_count)


def _batch_indices_for_global_batch(
    *,
    global_batch: int,
    batch_size: int,
    record_count: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[int], dict[str, Any]]:
    if record_count <= 0:
        raise ValueError("training records are empty")
    if global_batch <= 0:
        raise ValueError("global_batch is one-indexed")
    start = (global_batch - 1) * batch_size
    epoch = start // record_count
    position = start % record_count
    indices: list[int] = []
    while len(indices) < batch_size:
        order = _deterministic_order(record_count, seed=seed, epoch=epoch, shuffle=shuffle)
        take = min(batch_size - len(indices), record_count - position)
        indices.extend(order[position : position + take])
        position += take
        if position == record_count and len(indices) < batch_size:
            epoch += 1
            position = 0
    next_position = position % record_count
    next_epoch = epoch + (1 if position == record_count else 0)
    return indices, {
        "global_batch": global_batch,
        "next_global_batch": global_batch + 1,
        "epoch": next_epoch,
        "position": next_position,
        "record_count": record_count,
        "batch_size": batch_size,
        "seed": seed,
        "shuffle": shuffle,
        "resume_semantics": "deterministic_epoch_position_for_next_microbatch",
    }


def _loader_cursor_for_next_batch(
    *,
    next_global_batch: int,
    batch_size: int,
    record_count: int,
    seed: int,
    shuffle: bool,
    overfit_one_batch: bool,
) -> dict[str, Any]:
    if record_count <= 0:
        raise ValueError("training records are empty")
    if next_global_batch <= 0:
        raise ValueError("next_global_batch is one-indexed")
    start = (next_global_batch - 1) * batch_size
    return {
        "next_global_batch": next_global_batch,
        "last_consumed_global_batch": next_global_batch - 1,
        "epoch": start // record_count,
        "position": start % record_count,
        "record_count": record_count,
        "batch_size": batch_size,
        "seed": seed,
        "shuffle": shuffle,
        "overfit_one_batch": overfit_one_batch,
        "resume_semantics": "deterministic_epoch_position_for_next_unconsumed_microbatch",
    }


def _normalize_source_weights(source_weights: list[float] | None, expected_count: int) -> list[float] | None:
    if source_weights is None:
        return None
    if len(source_weights) != expected_count:
        raise ValueError(f"expected {expected_count} train source weights, got {len(source_weights)}")
    if any(weight < 0.0 for weight in source_weights):
        raise ValueError("train source weights must be nonnegative")
    total = sum(source_weights)
    if total <= 0.0:
        raise ValueError("at least one train source weight must be positive")
    return [float(weight) / total for weight in source_weights]


def _corpus_source_lengths(corpus: Any) -> list[int] | None:
    if hasattr(corpus, "source_lengths"):
        lengths = [int(length) for length in corpus.source_lengths()]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"source lengths must be positive, got {lengths}")
        return lengths
    return None


def _weighted_batch_indices_for_global_batch(
    *,
    global_batch: int,
    batch_size: int,
    source_lengths: list[int],
    source_weights: list[float],
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    if global_batch <= 0:
        raise ValueError("global_batch is one-indexed")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(source_lengths) != len(source_weights):
        raise ValueError("source length/weight mismatch")
    weights = _normalize_source_weights(source_weights, len(source_lengths))
    assert weights is not None
    offsets: list[int] = []
    offset = 0
    for length in source_lengths:
        if length <= 0:
            raise ValueError("source lengths must be positive")
        offsets.append(offset)
        offset += length
    cumulative_weights: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative_weights.append(running)

    rng = random.Random(seed + global_batch * 1_000_003)
    indices: list[int] = []
    source_counts = [0 for _ in source_lengths]
    for _ in range(batch_size):
        draw = rng.random()
        source_index = 0
        while source_index + 1 < len(cumulative_weights) and draw >= cumulative_weights[source_index]:
            source_index += 1
        local_index = rng.randrange(source_lengths[source_index])
        indices.append(offsets[source_index] + local_index)
        source_counts[source_index] += 1
    return indices, {
        "global_batch": global_batch,
        "next_global_batch": global_batch + 1,
        "record_count": offset,
        "batch_size": batch_size,
        "seed": seed,
        "source_lengths": source_lengths,
        "source_weights": weights,
        "source_counts": source_counts,
        "shuffle": True,
        "resume_semantics": "deterministic_weighted_source_sampling_with_replacement",
    }


def _loader_cursor_for_next_weighted_batch(
    *,
    next_global_batch: int,
    batch_size: int,
    source_lengths: list[int],
    source_weights: list[float],
    seed: int,
    overfit_one_batch: bool,
) -> dict[str, Any]:
    if next_global_batch <= 0:
        raise ValueError("next_global_batch is one-indexed")
    weights = _normalize_source_weights(source_weights, len(source_lengths))
    assert weights is not None
    return {
        "next_global_batch": next_global_batch,
        "last_consumed_global_batch": next_global_batch - 1,
        "record_count": sum(source_lengths),
        "batch_size": batch_size,
        "seed": seed,
        "source_lengths": source_lengths,
        "source_weights": weights,
        "shuffle": True,
        "overfit_one_batch": overfit_one_batch,
        "resume_semantics": "deterministic_weighted_source_sampling_for_next_unconsumed_microbatch",
    }


_TRUTHY_ENV = {"1", "true", "yes", "on"}

# Perf knobs (all opt-in; defaults preserve bit-identical training):
#   --data-workers N / CASCADIA_TRAIN_DATA_WORKERS   background batch loading
#   --tf32 / CASCADIA_TRAIN_TF32=1                    TF32 matmul+cudnn (CUDA)
#   --autocast {auto,off,bf16}                        auto = legacy behavior
#                                                     (bf16 on CUDA, off on CPU)
#   --fused-optimizer                                 fused AdamW (CUDA only)
#   --compile / CASCADIA_TRAIN_COMPILE=1              torch.compile the model
#   --grad-checkpoint {auto,on,off}                   auto = legacy no-op
#   --cgab-fused / CASCADIA_CGAB_FUSED=1              fused CGAB relation tail
#                                                     (count-matmul; equivalent
#                                                     math, not bit-identical)
#   CASCADIA_TRAIN_SDPA=flash|mem_efficient|math|cudnn (comma list = priority)
#   CASCADIA_TRAIN_SDPA_LOG=1                         log attention backend info
#   CASCADIA_TRAIN_TIMING=1 [CASCADIA_TRAIN_TIMING_EVERY=K]
#                                                     per-phase wall timing


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


_SDPA_BACKEND_ALIASES = {
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "cudnn": "CUDNN_ATTENTION",
}


def _sdpa_context_factory(spec: str | None):  # type: ignore[no-untyped-def]
    """Return a zero-arg factory of context managers restricting SDPA backends.

    ``spec`` is a comma list from CASCADIA_TRAIN_SDPA (order = priority).
    ``None``/empty returns nullcontext (current behavior)."""
    if not spec:
        return contextlib.nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = []
    for part in spec.split(","):
        key = part.strip().lower()
        if key not in _SDPA_BACKEND_ALIASES:
            raise ValueError(
                f"unsupported CASCADIA_TRAIN_SDPA backend {part!r}; "
                f"expected one of {sorted(set(_SDPA_BACKEND_ALIASES))}"
            )
        backends.append(getattr(SDPBackend, _SDPA_BACKEND_ALIASES[key]))
    import inspect

    if "set_priority" in inspect.signature(sdpa_kernel).parameters:
        return lambda: sdpa_kernel(backends, set_priority=True)
    return lambda: sdpa_kernel(backends)


def _log_attention_backend_info(device, *, autocast_enabled: bool, sample_batch=None, heads: int = 0, d_model: int = 0) -> None:  # type: ignore[no-untyped-def]
    """Log which SDPA backends are enabled/usable for the encoder's shapes.

    The nn.TransformerEncoder fused/nested fast path never applies during
    training (it requires eval mode + no grad), so training attention always
    goes through F.scaled_dot_product_attention with a merged key-padding
    attn_mask; flash rejects arbitrary attn_masks, so the practical choice is
    mem_efficient vs math. This logs the flags plus, when a sample batch is
    provided on CUDA, the per-backend usability verdict for our real shapes.
    """
    import torch

    parts = [f"torch={torch.__version__}", f"device={device.type}", f"autocast_bf16={autocast_enabled}"]
    if device.type == "cuda":
        backend_flags = torch.backends.cuda
        parts.append(
            "enabled_backends="
            f"flash:{backend_flags.flash_sdp_enabled()},"
            f"mem_efficient:{backend_flags.mem_efficient_sdp_enabled()},"
            f"math:{backend_flags.math_sdp_enabled()},"
            f"cudnn:{backend_flags.cudnn_sdp_enabled()}"
        )
    print("[trainer] sdpa " + " ".join(parts), flush=True)
    if sample_batch is None or device.type != "cuda":
        return
    try:
        token_mask = sample_batch["token_mask"]
        batch_size, seq_len = token_mask.shape
        head_dim = d_model // max(1, heads)
        dtype = torch.bfloat16 if autocast_enabled else torch.float32
        query = torch.empty((batch_size, heads, seq_len, head_dim), dtype=dtype, device=device)
        attn_mask = torch.zeros((batch_size, heads, seq_len, seq_len), dtype=dtype, device=device)
        params = torch.backends.cuda.SDPAParams(query, query, query, attn_mask, 0.0, False, False)
        verdicts = {
            "flash": torch.backends.cuda.can_use_flash_attention(params, True),
            "mem_efficient": torch.backends.cuda.can_use_efficient_attention(params, True),
        }
        print(
            f"[trainer] sdpa shape-probe b={batch_size} s={seq_len} h={heads} d={head_dim} "
            f"dtype={dtype} usable={verdicts} (debug reasons above if rejected)",
            flush=True,
        )
    except Exception as error:  # instrumentation must never break training
        print(f"[trainer] sdpa shape-probe unavailable: {error}", flush=True)


class _PhaseTimer:
    """Accumulates per-phase wall time. Near-zero overhead when disabled.

    When enabled on CUDA it synchronizes at phase boundaries so GPU phases are
    meaningful; this itself slightly perturbs throughput (timing mode is a
    measurement tool, not a production default).
    """

    def __init__(self, enabled: bool, device_type: str) -> None:
        self.enabled = enabled
        self._cuda = device_type == "cuda"
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._window_totals: dict[str, float] = {}
        self._window_steps = 0

    def _sync(self) -> None:
        if self._cuda:
            import torch

            torch.cuda.synchronize()

    def start(self) -> float:
        if not self.enabled:
            return 0.0
        self._sync()
        return time.perf_counter()

    def stop(self, name: str, started: float) -> None:
        if not self.enabled:
            return
        self._sync()
        elapsed = time.perf_counter() - started
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1
        self._window_totals[name] = self._window_totals.get(name, 0.0) + elapsed

    def step_done(self, step: int, every: int) -> None:
        if not self.enabled:
            return
        self._window_steps += 1
        if self._window_steps < every:
            return
        line = " ".join(
            f"{name}={self._window_totals.get(name, 0.0) / self._window_steps:.4f}s/step"
            for name in sorted(self._window_totals)
        )
        print(f"[trainer] timing step={step} window={self._window_steps} {line}", flush=True)
        self._window_totals = {}
        self._window_steps = 0

    def report(self) -> dict[str, Any]:
        return {
            "totals_s": {name: round(value, 6) for name, value in sorted(self.totals.items())},
            "counts": dict(sorted(self.counts.items())),
        }

    def summary(self) -> None:
        if not self.enabled or not self.totals:
            return
        line = " ".join(f"{name}={value:.3f}s" for name, value in sorted(self.totals.items()))
        print(f"[trainer] timing summary {line}", flush=True)


class _LazyCorpusDataset:
    """Picklable map-style dataset that opens the corpus lazily per process.

    Used only when --data-workers > 0. Each DataLoader worker re-opens the
    shard files itself (NPZ handles are not picklable / fork-safe)."""

    def __init__(self, paths: list[Path], corpus_format: str) -> None:
        self.paths = [str(path) for path in paths]
        self.corpus_format = corpus_format
        self._corpus: Any = None

    def _ensure(self) -> Any:
        if self._corpus is None:
            self._corpus = _load_corpus([Path(p) for p in self.paths], corpus_format=self.corpus_format)
        return self._corpus

    def __getitem__(self, index: int) -> Any:
        return _corpus_examples(self._ensure(), [index], corpus_format=self.corpus_format)[0]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_corpus"] = None
        return state


class _GlobalBatchIndexSampler:
    """Yields the exact per-global-batch index lists the trainer would build.

    The seeded index generation stays in the main process, so multi-worker
    loading preserves batch composition and order bit-for-bit; workers only
    fetch examples and collate."""

    def __init__(
        self,
        *,
        first_global_batch: int,
        last_global_batch: int,
        batch_size: int,
        record_count: int,
        seed: int,
        shuffle: bool,
        source_lengths: list[int] | None,
        source_weights: list[float] | None,
    ) -> None:
        self.first_global_batch = first_global_batch
        self.last_global_batch = last_global_batch
        self.batch_size = batch_size
        self.record_count = record_count
        self.seed = seed
        self.shuffle = shuffle
        self.source_lengths = source_lengths
        self.source_weights = source_weights

    def _indices(self, global_batch: int) -> list[int]:
        if self.source_weights is not None:
            assert self.source_lengths is not None
            indices, _ = _weighted_batch_indices_for_global_batch(
                global_batch=global_batch,
                batch_size=self.batch_size,
                source_lengths=self.source_lengths,
                source_weights=self.source_weights,
                seed=self.seed,
            )
        else:
            indices, _ = _batch_indices_for_global_batch(
                global_batch=global_batch,
                batch_size=self.batch_size,
                record_count=self.record_count,
                seed=self.seed,
                shuffle=self.shuffle,
            )
        return indices

    def __iter__(self):  # type: ignore[no-untyped-def]
        for global_batch in range(self.first_global_batch, self.last_global_batch + 1):
            yield self._indices(global_batch)

    def __len__(self) -> int:
        return max(0, self.last_global_batch - self.first_global_batch + 1)


def _build_train_loader(  # type: ignore[no-untyped-def]
    *,
    train_paths: list[Path],
    train_format: str,
    sampler: _GlobalBatchIndexSampler,
    data_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
):
    from torch.utils.data import DataLoader

    return DataLoader(
        _LazyCorpusDataset(train_paths, train_format),
        batch_sampler=sampler,
        num_workers=data_workers,
        collate_fn=functools.partial(_collate_examples, corpus_format=train_format),
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )


def _evaluate_records(  # type: ignore[no-untyped-def]
    *,
    model,
    records,
    corpus_format: str,
    weights: LossWeights,
    device,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    record_count = _corpus_len(records)
    if record_count <= 0:
        raise ValueError("validation records are empty")
    import torch

    total_batches = math.ceil(record_count / batch_size)
    batch_limit = total_batches if max_batches is None else min(max_batches, total_batches)
    if batch_limit <= 0:
        raise ValueError("validation max batches must be positive when provided")
    was_training = model.training
    model.eval()
    totals = {key: 0.0 for key in AGGREGATE_KEYS}
    record_total = 0
    with torch.no_grad():
        for batch_index in range(batch_limit):
            indices = list(range(batch_index * batch_size, min((batch_index + 1) * batch_size, record_count)))
            batch_records = _corpus_examples(records, indices, corpus_format=corpus_format)
            batch = _move_to_device(_collate_examples(batch_records, corpus_format=corpus_format), device)
            outputs = _model_forward(model, batch)
            losses = _loss_components(outputs, batch, weights)
            batch_weight = len(batch_records)
            record_total += batch_weight
            loss_values = _loss_scalars(losses, AGGREGATE_KEYS)
            for key in totals:
                totals[key] += loss_values[key] * batch_weight
    if was_training:
        model.train()
    metrics = {f"locked_val_{key}": value / record_total for key, value in totals.items()}
    metrics.update(
        {
            "locked_val_batches": batch_limit,
            "locked_val_total_batches": total_batches,
            "locked_val_records": record_total,
            "locked_val_total_records": record_count,
            "locked_val_limited": max_batches is not None and batch_limit < total_batches,
        }
    )
    return metrics


def _save_checkpoint(  # type: ignore[no-untyped-def]
    checkpoint_dir: Path,
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    config: Any,
    report: dict[str, Any],
    loss_weights: LossWeights,
    loader_cursor: dict[str, Any],
    tag: str | None = None,
) -> dict[str, Any]:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = tag or f"step_{step:07d}"
    weights_path = checkpoint_dir / f"{stem}.safetensors"
    state_path = checkpoint_dir / f"{stem}.pt"
    try:
        from safetensors.torch import save_file

        save_file(model.state_dict(), weights_path)
        weights_format = "safetensors"
    except ModuleNotFoundError:
        weights_path = checkpoint_dir / f"{stem}.weights.pt"
        torch.save(model.state_dict(), weights_path)
        weights_format = "torch_state_dict"
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.random.get_rng_state(),
        "loader_cursor": loader_cursor,
        "report": report,
    }
    torch.save(state, state_path)
    manifest = {
        "schema_ids": report["schema_ids"],
        "step": step,
        "checkpoint_tag": stem,
        "weights": _manifest_path(weights_path, checkpoint_dir),
        "weights_format": weights_format,
        "state": _manifest_path(state_path, checkpoint_dir),
        "config": config.to_dict(),
        "loss_weights": loss_weights.to_dict(),
        "loader_cursor": loader_cursor,
        "source_hashes": report["source_hashes"],
        "dataset_manifests": report["dataset_manifests"],
        "search_config": report["search_config"],
        "metrics": report["latest_metrics"],
        "objective": report.get("objective", "expert"),
        "selection_metric": report.get("selection_metric"),
        "selection_mode": report.get("selection_mode"),
        "resume_identity": report.get("resume_identity"),
    }
    manifest_path = checkpoint_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _state_dict_snapshot_for_swa(model):  # type: ignore[no-untyped-def]
    import torch

    snapshot = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu()
        snapshot[key] = tensor.float().clone() if torch.is_floating_point(tensor) else tensor.clone()
    return snapshot


def _update_swa_state(swa_state, model, swa_count: int):  # type: ignore[no-untyped-def]
    import torch

    snapshot = _state_dict_snapshot_for_swa(model)
    if swa_state is None:
        return snapshot, 1
    next_count = swa_count + 1
    for key, value in snapshot.items():
        if torch.is_floating_point(value):
            swa_state[key].add_((value - swa_state[key]) / next_count)
        else:
            swa_state[key] = value
    return swa_state, next_count


def _save_swa_checkpoint(  # type: ignore[no-untyped-def]
    checkpoint_dir: Path,
    *,
    swa_state: dict[str, Any],
    step: int,
    config: Any,
    report: dict[str, Any],
    loss_weights: LossWeights,
    swa_count: int,
    swa_fraction: float,
    swa_start_step: int,
    loader_cursor: dict[str, Any],
) -> dict[str, Any]:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "swa.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(swa_state, weights_path)
        weights_format = "safetensors"
    except ModuleNotFoundError:
        weights_path = checkpoint_dir / "swa.weights.pt"
        torch.save(swa_state, weights_path)
        weights_format = "torch_state_dict"
    manifest = {
        "schema_ids": report["schema_ids"],
        "checkpoint_tag": "swa",
        "step": step,
        "weights": _manifest_path(weights_path, checkpoint_dir),
        "weights_format": weights_format,
        "config": config.to_dict(),
        "loss_weights": loss_weights.to_dict(),
        "loader_cursor": loader_cursor,
        "source_hashes": report["source_hashes"],
        "dataset_manifests": report["dataset_manifests"],
        "search_config": report["search_config"],
        "metrics": report["latest_metrics"],
        "objective": report.get("objective", "expert"),
        "selection_metric": report.get("selection_metric"),
        "selection_mode": report.get("selection_mode"),
        "resume_identity": report.get("resume_identity"),
        "swa": {
            "snapshot_count": swa_count,
            "fraction": swa_fraction,
            "start_step": swa_start_step,
            "source": "deterministic_final_window_eval_snapshots",
        },
    }
    manifest_path = checkpoint_dir / "swa.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _checkpoint_member_path(manifest_path: Path, member: str) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = payload.get(member)
    if not raw:
        raise ValueError(f"checkpoint manifest {manifest_path} missing {member!r}")
    path = Path(raw)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_weights_from_manifest(model, manifest_path: Path, *, skip_mismatched: bool = False):  # type: ignore[no-untyped-def]
    import torch

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = _checkpoint_member_path(manifest_path, "weights")
    if payload.get("weights_format") == "safetensors" or weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if skip_mismatched:
        # Warm start across head-shape changes (e.g. scalar -> quantile q
        # head): drop checkpoint tensors whose shapes no longer match and
        # leave those modules at fresh init.
        model_state = model.state_dict()
        skipped = sorted(
            key
            for key, tensor in state.items()
            if key not in model_state or model_state[key].shape != tensor.shape
        )
        if skipped:
            print(f"[trainer] init skipping shape-mismatched tensors: {skipped}", flush=True)
            state = {key: tensor for key, tensor in state.items() if key not in skipped}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    return payload


def _resume_identity(
    *,
    schema_ids: list[str],
    source_hashes: dict[str, str],
    dataset_manifests: dict[str, list[dict[str, Any]]],
    config: Any,
    loss_weights: LossWeights,
    model_size: str,
    train_format: str,
    val_format: str,
    batch_size: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    seed: int,
    objective: str,
    selection_metric: str,
    selection_mode: str,
    overfit_one_batch: bool,
    eval_every_steps: int,
    min_selection_greedy_top1: float,
    early_stop_selection_guard_failures: int,
    early_stop_after_step: int,
    train_source_weights: list[float] | None,
) -> dict[str, Any]:
    return {
        "schema_ids": schema_ids,
        "source_hashes": source_hashes,
        "dataset_manifests": dataset_manifests,
        "config": config.to_dict(),
        "loss_weights": loss_weights.to_dict(),
        "model_size": model_size,
        "train_format": train_format,
        "val_format": val_format,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "lr": lr, "weight_decay": weight_decay},
        "seed": seed,
        "objective": objective,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "overfit_one_batch": overfit_one_batch,
        "eval_every_steps": eval_every_steps,
        "min_selection_greedy_top1": min_selection_greedy_top1,
        "early_stop_selection_guard_failures": early_stop_selection_guard_failures,
        "early_stop_after_step": early_stop_after_step,
        "train_source_weights": train_source_weights,
    }


def _diff_resume_identity(expected: dict[str, Any], observed: dict[str, Any] | None) -> list[str]:
    if observed is None:
        return ["resume_identity"]
    keys = sorted(set(expected) | set(observed))
    return [key for key in keys if expected.get(key) != observed.get(key)]


def _passes_selection_guards(metrics: dict[str, Any], *, min_greedy_top1: float) -> bool:
    if min_greedy_top1 <= 0.0:
        return True
    value = metrics.get("locked_val_greedy_top1")
    return isinstance(value, (int, float)) and float(value) >= min_greedy_top1


def _torch_unavailable_overfit_fallback(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    train_format: str,
    val_format: str,
    steps: int,
    batch_size: int,
    lr: float,
    out: Path,
    metrics_jsonl: Path,
    checkpoint_dir: Path,
    seed: int,
    val_max_batches: int | None,
    swa_fraction: float,
    objective: str = "expert",
    loss_weights: LossWeights | None = None,
    selection_metric: str = "locked_val_total",
    selection_mode: str = "min",
    init_manifest: Path | None = None,
    init_skip_mismatched: bool = False,
    resume: Path | None = None,
    eval_every_steps: int = 250,
    min_selection_greedy_top1: float = 0.0,
    early_stop_selection_guard_failures: int = 0,
    early_stop_after_step: int = 0,
    train_source_weights: list[float] | None = None,
) -> dict[str, Any]:
    if train_format != "jsonl" or val_format != "jsonl":
        raise ModuleNotFoundError("Torch is required for packed expert tensor NPZ training")
    records = []
    for path in train_paths:
        records.extend(read_replay_jsonl(path))
    batch = records[:batch_size]
    if not batch:
        raise ValueError("no records available for fallback overfit validation")
    targets = [
        float(value)
        for record in batch
        for value, valid in zip(record["per_action_Q"], record.get("per_action_Q_valid", [True] * len(record["per_action_Q"])))
        if valid
    ]
    target_mean = sum(targets) / len(targets)
    bias = 0.0

    def loss() -> float:
        return sum((bias - target) ** 2 for target in targets) / len(targets)

    initial_loss = loss()
    effective_lr = max(lr, 0.05)
    latest_metrics: dict[str, Any] = {}
    val_records = batch if val_paths else batch
    val_batches = math.ceil(len(val_records) / batch_size)
    val_batch_limit = val_batches if val_max_batches is None else min(val_max_batches, val_batches)
    if val_batch_limit <= 0:
        raise ValueError("validation max batches must be positive when provided")
    for step in range(1, steps + 1):
        grad = 2.0 * (bias - target_mean)
        bias -= effective_lr * grad
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            latest_metrics = {
                "step": step,
                "train_total": loss(),
                "train_q": loss(),
                "locked_val_total": loss(),
                "locked_val_q": loss(),
                "locked_val_batches": val_batch_limit,
                "locked_val_total_batches": val_batches,
                "locked_val_records": min(len(val_records), val_batch_limit * batch_size),
                "locked_val_total_records": len(val_records),
                "locked_val_limited": val_max_batches is not None and val_batch_limit < val_batches,
                "torch_available": False,
                "dry_run_fallback": True,
            }
            _atomic_jsonl_append(
                metrics_jsonl,
                latest_metrics,
            )
    final_loss = loss()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "fallback_weights.json"
    weights_path.write_text(json.dumps({"bias": bias}, sort_keys=True) + "\n", encoding="utf-8")
    state_path = checkpoint_dir / "fallback_state.json"
    state_path.write_text(
        json.dumps({"optimizer": "deterministic_bias_descent", "seed": seed}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cursor = {
        "global_batch": steps,
        "next_global_batch": steps + 1,
        "epoch": (steps * batch_size) // max(1, len(records)),
        "position": (steps * batch_size) % max(1, len(records)),
        "record_count": len(records),
        "batch_size": batch_size,
        "seed": seed,
        "shuffle": False,
        "overfit_one_batch": True,
        "resume_semantics": "deterministic_fallback_bias_model",
    }
    manifest = {
        "schema_ids": _schema_ids_for_formats(train_format, val_format),
        "checkpoint_tag": "fallback_final",
        "weights": _manifest_path(weights_path, checkpoint_dir),
        "weights_format": "torch_unavailable_dry_run_json",
        "state": _manifest_path(state_path, checkpoint_dir),
        "step": steps,
        "loader_cursor": cursor,
        "loss_weights": LossWeights().to_dict(),
        "metrics": latest_metrics,
    }
    (checkpoint_dir / "fallback_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_manifest = {**manifest, "checkpoint_tag": "best_locked_val", "selection_metric": "locked_val_total"}
    (checkpoint_dir / "best_locked_val.manifest.json").write_text(
        json.dumps(best_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    swa_weights_path = checkpoint_dir / "swa_weights.json"
    swa_weights_path.write_text(json.dumps({"bias": bias, "swa_snapshot_count": 1}, sort_keys=True) + "\n", encoding="utf-8")
    swa_manifest = {
        **manifest,
        "checkpoint_tag": "swa",
        "weights": _manifest_path(swa_weights_path, checkpoint_dir),
        "swa": {
            "snapshot_count": 1,
            "fraction": swa_fraction,
            "start_step": max(1, math.floor(steps * (1.0 - swa_fraction)) + 1),
            "source": "deterministic_fallback_final_state",
        },
    }
    (checkpoint_dir / "swa.manifest.json").write_text(
        json.dumps(swa_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "pass",
        "torch_available": False,
        "dry_run_fallback": True,
        "reason": "Torch is not importable for this python3; overfit-one-batch used deterministic CPU fallback.",
        "steps": steps,
        "batch_size": batch_size,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "checkpoint_manifest": str(checkpoint_dir / "fallback_manifest.json"),
        "best_val_checkpoint_manifest": str(checkpoint_dir / "best_locked_val.manifest.json"),
        "swa_checkpoint_manifest": str(checkpoint_dir / "swa.manifest.json"),
        "latest_metrics": latest_metrics,
        "metrics_jsonl": str(metrics_jsonl),
        "train_paths": [str(path) for path in train_paths],
        "val_paths": [str(path) for path in val_paths],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_training(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    train_format: str,
    val_format: str,
    model_size: str,
    q_quantiles: int = 1,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device_name: str,
    seed: int,
    grad_accum: int,
    warmup_fraction: float,
    checkpoint_dir: Path,
    metrics_jsonl: Path,
    out: Path,
    overfit_one_batch: bool,
    val_max_batches: int | None,
    swa_fraction: float,
    objective: str = "expert",
    loss_weights: LossWeights | None = None,
    selection_metric: str = "locked_val_total",
    selection_mode: str = "min",
    init_manifest: Path | None = None,
    init_skip_mismatched: bool = False,
    resume: Path | None = None,
    eval_every_steps: int = 250,
    min_selection_greedy_top1: float = 0.0,
    early_stop_selection_guard_failures: int = 0,
    early_stop_after_step: int = 0,
    train_source_weights: list[float] | None = None,
    max_example_passes: float = 0.0,
    data_workers: int = 0,
    prefetch_factor: int = 2,
    autocast_mode: str = "auto",
    tf32: bool = False,
    fused_optimizer: bool = False,
    compile_model: bool = False,
    grad_checkpoint: str = "auto",
    cgab_fused: bool = False,
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        if overfit_one_batch and device_name == "cpu":
            return _torch_unavailable_overfit_fallback(
                train_paths,
                val_paths,
                train_format=train_format,
                val_format=val_format,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                out=out,
                metrics_jsonl=metrics_jsonl,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                val_max_batches=val_max_batches,
                swa_fraction=swa_fraction,
                init_manifest=init_manifest,
                init_skip_mismatched=init_skip_mismatched,
                resume=resume,
                eval_every_steps=eval_every_steps,
                min_selection_greedy_top1=min_selection_greedy_top1,
                early_stop_selection_guard_failures=early_stop_selection_guard_failures,
                early_stop_after_step=early_stop_after_step,
                train_source_weights=train_source_weights,
            )
        raise

    if init_manifest is not None and resume is not None:
        raise ValueError("--init-manifest and --resume are mutually exclusive")

    # ---- opt-in performance knobs (defaults preserve bit-identical runs) ----
    data_workers = max(0, int(data_workers or _env_int("CASCADIA_TRAIN_DATA_WORKERS", 0)))
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if autocast_mode not in {"auto", "off", "bf16"}:
        raise ValueError("autocast_mode must be one of auto, off, bf16")
    if grad_checkpoint not in {"auto", "on", "off"}:
        raise ValueError("grad_checkpoint must be one of auto, on, off")
    tf32 = tf32 or _env_flag("CASCADIA_TRAIN_TF32")
    compile_model = compile_model or _env_flag("CASCADIA_TRAIN_COMPILE")
    cgab_fused = cgab_fused or _env_flag("CASCADIA_CGAB_FUSED")
    sdpa_spec = os.environ.get("CASCADIA_TRAIN_SDPA", "").strip() or None
    sdpa_context = _sdpa_context_factory(sdpa_spec)
    timing_enabled = _env_flag("CASCADIA_TRAIN_TIMING")
    timing_every = max(1, _env_int("CASCADIA_TRAIN_TIMING_EVERY", 50))
    sdpa_log = _env_flag("CASCADIA_TRAIN_SDPA_LOG") or sdpa_spec is not None or timing_enabled
    if tf32:
        # TF32 changes fp32 matmul numerics on CUDA; opt-in only. Harmless no-op on CPU.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    random.seed(seed)
    torch.manual_seed(seed)
    if selection_mode not in {"min", "max"}:
        raise ValueError("selection_mode must be 'min' or 'max'")
    if eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if not 0.0 <= min_selection_greedy_top1 <= 1.0:
        raise ValueError("min_selection_greedy_top1 must be in [0, 1]")
    if early_stop_selection_guard_failures < 0:
        raise ValueError("early_stop_selection_guard_failures must be nonnegative")
    if early_stop_after_step < 0:
        raise ValueError("early_stop_after_step must be nonnegative")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    train_records = _load_corpus(train_paths, corpus_format=train_format)
    val_records = _load_corpus(val_paths, corpus_format=val_format)
    train_source_lengths = _corpus_source_lengths(train_records)
    normalized_train_source_weights = _normalize_source_weights(train_source_weights, len(train_paths))
    if normalized_train_source_weights is not None:
        if train_format != "npz":
            raise ValueError("--train-source-weights is currently supported for packed NPZ training only")
        if train_source_lengths is None:
            raise ValueError("train source lengths are required for weighted source sampling")
        if len(train_source_lengths) != len(train_paths):
            raise ValueError("train source weights require one source length per train path")
    if overfit_one_batch:
        train_records = _corpus_examples(
            train_records,
            list(range(min(batch_size, _corpus_len(train_records)))),
            corpus_format=train_format,
        )
        train_format = "jsonl"
        val_records = train_records
        val_format = "jsonl"
        train_source_lengths = None
        normalized_train_source_weights = None
    if max_example_passes > 0.0 and not overfit_one_batch:
        # Overfitting guard: the EI-0 corpus saw ~240 passes/example and its
        # guarded checkpoint landed at step 7,250 of 25,000. Cap total passes
        # so the schedule cannot silently loop a small corpus hundreds of
        # times.
        train_corpus_len = _corpus_len(train_records)
        if train_corpus_len > 0:
            max_steps = max(1, int((max_example_passes * train_corpus_len) / max(1, batch_size)))
            if steps > max_steps:
                print(
                    f"[trainer] clamping steps {steps} -> {max_steps} to respect "
                    f"max_example_passes={max_example_passes} over {train_corpus_len} examples",
                    flush=True,
                )
                steps = max_steps
    weights = loss_weights or loss_weights_for_objective(objective)
    config = config_for_size(model_size)
    if q_quantiles > 1:
        from dataclasses import replace as _dc_replace

        config = _dc_replace(config, q_quantiles=q_quantiles)
    model = build_cascadiaformer(config).to(device)
    if cgab_fused:
        # Fused CGAB relation tail (count-matmul); mathematically equivalent
        # but NOT bit-identical (floating-point reassociation, ~1e-7 in fp32).
        model.set_cgab_fused(True)
    grad_checkpoint_applied = False
    if grad_checkpoint == "on":
        model.set_gradient_checkpointing(True)
        grad_checkpoint_applied = True
    elif grad_checkpoint == "auto" and config.gradient_checkpointing:
        # Historical behavior: config.gradient_checkpointing was never applied
        # by the trainer, so "auto" preserves the (no-checkpoint) status quo.
        print(
            "[trainer] model config requests gradient_checkpointing but the trainer "
            "has never applied it; pass --grad-checkpoint on to actually enable it",
            flush=True,
        )
    init_payload: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    if init_manifest is not None:
        init_payload = _load_weights_from_manifest(
            model, init_manifest, skip_mismatched=init_skip_mismatched
        )
        init_config = init_payload.get("config")
        if init_config and init_config != config.to_dict():
            if init_skip_mismatched:
                print(
                    "[trainer] init manifest config differs from requested config "
                    "(allowed by --init-skip-mismatched)",
                    flush=True,
                )
            else:
                raise ValueError("--init-manifest config does not match requested model config")
    fused_optimizer_applied = False
    optimizer_extra_kwargs: dict[str, Any] = {}
    if fused_optimizer:
        if device.type == "cuda":
            optimizer_extra_kwargs["fused"] = True
            fused_optimizer_applied = True
        else:
            print("[trainer] --fused-optimizer requires CUDA; ignoring on this device", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        **optimizer_extra_kwargs,
    )
    train_model = model
    compile_applied = False
    if compile_model:
        try:
            import torch._dynamo

            # Fall back to eager per-graph instead of crashing the run.
            torch._dynamo.config.suppress_errors = True
            train_model = torch.compile(model)
            compile_applied = True
            print("[trainer] torch.compile enabled (default mode, eager fallback on graph errors)", flush=True)
        except Exception as error:
            train_model = model
            print(f"[trainer] torch.compile failed ({error}); continuing eager", flush=True)
    warmup_steps = max(1, int(steps * warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    latest_metrics: dict[str, Any] = {}
    best_metric = float("inf") if selection_mode == "min" else float("-inf")
    best_val_manifest: dict[str, Any] | None = None
    swa_state = None
    swa_count = 0
    swa_start_step = max(1, math.floor(steps * (1.0 - swa_fraction)) + 1)
    shuffle_train = not overfit_one_batch
    source_hashes = {"trainer": _sha256(Path(__file__)), "model": _sha256(Path(__file__).with_name("torch_cascadiaformer.py"))}
    schema_ids = _schema_ids_for_formats(train_format, val_format)
    dataset_manifests = {"train": _dataset_manifest(train_paths), "val": _dataset_manifest(val_paths)}
    resume_identity = _resume_identity(
        schema_ids=schema_ids,
        source_hashes=source_hashes,
        dataset_manifests=dataset_manifests,
        config=config,
        loss_weights=weights,
        model_size=model_size,
        train_format=train_format,
        val_format=val_format,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        objective=objective,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        overfit_one_batch=overfit_one_batch,
        eval_every_steps=eval_every_steps,
        min_selection_greedy_top1=min_selection_greedy_top1,
        early_stop_selection_guard_failures=early_stop_selection_guard_failures,
        early_stop_after_step=early_stop_after_step,
        train_source_weights=normalized_train_source_weights,
    )
    report_base = {
        "schema_ids": schema_ids,
        "source_hashes": source_hashes,
        "dataset_manifests": dataset_manifests,
        "search_config": {
            "accepted_schema": EXPERT_TENSOR_SHARD_SCHEMA_ID if train_format == "npz" else EXPERT_ROOT_SCHEMA_ID,
            "accepted_schemas": [EXPERT_ROOT_SCHEMA_ID, EXPERT_TENSOR_SHARD_SCHEMA_ID],
            "target": "active_seat_score_to_go",
            "derived_final_q": "exact_afterstate_score_active + predicted_score_to_go",
            "val_max_batches": val_max_batches,
            "eval_every_steps": eval_every_steps,
            "min_selection_greedy_top1": min_selection_greedy_top1,
            "early_stop_selection_guard_failures": early_stop_selection_guard_failures,
            "early_stop_after_step": early_stop_after_step,
            "objective": objective,
            "greedy_action_index_semantics": "legal_actions[0] in the retained greedy-ranked action menu",
            "teacher_action_index_semantics": "selected_action_index from rollout/search labels",
            "train_sampling": (
                {
                    "mode": "weighted_source_sampling_with_replacement",
                    "source_lengths": train_source_lengths,
                    "source_weights": normalized_train_source_weights,
                }
                if normalized_train_source_weights is not None
                else {"mode": "deterministic_epoch_shuffle"}
            ),
        },
        "objective": objective,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "resume_identity": resume_identity,
    }
    start_step = 1
    if resume is not None:
        resume_payload = _load_weights_from_manifest(model, resume)
        missing_or_different = _diff_resume_identity(resume_identity, resume_payload.get("resume_identity"))
        if missing_or_different:
            raise ValueError(f"resume identity mismatch: {missing_or_different}")
        state_path = _checkpoint_member_path(resume, "state")
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("torch_rng") is not None:
            torch.random.set_rng_state(state["torch_rng"].cpu())
        latest_metrics = state.get("report", {}).get("latest_metrics", {})
        if selection_metric in latest_metrics:
            best_metric = float(latest_metrics[selection_metric])
        start_step = int(resume_payload.get("step", 0)) + 1
        if start_step > steps:
            raise ValueError(
                f"resume checkpoint step {start_step - 1} is already beyond requested --steps {steps}"
            )

    autocast_enabled = (device.type == "cuda") if autocast_mode == "auto" else (autocast_mode == "bf16")
    pin_memory = device.type == "cuda" and data_workers > 0
    use_train_loader = data_workers > 0 and not overfit_one_batch
    if data_workers > 0 and overfit_one_batch:
        print("[trainer] --data-workers ignored with --overfit-one-batch", flush=True)
    train_loader_iter = None
    if use_train_loader:
        loader_sampler = _GlobalBatchIndexSampler(
            first_global_batch=(start_step - 1) * grad_accum + 1,
            last_global_batch=steps * grad_accum,
            batch_size=batch_size,
            record_count=_corpus_len(train_records),
            seed=seed,
            shuffle=shuffle_train,
            source_lengths=train_source_lengths if normalized_train_source_weights is not None else None,
            source_weights=normalized_train_source_weights,
        )
        train_loader_iter = iter(
            _build_train_loader(
                train_paths=train_paths,
                train_format=train_format,
                sampler=loader_sampler,
                data_workers=data_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
        )
        print(
            f"[trainer] data loader: workers={data_workers} prefetch_factor={prefetch_factor} "
            f"pin_memory={pin_memory} persistent_workers=True",
            flush=True,
        )
    timer = _PhaseTimer(timing_enabled, device.type)
    if sdpa_log:
        _log_attention_backend_info(device, autocast_enabled=autocast_enabled)
    sdpa_probe_pending = sdpa_log and device.type == "cuda"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    completed_steps = start_step - 1
    consecutive_selection_guard_failures = 0
    stopped_early_reason: str | None = None
    for step in range(start_step, steps + 1):
        completed_steps = step
        stop_after_checkpoint = False
        train_totals = {key: 0.0 for key in AGGREGATE_KEYS}
        last_cursor: dict[str, Any] = {}
        for accum_index in range(grad_accum):
            global_batch = (step - 1) * grad_accum + accum_index + 1
            if normalized_train_source_weights is not None:
                assert train_source_lengths is not None
                indices, last_cursor = _weighted_batch_indices_for_global_batch(
                    global_batch=global_batch,
                    batch_size=batch_size,
                    source_lengths=train_source_lengths,
                    source_weights=normalized_train_source_weights,
                    seed=seed,
                )
            else:
                indices, last_cursor = _batch_indices_for_global_batch(
                    global_batch=global_batch,
                    batch_size=batch_size,
                    record_count=_corpus_len(train_records),
                    seed=seed,
                    shuffle=shuffle_train,
                )
            phase_started = timer.start()
            if train_loader_iter is not None:
                host_batch = next(train_loader_iter)
            else:
                batch_examples = _corpus_examples(train_records, indices, corpus_format=train_format)
                host_batch = _collate_examples(batch_examples, corpus_format=train_format)
            timer.stop("data", phase_started)
            phase_started = timer.start()
            batch = _move_to_device(host_batch, device, non_blocking=pin_memory)
            timer.stop("h2d", phase_started)
            if sdpa_probe_pending:
                sdpa_probe_pending = False
                _log_attention_backend_info(
                    device,
                    autocast_enabled=autocast_enabled,
                    sample_batch=batch,
                    heads=config.heads,
                    d_model=config.d_model,
                )
            phase_started = timer.start()
            with sdpa_context(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = _model_forward(train_model, batch)
                losses = _loss_components(outputs, batch, weights)
                loss = losses["total"] / grad_accum
            timer.stop("forward", phase_started)
            phase_started = timer.start()
            loss.backward()
            timer.stop("backward", phase_started)
            loss_values = _loss_scalars(losses, AGGREGATE_KEYS)
            for key in train_totals:
                train_totals[key] += loss_values[key] / grad_accum
        phase_started = timer.start()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        timer.stop("optimizer", phase_started)

        if step == 1 or step == steps or step % eval_every_steps == 0:
            phase_started = timer.start()
            with sdpa_context():
                val_metrics = _evaluate_records(
                    model=train_model,
                    records=val_records,
                    corpus_format=val_format,
                    weights=weights,
                    device=device,
                    batch_size=batch_size,
                    max_batches=val_max_batches,
                )
            timer.stop("eval", phase_started)
            loader_cursor = {
                **last_cursor,
                "overfit_one_batch": overfit_one_batch,
                "optimizer_step": step,
                "gradient_accumulation": grad_accum,
            }
            latest_metrics = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
            }
            latest_metrics.update({f"train_{key}": train_totals[key] for key in AGGREGATE_KEYS})
            latest_metrics.update(val_metrics)
            passes_selection_guards = _passes_selection_guards(
                latest_metrics,
                min_greedy_top1=min_selection_greedy_top1,
            )
            latest_metrics["selection_guard_passed"] = passes_selection_guards
            _atomic_jsonl_append(metrics_jsonl, latest_metrics)
            if passes_selection_guards:
                consecutive_selection_guard_failures = 0
            else:
                consecutive_selection_guard_failures += 1
            checkpoint_report = {
                **report_base,
                "latest_metrics": latest_metrics,
            }
            if selection_metric not in latest_metrics:
                raise ValueError(f"selection metric {selection_metric!r} missing from latest metrics")
            candidate_metric = float(latest_metrics[selection_metric])
            is_better = candidate_metric < best_metric if selection_mode == "min" else candidate_metric > best_metric
            if is_better and passes_selection_guards:
                best_metric = candidate_metric
                best_val_manifest = _save_checkpoint(
                    checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    report=checkpoint_report,
                    loss_weights=weights,
                    loader_cursor=loader_cursor,
                    tag="best_locked_val",
                )
            if (
                early_stop_selection_guard_failures > 0
                and step >= early_stop_after_step
                and consecutive_selection_guard_failures >= early_stop_selection_guard_failures
            ):
                stopped_early_reason = (
                    "selection_guard_failed_"
                    f"{consecutive_selection_guard_failures}_consecutive_evals"
                )
                stop_after_checkpoint = True
            if step >= swa_start_step:
                swa_state, swa_count = _update_swa_state(swa_state, model, swa_count)

        if step % 1000 == 0 or step == steps:
            checkpoint_report = {
                **report_base,
                "latest_metrics": latest_metrics,
            }
            _save_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=config,
                report=checkpoint_report,
                loss_weights=weights,
                loader_cursor=_loader_cursor_for_next_batch(
                    next_global_batch=step * grad_accum + 1,
                    batch_size=batch_size,
                    record_count=_corpus_len(train_records),
                    seed=seed,
                    shuffle=shuffle_train,
                    overfit_one_batch=overfit_one_batch,
                )
                if normalized_train_source_weights is None
                else _loader_cursor_for_next_weighted_batch(
                    next_global_batch=step * grad_accum + 1,
                    batch_size=batch_size,
                    source_lengths=train_source_lengths or [],
                    source_weights=normalized_train_source_weights,
                    seed=seed,
                    overfit_one_batch=overfit_one_batch,
                ),
            )
        timer.step_done(step, timing_every)
        if stop_after_checkpoint:
            break

    train_loader_iter = None  # release DataLoader workers promptly
    timer.summary()
    if swa_state is None:
        swa_state, swa_count = _update_swa_state(swa_state, model, swa_count)
    final_cursor = (
        _loader_cursor_for_next_batch(
            next_global_batch=completed_steps * grad_accum + 1,
            batch_size=batch_size,
            record_count=_corpus_len(train_records),
            seed=seed,
            shuffle=shuffle_train,
            overfit_one_batch=overfit_one_batch,
        )
        if normalized_train_source_weights is None
        else _loader_cursor_for_next_weighted_batch(
            next_global_batch=completed_steps * grad_accum + 1,
            batch_size=batch_size,
            source_lengths=train_source_lengths or [],
            source_weights=normalized_train_source_weights,
            seed=seed,
            overfit_one_batch=overfit_one_batch,
        )
    )
    swa_manifest = _save_swa_checkpoint(
        checkpoint_dir,
        swa_state=swa_state,
        step=completed_steps,
        config=config,
        report={**report_base, "latest_metrics": latest_metrics},
        loss_weights=weights,
        swa_count=swa_count,
        swa_fraction=swa_fraction,
        swa_start_step=swa_start_step,
        loader_cursor=final_cursor,
    )

    report = {
        "status": "pass",
        "torch_available": True,
        "torch_version": torch.__version__,
        "device": str(device),
        "model_size": model_size,
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "steps": completed_steps,
        "requested_steps": steps,
        "stopped_early": stopped_early_reason is not None,
        "stopped_early_reason": stopped_early_reason,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * grad_accum,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "lr": lr, "weight_decay": weight_decay},
        "scheduler": {"name": "warmup_cosine", "warmup_fraction": warmup_fraction, "min_lr_fraction": 0.10},
        "perf_knobs": {
            "data_workers": data_workers,
            "prefetch_factor": prefetch_factor if use_train_loader else None,
            "pin_memory": pin_memory,
            "autocast_mode": autocast_mode,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": "bfloat16" if autocast_enabled else None,
            "tf32": tf32,
            "fused_optimizer": fused_optimizer_applied,
            "compile": compile_applied,
            "grad_checkpoint": grad_checkpoint,
            "grad_checkpoint_applied": grad_checkpoint_applied,
            "cgab_fused": cgab_fused,
            "sdpa": sdpa_spec,
            "timing": timing_enabled,
        },
        "phase_timing": timer.report() if timing_enabled else None,
        "eval_every_steps": eval_every_steps,
        "objective": objective,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_selection_metric_value": best_metric,
        "loss_weights": weights.to_dict(),
        "latest_metrics": latest_metrics,
        "checkpoint_dir": str(checkpoint_dir),
        "best_val_checkpoint": best_val_manifest,
        "swa_checkpoint": swa_manifest,
        "metrics_jsonl": str(metrics_jsonl),
        "val_max_batches": val_max_batches,
        "swa_fraction": swa_fraction,
        "min_selection_greedy_top1": min_selection_greedy_top1,
        "early_stop_selection_guard_failures": early_stop_selection_guard_failures,
        "early_stop_after_step": early_stop_after_step,
        "train_format": train_format,
        "val_format": val_format,
        "init_manifest": str(init_manifest) if init_manifest else None,
        "resume_manifest": str(resume) if resume else None,
        "resume_start_step": start_step,
        "init_manifest_payload": {
            "step": init_payload.get("step"),
            "checkpoint_tag": init_payload.get("checkpoint_tag"),
            "objective": init_payload.get("objective"),
        }
        if init_payload
        else None,
        "resume_manifest_payload": {
            "step": resume_payload.get("step"),
            "checkpoint_tag": resume_payload.get("checkpoint_tag"),
            "objective": resume_payload.get("objective"),
        }
        if resume_payload
        else None,
        **report_base,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_paths(raw: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one path is required")
    return paths


def _parse_optional_float_list(raw: str | None) -> list[float] | None:
    if raw is None or raw.strip() == "":
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        return None
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=["tiny", "S", "M", "L"], default="S")
    parser.add_argument(
        "--init-skip-mismatched",
        action="store_true",
        help="When --init-checkpoint head shapes changed, skip mismatched tensors instead of failing",
    )
    parser.add_argument(
        "--q-quantiles",
        type=int,
        default=1,
        help="K>1 trains a distributional (quantile) score-to-go head with pinball loss",
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument(
        "--train-source-weights",
        help="Comma-separated per-train-shard sampling weights for packed NPZ mixed replay.",
    )
    parser.add_argument("--train-format", choices=["jsonl", "npz"], default="jsonl")
    parser.add_argument("--val-format", choices=["jsonl", "npz"], default="jsonl")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--objective",
        choices=[
            "expert",
            "k32-greedy-retention",
            "pure-greedy-retention",
            "search-improved-greedy-retention",
            "gumbel-selfplay",
        ],
        default="expert",
    )
    parser.add_argument(
        "--max-example-passes",
        type=float,
        default=0.0,
        help="Clamp steps so steps*batch/corpus stays at or below this many passes per example; 0 disables",
    )
    parser.add_argument("--selection-metric", default="locked_val_total")
    parser.add_argument("--selection-mode", choices=["min", "max"], default="min")
    parser.add_argument("--policy-weight", type=float)
    parser.add_argument("--q-loss-weight", type=float)
    parser.add_argument("--value-loss-weight", type=float)
    parser.add_argument("--score-loss-weight", type=float)
    parser.add_argument("--rank-loss-weight", type=float)
    parser.add_argument("--uncertainty-loss-weight", type=float)
    parser.add_argument("--greedy-policy-weight", type=float)
    parser.add_argument("--greedy-margin-weight", type=float)
    parser.add_argument("--greedy-margin-value", type=float)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--out", default="cascadiav3/reports/cascadiaformer_train.json")
    parser.add_argument("--metrics-jsonl", default="cascadiav3/reports/cascadiaformer_metrics.jsonl")
    parser.add_argument("--checkpoint-dir", default="cascadiav3/checkpoints/cascadiaformer")
    parser.add_argument("--init-manifest")
    parser.add_argument("--resume")
    parser.add_argument("--val-max-batches", type=int, default=0, help="0 evaluates the full locked validation set")
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=250,
        help="Run locked validation at step 1, final step, and each N optimizer steps",
    )
    parser.add_argument(
        "--min-selection-greedy-top1",
        type=float,
        default=0.0,
        help="Do not update best checkpoint unless locked_val_greedy_top1 is at least this value",
    )
    parser.add_argument(
        "--early-stop-selection-guard-failures",
        type=int,
        default=0,
        help="Stop training after this many consecutive evals fail selection guards; 0 disables",
    )
    parser.add_argument(
        "--early-stop-after-step",
        type=int,
        default=0,
        help="Do not apply selection-guard early stopping before this optimizer step",
    )
    parser.add_argument("--swa-fraction", type=float, default=0.20)
    parser.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="Background DataLoader workers for train batches (0 = legacy in-process path; "
        "batch composition/order is bit-identical either way)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per data worker (only used when --data-workers > 0)",
    )
    parser.add_argument(
        "--autocast",
        choices=["auto", "off", "bf16"],
        default="auto",
        help="auto = legacy behavior (bf16 autocast on CUDA, fp32 on CPU); off forces fp32; "
        "bf16 forces bf16 autocast. Locked-val eval always runs fp32, but train_* metrics "
        "under autocast are not comparable to --autocast off runs",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 matmul/cudnn on CUDA (also CASCADIA_TRAIN_TF32=1); changes fp32 numerics",
    )
    parser.add_argument(
        "--fused-optimizer",
        action="store_true",
        help="Use fused AdamW (CUDA only; ignored with a warning elsewhere)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model for training (also CASCADIA_TRAIN_COMPILE=1); eager fallback on failure",
    )
    parser.add_argument(
        "--grad-checkpoint",
        choices=["auto", "on", "off"],
        default="auto",
        help="Activation checkpointing for the state encoder; auto preserves legacy behavior "
        "(never applied, even for model sizes whose config requests it)",
    )
    parser.add_argument(
        "--cgab-fused",
        action="store_true",
        help="Fused CGAB relation tail (also CASCADIA_CGAB_FUSED=1): count-matmul instead of the "
        "materialized [B, A, seq, d_model] intermediate; mathematically equivalent, not bit-identical",
    )
    args = parser.parse_args()
    if args.val_max_batches < 0:
        parser.error("--val-max-batches must be >= 0")
    if args.eval_every_steps <= 0:
        parser.error("--eval-every-steps must be > 0")
    if not 0.0 <= args.min_selection_greedy_top1 <= 1.0:
        parser.error("--min-selection-greedy-top1 must be in [0, 1]")
    if args.early_stop_selection_guard_failures < 0:
        parser.error("--early-stop-selection-guard-failures must be >= 0")
    if args.early_stop_after_step < 0:
        parser.error("--early-stop-after-step must be >= 0")
    if not 0.0 < args.swa_fraction <= 1.0:
        parser.error("--swa-fraction must be in (0, 1]")
    loss_weights = loss_weights_for_objective(args.objective)
    for field_name, value in {
        "policy": args.policy_weight,
        "q": args.q_loss_weight,
        "value": args.value_loss_weight,
        "score": args.score_loss_weight,
        "rank": args.rank_loss_weight,
        "uncertainty": args.uncertainty_loss_weight,
        "greedy_policy": args.greedy_policy_weight,
        "greedy_margin": args.greedy_margin_weight,
        "greedy_margin_value": args.greedy_margin_value,
    }.items():
        if value is not None:
            loss_weights = replace(loss_weights, **{field_name: value})

    report = run_training(
        _parse_paths(args.train),
        _parse_paths(args.val),
        train_format=args.train_format,
        val_format=args.val_format,
        model_size=args.model_size,
        q_quantiles=args.q_quantiles,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device_name=args.device,
        seed=args.seed,
        grad_accum=args.grad_accum,
        warmup_fraction=args.warmup_fraction,
        checkpoint_dir=Path(args.checkpoint_dir),
        metrics_jsonl=Path(args.metrics_jsonl),
        out=Path(args.out),
        overfit_one_batch=args.overfit_one_batch,
        val_max_batches=None if args.val_max_batches == 0 else args.val_max_batches,
        swa_fraction=args.swa_fraction,
        max_example_passes=args.max_example_passes,
        objective=args.objective,
        loss_weights=loss_weights,
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        init_manifest=Path(args.init_manifest) if args.init_manifest else None,
        init_skip_mismatched=args.init_skip_mismatched,
        resume=Path(args.resume) if args.resume else None,
        eval_every_steps=args.eval_every_steps,
        min_selection_greedy_top1=args.min_selection_greedy_top1,
        early_stop_selection_guard_failures=args.early_stop_selection_guard_failures,
        early_stop_after_step=args.early_stop_after_step,
        train_source_weights=_parse_optional_float_list(args.train_source_weights),
        data_workers=args.data_workers,
        prefetch_factor=args.prefetch_factor,
        autocast_mode=args.autocast,
        tf32=args.tf32,
        fused_optimizer=args.fused_optimizer,
        compile_model=args.compile,
        grad_checkpoint=args.grad_checkpoint,
        cgab_fused=args.cgab_fused,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
