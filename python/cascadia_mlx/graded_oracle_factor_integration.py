"""Frozen candidate-factor caches and integration probes for ADR 0097."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import socket
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import (
    EXPECTED_WARM_START_CHECKPOINT,
    EXPECTED_WARM_START_MANIFEST_BLAKE3,
    EXPECTED_WARM_START_MODEL_BLAKE3,
    checksum,
    load_frontier_warm_start,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_CANDIDATE_FACTOR_NAMES,
    encode_graded_oracle_factor_batch,
)
from cascadia_mlx.graded_oracle_prepool_context import (
    stable_screen_topk_indices,
)
from cascadia_mlx.model import SetAttentionBlock

EXPERIMENT_ID = "complete-action-frontier-factor-integration-v1"
CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "graded-oracle-frozen-candidate-factors-v1"
FACTOR_COUNT = len(GRADED_ORACLE_CANDIDATE_FACTOR_NAMES)
FACTOR_DIM = 192
FLATTENED_FACTOR_DIM = FACTOR_COUNT * FACTOR_DIM
MINIMUM_FREE_BYTES = 24 * 1024**3

WIDE_CONCAT = "wide-concat"
SCREEN_RELATIVE = "screen-relative"
FACTOR_ATTENTION = "factor-attention"
PAIRWISE_GATED = "pairwise-gated"
PROBE_KINDS = (
    WIDE_CONCAT,
    SCREEN_RELATIVE,
    FACTOR_ATTENTION,
    PAIRWISE_GATED,
)
PROBE_SEEDS = {
    WIDE_CONCAT: 2026061617,
    SCREEN_RELATIVE: 2026061618,
    FACTOR_ATTENTION: 2026061619,
    PAIRWISE_GATED: 2026061620,
}
PROBE_EPOCHS = 20
PROBE_LEARNING_RATE = 3e-4
PROBE_WEIGHT_DECAY = 1e-4
SCREEN_CONTEXT_WIDTH = 64
MLX_CACHE_LIMIT_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class FactorCacheBatch:
    factors_path: Path
    metadata_path: Path
    groups: int
    candidates: int


class FrozenFactorCache:
    """Manifest-backed exact inputs to candidate_projection."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        manifest: Mapping[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.manifest = (
            dict(manifest)
            if manifest is not None
            else json.loads((self.root / "cache.json").read_text())
        )
        if (
            self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or self.manifest.get("cache_schema") != CACHE_SCHEMA
        ):
            raise ValueError("unsupported candidate factor cache")
        self.split = str(self.manifest["split"])
        self.factor_count = int(self.manifest["factor_count"])
        self.factor_dim = int(self.manifest["factor_dim"])
        self.group_count = int(self.manifest["groups"])
        self.candidate_count = int(self.manifest["candidates"])
        self.payload_blake3 = str(self.manifest["payload_blake3"])
        if (
            tuple(self.manifest["factor_names"]) != GRADED_ORACLE_CANDIDATE_FACTOR_NAMES
            or self.factor_count != FACTOR_COUNT
            or self.factor_dim != FACTOR_DIM
        ):
            raise ValueError("candidate factor cache shape drifted")
        self.batches = tuple(
            FactorCacheBatch(
                factors_path=self.root / entry["factors_file"],
                metadata_path=self.root / entry["metadata_file"],
                groups=int(entry["groups"]),
                candidates=int(entry["candidates"]),
            )
            for entry in self.manifest["batches"]
        )
        if sum(batch.groups for batch in self.batches) != self.group_count:
            raise ValueError("factor cache group total drifted")
        if sum(batch.candidates for batch in self.batches) != self.candidate_count:
            raise ValueError("factor cache candidate total drifted")
        if verify_checksums:
            for batch, entry in zip(
                self.batches,
                self.manifest["batches"],
                strict=True,
            ):
                if (
                    checksum(batch.factors_path) != entry["factors_blake3"]
                    or checksum(batch.metadata_path) != entry["metadata_blake3"]
                ):
                    raise ValueError("candidate factor cache checksum mismatch")
            if factor_payload_blake3(self.manifest) != self.payload_blake3:
                raise ValueError("candidate factor cache payload identity drifted")

    def iter_batches(
        self,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        order = np.arange(len(self.batches))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for index_value in order:
            batch = self.batches[int(index_value)]
            factors = np.load(batch.factors_path, mmap_mode="r")
            with np.load(batch.metadata_path) as loaded:
                metadata = {name: loaded[name] for name in loaded.files}
            yield factors, metadata


def configure_mlx_memory(
    cache_limit_bytes: int = MLX_CACHE_LIMIT_BYTES,
) -> dict[str, int]:
    """Bound MLX's free-buffer cache without changing numerical execution."""
    if cache_limit_bytes < 0:
        raise ValueError("MLX cache limit must be non-negative")
    previous_cache_limit_bytes = int(mx.set_cache_limit(cache_limit_bytes))
    mx.clear_cache()
    mx.reset_peak_memory()
    return {
        "cache_limit_bytes": cache_limit_bytes,
        "previous_cache_limit_bytes": previous_cache_limit_bytes,
    }


def mlx_memory_snapshot() -> dict[str, int]:
    """Return allocator-native unified-memory telemetry."""
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def factor_payload_blake3(manifest: dict[str, Any]) -> str:
    """Hash only portable scientific factor-cache identity."""
    payload = {
        "cache_schema": manifest["cache_schema"],
        "split": manifest["split"],
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_blake3": manifest["dataset_manifest_blake3"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_manifest_blake3": manifest["checkpoint_manifest_blake3"],
        "model_blake3": manifest["model_blake3"],
        "factor_names": manifest["factor_names"],
        "factor_count": manifest["factor_count"],
        "factor_dim": manifest["factor_dim"],
        "groups": manifest["groups"],
        "candidates": manifest["candidates"],
        "batches": [
            {
                "index": entry["index"],
                "factors_file": entry["factors_file"],
                "metadata_file": entry["metadata_file"],
                "factors_blake3": entry["factors_blake3"],
                "metadata_blake3": entry["metadata_blake3"],
                "groups": entry["groups"],
                "candidates": entry["candidates"],
            }
            for entry in manifest["batches"]
        ],
    }
    return blake3.blake3(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def export_factor_cache(
    *,
    dataset_root: Path,
    checkpoint_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export every open action's seven exact pre-compression factors."""
    if output_root.exists():
        raise ValueError("candidate factor cache output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(output_root.parent)
    if disk_before.free < MINIMUM_FREE_BYTES:
        raise ValueError("candidate factor cache requires at least 24 GiB free")
    dataset = GradedOracleDataset(dataset_root, verify_checksums=True)
    if dataset.split not in {"train", "validation"}:
        raise ValueError("candidate factor cache accepts only open splits")
    model = load_frontier_warm_start(checkpoint_dir)
    model.eval()
    if model.config.hidden_dim != FACTOR_DIM:
        raise ValueError("candidate factor dimension drifted")

    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "batches").mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    groups = 0
    candidates = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(
        dataset.batches(
            64,
            maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
            maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        )
    ):
        encoded = encode_graded_oracle_factor_batch(model, batch)
        mx.eval(encoded)
        factor_values = np.asarray(encoded, dtype=np.float32)
        candidate_mask = np.asarray(batch.candidate_mask)
        counts = np.sum(candidate_mask, axis=1, dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
        targets = build_frontier_anchored_target_mask(
            r1200_mean=np.asarray(batch.r1200_mean),
            r1200_mask=np.asarray(batch.r1200_mask),
            source_flags=np.asarray(batch.source_flags),
            candidate_mask=candidate_mask,
            action_hashes=np.asarray(batch.action_hash),
        )
        flattened = {
            "group_offsets": offsets,
            "target": _flatten_valid(targets, counts),
            "source_flags": _flatten_valid(
                np.asarray(batch.source_flags),
                counts,
            ),
            "screen_rank": _flatten_valid(
                np.asarray(batch.screen_rank, dtype=np.int32),
                counts,
            ),
            "action_hash": _flatten_valid(
                np.asarray(batch.action_hash),
                counts,
            ),
            "selected_index": np.asarray(batch.selected_index, dtype=np.int32),
            "r4800_mean": _flatten_valid(
                np.asarray(batch.r4800_mean, dtype=np.float32),
                counts,
            ),
            "r4800_mask": _flatten_valid(
                np.asarray(batch.r4800_mask),
                counts,
            ),
        }
        factors_relative = Path("batches") / f"batch-{batch_index:06d}-factors.npy"
        metadata_relative = Path("batches") / f"batch-{batch_index:06d}-metadata.npz"
        factors_path = temporary / factors_relative
        metadata_path = temporary / metadata_relative
        with factors_path.open("wb") as handle:
            np.save(
                handle,
                _flatten_valid(factor_values, counts),
                allow_pickle=False,
            )
        with metadata_path.open("wb") as handle:
            np.savez(handle, **flattened)
        entries.append(
            {
                "index": batch_index,
                "factors_file": factors_relative.as_posix(),
                "metadata_file": metadata_relative.as_posix(),
                "groups": len(counts),
                "candidates": int(offsets[-1]),
                "factors_bytes": factors_path.stat().st_size,
                "metadata_bytes": metadata_path.stat().st_size,
                "factors_blake3": checksum(factors_path),
                "metadata_blake3": checksum(metadata_path),
            }
        )
        groups += len(counts)
        candidates += int(offsets[-1])

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    disk_after = shutil.disk_usage(output_root.parent)
    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname().split(".")[0],
        "split": dataset.split,
        "dataset_root": str(dataset.root.resolve()),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "checkpoint": EXPECTED_WARM_START_CHECKPOINT,
        "checkpoint_manifest_blake3": EXPECTED_WARM_START_MANIFEST_BLAKE3,
        "model_blake3": EXPECTED_WARM_START_MODEL_BLAKE3,
        "candidate_point": "candidate_projection_input_factors",
        "factor_names": list(GRADED_ORACLE_CANDIDATE_FACTOR_NAMES),
        "factor_dtype": "float32",
        "factor_count": FACTOR_COUNT,
        "factor_dim": FACTOR_DIM,
        "flattened_dim": FLATTENED_FACTOR_DIM,
        "groups": groups,
        "candidates": candidates,
        "batches": entries,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "disk_free_before_bytes": disk_before.free,
            "disk_free_after_bytes": disk_after.free,
        },
        "features_contain_targets_or_teacher_values": False,
        "test_split_opened": False,
    }
    manifest["payload_blake3"] = factor_payload_blake3(manifest)
    _write_json_atomic(temporary / "cache.json", manifest)
    os.replace(temporary, output_root)
    return manifest


def _flatten_valid(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values[index, :count] for index, count in enumerate(counts)],
        axis=0,
    )


class WideConcatProbe(nn.Module):
    """Wide dense integration without a 192-dimensional bottleneck."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FLATTENED_FACTOR_DIM, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )

    def __call__(
        self,
        factors: mx.array,
        group_offsets: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        del group_offsets, screen_rank, action_hash
        return self.network(factors.reshape(len(factors), -1)).reshape(-1)


class ScreenRelativeProbe(nn.Module):
    """Pre-compression candidate context around observable screen landmarks."""

    def __init__(self):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(FLATTENED_FACTOR_DIM, 384),
            nn.GELU(),
            nn.LayerNorm(384),
        )
        self.network = nn.Sequential(
            nn.Linear(384 * 7, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 1),
        )

    def __call__(
        self,
        factors: mx.array,
        group_offsets: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        reduced = self.reduction(factors.reshape(len(factors), -1))
        contexts = []
        for start, end in pairwise(group_offsets):
            group = reduced[start:end]
            mean = mx.mean(group, axis=0, keepdims=True)
            maximum = mx.max(group, axis=0, keepdims=True)
            indices = stable_screen_topk_indices(
                screen_rank[start:end],
                action_hash[start:end],
                width=SCREEN_CONTEXT_WIDTH,
            )
            landmarks = group[mx.array(indices)]
            landmark_mean = mx.mean(landmarks, axis=0, keepdims=True)
            landmark_maximum = mx.max(landmarks, axis=0, keepdims=True)
            contexts.append(
                mx.concatenate(
                    [
                        group,
                        mx.broadcast_to(mean, group.shape),
                        mx.broadcast_to(maximum, group.shape),
                        mx.broadcast_to(landmark_mean, group.shape),
                        mx.broadcast_to(landmark_maximum, group.shape),
                        group - landmark_mean,
                        group - landmark_maximum,
                    ],
                    axis=-1,
                )
            )
        return self.network(mx.concatenate(contexts, axis=0)).reshape(-1)


class FactorAttentionProbe(nn.Module):
    """Typed self-attention over the seven candidate factor tokens."""

    def __init__(self):
        super().__init__()
        self.factor_embedding = nn.Embedding(FACTOR_COUNT, FACTOR_DIM)
        self.blocks = [SetAttentionBlock(FACTOR_DIM, 6, 4) for _ in range(2)]
        self.network = nn.Sequential(
            nn.Linear(FACTOR_DIM * 2, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def __call__(
        self,
        factors: mx.array,
        group_offsets: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        del group_offsets, screen_rank, action_hash
        mask = mx.ones((len(factors), FACTOR_COUNT), dtype=mx.bool_)
        values = factors + self.factor_embedding(mx.arange(FACTOR_COUNT))[None]
        for block in self.blocks:
            values = block(values, mask)
        pooled = mx.concatenate(
            [mx.mean(values, axis=1), mx.max(values, axis=1)],
            axis=-1,
        )
        return self.network(pooled).reshape(-1)


class PairwiseGatedProbe(nn.Module):
    """Identity-specific factors with explicit pooled pair interactions."""

    def __init__(self):
        super().__init__()
        self.factor_projections = [
            nn.Sequential(
                nn.Linear(FACTOR_DIM, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
            for _ in range(FACTOR_COUNT)
        ]
        self.gate = nn.Linear(256, 1)
        self.network = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def __call__(
        self,
        factors: mx.array,
        group_offsets: tuple[int, ...],
        screen_rank: np.ndarray,
        action_hash: np.ndarray,
    ) -> mx.array:
        del group_offsets, screen_rank, action_hash
        projected = mx.stack(
            [
                projection(factors[:, index])
                for index, projection in enumerate(self.factor_projections)
            ],
            axis=1,
        )
        gates = mx.softmax(self.gate(projected).reshape(len(factors), -1), axis=1)
        weighted = mx.sum(projected * gates[..., None], axis=1)
        total = mx.sum(projected, axis=1)
        pairwise_interactions = (total * total - mx.sum(projected * projected, axis=1)) / 42.0
        maximum = mx.max(projected, axis=1)
        return self.network(
            mx.concatenate(
                [weighted, pairwise_interactions, maximum],
                axis=-1,
            )
        ).reshape(-1)


@dataclass(frozen=True)
class FactorProbeConfig:
    kind: str
    seed: int
    epochs: int = PROBE_EPOCHS
    learning_rate: float = PROBE_LEARNING_RATE
    weight_decay: float = PROBE_WEIGHT_DECAY

    def validate(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError("unsupported factor integration probe")
        if (
            self.seed != PROBE_SEEDS[self.kind]
            or self.epochs != PROBE_EPOCHS
            or self.learning_rate != PROBE_LEARNING_RATE
            or self.weight_decay != PROBE_WEIGHT_DECAY
        ):
            raise ValueError("factor integration probe configuration drifted")


def build_factor_probe(kind: str) -> nn.Module:
    """Construct one frozen ADR 0097 architecture."""
    if kind == WIDE_CONCAT:
        return WideConcatProbe()
    if kind == SCREEN_RELATIVE:
        return ScreenRelativeProbe()
    if kind == FACTOR_ATTENTION:
        return FactorAttentionProbe()
    if kind == PAIRWISE_GATED:
        return PairwiseGatedProbe()
    raise ValueError("unsupported factor integration probe")


def balanced_factor_binary_loss(
    model: nn.Module,
    factors: mx.array,
    target: mx.array,
    eligible: mx.array,
    group_offsets: tuple[int, ...],
    screen_rank: np.ndarray,
    action_hash: np.ndarray,
) -> mx.array:
    """Give every group and class equal weight regardless of action count."""
    logits = model(
        factors,
        group_offsets,
        screen_rank,
        action_hash,
    )
    losses = []
    for start, end in pairwise(group_offsets):
        group_target = target[start:end]
        group_eligible = eligible[start:end]
        group_logits = logits[start:end]
        negative_mask = group_eligible & ~group_target
        losses.append(
            mx.sum(mx.where(group_target, nn.softplus(-group_logits), 0.0))
            / mx.maximum(mx.sum(group_target), 1)
            + mx.sum(mx.where(negative_mask, nn.softplus(group_logits), 0.0))
            / mx.maximum(mx.sum(negative_mask), 1)
        )
    return mx.mean(mx.stack(losses))


def train_factor_probe(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
    config: FactorProbeConfig,
) -> dict[str, Any]:
    """Train one frozen candidate-factor integration probe."""
    config.validate()
    allocator = configure_mlx_memory()
    if output_root.exists():
        raise ValueError("factor integration probe output already exists")
    train_cache = FrozenFactorCache(train_cache_root)
    validation_cache = FrozenFactorCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("factor integration cache split mismatch")
    mx.random.seed(config.seed)
    model = build_factor_probe(config.kind)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, balanced_factor_binary_loss)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for factor_values, metadata in train_cache.iter_batches(
            shuffle=True,
            seed=config.seed + epoch,
        ):
            factors = mx.array(np.asarray(factor_values))
            target = mx.array(metadata["target"])
            source_flags = mx.array(metadata["source_flags"])
            eligible = (source_flags.astype(mx.int32) & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
            offsets = tuple(int(value) for value in metadata["group_offsets"])
            loss, gradients = loss_and_grad(
                model,
                factors,
                target,
                eligible,
                offsets,
                metadata["screen_rank"],
                metadata["action_hash"],
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            batches += 1
        train_metrics = evaluate_factor_probe(
            model,
            train_cache,
        )
        validation_metrics = evaluate_factor_probe(
            model,
            validation_cache,
        )
        epoch_memory_before_clear = mlx_memory_snapshot()
        mx.clear_cache()
        epoch_memory_after_clear = mlx_memory_snapshot()
        event = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "train": train_metrics,
            "validation": validation_metrics,
            "mlx_memory_before_clear": epoch_memory_before_clear,
            "mlx_memory_after_clear": epoch_memory_after_clear,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        key = (
            float(train_metrics["target_positive_recall"]),
            float(train_metrics["target_set_exact_fraction"]),
            float(validation_metrics["target_positive_recall"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            mx.save_safetensors(
                str(output_root / "best.safetensors"),
                dict(tree_flatten(model.parameters())),
            )
            _write_json_atomic(output_root / "best.json", event)

    model.load_weights(str(output_root / "best.safetensors"))
    mx.eval(model.parameters())
    train_metrics = evaluate_factor_probe(model, train_cache)
    validation_metrics = evaluate_factor_probe(model, validation_cache)
    final_memory_before_clear = mlx_memory_snapshot()
    mx.clear_cache()
    final_memory_after_clear = mlx_memory_snapshot()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "probe": asdict(config),
        "host": socket.gethostname().split(".")[0],
        "best_epoch": best_epoch,
        "train_cache_payload_blake3": train_cache.payload_blake3,
        "validation_cache_payload_blake3": validation_cache.payload_blake3,
        "train": train_metrics,
        "validation": validation_metrics,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "mlx_allocator": allocator,
            "mlx_memory_before_clear": final_memory_before_clear,
            "mlx_memory_after_clear": final_memory_after_clear,
        },
        "test_split_opened": False,
    }
    _write_json_atomic(output_root / "report.json", report)
    return report


def evaluate_factor_probe(
    model: nn.Module,
    cache: FrozenFactorCache,
) -> dict[str, Any]:
    """Measure target separation under the anchored width-64 selector."""
    model.eval()
    groups = 0
    candidates = 0
    target_positives = 0
    target_hits = 0
    exact_sets = 0
    winner_hits = 0
    regret = 0.0
    finite = True
    total_loss = 0.0
    for factor_values, metadata in cache.iter_batches():
        target = metadata["target"].astype(np.bool_, copy=False)
        source_flags = metadata["source_flags"].astype(np.int32, copy=False)
        eligible = (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        offsets = tuple(int(value) for value in metadata["group_offsets"])
        factors = mx.array(np.asarray(factor_values))
        logits = model(
            factors,
            offsets,
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        loss = balanced_factor_binary_loss(
            model,
            factors,
            mx.array(target),
            mx.array(eligible),
            offsets,
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        mx.eval(logits, loss)
        values = np.asarray(logits)
        finite &= bool(np.all(np.isfinite(values)))
        total_loss += float(loss.item()) * (len(offsets) - 1)
        selected_indices = metadata["selected_index"]
        r4800_mean = metadata["r4800_mean"]
        r4800_mask = metadata["r4800_mask"]
        action_hash = metadata["action_hash"]
        for group_index, (start, end) in enumerate(pairwise(offsets)):
            group_scores = values[start:end]
            group_flags = source_flags[start:end]
            group_hashes = action_hash[start:end]
            group_target = target[start:end]
            retained = frontier_anchored_retained_indices(
                scores=group_scores,
                source_flags=group_flags,
                action_hashes=group_hashes,
            )
            retained_nonfrontier = retained[
                (group_flags[retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
            ]
            quota = int(np.sum(group_target))
            recalled = int(np.sum(group_target[retained_nonfrontier]))
            target_positives += quota
            target_hits += recalled
            exact_sets += int(recalled == quota)
            winner = int(selected_indices[group_index])
            winner_hits += int(winner in retained)
            labeled = r4800_mask[start:end]
            retained_labeled = retained[labeled[retained]]
            if np.any(labeled) and len(retained_labeled):
                regret += max(
                    0.0,
                    float(np.max(r4800_mean[start:end][labeled]))
                    - float(np.max(r4800_mean[start:end][retained_labeled])),
                )
            groups += 1
            candidates += end - start
    return {
        "groups": groups,
        "candidates": candidates,
        "balanced_binary_loss": total_loss / groups,
        "target_positives": target_positives,
        "target_positive_recall": target_hits / target_positives,
        "target_set_exact_fraction": exact_sets / groups,
        "top64_r4800_winner_recall": winner_hits / groups,
        "mean_top64_retained_r4800_regret": regret / groups,
        "all_scores_finite": finite,
        "all_groups_scored_once": groups == cache.group_count,
        "all_candidates_scored_once": candidates == cache.candidate_count,
    }


def factor_integration_classification(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Classify the smallest pre-compression integration that passes."""
    if set(reports) != set(PROBE_KINDS):
        raise ValueError("factor integration classification requires all probes")
    train_gates = {kind: _train_gate(report) for kind, report in reports.items()}
    validation_gates = {kind: _validation_gate(report) for kind, report in reports.items()}
    passing = {kind: train_gates[kind] and validation_gates[kind] for kind in PROBE_KINDS}
    if passing[WIDE_CONCAT]:
        classification = "wide_concat_sufficient"
    elif passing[PAIRWISE_GATED]:
        classification = "pairwise_factor_sufficient"
    elif passing[FACTOR_ATTENTION]:
        classification = "factor_attention_sufficient"
    elif passing[SCREEN_RELATIVE]:
        classification = "screen_relative_factor_context_sufficient"
    elif any(train_gates.values()):
        classification = "candidate_factors_train_separable_not_generalized"
    else:
        classification = "candidate_factor_inputs_insufficient"
    return {
        "train_gates": train_gates,
        "validation_gates": validation_gates,
        "classification": classification,
    }


def _train_gate(report: dict[str, Any]) -> bool:
    return (
        float(report["train"]["target_positive_recall"]) >= 0.80
        and float(report["train"]["target_set_exact_fraction"]) >= 0.25
    )


def _validation_gate(report: dict[str, Any]) -> bool:
    return (
        float(report["validation"]["target_positive_recall"]) >= 0.50
        and float(report["validation"]["target_set_exact_fraction"]) >= 0.01
    )


def load_factor_probe(*, kind: str, weights: Path) -> nn.Module:
    """Load one portable candidate-factor integration probe."""
    model = build_factor_probe(kind)
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def evaluate_saved_factor_probe(
    *,
    kind: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Cross-host replay one factor integration probe."""
    allocator = configure_mlx_memory()
    train_cache = FrozenFactorCache(train_cache_root)
    validation_cache = FrozenFactorCache(validation_cache_root)
    model = load_factor_probe(kind=kind, weights=weights)
    train_metrics = evaluate_factor_probe(model, train_cache)
    validation_metrics = evaluate_factor_probe(model, validation_cache)
    memory_before_clear = mlx_memory_snapshot()
    mx.clear_cache()
    memory_after_clear = mlx_memory_snapshot()
    scientific = {
        "kind": kind,
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": train_cache.payload_blake3,
        "validation_cache_payload_blake3": validation_cache.payload_blake3,
        "train": train_metrics,
        "validation": validation_metrics,
        "test_split_opened": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname().split(".")[0],
        "scientific": scientific,
        "execution": {
            "mlx_allocator": allocator,
            "mlx_memory_before_clear": memory_before_clear,
            "mlx_memory_after_clear": memory_after_clear,
        },
        "scientific_blake3": blake3.blake3(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _cache_main(args: argparse.Namespace) -> dict[str, Any]:
    return export_factor_cache(
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint,
        output_root=args.output,
    )


def _probe_main(args: argparse.Namespace) -> dict[str, Any]:
    return train_factor_probe(
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
        output_root=args.output,
        config=FactorProbeConfig(
            kind=args.kind,
            seed=PROBE_SEEDS[args.kind],
        ),
    )


def _evaluate_main(args: argparse.Namespace) -> dict[str, Any]:
    report = evaluate_saved_factor_probe(
        kind=args.kind,
        weights=args.weights,
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
    )
    _write_json_atomic(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--kind", choices=PROBE_KINDS, required=True)
    probe.add_argument("--train-cache", type=Path, required=True)
    probe.add_argument("--validation-cache", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--kind", choices=PROBE_KINDS, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--train-cache", type=Path, required=True)
    evaluate.add_argument("--validation-cache", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "cache":
        report = _cache_main(args)
        compact = {
            "experiment_id": report["experiment_id"],
            "host": report["host"],
            "split": report["split"],
            "groups": report["groups"],
            "candidates": report["candidates"],
            "payload_blake3": report["payload_blake3"],
            "execution": report["execution"],
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
        return
    report = _probe_main(args) if args.command == "probe" else _evaluate_main(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
