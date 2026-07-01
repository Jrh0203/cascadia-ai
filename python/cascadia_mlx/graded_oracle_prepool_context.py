"""Pre-pool candidate caches and linear-memory context probes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import socket
import time
from collections.abc import Iterator
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
from cascadia_mlx.graded_oracle_embedding_probe import (
    balanced_group_binary_loss,
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
    encode_graded_oracle_prepool_batch,
)

EXPERIMENT_ID = "complete-action-frontier-prepool-context-v1"
CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "graded-oracle-frozen-prepool-candidates-v1"
CANDIDATE_ONLY = "candidate-only"
LEGACY_CONTEXT = "legacy-context"
MOMENT_CONTEXT = "moment-context"
SCREEN_TOP64_CONTEXT = "screen-top64-context"
PROBE_KINDS = (
    CANDIDATE_ONLY,
    LEGACY_CONTEXT,
    MOMENT_CONTEXT,
    SCREEN_TOP64_CONTEXT,
)
PROBE_SEEDS = {
    CANDIDATE_ONLY: 2026061613,
    LEGACY_CONTEXT: 2026061614,
    MOMENT_CONTEXT: 2026061615,
    SCREEN_TOP64_CONTEXT: 2026061616,
}
PROBE_EPOCHS = 20
PROBE_LEARNING_RATE = 3e-4
PROBE_WEIGHT_DECAY = 1e-4
PROBE_HIDDEN_DIM = 256
SCREEN_CONTEXT_WIDTH = 64


@dataclass(frozen=True)
class PrepoolCacheBatch:
    candidates_path: Path
    metadata_path: Path
    groups: int
    candidates: int


class FrozenPrepoolCache:
    """Manifest-backed post-candidate-projection vectors."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "cache.json").read_text())
        if (
            self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or self.manifest.get("cache_schema") != CACHE_SCHEMA
        ):
            raise ValueError("unsupported prepool candidate cache")
        self.split = str(self.manifest["split"])
        self.candidate_dim = int(self.manifest["candidate_dim"])
        self.group_count = int(self.manifest["groups"])
        self.candidate_count = int(self.manifest["candidates"])
        self.payload_blake3 = str(self.manifest["payload_blake3"])
        self.batches = tuple(
            PrepoolCacheBatch(
                candidates_path=self.root / entry["candidates_file"],
                metadata_path=self.root / entry["metadata_file"],
                groups=int(entry["groups"]),
                candidates=int(entry["candidates"]),
            )
            for entry in self.manifest["batches"]
        )
        if sum(batch.groups for batch in self.batches) != self.group_count:
            raise ValueError("prepool cache group total drifted")
        if sum(batch.candidates for batch in self.batches) != self.candidate_count:
            raise ValueError("prepool cache candidate total drifted")
        if verify_checksums:
            for batch, entry in zip(
                self.batches,
                self.manifest["batches"],
                strict=True,
            ):
                if (
                    checksum(batch.candidates_path)
                    != entry["candidates_blake3"]
                    or checksum(batch.metadata_path)
                    != entry["metadata_blake3"]
                ):
                    raise ValueError("prepool cache checksum mismatch")
            if prepool_payload_blake3(self.manifest) != self.payload_blake3:
                raise ValueError("prepool cache payload identity drifted")

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
            candidates = np.load(batch.candidates_path, mmap_mode="r")
            with np.load(batch.metadata_path) as loaded:
                metadata = {name: loaded[name] for name in loaded.files}
            yield candidates, metadata


def prepool_payload_blake3(manifest: dict[str, Any]) -> str:
    """Hash only portable scientific cache identity."""
    payload = {
        "cache_schema": manifest["cache_schema"],
        "split": manifest["split"],
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_blake3": manifest["dataset_manifest_blake3"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_manifest_blake3": manifest["checkpoint_manifest_blake3"],
        "model_blake3": manifest["model_blake3"],
        "candidate_dim": manifest["candidate_dim"],
        "groups": manifest["groups"],
        "candidates": manifest["candidates"],
        "batches": [
            {
                "index": entry["index"],
                "candidates_file": entry["candidates_file"],
                "metadata_file": entry["metadata_file"],
                "candidates_blake3": entry["candidates_blake3"],
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


def export_prepool_cache(
    *,
    dataset_root: Path,
    checkpoint_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export every open action's exact pre-pool candidate vector."""
    if output_root.exists():
        raise ValueError("prepool cache output already exists")
    dataset = GradedOracleDataset(dataset_root, verify_checksums=True)
    if dataset.split not in {"train", "validation"}:
        raise ValueError("prepool cache accepts only open splits")
    model = load_frontier_warm_start(checkpoint_dir)
    model.eval()
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
        encoded = encode_graded_oracle_prepool_batch(model, batch)
        mx.eval(encoded)
        candidate_values = np.asarray(encoded, dtype=np.float32)
        candidate_mask = np.asarray(batch.candidate_mask)
        counts = np.sum(candidate_mask, axis=1, dtype=np.int64)
        offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts)]
        )
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
        candidate_relative = (
            Path("batches") / f"batch-{batch_index:06d}-candidates.npy"
        )
        metadata_relative = (
            Path("batches") / f"batch-{batch_index:06d}-metadata.npz"
        )
        candidate_path = temporary / candidate_relative
        metadata_path = temporary / metadata_relative
        with candidate_path.open("wb") as handle:
            np.save(
                handle,
                _flatten_valid(candidate_values, counts),
                allow_pickle=False,
            )
        with metadata_path.open("wb") as handle:
            np.savez(handle, **flattened)
        entries.append(
            {
                "index": batch_index,
                "candidates_file": candidate_relative.as_posix(),
                "metadata_file": metadata_relative.as_posix(),
                "groups": len(counts),
                "candidates": int(offsets[-1]),
                "candidates_bytes": candidate_path.stat().st_size,
                "metadata_bytes": metadata_path.stat().st_size,
                "candidates_blake3": checksum(candidate_path),
                "metadata_blake3": checksum(metadata_path),
            }
        )
        groups += len(counts)
        candidates += int(offsets[-1])

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
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
        "candidate_point": "candidate_projection_pre_pool",
        "candidate_dtype": "float32",
        "candidate_dim": model.config.hidden_dim,
        "groups": groups,
        "candidates": candidates,
        "batches": entries,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        },
        "features_contain_targets_or_teacher_values": False,
        "test_split_opened": False,
    }
    manifest["payload_blake3"] = prepool_payload_blake3(manifest)
    _write_json_atomic(temporary / "cache.json", manifest)
    os.replace(temporary, output_root)
    return manifest


def _flatten_valid(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values[index, :count] for index, count in enumerate(counts)],
        axis=0,
    )


def context_input_dim(kind: str, candidate_dim: int) -> int:
    if kind == CANDIDATE_ONLY:
        return candidate_dim
    if kind == LEGACY_CONTEXT:
        return candidate_dim * 4
    if kind in {MOMENT_CONTEXT, SCREEN_TOP64_CONTEXT}:
        return candidate_dim * 7
    raise ValueError("unsupported prepool context kind")


def stable_screen_topk_indices(
    screen_rank: np.ndarray,
    action_hash: np.ndarray,
    width: int = SCREEN_CONTEXT_WIDTH,
) -> np.ndarray:
    """Return deterministic observable screen landmarks."""
    ranks = np.asarray(screen_rank)
    hashes = np.asarray(action_hash)
    if ranks.ndim != 1 or hashes.shape != (len(ranks), 32):
        raise ValueError("screen landmark arrays have inconsistent shapes")
    if len(ranks) == 0 or width <= 0:
        raise ValueError("screen landmark selection requires candidates and width")
    keys = (
        *(hashes[:, column] for column in range(31, -1, -1)),
        ranks,
    )
    order = np.lexsort(keys)
    return order[: min(width, len(order))].astype(np.int32, copy=False)


def build_context_features(
    kind: str,
    candidates: mx.array,
    metadata: dict[str, np.ndarray],
) -> mx.array:
    """Build one frozen permutation-equivariant candidate context."""
    offsets = tuple(int(value) for value in metadata["group_offsets"])
    if candidates.ndim != 2:
        raise ValueError("prepool candidates must be a rank-two array")
    outputs = []
    for start, end in pairwise(offsets):
        group = candidates[start:end]
        if kind == CANDIDATE_ONLY:
            outputs.append(group)
            continue
        mean = mx.mean(group, axis=0, keepdims=True)
        maximum = mx.max(group, axis=0, keepdims=True)
        if kind == LEGACY_CONTEXT:
            outputs.append(
                mx.concatenate(
                    [
                        group,
                        mx.broadcast_to(mean, group.shape),
                        mx.broadcast_to(maximum, group.shape),
                        group - mean,
                    ],
                    axis=-1,
                )
            )
            continue
        if kind == MOMENT_CONTEXT:
            minimum = mx.min(group, axis=0, keepdims=True)
            standard_deviation = mx.sqrt(
                mx.mean((group - mean) ** 2, axis=0, keepdims=True)
            )
            outputs.append(
                mx.concatenate(
                    [
                        group,
                        mx.broadcast_to(mean, group.shape),
                        mx.broadcast_to(maximum, group.shape),
                        mx.broadcast_to(minimum, group.shape),
                        mx.broadcast_to(standard_deviation, group.shape),
                        group - mean,
                        group - maximum,
                    ],
                    axis=-1,
                )
            )
            continue
        if kind == SCREEN_TOP64_CONTEXT:
            indices = stable_screen_topk_indices(
                metadata["screen_rank"][start:end],
                metadata["action_hash"][start:end],
            )
            landmarks = group[mx.array(indices)]
            landmark_mean = mx.mean(landmarks, axis=0, keepdims=True)
            landmark_maximum = mx.max(landmarks, axis=0, keepdims=True)
            outputs.append(
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
            continue
        raise ValueError("unsupported prepool context kind")
    features = mx.concatenate(outputs, axis=0)
    expected = context_input_dim(kind, int(candidates.shape[-1]))
    if features.shape[-1] != expected:
        raise AssertionError("prepool context width drifted")
    return features


@dataclass(frozen=True)
class PrepoolProbeConfig:
    kind: str
    seed: int
    epochs: int = PROBE_EPOCHS
    learning_rate: float = PROBE_LEARNING_RATE
    weight_decay: float = PROBE_WEIGHT_DECAY
    hidden_dim: int = PROBE_HIDDEN_DIM

    def validate(self) -> None:
        if self.kind not in PROBE_KINDS:
            raise ValueError("unsupported prepool probe kind")
        if (
            self.seed != PROBE_SEEDS[self.kind]
            or self.epochs != PROBE_EPOCHS
            or self.learning_rate != PROBE_LEARNING_RATE
            or self.weight_decay != PROBE_WEIGHT_DECAY
            or self.hidden_dim != PROBE_HIDDEN_DIM
        ):
            raise ValueError("prepool probe configuration drifted")


class PrepoolContextProbe(nn.Module):
    """One hidden-layer probe shared by all frozen context arms."""

    def __init__(self, input_dim: int, hidden_dim: int = PROBE_HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, features: mx.array) -> mx.array:
        return self.network(features).reshape(-1)


def train_prepool_probe(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
    config: PrepoolProbeConfig,
) -> dict[str, Any]:
    """Train one frozen pre-pool context probe."""
    config.validate()
    if output_root.exists():
        raise ValueError("prepool probe output already exists")
    train_cache = FrozenPrepoolCache(train_cache_root)
    validation_cache = FrozenPrepoolCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("prepool probe cache split mismatch")
    if train_cache.candidate_dim != validation_cache.candidate_dim:
        raise ValueError("prepool candidate dimension mismatch")
    input_dim = context_input_dim(config.kind, train_cache.candidate_dim)
    mx.random.seed(config.seed)
    model = PrepoolContextProbe(input_dim, config.hidden_dim)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, balanced_group_binary_loss)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for candidates, metadata in train_cache.iter_batches(
            shuffle=True,
            seed=config.seed + epoch,
        ):
            features = build_context_features(
                config.kind,
                mx.array(np.asarray(candidates)),
                metadata,
            )
            target = mx.array(metadata["target"])
            source_flags = mx.array(metadata["source_flags"])
            eligible = (
                source_flags.astype(mx.int32)
                & GRADED_SOURCE_CHAMPION_FRONTIER
            ) == 0
            offsets = tuple(int(value) for value in metadata["group_offsets"])
            loss, gradients = loss_and_grad(
                model,
                features,
                target,
                eligible,
                offsets,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            batches += 1
        train_metrics = evaluate_prepool_probe(
            model,
            train_cache,
            config.kind,
        )
        validation_metrics = evaluate_prepool_probe(
            model,
            validation_cache,
            config.kind,
        )
        event = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "train": train_metrics,
            "validation": validation_metrics,
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
    train_metrics = evaluate_prepool_probe(model, train_cache, config.kind)
    validation_metrics = evaluate_prepool_probe(
        model,
        validation_cache,
        config.kind,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "probe": asdict(config),
        "input_dim": input_dim,
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
        },
        "test_split_opened": False,
    }
    _write_json_atomic(output_root / "report.json", report)
    return report


def evaluate_prepool_probe(
    model: nn.Module,
    cache: FrozenPrepoolCache,
    kind: str,
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
    for candidate_values, metadata in cache.iter_batches():
        target = metadata["target"].astype(np.bool_, copy=False)
        source_flags = metadata["source_flags"].astype(np.int32, copy=False)
        eligible = (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        offsets = tuple(int(value) for value in metadata["group_offsets"])
        features = build_context_features(
            kind,
            mx.array(np.asarray(candidate_values)),
            metadata,
        )
        logits = model(features)
        loss = balanced_group_binary_loss(
            model,
            features,
            mx.array(target),
            mx.array(eligible),
            offsets,
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
                (
                    group_flags[retained]
                    & GRADED_SOURCE_CHAMPION_FRONTIER
                )
                == 0
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


def prepool_context_classification(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Classify the smallest candidate-context representation that passes."""
    if set(reports) != set(PROBE_KINDS):
        raise ValueError("prepool classification requires all four probes")
    train_gates = {
        kind: _train_gate(report) for kind, report in reports.items()
    }
    validation_gates = {
        kind: _validation_gate(report) for kind, report in reports.items()
    }
    passing = {
        kind: train_gates[kind] and validation_gates[kind]
        for kind in PROBE_KINDS
    }
    if passing[CANDIDATE_ONLY]:
        classification = "candidate_projection_separable"
    elif passing[LEGACY_CONTEXT]:
        classification = "legacy_output_trunk_collapse"
    elif passing[MOMENT_CONTEXT]:
        classification = "rich_global_context_sufficient"
    elif passing[SCREEN_TOP64_CONTEXT]:
        classification = "screen_frontier_context_sufficient"
    elif any(train_gates.values()):
        classification = "prepool_train_separable_not_generalized"
    else:
        classification = "candidate_projection_insufficient"
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


def load_prepool_probe(
    *,
    kind: str,
    candidate_dim: int,
    weights: Path,
) -> nn.Module:
    """Load one portable pre-pool context probe."""
    model = PrepoolContextProbe(context_input_dim(kind, candidate_dim))
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def evaluate_saved_prepool_probe(
    *,
    kind: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Cross-host replay one context probe on both open splits."""
    train_cache = FrozenPrepoolCache(train_cache_root)
    validation_cache = FrozenPrepoolCache(validation_cache_root)
    if train_cache.candidate_dim != validation_cache.candidate_dim:
        raise ValueError("prepool replay candidate dimension mismatch")
    model = load_prepool_probe(
        kind=kind,
        candidate_dim=train_cache.candidate_dim,
        weights=weights,
    )
    scientific = {
        "kind": kind,
        "input_dim": context_input_dim(kind, train_cache.candidate_dim),
        "weights_blake3": checksum(weights),
        "train_cache_payload_blake3": train_cache.payload_blake3,
        "validation_cache_payload_blake3": validation_cache.payload_blake3,
        "train": evaluate_prepool_probe(model, train_cache, kind),
        "validation": evaluate_prepool_probe(model, validation_cache, kind),
        "test_split_opened": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname().split(".")[0],
        "scientific": scientific,
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
    return export_prepool_cache(
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint,
        output_root=args.output,
    )


def _probe_main(args: argparse.Namespace) -> dict[str, Any]:
    return train_prepool_probe(
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
        output_root=args.output,
        config=PrepoolProbeConfig(
            kind=args.kind,
            seed=PROBE_SEEDS[args.kind],
        ),
    )


def _evaluate_main(args: argparse.Namespace) -> dict[str, Any]:
    report = evaluate_saved_prepool_probe(
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
    elif args.command == "probe":
        report = _probe_main(args)
    else:
        report = _evaluate_main(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
