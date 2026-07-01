"""Frozen candidate-embedding cache and separability probes."""

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
from cascadia_mlx.graded_oracle_model import encode_graded_oracle_batch

EXPERIMENT_ID = "complete-action-frontier-embedding-separability-v1"
CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "graded-oracle-frozen-candidate-embeddings-v1"
LINEAR_PROBE_SEED = 2026061608
NONLINEAR_PROBE_SEED = 2026061609
PROBE_EPOCHS = 20
LINEAR_PROBE_LEARNING_RATE = 1e-3
NONLINEAR_PROBE_LEARNING_RATE = 3e-4
PROBE_WEIGHT_DECAY = 1e-4
NONLINEAR_HIDDEN_DIM = 128


@dataclass(frozen=True)
class EmbeddingCacheBatch:
    embeddings_path: Path
    metadata_path: Path
    groups: int
    candidates: int


class FrozenEmbeddingCache:
    """Manifest-backed, memory-mapped pre-head candidate embeddings."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "cache.json").read_text())
        if (
            self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or self.manifest.get("cache_schema") != CACHE_SCHEMA
        ):
            raise ValueError("unsupported frozen embedding cache")
        self.split = str(self.manifest["split"])
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.group_count = int(self.manifest["groups"])
        self.candidate_count = int(self.manifest["candidates"])
        self.batches = tuple(
            EmbeddingCacheBatch(
                embeddings_path=self.root / entry["embeddings_file"],
                metadata_path=self.root / entry["metadata_file"],
                groups=int(entry["groups"]),
                candidates=int(entry["candidates"]),
            )
            for entry in self.manifest["batches"]
        )
        if verify_checksums:
            for batch, entry in zip(
                self.batches,
                self.manifest["batches"],
                strict=True,
            ):
                if (
                    checksum(batch.embeddings_path)
                    != entry["embeddings_blake3"]
                    or checksum(batch.metadata_path)
                    != entry["metadata_blake3"]
                ):
                    raise ValueError("frozen embedding cache checksum mismatch")
        if sum(batch.groups for batch in self.batches) != self.group_count:
            raise ValueError("frozen embedding cache group total drifted")
        if sum(batch.candidates for batch in self.batches) != self.candidate_count:
            raise ValueError("frozen embedding cache candidate total drifted")

    def iter_batches(
        self,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        order = np.arange(len(self.batches))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for index in order:
            batch = self.batches[int(index)]
            embeddings = np.load(batch.embeddings_path, mmap_mode="r")
            with np.load(batch.metadata_path) as loaded:
                metadata = {name: loaded[name] for name in loaded.files}
            yield embeddings, metadata


def export_embedding_cache(
    *,
    dataset_root: Path,
    checkpoint_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export every open candidate's exact pre-head representation."""
    if output_root.exists():
        raise ValueError("embedding cache output already exists")
    dataset = GradedOracleDataset(dataset_root, verify_checksums=True)
    if dataset.split not in {"train", "validation"}:
        raise ValueError("embedding cache accepts only open splits")
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
        embeddings = encode_graded_oracle_batch(model, batch)
        mx.eval(embeddings)
        embedding_values = np.asarray(embeddings, dtype=np.float32)
        candidate_mask = np.asarray(batch.candidate_mask)
        counts = np.sum(candidate_mask, axis=1, dtype=np.int64)
        offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts)]
        )
        flattened_embeddings = _flatten_valid(embedding_values, counts)
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
        embedding_relative = Path("batches") / f"batch-{batch_index:06d}-embeddings.npy"
        metadata_relative = Path("batches") / f"batch-{batch_index:06d}-metadata.npz"
        embedding_path = temporary / embedding_relative
        metadata_path = temporary / metadata_relative
        with embedding_path.open("wb") as handle:
            np.save(handle, flattened_embeddings, allow_pickle=False)
        with metadata_path.open("wb") as handle:
            np.savez(handle, **flattened)
        entry = {
            "index": batch_index,
            "embeddings_file": embedding_relative.as_posix(),
            "metadata_file": metadata_relative.as_posix(),
            "groups": len(counts),
            "candidates": int(offsets[-1]),
            "embeddings_bytes": embedding_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
            "embeddings_blake3": checksum(embedding_path),
            "metadata_blake3": checksum(metadata_path),
        }
        entries.append(entry)
        groups += len(counts)
        candidates += int(offsets[-1])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    manifest = {
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
        "embedding_point": "output_trunk_pre_residual_head",
        "embedding_dtype": "float32",
        "embedding_dim": model.config.hidden_dim,
        "groups": groups,
        "candidates": candidates,
        "batches": entries,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        },
        "test_split_opened": False,
    }
    _write_json_atomic(temporary / "cache.json", manifest)
    os.replace(temporary, output_root)
    return manifest


def _flatten_valid(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values[index, :count] for index, count in enumerate(counts)],
        axis=0,
    )


@dataclass(frozen=True)
class EmbeddingProbeConfig:
    kind: str
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float = PROBE_WEIGHT_DECAY
    hidden_dim: int = NONLINEAR_HIDDEN_DIM

    def validate(self) -> None:
        expected = {
            "linear": (
                LINEAR_PROBE_SEED,
                LINEAR_PROBE_LEARNING_RATE,
            ),
            "nonlinear": (
                NONLINEAR_PROBE_SEED,
                NONLINEAR_PROBE_LEARNING_RATE,
            ),
        }
        if self.kind not in expected:
            raise ValueError("probe kind must be linear or nonlinear")
        seed, learning_rate = expected[self.kind]
        if (
            self.seed != seed
            or self.epochs != PROBE_EPOCHS
            or self.learning_rate != learning_rate
            or self.weight_decay != PROBE_WEIGHT_DECAY
            or self.hidden_dim != NONLINEAR_HIDDEN_DIM
        ):
            raise ValueError("embedding probe configuration drifted")


class LinearEmbeddingProbe(nn.Module):
    """One affine target-separability probe."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.head = nn.Linear(embedding_dim, 1)

    def __call__(self, embeddings: mx.array) -> mx.array:
        return self.head(embeddings).reshape(-1)


class NonlinearEmbeddingProbe(nn.Module):
    """One hidden-layer target-separability probe."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, embeddings: mx.array) -> mx.array:
        return self.network(embeddings).reshape(-1)


def balanced_group_binary_loss(
    model: nn.Module,
    embeddings: mx.array,
    target: mx.array,
    eligible: mx.array,
    group_offsets: tuple[int, ...],
) -> mx.array:
    """Give every group and class equal weight regardless of action count."""
    logits = model(embeddings)
    losses = []
    for start, end in pairwise(group_offsets):
        group_target = target[start:end]
        group_eligible = eligible[start:end]
        group_logits = logits[start:end]
        negative_mask = group_eligible & ~group_target
        losses.append(
            mx.sum(mx.where(group_target, nn.softplus(-group_logits), 0.0))
            / mx.maximum(mx.sum(group_target), 1)
            + mx.sum(
                mx.where(negative_mask, nn.softplus(group_logits), 0.0)
            )
            / mx.maximum(mx.sum(negative_mask), 1)
        )
    return mx.mean(mx.stack(losses))


def train_embedding_probe(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
    config: EmbeddingProbeConfig,
) -> dict[str, Any]:
    """Fit one frozen-trunk probe and select only by open train recovery."""
    config.validate()
    if output_root.exists():
        raise ValueError("probe output already exists")
    train_cache = FrozenEmbeddingCache(train_cache_root)
    validation_cache = FrozenEmbeddingCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("probe cache split mismatch")
    if train_cache.embedding_dim != validation_cache.embedding_dim:
        raise ValueError("probe embedding dimension mismatch")
    mx.random.seed(config.seed)
    model: nn.Module
    if config.kind == "linear":
        model = LinearEmbeddingProbe(train_cache.embedding_dim)
    else:
        model = NonlinearEmbeddingProbe(
            train_cache.embedding_dim,
            config.hidden_dim,
        )
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
        for embeddings, metadata in train_cache.iter_batches(
            shuffle=True,
            seed=config.seed + epoch,
        ):
            target = mx.array(metadata["target"])
            source_flags = mx.array(metadata["source_flags"])
            eligible = (
                source_flags.astype(mx.int32)
                & GRADED_SOURCE_CHAMPION_FRONTIER
            ) == 0
            offsets = tuple(
                int(value) for value in metadata["group_offsets"]
            )
            loss, gradients = loss_and_grad(
                model,
                mx.array(np.asarray(embeddings)),
                target,
                eligible,
                offsets,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            batches += 1
        train_metrics = evaluate_embedding_probe(model, train_cache)
        validation_metrics = evaluate_embedding_probe(model, validation_cache)
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
    train_metrics = evaluate_embedding_probe(model, train_cache)
    validation_metrics = evaluate_embedding_probe(model, validation_cache)
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
        "train_cache_blake3": checksum(train_cache.root / "cache.json"),
        "validation_cache_blake3": checksum(
            validation_cache.root / "cache.json"
        ),
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


def evaluate_embedding_probe(
    model: nn.Module,
    cache: FrozenEmbeddingCache,
) -> dict[str, Any]:
    """Measure exact target separation under the anchored width contract."""
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
    for embeddings, metadata in cache.iter_batches():
        target = metadata["target"].astype(np.bool_, copy=False)
        source_flags = metadata["source_flags"].astype(np.int32, copy=False)
        eligible = (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        offsets = tuple(int(value) for value in metadata["group_offsets"])
        embedding_array = mx.array(np.asarray(embeddings))
        logits = model(embedding_array)
        loss = balanced_group_binary_loss(
            model,
            embedding_array,
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
        for group_index, (start, end) in enumerate(
            pairwise(offsets)
        ):
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


def probe_classification(
    linear: dict[str, Any],
    nonlinear: dict[str, Any],
) -> dict[str, Any]:
    """Select the next mechanism from preregistered train-fit gates."""
    linear_train = linear["train"]
    nonlinear_train = nonlinear["train"]
    linear_fit = (
        float(linear_train["target_positive_recall"]) >= 0.60
        and float(linear_train["target_set_exact_fraction"]) >= 0.05
    )
    nonlinear_fit = (
        float(nonlinear_train["target_positive_recall"]) >= 0.80
        and float(nonlinear_train["target_set_exact_fraction"]) >= 0.25
    )
    nonlinear_generalizes = (
        float(nonlinear["validation"]["target_positive_recall"]) >= 0.50
        and float(nonlinear["validation"]["target_set_exact_fraction"]) >= 0.01
    )
    if linear_fit:
        classification = "linear_head_or_optimizer_scope_sufficient"
    elif nonlinear_fit and nonlinear_generalizes:
        classification = "nonlinear_head_capacity_sufficient"
    elif nonlinear_fit:
        classification = "frozen_representation_train_separable_not_generalized"
    else:
        classification = "frozen_representation_insufficient"
    return {
        "linear_train_fit_gate": linear_fit,
        "nonlinear_train_fit_gate": nonlinear_fit,
        "nonlinear_validation_transfer_gate": nonlinear_generalizes,
        "classification": classification,
    }


def load_embedding_probe(
    *,
    kind: str,
    embedding_dim: int,
    weights: Path,
) -> nn.Module:
    """Load one portable probe using the frozen architecture contract."""
    model: nn.Module
    if kind == "linear":
        model = LinearEmbeddingProbe(embedding_dim)
    elif kind == "nonlinear":
        model = NonlinearEmbeddingProbe(
            embedding_dim,
            NONLINEAR_HIDDEN_DIM,
        )
    else:
        raise ValueError("probe kind must be linear or nonlinear")
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def evaluate_saved_probe(
    *,
    kind: str,
    weights: Path,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Cross-host replay one frozen probe on both open caches."""
    train_cache = FrozenEmbeddingCache(train_cache_root)
    validation_cache = FrozenEmbeddingCache(validation_cache_root)
    if train_cache.embedding_dim != validation_cache.embedding_dim:
        raise ValueError("probe evaluation embedding dimension mismatch")
    model = load_embedding_probe(
        kind=kind,
        embedding_dim=train_cache.embedding_dim,
        weights=weights,
    )
    scientific = {
        "kind": kind,
        "weights_blake3": checksum(weights),
        "train_cache_blake3": checksum(train_cache.root / "cache.json"),
        "validation_cache_blake3": checksum(
            validation_cache.root / "cache.json"
        ),
        "train": evaluate_embedding_probe(model, train_cache),
        "validation": evaluate_embedding_probe(model, validation_cache),
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
    return export_embedding_cache(
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint,
        output_root=args.output,
    )


def _probe_main(args: argparse.Namespace) -> dict[str, Any]:
    kind = args.kind
    return train_embedding_probe(
        train_cache_root=args.train_cache,
        validation_cache_root=args.validation_cache,
        output_root=args.output,
        config=EmbeddingProbeConfig(
            kind=kind,
            seed=LINEAR_PROBE_SEED if kind == "linear" else NONLINEAR_PROBE_SEED,
            epochs=PROBE_EPOCHS,
            learning_rate=(
                LINEAR_PROBE_LEARNING_RATE
                if kind == "linear"
                else NONLINEAR_PROBE_LEARNING_RATE
            ),
        ),
    )


def _evaluate_main(args: argparse.Namespace) -> dict[str, Any]:
    report = evaluate_saved_probe(
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
    probe.add_argument("--kind", choices=["linear", "nonlinear"], required=True)
    probe.add_argument("--train-cache", type=Path, required=True)
    probe.add_argument("--validation-cache", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--kind",
        choices=["linear", "nonlinear"],
        required=True,
    )
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
