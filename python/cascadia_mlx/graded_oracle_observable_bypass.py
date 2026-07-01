"""Observable-feature sidecars and frozen-trunk bypass probes."""

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
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GRADED_ORACLE_PRIOR_DIM,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_embedding_probe import (
    FrozenEmbeddingCache,
    balanced_group_binary_loss,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

EXPERIMENT_ID = "complete-action-frontier-observable-bypass-v1"
SIDECAR_SCHEMA_VERSION = 1
SIDECAR_SCHEMA = "graded-oracle-observable-sidecar-v1"
RAW_FEATURE_DIM = GRADED_ORACLE_ACTION_DIM + GRADED_ORACLE_PRIOR_DIM
RAW_LINEAR_SEED = 2026061610
RAW_NONLINEAR_SEED = 2026061611
COMBINED_NONLINEAR_SEED = 2026061612
PROBE_EPOCHS = 20
LINEAR_LEARNING_RATE = 1e-3
NONLINEAR_LEARNING_RATE = 3e-4
PROBE_WEIGHT_DECAY = 1e-4
NONLINEAR_HIDDEN_DIM = 256


@dataclass(frozen=True)
class ObservableSidecarBatch:
    features_path: Path
    groups: int
    candidates: int


class FrozenObservableSidecar:
    """Manifest-backed observable action/prior features."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "sidecar.json").read_text())
        if (
            self.manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION
            or self.manifest.get("sidecar_schema") != SIDECAR_SCHEMA
        ):
            raise ValueError("unsupported observable sidecar")
        self.split = str(self.manifest["split"])
        self.raw_feature_dim = int(self.manifest["raw_feature_dim"])
        self.group_count = int(self.manifest["groups"])
        self.candidate_count = int(self.manifest["candidates"])
        self.payload_blake3 = str(self.manifest["payload_blake3"])
        self.embedding_cache_manifest_blake3 = str(
            self.manifest["embedding_cache_manifest_blake3"]
        )
        self.batches = tuple(
            ObservableSidecarBatch(
                features_path=self.root / entry["features_file"],
                groups=int(entry["groups"]),
                candidates=int(entry["candidates"]),
            )
            for entry in self.manifest["batches"]
        )
        if self.raw_feature_dim != RAW_FEATURE_DIM:
            raise ValueError("observable sidecar feature width drifted")
        if sum(batch.groups for batch in self.batches) != self.group_count:
            raise ValueError("observable sidecar group total drifted")
        if sum(batch.candidates for batch in self.batches) != self.candidate_count:
            raise ValueError("observable sidecar candidate total drifted")
        if verify_checksums:
            for batch, entry in zip(
                self.batches,
                self.manifest["batches"],
                strict=True,
            ):
                if checksum(batch.features_path) != entry["features_blake3"]:
                    raise ValueError("observable sidecar checksum mismatch")
            if sidecar_payload_blake3(self.manifest) != self.payload_blake3:
                raise ValueError("observable sidecar payload identity drifted")


class ObservableBypassCache:
    """Pair one ADR 0094 embedding cache with an aligned raw sidecar."""

    def __init__(
        self,
        embedding_root: str | Path,
        sidecar_root: str | Path,
        *,
        verify_checksums: bool = True,
    ):
        self.embedding = FrozenEmbeddingCache(
            embedding_root,
            verify_checksums=verify_checksums,
        )
        self.sidecar = FrozenObservableSidecar(
            sidecar_root,
            verify_checksums=verify_checksums,
        )
        if (
            self.embedding.split != self.sidecar.split
            or self.embedding.group_count != self.sidecar.group_count
            or self.embedding.candidate_count != self.sidecar.candidate_count
            or len(self.embedding.batches) != len(self.sidecar.batches)
        ):
            raise ValueError("observable bypass cache totals drifted")
        if self.sidecar.embedding_cache_manifest_blake3 != checksum(
            self.embedding.root / "cache.json"
        ):
            raise ValueError("observable sidecar embedding identity drifted")
        for embedding_batch, sidecar_batch in zip(
            self.embedding.batches,
            self.sidecar.batches,
            strict=True,
        ):
            if (
                embedding_batch.groups != sidecar_batch.groups
                or embedding_batch.candidates != sidecar_batch.candidates
            ):
                raise ValueError("observable bypass batch geometry drifted")
        self.split = self.embedding.split
        self.group_count = self.embedding.group_count
        self.candidate_count = self.embedding.candidate_count
        self.embedding_dim = self.embedding.embedding_dim

    def input_dim(self, kind: str) -> int:
        if kind in {"raw-linear", "raw-nonlinear"}:
            return RAW_FEATURE_DIM
        if kind == "combined-nonlinear":
            return self.embedding_dim + RAW_FEATURE_DIM
        raise ValueError("unsupported observable bypass probe kind")

    def iter_batches(
        self,
        kind: str,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        order = np.arange(len(self.embedding.batches))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for index_value in order:
            index = int(index_value)
            embedding_batch = self.embedding.batches[index]
            sidecar_batch = self.sidecar.batches[index]
            raw = np.load(sidecar_batch.features_path, mmap_mode="r")
            with np.load(embedding_batch.metadata_path) as loaded:
                metadata = {name: loaded[name] for name in loaded.files}
            if kind == "combined-nonlinear":
                embeddings = np.load(
                    embedding_batch.embeddings_path,
                    mmap_mode="r",
                )
                features = np.concatenate(
                    [np.asarray(embeddings), np.asarray(raw)],
                    axis=1,
                )
            elif kind in {"raw-linear", "raw-nonlinear"}:
                features = raw
            else:
                raise ValueError("unsupported observable bypass probe kind")
            yield features, metadata


def sidecar_payload_blake3(manifest: dict[str, Any]) -> str:
    """Hash only portable scientific payload identity."""
    payload = {
        "sidecar_schema": manifest["sidecar_schema"],
        "split": manifest["split"],
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_blake3": manifest["dataset_manifest_blake3"],
        "embedding_cache_manifest_blake3": manifest[
            "embedding_cache_manifest_blake3"
        ],
        "raw_feature_dim": manifest["raw_feature_dim"],
        "groups": manifest["groups"],
        "candidates": manifest["candidates"],
        "batches": [
            {
                "index": entry["index"],
                "features_file": entry["features_file"],
                "features_blake3": entry["features_blake3"],
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


def export_observable_sidecar(
    *,
    dataset_root: Path,
    embedding_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Materialize raw observable rows aligned to every cached embedding."""
    if output_root.exists():
        raise ValueError("observable sidecar output already exists")
    dataset = GradedOracleDataset(dataset_root, verify_checksums=True)
    embeddings = FrozenEmbeddingCache(embedding_cache_root)
    if dataset.split not in {"train", "validation"}:
        raise ValueError("observable sidecar accepts only open splits")
    if dataset.split != embeddings.split:
        raise ValueError("observable sidecar split mismatch")
    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "batches").mkdir(parents=True)

    dataset_batches = dataset.batches(
        64,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    )
    entries: list[dict[str, Any]] = []
    groups = 0
    candidates = 0
    started = time.perf_counter()
    for index, (batch, embedding_batch) in enumerate(
        zip(dataset_batches, embeddings.batches, strict=True)
    ):
        candidate_mask = np.asarray(batch.candidate_mask)
        counts = np.sum(candidate_mask, axis=1, dtype=np.int64)
        offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts)]
        )
        with np.load(embedding_batch.metadata_path) as loaded:
            cached_offsets = loaded["group_offsets"]
            cached_hashes = loaded["action_hash"]
        action_hashes = _flatten_valid(batch.action_hash, counts)
        if (
            not np.array_equal(offsets, cached_offsets)
            or not np.array_equal(action_hashes, cached_hashes)
        ):
            raise ValueError("observable sidecar action alignment drifted")
        raw = np.concatenate(
            [
                np.asarray(batch.action_features, dtype=np.float32),
                np.asarray(batch.prior_features, dtype=np.float32),
            ],
            axis=-1,
        )
        flattened = _flatten_valid(raw, counts)
        relative = Path("batches") / f"batch-{index:06d}-features.npy"
        path = temporary / relative
        with path.open("wb") as handle:
            np.save(handle, flattened, allow_pickle=False)
        entries.append(
            {
                "index": index,
                "features_file": relative.as_posix(),
                "features_blake3": checksum(path),
                "features_bytes": path.stat().st_size,
                "groups": len(counts),
                "candidates": int(offsets[-1]),
            }
        )
        groups += len(counts)
        candidates += int(offsets[-1])
    if (
        groups != embeddings.group_count
        or candidates != embeddings.candidate_count
        or len(entries) != len(embeddings.batches)
    ):
        raise ValueError("observable sidecar final totals drifted")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    manifest: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sidecar_schema": SIDECAR_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname().split(".")[0],
        "split": dataset.split,
        "dataset_root": str(dataset.root.resolve()),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "embedding_cache_manifest_blake3": checksum(
            embeddings.root / "cache.json"
        ),
        "raw_feature_dim": RAW_FEATURE_DIM,
        "raw_feature_schema": {
            "action_features": GRADED_ORACLE_ACTION_DIM,
            "observable_prior_features": GRADED_ORACLE_PRIOR_DIM,
        },
        "groups": groups,
        "candidates": candidates,
        "batches": entries,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        },
        "target_or_teacher_values_included": False,
        "test_split_opened": False,
    }
    manifest["payload_blake3"] = sidecar_payload_blake3(manifest)
    _write_json_atomic(temporary / "sidecar.json", manifest)
    os.replace(temporary, output_root)
    return manifest


def _flatten_valid(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values[index, :count] for index, count in enumerate(counts)],
        axis=0,
    )


@dataclass(frozen=True)
class ObservableBypassProbeConfig:
    kind: str
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float = PROBE_WEIGHT_DECAY
    hidden_dim: int = NONLINEAR_HIDDEN_DIM

    def validate(self) -> None:
        expected = {
            "raw-linear": (RAW_LINEAR_SEED, LINEAR_LEARNING_RATE),
            "raw-nonlinear": (
                RAW_NONLINEAR_SEED,
                NONLINEAR_LEARNING_RATE,
            ),
            "combined-nonlinear": (
                COMBINED_NONLINEAR_SEED,
                NONLINEAR_LEARNING_RATE,
            ),
        }
        if self.kind not in expected:
            raise ValueError("unsupported observable bypass probe kind")
        seed, learning_rate = expected[self.kind]
        if (
            self.seed != seed
            or self.epochs != PROBE_EPOCHS
            or self.learning_rate != learning_rate
            or self.weight_decay != PROBE_WEIGHT_DECAY
            or self.hidden_dim != NONLINEAR_HIDDEN_DIM
        ):
            raise ValueError("observable bypass probe configuration drifted")


class LinearObservableProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.head(features).reshape(-1)


class NonlinearObservableProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, features: mx.array) -> mx.array:
        return self.network(features).reshape(-1)


def create_probe(kind: str, input_dim: int) -> nn.Module:
    if kind == "raw-linear":
        return LinearObservableProbe(input_dim)
    if kind in {"raw-nonlinear", "combined-nonlinear"}:
        return NonlinearObservableProbe(input_dim, NONLINEAR_HIDDEN_DIM)
    raise ValueError("unsupported observable bypass probe kind")


def train_observable_bypass_probe(
    *,
    train_embedding_root: Path,
    train_sidecar_root: Path,
    validation_embedding_root: Path,
    validation_sidecar_root: Path,
    output_root: Path,
    config: ObservableBypassProbeConfig,
) -> dict[str, Any]:
    """Fit one frozen observable bypass probe."""
    config.validate()
    if output_root.exists():
        raise ValueError("observable bypass output already exists")
    train_cache = ObservableBypassCache(
        train_embedding_root,
        train_sidecar_root,
    )
    validation_cache = ObservableBypassCache(
        validation_embedding_root,
        validation_sidecar_root,
    )
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("observable bypass cache split mismatch")
    input_dim = train_cache.input_dim(config.kind)
    if input_dim != validation_cache.input_dim(config.kind):
        raise ValueError("observable bypass input width mismatch")

    mx.random.seed(config.seed)
    model = create_probe(config.kind, input_dim)
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
        for features, metadata in train_cache.iter_batches(
            config.kind,
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
                mx.array(np.asarray(features)),
                target,
                eligible,
                offsets,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            batches += 1
        train_metrics = evaluate_observable_bypass(
            model,
            train_cache,
            config.kind,
        )
        validation_metrics = evaluate_observable_bypass(
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
    train_metrics = evaluate_observable_bypass(
        model,
        train_cache,
        config.kind,
    )
    validation_metrics = evaluate_observable_bypass(
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
        "train_embedding_cache_blake3": checksum(
            train_cache.embedding.root / "cache.json"
        ),
        "validation_embedding_cache_blake3": checksum(
            validation_cache.embedding.root / "cache.json"
        ),
        "train_sidecar_payload_blake3": train_cache.sidecar.payload_blake3,
        "validation_sidecar_payload_blake3": (
            validation_cache.sidecar.payload_blake3
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


def evaluate_observable_bypass(
    model: nn.Module,
    cache: ObservableBypassCache,
    kind: str,
) -> dict[str, Any]:
    """Measure exact target separation under the deployed selector."""
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
    for features, metadata in cache.iter_batches(kind):
        target = metadata["target"].astype(np.bool_, copy=False)
        source_flags = metadata["source_flags"].astype(
            np.int32,
            copy=False,
        )
        eligible = (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        offsets = tuple(int(value) for value in metadata["group_offsets"])
        feature_array = mx.array(np.asarray(features))
        logits = model(feature_array)
        loss = balanced_group_binary_loss(
            model,
            feature_array,
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
                    - float(
                        np.max(
                            r4800_mean[start:end][retained_labeled]
                        )
                    ),
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


def observable_bypass_classification(
    raw_linear: dict[str, Any],
    raw_nonlinear: dict[str, Any],
    combined: dict[str, Any],
) -> dict[str, Any]:
    linear_fit = _train_gate(raw_linear, recall=0.60, exact=0.05)
    raw_nonlinear_fit = _train_gate(
        raw_nonlinear,
        recall=0.80,
        exact=0.25,
    )
    raw_nonlinear_transfer = _validation_gate(raw_nonlinear)
    combined_fit = _train_gate(combined, recall=0.80, exact=0.25)
    combined_transfer = _validation_gate(combined)
    if linear_fit:
        classification = "raw_linear_bypass_sufficient"
    elif raw_nonlinear_fit and raw_nonlinear_transfer:
        classification = "raw_nonlinear_bypass_sufficient"
    elif combined_fit and combined_transfer:
        classification = "combined_bypass_sufficient"
    elif combined_fit:
        classification = "combined_bypass_train_separable_not_generalized"
    else:
        classification = "observable_bypass_insufficient"
    return {
        "raw_linear_train_gate": linear_fit,
        "raw_nonlinear_train_gate": raw_nonlinear_fit,
        "raw_nonlinear_validation_gate": raw_nonlinear_transfer,
        "combined_train_gate": combined_fit,
        "combined_validation_gate": combined_transfer,
        "classification": classification,
    }


def _train_gate(
    report: dict[str, Any],
    *,
    recall: float,
    exact: float,
) -> bool:
    return (
        float(report["train"]["target_positive_recall"]) >= recall
        and float(report["train"]["target_set_exact_fraction"]) >= exact
    )


def _validation_gate(report: dict[str, Any]) -> bool:
    return (
        float(report["validation"]["target_positive_recall"]) >= 0.50
        and float(report["validation"]["target_set_exact_fraction"]) >= 0.01
    )


def load_observable_probe(
    *,
    kind: str,
    input_dim: int,
    weights: Path,
) -> nn.Module:
    model = create_probe(kind, input_dim)
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model


def evaluate_saved_observable_probe(
    *,
    kind: str,
    weights: Path,
    train_embedding_root: Path,
    train_sidecar_root: Path,
    validation_embedding_root: Path,
    validation_sidecar_root: Path,
) -> dict[str, Any]:
    train_cache = ObservableBypassCache(
        train_embedding_root,
        train_sidecar_root,
    )
    validation_cache = ObservableBypassCache(
        validation_embedding_root,
        validation_sidecar_root,
    )
    input_dim = train_cache.input_dim(kind)
    if input_dim != validation_cache.input_dim(kind):
        raise ValueError("observable bypass evaluation width mismatch")
    model = load_observable_probe(
        kind=kind,
        input_dim=input_dim,
        weights=weights,
    )
    scientific = {
        "kind": kind,
        "input_dim": input_dim,
        "weights_blake3": checksum(weights),
        "train_embedding_cache_blake3": checksum(
            train_cache.embedding.root / "cache.json"
        ),
        "validation_embedding_cache_blake3": checksum(
            validation_cache.embedding.root / "cache.json"
        ),
        "train_sidecar_payload_blake3": (
            train_cache.sidecar.payload_blake3
        ),
        "validation_sidecar_payload_blake3": (
            validation_cache.sidecar.payload_blake3
        ),
        "train": evaluate_observable_bypass(model, train_cache, kind),
        "validation": evaluate_observable_bypass(
            model,
            validation_cache,
            kind,
        ),
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


def _config(kind: str) -> ObservableBypassProbeConfig:
    seeds = {
        "raw-linear": RAW_LINEAR_SEED,
        "raw-nonlinear": RAW_NONLINEAR_SEED,
        "combined-nonlinear": COMBINED_NONLINEAR_SEED,
    }
    return ObservableBypassProbeConfig(
        kind=kind,
        seed=seeds[kind],
        epochs=PROBE_EPOCHS,
        learning_rate=(
            LINEAR_LEARNING_RATE
            if kind == "raw-linear"
            else NONLINEAR_LEARNING_RATE
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sidecar = subparsers.add_parser("sidecar")
    sidecar.add_argument("--dataset", type=Path, required=True)
    sidecar.add_argument("--embedding-cache", type=Path, required=True)
    sidecar.add_argument("--output", type=Path, required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument(
        "--kind",
        choices=[
            "raw-linear",
            "raw-nonlinear",
            "combined-nonlinear",
        ],
        required=True,
    )
    probe.add_argument("--train-embedding", type=Path, required=True)
    probe.add_argument("--train-sidecar", type=Path, required=True)
    probe.add_argument("--validation-embedding", type=Path, required=True)
    probe.add_argument("--validation-sidecar", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--kind",
        choices=[
            "raw-linear",
            "raw-nonlinear",
            "combined-nonlinear",
        ],
        required=True,
    )
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--train-embedding", type=Path, required=True)
    evaluate.add_argument("--train-sidecar", type=Path, required=True)
    evaluate.add_argument("--validation-embedding", type=Path, required=True)
    evaluate.add_argument("--validation-sidecar", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sidecar":
        report = export_observable_sidecar(
            dataset_root=args.dataset,
            embedding_cache_root=args.embedding_cache,
            output_root=args.output,
        )
    elif args.command == "probe":
        report = train_observable_bypass_probe(
            train_embedding_root=args.train_embedding,
            train_sidecar_root=args.train_sidecar,
            validation_embedding_root=args.validation_embedding,
            validation_sidecar_root=args.validation_sidecar,
            output_root=args.output,
            config=_config(args.kind),
        )
    else:
        report = evaluate_saved_observable_probe(
            kind=args.kind,
            weights=args.weights,
            train_embedding_root=args.train_embedding,
            train_sidecar_root=args.train_sidecar,
            validation_embedding_root=args.validation_embedding,
            validation_sidecar_root=args.validation_sidecar,
        )
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
