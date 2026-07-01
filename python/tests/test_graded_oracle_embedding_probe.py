from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.graded_oracle_embedding_probe import (
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    EmbeddingProbeConfig,
    FrozenEmbeddingCache,
    LinearEmbeddingProbe,
    balanced_group_binary_loss,
    evaluate_embedding_probe,
    probe_classification,
    train_embedding_probe,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum


def test_balanced_probe_loss_can_separate_two_groups() -> None:
    model = LinearEmbeddingProbe(2)
    model.head.weight = mx.array([[1.0, 0.0]])
    model.head.bias = mx.zeros_like(model.head.bias)
    embeddings = mx.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [3.0, 0.0],
            [-3.0, 0.0],
        ]
    )
    target = mx.array([True, False, True, False])
    eligible = mx.ones((4,), dtype=mx.bool_)
    separated = balanced_group_binary_loss(
        model,
        embeddings,
        target,
        eligible,
        (0, 2, 4),
    )
    model.head.weight = mx.array([[-1.0, 0.0]])
    reversed_loss = balanced_group_binary_loss(
        model,
        embeddings,
        target,
        eligible,
        (0, 2, 4),
    )
    mx.eval(separated, reversed_loss)
    assert float(separated.item()) < float(reversed_loss.item())


def test_probe_configuration_is_frozen() -> None:
    linear = EmbeddingProbeConfig(
        kind="linear",
        seed=2026061608,
        epochs=20,
        learning_rate=1e-3,
    )
    nonlinear = EmbeddingProbeConfig(
        kind="nonlinear",
        seed=2026061609,
        epochs=20,
        learning_rate=3e-4,
    )
    linear.validate()
    nonlinear.validate()
    with np.testing.assert_raises_regex(ValueError, "configuration drifted"):
        EmbeddingProbeConfig(
            kind="linear",
            seed=1,
            epochs=20,
            learning_rate=1e-3,
        ).validate()


def test_probe_classification_prioritizes_linear_then_nonlinear_fit() -> None:
    base = {
        "train": {
            "target_positive_recall": 0.2,
            "target_set_exact_fraction": 0.0,
        },
        "validation": {
            "target_positive_recall": 0.2,
            "target_set_exact_fraction": 0.0,
        },
    }
    linear = {
        **base,
        "train": {
            "target_positive_recall": 0.7,
            "target_set_exact_fraction": 0.1,
        },
    }
    assert (
        probe_classification(linear, base)["classification"]
        == "linear_head_or_optimizer_scope_sufficient"
    )
    nonlinear = {
        "train": {
            "target_positive_recall": 0.9,
            "target_set_exact_fraction": 0.3,
        },
        "validation": {
            "target_positive_recall": 0.6,
            "target_set_exact_fraction": 0.02,
        },
    }
    assert (
        probe_classification(base, nonlinear)["classification"]
        == "nonlinear_head_capacity_sufficient"
    )
    assert (
        probe_classification(base, base)["classification"]
        == "frozen_representation_insufficient"
    )


def _write_cache(root: Path, split: str) -> None:
    (root / "batches").mkdir(parents=True)
    embeddings_path = root / "batches/batch-000000-embeddings.npy"
    metadata_path = root / "batches/batch-000000-metadata.npz"
    with embeddings_path.open("wb") as handle:
        np.save(
            handle,
            np.array(
                [
                    [2.0, 0.0],
                    [-2.0, 0.0],
                    [3.0, 0.0],
                    [-3.0, 0.0],
                ],
                dtype=np.float32,
            ),
            allow_pickle=False,
        )
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.arange(4, dtype=np.uint8)
    with metadata_path.open("wb") as handle:
        np.savez(
            handle,
            group_offsets=np.array([0, 2, 4], dtype=np.int64),
            target=np.array([True, False, True, False]),
            source_flags=np.zeros(4, dtype=np.int32),
            action_hash=hashes,
            selected_index=np.array([0, 0], dtype=np.int32),
            r4800_mean=np.array([2.0, 1.0, 3.0, 1.0], dtype=np.float32),
            r4800_mask=np.ones(4, dtype=np.bool_),
        )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "split": split,
        "embedding_dim": 2,
        "groups": 2,
        "candidates": 4,
        "batches": [
            {
                "embeddings_file": "batches/batch-000000-embeddings.npy",
                "metadata_file": "batches/batch-000000-metadata.npz",
                "groups": 2,
                "candidates": 4,
                "embeddings_blake3": checksum(embeddings_path),
                "metadata_blake3": checksum(metadata_path),
            }
        ],
    }
    (root / "cache.json").write_text(json.dumps(manifest))


def test_linear_probe_trains_and_reloads_on_cache(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    _write_cache(train_root, "train")
    _write_cache(validation_root, "validation")
    output = tmp_path / "probe"
    report = train_embedding_probe(
        train_cache_root=train_root,
        validation_cache_root=validation_root,
        output_root=output,
        config=EmbeddingProbeConfig(
            kind="linear",
            seed=2026061608,
            epochs=20,
            learning_rate=1e-3,
        ),
    )
    assert report["train"]["target_positive_recall"] == 1.0
    assert report["validation"]["target_set_exact_fraction"] == 1.0
    assert (output / "best.safetensors").is_file()
    cache = FrozenEmbeddingCache(validation_root)
    model = LinearEmbeddingProbe(2)
    model.load_weights(str(output / "best.safetensors"))
    metrics = evaluate_embedding_probe(model, cache)
    assert metrics["all_candidates_scored_once"]
