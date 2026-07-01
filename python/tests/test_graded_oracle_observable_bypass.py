from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.graded_oracle_embedding_probe import (
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_observable_bypass import (
    RAW_FEATURE_DIM,
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_VERSION,
    FrozenObservableSidecar,
    ObservableBypassCache,
    ObservableBypassProbeConfig,
    create_probe,
    evaluate_observable_bypass,
    observable_bypass_classification,
    sidecar_payload_blake3,
    train_observable_bypass_probe,
)


def _write_fixture(root: Path, split: str) -> tuple[Path, Path]:
    embedding = root / "embedding"
    sidecar = root / "sidecar"
    (embedding / "batches").mkdir(parents=True)
    (sidecar / "batches").mkdir(parents=True)
    embeddings_path = embedding / "batches/batch-000000-embeddings.npy"
    metadata_path = embedding / "batches/batch-000000-metadata.npz"
    raw_path = sidecar / "batches/batch-000000-features.npy"
    np.save(
        embeddings_path,
        np.ones((4, 192), dtype=np.float32),
        allow_pickle=False,
    )
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.arange(4, dtype=np.uint8)
    np.savez(
        metadata_path,
        group_offsets=np.array([0, 2, 4], dtype=np.int64),
        target=np.array([True, False, True, False]),
        source_flags=np.zeros(4, dtype=np.int32),
        action_hash=hashes,
        selected_index=np.array([0, 0], dtype=np.int32),
        r4800_mean=np.array([2.0, 1.0, 3.0, 1.0], dtype=np.float32),
        r4800_mask=np.ones(4, dtype=np.bool_),
    )
    raw = np.zeros((4, RAW_FEATURE_DIM), dtype=np.float32)
    raw[:, 0] = np.array([2.0, -2.0, 3.0, -3.0])
    np.save(raw_path, raw, allow_pickle=False)
    embedding_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "split": split,
        "embedding_dim": 192,
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
    (embedding / "cache.json").write_text(json.dumps(embedding_manifest))
    sidecar_manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sidecar_schema": SIDECAR_SCHEMA,
        "split": split,
        "dataset_id": f"fixture-{split}",
        "dataset_manifest_blake3": "dataset",
        "embedding_cache_manifest_blake3": checksum(
            embedding / "cache.json"
        ),
        "raw_feature_dim": RAW_FEATURE_DIM,
        "groups": 2,
        "candidates": 4,
        "batches": [
            {
                "index": 0,
                "features_file": "batches/batch-000000-features.npy",
                "features_blake3": checksum(raw_path),
                "groups": 2,
                "candidates": 4,
            }
        ],
    }
    sidecar_manifest["payload_blake3"] = sidecar_payload_blake3(
        sidecar_manifest
    )
    (sidecar / "sidecar.json").write_text(json.dumps(sidecar_manifest))
    return embedding, sidecar


def test_observable_sidecar_and_combined_cache(tmp_path: Path) -> None:
    embedding, sidecar = _write_fixture(tmp_path, "train")
    frozen = FrozenObservableSidecar(sidecar)
    assert frozen.group_count == 2
    paired = ObservableBypassCache(embedding, sidecar)
    raw, metadata = next(paired.iter_batches("raw-linear"))
    combined, _ = next(paired.iter_batches("combined-nonlinear"))
    assert raw.shape == (4, RAW_FEATURE_DIM)
    assert combined.shape == (4, RAW_FEATURE_DIM + 192)
    assert metadata["target"].tolist() == [True, False, True, False]


def test_observable_probe_configuration_and_shapes() -> None:
    config = ObservableBypassProbeConfig(
        kind="raw-linear",
        seed=2026061610,
        epochs=20,
        learning_rate=1e-3,
    )
    config.validate()
    linear = create_probe("raw-linear", RAW_FEATURE_DIM)
    nonlinear = create_probe(
        "combined-nonlinear",
        RAW_FEATURE_DIM + 192,
    )
    linear_value = linear(mx.zeros((3, RAW_FEATURE_DIM)))
    nonlinear_value = nonlinear(
        mx.zeros((3, RAW_FEATURE_DIM + 192))
    )
    mx.eval(linear_value, nonlinear_value)
    assert linear_value.shape == (3,)
    assert nonlinear_value.shape == (3,)


def test_observable_bypass_classification_order() -> None:
    failed = {
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
        **failed,
        "train": {
            "target_positive_recall": 0.7,
            "target_set_exact_fraction": 0.1,
        },
    }
    assert (
        observable_bypass_classification(linear, failed, failed)[
            "classification"
        ]
        == "raw_linear_bypass_sufficient"
    )
    combined = {
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
        observable_bypass_classification(failed, failed, combined)[
            "classification"
        ]
        == "combined_bypass_sufficient"
    )
    assert (
        observable_bypass_classification(failed, failed, failed)[
            "classification"
        ]
        == "observable_bypass_insufficient"
    )


def test_raw_linear_probe_trains_and_reloads(tmp_path: Path) -> None:
    train_embedding, train_sidecar = _write_fixture(
        tmp_path / "train",
        "train",
    )
    validation_embedding, validation_sidecar = _write_fixture(
        tmp_path / "validation",
        "validation",
    )
    output = tmp_path / "probe"
    report = train_observable_bypass_probe(
        train_embedding_root=train_embedding,
        train_sidecar_root=train_sidecar,
        validation_embedding_root=validation_embedding,
        validation_sidecar_root=validation_sidecar,
        output_root=output,
        config=ObservableBypassProbeConfig(
            kind="raw-linear",
            seed=2026061610,
            epochs=20,
            learning_rate=1e-3,
        ),
    )
    assert report["train"]["target_positive_recall"] == 1.0
    assert report["validation"]["target_set_exact_fraction"] == 1.0
    cache = ObservableBypassCache(
        validation_embedding,
        validation_sidecar,
    )
    model = create_probe("raw-linear", RAW_FEATURE_DIM)
    model.load_weights(str(output / "best.safetensors"))
    metrics = evaluate_observable_bypass(model, cache, "raw-linear")
    assert metrics["all_candidates_scored_once"]
