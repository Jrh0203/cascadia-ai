from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_prepool_context import (
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    CANDIDATE_ONLY,
    LEGACY_CONTEXT,
    MOMENT_CONTEXT,
    PROBE_KINDS,
    PROBE_SEEDS,
    SCREEN_TOP64_CONTEXT,
    FrozenPrepoolCache,
    PrepoolProbeConfig,
    build_context_features,
    context_input_dim,
    evaluate_prepool_probe,
    prepool_context_classification,
    prepool_payload_blake3,
    stable_screen_topk_indices,
    train_prepool_probe,
)


def _write_fixture(root: Path, split: str) -> Path:
    (root / "batches").mkdir(parents=True)
    candidates_path = root / "batches/batch-000000-candidates.npy"
    metadata_path = root / "batches/batch-000000-metadata.npz"
    candidates = np.array(
        [
            [3.0, 0.0],
            [-3.0, 0.0],
            [4.0, 1.0],
            [-4.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(candidates_path, candidates, allow_pickle=False)
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.array([4, 3, 2, 1], dtype=np.uint8)
    np.savez(
        metadata_path,
        group_offsets=np.array([0, 2, 4], dtype=np.int64),
        target=np.array([True, False, True, False]),
        source_flags=np.zeros(4, dtype=np.int32),
        screen_rank=np.array([1, 2, 1, 2], dtype=np.int32),
        action_hash=hashes,
        selected_index=np.array([0, 0], dtype=np.int32),
        r4800_mean=np.array([2.0, 1.0, 3.0, 1.0], dtype=np.float32),
        r4800_mask=np.ones(4, dtype=np.bool_),
    )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "split": split,
        "dataset_id": f"fixture-{split}",
        "dataset_manifest_blake3": "dataset",
        "checkpoint": "checkpoint",
        "checkpoint_manifest_blake3": "checkpoint-manifest",
        "model_blake3": "model",
        "candidate_dim": 2,
        "groups": 2,
        "candidates": 4,
        "batches": [
            {
                "index": 0,
                "candidates_file": "batches/batch-000000-candidates.npy",
                "metadata_file": "batches/batch-000000-metadata.npz",
                "candidates_blake3": checksum(candidates_path),
                "metadata_blake3": checksum(metadata_path),
                "groups": 2,
                "candidates": 4,
            }
        ],
    }
    manifest["payload_blake3"] = prepool_payload_blake3(manifest)
    (root / "cache.json").write_text(json.dumps(manifest))
    return root


def test_prepool_cache_and_context_widths(tmp_path: Path) -> None:
    cache = FrozenPrepoolCache(_write_fixture(tmp_path, "train"))
    candidates, metadata = next(cache.iter_batches())
    for kind in PROBE_KINDS:
        features = build_context_features(
            kind,
            mx.array(np.asarray(candidates)),
            metadata,
        )
        mx.eval(features)
        assert features.shape == (4, context_input_dim(kind, 2))


def test_context_builders_are_permutation_equivariant() -> None:
    candidates = np.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0], [2.0, -2.0]],
        dtype=np.float32,
    )
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.arange(4, dtype=np.uint8)
    metadata = {
        "group_offsets": np.array([0, 4], dtype=np.int64),
        "screen_rank": np.array([2, 1, 4, 3], dtype=np.int32),
        "action_hash": hashes,
    }
    permutation = np.array([2, 0, 3, 1])
    permuted_metadata = {
        **metadata,
        "screen_rank": metadata["screen_rank"][permutation],
        "action_hash": metadata["action_hash"][permutation],
    }
    for kind in PROBE_KINDS:
        original = build_context_features(
            kind,
            mx.array(candidates),
            metadata,
        )
        permuted = build_context_features(
            kind,
            mx.array(candidates[permutation]),
            permuted_metadata,
        )
        mx.eval(original, permuted)
        np.testing.assert_allclose(
            np.asarray(permuted),
            np.asarray(original)[permutation],
            atol=0.0,
            rtol=0.0,
        )


def test_screen_topk_uses_rank_then_hash() -> None:
    ranks = np.array([1, 1, 2, 1], dtype=np.int32)
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.array([9, 3, 0, 7], dtype=np.uint8)
    assert stable_screen_topk_indices(ranks, hashes, width=3).tolist() == [
        1,
        3,
        0,
    ]


def test_prepool_classification_follows_smallest_passing_context() -> None:
    failed = {
        "train": {
            "target_positive_recall": 0.3,
            "target_set_exact_fraction": 0.0,
        },
        "validation": {
            "target_positive_recall": 0.3,
            "target_set_exact_fraction": 0.0,
        },
    }
    passed = {
        "train": {
            "target_positive_recall": 0.9,
            "target_set_exact_fraction": 0.3,
        },
        "validation": {
            "target_positive_recall": 0.6,
            "target_set_exact_fraction": 0.02,
        },
    }
    reports = {kind: failed for kind in PROBE_KINDS}
    reports[LEGACY_CONTEXT] = passed
    assert (
        prepool_context_classification(reports)["classification"]
        == "legacy_output_trunk_collapse"
    )
    reports = {kind: failed for kind in PROBE_KINDS}
    assert (
        prepool_context_classification(reports)["classification"]
        == "candidate_projection_insufficient"
    )


def test_candidate_only_probe_trains_and_reloads(tmp_path: Path) -> None:
    train_cache = _write_fixture(tmp_path / "train", "train")
    validation_cache = _write_fixture(
        tmp_path / "validation",
        "validation",
    )
    output = tmp_path / "probe"
    report = train_prepool_probe(
        train_cache_root=train_cache,
        validation_cache_root=validation_cache,
        output_root=output,
        config=PrepoolProbeConfig(
            kind=CANDIDATE_ONLY,
            seed=PROBE_SEEDS[CANDIDATE_ONLY],
        ),
    )
    assert report["train"]["target_positive_recall"] == 1.0
    assert report["validation"]["target_set_exact_fraction"] == 1.0
    cache = FrozenPrepoolCache(validation_cache)
    from cascadia_mlx.graded_oracle_prepool_context import load_prepool_probe

    model = load_prepool_probe(
        kind=CANDIDATE_ONLY,
        candidate_dim=cache.candidate_dim,
        weights=output / "best.safetensors",
    )
    metrics = evaluate_prepool_probe(model, cache, CANDIDATE_ONLY)
    assert metrics["all_candidates_scored_once"]
    assert metrics["target_positive_recall"] == 1.0


def test_probe_configuration_is_frozen() -> None:
    for kind in PROBE_KINDS:
        PrepoolProbeConfig(kind=kind, seed=PROBE_SEEDS[kind]).validate()
    with np.testing.assert_raises_regex(
        ValueError,
        "configuration drifted",
    ):
        PrepoolProbeConfig(
            kind=MOMENT_CONTEXT,
            seed=PROBE_SEEDS[MOMENT_CONTEXT],
            epochs=21,
        ).validate()
    assert context_input_dim(SCREEN_TOP64_CONTEXT, 192) == 1344
