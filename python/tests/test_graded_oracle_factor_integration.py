from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.graded_oracle_factor_integration import (
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    FACTOR_ATTENTION,
    FACTOR_COUNT,
    FACTOR_DIM,
    PAIRWISE_GATED,
    PROBE_KINDS,
    PROBE_SEEDS,
    SCREEN_RELATIVE,
    WIDE_CONCAT,
    FactorProbeConfig,
    FrozenFactorCache,
    balanced_factor_binary_loss,
    build_factor_probe,
    configure_mlx_memory,
    evaluate_factor_probe,
    factor_integration_classification,
    factor_payload_blake3,
    train_factor_probe,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from mlx.utils import tree_flatten


def _write_fixture(root: Path, split: str) -> Path:
    (root / "batches").mkdir(parents=True)
    factors_path = root / "batches/batch-000000-factors.npy"
    metadata_path = root / "batches/batch-000000-metadata.npz"
    factors = np.zeros((4, FACTOR_COUNT, FACTOR_DIM), dtype=np.float32)
    factors[0, 0, 0] = 4.0
    factors[1, 0, 0] = -4.0
    factors[2, 0, 0] = 5.0
    factors[3, 0, 0] = -5.0
    np.save(factors_path, factors, allow_pickle=False)
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
        "factor_names": [
            "action",
            "prior",
            "parent",
            "staged",
            "board_cross",
            "staged_cross",
            "action_parent_product",
        ],
        "factor_count": FACTOR_COUNT,
        "factor_dim": FACTOR_DIM,
        "groups": 2,
        "candidates": 4,
        "batches": [
            {
                "index": 0,
                "factors_file": "batches/batch-000000-factors.npy",
                "metadata_file": "batches/batch-000000-metadata.npz",
                "factors_blake3": checksum(factors_path),
                "metadata_blake3": checksum(metadata_path),
                "groups": 2,
                "candidates": 4,
            }
        ],
    }
    manifest["payload_blake3"] = factor_payload_blake3(manifest)
    (root / "cache.json").write_text(json.dumps(manifest))
    return root


def _metadata() -> dict[str, np.ndarray]:
    hashes = np.zeros((4, 32), dtype=np.uint8)
    hashes[:, 0] = np.array([9, 3, 7, 1], dtype=np.uint8)
    return {
        "group_offsets": np.array([0, 4], dtype=np.int64),
        "screen_rank": np.array([2, 1, 4, 3], dtype=np.int32),
        "action_hash": hashes,
    }


def test_factor_cache_and_all_probe_shapes(tmp_path: Path) -> None:
    cache = FrozenFactorCache(_write_fixture(tmp_path, "train"))
    factors, metadata = next(cache.iter_batches())
    for kind in PROBE_KINDS:
        model = build_factor_probe(kind)
        scores = model(
            mx.array(np.asarray(factors)),
            tuple(int(value) for value in metadata["group_offsets"]),
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        mx.eval(scores)
        assert scores.shape == (4,)
        assert np.all(np.isfinite(np.asarray(scores)))


def test_factor_probes_are_candidate_permutation_equivariant() -> None:
    rng = np.random.default_rng(42)
    factors = rng.normal(size=(4, FACTOR_COUNT, FACTOR_DIM)).astype(
        np.float32
    )
    metadata = _metadata()
    permutation = np.array([2, 0, 3, 1])
    permuted_metadata = {
        "group_offsets": metadata["group_offsets"],
        "screen_rank": metadata["screen_rank"][permutation],
        "action_hash": metadata["action_hash"][permutation],
    }
    for kind in PROBE_KINDS:
        model = build_factor_probe(kind)
        original = model(
            mx.array(factors),
            (0, 4),
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        permuted = model(
            mx.array(factors[permutation]),
            (0, 4),
            permuted_metadata["screen_rank"],
            permuted_metadata["action_hash"],
        )
        mx.eval(original, permuted)
        np.testing.assert_allclose(
            np.asarray(permuted),
            np.asarray(original)[permutation],
            atol=1e-6,
            rtol=1e-6,
        )


def test_factor_classification_prefers_simpler_passing_architecture() -> None:
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
    reports[FACTOR_ATTENTION] = passed
    reports[PAIRWISE_GATED] = passed
    assert (
        factor_integration_classification(reports)["classification"]
        == "pairwise_factor_sufficient"
    )
    reports = {kind: failed for kind in PROBE_KINDS}
    assert (
        factor_integration_classification(reports)["classification"]
        == "candidate_factor_inputs_insufficient"
    )


def test_wide_factor_probe_trains_and_reloads(tmp_path: Path) -> None:
    train_cache = _write_fixture(tmp_path / "train", "train")
    validation_cache = _write_fixture(
        tmp_path / "validation",
        "validation",
    )
    output = tmp_path / "probe"
    report = train_factor_probe(
        train_cache_root=train_cache,
        validation_cache_root=validation_cache,
        output_root=output,
        config=FactorProbeConfig(
            kind=WIDE_CONCAT,
            seed=PROBE_SEEDS[WIDE_CONCAT],
        ),
    )
    assert report["train"]["target_positive_recall"] == 1.0
    assert report["validation"]["target_set_exact_fraction"] == 1.0
    model = build_factor_probe(WIDE_CONCAT)
    model.load_weights(str(output / "best.safetensors"))
    metrics = evaluate_factor_probe(
        model,
        FrozenFactorCache(validation_cache),
    )
    assert metrics["all_candidates_scored_once"]
    assert metrics["target_positive_recall"] == 1.0


def test_factor_probe_configuration_is_frozen() -> None:
    for kind in PROBE_KINDS:
        FactorProbeConfig(kind=kind, seed=PROBE_SEEDS[kind]).validate()
    with np.testing.assert_raises_regex(
        ValueError,
        "configuration drifted",
    ):
        FactorProbeConfig(
            kind=SCREEN_RELATIVE,
            seed=PROBE_SEEDS[SCREEN_RELATIVE],
            epochs=21,
        ).validate()


def test_mlx_cache_policy_preserves_loss_and_gradients() -> None:
    rng = np.random.default_rng(7)
    factors = mx.array(
        rng.normal(size=(4, FACTOR_COUNT, FACTOR_DIM)).astype(np.float32)
    )
    target = mx.array(np.array([True, False, True, False]))
    eligible = mx.ones((4,), dtype=mx.bool_)
    metadata = _metadata()
    mx.random.seed(PROBE_SEEDS[WIDE_CONCAT])
    model = build_factor_probe(WIDE_CONCAT)
    loss_and_grad = nn.value_and_grad(
        model,
        balanced_factor_binary_loss,
    )
    baseline_loss, baseline_gradients = loss_and_grad(
        model,
        factors,
        target,
        eligible,
        (0, 4),
        metadata["screen_rank"],
        metadata["action_hash"],
    )
    mx.eval(baseline_loss, baseline_gradients)
    baseline_values = {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(baseline_gradients)
    }

    previous_limit = mx.set_cache_limit(0)
    try:
        allocator = configure_mlx_memory()
        bounded_loss, bounded_gradients = loss_and_grad(
            model,
            factors,
            target,
            eligible,
            (0, 4),
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        mx.eval(bounded_loss, bounded_gradients)
        bounded_values = dict(tree_flatten(bounded_gradients))
        assert allocator["cache_limit_bytes"] > 0
        assert np.array_equal(
            np.asarray(baseline_loss),
            np.asarray(bounded_loss),
        )
        assert baseline_values.keys() == bounded_values.keys()
        for name, expected in baseline_values.items():
            assert np.array_equal(expected, np.asarray(bounded_values[name]))
    finally:
        mx.set_cache_limit(previous_limit)
        mx.clear_cache()
