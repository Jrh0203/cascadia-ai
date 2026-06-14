from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.legacy_nnue import (
    LEGACY_NNUE_FEATURES,
    LegacyNnueError,
    LegacyNnueWeights,
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    pack_sparse_csr,
    pack_sparse_features,
    parse_legacy_nnue,
    reference_forward,
)
from cascadia_mlx.legacy_nnue_tool import _load_and_validate_fixture


def _write_small_nnue(path: Path, *, features: int = 3, hidden1: int = 4, hidden2: int = 2):
    values = np.arange(
        features * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1 + hidden2 + 1,
        dtype=np.float32,
    )
    path.write_bytes(b"NNUE" + struct.pack("<I", 1) + values.astype("<f4").tobytes())


def test_parse_legacy_nnue_reconstructs_strict_layout(tmp_path: Path) -> None:
    source = tmp_path / "small.nnue"
    _write_small_nnue(source)
    weights = parse_legacy_nnue(source, hidden1=4, hidden2=2, expected_features=3)

    assert weights.version == 1
    assert weights.w1.shape == (3, 4)
    assert weights.w2.shape == (4, 2)
    assert weights.w3.shape == (2,)
    assert weights.w3_policy.shape == (2,)


def test_parse_legacy_nnue_rejects_trailing_bytes(tmp_path: Path) -> None:
    source = tmp_path / "small.nnue"
    _write_small_nnue(source)
    source.write_bytes(source.read_bytes() + b"\0")

    with pytest.raises(LegacyNnueError, match="row aligned"):
        parse_legacy_nnue(source, hidden1=4, hidden2=2, expected_features=3)


def test_sparse_pack_preserves_multiplicity_and_rejects_bounds() -> None:
    indices, mask = pack_sparse_features([[1, 1]], feature_count=3)
    assert np.array_equal(np.asarray(indices), np.asarray([[1, 1]], dtype=np.int32))
    assert np.array_equal(np.asarray(mask), np.asarray([[True, True]]))
    with pytest.raises(LegacyNnueError, match="out-of-range"):
        pack_sparse_features([[3]], feature_count=3)


def test_sparse_mlx_forward_matches_reference() -> None:
    rng = np.random.default_rng(7)
    weights = LegacyNnueWeights(
        version=1,
        feature_count=3,
        hidden1=4,
        hidden2=2,
        w1=rng.normal(size=(3, 4)).astype(np.float32),
        b1=rng.normal(size=4).astype(np.float32),
        w2=rng.normal(size=(4, 2)).astype(np.float32),
        b2=rng.normal(size=2).astype(np.float32),
        w3=rng.normal(size=2).astype(np.float32),
        b3=rng.normal(size=1).astype(np.float32),
        w3_policy=np.zeros(2, dtype=np.float32),
        b3_policy=np.zeros(1, dtype=np.float32),
    )
    tensors = weights.tensors()
    model = object.__new__(LegacySparseNnue)
    model.tensors = tensors
    rows = [[], [0], [2, 1], [1, 1]]
    indices, mask = pack_sparse_features(rows, feature_count=3)
    actual = model(indices, mask)
    mx.eval(actual)
    expected = np.asarray([reference_forward(weights, row) for row in rows])

    assert np.allclose(np.asarray(actual), expected, atol=1e-5)


def test_rust_exact_sparse_mlx_forward_is_bit_identical() -> None:
    rng = np.random.default_rng(11)
    w1 = np.zeros((LEGACY_NNUE_FEATURES, 512), dtype=np.float32)
    w1[:3] = rng.normal(size=(3, 512)).astype(np.float32)
    weights = LegacyNnueWeights(
        version=1,
        feature_count=LEGACY_NNUE_FEATURES,
        hidden1=512,
        hidden2=64,
        w1=w1,
        b1=rng.normal(size=512).astype(np.float32),
        w2=rng.normal(size=(512, 64)).astype(np.float32),
        b2=rng.normal(size=64).astype(np.float32),
        w3=rng.normal(size=64).astype(np.float32),
        b3=rng.normal(size=1).astype(np.float32),
        w3_policy=np.zeros(64, dtype=np.float32),
        b3_policy=np.zeros(1, dtype=np.float32),
    )
    model = LegacyRustExactSparseNnue(weights.tensors())
    rows = [[], [0], [2, 1], [1, 1]]
    offsets, indices = pack_sparse_csr(rows)
    actual = model(offsets, indices)
    hidden, hidden_output = model.hidden_and_output(offsets, indices)
    mx.eval(actual, hidden, hidden_output)
    expected = np.asarray([reference_forward(weights, row) for row in rows])

    assert np.array_equal(np.asarray(actual), expected)
    assert np.array_equal(np.asarray(hidden_output), expected)
    assert np.array_equal(np.asarray(actual), np.asarray(hidden_output))
    assert np.asarray(hidden).shape == (len(rows), 64)
    assert np.all(np.asarray(hidden) >= 0.0)


def test_parity_fixture_validation_preserves_duplicate_metadata(tmp_path: Path) -> None:
    records = [
        {
            "game_index": 92_000,
            "decision_index": decision,
            "active_seat": decision % 4,
            "rust_value": float(decision),
            "features": [1, 1] if decision == 0 else [decision + 2],
        }
        for decision in range(80)
    ]
    fixture = {
        "schema_version": 1,
        "feature_schema": "legacy-mid-v4opp-sparse-u16-v1",
        "split": "train",
        "first_game_index": 92_000,
        "games": 1,
        "feature_count": 11_231,
        "hidden1": 512,
        "hidden2": 64,
        "records_with_duplicate_features": 1,
        "duplicate_feature_occurrences": 1,
        "maximum_feature_multiplicity": 2,
        "records": records,
        "provenance": {
            "source": {"v2_source_blake3": "a" * 64},
            "executable_blake3": "b" * 64,
            "weights_blake3": ("9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"),
            "legacy_environment": [
                ["MCE_LMR", "1"],
                ["MCE_DIVERSE_PREFILTER", "1"],
            ],
        },
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))

    assert _load_and_validate_fixture(path)["duplicate_feature_occurrences"] == 1
    fixture["duplicate_feature_occurrences"] = 0
    path.write_text(json.dumps(fixture))
    with pytest.raises(LegacyNnueError, match="duplicate metadata"):
        _load_and_validate_fixture(path)
