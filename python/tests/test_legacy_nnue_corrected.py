from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.legacy_nnue import (
    CORRECTED_NNUE_ARTIFACT_SCHEMA,
    CORRECTED_NNUE_CONTAINER_VERSION,
    CORRECTED_NNUE_HEADER,
    CORRECTED_NNUE_MAGIC,
    CORRECTED_NNUE_ROW_LAYOUT,
    CORRECTED_NNUE_SCHEMA_ID,
    CORRECTED_NNUE_SCHEMA_TAG,
    HISTORICAL_NNUE_SUPPORTED_FEATURES,
    LEGACY_NNUE_FEATURES,
    LEGACY_NNUE_HIDDEN1,
    LEGACY_NNUE_HIDDEN2,
    NNUE_SPLIT_HEADS,
    LegacyNnueError,
    LegacyNnueWeights,
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    convert_corrected_nnue,
    corrected_feature_for_historical,
    load_legacy_nnue_manifest,
    migrate_historical_nnue_to_corrected,
    pack_sparse_csr,
    parse_corrected_nnue,
    parse_legacy_nnue,
    reference_forward,
    remap_historical_features_to_corrected,
)


def _payload_float_count(
    features: int,
    hidden1: int,
    hidden2: int,
    version: int,
    *,
    policy: bool = True,
) -> int:
    count = features * hidden1
    count += hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1
    if policy:
        count += hidden2 + 1
        if version >= 2:
            count += 2 * (hidden2 + 1)
        if version >= 3:
            count += NNUE_SPLIT_HEADS * hidden2 + NNUE_SPLIT_HEADS
        if version >= 4:
            count += hidden2 + 1
    return count


def _write_corrected(
    path: Path,
    *,
    version: int = 1,
    features: int = 3,
    hidden1: int = 4,
    hidden2: int = 2,
    values: np.ndarray | None = None,
) -> np.ndarray:
    count = _payload_float_count(features, hidden1, hidden2, version)
    if values is None:
        values = np.arange(count, dtype=np.float32) + np.float32(0.25)
    assert values.shape == (count,)
    header = CORRECTED_NNUE_HEADER.pack(
        CORRECTED_NNUE_MAGIC,
        CORRECTED_NNUE_CONTAINER_VERSION,
        version,
        CORRECTED_NNUE_SCHEMA_TAG,
        features,
        hidden1,
        hidden2,
    )
    path.write_bytes(header + values.astype("<f4", copy=False).tobytes())
    return values


def _write_historical(
    path: Path,
    *,
    features: int,
    version: int = 4,
    hidden1: int = 2,
    hidden2: int = 2,
    policy: bool = True,
) -> np.ndarray:
    count = _payload_float_count(
        features,
        hidden1,
        hidden2,
        version,
        policy=policy,
    )
    values = np.arange(count, dtype=np.float32) + np.float32(0.5)
    path.write_bytes(
        b"NNUE" + struct.pack("<I", version) + values.astype("<f4", copy=False).tobytes()
    )
    return values


def _assert_bytes_equal(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    assert left.tobytes(order="C") == right.tobytes(order="C")


def _full_weights(version: int) -> LegacyNnueWeights:
    rng = np.random.default_rng(0xF5C0_13 + version)
    kwargs: dict[str, object] = {}
    if version >= 2:
        kwargs.update(
            has_split_value_heads=True,
            w3_wildlife=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
            b3_wildlife=rng.normal(size=1).astype(np.float32),
            w3_habitat=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
            b3_habitat=rng.normal(size=1).astype(np.float32),
        )
    if version >= 3:
        kwargs.update(
            has_split11_heads=True,
            w3_heads=rng.normal(size=(NNUE_SPLIT_HEADS, LEGACY_NNUE_HIDDEN2)).astype(np.float32),
            b3_heads=rng.normal(size=NNUE_SPLIT_HEADS).astype(np.float32),
        )
    if version >= 4:
        kwargs.update(
            has_heteroscedastic=True,
            w3_var=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
            b3_var=rng.normal(size=1).astype(np.float32),
        )
    w1 = np.zeros((LEGACY_NNUE_FEATURES, LEGACY_NNUE_HIDDEN1), dtype=np.float32)
    w1[:4] = rng.normal(size=(4, LEGACY_NNUE_HIDDEN1)).astype(np.float32)
    return LegacyNnueWeights(
        version=version,
        feature_count=LEGACY_NNUE_FEATURES,
        hidden1=LEGACY_NNUE_HIDDEN1,
        hidden2=LEGACY_NNUE_HIDDEN2,
        w1=w1,
        b1=rng.normal(size=LEGACY_NNUE_HIDDEN1).astype(np.float32),
        w2=rng.normal(size=(LEGACY_NNUE_HIDDEN1, LEGACY_NNUE_HIDDEN2)).astype(np.float32),
        b2=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
        w3=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
        b3=rng.normal(size=1).astype(np.float32),
        w3_policy=rng.normal(size=LEGACY_NNUE_HIDDEN2).astype(np.float32),
        b3_policy=rng.normal(size=1).astype(np.float32),
        container_magic=CORRECTED_NNUE_MAGIC,
        container_version=CORRECTED_NNUE_CONTAINER_VERSION,
        schema_id=CORRECTED_NNUE_SCHEMA_ID,
        **kwargs,
    )


def test_corrected_row_layout_is_dense_and_exposes_frozen_ranges() -> None:
    assert [
        (block.name, block.start, block.end, block.width) for block in CORRECTED_NNUE_ROW_LAYOUT
    ] == [
        ("historical_v2_base", 0, 10_561, 10_561),
        ("opponent_detail", 10_561, 10_930, 369),
        ("extended_tile_terrain_counts", 10_930, 11_080, 150),
        ("extended_tile_wildlife_capacity_counts", 11_080, 11_230, 150),
        ("overflow_used", 11_230, 11_231, 1),
    ]
    assert CORRECTED_NNUE_ROW_LAYOUT[-1].end == LEGACY_NNUE_FEATURES


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_parse_corrected_container_accepts_every_head_version(
    tmp_path: Path,
    version: int,
) -> None:
    source = tmp_path / f"corrected-v{version}.bin"
    _write_corrected(source, version=version)

    weights = parse_legacy_nnue(
        source,
        hidden1=4,
        hidden2=2,
        expected_features=3,
    )

    assert weights.is_corrected
    assert weights.version == version
    assert weights.schema_id == CORRECTED_NNUE_SCHEMA_ID
    assert weights.w1.shape == (3, 4)
    assert weights.has_split_value_heads is (version >= 2)
    assert weights.has_split11_heads is (version >= 3)
    assert weights.has_heteroscedastic is (version >= 4)
    assert set(weights.tensors()) == {
        "w1",
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "w3_policy",
        "b3_policy",
        *(("w3_wildlife", "b3_wildlife", "w3_habitat", "b3_habitat") if version >= 2 else ()),
        *(("w3_heads", "b3_heads") if version >= 3 else ()),
        *(("w3_var", "b3_var") if version >= 4 else ()),
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.__setitem__(slice(0, 4), b"NOPE"), "unknown magic"),
        (lambda raw: struct.pack_into("<I", raw, 4, 2), "container version"),
        (lambda raw: struct.pack_into("<I", raw, 8, 5), "head version"),
        (lambda raw: raw.__setitem__(12, raw[12] ^ 0xFF), "schema tag"),
        (lambda raw: struct.pack_into("<I", raw, 28, 4), "feature count"),
        (lambda raw: struct.pack_into("<I", raw, 32, 5), "architecture mismatch"),
        (lambda raw: struct.pack_into("<I", raw, 36, 3), "architecture mismatch"),
        (lambda raw: raw.extend(b"\0\0\0\0"), "file size mismatch"),
        (lambda raw: raw.__delitem__(slice(-4, None)), "file size mismatch"),
    ],
)
def test_corrected_container_corruptions_fail_closed(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    source = tmp_path / "corrected.bin"
    _write_corrected(source)
    raw = bytearray(source.read_bytes())
    mutation(raw)
    source.write_bytes(raw)

    with pytest.raises(LegacyNnueError, match=match):
        parse_legacy_nnue(
            source,
            hidden1=4,
            hidden2=2,
            expected_features=3,
        )


def test_corrected_container_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    source = tmp_path / "corrected.bin"
    values = _write_corrected(source)
    values[0] = np.nan
    _write_corrected(source, values=values)

    with pytest.raises(LegacyNnueError, match="non-finite"):
        parse_legacy_nnue(
            source,
            hidden1=4,
            hidden2=2,
            expected_features=3,
        )


@pytest.mark.parametrize("features", sorted(HISTORICAL_NNUE_SUPPORTED_FEATURES))
def test_every_recognized_historical_layout_migrates_exactly(
    tmp_path: Path,
    features: int,
) -> None:
    source = tmp_path / f"historical-{features}.bin"
    _write_historical(source, features=features)
    historical = parse_legacy_nnue(
        source,
        hidden1=2,
        hidden2=2,
        expected_features=None,
    )
    corrected = migrate_historical_nnue_to_corrected(historical)

    assert corrected.is_corrected
    assert corrected.feature_count == LEGACY_NNUE_FEATURES
    mapped_destinations = set()
    for source_feature in range(features):
        destination = corrected_feature_for_historical(
            source_feature,
            source_feature_count=features,
        )
        if destination is not None:
            mapped_destinations.add(destination)
            _assert_bytes_equal(
                historical.w1[source_feature],
                corrected.w1[destination],
            )
    unmapped = sorted(set(range(LEGACY_NNUE_FEATURES)) - mapped_destinations)
    assert np.all(corrected.w1[unmapped].view(np.uint32) == 0)

    for name in (
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "w3_policy",
        "b3_policy",
        "w3_wildlife",
        "b3_wildlife",
        "w3_habitat",
        "b3_habitat",
        "w3_heads",
        "b3_heads",
        "w3_var",
        "b3_var",
    ):
        _assert_bytes_equal(getattr(historical, name), getattr(corrected, name))


def test_historical_width_and_policyless_payload_compatibility(tmp_path: Path) -> None:
    policyless = tmp_path / "policyless.bin"
    _write_historical(
        policyless,
        features=5_197,
        version=1,
        policy=False,
    )
    weights = parse_legacy_nnue(
        policyless,
        hidden1=2,
        hidden2=2,
        expected_features=None,
    )
    assert not weights.has_policy
    assert np.array_equal(weights.w3_policy, np.zeros(2, dtype=np.float32))
    assert np.array_equal(weights.b3_policy, np.zeros(1, dtype=np.float32))

    unsupported = tmp_path / "unsupported.bin"
    _write_historical(unsupported, features=11_230, version=1)
    with pytest.raises(LegacyNnueError, match="unsupported first-layer width"):
        parse_corrected_nnue(unsupported, hidden1=2, hidden2=2)


def test_sparse_feature_remap_preserves_order_multiplicity_and_rejects_defect() -> None:
    assert remap_historical_features_to_corrected(
        [0, 10_862, 10_862, 11_230],
    ) == [0, 10_561, 10_561, 10_929]
    with pytest.raises(LegacyNnueError, match="discarded schema-defect"):
        remap_historical_features_to_corrected([10_561])
    assert remap_historical_features_to_corrected(
        [10_560, 10_561, 10_861, 10_862],
        reject_discarded=False,
    ) == [10_560, 10_561]


@pytest.mark.parametrize("version", [2, 4])
def test_rust_exact_mlx_matches_reference_for_extended_head_versions(version: int) -> None:
    weights = _full_weights(version)
    rows = [[], [0], [3, 2, 1], [1, 1, 0]]
    offsets, indices = pack_sparse_csr(rows)
    model = LegacyRustExactSparseNnue(weights.tensors())

    actual = model(offsets, indices)
    mx.eval(actual)
    expected = np.asarray([reference_forward(weights, row) for row in rows], dtype=np.float32)

    _assert_bytes_equal(np.asarray(actual), expected)


def test_corrected_artifact_roundtrip_and_manifest_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "corrected.bin"
    count = _payload_float_count(
        LEGACY_NNUE_FEATURES,
        LEGACY_NNUE_HIDDEN1,
        LEGACY_NNUE_HIDDEN2,
        1,
    )
    values = np.zeros(count, dtype=np.float32)
    _write_corrected(
        source,
        features=LEGACY_NNUE_FEATURES,
        hidden1=LEGACY_NNUE_HIDDEN1,
        hidden2=LEGACY_NNUE_HIDDEN2,
        values=values,
    )
    output = tmp_path / "mlx"
    manifest = convert_corrected_nnue(source, output)

    assert manifest["schema_version"] == CORRECTED_NNUE_ARTIFACT_SCHEMA
    assert manifest["source"]["schema_id"] == CORRECTED_NNUE_SCHEMA_ID
    assert "path" not in manifest["source"]
    assert "file" not in manifest["source"]
    assert LegacySparseNnue.load(output).tensors["w1"].shape == (
        LEGACY_NNUE_FEATURES,
        LEGACY_NNUE_HIDDEN1,
    )

    manifest_path = output / "model.json"
    corrupt = json.loads(manifest_path.read_text())
    corrupt["source"]["schema_tag_hex"] = "00" * 16
    manifest_path.write_text(json.dumps(corrupt))
    with pytest.raises(LegacyNnueError, match="source identity"):
        load_legacy_nnue_manifest(output)

    corrupt = manifest
    corrupt["source"]["bytes"] -= 4
    manifest_path.write_text(json.dumps(corrupt))
    with pytest.raises(LegacyNnueError, match="source identity"):
        load_legacy_nnue_manifest(output)
