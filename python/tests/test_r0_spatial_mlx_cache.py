from __future__ import annotations

import json
from pathlib import Path

import blake3
import cascadia_mlx.r0_spatial_mlx_cache as cache_module
import numpy as np
import pytest
from cascadia_mlx.r0_spatial_mlx_cache import (
    ARM_LOCAL_CAPACITY,
    ARM_TOKEN_CAPACITY,
    BOARD_SLOTS,
    D6_TRANSFORMS,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    MAX_ENTITIES_PER_BOARD,
    SLOT_SENTINEL,
    TARGET_DIM,
    TOKEN_FIELDS,
    R0SpatialMlxCache,
    R0SpatialMlxCacheError,
)


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _tiny_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = "exact-entity-control",
) -> tuple[Path, Path]:
    split_records = {"train": 2, "validation": 1}
    monkeypatch.setattr(cache_module, "EXPECTED_SPLIT_RECORDS", split_records)
    monkeypatch.setattr(cache_module, "EXPECTED_TOTAL_RECORDS", 3)
    staging = tmp_path / "staging"
    staging.mkdir()
    split_manifests: dict[str, dict] = {}
    scientific_files: dict[str, dict] = {}
    local_capacity = ARM_LOCAL_CAPACITY[arm]

    for split, records in split_records.items():
        slots = np.full(
            (records, D6_TRANSFORMS, BOARD_SLOTS, MAX_ENTITIES_PER_BOARD),
            SLOT_SENTINEL,
            dtype="<u2",
        )
        features = np.zeros(
            (
                records,
                D6_TRANSFORMS,
                BOARD_SLOTS,
                MAX_ENTITIES_PER_BOARD,
                TOKEN_FIELDS,
            ),
            dtype="i1",
        )
        slots[..., 0] = 0
        slots[..., 1] = 1 if arm == "exact-entity-control" else local_capacity
        features[..., 0, 4] = 1 if arm == "exact-entity-control" else 2
        features[..., 1, 4] = 1 if arm == "exact-entity-control" else 3
        features[..., :2, 5] = 1
        features[..., :2, 6] = 5
        features[..., :2, 9] = 5
        arrays = {
            "token_slots": slots,
            "token_features": features,
            "market_features": np.zeros((records, 4, MARKET_FEATURES), dtype="<f4"),
            "market_mask": np.ones((records, 4), dtype="u1"),
            "global_features": np.zeros((records, GLOBAL_FEATURES), dtype="<f4"),
            "targets": np.zeros((records, TARGET_DIM), dtype="<f4"),
            "game_index": np.arange(records, dtype="<u8"),
            "turn": np.zeros(records, dtype="u1"),
            "board_counts": np.full((records, BOARD_SLOTS), 2, dtype="u1"),
        }
        dtype_names = {
            "token_slots": "<u2",
            "token_features": "|i1",
            "market_features": "<f4",
            "market_mask": "|u1",
            "global_features": "<f4",
            "targets": "<f4",
            "game_index": "<u8",
            "turn": "|u1",
            "board_counts": "|u1",
        }
        files = {}
        for name, array in arrays.items():
            path = staging / f"{split}-{name}.bin"
            array.tofile(path)
            files[name] = {
                "file": path.name,
                "dtype": dtype_names[name],
                "shape": list(array.shape),
                "bytes": path.stat().st_size,
                "blake3": _checksum(path),
            }
        active = records * D6_TRANSFORMS * BOARD_SLOTS * 2
        total = records * D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD
        overflow_rows = 0 if arm == "exact-entity-control" else records * BOARD_SLOTS
        d6_overflow_rows = overflow_rows * D6_TRANSFORMS
        integrity = {
            "records": records,
            "source_entity_rows": records * BOARD_SLOTS * 2,
            "exported_active_token_rows": active,
            "exported_padding_token_rows": total - active,
            "identity_overflow_entity_rows": overflow_rows,
            "identity_positions_with_overflow": 0 if not overflow_rows else records,
            "d6_overflow_entity_rows": d6_overflow_rows,
            "d6_positions_with_overflow": 0 if not d6_overflow_rows else records * D6_TRANSFORMS,
        }
        split_manifests[split] = {
            "records": records,
            "files": files,
            "integrity": integrity,
        }
        scientific_files[split] = files

    lock_identity = {
        "feature_schema": "compact-entity-v2",
        "target_schema": "base-score-components-v1",
        "total_records": 3,
        "train_records": 2,
        "validation_records": 1,
        "source_v2_blake3": "1" * 64,
        "corpus_blake3": "2" * 64,
        "datasets": [{"order": index} for index in range(8)],
    }
    lock = {
        "schema_version": 1,
        "contract_id": "r0-frozen-60000-position-corpus-v1",
        "lock_id": cache_module._canonical_blake3(lock_identity),
        "identity": lock_identity,
    }
    lock_path = tmp_path / "corpus-lock.json"
    lock_path.write_text(json.dumps(lock))

    scientific_identity = {
        "arm": arm,
        "cache_schema": "r0-spatial-mlx-cache-v1",
        "corpus_blake3": lock_identity["corpus_blake3"],
        "corpus_lock_id": lock["lock_id"],
        "d6_semantic_blake3": "3" * 64,
        "d6_transform_ids": list(range(D6_TRANSFORMS)),
        "experiment_id": "r0-spatial-mlx-tournament-v1",
        "exporter_executable_blake3": "6" * 64,
        "exporter_source_v2_blake3": "7" * 64,
        "files": scientific_files,
        "source_semantic_blake3": "4" * 64,
        "spatial_token_capacity": ARM_TOKEN_CAPACITY[arm],
        "split_records": split_records,
        "target_blake3": "5" * 64,
    }
    cache_id = cache_module._canonical_blake3(scientific_identity)
    root = tmp_path / cache_id
    root.mkdir()
    for path in staging.iterdir():
        path.rename(root / path.name)
    staging.rmdir()
    manifest = {
        "schema_version": 1,
        "cache_schema": "r0-spatial-mlx-cache-v1",
        "experiment_id": "r0-spatial-mlx-tournament-v1",
        "cache_id": cache_id,
        "arm": arm,
        "scientific_identity": scientific_identity,
        "tensor_contract": {
            "board_slots": BOARD_SLOTS,
            "d6_transform_ids": list(range(D6_TRANSFORMS)),
            "global_feature_dim": GLOBAL_FEATURES,
            "local_capacity": local_capacity,
            "market_feature_dim": MARKET_FEATURES,
            "max_entities_per_board": MAX_ENTITIES_PER_BOARD,
            "spatial_token_capacity": ARM_TOKEN_CAPACITY[arm],
        },
        "corpus": {
            "contract_id": "r0-frozen-60000-position-corpus-v1",
            "lock_id": lock["lock_id"],
            "identity": lock_identity,
        },
        "semantic_integrity": {
            "identity_round_trip_verified": True,
            "packed_round_trip_verified": True,
            "packed_round_trip_records": 3,
            "d6_inverse_round_trip_verified": True,
            "source_semantic_blake3": "4" * 64,
            "d6_semantic_blake3": "3" * 64,
            "target_blake3": "5" * 64,
        },
        "overflow_integrity": {"exact_entities_retained": True},
        "splits": split_manifests,
        "exporter": {
            "executable_blake3": "6" * 64,
            "source_provenance": {"v2_source_blake3": "7" * 64},
        },
    }
    (root / "cache.json").write_text(json.dumps(manifest))
    return root, lock_path


@pytest.mark.parametrize(
    ("arm", "expected_capacity"),
    [
        ("exact-entity-control", 23),
        ("hex-radius-4-61", 84),
    ],
)
def test_cache_verifies_and_scatter_materializes_explicit_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    expected_capacity: int,
) -> None:
    root, lock = _tiny_cache(tmp_path, monkeypatch, arm=arm)
    cache = R0SpatialMlxCache(root, corpus_lock=lock)
    batch = cache.batch("train", [0, 1], transform_ids=[0, 11])
    tokens = np.asarray(batch.spatial_tokens)
    mask = np.asarray(batch.spatial_mask)
    assert tokens.shape == (2, 4, expected_capacity, TOKEN_FIELDS)
    assert mask.shape == (2, 4, expected_capacity)
    assert np.all(mask.sum(axis=-1) == 2)
    assert np.all(tokens[~mask] == 0)
    if arm == "exact-entity-control":
        assert np.all(tokens[:, :, :2, 4] == 1)
    else:
        assert np.all(tokens[:, :, 0, 4] == 2)
        assert np.all(tokens[:, :, ARM_LOCAL_CAPACITY[arm], 4] == 3)


def test_cache_rejects_checksum_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock = _tiny_cache(tmp_path, monkeypatch)
    target = root / "train-targets.bin"
    target.write_bytes(target.read_bytes()[:-1] + b"\x01")
    with pytest.raises(R0SpatialMlxCacheError, match="BLAKE3"):
        R0SpatialMlxCache(root, corpus_lock=lock)


def test_cache_rejects_nonzero_padding_even_without_checksum_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock = _tiny_cache(tmp_path, monkeypatch)
    path = root / "train-token_features.bin"
    values = np.memmap(
        path,
        mode="r+",
        dtype="i1",
        shape=(2, D6_TRANSFORMS, BOARD_SLOTS, MAX_ENTITIES_PER_BOARD, TOKEN_FIELDS),
    )
    values[0, 0, 0, 2, 0] = 1
    values.flush()
    with pytest.raises(R0SpatialMlxCacheError, match="padding"):
        R0SpatialMlxCache(
            root,
            corpus_lock=lock,
            verify_checksums=False,
        )


def test_cache_rejects_directory_name_that_is_not_the_content_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock = _tiny_cache(tmp_path, monkeypatch)
    renamed = root.with_name("wrong-cache-id")
    root.rename(renamed)
    with pytest.raises(R0SpatialMlxCacheError, match="directory name"):
        R0SpatialMlxCache(renamed, corpus_lock=lock)


def test_training_batches_are_resume_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock = _tiny_cache(tmp_path, monkeypatch)
    cache = R0SpatialMlxCache(root, corpus_lock=lock)
    first = cache.deterministic_training_batch(step=17, batch_size=8, seed=991)
    second = cache.deterministic_training_batch(step=17, batch_size=8, seed=991)
    assert np.array_equal(np.asarray(first.game_index), np.asarray(second.game_index))
    assert np.array_equal(np.asarray(first.spatial_tokens), np.asarray(second.spatial_tokens))
    different = cache.deterministic_training_batch(step=18, batch_size=8, seed=991)
    assert not np.array_equal(
        np.asarray(first.spatial_tokens),
        np.asarray(different.spatial_tokens),
    ) or not np.array_equal(
        np.asarray(first.game_index),
        np.asarray(different.game_index),
    )
