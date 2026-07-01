from __future__ import annotations

import struct
from pathlib import Path

import cascadia_mlx.s4_candidate_context_cache as cache_module
import numpy as np
import pytest
from cascadia_mlx.s4_candidate_context import (
    ANCHOR_LIMIT,
    MISSING_INDEX,
    RELATION_NEIGHBOR_LIMIT,
)
from cascadia_mlx.s4_candidate_context_cache import (
    CACHE_SCHEMA,
    EXPERIMENT_ID,
    FOUNDATION_REPORT_ID,
    S4CandidateContextCache,
    S4CandidateContextCacheError,
    _selected_relative_indices,
    merge_context_shards,
    read_context_container,
    write_context_container,
)
from cascadia_mlx.s4_candidate_relation_census import RELATIONS


def test_context_container_is_deterministic_and_detects_tampering(
    tmp_path: Path,
) -> None:
    identity = {"kind": "test", "value": 7}
    arrays = {
        "a": np.arange(9, dtype=np.uint16).reshape(3, 3),
        "b": np.arange(5, dtype=np.uint8),
    }
    first = tmp_path / "first.s4ctx"
    second = tmp_path / "second.s4ctx"

    first_id = write_context_container(
        first,
        scientific_identity=identity,
        arrays=arrays,
    )
    second_id = write_context_container(
        second,
        scientific_identity=identity,
        arrays=arrays,
    )

    assert first_id == second_id
    assert first.read_bytes() == second.read_bytes()
    loaded = read_context_container(first)
    assert loaded.container_id == first_id
    assert np.array_equal(loaded.arrays["a"], arrays["a"])
    assert np.array_equal(loaded.arrays["b"], arrays["b"])

    prefix = struct.Struct("<8sQ")
    _, header_bytes = prefix.unpack(first.read_bytes()[: prefix.size])
    payload_base = prefix.size + header_bytes
    with first.open("r+b") as handle:
        handle.seek(payload_base + arrays["a"].nbytes)
        handle.write(b"\x01")
    with pytest.raises(S4CandidateContextCacheError, match="padding"):
        read_context_container(first)

    first.write_bytes(second.read_bytes())
    with first.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(bytes([first.read_bytes()[-1] ^ 0xFF]))
    with pytest.raises(S4CandidateContextCacheError, match="checksum"):
        read_context_container(first)


def _synthetic_shard(
    path: Path,
    *,
    modulus: int,
    remainder: int,
) -> str:
    arrays: dict[str, np.ndarray] = {}
    row_summary: dict[str, list[int]] = {}
    for split, group_count in cache_module.EXPECTED_SPLIT_GROUPS.items():
        rows = np.arange(remainder, group_count, modulus, dtype=np.uint32)
        groups = len(rows)
        row_summary[split] = rows.astype(int).tolist()
        offsets = np.arange(groups + 1, dtype=np.uint64)
        hashes = np.zeros((groups, 32), dtype=np.uint8)
        hashes[:, 0] = rows.astype(np.uint8)
        hashes[:, 1] = 1 if split == "validation" else 0
        anchors = np.full(
            (groups, ANCHOR_LIMIT),
            MISSING_INDEX,
            dtype=np.uint16,
        )
        anchors[:, 0] = 0
        neighbors = np.full(
            (groups, len(RELATIONS), RELATION_NEIGHBOR_LIMIT),
            MISSING_INDEX,
            dtype=np.uint16,
        )
        arrays.update(
            {
                f"{split}/rows": rows,
                f"{split}/group_ids": (
                    rows.astype(np.uint64)
                    + (10_000 if split == "validation" else 0)
                ),
                f"{split}/candidate_offsets": offsets,
                f"{split}/selected_indices": np.zeros(
                    groups,
                    dtype=np.uint16,
                ),
                f"{split}/action_hashes": hashes,
                f"{split}/anchor_indices": anchors,
                f"{split}/relation_neighbor_indices": neighbors,
                f"{split}/relation_neighbor_counts": np.zeros(
                    (groups, len(RELATIONS)),
                    dtype=np.uint8,
                ),
                f"{split}/relation_anchor_sibling_counts": np.zeros(
                    (groups, len(RELATIONS)),
                    dtype=np.uint16,
                ),
            }
        )
    identity = {
        "schema_version": 1,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "kind": "shard",
        "source_bundle_id": "a" * 64,
        "foundation_report_id": FOUNDATION_REPORT_ID,
        "open_data_verification_id": "b" * 64,
        "r3_cache_id": "c" * 64,
        "anchor_limit": ANCHOR_LIMIT,
        "relation_neighbor_limit": RELATION_NEIGHBOR_LIMIT,
        "relations": list(RELATIONS),
        "row_shard": {"modulus": modulus, "remainder": remainder},
        "rows": row_summary,
    }
    return write_context_container(
        path,
        scientific_identity=identity,
        arrays=arrays,
    )


def test_context_shards_merge_order_invariant_and_bind_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cache_module,
        "EXPECTED_SPLIT_GROUPS",
        {"train": 4, "validation": 2},
    )
    monkeypatch.setattr(
        cache_module,
        "EXPECTED_SPLIT_CANDIDATES",
        {"validation": 2},
    )
    first = tmp_path / "shard-0.s4ctx"
    second = tmp_path / "shard-1.s4ctx"
    _synthetic_shard(first, modulus=2, remainder=0)
    _synthetic_shard(second, modulus=2, remainder=1)

    forward, forward_manifest, _ = merge_context_shards(
        [first, second],
        output_root=tmp_path / "forward",
    )
    reverse, reverse_manifest, _ = merge_context_shards(
        [second, first],
        output_root=tmp_path / "reverse",
    )

    assert forward_manifest["cache_id"] == reverse_manifest["cache_id"]
    assert (forward / "context.s4ctx").read_bytes() == (
        reverse / "context.s4ctx"
    ).read_bytes()
    cache = S4CandidateContextCache(forward)
    group = cache.splits["train"].group(3)
    assert group.group_id == 3
    assert group.context.anchor_indices[0] == 0
    assert np.all(group.context.anchor_indices[1:] == MISSING_INDEX)
    assert (
        cache.bind_action_hashes("train", 3, group.action_hashes).group_id
        == 3
    )
    drifted = group.action_hashes.copy()
    drifted[0, 0] ^= 1
    with pytest.raises(S4CandidateContextCacheError, match="identity drifted"):
        cache.bind_action_hashes("train", 3, drifted)

    hashes = np.zeros((2, 2, 32), dtype=np.uint8)
    masks = np.zeros((2, 2), dtype=np.bool_)
    for group_index, row in enumerate((3, 1)):
        value = cache.splits["train"].group(row).action_hashes
        hashes[group_index, : len(value)] = value
        masks[group_index, : len(value)] = True
    batch = cache.materialize(
        "train",
        np.asarray([3, 1]),
        action_hashes=hashes,
        candidate_mask=masks,
    )
    assert batch.rows.tolist() == [3, 1]
    assert batch.candidate_counts.tolist() == [1, 1]
    assert np.all(batch.anchor_mask[:, :1])
    assert not np.any(batch.relation_neighbor_mask)

    masks[0, 1] = True
    with pytest.raises(S4CandidateContextCacheError, match="identity drifted"):
        cache.materialize(
            "train",
            np.asarray([3, 1]),
            action_hashes=hashes,
            candidate_mask=masks,
        )


def test_selected_source_indices_are_converted_to_group_relative_indices() -> None:
    offsets = np.asarray([0, 3, 5], dtype=np.uint64)
    sources = np.asarray([7, 3, 9, 4, 8], dtype=np.uint16)
    selected = np.asarray([3, 8], dtype=np.uint16)

    assert _selected_relative_indices(offsets, sources, selected).tolist() == [
        1,
        1,
    ]

    with pytest.raises(S4CandidateContextCacheError, match="not unique"):
        _selected_relative_indices(
            offsets,
            sources,
            np.asarray([5, 8], dtype=np.uint16),
        )
