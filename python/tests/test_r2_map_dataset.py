import json
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.r2_map_dataset import (
    COMPACT_INDEX_SCHEMA,
    D6_SCHEMA,
    DRAFT_FRAME_KIND,
    FEATURE_SCHEMA,
    FRAME_VERSION,
    HEADER_SIZE,
    IMITATION_SUBSET_PARTS_PER_MILLION,
    IMITATION_SUBSET_SCHEMA,
    MAGIC,
    MARKET_FRAME_KIND,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SPLIT_SCHEMA,
    TARGET_SCHEMA,
    R2MapCompactDatasetAdapter,
    R2MapDatasetError,
    R2MapFrameRef,
    R2MapStreamReader,
    _bounded_whole_game_window_chunks,
    _concatenate_supervised_batches,
    _decode_state,
    _pack_one,
    _run_exporter,
    compact_packing_plan,
    compact_storage_projection,
    d6_transform_id,
    draft_is_imitation_subset,
    validate_compact_index,
    validate_compact_index_value,
)
from cascadia_mlx.r2_map_market_decision import market_decision_action_id
from cascadia_mlx.r2_map_tensor_contract import BOARD_TOKEN_CAPACITY, TOKEN_FEATURES


def _state() -> bytes:
    counts = np.zeros((4, 4), dtype="<u2")
    counts[:, 0] = 1
    return b"".join(
        [
            struct.pack("<H", 4),
            counts.tobytes(),
            bytes([1, 1, 1, 1]),
            bytes([0, 1, 2, 3]),
            bytes(4 * 52),
            np.zeros(4 * 31, dtype="<f4").tobytes(),
            bytes([1, 1, 1, 1]),
            np.zeros(4 * 23, dtype="<f4").tobytes(),
            bytes([1, 1, 1, 1]),
            np.zeros(96, dtype="<f4").tobytes(),
        ]
    )


def _boundary_state(motif_count: int) -> bytes:
    counts = np.zeros((4, 4), dtype="<u2")
    counts[0] = [23, 50, 46, motif_count]
    token_count = int(counts.sum())
    types = np.repeat(np.arange(1, 5, dtype=np.uint8), counts[0].astype(np.int64))
    return b"".join(
        [
            struct.pack("<H", token_count),
            counts.tobytes(),
            types.tobytes(),
            bytes(token_count),
            bytes(token_count * 52),
            np.zeros(4 * 31, dtype="<f4").tobytes(),
            bytes([1, 1, 1, 1]),
            np.zeros(4 * 23, dtype="<f4").tobytes(),
            bytes([1, 1, 1, 1]),
            np.zeros(96, dtype="<f4").tobytes(),
        ]
    )


def test_compact_decoder_admits_slot_138_and_rejects_140_token_overflow() -> None:
    assert BOARD_TOKEN_CAPACITY == 139
    encoded = _boundary_state(20)
    state, cursor = _decode_state(memoryview(encoded), 0)
    assert cursor == len(encoded)
    assert state.token_mask.shape == (4, BOARD_TOKEN_CAPACITY)
    assert int(state.token_mask.sum()) == BOARD_TOKEN_CAPACITY
    assert state.token_types[0, -1] == 4

    overflow = _boundary_state(21)
    assert struct.unpack_from("<H", overflow)[0] == 140
    with pytest.raises(R2MapDatasetError, match="board counts"):
        _decode_state(memoryview(overflow), 0)


def _fixture_value(
    *,
    epoch: int = 0,
    sampler_seed: int = 0,
    game_indices: tuple[int, ...] = (),
    draft_decision_id: bytes | None = None,
    policy_target: bool | None = None,
) -> tuple[dict, bytes]:
    game_id = bytes([1]) * 32
    draft_decision_id = draft_decision_id or bytes([2]) * 32
    expected_policy_target = draft_is_imitation_subset(
        collection_kind="bootstrap", draft_decision_id=draft_decision_id
    )
    if policy_target is None:
        policy_target = expected_policy_target
    transform_id = d6_transform_id(
        game_id=game_id,
        draft_decision_id=draft_decision_id,
        mode=0,
        epoch=epoch,
        sampler_seed=sampler_seed,
    )
    current = np.arange(11, dtype="<u2")
    residual = np.ones(11, dtype="<i2")
    terminal = (current.astype(np.int32) + residual).astype("<u2")
    draft_fixed = b"".join(
        [
            game_id,
            draft_decision_id,
            bytes([4]) * 32,
            struct.pack("<QH", 7, 0),
            bytes([0, 0, transform_id, 0b111, 1, int(policy_target)]),
            struct.pack("<II", 1, 0),
            current.tobytes(),
            residual.tobytes(),
            terminal.tobytes(),
            bytes(3 * 26),
            bytes([0, 1, 2, 3]),
            bytes([0, 0, 0, 1]),
            bytes([0, 0, 0, 3]),
        ]
    )
    action = bytearray(128)
    action[6] = 255
    payload = b"".join(
        [
            bytes([0, 1, 2, FRAME_VERSION]),
            draft_fixed,
            _state(),
            bytes([4]) * 32,
            bytes(action),
            struct.pack("<H", int(current.sum())),
            _state(),
        ]
    )
    draft_frame = struct.pack("<I", len(payload)) + blake3.blake3(payload).digest() + payload
    market_action = bytes.fromhex("0101020000000000")
    market_decision_id = bytes([10]) * 32
    market_action_id = bytes.fromhex(
        market_decision_action_id(market_decision_id.hex(), market_action)
    )
    ordered_digest = blake3.blake3(
        json.dumps([market_action_id.hex()], separators=(",", ":")).encode()
    ).digest()
    market_payload = b"".join(
        [
            bytes([1, 0, 1, FRAME_VERSION]),
            game_id,
            bytes([9]) * 32,
            market_decision_id,
            market_action_id,
            bytes([11]) * 32,
            bytes([12]) * 32,
            ordered_digest,
            struct.pack("<QH", 7, 0),
            bytes([0, 0, transform_id, 0, 10, 1]),
            bytes([2, 2, 2, 2, 2]),
            bytes([0, 1, 2, 3]),
            bytes(3),
            struct.pack("<IIHHhH", 1, 0, 55, 66, 11, 0),
            _state(),
            market_action_id,
            market_action,
        ]
    )
    market_frame = (
        struct.pack("<I", len(market_payload))
        + blake3.blake3(market_payload).digest()
        + market_payload
    )
    source = {
        "file_name": "source.r2sh",
        "bytes": 123,
        "blake3": "a" * 64,
        "first_game_index": 7,
        "next_game_index": 8,
        "game_count": 1,
        "example_count": 1,
        "imitation_example_count": int(expected_policy_target),
        "market_decision_count": 1,
        "market_policy_target_count": 1,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "feature_schema": FEATURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "split_schema": SPLIT_SCHEMA,
        "d6_schema": D6_SCHEMA,
        "imitation_subset_schema": IMITATION_SUBSET_SCHEMA,
        "imitation_subset_parts_per_million": IMITATION_SUBSET_PARTS_PER_MILLION,
        "round": {
            "campaign_id": "r2-map-dataset-test",
            "iteration": 0,
            "collection_kind": "bootstrap",
            "newest_checkpoint_blake3": None,
        },
        "game_count": 1,
        "example_count": 1,
        "imitation_example_count": int(expected_policy_target),
        "market_decision_count": 1,
        "market_policy_target_count": 1,
        "train_games": 1,
        "validation_games": 0,
        "sources": [source],
    }
    dataset_hash = blake3.blake3(
        json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    manifest = {**identity, "dataset_blake3": dataset_hash}
    header = struct.pack(
        "<8sHH32s32sQQB3xQQQ",
        MAGIC,
        SCHEMA_VERSION,
        HEADER_SIZE,
        bytes.fromhex(dataset_hash),
        blake3.blake3(
            json.dumps(
                [
                    PROTOCOL_ID,
                    dataset_hash,
                    {
                        "mode": "train",
                        "epoch": epoch,
                        "sampler_seed": sampler_seed,
                        "fixed_panel_games": 0,
                        "game_indices": list(game_indices),
                    },
                ],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).digest(),
        2,
        1,
        0,
        epoch,
        sampler_seed,
        0,
    )
    return manifest, header + market_frame + draft_frame


def _fixture(tmp_path: Path, *, game_indices: tuple[int, ...] = ()) -> tuple[Path, Path]:
    manifest, stream = _fixture_value(game_indices=game_indices)
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(manifest))
    stream_path = tmp_path / "train.r2map"
    stream_path.write_bytes(stream)
    return manifest_path, stream_path


def test_stream_reader_materializes_only_batch_padding_and_exact_masks(tmp_path: Path) -> None:
    manifest, stream = _fixture(tmp_path)
    with R2MapStreamReader(manifest, stream) as reader:
        batch = reader.batch([0])
        assert batch.validate() == (1, 1)
        assert batch.inputs.action_features.shape == (1, 1, 140)
        assert batch.inputs.candidates.token_features.shape == (
            1,
            1,
            4,
            BOARD_TOKEN_CAPACITY,
            TOKEN_FEATURES,
        )
        assert batch.score_target_mask.tolist() == [[True]]
        assert batch.opponent_valid_mask.tolist() == [[True, True, True]]
        assert batch.market_final_slot_mask.tolist() == [[False, False, False, True]]
        assert batch.score_to_go_targets.tolist() == [[11.0]]
        assert batch.bootstrap_policy_mask.tolist() == [False]
        assert batch.market_decisions is not None
        assert batch.market_decisions.policy_target_mask.tolist() == [False]
        panel = reader.fixed_selected_batch([0])
        assert panel.candidate_mask.shape == (1, 1)
        np.testing.assert_array_equal(
            np.asarray(panel.action_features),
            np.asarray(batch.inputs.action_features),
        )
        assert reader.refs[0].global_game_index == 7


def test_cyclic_d6_schedule_covers_each_transform_once_per_12_epochs() -> None:
    game_id = bytes.fromhex("12" * 32)
    decision_id = bytes.fromhex("ab" * 32)
    observed = [
        d6_transform_id(
            game_id=game_id,
            draft_decision_id=decision_id,
            mode=0,
            epoch=epoch,
            sampler_seed=20260618,
        )
        for epoch in range(12)
    ]
    assert sorted(observed) == list(range(12))
    assert len(set(observed)) == 12
    assert (
        d6_transform_id(
            game_id=game_id,
            draft_decision_id=decision_id,
            mode=0,
            epoch=12,
            sampler_seed=20260618,
        )
        == observed[0]
    )
    assert (
        d6_transform_id(
            game_id=game_id,
            draft_decision_id=decision_id,
            mode=1,
            epoch=99,
            sampler_seed=88,
        )
        == 0
    )


def test_in_memory_v3_fixture_decodes_every_d6_transform() -> None:
    observed: list[int] = []
    for epoch in range(12):
        manifest, stream = _fixture_value(epoch=epoch, sampler_seed=20260618)
        with R2MapStreamReader(manifest, bytearray(stream)) as reader:
            draft_ref = next(ref for ref in reader.refs if ref.frame_kind == DRAFT_FRAME_KIND)
            market_ref = next(
                ref for ref in reader.market_refs if ref.frame_kind == MARKET_FRAME_KIND
            )
            draft_transform = reader.decode(draft_ref).transform_id
            market_transform = reader.decode_market(market_ref).transform_id
            assert draft_transform == market_transform
            observed.append(draft_transform)
    assert sorted(observed) == list(range(12))


def test_imitation_subset_is_deterministic_and_bootstrap_only() -> None:
    member = next(
        value.to_bytes(32, "little")
        for value in range(10_000)
        if draft_is_imitation_subset(
            collection_kind="bootstrap",
            draft_decision_id=value.to_bytes(32, "little"),
        )
    )
    assert draft_is_imitation_subset(collection_kind="bootstrap", draft_decision_id=member)
    assert not draft_is_imitation_subset(
        collection_kind="iterative-training", draft_decision_id=member
    )
    assert not draft_is_imitation_subset(collection_kind="benchmark", draft_decision_id=member)


def test_bootstrap_value_only_reader_requires_explicit_receipt_bound_mode() -> None:
    member = next(
        value.to_bytes(32, "little")
        for value in range(10_000)
        if draft_is_imitation_subset(
            collection_kind="bootstrap",
            draft_decision_id=value.to_bytes(32, "little"),
        )
    )
    manifest, stream = _fixture_value(
        draft_decision_id=member,
        policy_target=False,
    )

    with (
        R2MapStreamReader(manifest, bytearray(stream)) as reader,
        pytest.raises(R2MapDatasetError, match="frame metadata"),
    ):
        reader.decode(reader.refs[0])

    with R2MapStreamReader(
        manifest,
        bytearray(stream),
        bootstrap_value_only=True,
    ) as reader:
        frame = reader.decode(reader.refs[0])
        assert frame.bootstrap_policy_target is False
        assert frame.ref.candidate_count == 1
        assert frame.ref.selected_index == 0


def test_bounded_in_memory_stream_reader_matches_local_reader_exactly(tmp_path: Path) -> None:
    manifest_path, stream_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    payload = stream_path.read_bytes()
    with R2MapStreamReader(manifest_path, stream_path) as local:
        local_batch = local.batch([0])
        local_identity = local_batch.batch_identity
    with R2MapStreamReader(manifest, bytearray(payload)) as remote:
        remote_batch = remote.batch([0])
        assert remote.manifest_path is None
        assert remote.stream_path is None
        assert remote_batch.batch_identity == local_identity
        assert remote_batch.score_to_go_targets.tolist() == (
            local_batch.score_to_go_targets.tolist()
        )
        assert remote_batch.inputs.action_features.tolist() == (
            local_batch.inputs.action_features.tolist()
        )

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(R2MapDatasetError, match="checksum"):
        R2MapStreamReader(manifest, corrupted)


def test_remote_compact_adapter_uses_one_bounded_memory_window_and_no_paths(
    tmp_path: Path,
) -> None:
    manifest_path, stream_path = _fixture(tmp_path, game_indices=(7,))
    manifest = json.loads(manifest_path.read_text())
    game = {
        "source_file_name": "source.r2sh",
        "source_blake3": "a" * 64,
        "global_game_index": 7,
        "game_id": (bytes([1]) * 32).hex(),
        "example_count": 1,
        "imitation_example_count": 0,
        "market_decision_count": 1,
        "market_policy_target_count": 1,
        "split": "train",
    }
    index = {
        "schema_version": 1,
        "protocol_id": COMPACT_INDEX_SCHEMA,
        "dataset_manifest": manifest,
        "games": [game],
    }
    index["index_blake3"] = blake3.blake3(
        json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    validate_compact_index_value(index)
    packing = compact_packing_plan(
        index,
        {"source.r2sh": {game["game_id"]: (1,)}},
        group_batch_size=64,
        maximum_candidates_per_batch=16_384,
        seed=0,
        epochs=12,
    )
    assert [epoch["steps"] for epoch in packing["epoch_plans"]] == [1] * 12
    assert packing["totals"] == {
        "steps": 12,
        "draft_groups": 12,
        "selected_only_groups": 12,
        "draft_policy_targets": 0,
        "draft_candidates": 12,
        "padded_draft_candidates": 12,
    }
    calls: list[tuple[str, str, int, int]] = []

    def load(
        source: str,
        mode: str,
        epoch: int,
        seed: int,
        chunk_index: int,
        game_indices: tuple[int, ...],
    ):
        calls.append((source, mode, epoch, seed))
        assert chunk_index == 0
        assert game_indices == (7,)
        return manifest, bytearray(stream_path.read_bytes())

    with R2MapCompactDatasetAdapter(
        index=index,
        window_loader=load,
        maximum_prefetch_windows=0,
        maximum_window_bytes=1 << 20,
    ) as adapter:
        cursor, sampler = adapter.initial_state(0)
        tampered = dict(cursor)
        tampered["chunk_blake3"] = "0" * 64
        with pytest.raises(R2MapDatasetError, match="chunk cursor identity"):
            adapter.training_batch(tampered, sampler)
        batch = adapter.training_batch(cursor, sampler)
        assert batch.batch.batch_identity
        resumed = R2MapCompactDatasetAdapter(
            index=index,
            window_loader=load,
            maximum_prefetch_windows=0,
            maximum_window_bytes=1 << 20,
        )
        try:
            replayed = resumed.training_batch(cursor, sampler)
            assert replayed.batch.batch_identity == batch.batch.batch_identity
            assert replayed.next_cursor == batch.next_cursor
        finally:
            resumed.close()
        assert adapter.window_root is None
        assert adapter.shard_root is None
    assert calls == [("source.r2sh", "train", 0, 0)] * 2

    with pytest.raises(ValueError, match="forbids a second"):
        R2MapCompactDatasetAdapter(index=index, window_loader=load)


def test_receipt_bound_exporter_passes_all_three_binding_paths(monkeypatch, tmp_path: Path) -> None:
    observed: list[str] = []

    def run(command, **kwargs):
        observed.extend(command)
        assert kwargs["check"] is True

    monkeypatch.setattr("cascadia_mlx.r2_map_dataset.subprocess.run", run)
    aggregate = tmp_path / "aggregate.json"
    index = tmp_path / "index.json"
    packing = tmp_path / "packing.json"
    _run_exporter(
        tmp_path / "exporter",
        shard=tmp_path / "source.r2sh",
        manifest=tmp_path / "manifest.json",
        stream=tmp_path / "stream.r2map",
        mode="train",
        epoch=3,
        sampler_seed=7,
        compact_index=index,
        semantic_validation_binding=(aggregate, packing),
        game_indices=(),
    )
    assert observed[-6:] == [
        "--validated-aggregate-receipt",
        str(aggregate),
        "--validated-compact-index",
        str(index),
        "--validated-packing-receipt",
        str(packing),
    ]


def test_fixed_whole_game_chunks_preserve_order_and_memory_bound() -> None:
    games = [
        {
            "global_game_index": index,
            "game_id": f"{index:064x}",
            "example_count": 80,
            "candidate_widths": [1] * 80,
        }
        for index in range(64)
    ]
    chunks = _bounded_whole_game_window_chunks(
        "source.r2sh",
        "train",
        0,
        7,
        games,
        group_batch_size=128,
        maximum_candidates_per_batch=16_384,
    )
    assert [len(chunk.game_indices) for chunk in chunks] == [24, 24, 16]
    assert [item for chunk in chunks for item in chunk.game_indices] == list(range(64))
    assert all(len(chunk.game_indices) <= 32 for chunk in chunks)
    assert len({chunk.chunk_blake3 for chunk in chunks}) == len(chunks)

    high_branching_games = [
        {
            "global_game_index": index,
            "game_id": f"{index:064x}",
            "example_count": 1,
            "candidate_widths": [8_000],
        }
        for index in range(10)
    ]
    budgeted = _bounded_whole_game_window_chunks(
        "source.r2sh",
        "train",
        0,
        7,
        high_branching_games,
        group_batch_size=128,
        maximum_candidates_per_batch=16_384,
    )
    assert [len(chunk.game_indices) for chunk in budgeted] == [3, 3, 3, 1]
    assert all(chunk.nominal_bytes <= 384 << 20 for chunk in budgeted)


def test_full_batch_matches_concatenated_whole_game_window_parts(tmp_path: Path) -> None:
    manifest, stream = _fixture(tmp_path)
    with R2MapStreamReader(manifest, stream) as reader:
        full = reader.batch([0, 0])
        left, left_draft, left_market = reader._batch_with_identity_components([0])
        right, right_draft, right_market = reader._batch_with_identity_components([0])
        stitched = _concatenate_supervised_batches(
            [left, right],
            [*left_draft, *right_draft],
            [*left_market, *right_market],
        )

    assert stitched.validate() == full.validate() == (2, 1)
    assert stitched.batch_identity == full.batch_identity
    assert stitched.market_decisions is not None
    assert full.market_decisions is not None
    assert stitched.market_decisions.batch_identity == full.market_decisions.batch_identity
    for name in (
        "score_to_go_targets",
        "score_component_targets",
        "score_target_mask",
        "selected_action_index",
        "bootstrap_policy_mask",
        "opponent_tile_slot_targets",
        "opponent_wildlife_slot_targets",
        "opponent_draft_kind_targets",
        "opponent_drafted_wildlife_targets",
        "opponent_replace_three_targets",
        "opponent_paid_wipe_count_targets",
        "opponent_paid_wipe_mask_targets",
        "opponent_paid_wipe_mask_valid",
        "opponent_valid_mask",
        "market_disposition_targets",
        "market_pair_survival_targets",
        "market_final_slot_targets",
        "market_disposition_mask",
        "market_pair_survival_mask",
        "market_final_slot_mask",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(stitched, name)), np.asarray(getattr(full, name))
        )
    for name in (
        "candidate_mask",
        "action_features",
        "exact_afterstate_scores",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(stitched.inputs, name)),
            np.asarray(getattr(full.inputs, name)),
        )


def test_stream_reader_rejects_payload_corruption(tmp_path: Path) -> None:
    manifest, stream = _fixture(tmp_path)
    corrupted = bytearray(stream.read_bytes())
    corrupted[-1] ^= 1
    stream.write_bytes(corrupted)
    with pytest.raises(R2MapDatasetError, match="checksum"):
        R2MapStreamReader(manifest, stream)


def test_manifest_hash_and_source_accounting_fail_closed(tmp_path: Path) -> None:
    manifest, stream = _fixture(tmp_path)
    value = json.loads(manifest.read_text())
    value["sources"][0]["game_count"] = 2
    manifest.write_text(json.dumps(value))
    with pytest.raises(R2MapDatasetError):
        R2MapStreamReader(manifest, stream)


def test_compact_100k_projection_fits_budget_while_expanded_corpus_fails(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    value = {
        "schema_version": 1,
        "protocol_id": COMPACT_INDEX_SCHEMA,
        "dataset_manifest": manifest,
        "games": [
            {
                "source_file_name": "source.r2sh",
                "source_blake3": "a" * 64,
                "global_game_index": 7,
                "game_id": "01" * 32,
                "example_count": 1,
                "imitation_example_count": 0,
                "market_decision_count": 1,
                "market_policy_target_count": 1,
                "split": "train",
            }
        ],
    }
    value["index_blake3"] = blake3.blake3(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    index = tmp_path / "compact-index.json"
    index.write_text(json.dumps(value, separators=(",", ":")))
    assert validate_compact_index(index)["index_blake3"] == value["index_blake3"]

    projection = compact_storage_projection(
        index,
        target_games=100_000,
        maximum_window_bytes=1 << 30,
        maximum_prefetch_windows=1,
        expanded_bytes_per_game=2_000_000,
    )
    assert projection.compact_fits_run_budget is True
    assert projection.expanded_fits_run_budget is False
    assert projection.projected_peak_additional_bytes < 40 * (1 << 30)
    assert projection.projected_expanded_bytes == 200_000_000_000


def _frame_ref(candidate_count: int, ordinal: int) -> R2MapFrameRef:
    return R2MapFrameRef(
        payload_offset=0,
        payload_size=0,
        game_id=bytes([ordinal]) * 32,
        position_id=bytes([ordinal + 1]) * 32,
        global_game_index=ordinal,
        turn=ordinal,
        candidate_count=candidate_count,
        selected_index=0,
        frame_kind=0,
        ordinal=ordinal,
        stage=0,
        decision_id=bytes(32),
    )


def test_candidate_budget_accounts_for_rectangular_padding_and_preserves_cursor() -> None:
    refs = tuple(_frame_ref(width, index) for index, width in enumerate((3, 5, 2)))
    order = (0, 1, 2)
    assert _pack_one(refs, order, 0, groups=3, budget=10) == [0, 1]
    assert _pack_one(refs, order, 2, groups=3, budget=10) == [2]
    with pytest.raises(R2MapDatasetError, match="one R2-MAP group"):
        _pack_one((_frame_ref(11, 0),), (0,), 0, groups=1, budget=10)


def test_compact_index_hash_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    value = {
        "schema_version": 1,
        "protocol_id": COMPACT_INDEX_SCHEMA,
        "dataset_manifest": manifest,
        "games": [
            {
                "source_file_name": "source.r2sh",
                "source_blake3": "a" * 64,
                "global_game_index": 7,
                "game_id": "01" * 32,
                "example_count": 1,
                "imitation_example_count": 0,
                "market_decision_count": 1,
                "market_policy_target_count": 1,
                "split": "train",
            }
        ],
        "index_blake3": "0" * 64,
    }
    index = tmp_path / "compact-index.json"
    index.write_text(json.dumps(value))
    with pytest.raises(R2MapDatasetError, match="identity"):
        validate_compact_index(index)
