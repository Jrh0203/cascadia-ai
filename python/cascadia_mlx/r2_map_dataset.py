"""Strict lazy reader for replay-authoritative R2-MAP exact-R2 streams."""

from __future__ import annotations

import json
import mmap
import os
import struct
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_FEATURE_SIZE,
    GRADED_ORACLE_MAX_WILDLIFE_WIPES,
    decode_graded_action_feature_bytes,
)
from cascadia_mlx.r2_map_contracts import (
    CAMPAIGN_ROOT,
    StoragePreflightError,
    require_local_storage_authority,
)
from cascadia_mlx.r2_map_market_decision import (
    MARKET_DECISION_ACTION_SIZE,
    MarketDecisionActionKind,
    MarketDecisionKind,
    decode_market_decision_action_bytes,
    market_decision_action_id,
    validate_canonical_market_action_order,
)
from cascadia_mlx.r2_map_model import (
    R2MapBatch,
    R2MapMarketDecisionBatch,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
    TOKEN_PAYLOAD_WIDTH,
)
from cascadia_mlx.r2_map_training_contract import (
    R2MapAdapterStep,
    R2MapMarketDecisionSupervision,
    R2MapSupervisedBatch,
    R2MapTrainingAdapter,
)
from cascadia_mlx.r2_sparse_mlx_cache import _materialize_token_features

SCHEMA_VERSION = 3
PROTOCOL_ID = "r2-map-streaming-exact-r2-v3"
COMPACT_PROTOCOL_ID = "r2-map-compact-bounded-window-v7"
COMPACT_INDEX_SCHEMA = "r2-map-compact-index-v4"
MAGIC = b"CSDR2MP\0"
HEADER_SIZE = 120
FRAME_HEADER_SIZE = 36
FRAME_PREFIX_SIZE = 4
DRAFT_FRAME_KIND = 0
MARKET_FRAME_KIND = 1
FRAME_VERSION = 2
MARKET_FIXED_SIZE = 272
COMPONENTS = 11
OPPONENTS = 3
MARKET_SLOTS = 4
SPLIT_SCHEMA = "r2-map-whole-game-split-v1"
D6_SCHEMA = "r2-map-d6-cyclic-offset-v1"
IMITATION_SUBSET_SCHEMA = "r2-map-draft-imitation-subset-v1"
IMITATION_SUBSET_PARTS_PER_MILLION = 10_000
FEATURE_SCHEMA = "exact-r2-staged-public-market-and-draft-v3"
TARGET_SCHEMA = "selected-value-plus-deterministic-full-draft-imitation-v3"
MAX_IN_MEMORY_STREAM_BYTES = 1 << 30
WINDOW_CHUNKING_PROTOCOL = "candidate-budgeted-whole-game-window-with-batch-stitching-v5"
WINDOW_TARGET_GAMES = 24
WINDOW_MAXIMUM_NOMINAL_GAMES = 32
WINDOW_NOMINAL_BYTES_PER_DRAFT_GROUP = 64 << 10
WINDOW_NOMINAL_BYTES_PER_CANDIDATE = 16 << 10
WINDOW_PREFERRED_TARGET_BYTES = 384 << 20
WINDOW_NOMINAL_TARGET_BYTES = 512 << 20
WINDOW_PLAN_EPOCHS = 12

_HEADER = struct.Struct("<8sHH32s32sQQB3xQQQ")


class R2MapDatasetError(ValueError):
    """Raised when an R2-MAP manifest or stream fails closed validation."""


@dataclass(frozen=True, slots=True)
class R2MapFrameRef:
    payload_offset: int
    payload_size: int
    game_id: bytes
    position_id: bytes
    global_game_index: int
    turn: int
    candidate_count: int
    selected_index: int
    frame_kind: int
    ordinal: int
    stage: int
    decision_id: bytes


@dataclass(frozen=True, slots=True)
class _State:
    token_features: np.ndarray
    token_types: np.ndarray
    token_mask: np.ndarray
    market_features: np.ndarray
    market_mask: np.ndarray
    player_features: np.ndarray
    player_mask: np.ndarray
    global_features: np.ndarray


@dataclass(frozen=True, slots=True)
class _Frame:
    ref: R2MapFrameRef
    parent: _State
    candidates: tuple[_State, ...]
    action_features: np.ndarray
    exact_afterstate_scores: np.ndarray
    current_components: np.ndarray
    residual_components: np.ndarray
    terminal_components: np.ndarray
    opponent_valid: np.ndarray
    opponent_targets: np.ndarray
    opponent_paid_wipe_count: np.ndarray
    opponent_paid_wipe_masks: np.ndarray
    opponent_paid_wipe_mask_valid: np.ndarray
    bootstrap_policy_target: bool
    market_valid: bool
    market_disposition: np.ndarray
    market_pair_survival: np.ndarray
    market_final_slot: np.ndarray
    transform_id: int


@dataclass(frozen=True, slots=True)
class _MarketFrame:
    ref: R2MapFrameRef
    parent: _State
    action_bytes: np.ndarray
    action_features: np.ndarray
    action_ids: tuple[bytes, ...]
    exact_current_score: float
    final_score: float
    score_to_go: float
    selected_action_id: bytes
    parent_public_hash: bytes
    resulting_public_hash: bytes
    public_nature_tokens: int
    public_wildlife_bag_total: int
    public_wildlife_bag_counts: tuple[int, ...]
    public_market_wildlife: tuple[int, ...]
    policy_target: bool
    transform_id: int


class R2MapStreamReader:
    """Read verified frames from a local mmap or a bounded in-memory remote object."""

    def __init__(
        self,
        manifest: str | Path | Mapping[str, Any],
        stream: str | Path | bytes | bytearray,
        *,
        game_indices: Sequence[int] = (),
        ordered_game_indices: bool = False,
        bootstrap_value_only: bool = False,
    ):
        self._closed = False
        self.game_indices = tuple(int(value) for value in game_indices)
        self.ordered_game_indices = bool(ordered_game_indices)
        self.bootstrap_value_only = bool(bootstrap_value_only)
        if any(value < 0 for value in self.game_indices) or len(set(self.game_indices)) != len(
            self.game_indices
        ):
            raise R2MapDatasetError("R2-MAP reader game-index binding is invalid")
        self._owned_bytes: bytes | bytearray | None = None
        if isinstance(manifest, Mapping) and isinstance(stream, (bytes, bytearray)):
            if len(stream) > MAX_IN_MEMORY_STREAM_BYTES:
                raise R2MapDatasetError("remote R2-MAP stream exceeds the 1 GiB memory gate")
            self.manifest_path: Path | None = None
            self.stream_path: Path | None = None
            self.manifest = _validate_manifest_value(dict(manifest))
            self._file = None
            self._owned_bytes = stream
            self._map: mmap.mmap | memoryview = memoryview(self._owned_bytes).toreadonly()
        elif not isinstance(manifest, Mapping) and not isinstance(stream, (bytes, bytearray)):
            self.manifest_path = Path(manifest)
            self.stream_path = Path(stream)
            self.manifest = _read_manifest(self.manifest_path)
            self._file = self.stream_path.open("rb")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            raise TypeError("R2-MAP manifest and stream must both be paths or memory objects")
        (
            self.dataset_blake3,
            self.config_blake3,
            self.mode,
            self.epoch,
            self.sampler_seed,
            expected,
        ) = self._header()
        if self.dataset_blake3 != self.manifest["dataset_blake3"]:
            raise R2MapDatasetError("R2-MAP stream dataset hash differs from manifest")
        all_refs = self._scan(expected)
        observed_game_indices = {ref.global_game_index for ref in all_refs}
        if self.game_indices and observed_game_indices != set(self.game_indices):
            raise R2MapDatasetError("R2-MAP bounded stream game indices differ")
        self.refs = tuple(ref for ref in all_refs if ref.frame_kind == DRAFT_FRAME_KIND)
        self.market_refs = tuple(ref for ref in all_refs if ref.frame_kind == MARKET_FRAME_KIND)
        if len(self.refs) > self.manifest["example_count"]:
            raise R2MapDatasetError("R2-MAP draft frame count exceeds dataset manifest")
        if len(self.market_refs) > self.manifest["market_decision_count"]:
            raise R2MapDatasetError("R2-MAP market frame count exceeds dataset manifest")
        market_by_turn: dict[tuple[int, int], list[R2MapFrameRef]] = {}
        for ref in self.market_refs:
            market_by_turn.setdefault((ref.global_game_index, ref.turn), []).append(ref)
        self.market_by_turn = {key: tuple(values) for key, values in market_by_turn.items()}
        self._validate_turn_frame_sequences()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._map.close() if isinstance(self._map, mmap.mmap) else self._map.release()
        if self._file is not None:
            self._file.close()
        self._owned_bytes = None

    def __enter__(self) -> R2MapStreamReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _header(self) -> tuple[str, str, int, int, int, int]:
        if len(self._map) < HEADER_SIZE:
            raise R2MapDatasetError("R2-MAP stream header is truncated")
        (
            magic,
            version,
            size,
            dataset,
            config,
            frames,
            games,
            mode,
            epoch,
            sampler_seed,
            fixed_panel_games,
        ) = _HEADER.unpack_from(self._map)
        if magic != MAGIC or version != SCHEMA_VERSION or size != HEADER_SIZE:
            raise R2MapDatasetError("unsupported R2-MAP stream envelope")
        if mode not in (0, 1, 2) or games > self.manifest["game_count"]:
            raise R2MapDatasetError("R2-MAP stream mode or game count is invalid")
        mode_name = ("train", "validation", "fixed-panel")[mode]
        config_identity = [
            PROTOCOL_ID,
            self.manifest["dataset_blake3"],
            {
                "mode": mode_name,
                "epoch": epoch,
                "sampler_seed": sampler_seed,
                "fixed_panel_games": fixed_panel_games,
                "game_indices": list(self.game_indices),
            },
        ]
        expected_config = blake3.blake3(
            json.dumps(config_identity, separators=(",", ":"), ensure_ascii=True).encode()
        ).digest()
        if config != expected_config:
            raise R2MapDatasetError("R2-MAP stream configuration hash failed")
        return dataset.hex(), config.hex(), mode, epoch, sampler_seed, frames

    def _scan(self, expected: int) -> tuple[R2MapFrameRef, ...]:
        cursor = HEADER_SIZE
        refs: list[R2MapFrameRef] = []
        previous_key: tuple[int, int, int, int] | None = None
        for _ in range(expected):
            if cursor + FRAME_HEADER_SIZE > len(self._map):
                raise R2MapDatasetError("R2-MAP frame header is truncated")
            size = struct.unpack_from("<I", self._map, cursor)[0]
            digest = self._map[cursor + 4 : cursor + FRAME_HEADER_SIZE]
            payload_offset = cursor + FRAME_HEADER_SIZE
            payload_end = payload_offset + size
            if payload_end > len(self._map):
                raise R2MapDatasetError("R2-MAP frame payload is truncated")
            payload = self._map[payload_offset:payload_end]
            if blake3.blake3(payload).digest() != digest:
                raise R2MapDatasetError("R2-MAP frame checksum failed")
            if size < FRAME_PREFIX_SIZE:
                raise R2MapDatasetError("R2-MAP frame is shorter than its kind prefix")
            frame_kind, ordinal, stage, frame_version = payload[:FRAME_PREFIX_SIZE]
            if frame_version != FRAME_VERSION or frame_kind not in {
                DRAFT_FRAME_KIND,
                MARKET_FRAME_KIND,
            }:
                raise R2MapDatasetError("R2-MAP frame kind or version is unsupported")
            if frame_kind == DRAFT_FRAME_KIND:
                if size < 124 or stage != 2:
                    raise R2MapDatasetError("R2-MAP draft frame prefix is invalid")
                game_id = bytes(payload[4:36])
                position_id = bytes(payload[36:68])
                decision_id = position_id
                game_index = struct.unpack_from("<Q", payload, 100)[0]
                turn = struct.unpack_from("<H", payload, 108)[0]
                candidate_count, selected_index = struct.unpack_from("<II", payload, 116)
            else:
                if size < MARKET_FIXED_SIZE or stage not in {0, 1}:
                    raise R2MapDatasetError("R2-MAP market frame prefix is invalid")
                game_id = bytes(payload[4:36])
                position_id = bytes(payload[36:68])
                decision_id = bytes(payload[68:100])
                game_index = struct.unpack_from("<Q", payload, 228)[0]
                turn = struct.unpack_from("<H", payload, 236)[0]
                candidate_count, selected_index = struct.unpack_from("<II", payload, 256)
            if candidate_count == 0 or selected_index >= candidate_count:
                raise R2MapDatasetError("R2-MAP candidate count or selected index is invalid")
            game_order = (
                self.game_indices.index(game_index) if self.ordered_game_indices else game_index
            )
            key = (game_order, turn, ordinal, frame_kind)
            if previous_key is not None and key <= previous_key:
                raise R2MapDatasetError("R2-MAP frames are not in canonical staged-turn order")
            previous_key = key
            refs.append(
                R2MapFrameRef(
                    payload_offset=payload_offset,
                    payload_size=size,
                    game_id=game_id,
                    position_id=position_id,
                    global_game_index=game_index,
                    turn=turn,
                    candidate_count=candidate_count,
                    selected_index=selected_index,
                    frame_kind=int(frame_kind),
                    ordinal=int(ordinal),
                    stage=int(stage),
                    decision_id=decision_id,
                )
            )
            cursor = payload_end
        if cursor != len(self._map):
            raise R2MapDatasetError("R2-MAP stream contains trailing bytes")
        return tuple(refs)

    def _validate_turn_frame_sequences(self) -> None:
        draft_keys = {(ref.global_game_index, ref.turn): ref for ref in self.refs}
        if set(self.market_by_turn) != set(draft_keys):
            raise R2MapDatasetError("every retained draft must have its exact market prelude")
        for key, draft in draft_keys.items():
            refs = self.market_by_turn[key]
            if not refs or tuple(ref.ordinal for ref in refs) != tuple(range(len(refs))):
                raise R2MapDatasetError("market decision ordinals are incomplete or reordered")
            if draft.ordinal != len(refs):
                raise R2MapDatasetError("draft ordinal does not follow every market decision")
            decoded = [self.decode_market(ref) for ref in refs]
            expected_transform = d6_transform_id(
                game_id=draft.game_id,
                draft_decision_id=draft.position_id,
                mode=self.mode,
                epoch=self.epoch,
                sampler_seed=self.sampler_seed,
            )
            if any(frame.transform_id != expected_transform for frame in decoded):
                raise R2MapDatasetError(
                    "market and draft frames disagree with the frozen D6 schedule"
                )
            stages = [frame.ref.stage for frame in decoded]
            if stages[0] == 0:
                if stages[1:] != [1] * (len(stages) - 1):
                    raise R2MapDatasetError("free replacement must precede every paid-wipe stage")
            elif stages != [1] * len(stages):
                raise R2MapDatasetError("market stage sequence is invalid")
            selected_kinds = [
                MarketDecisionActionKind(int(frame.action_bytes[frame.ref.selected_index, 2]))
                for frame in decoded
            ]
            if selected_kinds[-1] is not MarketDecisionActionKind.STOP:
                raise R2MapDatasetError("market prelude must end with explicit Stop")
            if any(kind is MarketDecisionActionKind.STOP for kind in selected_kinds[:-1]):
                raise R2MapDatasetError("market prelude continues after Stop")

    def decode(self, ref: R2MapFrameRef) -> _Frame:
        if ref.frame_kind != DRAFT_FRAME_KIND:
            raise R2MapDatasetError("draft decoder received a market frame")
        payload = memoryview(self._map)[ref.payload_offset : ref.payload_offset + ref.payload_size]
        cursor = 108
        turn = _u16(payload, cursor)
        cursor += 2
        (_seat, _split, transform_id, opponent_bits, market_valid, policy_target) = payload[
            cursor : cursor + 6
        ]
        cursor += 6
        expected_transform = d6_transform_id(
            game_id=ref.game_id,
            draft_decision_id=ref.position_id,
            mode=self.mode,
            epoch=self.epoch,
            sampler_seed=self.sampler_seed,
        )
        expected_policy_target = (
            False
            if self.bootstrap_value_only
            else draft_is_imitation_subset(
                collection_kind=self.manifest["round"]["collection_kind"],
                draft_decision_id=ref.position_id,
            )
        )
        if (
            turn != ref.turn
            or transform_id != expected_transform
            or policy_target not in {0, 1}
            or bool(policy_target) != expected_policy_target
            or opponent_bits & ~0b111
        ):
            raise R2MapDatasetError("R2-MAP frame metadata is invalid")
        candidate_count, selected_index = struct.unpack_from("<II", payload, cursor)
        cursor += 8
        if candidate_count != ref.candidate_count or selected_index != ref.selected_index:
            raise R2MapDatasetError("R2-MAP frame index metadata drifted")
        if not policy_target and (candidate_count != 1 or selected_index != 0):
            raise R2MapDatasetError("selected-only draft frame materialized extra candidates")
        current = _array(payload, cursor, "<u2", COMPONENTS)
        cursor += COMPONENTS * 2
        residual = _array(payload, cursor, "<i2", COMPONENTS)
        cursor += COMPONENTS * 2
        terminal = _array(payload, cursor, "<u2", COMPONENTS)
        cursor += COMPONENTS * 2
        if not np.array_equal(current.astype(np.int32) + residual, terminal):
            raise R2MapDatasetError("R2-MAP component targets violate current+residual=terminal")
        opponent_width = 6 + GRADED_ORACLE_MAX_WILDLIFE_WIPES
        opponents = _array(payload, cursor, "u1", OPPONENTS * opponent_width).reshape(
            OPPONENTS, opponent_width
        )
        cursor += OPPONENTS * opponent_width
        opponent_core = opponents[:, :5]
        opponent_wipe_count = opponents[:, 5]
        opponent_wipe_masks = opponents[:, 6:]
        wipe_ordinals = np.arange(GRADED_ORACLE_MAX_WILDLIFE_WIPES)[None, :]
        opponent_wipe_valid = wipe_ordinals < opponent_wipe_count[:, None]
        opponent_valid = np.asarray(
            [(opponent_bits >> index) & 1 for index in range(3)], dtype=np.bool_
        )
        if (
            np.any(opponent_core[:, 0] > 3)
            or np.any(opponent_core[:, 1] > 3)
            or np.any(opponent_core[:, 2] > 1)
            or np.any(opponent_core[:, 3] > 4)
            or np.any(opponent_core[:, 4] > 1)
            or np.any(opponent_wipe_count > GRADED_ORACLE_MAX_WILDLIFE_WIPES)
            or np.any(opponent_wipe_masks[opponent_wipe_valid] == 0)
            or np.any(opponent_wipe_masks[opponent_wipe_valid] > 15)
            or np.any(opponent_wipe_masks[~opponent_wipe_valid] != 0)
            or np.any(opponent_core[~opponent_valid] != 0)
            or np.any(opponent_wipe_count[~opponent_valid] != 0)
        ):
            raise R2MapDatasetError("R2-MAP ordered opponent paid-wipe target is invalid")
        disposition = _array(payload, cursor, "u1", MARKET_SLOTS).astype(np.int32)
        cursor += MARKET_SLOTS
        pair = _array(payload, cursor, "u1", MARKET_SLOTS).astype(np.int32)
        cursor += MARKET_SLOTS
        final_slot = _array(payload, cursor, "u1", MARKET_SLOTS).astype(np.int32)
        cursor += MARKET_SLOTS
        parent, cursor = _decode_state(payload, cursor)
        candidates: list[_State] = []
        raw_actions = np.empty((candidate_count, GRADED_ORACLE_ACTION_FEATURE_SIZE), dtype=np.uint8)
        scores = np.empty(candidate_count, dtype=np.float32)
        action_hashes: set[bytes] = set()
        ordered_action_hashes: list[bytes] = []
        for candidate in range(candidate_count):
            action_hash = bytes(payload[cursor : cursor + 32])
            cursor += 32
            if len(action_hash) != 32 or action_hash in action_hashes:
                raise R2MapDatasetError("R2-MAP candidate action hashes are invalid or repeated")
            action_hashes.add(action_hash)
            ordered_action_hashes.append(action_hash)
            raw_actions[candidate] = np.frombuffer(
                payload[cursor : cursor + GRADED_ORACLE_ACTION_FEATURE_SIZE], dtype=np.uint8
            )
            cursor += GRADED_ORACLE_ACTION_FEATURE_SIZE
            scores[candidate] = float(_u16(payload, cursor))
            cursor += 2
            state, cursor = _decode_state(payload, cursor)
            candidates.append(state)
        if ordered_action_hashes[selected_index] != bytes(payload[68:100]):
            raise R2MapDatasetError("R2-MAP selected draft action identity differs")
        if cursor != len(payload):
            raise R2MapDatasetError("R2-MAP frame has trailing or missing payload bytes")
        if market_valid and (np.any(disposition > 3) or np.any(pair > 1) or np.any(final_slot > 3)):
            raise R2MapDatasetError("R2-MAP market targets are outside their class bounds")
        return _Frame(
            ref=ref,
            parent=parent,
            candidates=tuple(candidates),
            action_features=decode_graded_action_feature_bytes(raw_actions),
            exact_afterstate_scores=scores,
            current_components=current.astype(np.float32),
            residual_components=residual.astype(np.float32),
            terminal_components=terminal.astype(np.float32),
            opponent_valid=opponent_valid,
            opponent_targets=opponent_core.astype(np.int32),
            opponent_paid_wipe_count=opponent_wipe_count.astype(np.int32),
            opponent_paid_wipe_masks=opponent_wipe_masks.astype(np.int32),
            opponent_paid_wipe_mask_valid=opponent_wipe_valid.astype(np.bool_),
            bootstrap_policy_target=bool(policy_target),
            market_valid=bool(market_valid),
            market_disposition=disposition,
            market_pair_survival=pair,
            market_final_slot=final_slot,
            transform_id=int(transform_id),
        )

    def decode_market(self, ref: R2MapFrameRef) -> _MarketFrame:
        if ref.frame_kind != MARKET_FRAME_KIND:
            raise R2MapDatasetError("market decoder received a draft frame")
        payload = memoryview(self._map)[ref.payload_offset : ref.payload_offset + ref.payload_size]
        if len(payload) < MARKET_FIXED_SIZE:
            raise R2MapDatasetError("R2-MAP market frame fixed prefix is truncated")
        game_id = bytes(payload[4:36])
        position_id = bytes(payload[36:68])
        decision_id = bytes(payload[68:100])
        selected_action_id = bytes(payload[100:132])
        parent_public_hash = bytes(payload[132:164])
        resulting_public_hash = bytes(payload[164:196])
        ordered_digest = bytes(payload[196:228])
        game_index = struct.unpack_from("<Q", payload, 228)[0]
        turn = struct.unpack_from("<H", payload, 236)[0]
        seat, split, transform_id, tokens, bag_total, policy_target = payload[238:244]
        bag_counts = tuple(int(value) for value in payload[244:249])
        market_wildlife = tuple(int(value) for value in payload[249:253])
        reserved = payload[253:256]
        legal_count, selected_index = struct.unpack_from("<II", payload, 256)
        exact_current, final_score, score_to_go, score_reserved = struct.unpack_from(
            "<HHhH", payload, 264
        )
        expected_policy_target = self.manifest["round"]["collection_kind"] == "bootstrap"
        if (
            game_id != ref.game_id
            or position_id != ref.position_id
            or decision_id != ref.decision_id
            or game_index != ref.global_game_index
            or turn != ref.turn
            or legal_count != ref.candidate_count
            or selected_index != ref.selected_index
            or seat >= BOARD_SLOTS
            or split not in {0, 1}
            or transform_id >= 12
            or policy_target not in {0, 1}
            or bool(policy_target) != expected_policy_target
            or bytes(reserved) != b"\0\0\0"
            or sum(bag_counts) != bag_total
            or score_reserved != 0
            or int(final_score) - int(exact_current) != int(score_to_go)
        ):
            raise R2MapDatasetError("R2-MAP market frame fixed metadata or score algebra differs")
        parent, cursor = _decode_state(payload, MARKET_FIXED_SIZE)
        action_ids: list[bytes] = []
        action_bytes = np.empty((legal_count, MARKET_DECISION_ACTION_SIZE), dtype=np.uint8)
        for index in range(legal_count):
            if cursor + 32 + MARKET_DECISION_ACTION_SIZE > len(payload):
                raise R2MapDatasetError("R2-MAP market legal row is truncated")
            action_id = bytes(payload[cursor : cursor + 32])
            cursor += 32
            row = bytes(payload[cursor : cursor + MARKET_DECISION_ACTION_SIZE])
            cursor += MARKET_DECISION_ACTION_SIZE
            expected_id = market_decision_action_id(decision_id.hex(), row)
            if action_id.hex() != expected_id or action_id in action_ids:
                raise R2MapDatasetError("R2-MAP market action identity is invalid or repeated")
            action_ids.append(action_id)
            action_bytes[index] = np.frombuffer(row, dtype=np.uint8)
        if cursor != len(payload):
            raise R2MapDatasetError("R2-MAP market frame has trailing bytes")
        decision_kind = MarketDecisionKind(ref.stage)
        validate_canonical_market_action_order(
            action_bytes,
            decision_kind=decision_kind,
            public_nature_tokens=int(tokens),
            public_wildlife_bag_total=int(bag_total),
            public_wildlife_bag_counts=bag_counts,
            public_market_wildlife=market_wildlife,
        )
        ordered = blake3.blake3(
            json.dumps([value.hex() for value in action_ids], separators=(",", ":")).encode()
        ).digest()
        if ordered != ordered_digest or action_ids[selected_index] != selected_action_id:
            raise R2MapDatasetError("R2-MAP market selected or ordered action identity differs")
        return _MarketFrame(
            ref=ref,
            parent=parent,
            action_bytes=action_bytes,
            action_features=decode_market_decision_action_bytes(action_bytes),
            action_ids=tuple(action_ids),
            exact_current_score=float(exact_current),
            final_score=float(final_score),
            score_to_go=float(score_to_go),
            selected_action_id=selected_action_id,
            parent_public_hash=parent_public_hash,
            resulting_public_hash=resulting_public_hash,
            public_nature_tokens=int(tokens),
            public_wildlife_bag_total=int(bag_total),
            public_wildlife_bag_counts=bag_counts,
            public_market_wildlife=market_wildlife,
            policy_target=bool(policy_target),
            transform_id=int(transform_id),
        )

    def batch(self, indices: list[int] | tuple[int, ...]) -> R2MapSupervisedBatch:
        result, _, _ = self._batch_with_identity_components(indices)
        return result

    def _batch_with_identity_components(
        self, indices: list[int] | tuple[int, ...]
    ) -> tuple[R2MapSupervisedBatch, tuple[str, ...], tuple[str, ...]]:
        """Build a batch and retain the ordered identity preimages for stitching.

        Batch identities hash an ordered list of replay identities.  A digest of a
        partial batch cannot be composed into the digest of the original full
        batch, so bounded whole-game windows retain the preimages only while a
        boundary batch is being assembled.  They are not model inputs.
        """
        if not indices or any(index < 0 or index >= len(self.refs) for index in indices):
            raise IndexError("R2-MAP batch contains an invalid frame index")
        frames = [self.decode(self.refs[index]) for index in indices]
        groups = len(frames)
        width = max(frame.ref.candidate_count for frame in frames)
        parents = _stack_states([frame.parent for frame in frames])
        candidate_arrays = _empty_state_arrays(groups, width)
        candidate_mask = np.zeros((groups, width), dtype=np.bool_)
        action_features = np.zeros((groups, width, 140), dtype=np.float32)
        exact_scores = np.zeros((groups, width), dtype=np.float32)
        score_targets = np.zeros((groups, width), dtype=np.float32)
        component_targets = np.zeros((groups, width, COMPONENTS), dtype=np.float32)
        score_mask = np.zeros((groups, width), dtype=np.bool_)
        selected = np.empty(groups, dtype=np.int32)
        opponent_targets = np.zeros((groups, OPPONENTS, 5), dtype=np.int32)
        opponent_wipe_count = np.zeros((groups, OPPONENTS), dtype=np.int32)
        opponent_wipe_masks = np.zeros(
            (groups, OPPONENTS, GRADED_ORACLE_MAX_WILDLIFE_WIPES), dtype=np.int32
        )
        opponent_wipe_valid = np.zeros_like(opponent_wipe_masks, dtype=np.bool_)
        opponent_mask = np.zeros((groups, OPPONENTS), dtype=np.bool_)
        disposition = np.zeros((groups, MARKET_SLOTS), dtype=np.int32)
        pair = np.zeros((groups, MARKET_SLOTS), dtype=np.int32)
        final_slot = np.zeros((groups, MARKET_SLOTS), dtype=np.int32)
        market_mask = np.zeros((groups, MARKET_SLOTS), dtype=np.bool_)
        final_mask = np.zeros((groups, MARKET_SLOTS), dtype=np.bool_)
        identities = []
        for row, frame in enumerate(frames):
            count = frame.ref.candidate_count
            _assign_states(candidate_arrays, row, frame.candidates)
            candidate_mask[row, :count] = True
            action_features[row, :count] = frame.action_features
            exact_scores[row, :count] = frame.exact_afterstate_scores
            selected[row] = frame.ref.selected_index
            score_targets[row, selected[row]] = float(frame.residual_components.sum())
            component_targets[row, selected[row]] = frame.residual_components
            score_mask[row, selected[row]] = True
            opponent_targets[row] = frame.opponent_targets
            opponent_wipe_count[row] = frame.opponent_paid_wipe_count
            opponent_wipe_masks[row] = frame.opponent_paid_wipe_masks
            opponent_wipe_valid[row] = frame.opponent_paid_wipe_mask_valid
            opponent_mask[row] = frame.opponent_valid
            disposition[row] = frame.market_disposition
            pair[row] = frame.market_pair_survival
            final_slot[row] = frame.market_final_slot
            market_mask[row] = frame.market_valid
            final_mask[row] = frame.market_valid & (frame.market_disposition == 3)
            identities.append(
                f"{frame.ref.position_id.hex()}:{frame.transform_id}:{frame.ref.candidate_count}"
            )
        batch_id = blake3.blake3("|".join(identities).encode()).hexdigest()
        market_supervision, market_identities = self._market_supervision_with_identities(frames)
        result = R2MapSupervisedBatch(
            inputs=R2MapBatch(
                parent=_to_public_state(parents),
                candidates=_to_public_state(candidate_arrays),
                candidate_mask=mx.array(candidate_mask),
                action_features=mx.array(action_features),
                exact_afterstate_scores=mx.array(exact_scores),
            ),
            score_to_go_targets=mx.array(score_targets),
            score_component_targets=mx.array(component_targets),
            score_target_mask=mx.array(score_mask),
            selected_action_index=mx.array(selected),
            bootstrap_policy_mask=mx.array(
                [
                    frame.bootstrap_policy_target and frame.ref.candidate_count > 1
                    for frame in frames
                ]
            ),
            opponent_tile_slot_targets=mx.array(opponent_targets[..., 0]),
            opponent_wildlife_slot_targets=mx.array(opponent_targets[..., 1]),
            opponent_draft_kind_targets=mx.array(opponent_targets[..., 2]),
            opponent_drafted_wildlife_targets=mx.array(opponent_targets[..., 3]),
            opponent_replace_three_targets=mx.array(opponent_targets[..., 4]),
            opponent_paid_wipe_count_targets=mx.array(opponent_wipe_count),
            opponent_paid_wipe_mask_targets=mx.array(opponent_wipe_masks),
            opponent_paid_wipe_mask_valid=mx.array(opponent_wipe_valid),
            opponent_valid_mask=mx.array(opponent_mask),
            market_disposition_targets=mx.array(disposition),
            market_pair_survival_targets=mx.array(pair),
            market_final_slot_targets=mx.array(final_slot),
            market_disposition_mask=mx.array(market_mask),
            market_pair_survival_mask=mx.array(market_mask),
            market_final_slot_mask=mx.array(final_mask),
            batch_identity=batch_id,
            market_decisions=market_supervision,
        )
        result.validate()
        return result, tuple(identities), market_identities

    def fixed_selected_batch(self, indices: Sequence[int]) -> R2MapBatch:
        """Build a bounded all-frame panel using each replay-selected action."""
        if not indices or any(index < 0 or index >= len(self.refs) for index in indices):
            raise IndexError("R2-MAP fixed panel contains an invalid frame index")
        parents: list[_State] = []
        selected_states: list[_State] = []
        action_features = np.empty((len(indices), 1, 140), dtype=np.float32)
        exact_scores = np.empty((len(indices), 1), dtype=np.float32)
        for row, index in enumerate(indices):
            frame = self.decode(self.refs[index])
            selected = frame.ref.selected_index
            parents.append(frame.parent)
            selected_states.append(frame.candidates[selected])
            action_features[row, 0] = frame.action_features[selected]
            exact_scores[row, 0] = frame.exact_afterstate_scores[selected]
        candidate_arrays = _empty_state_arrays(len(indices), 1)
        for row, state in enumerate(selected_states):
            _assign_states(candidate_arrays, row, (state,))
        result = R2MapBatch(
            parent=_to_public_state(_stack_states(parents)),
            candidates=_to_public_state(candidate_arrays),
            candidate_mask=mx.ones((len(indices), 1), dtype=mx.bool_),
            action_features=mx.array(action_features),
            exact_afterstate_scores=mx.array(exact_scores),
        )
        result.validate()
        return result

    def _market_supervision(self, draft_frames: Sequence[_Frame]) -> R2MapMarketDecisionSupervision:
        result, _ = self._market_supervision_with_identities(draft_frames)
        return result

    def _market_supervision_with_identities(
        self, draft_frames: Sequence[_Frame]
    ) -> tuple[R2MapMarketDecisionSupervision, tuple[str, ...]]:
        frames = [
            self.decode_market(ref)
            for draft in draft_frames
            for ref in self.market_by_turn[(draft.ref.global_game_index, draft.ref.turn)]
        ]
        if not frames:
            raise R2MapDatasetError("R2-MAP batch omitted every market decision")
        groups = len(frames)
        width = max(frame.ref.candidate_count for frame in frames)
        parents = _stack_states([frame.parent for frame in frames])
        action_mask = np.zeros((groups, width), dtype=np.bool_)
        action_features = np.zeros(
            (groups, width, frames[0].action_features.shape[-1]), dtype=np.float32
        )
        exact_current = np.empty(groups, dtype=np.float32)
        targets = np.zeros((groups, width), dtype=np.float32)
        target_mask = np.zeros((groups, width), dtype=np.bool_)
        selected = np.empty(groups, dtype=np.int32)
        policy_target = np.zeros(groups, dtype=np.bool_)
        identities = []
        for row, frame in enumerate(frames):
            count = frame.ref.candidate_count
            action_mask[row, :count] = True
            action_features[row, :count] = frame.action_features
            exact_current[row] = frame.exact_current_score
            selected[row] = frame.ref.selected_index
            targets[row, selected[row]] = frame.score_to_go
            target_mask[row, selected[row]] = True
            policy_target[row] = frame.policy_target and count > 1
            identities.append(
                f"{frame.ref.decision_id.hex()}:{frame.ref.ordinal}:{count}:"
                f"{frame.selected_action_id.hex()}"
            )
        result = R2MapMarketDecisionSupervision(
            inputs=R2MapMarketDecisionBatch(
                public_state=_to_public_state(parents),
                action_mask=mx.array(action_mask),
                action_features=mx.array(action_features),
                exact_current_scores=mx.array(exact_current),
            ),
            score_to_go_targets=mx.array(targets),
            score_target_mask=mx.array(target_mask),
            selected_action_index=mx.array(selected),
            policy_target_mask=mx.array(policy_target),
            batch_identity=blake3.blake3("|".join(identities).encode()).hexdigest(),
        )
        result.validate()
        return result, tuple(identities)


class R2MapDatasetAdapter(R2MapTrainingAdapter):
    """Deterministic candidate-budget packing over train/validation/panel streams."""

    protocol_id = PROTOCOL_ID

    def __init__(
        self,
        train: R2MapStreamReader,
        validation: R2MapStreamReader,
        panel: R2MapStreamReader,
        *,
        group_batch_size: int = 2,
        maximum_candidates_per_batch: int = 16_384,
    ):
        if train.mode != 0 or validation.mode != 1 or panel.mode != 2:
            raise R2MapDatasetError("R2-MAP adapter stream modes are inconsistent")
        if len({train.dataset_blake3, validation.dataset_blake3, panel.dataset_blake3}) != 1:
            raise R2MapDatasetError("R2-MAP adapter streams bind different datasets")
        if group_batch_size <= 0 or maximum_candidates_per_batch <= 0:
            raise ValueError("R2-MAP batch limits must be positive")
        self.train = train
        self.validation = validation
        self.panel = panel
        self.dataset_blake3 = train.dataset_blake3
        self.dataset_contract = _training_dataset_contract(train.manifest)
        self.group_batch_size = group_batch_size
        self.maximum_candidates_per_batch = maximum_candidates_per_batch

    @classmethod
    def open(
        cls,
        *,
        train_manifest: str | Path,
        train_stream: str | Path,
        validation_manifest: str | Path,
        validation_stream: str | Path,
        panel_manifest: str | Path,
        panel_stream: str | Path,
        group_batch_size: int = 2,
        maximum_candidates_per_batch: int = 16_384,
    ) -> R2MapDatasetAdapter:
        readers: list[R2MapStreamReader] = []
        try:
            readers.append(R2MapStreamReader(train_manifest, train_stream))
            readers.append(R2MapStreamReader(validation_manifest, validation_stream))
            readers.append(R2MapStreamReader(panel_manifest, panel_stream))
            return cls(
                *readers,
                group_batch_size=group_batch_size,
                maximum_candidates_per_batch=maximum_candidates_per_batch,
            )
        except Exception:
            for reader in reversed(readers):
                reader.close()
            raise

    def close(self) -> None:
        for reader in (self.panel, self.validation, self.train):
            reader.close()

    def __enter__(self) -> R2MapDatasetAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"epoch": 0, "offset": 0}, {"seed": int(seed)}

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        epoch = int(cursor["epoch"])
        offset = int(cursor["offset"])
        seed = int(sampler_state["seed"])
        order = self._order(epoch, seed)
        if offset >= len(order):
            epoch += 1
            offset = 0
            order = self._order(epoch, seed)
        selected = _pack_one(
            self.train.refs,
            order,
            offset,
            self.group_batch_size,
            self.maximum_candidates_per_batch,
        )
        next_offset = offset + len(selected)
        if next_offset == len(order):
            epoch += 1
            next_offset = 0
        return R2MapAdapterStep(
            batch=self.train.batch(selected),
            next_cursor={"epoch": epoch, "offset": next_offset},
            next_sampler_state={"seed": seed},
        )

    def validation_batches(self) -> tuple[R2MapSupervisedBatch, ...]:
        order = tuple(range(len(self.validation.refs)))
        return tuple(
            self.validation.batch(list(group))
            for group in _pack_all(
                self.validation.refs,
                order,
                self.group_batch_size,
                self.maximum_candidates_per_batch,
            )
        )

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch:
        if not panel_id or not self.panel.refs:
            raise R2MapDatasetError("R2-MAP fixed panel is absent or unnamed")
        return self.panel.fixed_selected_batch(list(range(len(self.panel.refs))))

    def _order(self, epoch: int, seed: int) -> tuple[int, ...]:
        ranked = []
        for index, ref in enumerate(self.train.refs):
            digest = blake3.blake3(
                b"r2-map-sampler-order-v1"
                + seed.to_bytes(8, "little", signed=False)
                + epoch.to_bytes(8, "little", signed=False)
                + ref.position_id
            ).digest()
            ranked.append((digest, index))
        ranked.sort()
        return tuple(index for _, index in ranked)


@dataclass(frozen=True, slots=True)
class CompactStorageProjection:
    target_games: int
    measured_games: int
    measured_compact_bytes: int
    projected_compact_bytes: int
    projected_index_bytes: int
    maximum_window_bytes: int
    maximum_prefetch_windows: int
    projected_peak_additional_bytes: int
    expanded_bytes_per_game: int
    projected_expanded_bytes: int
    run_budget_bytes: int
    compact_fits_run_budget: bool
    expanded_fits_run_budget: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def build_compact_index(
    shards: Sequence[str | Path],
    *,
    exporter: str | Path,
    output: str | Path,
    scratch: str | Path,
    maximum_window_bytes: int = 1 << 30,
) -> dict[str, Any]:
    """Build a small replay index via bounded, disposable per-shard windows."""
    if not shards:
        raise R2MapDatasetError("compact R2-MAP index requires source shards")
    if maximum_window_bytes <= 0:
        raise ValueError("compact index window limit must be positive")
    exporter = Path(exporter).resolve(strict=True)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    source_manifests: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    try:
        for ordinal, raw_shard in enumerate(shards):
            shard = Path(raw_shard).resolve(strict=True)
            if shard.name in seen_names:
                raise R2MapDatasetError("compact R2-MAP shard names must be unique")
            seen_names.add(shard.name)
            observed: dict[tuple[int, str], dict[str, Any]] = {}
            local_manifest: dict[str, Any] | None = None
            for mode in ("train", "validation"):
                manifest_path = scratch / f"index-{ordinal:05d}-{mode}.json"
                stream_path = scratch / f"index-{ordinal:05d}-{mode}.r2map"
                _run_exporter(
                    exporter,
                    shard=shard,
                    manifest=manifest_path,
                    stream=stream_path,
                    mode=mode,
                    epoch=0,
                    sampler_seed=0,
                    compact_index=None,
                    semantic_validation_binding=None,
                    game_indices=(),
                )
                if stream_path.stat().st_size > maximum_window_bytes:
                    raise R2MapDatasetError("compact index source window exceeds storage gate")
                manifest = _read_manifest(manifest_path)
                if local_manifest is None:
                    local_manifest = manifest
                elif manifest != local_manifest:
                    raise R2MapDatasetError("per-mode compact source manifests differ")
                with R2MapStreamReader(manifest_path, stream_path) as reader:
                    grouped: dict[tuple[int, str], int] = {}
                    grouped_imitation: dict[tuple[int, str], int] = {}
                    for ref in reader.refs:
                        key = (ref.global_game_index, ref.game_id.hex())
                        grouped[key] = grouped.get(key, 0) + 1
                        grouped_imitation[key] = grouped_imitation.get(key, 0) + int(
                            draft_is_imitation_subset(
                                collection_kind=manifest["round"]["collection_kind"],
                                draft_decision_id=ref.position_id,
                            )
                        )
                    grouped_market: dict[tuple[int, str], int] = {}
                    for ref in reader.market_refs:
                        key = (ref.global_game_index, ref.game_id.hex())
                        grouped_market[key] = grouped_market.get(key, 0) + 1
                    if set(grouped_market) != set(grouped):
                        raise R2MapDatasetError("compact index market and draft game sets differ")
                    for (game_index, game_id), example_count in grouped.items():
                        key = (game_index, game_id)
                        if key in observed:
                            raise R2MapDatasetError("compact index repeats a game across splits")
                        observed[key] = {
                            "source_file_name": shard.name,
                            "source_blake3": manifest["sources"][0]["blake3"],
                            "global_game_index": game_index,
                            "game_id": game_id,
                            "example_count": example_count,
                            "imitation_example_count": grouped_imitation[key],
                            "market_decision_count": grouped_market[(game_index, game_id)],
                            "market_policy_target_count": (
                                grouped_market[(game_index, game_id)]
                                if manifest["round"]["collection_kind"] == "bootstrap"
                                else 0
                            ),
                            "split": mode,
                        }
                manifest_path.unlink(missing_ok=True)
                stream_path.unlink(missing_ok=True)
            assert local_manifest is not None
            if len(local_manifest["sources"]) != 1:
                raise R2MapDatasetError("compact indexing must inspect one shard at a time")
            if len(observed) != local_manifest["game_count"]:
                raise R2MapDatasetError("compact index game accounting differs from manifest")
            if (
                sum(game["example_count"] for game in observed.values())
                != local_manifest["example_count"]
                or sum(game["imitation_example_count"] for game in observed.values())
                != local_manifest["imitation_example_count"]
                or sum(game["market_policy_target_count"] for game in observed.values())
                != local_manifest["market_policy_target_count"]
            ):
                raise R2MapDatasetError("compact index example accounting differs from manifest")
            source_manifests.append(local_manifest)
            games.extend(observed.values())
    finally:
        for path in scratch.glob("index-*.r2map"):
            path.unlink(missing_ok=True)
        for path in scratch.glob("index-*.json"):
            path.unlink(missing_ok=True)
    manifest = _aggregate_source_manifests(source_manifests)
    games.sort(key=lambda game: (game["global_game_index"], game["game_id"]))
    if [game["global_game_index"] for game in games] != sorted(
        {game["global_game_index"] for game in games}
    ):
        raise R2MapDatasetError("compact index global game indices repeat")
    value: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": COMPACT_INDEX_SCHEMA,
        "dataset_manifest": manifest,
        "games": games,
    }
    value["index_blake3"] = _canonical_blake3(value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    # The Rust v1 manifest digest binds serde field order, so preserve the
    # nested manifest/source insertion order on disk. The outer index digest
    # itself is independently canonicalized with sorted keys.
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    return validate_compact_index(output)


def validate_compact_index(
    path: str | Path,
    *,
    shard_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R2MapDatasetError("cannot read compact R2-MAP index") from error
    return validate_compact_index_value(value, shard_root=shard_root)


def validate_compact_index_value(
    value: Any,
    *,
    shard_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an index streamed from authoritative storage without local I/O."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "protocol_id",
        "dataset_manifest",
        "games",
        "index_blake3",
    }:
        raise R2MapDatasetError("compact R2-MAP index schema differs")
    if value["schema_version"] != 1 or value["protocol_id"] != COMPACT_INDEX_SCHEMA:
        raise R2MapDatasetError("compact R2-MAP index protocol differs")
    if value["index_blake3"] != _canonical_blake3(
        {key: item for key, item in value.items() if key != "index_blake3"}
    ):
        raise R2MapDatasetError("compact R2-MAP index identity failed")
    manifest = _validate_manifest_value(value["dataset_manifest"])
    games = value["games"]
    required = {
        "source_file_name",
        "source_blake3",
        "global_game_index",
        "game_id",
        "example_count",
        "imitation_example_count",
        "market_decision_count",
        "market_policy_target_count",
        "split",
    }
    width_required = required | {"candidate_widths"}
    sources = {source["file_name"]: source for source in manifest["sources"]}
    if not isinstance(games, list) or len(games) != manifest["game_count"]:
        raise R2MapDatasetError("compact R2-MAP game index count differs")
    seen: set[tuple[int, str]] = set()
    source_counts = {name: [0, 0, 0, 0, 0] for name in sources}
    for game in games:
        if not isinstance(game, dict) or set(game) not in (required, width_required):
            raise R2MapDatasetError("compact R2-MAP game index schema differs")
        source = sources.get(game["source_file_name"])
        key = (game["global_game_index"], game["game_id"])
        widths = game.get("candidate_widths")
        if widths is None:
            if game.get("imitation_example_count") != 0:
                raise R2MapDatasetError("compact R2-MAP imitation game omits candidate widths")
            widths = [1] * game.get("example_count", 0)
        if (
            source is None
            or game["source_blake3"] != source["blake3"]
            or not isinstance(game["global_game_index"], int)
            or game["global_game_index"] < 0
            or not _digest(game["game_id"])
            or not isinstance(game["example_count"], int)
            or game["example_count"] <= 0
            or not isinstance(game["imitation_example_count"], int)
            or isinstance(game["imitation_example_count"], bool)
            or not 0 <= game["imitation_example_count"] <= game["example_count"]
            or not isinstance(game["market_decision_count"], int)
            or game["market_decision_count"] <= 0
            or not isinstance(game["market_policy_target_count"], int)
            or isinstance(game["market_policy_target_count"], bool)
            or not 0 <= game["market_policy_target_count"] <= game["market_decision_count"]
            or game["split"] not in {"train", "validation"}
            or not isinstance(widths, list)
            or len(widths) != game["example_count"]
            or any(
                not isinstance(width, int) or isinstance(width, bool) or not 1 <= width <= 16_384
                for width in widths
            )
            or sum(width > 1 for width in widths) != game["imitation_example_count"]
            or key in seen
        ):
            raise R2MapDatasetError("compact R2-MAP game index identity is invalid")
        seen.add(key)
        source_counts[game["source_file_name"]][0] += 1
        source_counts[game["source_file_name"]][1] += game["example_count"]
        source_counts[game["source_file_name"]][2] += game["imitation_example_count"]
        source_counts[game["source_file_name"]][3] += game["market_decision_count"]
        source_counts[game["source_file_name"]][4] += game["market_policy_target_count"]
    for name, source in sources.items():
        if source_counts[name] != [
            source["game_count"],
            source["example_count"],
            source["imitation_example_count"],
            source["market_decision_count"],
            source["market_policy_target_count"],
        ]:
            raise R2MapDatasetError("compact R2-MAP source index accounting differs")
    if sum(game["split"] == "train" for game in games) != manifest["train_games"]:
        raise R2MapDatasetError("compact R2-MAP train split accounting differs")
    if sum(game["split"] == "validation" for game in games) != manifest["validation_games"]:
        raise R2MapDatasetError("compact R2-MAP validation split accounting differs")
    if shard_root is not None:
        root = Path(shard_root).resolve(strict=True)
        for source in manifest["sources"]:
            shard = (root / source["file_name"]).resolve(strict=True)
            try:
                shard.relative_to(root)
            except ValueError as error:
                raise R2MapDatasetError("compact source escapes shard root") from error
            if shard.stat().st_size != source["bytes"] or _file_blake3(shard) != source["blake3"]:
                raise R2MapDatasetError("compact source bytes or hash differ from index")
    return value


def compact_storage_projection(
    index: dict[str, Any] | str | Path,
    *,
    target_games: int = 100_000,
    maximum_window_bytes: int = 1 << 30,
    maximum_prefetch_windows: int = 1,
    expanded_bytes_per_game: int = 2_000_000,
    run_budget_bytes: int = 40 * (1 << 30),
) -> CompactStorageProjection:
    value = (
        validate_compact_index_value(index)
        if isinstance(index, dict)
        else validate_compact_index(index)
    )
    integer_limits = (
        target_games,
        maximum_window_bytes,
        maximum_prefetch_windows,
        expanded_bytes_per_game,
        run_budget_bytes,
    )
    if (
        any(not isinstance(value, int) or isinstance(value, bool) for value in integer_limits)
        or target_games <= 0
        or maximum_window_bytes <= 0
        or maximum_prefetch_windows not in (0, 1)
        or expanded_bytes_per_game <= 0
        or run_budget_bytes <= 0
    ):
        raise ValueError("compact storage projection limits are invalid")
    manifest = value["dataset_manifest"]
    measured_games = manifest["game_count"]
    compact_bytes = sum(source["bytes"] for source in manifest["sources"])
    projected_compact = math_ceil_div(compact_bytes * target_games, measured_games)
    encoded_index_bytes = len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    )
    projected_index = math_ceil_div(encoded_index_bytes * target_games, measured_games)
    window_bytes = maximum_window_bytes * (1 + maximum_prefetch_windows)
    projected_peak = projected_compact + projected_index + window_bytes
    projected_expanded = target_games * expanded_bytes_per_game
    return CompactStorageProjection(
        target_games=target_games,
        measured_games=measured_games,
        measured_compact_bytes=compact_bytes,
        projected_compact_bytes=projected_compact,
        projected_index_bytes=projected_index,
        maximum_window_bytes=maximum_window_bytes,
        maximum_prefetch_windows=maximum_prefetch_windows,
        projected_peak_additional_bytes=projected_peak,
        expanded_bytes_per_game=expanded_bytes_per_game,
        projected_expanded_bytes=projected_expanded,
        run_budget_bytes=run_budget_bytes,
        compact_fits_run_budget=projected_peak <= run_budget_bytes,
        expanded_fits_run_budget=projected_expanded <= run_budget_bytes,
    )


def compact_packing_plan(
    index: dict[str, Any],
    candidate_widths: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    group_batch_size: int,
    maximum_candidates_per_batch: int,
    seed: int,
    epochs: int = 12,
) -> dict[str, Any]:
    """Plan exact compact-adapter batches from verified per-turn widths.

    Widths are read from disposable replay-authoritative streams; the compact
    index supplies immutable source/game order. The planner mirrors the live
    source-boundary, whole-game, and rectangular-padding rules without
    materializing any R2 tensors.
    """
    value = validate_compact_index_value(index)
    limits = (group_batch_size, maximum_candidates_per_batch, seed, epochs)
    if (
        any(not isinstance(item, int) or isinstance(item, bool) for item in limits)
        or group_batch_size <= 0
        or maximum_candidates_per_batch <= 0
        or seed < 0
        or epochs <= 0
    ):
        raise ValueError("compact packing-plan limits are invalid")
    train_games = [game for game in value["games"] if game["split"] == "train"]
    if not train_games:
        raise R2MapDatasetError("compact packing plan requires training games")
    games_by_source: dict[str, list[dict[str, Any]]] = {}
    for game in train_games:
        games_by_source.setdefault(game["source_file_name"], []).append(game)
    if set(candidate_widths) != set(games_by_source):
        raise R2MapDatasetError("compact packing width source set differs from the index")

    normalized: dict[str, dict[str, tuple[int, ...]]] = {}
    for source, games in games_by_source.items():
        supplied = candidate_widths[source]
        if set(supplied) != {game["game_id"] for game in games}:
            raise R2MapDatasetError("compact packing width game set differs from the index")
        normalized[source] = {}
        for game in games:
            widths = tuple(supplied[game["game_id"]])
            if (
                len(widths) != game["example_count"]
                or any(
                    not isinstance(width, int)
                    or isinstance(width, bool)
                    or width <= 0
                    or width > maximum_candidates_per_batch
                    for width in widths
                )
                or sum(width > 1 for width in widths) != game["imitation_example_count"]
            ):
                raise R2MapDatasetError(
                    "compact packing widths differ from indexed example accounting"
                )
            normalized[source][game["game_id"]] = widths

    epoch_plans: list[dict[str, int]] = []
    for epoch in range(epochs):
        sources = sorted(
            games_by_source,
            key=lambda source: _sampler_hash(seed, epoch, source),
        )
        epoch_stats = _empty_packing_statistics(epoch)
        for source in sources:
            games = sorted(
                games_by_source[source],
                key=lambda game: _sampler_hash(seed, epoch, game["game_id"]),
            )
            widths = tuple(width for game in games for width in normalized[source][game["game_id"]])
            _accumulate_packing_statistics(
                epoch_stats,
                widths,
                group_batch_size=group_batch_size,
                maximum_candidates_per_batch=maximum_candidates_per_batch,
            )
        epoch_plans.append(epoch_stats)

    summed_fields = (
        "steps",
        "draft_groups",
        "selected_only_groups",
        "draft_policy_targets",
        "draft_candidates",
        "padded_draft_candidates",
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": "r2-map-compact-packing-plan-v1",
        "dataset_blake3": value["dataset_manifest"]["dataset_blake3"],
        "seed": seed,
        "epochs": epochs,
        "group_batch_size": group_batch_size,
        "maximum_candidates_per_batch": maximum_candidates_per_batch,
        "epoch_plans": epoch_plans,
        "totals": {name: sum(plan[name] for plan in epoch_plans) for name in summed_fields},
        "maximum_candidate_width": max(
            width
            for source in normalized.values()
            for widths in source.values()
            for width in widths
        ),
        "maximum_batch_groups": max(plan["maximum_batch_groups"] for plan in epoch_plans),
        "minimum_batch_groups": min(plan["minimum_batch_groups"] for plan in epoch_plans),
    }
    result["plan_blake3"] = _canonical_blake3(result)
    return result


def _empty_packing_statistics(epoch: int) -> dict[str, int]:
    return {
        "epoch": epoch,
        "steps": 0,
        "draft_groups": 0,
        "selected_only_groups": 0,
        "draft_policy_targets": 0,
        "draft_candidates": 0,
        "padded_draft_candidates": 0,
        "maximum_batch_groups": 0,
        "minimum_batch_groups": 0,
    }


def _accumulate_packing_statistics(
    statistics: dict[str, int],
    widths: Sequence[int],
    *,
    group_batch_size: int,
    maximum_candidates_per_batch: int,
) -> None:
    offset = 0
    while offset < len(widths):
        batch_width = 0
        batch_groups = 0
        batch_candidates = 0
        batch_policy_targets = 0
        while offset + batch_groups < len(widths) and batch_groups < group_batch_size:
            candidate_width = widths[offset + batch_groups]
            next_width = max(batch_width, candidate_width)
            if next_width * (batch_groups + 1) > maximum_candidates_per_batch:
                if batch_groups == 0:
                    # One exact legal screen cannot be split by the packing
                    # contract.  Isolate it; the cap continues to bound every
                    # avoidable multi-group padded tensor.
                    batch_width = candidate_width
                    batch_groups = 1
                    batch_candidates = candidate_width
                    batch_policy_targets = int(candidate_width > 1)
                break
            batch_width = next_width
            batch_groups += 1
            batch_candidates += candidate_width
            batch_policy_targets += int(candidate_width > 1)
        statistics["steps"] += 1
        statistics["draft_groups"] += batch_groups
        statistics["selected_only_groups"] += batch_groups - batch_policy_targets
        statistics["draft_policy_targets"] += batch_policy_targets
        statistics["draft_candidates"] += batch_candidates
        statistics["padded_draft_candidates"] += batch_width * batch_groups
        statistics["maximum_batch_groups"] = max(statistics["maximum_batch_groups"], batch_groups)
        prior_minimum = statistics["minimum_batch_groups"]
        statistics["minimum_batch_groups"] = (
            batch_groups if prior_minimum == 0 else min(prior_minimum, batch_groups)
        )
        offset += batch_groups


@dataclass(frozen=True)
class CompactWindowChunk:
    source: str
    mode: str
    epoch: int
    sampler_seed: int
    chunk_index: int
    first_game_offset: int
    next_game_offset: int
    game_indices: tuple[int, ...]
    nominal_bytes: int
    chunk_blake3: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "mode": self.mode,
            "epoch": self.epoch,
            "sampler_seed": self.sampler_seed,
            "chunk_index": self.chunk_index,
            "first_game_offset": self.first_game_offset,
            "next_game_offset": self.next_game_offset,
            "game_indices": list(self.game_indices),
            "nominal_bytes": self.nominal_bytes,
            "chunk_blake3": self.chunk_blake3,
        }


def _bounded_whole_game_window_chunks(
    source: str,
    mode: str,
    epoch: int,
    sampler_seed: int,
    games: Sequence[dict[str, Any]],
    *,
    group_batch_size: int,
    maximum_candidates_per_batch: int,
) -> tuple[CompactWindowChunk, ...]:
    if not games:
        return ()
    for game in games:
        widths = game.get("candidate_widths")
        if widths is None and game.get("imitation_example_count") == 0:
            widths = [1] * game["example_count"]
        if not isinstance(widths, list) or len(widths) != game["example_count"]:
            raise R2MapDatasetError("bounded window planning requires exact candidate widths")
        for width in widths:
            if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
                raise R2MapDatasetError("bounded window candidate width is invalid")
            if width > maximum_candidates_per_batch:
                raise R2MapDatasetError("one bounded-window group exceeds the candidate budget")
    ranges: list[tuple[int, int, int]] = []
    first = 0
    nominal_bytes = 0
    for game_offset, game in enumerate(games):
        widths = game.get("candidate_widths")
        if widths is None:
            widths = [1] * game["example_count"]
        game_bytes = (
            int(game["example_count"]) * WINDOW_NOMINAL_BYTES_PER_DRAFT_GROUP
            + sum(widths) * WINDOW_NOMINAL_BYTES_PER_CANDIDATE
        )
        if game_bytes > WINDOW_NOMINAL_TARGET_BYTES:
            raise R2MapDatasetError("one whole game exceeds the bounded-window nominal byte target")
        games_in_chunk = game_offset - first
        if games_in_chunk and (
            games_in_chunk >= WINDOW_TARGET_GAMES
            or nominal_bytes + game_bytes > WINDOW_PREFERRED_TARGET_BYTES
        ):
            ranges.append((first, game_offset, nominal_bytes))
            first = game_offset
            nominal_bytes = 0
        nominal_bytes += game_bytes
    ranges.append((first, len(games), nominal_bytes))
    chunks: list[CompactWindowChunk] = []
    for chunk_index, (first, next_offset, nominal_bytes) in enumerate(ranges):
        game_indices = tuple(int(game["global_game_index"]) for game in games[first:next_offset])
        descriptor = {
            "source": source,
            "mode": mode,
            "epoch": epoch,
            "sampler_seed": sampler_seed,
            "chunk_index": chunk_index,
            "first_game_offset": first,
            "next_game_offset": next_offset,
            "game_indices": list(game_indices),
            "nominal_bytes": nominal_bytes,
        }
        chunks.append(
            CompactWindowChunk(
                source=source,
                mode=mode,
                epoch=epoch,
                sampler_seed=sampler_seed,
                chunk_index=chunk_index,
                first_game_offset=first,
                next_game_offset=next_offset,
                game_indices=game_indices,
                nominal_bytes=nominal_bytes,
                chunk_blake3=_canonical_blake3(descriptor),
            )
        )
    return tuple(chunks)


def _pad_axis_one(value: mx.array, width: int) -> mx.array:
    current = int(value.shape[1])
    if current > width:
        raise R2MapDatasetError("stitched batch padding would truncate an action axis")
    if current == width:
        return value
    padding = [(0, 0)] * value.ndim
    padding[1] = (0, width - current)
    return mx.pad(value, padding)


def _concatenate_public_states(
    states: Sequence[R2MapPublicState], *, candidates: bool, width: int = 0
) -> R2MapPublicState:
    if not states:
        raise R2MapDatasetError("cannot concatenate an empty public-state sequence")

    def combine(name: str) -> mx.array:
        values = [getattr(state, name) for state in states]
        if candidates:
            values = [_pad_axis_one(value, width) for value in values]
        return mx.concatenate(values, axis=0)

    result = R2MapPublicState(
        token_features=combine("token_features"),
        token_types=combine("token_types"),
        token_mask=combine("token_mask"),
        market_features=combine("market_features"),
        market_mask=combine("market_mask"),
        player_features=combine("player_features"),
        player_mask=combine("player_mask"),
        global_features=combine("global_features"),
    )
    result.validate(candidates=candidates)
    return result


def _concatenate_market_supervision(
    values: Sequence[R2MapMarketDecisionSupervision],
    identity_components: Sequence[str],
) -> R2MapMarketDecisionSupervision:
    if not values or not identity_components:
        raise R2MapDatasetError("stitched market supervision is empty")
    width = max(int(value.inputs.action_mask.shape[1]) for value in values)
    result = R2MapMarketDecisionSupervision(
        inputs=R2MapMarketDecisionBatch(
            public_state=_concatenate_public_states(
                [value.inputs.public_state for value in values], candidates=False
            ),
            action_mask=mx.concatenate(
                [_pad_axis_one(value.inputs.action_mask, width) for value in values],
                axis=0,
            ),
            action_features=mx.concatenate(
                [_pad_axis_one(value.inputs.action_features, width) for value in values],
                axis=0,
            ),
            exact_current_scores=mx.concatenate(
                [value.inputs.exact_current_scores for value in values], axis=0
            ),
        ),
        score_to_go_targets=mx.concatenate(
            [_pad_axis_one(value.score_to_go_targets, width) for value in values], axis=0
        ),
        score_target_mask=mx.concatenate(
            [_pad_axis_one(value.score_target_mask, width) for value in values], axis=0
        ),
        selected_action_index=mx.concatenate(
            [value.selected_action_index for value in values], axis=0
        ),
        policy_target_mask=mx.concatenate([value.policy_target_mask for value in values], axis=0),
        batch_identity=blake3.blake3("|".join(identity_components).encode()).hexdigest(),
    )
    result.validate()
    return result


def _concatenate_supervised_batches(
    values: Sequence[R2MapSupervisedBatch],
    draft_identity_components: Sequence[str],
    market_identity_components: Sequence[str],
) -> R2MapSupervisedBatch:
    """Reassemble one original packed batch from adjacent whole-game windows."""
    if not values or not draft_identity_components:
        raise R2MapDatasetError("stitched supervised batch is empty")
    if len(values) == 1:
        return values[0]
    width = max(int(value.inputs.candidate_mask.shape[1]) for value in values)

    def join(name: str, *, candidate_axis: bool = False) -> mx.array:
        arrays = [getattr(value, name) for value in values]
        if candidate_axis:
            arrays = [_pad_axis_one(array, width) for array in arrays]
        return mx.concatenate(arrays, axis=0)

    market_values = [value.market_decisions for value in values]
    if any(value is None for value in market_values):
        raise R2MapDatasetError("stitched supervised batch omitted market supervision")
    result = R2MapSupervisedBatch(
        inputs=R2MapBatch(
            parent=_concatenate_public_states(
                [value.inputs.parent for value in values], candidates=False
            ),
            candidates=_concatenate_public_states(
                [value.inputs.candidates for value in values],
                candidates=True,
                width=width,
            ),
            candidate_mask=mx.concatenate(
                [_pad_axis_one(value.inputs.candidate_mask, width) for value in values],
                axis=0,
            ),
            action_features=mx.concatenate(
                [_pad_axis_one(value.inputs.action_features, width) for value in values],
                axis=0,
            ),
            exact_afterstate_scores=mx.concatenate(
                [_pad_axis_one(value.inputs.exact_afterstate_scores, width) for value in values],
                axis=0,
            ),
        ),
        score_to_go_targets=join("score_to_go_targets", candidate_axis=True),
        score_component_targets=join("score_component_targets", candidate_axis=True),
        score_target_mask=join("score_target_mask", candidate_axis=True),
        selected_action_index=join("selected_action_index"),
        bootstrap_policy_mask=join("bootstrap_policy_mask"),
        opponent_tile_slot_targets=join("opponent_tile_slot_targets"),
        opponent_wildlife_slot_targets=join("opponent_wildlife_slot_targets"),
        opponent_draft_kind_targets=join("opponent_draft_kind_targets"),
        opponent_drafted_wildlife_targets=join("opponent_drafted_wildlife_targets"),
        opponent_replace_three_targets=join("opponent_replace_three_targets"),
        opponent_paid_wipe_count_targets=join("opponent_paid_wipe_count_targets"),
        opponent_paid_wipe_mask_targets=join("opponent_paid_wipe_mask_targets"),
        opponent_paid_wipe_mask_valid=join("opponent_paid_wipe_mask_valid"),
        opponent_valid_mask=join("opponent_valid_mask"),
        market_disposition_targets=join("market_disposition_targets"),
        market_pair_survival_targets=join("market_pair_survival_targets"),
        market_final_slot_targets=join("market_final_slot_targets"),
        market_disposition_mask=join("market_disposition_mask"),
        market_pair_survival_mask=join("market_pair_survival_mask"),
        market_final_slot_mask=join("market_final_slot_mask"),
        batch_identity=blake3.blake3("|".join(draft_identity_components).encode()).hexdigest(),
        market_decisions=_concatenate_market_supervision(
            [value for value in market_values if value is not None],
            market_identity_components,
        ),
    )
    result.validate()
    return result


class R2MapCompactDatasetAdapter(R2MapTrainingAdapter):
    """On-demand `.r2sh` adapter with at most two disposable source windows."""

    protocol_id = COMPACT_PROTOCOL_ID

    def __init__(
        self,
        *,
        index: str | Path | dict[str, Any],
        shard_root: str | Path | None = None,
        exporter: str | Path | None = None,
        window_root: str | Path | None = None,
        validated_aggregate_receipt: str | Path | None = None,
        validated_packing_receipt: str | Path | None = None,
        window_loader: (
            Callable[
                [str, str, int, int, int, tuple[int, ...]],
                tuple[dict[str, Any], bytes | bytearray],
            ]
            | None
        ) = None,
        group_batch_size: int = 2,
        maximum_candidates_per_batch: int = 16_384,
        maximum_window_bytes: int = 1 << 30,
        maximum_prefetch_windows: int = 1,
        fixed_panel_games: int = 1,
        require_ssd: bool = False,
        sampler_seed: int = 0,
        packing_epochs: int = WINDOW_PLAN_EPOCHS,
    ):
        if group_batch_size <= 0 or maximum_candidates_per_batch <= 0:
            raise ValueError("R2-MAP compact batch limits must be positive")
        if (
            maximum_prefetch_windows not in (0, 1)
            or fixed_panel_games <= 0
            or sampler_seed < 0
            or packing_epochs <= 0
        ):
            raise ValueError("R2-MAP compact prefetch or panel limits are invalid")
        self._window_loader = window_loader
        if window_loader is None:
            if isinstance(index, dict) or None in (shard_root, exporter, window_root):
                raise ValueError("local compact adapter requires index and all local paths")
            binding_paths = (validated_aggregate_receipt, validated_packing_receipt)
            if any(binding_paths) and not all(binding_paths):
                raise ValueError("local receipt-bound export requires both receipt paths")
            self.index_path: Path | None = Path(index).resolve(strict=True)
            self.index = validate_compact_index(self.index_path, shard_root=shard_root)
            self.shard_root: Path | None = Path(shard_root).resolve(strict=True)
            self.exporter: Path | None = Path(exporter).resolve(strict=True)
            self.window_root: Path | None = Path(window_root)
            self.semantic_validation_binding: tuple[Path, Path] | None = (
                (
                    Path(validated_aggregate_receipt).resolve(strict=True),
                    Path(validated_packing_receipt).resolve(strict=True),
                )
                if all(binding_paths)
                else None
            )
            self.window_root.mkdir(parents=True, exist_ok=True)
            if require_ssd:
                _require_campaign_ssd(self.window_root)
                _require_campaign_ssd(self.shard_root)
            for stale in (
                *self.window_root.glob("window-*.r2map"),
                *self.window_root.glob("window-*.json"),
            ):
                stale.unlink(missing_ok=True)
        else:
            if not isinstance(index, dict) or any(
                value is not None
                for value in (
                    shard_root,
                    exporter,
                    window_root,
                    validated_aggregate_receipt,
                    validated_packing_receipt,
                )
            ):
                raise ValueError(
                    "remote compact adapter requires an in-memory index and no local paths"
                )
            if require_ssd:
                raise ValueError("remote compact adapter cannot authorize a local SSD path")
            if maximum_prefetch_windows != 0:
                raise ValueError("remote compact adapter forbids a second in-memory window")
            if maximum_window_bytes > MAX_IN_MEMORY_STREAM_BYTES:
                raise ValueError("remote compact window exceeds the 1 GiB memory ceiling")
            self.index = validate_compact_index_value(index)
            self.shard_root = None
            self.exporter = None
            self.window_root = None
            self.index_path = None
            self.semantic_validation_binding = None
        self.dataset_blake3 = self.index["dataset_manifest"]["dataset_blake3"]
        self.dataset_contract = _training_dataset_contract(self.index["dataset_manifest"])
        self.group_batch_size = group_batch_size
        self.maximum_candidates_per_batch = maximum_candidates_per_batch
        self.maximum_window_bytes = maximum_window_bytes
        self.maximum_prefetch_windows = maximum_prefetch_windows
        self.fixed_panel_games = fixed_panel_games
        self.sampler_seed = sampler_seed
        self.packing_epochs = packing_epochs
        self._games_by_source: dict[str, list[dict[str, Any]]] = {}
        for game in self.index["games"]:
            self._games_by_source.setdefault(game["source_file_name"], []).append(game)
        self._sources = {
            source["file_name"]: source for source in self.index["dataset_manifest"]["sources"]
        }
        self._chunk_cache: dict[tuple[str, str, int, int], tuple[CompactWindowChunk, ...]] = {}
        chunk_plan = []
        for planned_epoch in range(self.packing_epochs + 1):
            for source_name in self._source_order(planned_epoch, self.sampler_seed, split="train"):
                chunk_plan.extend(
                    chunk.to_dict()
                    for chunk in self._window_chunks(
                        source_name, "train", planned_epoch, self.sampler_seed
                    )
                )
        for source_name in self._source_order(0, 0, split="validation"):
            chunk_plan.extend(
                chunk.to_dict() for chunk in self._window_chunks(source_name, "validation", 0, 0)
            )
        self.chunk_plan_blake3 = _canonical_blake3(chunk_plan)
        self.dataset_contract.update(
            {
                "compact_index_blake3": self.index["index_blake3"],
                "window_chunking_protocol": WINDOW_CHUNKING_PROTOCOL,
                "window_target_games": WINDOW_TARGET_GAMES,
                "window_maximum_nominal_games": WINDOW_MAXIMUM_NOMINAL_GAMES,
                "window_nominal_bytes_per_draft_group": (WINDOW_NOMINAL_BYTES_PER_DRAFT_GROUP),
                "window_nominal_bytes_per_candidate": WINDOW_NOMINAL_BYTES_PER_CANDIDATE,
                "window_preferred_target_bytes": WINDOW_PREFERRED_TARGET_BYTES,
                "window_nominal_target_bytes": WINDOW_NOMINAL_TARGET_BYTES,
                "window_hard_maximum_bytes": self.maximum_window_bytes,
                "window_chunk_plan_blake3": self.chunk_plan_blake3,
                "window_chunk_plan_epochs": self.packing_epochs,
                "window_chunk_plan_includes_terminal_next_epoch": True,
                "window_chunk_count": len(chunk_plan),
            }
        )
        self._executor = ThreadPoolExecutor(max_workers=1) if maximum_prefetch_windows else None
        self._future: (
            tuple[
                tuple[str, str, int, int, int],
                Future[tuple[Path, Path] | tuple[dict[str, Any], bytes | bytearray]],
            ]
            | None
        ) = None
        self._current_key: tuple[str, str, int, int, int] | None = None
        self._current_reader: R2MapStreamReader | None = None
        self._closed = False

    def __enter__(self) -> R2MapCompactDatasetAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_current()
        if self._future is not None:
            _, future = self._future
            self._discard_window(future.result())
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed != self.sampler_seed:
            raise R2MapDatasetError("training seed differs from the bounded window plan")
        return (
            self._cursor_value(0, 0, 0, 0, seed),
            {"seed": int(seed), "sampler_protocol": "shard-game-turn-v1"},
        )

    def _cursor_value(
        self,
        epoch: int,
        source_offset: int,
        game_offset: int,
        turn_offset: int,
        seed: int,
    ) -> dict[str, Any]:
        source_order = self._source_order(epoch, seed, split="train")
        if source_offset >= len(source_order):
            raise R2MapDatasetError("bounded cursor source offset exceeds its epoch")
        source = source_order[source_offset]
        chunk = self._chunk_for_game_offset(source, "train", epoch, seed, game_offset)
        return {
            "epoch": epoch,
            "source_offset": source_offset,
            "chunk_index": chunk.chunk_index,
            "chunk_first_game_offset": chunk.first_game_offset,
            "chunk_next_game_offset": chunk.next_game_offset,
            "chunk_blake3": chunk.chunk_blake3,
            "game_offset": game_offset,
            "turn_offset": turn_offset,
        }

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        epoch, source_offset, game_offset, turn_offset, seed = self._validate_cursor(
            cursor, sampler_state
        )
        source_order = self._source_order(epoch, seed, split="train")
        if not source_order:
            raise R2MapDatasetError("compact R2-MAP index has no training games")
        if source_offset >= len(source_order):
            epoch += 1
            source_offset = game_offset = turn_offset = 0
            source_order = self._source_order(epoch, seed, split="train")
        source_name = source_order[source_offset]
        game_order = self._game_order(source_name, epoch, seed, split="train")
        positions, next_game, next_turn = self._batch_positions(
            game_order, game_offset, turn_offset
        )
        batch = self._batch_from_positions(source_name, "train", epoch, seed, game_order, positions)
        next_source = source_offset
        if next_game == len(game_order):
            next_source += 1
            next_game = next_turn = 0
        if next_source == len(source_order):
            epoch += 1
            next_source = next_game = next_turn = 0
            source_order = self._source_order(epoch, seed, split="train")
        if next_source < len(source_order):
            next_source_name = source_order[next_source]
            next_chunk = self._chunk_for_game_offset(
                next_source_name, "train", epoch, seed, next_game
            )
            self._schedule(next_source_name, "train", epoch, seed, next_chunk.chunk_index)
        return R2MapAdapterStep(
            batch=batch,
            next_cursor=self._cursor_value(epoch, next_source, next_game, next_turn, seed),
            next_sampler_state=dict(sampler_state),
        )

    def validation_batches(self) -> Iterable[R2MapSupervisedBatch]:
        def batches() -> Iterator[R2MapSupervisedBatch]:
            sources = self._source_order(0, 0, split="validation")
            for source_name in sources:
                game_order = self._game_order(source_name, 0, 0, split="validation")
                game_offset = turn_offset = 0
                while game_offset < len(game_order):
                    positions, game_offset, turn_offset = self._batch_positions(
                        game_order, game_offset, turn_offset
                    )
                    yield self._batch_from_positions(
                        source_name,
                        "validation",
                        0,
                        0,
                        game_order,
                        positions,
                    )

        return batches()

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch:
        if not panel_id:
            raise R2MapDatasetError("R2-MAP fixed panel must be named")
        games = sorted(
            (game for game in self.index["games"] if game["split"] == "validation"),
            key=lambda game: (game["global_game_index"], game["game_id"]),
        )[: self.fixed_panel_games]
        if not games:
            raise R2MapDatasetError("compact R2-MAP index has no fixed-panel games")
        source_names = {game["source_file_name"] for game in games}
        if len(source_names) != 1:
            raise R2MapDatasetError(
                "fixed panel crosses source windows; lower fixed-panel-games or shard larger"
            )
        source_name = next(iter(source_names))
        game_order = self._game_order(source_name, 0, 0, split="validation")
        first_offset = next(
            index for index, game in enumerate(game_order) if game["game_id"] == games[0]["game_id"]
        )
        panel_offsets = [
            next(
                index
                for index, ordered_game in enumerate(game_order)
                if ordered_game["game_id"] == game["game_id"]
            )
            for game in games
        ]
        panel_chunks = {
            self._chunk_for_game_offset(source_name, "validation", 0, 0, offset).chunk_index
            for offset in panel_offsets
        }
        if len(panel_chunks) != 1:
            raise R2MapDatasetError("fixed panel crosses bounded windows; lower fixed-panel-games")
        chunk = self._chunk_for_game_offset(source_name, "validation", 0, 0, first_offset)
        reader = self._window(source_name, "validation", 0, 0, chunk.chunk_index)
        refs_by_game = _refs_by_game(reader.refs)
        indices = [index for game in games for index in refs_by_game[game["game_id"]]]
        return reader.fixed_selected_batch(indices)

    @staticmethod
    def _game_widths(game: Mapping[str, Any]) -> tuple[int, ...]:
        widths = game.get("candidate_widths")
        if widths is None and game.get("imitation_example_count") == 0:
            widths = [1] * int(game["example_count"])
        if not isinstance(widths, list) or len(widths) != game["example_count"]:
            raise R2MapDatasetError("compact game omits exact candidate widths")
        return tuple(int(width) for width in widths)

    def _batch_positions(
        self,
        game_order: Sequence[dict[str, Any]],
        game_offset: int,
        turn_offset: int,
    ) -> tuple[tuple[tuple[int, int], ...], int, int]:
        selected: list[tuple[int, int]] = []
        padded_width = 0
        next_game, next_turn = game_offset, turn_offset
        while len(selected) < self.group_batch_size and next_game < len(game_order):
            widths = self._game_widths(game_order[next_game])
            if next_turn >= len(widths):
                raise R2MapDatasetError("compact training turn cursor exceeds indexed game")
            next_width = max(padded_width, widths[next_turn])
            if next_width * (len(selected) + 1) > self.maximum_candidates_per_batch:
                if not selected:
                    raise R2MapDatasetError(
                        "one compact training group exceeds the padded candidate budget"
                    )
                break
            selected.append((next_game, next_turn))
            padded_width = next_width
            next_turn += 1
            if next_turn == len(widths):
                next_game += 1
                next_turn = 0
        if not selected:
            raise R2MapDatasetError("compact training cursor selected no examples")
        return tuple(selected), next_game, next_turn

    def _batch_from_positions(
        self,
        source_name: str,
        mode: str,
        epoch: int,
        seed: int,
        game_order: Sequence[dict[str, Any]],
        positions: Sequence[tuple[int, int]],
    ) -> R2MapSupervisedBatch:
        segments: list[tuple[int, list[tuple[int, int]]]] = []
        for game_offset, turn_offset in positions:
            chunk_index = self._chunk_for_game_offset(
                source_name, mode, epoch, seed, game_offset
            ).chunk_index
            if not segments or segments[-1][0] != chunk_index:
                segments.append((chunk_index, []))
            segments[-1][1].append((game_offset, turn_offset))

        partials: list[R2MapSupervisedBatch] = []
        draft_identities: list[str] = []
        market_identities: list[str] = []
        for chunk_index, segment in segments:
            reader = self._window(source_name, mode, epoch, seed, chunk_index)
            refs_by_game = _refs_by_game(reader.refs)
            selected: list[int] = []
            for game_offset, turn_offset in segment:
                game = game_order[game_offset]
                refs = refs_by_game.get(game["game_id"])
                if refs is None or len(refs) != game["example_count"]:
                    raise R2MapDatasetError("on-demand bounded window differs from compact index")
                ref_index = refs[turn_offset]
                expected_width = self._game_widths(game)[turn_offset]
                if reader.refs[ref_index].candidate_count != expected_width:
                    raise R2MapDatasetError(
                        "on-demand bounded window candidate width differs from compact index"
                    )
                selected.append(ref_index)
            partial, draft_parts, market_parts = reader._batch_with_identity_components(selected)
            partials.append(partial)
            draft_identities.extend(draft_parts)
            market_identities.extend(market_parts)
        return _concatenate_supervised_batches(partials, draft_identities, market_identities)

    def _validate_cursor(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> tuple[int, int, int, int, int]:
        if set(cursor) != {
            "epoch",
            "source_offset",
            "chunk_index",
            "chunk_first_game_offset",
            "chunk_next_game_offset",
            "chunk_blake3",
            "game_offset",
            "turn_offset",
        }:
            raise R2MapDatasetError("compact training cursor schema differs")
        if (
            set(sampler_state) != {"seed", "sampler_protocol"}
            or sampler_state["sampler_protocol"] != "shard-game-turn-v1"
        ):
            raise R2MapDatasetError("compact sampler state schema differs")
        integer_names = (
            "epoch",
            "source_offset",
            "chunk_index",
            "chunk_first_game_offset",
            "chunk_next_game_offset",
            "game_offset",
            "turn_offset",
        )
        values = tuple(int(cursor[name]) for name in integer_names)
        if any(value < 0 for value in values) or not isinstance(cursor["chunk_blake3"], str):
            raise R2MapDatasetError("compact training cursor is negative")
        epoch, source_offset, chunk_index, chunk_first, chunk_next, game_offset, turn = values
        seed = int(sampler_state["seed"])
        source_order = self._source_order(epoch, seed, split="train")
        if source_offset >= len(source_order):
            raise R2MapDatasetError("compact training source cursor exceeds its epoch")
        chunk = self._chunk_for_game_offset(
            source_order[source_offset], "train", epoch, seed, game_offset
        )
        if (
            chunk.chunk_index != chunk_index
            or chunk.first_game_offset != chunk_first
            or chunk.next_game_offset != chunk_next
            or chunk.chunk_blake3 != cursor["chunk_blake3"]
        ):
            raise R2MapDatasetError("compact training chunk cursor identity differs")
        return epoch, source_offset, game_offset, turn, seed

    def _source_order(self, epoch: int, seed: int, *, split: str) -> tuple[str, ...]:
        eligible = [
            name
            for name, games in self._games_by_source.items()
            if any(game["split"] == split for game in games)
        ]
        if split == "validation":
            return tuple(sorted(eligible, key=lambda name: self._sources[name]["first_game_index"]))
        return tuple(sorted(eligible, key=lambda name: _sampler_hash(seed, epoch, name)))

    def _game_order(
        self, source_name: str, epoch: int, seed: int, *, split: str
    ) -> tuple[dict[str, Any], ...]:
        games = [game for game in self._games_by_source[source_name] if game["split"] == split]
        if split == "validation":
            games.sort(key=lambda game: game["global_game_index"])
        else:
            games.sort(key=lambda game: _sampler_hash(seed, epoch, game["game_id"]))
        return tuple(games)

    def _window_chunks(
        self, source: str, mode: str, epoch: int, seed: int
    ) -> tuple[CompactWindowChunk, ...]:
        key = (source, mode, epoch, seed)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            return cached
        split = "train" if mode == "train" else "validation"
        chunks = _bounded_whole_game_window_chunks(
            source,
            mode,
            epoch,
            seed,
            self._game_order(source, epoch, seed, split=split),
            group_batch_size=self.group_batch_size,
            maximum_candidates_per_batch=self.maximum_candidates_per_batch,
        )
        if not chunks:
            raise R2MapDatasetError("bounded window plan has no chunks")
        self._chunk_cache[key] = chunks
        return chunks

    def _chunk_for_game_offset(
        self, source: str, mode: str, epoch: int, seed: int, game_offset: int
    ) -> CompactWindowChunk:
        for chunk in self._window_chunks(source, mode, epoch, seed):
            if chunk.first_game_offset <= game_offset < chunk.next_game_offset:
                return chunk
        raise R2MapDatasetError("training cursor is outside the bounded window plan")

    def _window(
        self, source: str, mode: str, epoch: int, seed: int, chunk_index: int
    ) -> R2MapStreamReader:
        key = (source, mode, epoch, seed, chunk_index)
        if key == self._current_key and self._current_reader is not None:
            return self._current_reader
        self._close_current()
        window: tuple[Path, Path] | tuple[dict[str, Any], bytes | bytearray] | None = None
        if self._future is not None:
            future_key, future = self._future
            future_paths = future.result()
            self._future = None
            if future_key == key:
                window = future_paths
            else:
                self._discard_window(future_paths)
        if window is None:
            window = self._materialize(key)
        manifest, stream = window
        chunk = self._window_chunks(key[0], key[1], key[2], key[3])[key[4]]
        self._current_reader = R2MapStreamReader(manifest, stream, game_indices=chunk.game_indices)
        self._current_key = key
        return self._current_reader

    def _schedule(self, source: str, mode: str, epoch: int, seed: int, chunk_index: int) -> None:
        if self._executor is None or self._future is not None:
            return
        key = (source, mode, epoch, seed, chunk_index)
        if key == self._current_key:
            return
        self._future = (key, self._executor.submit(self._materialize, key))

    def _materialize(
        self, key: tuple[str, str, int, int, int]
    ) -> tuple[Path, Path] | tuple[dict[str, Any], bytes | bytearray]:
        source, mode, epoch, seed, chunk_index = key
        chunks = self._window_chunks(source, mode, epoch, seed)
        if chunk_index >= len(chunks):
            raise R2MapDatasetError("bounded window chunk index exceeds its plan")
        chunk = chunks[chunk_index]
        if self._window_loader is not None:
            manifest_value, stream_bytes = self._window_loader(
                source, mode, epoch, seed, chunk_index, chunk.game_indices
            )
            if (
                not isinstance(stream_bytes, (bytes, bytearray))
                or len(stream_bytes) > self.maximum_window_bytes
            ):
                raise R2MapDatasetError("remote R2-MAP window exceeds bounded memory gate")
            local = _validate_manifest_value(manifest_value)
            expected = self._sources[source]
            if len(local["sources"]) != 1 or local["sources"][0] != expected:
                raise R2MapDatasetError("remote R2-MAP window source identity differs")
            return local, stream_bytes
        assert self.window_root is not None
        assert self.exporter is not None
        assert self.shard_root is not None
        identity = blake3.blake3("|".join(map(str, key)).encode()).hexdigest()[:24]
        manifest = self.window_root / f"window-{identity}.json"
        stream = self.window_root / f"window-{identity}.r2map"
        _run_exporter(
            self.exporter,
            shard=self.shard_root / source,
            manifest=manifest,
            stream=stream,
            mode=mode,
            epoch=epoch if mode == "train" else 0,
            sampler_seed=seed if mode == "train" else 0,
            compact_index=self.index_path,
            semantic_validation_binding=self.semantic_validation_binding,
            game_indices=chunk.game_indices,
        )
        if stream.stat().st_size > self.maximum_window_bytes:
            _remove_window((manifest, stream))
            raise R2MapDatasetError("on-demand R2-MAP window exceeds bounded storage gate")
        local = _read_manifest(manifest)
        expected = self._sources[source]
        if len(local["sources"]) != 1 or local["sources"][0] != expected:
            _remove_window((manifest, stream))
            raise R2MapDatasetError("on-demand R2-MAP window source identity differs")
        return manifest, stream

    def _close_current(self) -> None:
        if self._current_reader is None:
            return
        paths = (self._current_reader.manifest_path, self._current_reader.stream_path)
        self._current_reader.close()
        self._current_reader = None
        self._current_key = None
        if all(path is not None for path in paths):
            _remove_window((paths[0], paths[1]))  # type: ignore[arg-type]

    @staticmethod
    def _discard_window(
        window: tuple[Path, Path] | tuple[dict[str, Any], bytes | bytearray],
    ) -> None:
        if isinstance(window[0], Path):
            _remove_window((window[0], window[1]))  # type: ignore[arg-type]


def _aggregate_source_manifests(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        raise R2MapDatasetError("cannot aggregate an empty compact source set")
    first = values[0]
    invariant = (
        "schema_version",
        "protocol_id",
        "feature_schema",
        "target_schema",
        "split_schema",
        "d6_schema",
        "imitation_subset_schema",
        "imitation_subset_parts_per_million",
        "round",
    )
    if any(any(value[name] != first[name] for name in invariant) for value in values[1:]):
        raise R2MapDatasetError("compact sources mix dataset contracts or rounds")
    sources = sorted(
        (source for value in values for source in value["sources"]),
        key=lambda source: (source["first_game_index"], source["blake3"]),
    )
    for left, right in pairwise(sources):
        if left["next_game_index"] > right["first_game_index"]:
            raise R2MapDatasetError("compact source game ranges overlap")
    manifest: dict[str, Any] = {
        "schema_version": first["schema_version"],
        "protocol_id": first["protocol_id"],
        "feature_schema": first["feature_schema"],
        "target_schema": first["target_schema"],
        "split_schema": first["split_schema"],
        "d6_schema": first["d6_schema"],
        "imitation_subset_schema": first["imitation_subset_schema"],
        "imitation_subset_parts_per_million": first["imitation_subset_parts_per_million"],
        "round": first["round"],
        "game_count": sum(value["game_count"] for value in values),
        "example_count": sum(value["example_count"] for value in values),
        "imitation_example_count": sum(value["imitation_example_count"] for value in values),
        "market_decision_count": sum(value["market_decision_count"] for value in values),
        "market_policy_target_count": sum(value["market_policy_target_count"] for value in values),
        "train_games": sum(value["train_games"] for value in values),
        "validation_games": sum(value["validation_games"] for value in values),
        "sources": sources,
    }
    identity = dict(manifest)
    manifest["dataset_blake3"] = blake3.blake3(
        json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    # Match the published manifest field order while validation remains order-independent.
    return _validate_manifest_value(
        {
            **{name: manifest[name] for name in invariant},
            "dataset_blake3": manifest["dataset_blake3"],
            **{
                name: manifest[name]
                for name in (
                    "game_count",
                    "example_count",
                    "imitation_example_count",
                    "market_decision_count",
                    "market_policy_target_count",
                    "train_games",
                    "validation_games",
                    "sources",
                )
            },
        }
    )


def _run_exporter(
    exporter: Path,
    *,
    shard: Path,
    manifest: Path,
    stream: Path,
    mode: str,
    epoch: int,
    sampler_seed: int,
    compact_index: Path | None,
    semantic_validation_binding: tuple[Path, Path] | None,
    game_indices: Sequence[int],
) -> None:
    if mode not in {"train", "validation"}:
        raise R2MapDatasetError("compact window mode is unsupported")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(exporter),
        "export-r2-map-dataset",
        "--shard",
        str(shard),
        "--manifest",
        str(manifest),
        "--stream",
        str(stream),
        "--mode",
        mode,
        "--epoch",
        str(epoch),
        "--sampler-seed",
        str(sampler_seed),
    ]
    if semantic_validation_binding is not None:
        if compact_index is None:
            raise R2MapDatasetError("receipt-bound export omitted the compact index path")
        aggregate_receipt, packing_receipt = semantic_validation_binding
        command.extend(
            [
                "--validated-aggregate-receipt",
                str(aggregate_receipt),
                "--validated-compact-index",
                str(compact_index),
                "--validated-packing-receipt",
                str(packing_receipt),
            ]
        )
    for game_index in game_indices:
        command.extend(["--game-index", str(game_index)])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "")
        raise R2MapDatasetError(f"compact R2-MAP exporter failed: {detail}") from error


def _refs_by_game(refs: Sequence[R2MapFrameRef]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    previous_turn: dict[str, int] = {}
    for index, ref in enumerate(refs):
        game_id = ref.game_id.hex()
        expected = previous_turn.get(game_id, -1) + 1
        if ref.turn != expected:
            raise R2MapDatasetError("on-demand R2-MAP game turns are not contiguous")
        previous_turn[game_id] = ref.turn
        grouped.setdefault(game_id, []).append(index)
    return {game_id: tuple(indices) for game_id, indices in grouped.items()}


def _sampler_hash(seed: int, epoch: int, identity: str) -> bytes:
    return blake3.blake3(
        b"r2-map-compact-sampler-v1"
        + seed.to_bytes(8, "little", signed=False)
        + epoch.to_bytes(8, "little", signed=False)
        + identity.encode()
    ).digest()


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _remove_window(paths: tuple[Path, Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _require_campaign_ssd(path: Path) -> None:
    try:
        require_local_storage_authority()
    except StoragePreflightError as error:
        raise R2MapDatasetError(
            "local compact windows are restricted to john1's primary campaign store"
        ) from error
    root = CAMPAIGN_ROOT.resolve(strict=True)
    candidate = path.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise R2MapDatasetError(
            f"compact R2-MAP path must remain below john1 root {root}"
        ) from error


def math_ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def d6_transform_id(
    *,
    game_id: bytes,
    draft_decision_id: bytes,
    mode: int,
    epoch: int,
    sampler_seed: int,
) -> int:
    """Independently reproduce the frozen cyclic 12-transform schedule."""
    if len(game_id) != 32 or len(draft_decision_id) != 32:
        raise R2MapDatasetError("D6 schedule identities must be 32 bytes")
    if (
        not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode not in {0, 1, 2}
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or not 0 <= epoch <= 0xFFFFFFFFFFFFFFFF
        or not isinstance(sampler_seed, int)
        or isinstance(sampler_seed, bool)
        or not 0 <= sampler_seed <= 0xFFFFFFFFFFFFFFFF
    ):
        raise R2MapDatasetError("D6 schedule counters are invalid")
    if mode != 0:
        return 0
    digest = _hash_parts(
        D6_SCHEMA.encode(),
        (game_id, draft_decision_id, sampler_seed.to_bytes(8, "little")),
    )
    return (digest[0] % 12 + epoch % 12) % 12


def draft_is_imitation_subset(*, collection_kind: str, draft_decision_id: bytes) -> bool:
    """Recompute the frozen bootstrap-only 1% full-screen exposure flag."""
    if collection_kind not in {"bootstrap", "iterative-training", "benchmark"}:
        raise R2MapDatasetError("R2-MAP collection kind is invalid")
    if len(draft_decision_id) != 32:
        raise R2MapDatasetError("draft imitation identity must be 32 bytes")
    if collection_kind != "bootstrap":
        return False
    digest = _hash_parts(IMITATION_SUBSET_SCHEMA.encode(), (draft_decision_id,))
    draw = int.from_bytes(digest[:8], "little")
    return draw * 1_000_000 < 0xFFFFFFFFFFFFFFFF * IMITATION_SUBSET_PARTS_PER_MILLION


def _hash_parts(domain: bytes, parts: Sequence[bytes]) -> bytes:
    digest = blake3.blake3()
    digest.update(len(domain).to_bytes(8, "little"))
    digest.update(domain)
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.digest()


def _training_dataset_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_blake3": manifest["dataset_blake3"],
        "d6_schema": manifest["d6_schema"],
        "d6_cycle_epochs": 12,
        "imitation_subset_schema": manifest["imitation_subset_schema"],
        "imitation_subset_parts_per_million": manifest["imitation_subset_parts_per_million"],
        "collection_kind": manifest["round"]["collection_kind"],
        "example_count": manifest["example_count"],
        "imitation_example_count": manifest["imitation_example_count"],
        "market_decision_count": manifest["market_decision_count"],
        "market_policy_target_count": manifest["market_policy_target_count"],
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R2MapDatasetError("cannot read R2-MAP manifest") from error
    return _validate_manifest_value(value)


def _validate_manifest_value(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_id",
        "feature_schema",
        "target_schema",
        "split_schema",
        "d6_schema",
        "imitation_subset_schema",
        "imitation_subset_parts_per_million",
        "round",
        "dataset_blake3",
        "game_count",
        "example_count",
        "imitation_example_count",
        "market_decision_count",
        "market_policy_target_count",
        "train_games",
        "validation_games",
        "sources",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise R2MapDatasetError("R2-MAP manifest schema is incomplete")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["protocol_id"] != PROTOCOL_ID
        or value["feature_schema"] != FEATURE_SCHEMA
        or value["target_schema"] != TARGET_SCHEMA
        or value["split_schema"] != SPLIT_SCHEMA
        or value["d6_schema"] != D6_SCHEMA
        or value["imitation_subset_schema"] != IMITATION_SUBSET_SCHEMA
        or value["imitation_subset_parts_per_million"] != IMITATION_SUBSET_PARTS_PER_MILLION
    ):
        raise R2MapDatasetError("R2-MAP manifest contract drifted")
    sources = value["sources"]
    round_identity = value["round"]
    if (
        not isinstance(round_identity, dict)
        or set(round_identity)
        != {"campaign_id", "iteration", "collection_kind", "newest_checkpoint_blake3"}
        or not isinstance(round_identity["campaign_id"], str)
        or not round_identity["campaign_id"]
        or not isinstance(round_identity["iteration"], int)
        or isinstance(round_identity["iteration"], bool)
        or round_identity["iteration"] < 0
        or round_identity["collection_kind"] not in {"bootstrap", "iterative-training", "benchmark"}
        or not _optional_digest(round_identity["newest_checkpoint_blake3"])
        or (
            round_identity["collection_kind"] == "iterative-training"
            and round_identity["newest_checkpoint_blake3"] is None
        )
        or (
            round_identity["collection_kind"] != "iterative-training"
            and round_identity["newest_checkpoint_blake3"] is not None
        )
    ):
        raise R2MapDatasetError("R2-MAP round identity is invalid")
    source_keys = {
        "file_name",
        "bytes",
        "blake3",
        "first_game_index",
        "next_game_index",
        "game_count",
        "example_count",
        "imitation_example_count",
        "market_decision_count",
        "market_policy_target_count",
    }
    if not isinstance(sources, list) or not sources:
        raise R2MapDatasetError("R2-MAP manifest requires source shards")
    top_counts = tuple(
        value[name]
        for name in (
            "game_count",
            "example_count",
            "imitation_example_count",
            "market_decision_count",
            "market_policy_target_count",
            "train_games",
            "validation_games",
        )
    )
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in top_counts):
        raise R2MapDatasetError("R2-MAP manifest counts are invalid")
    previous_next = None
    for source in sources:
        if not isinstance(source, dict) or set(source) != source_keys:
            raise R2MapDatasetError("R2-MAP source identity schema drifted")
        digest = source["blake3"]
        numeric = (
            source["bytes"],
            source["first_game_index"],
            source["next_game_index"],
            source["game_count"],
            source["example_count"],
            source["imitation_example_count"],
            source["market_decision_count"],
            source["market_policy_target_count"],
        )
        if (
            not isinstance(source["file_name"], str)
            or Path(source["file_name"]).name != source["file_name"]
            or any(not isinstance(item, int) or isinstance(item, bool) for item in numeric)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or source["bytes"] <= 0
            or source["game_count"] <= 0
            or source["example_count"] <= 0
            or not 0 <= source["imitation_example_count"] <= source["example_count"]
            or source["market_decision_count"] <= 0
            or not 0 <= source["market_policy_target_count"] <= source["market_decision_count"]
            or source["next_game_index"] <= source["first_game_index"]
            or source["next_game_index"] - source["first_game_index"] != source["game_count"]
            or (previous_next is not None and source["first_game_index"] < previous_next)
        ):
            raise R2MapDatasetError("R2-MAP source identity is invalid")
        previous_next = source["next_game_index"]
    if (
        sum(source["game_count"] for source in sources) != value["game_count"]
        or sum(source["example_count"] for source in sources) != value["example_count"]
        or sum(source["imitation_example_count"] for source in sources)
        != value["imitation_example_count"]
        or sum(source["market_decision_count"] for source in sources)
        != value["market_decision_count"]
        or sum(source["market_policy_target_count"] for source in sources)
        != value["market_policy_target_count"]
        or value["train_games"] + value["validation_games"] != value["game_count"]
        or value["imitation_example_count"] > value["example_count"]
        or value["market_policy_target_count"] > value["market_decision_count"]
        or (
            round_identity["collection_kind"] == "bootstrap"
            and value["market_policy_target_count"] != value["market_decision_count"]
        )
        or (
            round_identity["collection_kind"] != "bootstrap"
            and (value["imitation_example_count"] != 0 or value["market_policy_target_count"] != 0)
        )
    ):
        raise R2MapDatasetError("R2-MAP manifest source accounting drifted")
    canonical_round = {
        key: round_identity[key]
        for key in (
            "campaign_id",
            "iteration",
            "collection_kind",
            "newest_checkpoint_blake3",
        )
    }
    canonical_sources = [
        {
            key: source[key]
            for key in (
                "file_name",
                "bytes",
                "blake3",
                "first_game_index",
                "next_game_index",
                "game_count",
                "example_count",
                "imitation_example_count",
                "market_decision_count",
                "market_policy_target_count",
            )
        }
        for source in sources
    ]
    identity = {
        "schema_version": value["schema_version"],
        "protocol_id": value["protocol_id"],
        "feature_schema": value["feature_schema"],
        "target_schema": value["target_schema"],
        "split_schema": value["split_schema"],
        "d6_schema": value["d6_schema"],
        "imitation_subset_schema": value["imitation_subset_schema"],
        "imitation_subset_parts_per_million": value["imitation_subset_parts_per_million"],
        "round": canonical_round,
        "game_count": value["game_count"],
        "example_count": value["example_count"],
        "imitation_example_count": value["imitation_example_count"],
        "market_decision_count": value["market_decision_count"],
        "market_policy_target_count": value["market_policy_target_count"],
        "train_games": value["train_games"],
        "validation_games": value["validation_games"],
        "sources": canonical_sources,
    }
    encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    if blake3.blake3(encoded).hexdigest() != value["dataset_blake3"]:
        raise R2MapDatasetError("R2-MAP manifest dataset identity failed")
    return value


def _optional_digest(value: object) -> bool:
    return value is None or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_state(payload: memoryview, cursor: int) -> tuple[_State, int]:
    token_count = _u16(payload, cursor)
    cursor += 2
    counts = _array(payload, cursor, "<u2", BOARD_SLOTS * 4).reshape(BOARD_SLOTS, 4)
    cursor += BOARD_SLOTS * 4 * 2
    board_counts = counts.sum(axis=1)
    if token_count != int(board_counts.sum()) or np.any(board_counts > BOARD_TOKEN_CAPACITY):
        raise R2MapDatasetError("compact R2 board counts are inconsistent")
    types_compact = _array(payload, cursor, "u1", token_count)
    cursor += token_count
    seats_compact = _array(payload, cursor, "u1", token_count)
    cursor += token_count
    compact_payload = _array(payload, cursor, "i1", token_count * TOKEN_PAYLOAD_WIDTH).reshape(
        token_count, TOKEN_PAYLOAD_WIDTH
    )
    cursor += token_count * TOKEN_PAYLOAD_WIDTH
    types = np.zeros((BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype=np.uint8)
    seats = np.zeros_like(types)
    token_payload = np.zeros((*types.shape, TOKEN_PAYLOAD_WIDTH), dtype=np.int8)
    source = 0
    for board in range(BOARD_SLOTS):
        count = int(board_counts[board])
        expected = np.repeat(np.arange(1, 5, dtype=np.uint8), counts[board].astype(np.int64))
        if not np.array_equal(types_compact[source : source + count], expected):
            raise R2MapDatasetError("compact R2 token type ordering drifted")
        if np.any(seats_compact[source : source + count] != board):
            raise R2MapDatasetError("compact R2 relative-seat ownership drifted")
        types[board, :count] = types_compact[source : source + count]
        seats[board, :count] = seats_compact[source : source + count]
        token_payload[board, :count] = compact_payload[source : source + count]
        source += count
    mask = types != 0
    features = _materialize_token_features(types, seats, token_payload, mask)
    market = _array(payload, cursor, "<f4", MARKET_SLOTS * MARKET_FEATURES).reshape(
        MARKET_SLOTS, MARKET_FEATURES
    )
    cursor += MARKET_SLOTS * MARKET_FEATURES * 4
    market_mask = _array(payload, cursor, "u1", MARKET_SLOTS).astype(np.bool_)
    cursor += MARKET_SLOTS
    players = _array(payload, cursor, "<f4", BOARD_SLOTS * PLAYER_FEATURES).reshape(
        BOARD_SLOTS, PLAYER_FEATURES
    )
    cursor += BOARD_SLOTS * PLAYER_FEATURES * 4
    player_mask = _array(payload, cursor, "u1", BOARD_SLOTS).astype(np.bool_)
    cursor += BOARD_SLOTS
    global_features = _array(payload, cursor, "<f4", GLOBAL_FEATURES)
    cursor += GLOBAL_FEATURES * 4
    state = _State(
        features,
        types.astype(np.int32),
        mask,
        market,
        market_mask,
        players,
        player_mask,
        global_features,
    )
    return state, cursor


def _stack_states(states: list[_State]) -> dict[str, np.ndarray]:
    return {
        name: np.stack([getattr(state, name) for state in states])
        for name in _State.__dataclass_fields__
    }


def _empty_state_arrays(groups: int, width: int) -> dict[str, np.ndarray]:
    return {
        "token_features": np.zeros(
            (groups, width, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
            np.float32,
        ),
        "token_types": np.zeros((groups, width, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), np.int32),
        "token_mask": np.zeros((groups, width, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), np.bool_),
        "market_features": np.zeros((groups, width, MARKET_SLOTS, MARKET_FEATURES), np.float32),
        "market_mask": np.zeros((groups, width, MARKET_SLOTS), np.bool_),
        "player_features": np.zeros((groups, width, BOARD_SLOTS, PLAYER_FEATURES), np.float32),
        "player_mask": np.zeros((groups, width, BOARD_SLOTS), np.bool_),
        "global_features": np.zeros((groups, width, GLOBAL_FEATURES), np.float32),
    }


def _assign_states(arrays: dict[str, np.ndarray], row: int, states: tuple[_State, ...]) -> None:
    for name in arrays:
        arrays[name][row, : len(states)] = np.stack([getattr(state, name) for state in states])


def _to_public_state(values: dict[str, np.ndarray]) -> R2MapPublicState:
    return R2MapPublicState(**{name: mx.array(value) for name, value in values.items()})


def _pack_one(
    refs: tuple[R2MapFrameRef, ...], order: tuple[int, ...], offset: int, groups: int, budget: int
) -> list[int]:
    selected: list[int] = []
    width = 0
    while offset + len(selected) < len(order) and len(selected) < groups:
        candidate = order[offset + len(selected)]
        next_width = max(width, refs[candidate].candidate_count)
        if next_width * (len(selected) + 1) > budget:
            if not selected:
                raise R2MapDatasetError("one R2-MAP group exceeds the padded candidate budget")
            break
        selected.append(candidate)
        width = next_width
    if not selected:
        raise R2MapDatasetError("R2-MAP training stream contains no frames")
    return selected


def _pack_all(refs: tuple[R2MapFrameRef, ...], order: tuple[int, ...], groups: int, budget: int):
    offset = 0
    while offset < len(order):
        selected = _pack_one(refs, order, offset, groups, budget)
        yield tuple(selected)
        offset += len(selected)


def _u16(payload: memoryview, offset: int) -> int:
    if offset + 2 > len(payload):
        raise R2MapDatasetError("R2-MAP payload is truncated")
    return struct.unpack_from("<H", payload, offset)[0]


def _array(payload: memoryview, offset: int, dtype: str, count: int) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    if count < 0 or offset + count * itemsize > len(payload):
        raise R2MapDatasetError("R2-MAP payload array is truncated")
    return np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
