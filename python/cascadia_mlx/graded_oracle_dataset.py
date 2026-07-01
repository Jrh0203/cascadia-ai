"""Streaming decoder for lossless complete-action graded-oracle datasets."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import (
    _RECORD_DTYPE,
    FEATURE_SCHEMA,
    Batch,
    DatasetError,
    _mask_bits,
    _one_hot,
    _one_hot_with_none,
    decode_market_entities,
    decode_records,
)
from cascadia_mlx.hex_symmetry import rotate_axial, rotate_one_hot, rotation_steps

GRADED_ORACLE_DATASET_SCHEMA_VERSION = 1
GRADED_ORACLE_FEATURE_SCHEMA = "complete-action-graded-oracle-v1"
GRADED_ORACLE_TARGET_SCHEMA = "screen-r600-r1200-r4800-graded-v1"
GRADED_ORACLE_SHARD_MAGIC = b"CSD2GOV\0"
GRADED_ORACLE_HEADER_SIZE = 112
GRADED_ORACLE_GROUP_HEADER_SIZE = 960
GRADED_ORACLE_ACTION_FEATURE_SIZE = 128
GRADED_ORACLE_CANDIDATE_RECORD_SIZE = 224
GRADED_ORACLE_MAX_WILDLIFE_WIPES = 20
GRADED_ORACLE_PUBLIC_SUPPLY_SIZE = 30
GRADED_ORACLE_ACTION_DIM = 140
GRADED_ORACLE_PRIOR_SCHEMA = "observable-screen-priors-v1"
GRADED_ORACLE_PRIOR_FEATURES = (
    "model_immediate_score",
    "model_remaining_value",
    "screen_value",
    "screen_rank_scaled",
    "screen_inverse_rank",
    "uniform_market_survival_proxy",
    "visible_wildlife_count",
    "public_bag_wildlife_count",
)
GRADED_ORACLE_PRIOR_DIM = len(GRADED_ORACLE_PRIOR_FEATURES)
GRADED_ORACLE_PACKED_ACTION_LIMIT = 8192
GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS = 16384

GRADED_ACTION_TILE_Q_INDEX = 34
GRADED_ACTION_TILE_R_INDEX = 35
GRADED_ACTION_ROTATION_SLICE = slice(36, 42)
GRADED_ACTION_WILDLIFE_Q_INDEX = 43
GRADED_ACTION_WILDLIFE_R_INDEX = 44

GRADED_SOURCE_R600 = 1 << 6
GRADED_SOURCE_R1200 = 1 << 7
GRADED_SOURCE_R4800 = 1 << 8
GRADED_SOURCE_COMPLETE_LEGAL = 1 << 9

GRADED_FIDELITY_R600 = 1 << 0
GRADED_FIDELITY_R1200 = 1 << 1
GRADED_FIDELITY_R4800 = 1 << 2

_PUBLIC_SUPPLY_SCALES = np.array(
    [20.0] * 5 + [81.0] * 25,
    dtype=np.float32,
)
_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")
_GROUP_HEADER_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "raw_seed",
            "candidate_count",
            "selected_index",
            "champion_index",
            "completed_turns",
            "current_player",
            "personal_turn",
            "phase",
            "public_state_hash",
            "position",
            "public_supply",
        ],
        "formats": [
            "<u8",
            "<u8",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "u1",
            "u1",
            "u1",
            ("u1", (32,)),
            _RECORD_DTYPE,
            ("u1", (GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,)),
        ],
        "offsets": [0, 8, 16, 18, 20, 22, 24, 25, 26, 28, 60, 924],
        "itemsize": GRADED_ORACLE_GROUP_HEADER_SIZE,
    }
)
_ESTIMATE_DTYPE = np.dtype(
    {
        "names": ["mean", "stddev", "samples"],
        "formats": ["<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8],
        "itemsize": 12,
    }
)
_ACTION_DTYPE = np.dtype(
    {
        "names": [
            "same_slot_independent",
            "draft_kind",
            "tile_slot",
            "wildlife_slot",
            "tile_id",
            "tile_terrain_a",
            "tile_terrain_b",
            "tile_wildlife_mask",
            "tile_keystone",
            "drafted_wildlife",
            "tile_q",
            "tile_r",
            "rotation",
            "wildlife_present",
            "wildlife_q",
            "wildlife_r",
            "replace_three_of_a_kind",
            "wipe_count",
            "wipe_masks",
            "staged_active_nature_tokens",
            "staged_market_entities",
            "staged_public_supply",
            "immediate_score",
            "immediate_deltas",
        ],
        "formats": [
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            ("u1", (GRADED_ORACLE_MAX_WILDLIFE_WIPES,)),
            "u1",
            ("u1", (4, 8)),
            ("u1", (GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,)),
            "<u2",
            ("<i2", (11,)),
        ],
        "offsets": [
            *range(18),
            18,
            38,
            42,
            74,
            104,
            106,
        ],
        "itemsize": GRADED_ORACLE_ACTION_FEATURE_SIZE,
    }
)
_CANDIDATE_DTYPE = np.dtype(
    {
        "names": [
            "action_hash",
            "canonical_index",
            "screen_rank",
            "source_flags",
            "fidelity_mask",
            "model_immediate_score",
            "model_remaining_value",
            "screen_value",
            "uniform_market_survival_proxy",
            "visible_wildlife_count",
            "public_bag_wildlife_count",
            "action",
            "r600",
            "r1200",
            "r4800",
        ],
        "formats": [
            ("u1", (32,)),
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "u1",
            "u1",
            _ACTION_DTYPE,
            _ESTIMATE_DTYPE,
            _ESTIMATE_DTYPE,
            _ESTIMATE_DTYPE,
        ],
        "offsets": [0, 32, 34, 36, 38, 40, 44, 48, 52, 56, 57, 60, 188, 200, 212],
        "itemsize": GRADED_ORACLE_CANDIDATE_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class GradedOracleBatch:
    """Padded complete decisions with explicit public action context."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    public_supply: mx.array
    action_features: mx.array
    prior_features: mx.array
    staged_market_entities: mx.array
    staged_market_mask: mx.array
    staged_public_supply: mx.array
    candidate_mask: mx.array
    model_immediate_score: mx.array
    model_remaining_value: mx.array
    screen_value: mx.array
    screen_rank: mx.array
    source_flags: mx.array
    fidelity_mask: mx.array
    r600_mean: mx.array
    r600_stddev: mx.array
    r600_samples: mx.array
    r600_mask: mx.array
    r1200_mean: mx.array
    r1200_stddev: mx.array
    r1200_samples: mx.array
    r1200_mask: mx.array
    r4800_mean: mx.array
    r4800_stddev: mx.array
    r4800_samples: mx.array
    r4800_mask: mx.array
    selected: mx.array
    champion: mx.array
    selected_index: mx.array
    champion_index: mx.array
    canonical_index: mx.array
    action_hash: np.ndarray
    public_state_hash: np.ndarray
    group_id: mx.array
    game_index: mx.array
    turn: mx.array
    current_player: mx.array
    personal_turn: mx.array
    phase: mx.array
    active_nature_tokens: mx.array
    same_slot_independent: mx.array
    draft_kind: mx.array
    replace_three_of_a_kind: mx.array
    wipe_count: mx.array


@dataclass(frozen=True)
class GradedOracleGroupRef:
    header_offset: int
    candidate_offset: int
    candidate_count: int


@dataclass(frozen=True)
class GradedOracleShard:
    path: Path
    record_count: int
    group_count: int
    game_count: int
    first_game_index: int
    groups: tuple[GradedOracleGroupRef, ...]

    def bytes(self) -> np.memmap:
        return np.memmap(self.path, mode="r", dtype=np.uint8)


@dataclass(frozen=True)
class GradedOracleGroupHeader:
    """Copied public group metadata without retaining candidate-record views."""

    group_id: int
    public_state_hash: np.ndarray
    candidate_count: int
    selected_index: int
    champion_index: int
    turn: int
    selected_draft_kind: int


@dataclass(frozen=True)
class GradedOracleGroupIdentity(GradedOracleGroupHeader):
    """Zero-copy action-identity view used only by exhaustive verification."""

    action_hashes: np.ndarray


class GradedOracleDataset:
    """Manifest-backed complete-action graded-oracle dataset."""

    DATASET_SCHEMA_VERSION = GRADED_ORACLE_DATASET_SCHEMA_VERSION
    TARGET_SCHEMA = GRADED_ORACLE_TARGET_SCHEMA
    SHARD_MAGIC = GRADED_ORACLE_SHARD_MAGIC

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "dataset.json"
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read graded-oracle manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def group_count(self) -> int:
        return int(self.manifest["total_groups"])

    @property
    def candidate_count(self) -> int:
        return int(self.manifest["total_records"])

    def batches(
        self,
        group_batch_size: int,
        *,
        maximum_actions_per_batch: int | None = None,
        maximum_group_actions: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[GradedOracleBatch]:
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        if maximum_actions_per_batch is not None and maximum_actions_per_batch <= 0:
            raise ValueError("maximum_actions_per_batch must be positive")
        if maximum_group_actions is not None and maximum_group_actions <= 0:
            raise ValueError("maximum_group_actions must be positive")
        rng = np.random.default_rng(seed)
        shard_indices = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            shard = self.shards[int(shard_index)]
            group_indices = np.arange(len(shard.groups))
            if shuffle:
                rng.shuffle(group_indices)
            raw = shard.bytes()
            refs = tuple(shard.groups[int(index)] for index in group_indices)
            for batch_refs in _pack_groups(
                refs,
                group_batch_size,
                maximum_actions_per_batch,
                maximum_group_actions,
            ):
                yield decode_graded_oracle_groups(raw, batch_refs)

    def raw_groups(
        self,
    ) -> Iterator[tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupIdentity]]:
        """Iterate exact group identities without decoding model features."""
        for raw, ref, header in self.raw_group_headers():
            candidates = inspect_graded_oracle_candidate_records(raw, ref)
            yield (
                raw,
                ref,
                GradedOracleGroupIdentity(
                    group_id=header.group_id,
                    public_state_hash=header.public_state_hash,
                    candidate_count=header.candidate_count,
                    selected_index=header.selected_index,
                    champion_index=header.champion_index,
                    turn=header.turn,
                    selected_draft_kind=header.selected_draft_kind,
                    action_hashes=np.asarray(candidates["action_hash"], dtype=np.uint8),
                ),
            )

    def raw_group_headers(
        self,
    ) -> Iterator[tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupHeader]]:
        """Iterate copied group metadata while touching one selected action only."""
        for shard in self.shards:
            raw = shard.bytes()
            for ref in shard.groups:
                yield raw, ref, inspect_graded_oracle_group_header(raw, ref)

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("schema_version") != self.DATASET_SCHEMA_VERSION:
            raise DatasetError("unsupported graded-oracle dataset schema version")
        if manifest.get("feature_schema") != GRADED_ORACLE_FEATURE_SCHEMA:
            raise DatasetError("unsupported graded-oracle feature schema")
        if manifest.get("position_feature_schema") != FEATURE_SCHEMA:
            raise DatasetError("unsupported graded-oracle position feature schema")
        if manifest.get("target_schema") != self.TARGET_SCHEMA:
            raise DatasetError("unsupported graded-oracle target schema")
        if manifest.get("group_header_size") != GRADED_ORACLE_GROUP_HEADER_SIZE:
            raise DatasetError("unsupported graded-oracle group header size")
        if manifest.get("candidate_record_size") != GRADED_ORACLE_CANDIDATE_RECORD_SIZE:
            raise DatasetError("unsupported graded-oracle candidate record size")
        if manifest.get("action_feature_size") != GRADED_ORACLE_ACTION_FEATURE_SIZE:
            raise DatasetError("unsupported graded-oracle action feature size")
        if manifest.get("public_supply_size") != GRADED_ORACLE_PUBLIC_SUPPLY_SIZE:
            raise DatasetError("unsupported graded-oracle public-supply size")
        if manifest.get("maximum_wildlife_wipes") != GRADED_ORACLE_MAX_WILDLIFE_WIPES:
            raise DatasetError("unsupported graded-oracle wipe capacity")
        if not isinstance(manifest.get("teacher"), dict):
            raise DatasetError("graded-oracle manifest requires frozen teacher identity")
        seeds = manifest.get("seeds")
        if not isinstance(seeds, list) or seeds != sorted(set(seeds)):
            raise DatasetError("graded-oracle manifest seeds are invalid")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("graded-oracle manifest shards must be a list")
        if sum(int(shard["record_count"]) for shard in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise DatasetError("graded-oracle manifest record total does not match shards")
        if sum(int(shard["group_count"]) for shard in shards) != int(
            manifest.get("total_groups", -1)
        ):
            raise DatasetError("graded-oracle manifest group total does not match shards")
        if sum(int(shard["game_count"]) for shard in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise DatasetError("graded-oracle manifest game total does not match shards")
        if [int(shard["first_game_index"]) for shard in shards] != seeds[: len(shards)]:
            raise DatasetError("graded-oracle shard seeds do not match the manifest")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> GradedOracleShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            header = path.read_bytes()[:GRADED_ORACLE_HEADER_SIZE]
        except OSError as error:
            raise DatasetError(f"cannot read graded-oracle shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"graded-oracle shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"graded-oracle shard checksum mismatch: {path}")
        if len(header) != GRADED_ORACLE_HEADER_SIZE:
            raise DatasetError(f"truncated graded-oracle shard header: {path}")
        (
            magic,
            schema,
            header_size,
            group_header_size,
            candidate_record_size,
            record_count,
            group_count,
            game_count,
            _split,
            _players,
            _bonuses,
            _reserved_byte,
            first_game_index,
            feature_hash,
            target_hash,
            _reserved_tail,
        ) = _HEADER.unpack(header)
        if (
            magic != self.SHARD_MAGIC
            or schema != self.DATASET_SCHEMA_VERSION
            or header_size != GRADED_ORACLE_HEADER_SIZE
            or group_header_size != GRADED_ORACLE_GROUP_HEADER_SIZE
            or candidate_record_size != GRADED_ORACLE_CANDIDATE_RECORD_SIZE
            or feature_hash != blake3.blake3(GRADED_ORACLE_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(self.TARGET_SCHEMA.encode()).digest()
        ):
            raise DatasetError(f"incompatible graded-oracle shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"graded-oracle shard header disagrees with manifest: {path}")

        raw = np.memmap(path, mode="r", dtype=np.uint8)
        groups: list[GradedOracleGroupRef] = []
        offset = GRADED_ORACLE_HEADER_SIZE
        observed_records = 0
        for _ in range(group_count):
            if offset + GRADED_ORACLE_GROUP_HEADER_SIZE > stat.st_size:
                raise DatasetError(f"truncated graded-oracle group header: {path}")
            header_record = np.frombuffer(
                raw,
                dtype=_GROUP_HEADER_DTYPE,
                count=1,
                offset=offset,
            )[0]
            candidate_count = int(header_record["candidate_count"])
            if candidate_count < 2:
                raise DatasetError(f"invalid graded-oracle group width: {path}")
            candidate_offset = offset + GRADED_ORACLE_GROUP_HEADER_SIZE
            groups.append(
                GradedOracleGroupRef(
                    header_offset=offset,
                    candidate_offset=candidate_offset,
                    candidate_count=candidate_count,
                )
            )
            observed_records += candidate_count
            offset = candidate_offset + candidate_count * GRADED_ORACLE_CANDIDATE_RECORD_SIZE
        if offset != stat.st_size or observed_records != record_count:
            raise DatasetError(f"graded-oracle shard framing mismatch: {path}")
        return GradedOracleShard(
            path=path,
            record_count=record_count,
            group_count=group_count,
            game_count=game_count,
            first_game_index=first_game_index,
            groups=tuple(groups),
        )


def inspect_graded_oracle_candidate_records(
    raw: np.ndarray,
    ref: GradedOracleGroupRef,
) -> np.ndarray:
    """Return a zero-copy, read-only view of one group's structured candidates."""
    candidates = np.frombuffer(
        raw,
        dtype=_CANDIDATE_DTYPE,
        count=ref.candidate_count,
        offset=ref.candidate_offset,
    )
    candidates.flags.writeable = False
    return candidates


def decode_graded_oracle_groups(
    raw: np.ndarray,
    refs: tuple[GradedOracleGroupRef, ...],
    *,
    candidate_indices: Sequence[Sequence[int] | np.ndarray] | None = None,
    require_selected_action: bool = True,
    require_champion_action: bool = True,
) -> GradedOracleBatch:
    """Decode and pad a set of complete legal decision groups."""
    if not refs:
        raise ValueError("graded-oracle batch must contain at least one group")
    if candidate_indices is not None and len(candidate_indices) != len(refs):
        raise ValueError("candidate index selections must align with group refs")
    selections: list[np.ndarray | None] = []
    for group_index, ref in enumerate(refs):
        if candidate_indices is None:
            selections.append(None)
            continue
        selected = np.asarray(candidate_indices[group_index], dtype=np.int64)
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= ref.candidate_count)
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError(
                "candidate selections must be nonempty, strictly increasing, and in range"
            )
        selections.append(selected)
    group_count = len(refs)
    max_candidates = max(
        ref.candidate_count if selected is None else len(selected)
        for ref, selected in zip(refs, selections, strict=True)
    )
    positions = np.zeros(group_count, dtype=_RECORD_DTYPE)
    public_supply = np.zeros(
        (group_count, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE),
        dtype=np.float32,
    )
    candidates = np.zeros((group_count, max_candidates), dtype=_CANDIDATE_DTYPE)
    candidate_mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    selected = np.zeros((group_count, max_candidates), dtype=np.bool_)
    champion = np.zeros((group_count, max_candidates), dtype=np.bool_)
    selected_index = np.zeros(group_count, dtype=np.int32)
    champion_index = np.zeros(group_count, dtype=np.int32)
    group_id = np.zeros(group_count, dtype=np.uint64)
    current_player = np.zeros(group_count, dtype=np.int32)
    personal_turn = np.zeros(group_count, dtype=np.int32)
    phase = np.zeros(group_count, dtype=np.int32)
    public_state_hash = np.zeros((group_count, 32), dtype=np.uint8)

    for group_index, (ref, source_selection) in enumerate(zip(refs, selections, strict=True)):
        header = np.frombuffer(
            raw,
            dtype=_GROUP_HEADER_DTYPE,
            count=1,
            offset=ref.header_offset,
        )[0]
        full_group_candidates = np.frombuffer(
            raw,
            dtype=_CANDIDATE_DTYPE,
            count=ref.candidate_count,
            offset=ref.candidate_offset,
        )
        source_selected = int(header["selected_index"])
        source_champion = int(header["champion_index"])
        if source_selected >= ref.candidate_count or source_champion >= ref.candidate_count:
            raise DatasetError("graded-oracle selected or champion index is invalid")
        if source_selection is None:
            group_candidates = full_group_candidates
            selected_value = source_selected
            champion_value = source_champion
        else:
            selected_matches = np.flatnonzero(source_selection == source_selected)
            champion_matches = np.flatnonzero(source_selection == source_champion)
            if len(selected_matches) > 1 or len(champion_matches) > 1:
                raise DatasetError("candidate subset duplicated the selected or champion action")
            if require_selected_action and len(selected_matches) != 1:
                raise DatasetError("candidate subset omitted the selected action")
            if require_champion_action and len(champion_matches) != 1:
                raise DatasetError("candidate subset omitted the champion action")
            group_candidates = full_group_candidates[source_selection]
            selected_value = int(selected_matches[0]) if len(selected_matches) else -1
            champion_value = int(champion_matches[0]) if len(champion_matches) else -1
        count = len(group_candidates)
        positions[group_index] = header["position"]
        public_supply[group_index] = (
            header["public_supply"].astype(np.float32) / _PUBLIC_SUPPLY_SCALES
        )
        candidates[group_index, :count] = group_candidates
        candidate_mask[group_index, :count] = True
        if selected_value >= 0:
            selected[group_index, selected_value] = True
        if champion_value >= 0:
            champion[group_index, champion_value] = True
        selected_index[group_index] = selected_value
        champion_index[group_index] = champion_value
        group_id[group_index] = header["group_id"]
        current_player[group_index] = header["current_player"]
        personal_turn[group_index] = header["personal_turn"]
        phase[group_index] = header["phase"]
        public_state_hash[group_index] = header["public_state_hash"]

    decoded: Batch = decode_records(positions)
    raw_actions = candidates["action"]
    action_features = decode_graded_action_features(raw_actions)
    action_features *= candidate_mask[..., None]
    prior_features = decode_graded_prior_features(candidates)
    prior_features *= candidate_mask[..., None]
    staged_market_entities, staged_market_mask = decode_market_entities(
        raw_actions["staged_market_entities"]
    )
    staged_market_entities *= candidate_mask[..., None, None]
    staged_market_mask &= candidate_mask[..., None]
    staged_public_supply = (
        raw_actions["staged_public_supply"].astype(np.float32) / _PUBLIC_SUPPLY_SCALES
    )
    staged_public_supply *= candidate_mask[..., None]

    fidelity: dict[str, dict[str, np.ndarray]] = {}
    for tier in ("r600", "r1200", "r4800"):
        samples = candidates[tier]["samples"].astype(np.float32)
        fidelity[tier] = {
            "mean": candidates[tier]["mean"].astype(np.float32),
            "stddev": candidates[tier]["stddev"].astype(np.float32),
            "samples": samples,
            "mask": (samples > 0) & candidate_mask,
        }

    return GradedOracleBatch(
        board_entities=decoded.board_entities,
        board_mask=decoded.board_mask,
        market_entities=decoded.market_entities,
        market_mask=decoded.market_mask,
        global_features=decoded.global_features,
        public_supply=mx.array(public_supply),
        action_features=mx.array(action_features),
        prior_features=mx.array(prior_features),
        staged_market_entities=mx.array(staged_market_entities),
        staged_market_mask=mx.array(staged_market_mask),
        staged_public_supply=mx.array(staged_public_supply),
        candidate_mask=mx.array(candidate_mask),
        model_immediate_score=mx.array(candidates["model_immediate_score"].astype(np.float32)),
        model_remaining_value=mx.array(candidates["model_remaining_value"].astype(np.float32)),
        screen_value=mx.array(candidates["screen_value"].astype(np.float32)),
        screen_rank=mx.array(candidates["screen_rank"].astype(np.float32)),
        source_flags=mx.array(candidates["source_flags"].astype(np.int32)),
        fidelity_mask=mx.array(candidates["fidelity_mask"].astype(np.int32)),
        r600_mean=mx.array(fidelity["r600"]["mean"]),
        r600_stddev=mx.array(fidelity["r600"]["stddev"]),
        r600_samples=mx.array(fidelity["r600"]["samples"]),
        r600_mask=mx.array(fidelity["r600"]["mask"]),
        r1200_mean=mx.array(fidelity["r1200"]["mean"]),
        r1200_stddev=mx.array(fidelity["r1200"]["stddev"]),
        r1200_samples=mx.array(fidelity["r1200"]["samples"]),
        r1200_mask=mx.array(fidelity["r1200"]["mask"]),
        r4800_mean=mx.array(fidelity["r4800"]["mean"]),
        r4800_stddev=mx.array(fidelity["r4800"]["stddev"]),
        r4800_samples=mx.array(fidelity["r4800"]["samples"]),
        r4800_mask=mx.array(fidelity["r4800"]["mask"]),
        selected=mx.array(selected),
        champion=mx.array(champion),
        selected_index=mx.array(selected_index),
        champion_index=mx.array(champion_index),
        canonical_index=mx.array(candidates["canonical_index"].astype(np.int32)),
        action_hash=candidates["action_hash"].copy(),
        public_state_hash=public_state_hash,
        group_id=mx.array(group_id.astype(np.int64)),
        game_index=decoded.game_index,
        turn=decoded.turn,
        current_player=mx.array(current_player),
        personal_turn=mx.array(personal_turn),
        phase=mx.array(phase),
        active_nature_tokens=mx.array(positions["nature_tokens"][:, 0].astype(np.int32)),
        same_slot_independent=mx.array(raw_actions["same_slot_independent"].astype(np.int32)),
        draft_kind=mx.array(raw_actions["draft_kind"].astype(np.int32)),
        replace_three_of_a_kind=mx.array(raw_actions["replace_three_of_a_kind"].astype(np.int32)),
        wipe_count=mx.array(raw_actions["wipe_count"].astype(np.int32)),
    )


def inspect_graded_oracle_group(
    raw: np.ndarray,
    ref: GradedOracleGroupRef,
) -> GradedOracleGroupIdentity:
    """Read only identity fields and action hashes for one source group."""
    header = inspect_graded_oracle_group_header(raw, ref)
    candidates = np.frombuffer(
        raw,
        dtype=_CANDIDATE_DTYPE,
        count=ref.candidate_count,
        offset=ref.candidate_offset,
    )
    return GradedOracleGroupIdentity(
        group_id=header.group_id,
        public_state_hash=header.public_state_hash,
        candidate_count=header.candidate_count,
        selected_index=header.selected_index,
        champion_index=header.champion_index,
        turn=header.turn,
        selected_draft_kind=header.selected_draft_kind,
        action_hashes=np.asarray(candidates["action_hash"], dtype=np.uint8),
    )


def inspect_graded_oracle_group_header(
    raw: np.ndarray,
    ref: GradedOracleGroupRef,
) -> GradedOracleGroupHeader:
    """Read copied group metadata without walking the complete candidate body."""
    header = np.frombuffer(
        raw,
        dtype=_GROUP_HEADER_DTYPE,
        count=1,
        offset=ref.header_offset,
    )[0]
    selected_index = int(header["selected_index"])
    champion_index = int(header["champion_index"])
    if selected_index >= ref.candidate_count or champion_index >= ref.candidate_count:
        raise DatasetError("graded-oracle selected or champion index is invalid")
    selected = np.frombuffer(
        raw,
        dtype=_CANDIDATE_DTYPE,
        count=1,
        offset=(ref.candidate_offset + selected_index * GRADED_ORACLE_CANDIDATE_RECORD_SIZE),
    )[0]
    return GradedOracleGroupHeader(
        group_id=int(header["group_id"]),
        public_state_hash=np.asarray(header["public_state_hash"], dtype=np.uint8).copy(),
        candidate_count=ref.candidate_count,
        selected_index=selected_index,
        champion_index=champion_index,
        turn=int(header["position"]["turn"]),
        selected_draft_kind=int(selected["action"]["draft_kind"]),
    )


def decode_graded_action_features(actions: np.ndarray) -> np.ndarray:
    """Vectorize lossless action rows without teacher labels."""
    actions = np.asarray(actions)
    presence = actions["wildlife_present"].astype(np.float32)[..., None]
    wipe_bits = _mask_bits(actions["wipe_masks"], 4).reshape(
        *actions.shape,
        GRADED_ORACLE_MAX_WILDLIFE_WIPES * 4,
    )
    features = np.concatenate(
        [
            actions["same_slot_independent"].astype(np.float32)[..., None],
            _one_hot(actions["draft_kind"], 2),
            _one_hot(actions["tile_slot"], 4),
            _one_hot(actions["wildlife_slot"], 4),
            actions["tile_id"].astype(np.float32)[..., None] / 84.0,
            _one_hot(actions["tile_terrain_a"], 5),
            _one_hot_with_none(actions["tile_terrain_b"], 5),
            _mask_bits(actions["tile_wildlife_mask"], 5),
            actions["tile_keystone"].astype(np.float32)[..., None],
            _one_hot(actions["drafted_wildlife"], 5),
            actions["tile_q"].astype(np.float32)[..., None] / 24.0,
            actions["tile_r"].astype(np.float32)[..., None] / 24.0,
            _one_hot(actions["rotation"], 6),
            presence,
            actions["wildlife_q"].astype(np.float32)[..., None] / 24.0 * presence,
            actions["wildlife_r"].astype(np.float32)[..., None] / 24.0 * presence,
            actions["replace_three_of_a_kind"].astype(np.float32)[..., None],
            actions["wipe_count"].astype(np.float32)[..., None] / GRADED_ORACLE_MAX_WILDLIFE_WIPES,
            wipe_bits,
            actions["staged_active_nature_tokens"].astype(np.float32)[..., None] / 20.0,
            actions["immediate_score"].astype(np.float32)[..., None] / 100.0,
            actions["immediate_deltas"].astype(np.float32) / 20.0,
        ],
        axis=-1,
    )
    if features.shape[-1] != GRADED_ORACLE_ACTION_DIM:
        raise AssertionError("graded-oracle action dimension drifted")
    return features


def decode_graded_action_feature_bytes(raw_actions: np.ndarray) -> np.ndarray:
    """Decode canonical Rust 128-byte action rows into the frozen 140-vector."""
    raw = np.ascontiguousarray(raw_actions, dtype=np.uint8)
    if raw.ndim < 2 or raw.shape[-1] != GRADED_ORACLE_ACTION_FEATURE_SIZE:
        raise ValueError("graded-oracle raw action rows must end in 128 bytes")
    actions = raw.view(_ACTION_DTYPE).reshape(raw.shape[:-1])
    return decode_graded_action_features(actions)


def decode_graded_prior_features(candidates: np.ndarray) -> np.ndarray:
    """Decode only screen priors that are computable during live play."""
    candidates = np.asarray(candidates)
    rank = candidates["screen_rank"].astype(np.float32)
    features = np.concatenate(
        [
            candidates["model_immediate_score"].astype(np.float32)[..., None] / 100.0,
            candidates["model_remaining_value"].astype(np.float32)[..., None] / 100.0,
            candidates["screen_value"].astype(np.float32)[..., None] / 100.0,
            rank[..., None] / 4096.0,
            1.0 / np.maximum(rank[..., None], 1.0),
            candidates["uniform_market_survival_proxy"].astype(np.float32)[..., None],
            candidates["visible_wildlife_count"].astype(np.float32)[..., None] / 4.0,
            candidates["public_bag_wildlife_count"].astype(np.float32)[..., None] / 20.0,
        ],
        axis=-1,
    )
    if features.shape[-1] != GRADED_ORACLE_PRIOR_DIM:
        raise AssertionError("graded-oracle prior dimension drifted")
    return features


def rotate_graded_oracle_batch(
    batch: GradedOracleBatch,
    rotations: int | Sequence[int],
) -> GradedOracleBatch:
    """Rotate every board and complete action in a group by the same symmetry."""
    groups = batch.action_features.shape[0]
    steps = rotation_steps(rotations, groups)

    boards = batch.board_entities
    board_q, board_r = rotate_axial(boards[..., 0], boards[..., 1], steps)
    board_rotations = rotate_one_hot(
        boards[..., 13:19],
        steps,
        batch.board_mask,
    )
    rotated_boards = mx.concatenate(
        [
            board_q[..., None],
            board_r[..., None],
            boards[..., 2:13],
            board_rotations,
            boards[..., 19:],
        ],
        axis=-1,
    )

    actions = batch.action_features
    tile_q, tile_r = rotate_axial(
        actions[..., GRADED_ACTION_TILE_Q_INDEX],
        actions[..., GRADED_ACTION_TILE_R_INDEX],
        steps,
    )
    wildlife_q, wildlife_r = rotate_axial(
        actions[..., GRADED_ACTION_WILDLIFE_Q_INDEX],
        actions[..., GRADED_ACTION_WILDLIFE_R_INDEX],
        steps,
    )
    action_rotations = rotate_one_hot(
        actions[..., GRADED_ACTION_ROTATION_SLICE],
        steps,
        batch.candidate_mask,
    )
    rotated_actions = mx.concatenate(
        [
            actions[..., :GRADED_ACTION_TILE_Q_INDEX],
            tile_q[..., None],
            tile_r[..., None],
            action_rotations,
            actions[..., 42:GRADED_ACTION_WILDLIFE_Q_INDEX],
            wildlife_q[..., None],
            wildlife_r[..., None],
            actions[..., 45:],
        ],
        axis=-1,
    )
    return replace(
        batch,
        board_entities=rotated_boards,
        action_features=rotated_actions,
    )


def randomly_rotate_graded_oracle_batch(
    batch: GradedOracleBatch,
    seed: int,
) -> GradedOracleBatch:
    """Sample one exact, uniform hex rotation per complete decision group."""
    rng = np.random.default_rng(seed)
    return rotate_graded_oracle_batch(
        batch,
        rng.integers(0, 6, size=batch.action_features.shape[0]),
    )


def _pack_groups(
    refs: tuple[GradedOracleGroupRef, ...],
    group_batch_size: int,
    maximum_actions_per_batch: int | None,
    maximum_group_actions: int | None = None,
) -> Iterator[tuple[GradedOracleGroupRef, ...]]:
    batch: list[GradedOracleGroupRef] = []
    maximum_width = 0
    for ref in refs:
        if maximum_group_actions is not None and ref.candidate_count > maximum_group_actions:
            raise DatasetError("one graded-oracle group exceeds maximum_group_actions")
        next_width = max(maximum_width, ref.candidate_count)
        padded_actions = next_width * (len(batch) + 1)
        if batch and (
            len(batch) >= group_batch_size
            or (
                maximum_actions_per_batch is not None and padded_actions > maximum_actions_per_batch
            )
        ):
            yield tuple(batch)
            batch = []
            maximum_width = 0
        batch.append(ref)
        maximum_width = max(maximum_width, ref.candidate_count)
    if batch:
        yield tuple(batch)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
