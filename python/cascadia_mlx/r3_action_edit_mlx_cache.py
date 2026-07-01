"""Fail-closed R3 action-edit cache and graded-oracle binding for MLX."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GradedOracleBatch,
    GradedOracleDataset,
    GradedOracleGroupHeader,
    GradedOracleGroupIdentity,
    GradedOracleGroupRef,
    decode_graded_oracle_groups,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
    TOKEN_PAYLOAD_WIDTH,
    _layout_from_board_type_counts,
    _materialize_token_features,
    _transform_payload_in_place,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    RELATIONAL_ARM,
    S1ExactSupplyCache,
    transform_s1_exact_supply_batch,
)

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "r3-action-edit-mlx-cache-v1"
EXPERIMENT_ID = "r3-action-edit-mlx-comparison-v1"
PROTOCOL_ID = "r3-action-edit-mlx-matched-comparison-v1"
ADR_ID = "0150"

ARMS = (
    "c0-full-r2-afterstate",
    "t1-r3-radius3-global",
    "t2-r3-radius2-global",
    "t3-r3-radius1-global",
)
ARM_RADII = {
    ARMS[1]: 3,
    ARMS[2]: 2,
    ARMS[3]: 1,
}
CONTROL_ARM = ARMS[0]
CONTROL_MATERIALIZATION_VERIFIED = "verified-per-candidate"
CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED = "preverified-vectorized"
CONTROL_MATERIALIZATIONS = (
    CONTROL_MATERIALIZATION_VERIFIED,
    CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
)

R3_TOKEN_TYPE_COUNT = 10
R3_OPERATION_COUNT = 6
R3_TOKEN_PAYLOAD_WIDTH = 64
R3_TOKEN_FEATURES = R3_TOKEN_TYPE_COUNT + R3_OPERATION_COUNT + R3_TOKEN_PAYLOAD_WIDTH
R3_LOCAL_PATCH_TOKEN = 2
R3_CONTROL_OPERATION = 5
R3_SCHEMA_VERSION = 1
D6_TRANSFORMS = 12

EXPECTED_SPLIT_COUNTS = {
    "train": (560, 2_135_111),
    "validation": (240, 860_203),
}
EXPECTED_TENSOR_CONTRACT = {
    "parent": {
        "boards": BOARD_SLOTS,
        "tokens_per_board": BOARD_TOKEN_CAPACITY,
        "token_payload_width": TOKEN_PAYLOAD_WIDTH,
        "market_feature_dim": MARKET_FEATURES,
        "player_feature_dim": PLAYER_FEATURES,
        "global_feature_dim": GLOBAL_FEATURES,
        "one_parent_encoding_per_group": True,
    },
    "candidate": {
        "train_candidate_cap": 512,
        "validation_is_complete": True,
        "control": "canonical-parent-multiset-removals-plus-exact-additions",
        "r3_payload_width": R3_TOKEN_PAYLOAD_WIDTH,
        "r3_radius_three_cached_once": True,
        "radius_one_and_two": "exact-loader-crop-of-local-patch-tokens",
        "silent_truncation": False,
    },
}
EXPECTED_HIDDEN_BOUNDARY = {
    "open_train_and_validation_only": True,
    "source_seed_used_for_authoritative_replay": True,
    "hidden_order_exported": False,
    "excluded_tile_identity_exported": False,
    "future_refill_exported": False,
    "sealed_test_opened": False,
    "gameplay_opened": False,
}

_DTYPES = {
    "|i1": np.dtype("i1"),
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<u8": np.dtype("<u8"),
    "<f4": np.dtype("<f4"),
}
_CONTROL_TYPE_MAP = np.asarray([0, 3, 4, 6, 7], dtype=np.uint8)
_R3_TOKEN_DOMAIN = b"r3-mlx-action-token-stream-v1"
_CANDIDATE_IDENTITY_DOMAIN = b"r3-mlx-candidate-identity-v1"
_COHORT_DOMAIN = b"r3-mlx-train-cohort-v1"
_CONTROL_TOKEN_DOMAIN = b"r3-mlx-control-token-multiset-v1"
_TRAINING_SCHEDULE_DOMAIN = b"r3-mlx-training-schedule-v1"
LOW_SUPPLY_MAX_UNSEEN = 20


class R3ActionEditMlxCacheError(ValueError):
    """The R3 cache or one of its bound public datasets is inconsistent."""


@dataclass(frozen=True)
class R3ParentBatch:
    """One exact R2 Perceiver input per decision group."""

    token_features: mx.array
    token_types: mx.array
    token_mask: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array
    transform_ids: mx.array


@dataclass(frozen=True)
class R3ActionEditBatch:
    """Shared parent/factual inputs plus one arm-routed candidate token surface."""

    base: GradedOracleBatch
    supply_vector: mx.array
    staged_supply_vector: mx.array
    supply_tokens: mx.array
    supply_mask: mx.array
    refill_target: mx.array
    selected_archetype: mx.array
    frontier_features: mx.array
    parent: R3ParentBatch
    candidate_token_features: mx.array
    candidate_token_mask: mx.array
    candidate_token_counts: mx.array
    source_candidate_indices: mx.array
    canonical_transform_ids: mx.array
    arm: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


@dataclass(frozen=True)
class _Tensor:
    path: Path
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    blake3: str

    def memmap(self) -> np.memmap:
        return np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape)


@dataclass(frozen=True)
class _Split:
    groups: int
    source_candidates: int
    retained_candidates: int
    tensors: dict[str, np.memmap]
    group_rows: dict[int, int]


@dataclass(frozen=True)
class _GroupLocation:
    raw: np.memmap
    ref: GradedOracleGroupRef
    identity: GradedOracleGroupHeader


class R3ActionEditMlxCache:
    """Content-addressed R3 sidecar with exact R2-control reconstruction."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.manifest = _read_json(self.root / "cache.json", "R3 cache manifest")
        self._validate_envelope(require_complete=require_complete)
        self.splits = {
            split: self._load_split(split, verify_checksums=verify_checksums)
            for split in EXPECTED_SPLIT_COUNTS
        }
        if verify_semantics:
            for split in EXPECTED_SPLIT_COUNTS:
                self._verify_split_semantics(split)

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def bind_dataset(
        self,
        root: str | Path,
        *,
        s1_cache: S1ExactSupplyCache,
        verify_dataset_checksums: bool = True,
        preverified_open_data_proof_id: str | None = None,
    ) -> R3ActionEditMlxDataset:
        return R3ActionEditMlxDataset(
            root,
            cache=self,
            s1_cache=s1_cache,
            verify_dataset_checksums=verify_dataset_checksums,
            preverified_open_data_proof_id=preverified_open_data_proof_id,
        )

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
        ):
            raise R3ActionEditMlxCacheError("unsupported R3 action-edit cache envelope")
        identity = manifest.get("scientific_identity")
        if (
            not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise R3ActionEditMlxCacheError(
                "R3 cache content address or directory identity is invalid"
            )
        if manifest.get("tensor_contract") != EXPECTED_TENSOR_CONTRACT:
            raise R3ActionEditMlxCacheError("R3 tensor contract drifted")
        if manifest.get("hidden_information") != EXPECTED_HIDDEN_BOUNDARY:
            raise R3ActionEditMlxCacheError("R3 hidden-information boundary drifted")
        exporter = manifest.get("exporter")
        if not isinstance(exporter, dict):
            raise R3ActionEditMlxCacheError("R3 exporter identity is absent")
        _require_blake3(exporter.get("executable_blake3"), "R3 exporter executable")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise R3ActionEditMlxCacheError(
                "production R3 cache must cover the complete open corpus"
            )
        raw_splits = manifest.get("splits")
        if not isinstance(raw_splits, dict) or set(raw_splits) != set(EXPECTED_SPLIT_COUNTS):
            raise R3ActionEditMlxCacheError("R3 cache split manifests are incomplete")
        if identity.get("splits") != raw_splits:
            raise R3ActionEditMlxCacheError(
                "R3 scientific identity does not bind its split manifests"
            )

    def _load_split(self, split: str, *, verify_checksums: bool) -> _Split:
        raw = self.manifest["splits"][split]
        if not isinstance(raw, dict):
            raise R3ActionEditMlxCacheError(f"{split} R3 split manifest is malformed")
        groups = int(raw.get("groups", -1))
        source_candidates = int(raw.get("source_candidates", -1))
        retained_candidates = int(raw.get("retained_candidates", -1))
        if groups <= 0 or source_candidates <= 0 or retained_candidates <= 0:
            raise R3ActionEditMlxCacheError(f"{split} R3 split counts are invalid")
        if (
            raw.get("complete_open_split") is True
            and (groups, source_candidates) != EXPECTED_SPLIT_COUNTS[split]
        ):
            raise R3ActionEditMlxCacheError(
                f"{split} claims complete coverage with incorrect counts"
            )
        expected_shapes = _expected_shapes(
            groups=groups,
            candidates=retained_candidates,
            removed=int(raw.get("control_removed_tokens", -1)),
            added=int(raw.get("control_added_tokens", -1)),
            r3_tokens=int(raw.get("r3_tokens", -1)),
        )
        raw_files = raw.get("files")
        if not isinstance(raw_files, dict) or set(raw_files) != set(expected_shapes):
            raise R3ActionEditMlxCacheError(f"{split} R3 tensor set drifted")
        tensors: dict[str, np.memmap] = {}
        for name, shape in expected_shapes.items():
            tensor = self._tensor(raw_files[name], shape)
            if verify_checksums and _checksum(tensor.path) != tensor.blake3:
                raise R3ActionEditMlxCacheError(f"{split} R3 tensor checksum mismatch: {name}")
            tensors[name] = tensor.memmap()
        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        if len(set(int(value) for value in group_ids)) != groups:
            raise R3ActionEditMlxCacheError(f"{split} R3 group IDs are duplicated")
        return _Split(
            groups=groups,
            source_candidates=source_candidates,
            retained_candidates=retained_candidates,
            tensors=tensors,
            group_rows={int(group_id): row for row, group_id in enumerate(group_ids)},
        )

    def _tensor(self, raw: object, expected_shape: tuple[int, ...]) -> _Tensor:
        if not isinstance(raw, dict) or raw.get("dtype") not in _DTYPES:
            raise R3ActionEditMlxCacheError("R3 tensor specification is malformed")
        if raw.get("shape") != list(expected_shape):
            raise R3ActionEditMlxCacheError("R3 tensor shape drifted")
        path = self.root / str(raw.get("file"))
        if path.parent != self.root or not path.is_file():
            raise R3ActionEditMlxCacheError("R3 tensor path escapes or is absent")
        dtype = _DTYPES[str(raw["dtype"])]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        if raw.get("bytes") != expected_bytes or path.stat().st_size != expected_bytes:
            raise R3ActionEditMlxCacheError("R3 tensor byte count drifted")
        digest = raw.get("blake3")
        _require_blake3(digest, "R3 tensor")
        return _Tensor(
            path=path,
            dtype=dtype,
            shape=expected_shape,
            blake3=str(digest),
        )

    def _verify_split_semantics(self, split: str) -> None:
        source = self.splits[split]
        tensors = source.tensors
        group_offsets = np.asarray(tensors["candidate_offsets"], dtype=np.uint64)
        retained_counts = np.asarray(tensors["retained_candidate_counts"], dtype=np.int64)
        source_counts = np.asarray(tensors["source_candidate_counts"], dtype=np.int64)
        if (
            group_offsets[0] != 0
            or group_offsets[-1] != source.retained_candidates
            or np.any(np.diff(group_offsets) != retained_counts)
            or int(source_counts.sum()) != source.source_candidates
            or np.any(retained_counts <= 0)
            or np.any(retained_counts > source_counts)
        ):
            raise R3ActionEditMlxCacheError(f"{split} R3 candidate offset accounting drifted")
        if split == "train" and np.any(retained_counts > 512):
            raise R3ActionEditMlxCacheError("R3 train cohort exceeds 512 actions")
        if split == "validation" and np.any(retained_counts != source_counts):
            raise R3ActionEditMlxCacheError("R3 validation cache is not complete at every decision")

        for offsets_name, terminal in (
            ("control_remove_offsets", tensors["control_remove_indices"].shape[0]),
            ("control_add_offsets", tensors["control_add_types"].shape[0]),
            ("r3_token_offsets", tensors["r3_token_types"].shape[0]),
        ):
            offsets = np.asarray(tensors[offsets_name], dtype=np.uint64)
            if offsets[0] != 0 or offsets[-1] != terminal or np.any(np.diff(offsets) < 0):
                raise R3ActionEditMlxCacheError(f"{split} {offsets_name} accounting drifted")

        counts = np.asarray(tensors["parent_board_type_counts"], dtype=np.int64)
        expected_mask, expected_types = _layout_from_board_type_counts(counts)
        token_types = np.asarray(tensors["parent_token_types"], dtype=np.uint8)
        token_seats = np.asarray(tensors["parent_token_seats"], dtype=np.uint8)
        payload = np.asarray(tensors["parent_token_payload"], dtype=np.int8)
        if (
            not np.array_equal(token_types, expected_types)
            or np.any(payload[~expected_mask] != 0)
            or np.any(token_seats[~expected_mask] != 0)
        ):
            raise R3ActionEditMlxCacheError(f"{split} parent R2 padding or type layout drifted")
        for seat in range(BOARD_SLOTS):
            if np.any(token_seats[:, seat][expected_mask[:, seat]] != seat):
                raise R3ActionEditMlxCacheError(f"{split} parent relative-seat ownership drifted")
        if not all(
            np.isfinite(np.asarray(tensors[name], dtype=np.float32)).all()
            for name in (
                "parent_market_features",
                "parent_player_features",
                "parent_global_features",
            )
        ):
            raise R3ActionEditMlxCacheError(
                f"{split} parent public features contain non-finite values"
            )

        transform_ids = np.asarray(tensors["canonical_transform_ids"], dtype=np.uint8)
        r3_types = np.asarray(tensors["r3_token_types"], dtype=np.uint8)
        r3_operations = np.asarray(tensors["r3_token_operations"], dtype=np.uint8)
        control_types = np.asarray(tensors["control_add_types"], dtype=np.uint8)
        if (
            np.any(transform_ids >= D6_TRANSFORMS)
            or np.any((r3_types < 1) | (r3_types > 8))
            or np.any(r3_operations >= R3_OPERATION_COUNT)
            or np.any((control_types < 1) | (control_types > 4))
        ):
            raise R3ActionEditMlxCacheError(f"{split} candidate token codes are out of range")

        source_indices = np.asarray(tensors["source_candidate_indices"], dtype=np.int64)
        action_hashes = np.asarray(tensors["action_hashes"], dtype=np.uint8)
        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        selected = np.asarray(tensors["selected_source_indices"], dtype=np.int64)
        champion = np.asarray(tensors["champion_source_indices"], dtype=np.int64)
        cohort_hashes = np.asarray(tensors["cohort_hashes"], dtype=np.uint8)
        candidate_hashes = np.asarray(tensors["candidate_identity_hashes"], dtype=np.uint8)
        control_after_hashes = np.asarray(tensors["control_after_hashes"], dtype=np.uint8)
        r3_offsets = np.asarray(tensors["r3_token_offsets"], dtype=np.uint64)
        r3_payload = np.asarray(tensors["r3_token_payload"], dtype=np.int8)
        for row in range(source.groups):
            start = int(group_offsets[row])
            end = int(group_offsets[row + 1])
            indices = source_indices[start:end]
            hashes = action_hashes[start:end]
            if (
                np.any(np.diff(indices) <= 0)
                or indices[0] < 0
                or indices[-1] >= source_counts[row]
                or selected[row] not in indices
                or champion[row] not in indices
            ):
                raise R3ActionEditMlxCacheError(f"{split} retained cohort is invalid at row {row}")
            cohort = _cohort_blake3(int(group_ids[row]), indices, hashes)
            if cohort != bytes(cohort_hashes[row]):
                raise R3ActionEditMlxCacheError(f"{split} cohort identity drifted at row {row}")
            identity = blake3.blake3()
            identity.update(_CANDIDATE_IDENTITY_DOMAIN)
            identity.update(int(group_ids[row]).to_bytes(8, "little"))
            identity.update((end - start).to_bytes(8, "little"))
            for candidate in range(start, end):
                token_start = int(r3_offsets[candidate])
                token_end = int(r3_offsets[candidate + 1])
                r3_hash = _r3_token_blake3(
                    r3_types[token_start:token_end],
                    r3_operations[token_start:token_end],
                    r3_payload[token_start:token_end],
                )
                identity.update(int(source_indices[candidate]).to_bytes(2, "little"))
                identity.update(action_hashes[candidate].tobytes())
                identity.update(control_after_hashes[candidate].tobytes())
                identity.update(r3_hash)
            if identity.digest() != bytes(candidate_hashes[row]):
                raise R3ActionEditMlxCacheError(f"{split} candidate identity drifted at row {row}")

        checks = self.manifest["splits"][split].get("checks")
        if not _checks_cover_split(
            checks,
            groups=source.groups,
            candidates=source.retained_candidates,
            validation=split == "validation",
        ):
            raise R3ActionEditMlxCacheError(f"{split} R3 mechanical checks do not cover the split")


class R3ActionEditMlxDataset:
    """One exact open graded dataset joined to R3 and S1 sidecars."""

    def __init__(
        self,
        root: str | Path,
        *,
        cache: R3ActionEditMlxCache,
        s1_cache: S1ExactSupplyCache,
        verify_dataset_checksums: bool = True,
        preverified_open_data_proof_id: str | None = None,
    ):
        if preverified_open_data_proof_id is not None:
            _require_blake3(
                preverified_open_data_proof_id,
                "preverified open-data proof",
            )
        self.base = GradedOracleDataset(
            root,
            verify_checksums=verify_dataset_checksums,
        )
        if self.base.split not in EXPECTED_SPLIT_COUNTS:
            raise R3ActionEditMlxCacheError("R3 accepts only the open train and validation splits")
        self.split = self.base.split
        self.cache = cache
        self.s1_cache = s1_cache
        self.source = cache.splits[self.split]
        split_manifest = cache.manifest["splits"][self.split]
        if split_manifest.get("dataset_id") != self.base.manifest.get(
            "dataset_id"
        ) or split_manifest.get("dataset_manifest_blake3") != _checksum(
            self.base.root / "dataset.json"
        ):
            raise R3ActionEditMlxCacheError(
                "graded-oracle dataset identity disagrees with R3 cache"
            )

        wanted = set(self.source.group_rows)
        found: dict[int, _GroupLocation] = {}
        groups = (
            self.base.raw_groups()
            if preverified_open_data_proof_id is None
            else self.base.raw_group_headers()
        )
        for raw, ref, identity in groups:
            if identity.group_id not in wanted:
                continue
            row = self.source.group_rows[identity.group_id]
            tensors = self.source.tensors
            if (
                identity.candidate_count != int(tensors["source_candidate_counts"][row])
                or identity.selected_index != int(tensors["selected_source_indices"][row])
                or identity.champion_index != int(tensors["champion_source_indices"][row])
                or not np.array_equal(
                    identity.public_state_hash,
                    tensors["public_state_hashes"][row],
                )
            ):
                raise R3ActionEditMlxCacheError(
                    f"source group {identity.group_id} disagrees with R3 cache"
                )
            start = int(tensors["candidate_offsets"][row])
            end = int(tensors["candidate_offsets"][row + 1])
            source_indices = np.asarray(
                tensors["source_candidate_indices"][start:end],
                dtype=np.int64,
            )
            if preverified_open_data_proof_id is None:
                if not isinstance(identity, GradedOracleGroupIdentity):
                    raise R3ActionEditMlxCacheError(
                        "exhaustive source verification did not expose action identities"
                    )
                if not np.array_equal(
                    identity.action_hashes[source_indices],
                    tensors["action_hashes"][start:end],
                ):
                    raise R3ActionEditMlxCacheError(
                        f"source action hashes disagree at group {identity.group_id}"
                    )
                s1_cache.verify_group_candidate_identity(
                    self.split,
                    identity.group_id,
                    identity.action_hashes,
                )
            found[identity.group_id] = _GroupLocation(
                raw=raw,
                ref=ref,
                identity=_copy_group_header(identity),
            )
        if set(found) != wanted:
            missing = sorted(wanted - set(found))
            raise R3ActionEditMlxCacheError(
                f"graded-oracle dataset is missing R3 groups: {missing[:5]}"
            )
        if preverified_open_data_proof_id is not None:
            s1_cache.register_preverified_group_candidate_identities(
                self.split,
                set(found),
                proof_id=preverified_open_data_proof_id,
            )
        self.open_data_verification_id = preverified_open_data_proof_id
        self.locations = tuple(
            found[int(group_id)]
            for group_id in np.asarray(self.source.tensors["group_ids"], dtype=np.uint64)
        )
        self.low_supply_rows = np.asarray(
            [
                row
                for row, location in enumerate(self.locations)
                if 81 - location.identity.turn <= LOW_SUPPLY_MAX_UNSEEN
            ],
            dtype=np.int64,
        )
        self.independent_winner_rows = np.asarray(
            [
                row
                for row, location in enumerate(self.locations)
                if location.identity.selected_draft_kind == 1
            ],
            dtype=np.int64,
        )
        if (
            self.source.groups == EXPECTED_SPLIT_COUNTS[self.split][0]
            and self.split == "train"
            and (len(self.low_supply_rows) != 133 or len(self.independent_winner_rows) != 55)
        ):
            raise R3ActionEditMlxCacheError(
                "R3 train slice membership drifted from 133 low-supply and "
                "55 independent-winner groups"
            )

    @property
    def group_count(self) -> int:
        return self.source.groups

    @property
    def candidate_count(self) -> int:
        return self.source.retained_candidates

    def batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        arm: str,
        transform_ids: Sequence[int] | np.ndarray | None = None,
        candidate_positions: Sequence[Sequence[int] | np.ndarray] | None = None,
        require_selected_action: bool = True,
        require_champion_action: bool = True,
        verify_control_hashes: bool = True,
        include_parent_tokens: bool = True,
        control_materialization: str = CONTROL_MATERIALIZATION_VERIFIED,
    ) -> R3ActionEditBatch:
        if arm not in ARMS:
            raise ValueError(f"unknown R3 comparison arm: {arm}")
        if control_materialization not in CONTROL_MATERIALIZATIONS:
            raise ValueError(f"unknown R3 control materialization: {control_materialization}")
        if (
            control_materialization == CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
            and arm != CONTROL_ARM
        ):
            raise ValueError(
                "preverified control materialization applies only to the exact-R2 control"
            )
        if (
            control_materialization == CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
            and verify_control_hashes
        ):
            raise ValueError(
                "preverified control materialization cannot repeat per-candidate hashes"
            )
        if (
            control_materialization == CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
            and self.open_data_verification_id is None
        ):
            raise R3ActionEditMlxCacheError(
                "preverified control materialization requires an exhaustive open-data proof"
            )
        selected_rows, transforms = self._normalize_rows_and_transforms(
            rows,
            transform_ids,
        )
        selected_positions = self._normalize_candidate_positions(
            selected_rows,
            candidate_positions,
        )
        if (
            candidate_positions is not None
            and control_materialization == CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
        ):
            raise ValueError(
                "candidate subsets require verified control materialization"
            )

        individual: list[GradedOracleBatch] = []
        selected_sources: list[np.ndarray] = []
        for row, positions in zip(selected_rows, selected_positions, strict=True):
            location = self.locations[int(row)]
            start = int(self.source.tensors["candidate_offsets"][row])
            end = int(self.source.tensors["candidate_offsets"][row + 1])
            all_sources = np.asarray(
                self.source.tensors["source_candidate_indices"][start:end],
                dtype=np.int64,
            )
            sources = all_sources if positions is None else all_sources[positions]
            individual.append(
                decode_graded_oracle_groups(
                    location.raw,
                    (location.ref,),
                    candidate_indices=(sources,),
                    require_selected_action=require_selected_action,
                    require_champion_action=require_champion_action,
                )
            )
            selected_sources.append(sources)
        graded = _combine_graded_batches(individual)
        padded_sources = _pad_rows(selected_sources, fill=0, dtype=np.int64)
        facts = self.s1_cache.materialize(
            self.split,
            RELATIONAL_ARM,
            graded,
            source_candidate_indices=padded_sources,
        )
        facts = transform_s1_exact_supply_batch(facts, transforms)
        parent = self._parent_batch(
            selected_rows,
            transforms,
            include_tokens=include_parent_tokens,
        )
        (
            candidate_features,
            candidate_mask,
            candidate_counts,
            canonical_transforms,
        ) = self._candidate_batch(
            selected_rows,
            arm=arm,
            candidate_positions=selected_positions,
            verify_control_hashes=verify_control_hashes,
            control_materialization=control_materialization,
        )
        return R3ActionEditBatch(
            base=facts.base,
            supply_vector=facts.supply_vector,
            staged_supply_vector=facts.staged_supply_vector,
            supply_tokens=facts.supply_tokens,
            supply_mask=facts.supply_mask,
            refill_target=facts.refill_target,
            selected_archetype=facts.selected_archetype,
            frontier_features=facts.frontier_features,
            parent=parent,
            candidate_token_features=mx.array(candidate_features),
            candidate_token_mask=mx.array(candidate_mask),
            candidate_token_counts=mx.array(candidate_counts),
            source_candidate_indices=mx.array(padded_sources.astype(np.int32)),
            canonical_transform_ids=mx.array(canonical_transforms),
            arm=arm,
        )

    def parent_batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        transform_ids: Sequence[int] | np.ndarray | None = None,
        include_tokens: bool = True,
    ) -> R3ParentBatch:
        """Materialize only once-per-decision parent state."""
        selected_rows, transforms = self._normalize_rows_and_transforms(
            rows,
            transform_ids,
        )
        return self._parent_batch(
            selected_rows,
            transforms,
            include_tokens=include_tokens,
        )

    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        arm: str,
        verify_control_hashes: bool = True,
        control_materialization: str = CONTROL_MATERIALIZATION_VERIFIED,
    ) -> R3ActionEditBatch:
        if self.split != "train":
            raise ValueError("deterministic R3 training batches require the train split")
        rows = deterministic_training_rows(
            step=step,
            seed=seed,
            all_rows=np.arange(self.group_count, dtype=np.int64),
            low_supply_rows=self.low_supply_rows,
            independent_winner_rows=self.independent_winner_rows,
        )
        transforms = deterministic_transform_ids(
            step=step,
            seed=seed,
            slots=len(rows),
        )
        return self.batch(
            rows,
            arm=arm,
            transform_ids=transforms,
            verify_control_hashes=verify_control_hashes,
            control_materialization=control_materialization,
        )

    def _parent_batch(
        self,
        rows: np.ndarray,
        transforms: np.ndarray,
        *,
        include_tokens: bool = True,
    ) -> R3ParentBatch:
        tensors = self.source.tensors
        if include_tokens:
            token_types = np.asarray(tensors["parent_token_types"][rows]).copy()
            token_seats = np.asarray(tensors["parent_token_seats"][rows]).copy()
            payload = np.asarray(tensors["parent_token_payload"][rows]).copy()
            token_mask = token_types != 0
            _transform_payload_in_place(
                payload.reshape(len(rows), -1, TOKEN_PAYLOAD_WIDTH),
                token_types.reshape(len(rows), -1),
                transforms,
            )
            features = _materialize_token_features(
                token_types,
                token_seats,
                payload,
                token_mask,
            )
        else:
            token_types = np.zeros((len(rows), BOARD_SLOTS, 0), dtype=np.uint8)
            token_mask = np.zeros_like(token_types, dtype=np.bool_)
            features = np.zeros(
                (len(rows), BOARD_SLOTS, 0, TOKEN_FEATURES),
                dtype=np.float32,
            )
        return R3ParentBatch(
            token_features=mx.array(features),
            token_types=mx.array(token_types.astype(np.int32)),
            token_mask=mx.array(token_mask),
            market_features=mx.array(np.asarray(tensors["parent_market_features"][rows]).copy()),
            market_mask=mx.array(
                np.asarray(tensors["parent_market_mask"][rows], dtype=np.bool_).copy()
            ),
            player_features=mx.array(np.asarray(tensors["parent_player_features"][rows]).copy()),
            player_mask=mx.array(
                np.asarray(tensors["parent_player_mask"][rows], dtype=np.bool_).copy()
            ),
            global_features=mx.array(np.asarray(tensors["parent_global_features"][rows]).copy()),
            transform_ids=mx.array(transforms.astype(np.int32)),
        )

    def _normalize_rows_and_transforms(
        self,
        rows: Sequence[int] | np.ndarray,
        transform_ids: Sequence[int] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_rows = np.asarray(rows, dtype=np.int64)
        if (
            selected_rows.ndim != 1
            or not len(selected_rows)
            or np.any(selected_rows < 0)
            or np.any(selected_rows >= self.group_count)
        ):
            raise IndexError("R3 group rows must be a nonempty in-range vector")
        if transform_ids is None:
            transforms = np.zeros(len(selected_rows), dtype=np.int64)
        else:
            transforms = np.asarray(transform_ids, dtype=np.int64)
            if transforms.shape != selected_rows.shape:
                raise ValueError("R3 transform IDs must align with group rows")
            if np.any((transforms < 0) | (transforms >= D6_TRANSFORMS)):
                raise ValueError("R3 transform IDs must be in [0, 11]")
        return selected_rows, transforms

    def _normalize_candidate_positions(
        self,
        rows: np.ndarray,
        candidate_positions: Sequence[Sequence[int] | np.ndarray] | None,
    ) -> tuple[np.ndarray | None, ...]:
        if candidate_positions is None:
            return tuple(None for _ in rows)
        if len(candidate_positions) != len(rows):
            raise ValueError("R3 candidate selections must align with group rows")
        normalized: list[np.ndarray] = []
        offsets = self.source.tensors["candidate_offsets"]
        for row, raw_positions in zip(rows, candidate_positions, strict=True):
            positions = np.asarray(raw_positions, dtype=np.int64)
            count = int(offsets[int(row) + 1]) - int(offsets[int(row)])
            if (
                positions.ndim != 1
                or not len(positions)
                or np.any(positions < 0)
                or np.any(positions >= count)
                or np.any(np.diff(positions) <= 0)
            ):
                raise ValueError(
                    "R3 candidate positions must be nonempty, strictly increasing, and in range"
                )
            normalized.append(positions)
        return tuple(normalized)

    def _candidate_batch(
        self,
        rows: np.ndarray,
        *,
        arm: str,
        verify_control_hashes: bool,
        control_materialization: str,
        candidate_positions: tuple[np.ndarray | None, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if candidate_positions is None:
            candidate_positions = tuple(None for _ in rows)
        if (
            arm == CONTROL_ARM
            and control_materialization == CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED
        ):
            return self._preverified_control_candidate_batch(rows)

        sequences: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        canonical_rows: list[np.ndarray] = []
        maximum_tokens = 0
        maximum_candidates = 0
        for row, positions in zip(rows, candidate_positions, strict=True):
            start = int(self.source.tensors["candidate_offsets"][row])
            end = int(self.source.tensors["candidate_offsets"][row + 1])
            candidates = (
                np.arange(start, end, dtype=np.int64)
                if positions is None
                else start + positions
            )
            group_sequences = []
            for candidate in candidates:
                if arm == CONTROL_ARM:
                    sequence = self._control_sequence(
                        int(row),
                        int(candidate),
                        verify_hash=verify_control_hashes,
                    )
                else:
                    sequence = self._r3_sequence(int(candidate), ARM_RADII[arm])
                maximum_tokens = max(maximum_tokens, len(sequence[0]))
                group_sequences.append(sequence)
            maximum_candidates = max(maximum_candidates, len(group_sequences))
            sequences.append(group_sequences)
            canonical = np.asarray(
                self.source.tensors["canonical_transform_ids"][start:end],
                dtype=np.int32,
            )
            canonical_rows.append(canonical if positions is None else canonical[positions])

        token_types = np.zeros(
            (len(rows), maximum_candidates, maximum_tokens),
            dtype=np.uint8,
        )
        operations = np.zeros_like(token_types)
        payload = np.zeros(
            (
                len(rows),
                maximum_candidates,
                maximum_tokens,
                R3_TOKEN_PAYLOAD_WIDTH,
            ),
            dtype=np.int8,
        )
        candidate_mask = np.zeros_like(token_types, dtype=np.bool_)
        counts = np.zeros((len(rows), maximum_candidates), dtype=np.int32)
        canonical = np.zeros((len(rows), maximum_candidates), dtype=np.int32)
        for group, group_sequences in enumerate(sequences):
            canonical[group, : len(group_sequences)] = canonical_rows[group]
            for candidate, (types, ops, values) in enumerate(group_sequences):
                count = len(types)
                token_types[group, candidate, :count] = types
                operations[group, candidate, :count] = ops
                payload[group, candidate, :count] = values
                candidate_mask[group, candidate, :count] = True
                counts[group, candidate] = count
        return (
            _materialize_candidate_features(
                token_types,
                operations,
                payload,
                candidate_mask,
            ),
            candidate_mask,
            counts,
            canonical,
        )

    def _preverified_control_candidate_batch(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        row_batches = [self._preverified_control_candidate_row(int(row)) for row in rows]
        maximum_candidates = max(batch[0].shape[0] for batch in row_batches)
        maximum_tokens = max(batch[0].shape[1] for batch in row_batches)
        features = np.zeros(
            (
                len(row_batches),
                maximum_candidates,
                maximum_tokens,
                R3_TOKEN_FEATURES,
            ),
            dtype=np.float32,
        )
        mask = np.zeros(
            (len(row_batches), maximum_candidates, maximum_tokens),
            dtype=np.bool_,
        )
        counts = np.zeros(
            (len(row_batches), maximum_candidates),
            dtype=np.int32,
        )
        canonical = np.zeros(
            (len(row_batches), maximum_candidates),
            dtype=np.int32,
        )
        for batch_row, (
            row_features,
            row_mask,
            row_counts,
            row_canonical,
        ) in enumerate(row_batches):
            candidates, tokens = row_mask.shape
            features[batch_row, :candidates, :tokens] = row_features
            mask[batch_row, :candidates, :tokens] = row_mask
            counts[batch_row, :candidates] = row_counts
            canonical[batch_row, :candidates] = row_canonical
        return features, mask, counts, canonical

    def _preverified_control_candidate_row(
        self,
        group_row: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tensors = self.source.tensors
        candidate_start = int(tensors["candidate_offsets"][group_row])
        candidate_end = int(tensors["candidate_offsets"][group_row + 1])
        candidates = candidate_end - candidate_start
        if candidates <= 0:
            raise R3ActionEditMlxCacheError("preverified control row contains no candidates")

        active = int(
            np.asarray(
                tensors["parent_board_type_counts"][group_row, 0],
                dtype=np.int64,
            ).sum()
        )
        parent_types = np.asarray(
            tensors["parent_token_types"][group_row, 0, :active],
            dtype=np.uint8,
        )
        parent_payload = np.asarray(
            tensors["parent_token_payload"][group_row, 0, :active],
            dtype=np.int8,
        )
        if active <= 0 or np.any((parent_types < 1) | (parent_types > 4)):
            raise R3ActionEditMlxCacheError("preverified control parent token layout is invalid")

        transforms = np.asarray(
            tensors["canonical_transform_ids"][candidate_start:candidate_end],
            dtype=np.int16,
        )
        centers = np.asarray(
            tensors["transformed_centers"][candidate_start:candidate_end],
            dtype=np.int16,
        )
        frame_keys = np.empty((candidates, 3), dtype=np.int16)
        frame_keys[:, 0] = transforms
        frame_keys[:, 1:] = centers
        unique_frames, frame_inverse = np.unique(
            frame_keys,
            axis=0,
            return_inverse=True,
        )
        transformed_frames = np.empty(
            (len(unique_frames), active, TOKEN_PAYLOAD_WIDTH),
            dtype=np.int8,
        )
        parent_types_batch = parent_types[None, :]
        for frame_index, (transform_id, center_q, center_r) in enumerate(unique_frames):
            payload = parent_payload.copy()
            _transform_payload_in_place(
                payload[None, :, :],
                parent_types_batch,
                np.asarray([transform_id], dtype=np.int64),
            )
            _translate_r2_payloads_in_place(
                payload,
                parent_types,
                (int(center_q), int(center_r)),
            )
            transformed_frames[frame_index] = payload
        candidate_parent_payload = transformed_frames[frame_inverse]

        remove_offsets = np.asarray(
            tensors["control_remove_offsets"][candidate_start : candidate_end + 1],
            dtype=np.int64,
        )
        remove_counts = np.diff(remove_offsets)
        remove_indices = np.asarray(
            tensors["control_remove_indices"][int(remove_offsets[0]) : int(remove_offsets[-1])],
            dtype=np.int64,
        )
        remove_rows = np.repeat(
            np.arange(candidates, dtype=np.int64),
            remove_counts,
        )
        if (
            len(remove_indices) != len(remove_rows)
            or np.any(remove_counts < 0)
            or np.any(remove_indices < 0)
            or np.any(remove_indices >= active)
        ):
            raise R3ActionEditMlxCacheError("preverified control remove accounting is invalid")
        if len(remove_indices) > 1:
            same_candidate = remove_rows[1:] == remove_rows[:-1]
            if np.any(np.diff(remove_indices)[same_candidate] <= 0):
                raise R3ActionEditMlxCacheError(
                    "preverified control remove indices are not strictly ordered"
                )
        keep = np.ones((candidates, active), dtype=np.bool_)
        keep[remove_rows, remove_indices] = False
        kept_counts = keep.sum(axis=1, dtype=np.int64)

        add_offsets = np.asarray(
            tensors["control_add_offsets"][candidate_start : candidate_end + 1],
            dtype=np.int64,
        )
        add_counts = np.diff(add_offsets)
        added_types = np.asarray(
            tensors["control_add_types"][int(add_offsets[0]) : int(add_offsets[-1])],
            dtype=np.uint8,
        )
        added_payload = np.asarray(
            tensors["control_add_payload"][int(add_offsets[0]) : int(add_offsets[-1])],
            dtype=np.int8,
        )
        add_rows = np.repeat(
            np.arange(candidates, dtype=np.int64),
            add_counts,
        )
        if (
            len(added_types) != len(add_rows)
            or added_payload.shape != (len(add_rows), TOKEN_PAYLOAD_WIDTH)
            or np.any(add_counts < 0)
            or np.any((added_types < 1) | (added_types > 4))
        ):
            raise R3ActionEditMlxCacheError("preverified control addition accounting is invalid")

        token_counts = kept_counts + add_counts
        maximum_tokens = int(token_counts.max(initial=0))
        if maximum_tokens <= 0:
            raise R3ActionEditMlxCacheError(
                "preverified control materialization produced an empty afterstate"
            )
        features = np.zeros(
            (candidates, maximum_tokens, R3_TOKEN_FEATURES),
            dtype=np.float32,
        )
        mask = np.arange(maximum_tokens, dtype=np.int64)[None, :] < token_counts[:, None]

        keep_rows, parent_indices = np.nonzero(keep)
        parent_positions = (np.cumsum(keep, axis=1, dtype=np.int64) - 1)[keep_rows, parent_indices]
        mapped_parent_types = _CONTROL_TYPE_MAP[parent_types[parent_indices]]
        features[
            keep_rows,
            parent_positions,
            mapped_parent_types.astype(np.intp) - 1,
        ] = 1.0
        features[
            keep_rows,
            parent_positions,
            R3_TOKEN_TYPE_COUNT + R3_CONTROL_OPERATION,
        ] = 1.0
        features[
            keep_rows,
            parent_positions,
            R3_TOKEN_TYPE_COUNT + R3_OPERATION_COUNT : R3_TOKEN_TYPE_COUNT
            + R3_OPERATION_COUNT
            + TOKEN_PAYLOAD_WIDTH,
        ] = candidate_parent_payload[keep_rows, parent_indices].astype(np.float32) / 64.0

        if len(add_rows):
            addition_source_offsets = np.repeat(
                add_offsets[:-1],
                add_counts,
            )
            addition_positions = (
                np.arange(
                    int(add_offsets[0]),
                    int(add_offsets[-1]),
                    dtype=np.int64,
                )
                - addition_source_offsets
                + kept_counts[add_rows]
            )
            mapped_added_types = _CONTROL_TYPE_MAP[added_types]
            features[
                add_rows,
                addition_positions,
                mapped_added_types.astype(np.intp) - 1,
            ] = 1.0
            features[
                add_rows,
                addition_positions,
                R3_TOKEN_TYPE_COUNT + R3_CONTROL_OPERATION,
            ] = 1.0
            features[
                add_rows,
                addition_positions,
                R3_TOKEN_TYPE_COUNT + R3_OPERATION_COUNT : R3_TOKEN_TYPE_COUNT
                + R3_OPERATION_COUNT
                + TOKEN_PAYLOAD_WIDTH,
            ] = added_payload.astype(np.float32) / 64.0

        features *= mask[..., None]
        return (
            features,
            mask,
            token_counts.astype(np.int32),
            transforms.astype(np.int32),
        )

    def _r3_sequence(
        self,
        candidate: int,
        radius: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensors = self.source.tensors
        start = int(tensors["r3_token_offsets"][candidate])
        end = int(tensors["r3_token_offsets"][candidate + 1])
        token_types = np.asarray(tensors["r3_token_types"][start:end]).copy()
        operations = np.asarray(tensors["r3_token_operations"][start:end]).copy()
        payload = np.asarray(tensors["r3_token_payload"][start:end]).copy()
        if radius < 3:
            patch = token_types == R3_LOCAL_PATCH_TOKEN
            q = payload[:, 0].astype(np.int16)
            r = payload[:, 1].astype(np.int16)
            distance = np.maximum.reduce((np.abs(q), np.abs(r), np.abs(q + r)))
            keep = ~patch | (distance <= radius)
            token_types = token_types[keep]
            operations = operations[keep]
            payload = payload[keep]
        if not len(token_types):
            raise R3ActionEditMlxCacheError("R3 radius crop removed every token")
        return token_types, operations, payload

    def _control_sequence(
        self,
        group_row: int,
        candidate: int,
        *,
        verify_hash: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensors = self.source.tensors
        transform_id = int(tensors["canonical_transform_ids"][candidate])
        center = tuple(int(value) for value in tensors["transformed_centers"][candidate])
        active = int(
            np.asarray(
                tensors["parent_board_type_counts"][group_row, 0],
                dtype=np.int64,
            ).sum()
        )
        token_types = np.asarray(
            tensors["parent_token_types"][group_row, 0, :active],
            dtype=np.uint8,
        ).copy()
        payload = np.asarray(
            tensors["parent_token_payload"][group_row, 0, :active],
            dtype=np.int8,
        ).copy()
        _transform_payload_in_place(
            payload[None, :, :],
            token_types[None, :],
            np.asarray([transform_id], dtype=np.int64),
        )
        _translate_r2_payloads_in_place(payload, token_types, center)

        remove_start = int(tensors["control_remove_offsets"][candidate])
        remove_end = int(tensors["control_remove_offsets"][candidate + 1])
        remove = np.asarray(
            tensors["control_remove_indices"][remove_start:remove_end],
            dtype=np.int64,
        )
        if (
            np.any(remove < 0)
            or np.any(remove >= active)
            or (len(remove) > 1 and np.any(np.diff(remove) <= 0))
        ):
            raise R3ActionEditMlxCacheError("R3 control remove indices are invalid")
        keep = np.ones(active, dtype=np.bool_)
        keep[remove] = False

        add_start = int(tensors["control_add_offsets"][candidate])
        add_end = int(tensors["control_add_offsets"][candidate + 1])
        added_types = np.asarray(
            tensors["control_add_types"][add_start:add_end],
            dtype=np.uint8,
        )
        added_payload = np.asarray(
            tensors["control_add_payload"][add_start:add_end],
            dtype=np.int8,
        )
        after_types = np.concatenate([token_types[keep], added_types])
        after_payload = np.concatenate([payload[keep], added_payload], axis=0)
        if verify_hash:
            observed = _control_multiset_blake3(after_types, after_payload)
            expected = bytes(tensors["control_after_hashes"][candidate])
            if observed != expected:
                raise R3ActionEditMlxCacheError(
                    f"R3 control afterstate hash drifted at candidate {candidate}"
                )
        mapped_types = _CONTROL_TYPE_MAP[after_types]
        output_payload = np.zeros(
            (len(after_types), R3_TOKEN_PAYLOAD_WIDTH),
            dtype=np.int8,
        )
        output_payload[:, :TOKEN_PAYLOAD_WIDTH] = after_payload
        operations = np.full(
            len(after_types),
            R3_CONTROL_OPERATION,
            dtype=np.uint8,
        )
        return mapped_types, operations, output_payload


def _combine_graded_batches(batches: Sequence[GradedOracleBatch]) -> GradedOracleBatch:
    if not batches:
        raise ValueError("cannot combine an empty graded-oracle batch list")
    candidate_counts = [
        int(np.asarray(batch.candidate_mask, dtype=np.bool_).sum()) for batch in batches
    ]
    maximum = max(candidate_counts)
    values: dict[str, object] = {}
    for field in fields(GradedOracleBatch):
        items = [getattr(batch, field.name) for batch in batches]
        arrays = [np.asarray(item) for item in items]
        candidate_specific = all(
            array.ndim >= 2 and array.shape[1] == count
            for array, count in zip(arrays, candidate_counts, strict=True)
        )
        if candidate_specific:
            padded = []
            for array, count in zip(arrays, candidate_counts, strict=True):
                shape = list(array.shape)
                shape[1] = maximum
                target = np.zeros(shape, dtype=array.dtype)
                target[:, :count] = array
                padded.append(target)
            combined = np.concatenate(padded, axis=0)
        else:
            combined = np.concatenate(arrays, axis=0)
        values[field.name] = combined if isinstance(items[0], np.ndarray) else mx.array(combined)
    return GradedOracleBatch(**values)


def _materialize_candidate_features(
    token_types: np.ndarray,
    operations: np.ndarray,
    payload: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if (
        token_types.shape != operations.shape
        or token_types.shape != mask.shape
        or payload.shape != (*token_types.shape, R3_TOKEN_PAYLOAD_WIDTH)
    ):
        raise ValueError("R3 candidate token tensors have inconsistent shapes")
    features = np.zeros(
        (*token_types.shape, R3_TOKEN_FEATURES),
        dtype=np.float32,
    )
    active = np.nonzero(mask)
    features[(*active, token_types[active].astype(np.intp) - 1)] = 1.0
    features[(*active, R3_TOKEN_TYPE_COUNT + operations[active].astype(np.intp))] = 1.0
    features[..., R3_TOKEN_TYPE_COUNT + R3_OPERATION_COUNT :] = payload.astype(np.float32) / 64.0
    features *= mask[..., None]
    return features


def _translate_r2_payloads_in_place(
    payload: np.ndarray,
    token_types: np.ndarray,
    center: tuple[int, int],
) -> None:
    for index, token_type in enumerate(token_types):
        if token_type in (1, 2, 4):
            _translate_coord(payload[index], 0, 1, center)
        elif token_type == 3:
            members = int(payload[index, 2])
            if members < 0 or members > 23 or 6 + members * 2 > TOKEN_PAYLOAD_WIDTH:
                raise R3ActionEditMlxCacheError(
                    "R2 component member payload exceeds the exact board bound"
                )
            for member in range(members):
                _translate_coord(
                    payload[index],
                    6 + member * 2,
                    7 + member * 2,
                    center,
                )
        else:
            raise R3ActionEditMlxCacheError(f"cannot translate unknown R2 token type {token_type}")


def _translate_coord(
    payload: np.ndarray,
    q_slot: int,
    r_slot: int,
    center: tuple[int, int],
) -> None:
    q = int(payload[q_slot]) - center[0]
    r = int(payload[r_slot]) - center[1]
    if not -128 <= q <= 127 or not -128 <= r <= 127:
        raise R3ActionEditMlxCacheError(
            "canonical relative R2 coordinate exceeds signed-byte range"
        )
    payload[q_slot] = q
    payload[r_slot] = r


def _control_multiset_blake3(
    token_types: np.ndarray,
    payload: np.ndarray,
) -> bytes:
    ordered = sorted(
        zip(
            (int(value) for value in token_types),
            (tuple(int(value) for value in row) for row in payload),
            strict=True,
        )
    )
    digest = blake3.blake3()
    digest.update(_CONTROL_TOKEN_DOMAIN)
    digest.update(len(ordered).to_bytes(8, "little"))
    for token_type, values in ordered:
        digest.update(bytes([token_type]))
        digest.update(np.asarray(values, dtype=np.int8).view(np.uint8).tobytes())
    return digest.digest()


def _r3_token_blake3(
    token_types: np.ndarray,
    operations: np.ndarray,
    payload: np.ndarray,
) -> bytes:
    digest = blake3.blake3()
    digest.update(_R3_TOKEN_DOMAIN)
    digest.update(R3_SCHEMA_VERSION.to_bytes(2, "little"))
    digest.update(len(token_types).to_bytes(8, "little"))
    for token_type, operation, values in zip(
        token_types,
        operations,
        payload,
        strict=True,
    ):
        digest.update(bytes([int(token_type), int(operation)]))
        digest.update(np.asarray(values, dtype=np.int8).view(np.uint8).tobytes())
    return digest.digest()


def _cohort_blake3(
    group_id: int,
    source_indices: np.ndarray,
    action_hashes: np.ndarray,
) -> bytes:
    digest = blake3.blake3()
    digest.update(_COHORT_DOMAIN)
    digest.update(group_id.to_bytes(8, "little"))
    digest.update(len(source_indices).to_bytes(8, "little"))
    for source_index, action_hash in zip(
        source_indices,
        action_hashes,
        strict=True,
    ):
        digest.update(int(source_index).to_bytes(2, "little"))
        digest.update(np.asarray(action_hash, dtype=np.uint8).tobytes())
    return digest.digest()


def _pad_rows(
    rows: Sequence[np.ndarray],
    *,
    fill: int,
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    maximum = max(len(row) for row in rows)
    result = np.full((len(rows), maximum), fill, dtype=dtype)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row
    return result


def deterministic_training_rows(
    *,
    step: int,
    seed: int,
    all_rows: np.ndarray,
    low_supply_rows: np.ndarray,
    independent_winner_rows: np.ndarray,
) -> np.ndarray:
    """Return the frozen three-global-plus-one-slice group schedule."""
    if step < 0:
        raise ValueError("R3 training step cannot be negative")
    all_rows = _schedule_rows(all_rows, "all")
    low_supply_rows = _schedule_rows(low_supply_rows, "low-supply")
    independent_winner_rows = _schedule_rows(
        independent_winner_rows,
        "independent-winner",
    )
    selected = [
        _permutation_value(all_rows, seed=seed, stream=stream, cursor=step) for stream in range(3)
    ]
    slice_rows = low_supply_rows if step % 2 == 0 else independent_winner_rows
    selected.append(
        _permutation_value(
            slice_rows,
            seed=seed,
            stream=3 + step % 2,
            cursor=step // 2,
        )
    )
    return np.asarray(selected, dtype=np.int64)


def deterministic_transform_ids(
    *,
    step: int,
    seed: int,
    slots: int = 4,
) -> np.ndarray:
    if step < 0 or slots <= 0:
        raise ValueError("R3 transform schedule requires nonnegative step and slots")
    output = np.zeros(slots, dtype=np.int64)
    for slot in range(slots):
        digest = blake3.blake3()
        digest.update(_TRAINING_SCHEDULE_DOMAIN)
        digest.update(seed.to_bytes(8, "little", signed=False))
        digest.update(step.to_bytes(8, "little", signed=False))
        digest.update(slot.to_bytes(4, "little", signed=False))
        output[slot] = int.from_bytes(digest.digest(length=2), "little") % 12
    return output


def _schedule_rows(values: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError(f"R3 {label} schedule rows must be unique and nonempty")
    return rows


def _permutation_value(
    rows: np.ndarray,
    *,
    seed: int,
    stream: int,
    cursor: int,
) -> int:
    epoch, offset = divmod(cursor, len(rows))
    sequence = np.random.SeedSequence([seed, stream, epoch, 0x5233414354494F4E])
    permutation = np.random.default_rng(sequence).permutation(rows)
    return int(permutation[offset])


def _expected_shapes(
    *,
    groups: int,
    candidates: int,
    removed: int,
    added: int,
    r3_tokens: int,
) -> dict[str, tuple[int, ...]]:
    if min(groups, candidates, removed, added, r3_tokens) < 0:
        raise R3ActionEditMlxCacheError("R3 split token counts are negative")
    return {
        "group_ids": (groups,),
        "public_state_hashes": (groups, 32),
        "source_candidate_counts": (groups,),
        "retained_candidate_counts": (groups,),
        "selected_source_indices": (groups,),
        "champion_source_indices": (groups,),
        "cohort_hashes": (groups, 32),
        "candidate_identity_hashes": (groups, 32),
        "candidate_offsets": (groups + 1,),
        "parent_token_types": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_token_seats": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_token_payload": (
            groups,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            TOKEN_PAYLOAD_WIDTH,
        ),
        "parent_board_type_counts": (groups, BOARD_SLOTS, 4),
        "parent_market_features": (groups, 4, MARKET_FEATURES),
        "parent_market_mask": (groups, 4),
        "parent_player_features": (groups, BOARD_SLOTS, PLAYER_FEATURES),
        "parent_player_mask": (groups, BOARD_SLOTS),
        "parent_global_features": (groups, GLOBAL_FEATURES),
        "source_candidate_indices": (candidates,),
        "action_hashes": (candidates, 32),
        "canonical_transform_ids": (candidates,),
        "transformed_centers": (candidates, 2),
        "control_after_hashes": (candidates, 32),
        "control_remove_offsets": (candidates + 1,),
        "control_remove_indices": (removed,),
        "control_add_offsets": (candidates + 1,),
        "control_add_types": (added,),
        "control_add_payload": (added, TOKEN_PAYLOAD_WIDTH),
        "r3_token_offsets": (candidates + 1,),
        "r3_token_types": (r3_tokens,),
        "r3_token_operations": (r3_tokens,),
        "r3_token_payload": (r3_tokens, R3_TOKEN_PAYLOAD_WIDTH),
    }


def _checks_cover_split(
    checks: object,
    *,
    groups: int,
    candidates: int,
    validation: bool,
) -> bool:
    if not isinstance(checks, dict):
        return False
    candidate_checks = (
        "graded_action_reconstructions",
        "grouped_r3_action_matches",
        "r3_apply_checks",
        "authoritative_successor_checks",
        "canonical_transform_checks",
        "r2_afterstate_encodings",
        "control_delta_round_trips",
        "r3_token_round_trips",
    )
    group_checks = (
        "groups_replayed",
        "parent_r2_encodings",
        "position_record_checks",
        "public_state_hash_checks",
        "public_supply_checks",
        "selected_winner_retained",
        "champion_retained",
    )
    return (
        all(checks.get(name) == candidates for name in candidate_checks)
        and all(checks.get(name) == groups for name in group_checks)
        and checks.get("source_r600") == checks.get("r600_retained")
        and checks.get("source_r4800") == checks.get("r4800_retained")
        and (not validation or checks.get("source_r1200") == checks.get("r1200_retained"))
        and checks.get("silent_truncations") == 0
        and isinstance(checks.get("minimum_control_tokens"), int)
        and isinstance(checks.get("maximum_control_tokens"), int)
        and checks["minimum_control_tokens"] > 0
        and checks["maximum_control_tokens"] >= checks["minimum_control_tokens"]
        and isinstance(checks.get("minimum_r3_tokens"), int)
        and isinstance(checks.get("maximum_r3_tokens"), int)
        and checks["minimum_r3_tokens"] > 0
        and checks["maximum_r3_tokens"] >= checks["minimum_r3_tokens"]
    )


def open_data_verification_identity(
    *,
    cache: R3ActionEditMlxCache,
    s1_cache: S1ExactSupplyCache,
    train_dataset: str | Path,
    validation_dataset: str | Path,
) -> dict[str, Any]:
    """Bind content-addressed sidecars to the two exact open dataset manifests."""
    datasets: dict[str, dict[str, Any]] = {}
    for split, root_value in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        root = Path(root_value)
        manifest_path = root / "dataset.json"
        manifest = _read_json(manifest_path, f"{split} graded-oracle manifest")
        dataset_id = manifest.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise R3ActionEditMlxCacheError(f"{split} graded-oracle dataset ID is absent")
        manifest_blake3 = _checksum(manifest_path)
        r3_split = cache.manifest["splits"][split]
        s1_split = s1_cache.manifest["splits"][split]
        if (
            manifest.get("split") != split
            or r3_split.get("dataset_id") != dataset_id
            or r3_split.get("dataset_manifest_blake3") != manifest_blake3
            or s1_split.get("dataset_id") != dataset_id
            or s1_split.get("dataset_manifest_blake3") != manifest_blake3
        ):
            raise R3ActionEditMlxCacheError(
                f"{split} dataset manifest is not bound by both public sidecars"
            )
        datasets[split] = {
            "dataset_id": dataset_id,
            "manifest_blake3": manifest_blake3,
        }
    return {
        "cache_id": cache.cache_id,
        "cache_manifest_blake3": _checksum(cache.root / "cache.json"),
        "s1_cache_id": s1_cache.cache_id,
        "s1_cache_manifest_blake3": _checksum(s1_cache.manifest_path),
        "datasets": datasets,
    }


def open_data_verification_id(identity: dict[str, Any]) -> str:
    """Content-address one complete open-data verification identity."""
    return _canonical_blake3(identity)


def _copy_group_header(identity: GradedOracleGroupHeader) -> GradedOracleGroupHeader:
    return GradedOracleGroupHeader(
        group_id=identity.group_id,
        public_state_hash=identity.public_state_hash.copy(),
        candidate_count=identity.candidate_count,
        selected_index=identity.selected_index,
        champion_index=identity.champion_index,
        turn=identity.turn,
        selected_draft_kind=identity.selected_draft_kind,
    )


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_blake3(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R3ActionEditMlxCacheError(f"{label} is not a lowercase BLAKE3 digest")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R3ActionEditMlxCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R3ActionEditMlxCacheError(f"{label} must be a JSON object")
    return value
