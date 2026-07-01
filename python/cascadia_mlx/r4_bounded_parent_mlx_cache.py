"""Fail-closed bounded-parent sidecar joined to the accepted R3 action cache."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM as R3_CONTROL_ARM,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    D6_TRANSFORMS,
    R3ActionEditBatch,
    R3ActionEditMlxCache,
    R3ActionEditMlxDataset,
    _transform_payload_in_place,
    deterministic_training_rows,
    deterministic_transform_ids,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    open_data_verification_identity as r3_open_data_verification_identity,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "r4-bounded-parent-mlx-cache-v1"
EXPERIMENT_ID = "r4-bounded-quotient-mlx-comparison-v1"
PROTOCOL_ID = "r4-bounded-parent-mlx-matched-comparison-v1"
ADR_ID = "0156"
EXPECTED_R3_CACHE_ID = "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"

ARMS = (
    "c0-exact-r2-parent",
    "q1-seat-marginal-parent",
    "q2-directional-parent",
    "q3-affordance-parent",
)
CONTROL_ARM = ARMS[0]
ARM_TO_BOUNDED = {
    ARMS[1]: "q1-seat-marginal",
    ARMS[2]: "q2-directional",
    ARMS[3]: "q3-affordance",
}
BOUNDED_ARMS = tuple(ARM_TO_BOUNDED.values())
ARM_HARD_TOKEN_MAX = {
    "q1-seat-marginal": 184,
    "q2-directional": 204,
    "q3-affordance": 200,
}

BOARD_SLOTS = 4
UNIVERSAL_PARENT_CLASS_COUNT = 9
UNIVERSAL_PARENT_VALUE_WIDTH = 144
BOUNDED_KIND_COUNT = 5
R2_VALUE_WIDTH = 52
EXPECTED_SPLIT_COUNTS = {"train": 560, "validation": 240}
EXPECTED_TENSOR_CONTRACT = {
    "r3_parent_boundary": {
        "cache_id": EXPECTED_R3_CACHE_ID,
        "group_and_public_hash_alignment": True,
        "candidate_afterstate_stream_unchanged": True,
    },
    "bounded_parent": {
        "arms": list(BOUNDED_ARMS),
        "radius": "radius4-61",
        "d6_views_per_group": D6_TRANSFORMS,
        "ragged_active_i16_values": True,
        "universal_parent_class_count": UNIVERSAL_PARENT_CLASS_COUNT,
        "universal_parent_value_width": UNIVERSAL_PARENT_VALUE_WIDTH,
        "board_slots": BOARD_SLOTS,
        "bounded_kind_count": BOUNDED_KIND_COUNT,
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
_COMMON_FILES = {"group_ids", "public_state_hashes"}
_ARM_FILES = {
    "view_offsets",
    "token_kinds",
    "token_seats",
    "value_offsets",
    "token_values",
    "token_counts",
    "board_class_counts",
    "view_hashes",
    "adaptive_state_hashes",
}
_DTYPES = {
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<i2": np.dtype("<i2"),
    "<u8": np.dtype("<u8"),
}
_ACTIVE_WIDTHS = np.asarray([128, 64, 80, 80, 144], dtype=np.int64)
_CLASS_NAMES = (
    "r2_occupied",
    "r2_frontier",
    "r2_habitat_component",
    "r2_wildlife_motif",
    "q_near_cell",
    "q_far_habitat_component",
    "q_far_wildlife_component",
    "q_wildlife_summary",
    "q_frontier_summary",
)


class R4BoundedParentMlxCacheError(ValueError):
    """The bounded-parent cache or one of its exact bindings is inconsistent."""


@dataclass(frozen=True)
class R4ParentBatch:
    """One compact universal parent token batch."""

    token_values: mx.array
    token_classes: mx.array
    token_types: mx.array
    token_mask: mx.array
    token_counts: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array
    transform_ids: mx.array


@dataclass(frozen=True)
class _Tensor:
    path: Path
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    blake3: str

    def memmap(self) -> np.memmap:
        return np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape)


@dataclass(frozen=True)
class _ArmSplit:
    views: int
    tokens: int
    active_values: int
    hard_token_max: int
    tensors: dict[str, np.memmap]


@dataclass(frozen=True)
class _Split:
    groups: int
    tensors: dict[str, np.memmap]
    arms: dict[str, _ArmSplit]


class R4BoundedParentMlxCache:
    """Content-addressed bounded parent states aligned exactly to ADR 0150."""

    def __init__(
        self,
        root: str | Path,
        *,
        r3_cache: R3ActionEditMlxCache,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.r3_cache = r3_cache
        self.manifest = _read_json(self.root / "cache.json", "R4 parent cache manifest")
        self._validate_envelope(require_complete=require_complete)
        self.splits = {
            split: self._load_split(split, verify_checksums=verify_checksums)
            for split in EXPECTED_SPLIT_COUNTS
        }
        self.parent_capacities = self._parent_capacities()
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
    ) -> R4BoundedParentMlxDataset:
        r3 = self.r3_cache.bind_dataset(
            root,
            s1_cache=s1_cache,
            verify_dataset_checksums=verify_dataset_checksums,
            preverified_open_data_proof_id=preverified_open_data_proof_id,
        )
        return R4BoundedParentMlxDataset(cache=self, r3=r3)

    def parent_batch(
        self,
        split: str,
        rows: np.ndarray,
        transforms: np.ndarray,
        *,
        arm: str,
        r3_parent: object,
    ) -> R4ParentBatch:
        if arm == CONTROL_ARM:
            sequences = self._control_sequences(split, rows, transforms)
        elif arm in ARM_TO_BOUNDED:
            sequences = self._bounded_sequences(
                split,
                rows,
                transforms,
                bounded_arm=ARM_TO_BOUNDED[arm],
            )
        else:
            raise ValueError(f"unknown R4 parent arm: {arm}")
        classes, values, mask, counts = _materialize_parent_sequences(
            sequences,
            capacity=self.parent_capacities[arm],
        )
        return R4ParentBatch(
            token_values=mx.array(values),
            token_classes=mx.array(classes.astype(np.int32)),
            token_types=mx.array(classes.astype(np.int32)),
            token_mask=mx.array(mask),
            token_counts=mx.array(counts),
            market_features=r3_parent.market_features,
            market_mask=r3_parent.market_mask,
            player_features=r3_parent.player_features,
            player_mask=r3_parent.player_mask,
            global_features=r3_parent.global_features,
            transform_ids=mx.array(transforms.astype(np.int32)),
        )

    def parent_token_statistics(self, split: str, arm: str) -> dict[str, Any]:
        source = self.splits[split]
        if arm == CONTROL_ARM:
            counts = np.asarray(
                self.r3_cache.splits[split].tensors["parent_board_type_counts"][: source.groups],
                dtype=np.int64,
            )
            by_class = np.zeros(UNIVERSAL_PARENT_CLASS_COUNT, dtype=np.int64)
            by_class[:4] = counts.sum(axis=(0, 1))
            by_group = counts.sum(axis=(1, 2))
            per_board = counts.sum(axis=2)
        elif arm in ARM_TO_BOUNDED:
            bounded = source.arms[ARM_TO_BOUNDED[arm]]
            counts = np.asarray(
                bounded.tensors["board_class_counts"][:, 0],
                dtype=np.int64,
            )
            by_class = np.zeros(UNIVERSAL_PARENT_CLASS_COUNT, dtype=np.int64)
            by_class[4:] = counts.sum(axis=(0, 1))
            by_group = counts.sum(axis=(1, 2))
            per_board = counts.sum(axis=2)
        else:
            raise ValueError(f"unknown R4 parent arm: {arm}")
        return {
            "groups": source.groups,
            "tensor_capacity_per_board": self.parent_capacities[arm],
            "tokens": _distribution(by_group),
            "per_board_tokens": _distribution(per_board.reshape(-1)),
            "class_tokens": {name: int(by_class[index]) for index, name in enumerate(_CLASS_NAMES)},
        }

    def _parent_capacities(self) -> dict[str, int]:
        control = max(
            int(self.r3_cache.splits[split].tensors["parent_token_types"].shape[2])
            for split in EXPECTED_SPLIT_COUNTS
        )
        capacities = {CONTROL_ARM: control}
        for arm, bounded_arm in ARM_TO_BOUNDED.items():
            capacity = max(
                int(
                    np.asarray(
                        self.splits[split].arms[bounded_arm].tensors["board_class_counts"],
                        dtype=np.int64,
                    )
                    .sum(axis=-1)
                    .max()
                )
                for split in EXPECTED_SPLIT_COUNTS
            )
            if capacity <= 0 or capacity > ARM_HARD_TOKEN_MAX[bounded_arm]:
                raise R4BoundedParentMlxCacheError(
                    f"R4 parent capacity is invalid for {bounded_arm}"
                )
            capacities[arm] = capacity
        return capacities

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
        ):
            raise R4BoundedParentMlxCacheError("unsupported R4 bounded-parent cache envelope")
        identity = manifest.get("scientific_identity")
        if (
            not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise R4BoundedParentMlxCacheError(
                "R4 parent cache content address or directory identity is invalid"
            )
        if manifest.get("tensor_contract") != EXPECTED_TENSOR_CONTRACT:
            raise R4BoundedParentMlxCacheError("R4 parent tensor contract drifted")
        if manifest.get("hidden_information") != EXPECTED_HIDDEN_BOUNDARY:
            raise R4BoundedParentMlxCacheError("R4 parent hidden-information boundary drifted")
        r3 = manifest.get("r3_cache")
        if (
            not isinstance(r3, dict)
            or r3.get("cache_id") != EXPECTED_R3_CACHE_ID
            or r3.get("cache_id") != self.r3_cache.cache_id
            or r3.get("manifest_blake3") != _checksum(self.r3_cache.root / "cache.json")
        ):
            raise R4BoundedParentMlxCacheError("R4 parent cache is not bound to ADR 0150")
        exporter = manifest.get("exporter")
        if not isinstance(exporter, dict):
            raise R4BoundedParentMlxCacheError("R4 parent exporter identity is absent")
        _require_blake3(exporter.get("executable_blake3"), "R4 parent exporter executable")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise R4BoundedParentMlxCacheError(
                "production R4 parent cache must cover the complete open corpus"
            )
        raw_splits = manifest.get("splits")
        if not isinstance(raw_splits, dict) or set(raw_splits) != set(EXPECTED_SPLIT_COUNTS):
            raise R4BoundedParentMlxCacheError("R4 parent cache splits are incomplete")
        if identity.get("splits") != raw_splits:
            raise R4BoundedParentMlxCacheError(
                "R4 parent scientific identity does not bind its split manifests"
            )

    def _load_split(self, split: str, *, verify_checksums: bool) -> _Split:
        raw = self.manifest["splits"][split]
        if not isinstance(raw, dict):
            raise R4BoundedParentMlxCacheError(f"{split} R4 split is malformed")
        groups = int(raw.get("groups", -1))
        if groups <= 0:
            raise R4BoundedParentMlxCacheError(f"{split} R4 group count is invalid")
        if raw.get("complete_open_split") is True and groups != EXPECTED_SPLIT_COUNTS[split]:
            raise R4BoundedParentMlxCacheError(
                f"{split} R4 split claims complete coverage with incorrect counts"
            )
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != _COMMON_FILES:
            raise R4BoundedParentMlxCacheError(f"{split} R4 common tensor set drifted")
        common_shapes = {
            "group_ids": (groups,),
            "public_state_hashes": (groups, 32),
        }
        tensors = {
            name: self._tensor(files[name], shape, verify_checksum=verify_checksums).memmap()
            for name, shape in common_shapes.items()
        }
        raw_arms = raw.get("arms")
        if not isinstance(raw_arms, dict) or set(raw_arms) != set(BOUNDED_ARMS):
            raise R4BoundedParentMlxCacheError(f"{split} R4 arm set drifted")
        arms: dict[str, _ArmSplit] = {}
        for arm in BOUNDED_ARMS:
            arm_raw = raw_arms[arm]
            if (
                not isinstance(arm_raw, dict)
                or arm_raw.get("arm") != arm
                or arm_raw.get("universal_classes") != [5, 6, 7, 8, 9]
                or arm_raw.get("hard_token_max") != ARM_HARD_TOKEN_MAX[arm]
                or int(arm_raw.get("views", -1)) != groups * D6_TRANSFORMS
            ):
                raise R4BoundedParentMlxCacheError(f"{split} {arm} manifest drifted")
            tokens = int(arm_raw.get("tokens", -1))
            active_values = int(arm_raw.get("active_values", -1))
            if tokens <= 0 or active_values <= 0:
                raise R4BoundedParentMlxCacheError(f"{split} {arm} counts are invalid")
            arm_files = arm_raw.get("files")
            if not isinstance(arm_files, dict) or set(arm_files) != _ARM_FILES:
                raise R4BoundedParentMlxCacheError(f"{split} {arm} tensor set drifted")
            views = groups * D6_TRANSFORMS
            shapes = {
                "view_offsets": (views + 1,),
                "token_kinds": (tokens,),
                "token_seats": (tokens,),
                "value_offsets": (tokens + 1,),
                "token_values": (active_values,),
                "token_counts": (groups, D6_TRANSFORMS),
                "board_class_counts": (
                    groups,
                    D6_TRANSFORMS,
                    BOARD_SLOTS,
                    BOUNDED_KIND_COUNT,
                ),
                "view_hashes": (groups, D6_TRANSFORMS, 32),
                "adaptive_state_hashes": (groups, D6_TRANSFORMS, 32),
            }
            arm_tensors = {
                name: self._tensor(
                    arm_files[name],
                    shape,
                    verify_checksum=verify_checksums,
                ).memmap()
                for name, shape in shapes.items()
            }
            arms[arm] = _ArmSplit(
                views=views,
                tokens=tokens,
                active_values=active_values,
                hard_token_max=ARM_HARD_TOKEN_MAX[arm],
                tensors=arm_tensors,
            )
        return _Split(groups=groups, tensors=tensors, arms=arms)

    def _tensor(
        self,
        raw: object,
        expected_shape: tuple[int, ...],
        *,
        verify_checksum: bool,
    ) -> _Tensor:
        if not isinstance(raw, dict) or raw.get("dtype") not in _DTYPES:
            raise R4BoundedParentMlxCacheError("R4 parent tensor specification is malformed")
        if raw.get("shape") != list(expected_shape):
            raise R4BoundedParentMlxCacheError("R4 parent tensor shape drifted")
        path = self.root / str(raw.get("file"))
        if path.parent != self.root or not path.is_file():
            raise R4BoundedParentMlxCacheError("R4 parent tensor path escapes or is absent")
        dtype = _DTYPES[str(raw["dtype"])]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        if raw.get("bytes") != expected_bytes or path.stat().st_size != expected_bytes:
            raise R4BoundedParentMlxCacheError("R4 parent tensor byte count drifted")
        digest = raw.get("blake3")
        _require_blake3(digest, "R4 parent tensor")
        if verify_checksum and _checksum(path) != digest:
            raise R4BoundedParentMlxCacheError("R4 parent tensor checksum mismatch")
        return _Tensor(
            path=path,
            dtype=dtype,
            shape=expected_shape,
            blake3=str(digest),
        )

    def _verify_split_semantics(self, split: str) -> None:
        source = self.splits[split]
        r3 = self.r3_cache.splits[split]
        if not np.array_equal(
            source.tensors["group_ids"],
            r3.tensors["group_ids"][: source.groups],
        ):
            raise R4BoundedParentMlxCacheError(f"{split} group IDs differ from R3")
        if not np.array_equal(
            source.tensors["public_state_hashes"],
            r3.tensors["public_state_hashes"][: source.groups],
        ):
            raise R4BoundedParentMlxCacheError(f"{split} public hashes differ from R3")
        raw = self.manifest["splits"][split]
        if raw.get("dataset_id") != self.r3_cache.manifest["splits"][split].get(
            "dataset_id"
        ) or raw.get("dataset_manifest_blake3") != self.r3_cache.manifest["splits"][split].get(
            "dataset_manifest_blake3"
        ):
            raise R4BoundedParentMlxCacheError(f"{split} dataset binding differs from R3")
        for arm, bounded in source.arms.items():
            tensors = bounded.tensors
            view_offsets = np.asarray(tensors["view_offsets"], dtype=np.uint64)
            token_counts = np.asarray(tensors["token_counts"], dtype=np.int64).reshape(-1)
            value_offsets = np.asarray(tensors["value_offsets"], dtype=np.uint64)
            kinds = np.asarray(tensors["token_kinds"], dtype=np.int64)
            seats = np.asarray(tensors["token_seats"], dtype=np.int64)
            board_counts = np.asarray(tensors["board_class_counts"], dtype=np.int64).reshape(
                bounded.views,
                BOARD_SLOTS,
                BOUNDED_KIND_COUNT,
            )
            if (
                view_offsets[0] != 0
                or view_offsets[-1] != bounded.tokens
                or np.any(np.diff(view_offsets) != token_counts)
                or np.any(token_counts <= 0)
                or np.any(token_counts > bounded.hard_token_max)
                or value_offsets[0] != 0
                or value_offsets[-1] != bounded.active_values
                or np.any(np.diff(value_offsets) <= 0)
                or np.any((kinds < 1) | (kinds > BOUNDED_KIND_COUNT))
                or np.any((seats < 0) | (seats >= BOARD_SLOTS))
                or np.any(np.diff(value_offsets) > _ACTIVE_WIDTHS[kinds - 1])
                or np.any(board_counts.sum(axis=(1, 2)) != token_counts)
            ):
                raise R4BoundedParentMlxCacheError(f"{split} {arm} ragged accounting drifted")
            for view in range(bounded.views):
                start = int(view_offsets[view])
                end = int(view_offsets[view + 1])
                observed = np.zeros(
                    (BOARD_SLOTS, BOUNDED_KIND_COUNT),
                    dtype=np.int64,
                )
                np.add.at(observed, (seats[start:end], kinds[start:end] - 1), 1)
                if not np.array_equal(observed, board_counts[view]):
                    raise R4BoundedParentMlxCacheError(
                        f"{split} {arm} board/class counts drifted at view {view}"
                    )
            hashes = np.asarray(tensors["view_hashes"], dtype=np.uint8)
            adaptive = np.asarray(tensors["adaptive_state_hashes"], dtype=np.uint8)
            if np.any(np.all(hashes == 0, axis=-1)) or np.any(np.all(adaptive == 0, axis=-1)):
                raise R4BoundedParentMlxCacheError(f"{split} {arm} contains a zero identity")
            token_boundaries = view_offsets.astype(np.int64)
            active_values_per_view = (
                value_offsets[token_boundaries[1:]] - value_offsets[token_boundaries[:-1]]
            )
            checks = self.manifest["splits"][split]["arms"][arm].get("checks")
            expected_views = source.groups * D6_TRANSFORMS
            if (
                not isinstance(checks, dict)
                or checks.get("bounded_views") != expected_views
                or checks.get("bounded_envelope_round_trips") != expected_views
                or checks.get("source_accounting_checks") != expected_views
                or checks.get("hard_token_max_checks") != expected_views
                or checks.get("universal_class_checks") != bounded.tokens
                or checks.get("minimum_tokens") != int(token_counts.min())
                or checks.get("maximum_tokens") != int(token_counts.max())
                or checks.get("minimum_active_values") != int(active_values_per_view.min())
                or checks.get("maximum_active_values") != int(active_values_per_view.max())
            ):
                raise R4BoundedParentMlxCacheError(f"{split} {arm} checks are incomplete")

    def _control_sequences(
        self,
        split: str,
        rows: np.ndarray,
        transforms: np.ndarray,
    ) -> list[list[list[tuple[int, np.ndarray]]]]:
        tensors = self.r3_cache.splits[split].tensors
        token_types = np.asarray(tensors["parent_token_types"][rows]).copy()
        payload = np.asarray(tensors["parent_token_payload"][rows]).copy()
        _transform_payload_in_place(
            payload.reshape(len(rows), -1, R2_VALUE_WIDTH),
            token_types.reshape(len(rows), -1),
            transforms,
        )
        output: list[list[list[tuple[int, np.ndarray]]]] = []
        for batch_row in range(len(rows)):
            boards: list[list[tuple[int, np.ndarray]]] = []
            for board in range(BOARD_SLOTS):
                active = token_types[batch_row, board] != 0
                classes = token_types[batch_row, board, active].astype(np.int64)
                values = payload[batch_row, board, active].astype(np.int16)
                order = np.argsort(classes, kind="stable")
                boards.append([(int(classes[index]), values[index]) for index in order])
            output.append(boards)
        return output

    def _bounded_sequences(
        self,
        split: str,
        rows: np.ndarray,
        transforms: np.ndarray,
        *,
        bounded_arm: str,
    ) -> list[list[list[tuple[int, np.ndarray]]]]:
        source = self.splits[split].arms[bounded_arm]
        tensors = source.tensors
        view_offsets = tensors["view_offsets"]
        value_offsets = tensors["value_offsets"]
        kinds = tensors["token_kinds"]
        seats = tensors["token_seats"]
        values = tensors["token_values"]
        board_counts = tensors["board_class_counts"]
        output: list[list[list[tuple[int, np.ndarray]]]] = []
        for row, transform in zip(rows, transforms, strict=True):
            view = int(row) * D6_TRANSFORMS + int(transform)
            start = int(view_offsets[view])
            end = int(view_offsets[view + 1])
            boards: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(BOARD_SLOTS)]
            observed = np.zeros((BOARD_SLOTS, BOUNDED_KIND_COUNT), dtype=np.int64)
            for token in range(start, end):
                kind = int(kinds[token])
                seat = int(seats[token])
                value_start = int(value_offsets[token])
                value_end = int(value_offsets[token + 1])
                boards[seat].append(
                    (
                        kind + 4,
                        np.asarray(values[value_start:value_end], dtype=np.int16).copy(),
                    )
                )
                observed[seat, kind - 1] += 1
            if not np.array_equal(observed, board_counts[int(row), int(transform)]):
                raise R4BoundedParentMlxCacheError(
                    f"{split} {bounded_arm} materialization accounting drifted"
                )
            output.append(boards)
        return output


class R4BoundedParentMlxDataset:
    """One R3 dataset whose parent representation is routed by ADR 0156 arm."""

    def __init__(
        self,
        *,
        cache: R4BoundedParentMlxCache,
        r3: R3ActionEditMlxDataset,
    ):
        self.cache = cache
        self.r3 = r3
        self.base = r3.base
        self.split = r3.split
        self._group_count = self.cache.splits[self.split].groups
        self.low_supply_rows = r3.low_supply_rows[r3.low_supply_rows < self._group_count]
        self.independent_winner_rows = r3.independent_winner_rows[
            r3.independent_winner_rows < self._group_count
        ]
        offsets = np.asarray(
            r3.source.tensors["candidate_offsets"][: self._group_count + 1],
            dtype=np.uint64,
        )
        self._candidate_count = int(offsets[-1])

    @property
    def group_count(self) -> int:
        return self._group_count

    @property
    def candidate_count(self) -> int:
        return self._candidate_count

    def batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        arm: str,
        transform_ids: Sequence[int] | np.ndarray | None = None,
        verify_control_hashes: bool = True,
    ) -> R3ActionEditBatch:
        selected = np.asarray(rows, dtype=np.int64)
        if transform_ids is None:
            transforms = np.zeros(len(selected), dtype=np.int64)
        else:
            transforms = np.asarray(transform_ids, dtype=np.int64)
        common = self.r3.batch(
            selected,
            arm=R3_CONTROL_ARM,
            transform_ids=transforms,
            verify_control_hashes=verify_control_hashes,
        )
        parent = self.cache.parent_batch(
            self.split,
            selected,
            transforms,
            arm=arm,
            r3_parent=common.parent,
        )
        return replace(common, parent=parent, arm=arm)

    def parent_batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        arm: str,
        transform_ids: Sequence[int] | np.ndarray | None = None,
    ) -> R4ParentBatch:
        """Materialize only the once-per-decision parent representation."""
        selected = np.asarray(rows, dtype=np.int64)
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= self.group_count)
        ):
            raise IndexError("R4 parent rows must be a nonempty in-range vector")
        if transform_ids is None:
            transforms = np.zeros(len(selected), dtype=np.int64)
        else:
            transforms = np.asarray(transform_ids, dtype=np.int64)
            if transforms.shape != selected.shape:
                raise ValueError("R4 parent transform IDs must align with rows")
            if np.any((transforms < 0) | (transforms >= D6_TRANSFORMS)):
                raise ValueError("R4 parent transform IDs must be in [0, 11]")
        r3_parent = self.r3.parent_batch(
            selected,
            transform_ids=transforms,
        )
        return self.cache.parent_batch(
            self.split,
            selected,
            transforms,
            arm=arm,
            r3_parent=r3_parent,
        )

    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        arm: str,
        verify_control_hashes: bool = True,
    ) -> R3ActionEditBatch:
        if self.split != "train":
            raise ValueError("deterministic R4 parent batches require the train split")
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
        )

    def parent_token_statistics(self, arm: str) -> dict[str, Any]:
        return self.cache.parent_token_statistics(self.split, arm)


def open_data_verification_identity(
    *,
    cache: R4BoundedParentMlxCache,
    s1_cache: S1ExactSupplyCache,
    train_dataset: str | Path,
    validation_dataset: str | Path,
) -> dict[str, Any]:
    identity = r3_open_data_verification_identity(
        cache=cache.r3_cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    return {
        **identity,
        "r4_parent_cache_id": cache.cache_id,
        "r4_parent_cache_manifest_blake3": _checksum(cache.root / "cache.json"),
    }


def open_data_verification_id(identity: dict[str, Any]) -> str:
    return _canonical_blake3(identity)


def _materialize_parent_sequences(
    sequences: list[list[list[tuple[int, np.ndarray]]]],
    *,
    capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sequences or any(len(boards) != BOARD_SLOTS for boards in sequences):
        raise ValueError("R4 parent sequences must contain four boards per group")
    observed_maximum = max(len(tokens) for boards in sequences for tokens in boards)
    maximum = observed_maximum if capacity is None else capacity
    if maximum <= 0:
        raise ValueError("R4 parent sequence contains no tokens")
    if observed_maximum > maximum:
        raise R4BoundedParentMlxCacheError("R4 parent sequence exceeds its fixed tensor capacity")
    classes = np.zeros((len(sequences), BOARD_SLOTS, maximum), dtype=np.uint8)
    values = np.zeros(
        (
            len(sequences),
            BOARD_SLOTS,
            maximum,
            UNIVERSAL_PARENT_VALUE_WIDTH,
        ),
        dtype=np.int16,
    )
    mask = np.zeros((len(sequences), BOARD_SLOTS, maximum), dtype=np.bool_)
    counts = np.zeros((len(sequences), BOARD_SLOTS), dtype=np.int32)
    for group, boards in enumerate(sequences):
        for board, tokens in enumerate(boards):
            previous = 0
            for slot, (token_class, payload) in enumerate(tokens):
                if not 1 <= token_class <= UNIVERSAL_PARENT_CLASS_COUNT or token_class < previous:
                    raise R4BoundedParentMlxCacheError(
                        "R4 parent token classes are invalid or noncanonical"
                    )
                previous = token_class
                payload = np.asarray(payload, dtype=np.int16)
                if payload.ndim != 1 or len(payload) > UNIVERSAL_PARENT_VALUE_WIDTH:
                    raise R4BoundedParentMlxCacheError(
                        "R4 parent token payload exceeds the universal width"
                    )
                classes[group, board, slot] = token_class
                values[group, board, slot, : len(payload)] = payload
                mask[group, board, slot] = True
            counts[group, board] = len(tokens)
    return classes, values, mask, counts


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": int(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": int(values.max()),
    }


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
        raise R4BoundedParentMlxCacheError(f"{label} is not a lowercase BLAKE3 digest")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise R4BoundedParentMlxCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise R4BoundedParentMlxCacheError(f"{label} must be a JSON object")
    return value
