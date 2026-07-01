"""Fail-closed relational substrate joined to the accepted R3 action cache."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ARMS as R3_ARMS,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM as R3_CONTROL_ARM,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_MATERIALIZATION_VERIFIED,
    D6_TRANSFORMS,
    R3ActionEditBatch,
    R3ActionEditMlxCache,
    R3ActionEditMlxDataset,
    deterministic_training_rows,
    deterministic_transform_ids,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    open_data_verification_identity as r3_open_data_verification_identity,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "relational-substrate-mlx-cache-v1"
EXPERIMENT_ID = "relational-substrate-mlx-tournament-v1"
PROTOCOL_ID = "r5-s3-s5-matched-mlx-v1"
ADR_ID = "0161"
EXPECTED_R3_CACHE_ID = "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"

ARMS = (
    "c0-exact-r2",
    "q1-r5-quotient-local",
    "g2-r5-s3",
    "d3-r5-s3-s5",
)
CONTROL_ARM = ARMS[0]
R5_ARM = ARMS[1]
S3_ARM = ARMS[2]
S5_ARM = ARMS[3]
ARM_R3_CANDIDATE = {
    CONTROL_ARM: R3_CONTROL_ARM,
    R5_ARM: R3_ARMS[3],
    S3_ARM: R3_ARMS[3],
    S5_ARM: R3_ARMS[3],
}

RELATIONAL_CLASS_COUNT = 8
RELATIONAL_VALUE_WIDTH = 64
R5_MINIMAL_CLASS_COUNT = 6
S5_FEATURES = 154
PARENT_CLASS_COUNT = 12
OPPORTUNITY_NAMES = ("elk", "salmon", "hawk", "bear")
EXPECTED_SPLIT_COUNTS = {
    "train": (560, 2_135_111, 280_012),
    "validation": (240, 860_203, 860_203),
}
EXPECTED_TENSOR_CONTRACT = {
    "r3_boundary": {
        "cache_id": EXPECTED_R3_CACHE_ID,
        "group_public_candidate_and_action_alignment": True,
        "candidate_cohorts_and_labels_unchanged": True,
    },
    "parent": {
        "d6_views_per_group": D6_TRANSFORMS,
        "classes": RELATIONAL_CLASS_COUNT,
        "value_width": RELATIONAL_VALUE_WIDTH,
        "relative_boards": BOARD_SLOTS,
        "rich_view_stored_once": True,
        "r5_minimal_view_is_loader_projection": True,
        "silent_truncation": False,
    },
    "candidate": {
        "s5_width": S5_FEATURES,
        "raw_dtype": "signed_i16",
        "normalization_fit": "open_train_retained_candidates_only",
        "validation_fit": False,
        "teacher_values_used": False,
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
    "teacher_values_used_for_features": False,
}
_FILES = {
    "group_ids",
    "public_state_hashes",
    "candidate_identity_hashes",
    "opportunity_flags",
    "candidate_offsets",
    "source_candidate_indices",
    "action_hashes",
    "parent_view_offsets",
    "parent_token_classes",
    "parent_token_seats",
    "parent_token_values",
    "parent_view_counts",
    "parent_view_hashes",
    "s5_values",
}
_DTYPES = {
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<i2": np.dtype("<i2"),
    "<u8": np.dtype("<u8"),
}


class RelationalSubstrateMlxCacheError(ValueError):
    """The relational sidecar or one of its exact R3 bindings is inconsistent."""


@dataclass(frozen=True)
class RelationalParentBatch:
    """Native exact-R2 or relational parent tokens plus common public context."""

    r2_token_features: mx.array
    r2_token_types: mx.array
    r2_token_mask: mx.array
    relational_values: mx.array
    relational_classes: mx.array
    relational_mask: mx.array
    relational_counts: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array
    transform_ids: mx.array


@dataclass(frozen=True)
class RelationalSubstrateBatch:
    """One matched R3 batch with routed parent and S5 factual surfaces."""

    r3: R3ActionEditBatch
    parent: RelationalParentBatch
    derivative_features: mx.array
    opportunity_flags: mx.array
    arm: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.r3, name)


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
    parent_views: int
    parent_tokens: int
    tensors: dict[str, np.memmap]


class RelationalSubstrateMlxCache:
    """Content-addressed R5/S3/S5 sidecar aligned exactly to ADR 0150."""

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
        self.manifest = _read_json(
            self.root / "cache.json",
            "relational substrate cache manifest",
        )
        self._validate_envelope(require_complete=require_complete)
        self.normalization = self._load_normalization()
        self.splits = {
            split: self._load_split(split, verify_checksums=verify_checksums)
            for split in EXPECTED_SPLIT_COUNTS
        }
        self.parent_capacities = self._parent_capacities()
        if verify_semantics:
            for split in EXPECTED_SPLIT_COUNTS:
                self._verify_split_semantics(split)
            self._verify_normalization()

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
    ) -> RelationalSubstrateMlxDataset:
        r3 = self.r3_cache.bind_dataset(
            root,
            s1_cache=s1_cache,
            verify_dataset_checksums=verify_dataset_checksums,
            preverified_open_data_proof_id=preverified_open_data_proof_id,
        )
        return RelationalSubstrateMlxDataset(cache=self, r3=r3)

    def parent_batch(
        self,
        split: str,
        rows: np.ndarray,
        transforms: np.ndarray,
        *,
        arm: str,
        r3_parent: object,
    ) -> RelationalParentBatch:
        if arm not in ARMS:
            raise ValueError(f"unknown relational substrate arm: {arm}")
        if arm == CONTROL_ARM:
            values = np.zeros(
                (len(rows), BOARD_SLOTS, 0, RELATIONAL_VALUE_WIDTH),
                dtype=np.int16,
            )
            classes = np.zeros((len(rows), BOARD_SLOTS, 0), dtype=np.uint8)
            mask = np.zeros_like(classes, dtype=np.bool_)
            counts = np.zeros((len(rows), BOARD_SLOTS), dtype=np.int32)
        else:
            sequences = self._relational_sequences(
                split,
                rows,
                transforms,
                minimal=arm == R5_ARM,
            )
            classes, values, mask, counts = _materialize_relational_sequences(
                sequences,
                capacity=self.parent_capacities[arm],
            )
        return RelationalParentBatch(
            r2_token_features=r3_parent.token_features,
            r2_token_types=r3_parent.token_types,
            r2_token_mask=r3_parent.token_mask,
            relational_values=mx.array(values),
            relational_classes=mx.array(classes.astype(np.int32)),
            relational_mask=mx.array(mask),
            relational_counts=mx.array(counts),
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
            by_class = np.zeros(PARENT_CLASS_COUNT, dtype=np.int64)
            by_class[RELATIONAL_CLASS_COUNT:] = counts.sum(axis=(0, 1))
            per_board = counts.sum(axis=2)
        elif arm in ARMS[1:]:
            sequences = self._relational_sequences(
                split,
                np.arange(source.groups, dtype=np.int64),
                np.zeros(source.groups, dtype=np.int64),
                minimal=arm == R5_ARM,
            )
            per_board = np.asarray(
                [[len(tokens) for tokens in boards] for boards in sequences],
                dtype=np.int64,
            )
            by_class = np.zeros(PARENT_CLASS_COUNT, dtype=np.int64)
            for boards in sequences:
                for tokens in boards:
                    for token_class, _ in tokens:
                        by_class[token_class - 1] += 1
        else:
            raise ValueError(f"unknown relational substrate arm: {arm}")
        return {
            "groups": source.groups,
            "tensor_capacity_per_board": self.parent_capacities[arm],
            "tokens": _distribution(per_board.sum(axis=1)),
            "per_board_tokens": _distribution(per_board.reshape(-1)),
            "class_tokens": {
                _parent_class_name(index + 1): int(value) for index, value in enumerate(by_class)
            },
        }

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
        ):
            raise RelationalSubstrateMlxCacheError(
                "unsupported relational substrate cache envelope"
            )
        identity = manifest.get("scientific_identity")
        if (
            not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise RelationalSubstrateMlxCacheError(
                "relational substrate content address or directory identity is invalid"
            )
        if manifest.get("tensor_contract") != EXPECTED_TENSOR_CONTRACT:
            raise RelationalSubstrateMlxCacheError("relational substrate tensor contract drifted")
        if manifest.get("hidden_information") != EXPECTED_HIDDEN_BOUNDARY:
            raise RelationalSubstrateMlxCacheError(
                "relational substrate hidden-information boundary drifted"
            )
        r3 = manifest.get("r3_cache")
        if (
            not isinstance(r3, dict)
            or r3.get("cache_id") != EXPECTED_R3_CACHE_ID
            or r3.get("cache_id") != self.r3_cache.cache_id
            or r3.get("manifest_blake3") != _checksum(self.r3_cache.root / "cache.json")
        ):
            raise RelationalSubstrateMlxCacheError(
                "relational substrate cache is not bound to accepted R3"
            )
        exporter = manifest.get("exporter")
        if not isinstance(exporter, dict):
            raise RelationalSubstrateMlxCacheError(
                "relational substrate exporter identity is absent"
            )
        _require_blake3(
            exporter.get("executable_blake3"),
            "relational substrate exporter",
        )
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise RelationalSubstrateMlxCacheError(
                "production relational substrate cache must cover the open corpus"
            )
        raw_splits = manifest.get("splits")
        if not isinstance(raw_splits, dict) or set(raw_splits) != set(EXPECTED_SPLIT_COUNTS):
            raise RelationalSubstrateMlxCacheError(
                "relational substrate cache splits are incomplete"
            )
        if identity.get("splits") != raw_splits:
            raise RelationalSubstrateMlxCacheError(
                "relational substrate scientific identity does not bind its splits"
            )

    def _load_normalization(self) -> tuple[dict[str, Any], ...]:
        raw = self.manifest.get("normalization")
        if not isinstance(raw, list) or len(raw) != S5_FEATURES:
            raise RelationalSubstrateMlxCacheError("S5 normalization width differs from 154")
        output: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, value in enumerate(raw):
            if (
                not isinstance(value, dict)
                or value.get("index") != index
                or not isinstance(value.get("name"), str)
                or not value["name"]
                or value["name"] in names
                or value.get("transform")
                not in {"identity", "robust_divide", "signed_log1p_robust_divide"}
                or not isinstance(value.get("divisor"), int)
                or value["divisor"] <= 0
            ):
                raise RelationalSubstrateMlxCacheError(
                    f"S5 normalization field {index} is malformed"
                )
            names.add(value["name"])
            output.append(value)
        return tuple(output)

    def _load_split(self, split: str, *, verify_checksums: bool) -> _Split:
        raw = self.manifest["splits"][split]
        if not isinstance(raw, dict) or raw.get("split") != split:
            raise RelationalSubstrateMlxCacheError(
                f"{split} relational substrate split is malformed"
            )
        groups = int(raw.get("groups", -1))
        source_candidates = int(raw.get("source_candidates", -1))
        retained_candidates = int(raw.get("retained_candidates", -1))
        parent_views = int(raw.get("parent_views", -1))
        parent_tokens = int(raw.get("parent_tokens", -1))
        if (
            groups <= 0
            or source_candidates <= 0
            or retained_candidates <= 0
            or parent_views != groups * D6_TRANSFORMS
            or parent_tokens <= 0
        ):
            raise RelationalSubstrateMlxCacheError(
                f"{split} relational substrate counts are invalid"
            )
        if (
            raw.get("complete_open_split") is True
            and (
                groups,
                source_candidates,
                retained_candidates,
            )
            != EXPECTED_SPLIT_COUNTS[split]
        ):
            raise RelationalSubstrateMlxCacheError(
                f"{split} relational split claims complete coverage incorrectly"
            )
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != _FILES:
            raise RelationalSubstrateMlxCacheError(f"{split} relational tensor set drifted")
        shapes = {
            "group_ids": (groups,),
            "public_state_hashes": (groups, 32),
            "candidate_identity_hashes": (groups, 32),
            "opportunity_flags": (groups, len(OPPORTUNITY_NAMES)),
            "candidate_offsets": (groups + 1,),
            "source_candidate_indices": (retained_candidates,),
            "action_hashes": (retained_candidates, 32),
            "parent_view_offsets": (parent_views + 1,),
            "parent_token_classes": (parent_tokens,),
            "parent_token_seats": (parent_tokens,),
            "parent_token_values": (
                parent_tokens,
                RELATIONAL_VALUE_WIDTH,
            ),
            "parent_view_counts": (groups, D6_TRANSFORMS),
            "parent_view_hashes": (groups, D6_TRANSFORMS, 32),
            "s5_values": (retained_candidates, S5_FEATURES),
        }
        tensors = {
            name: self._tensor(
                files[name],
                shape,
                verify_checksum=verify_checksums,
            ).memmap()
            for name, shape in shapes.items()
        }
        return _Split(
            groups=groups,
            source_candidates=source_candidates,
            retained_candidates=retained_candidates,
            parent_views=parent_views,
            parent_tokens=parent_tokens,
            tensors=tensors,
        )

    def _tensor(
        self,
        raw: object,
        expected_shape: tuple[int, ...],
        *,
        verify_checksum: bool,
    ) -> _Tensor:
        if not isinstance(raw, dict) or raw.get("dtype") not in _DTYPES:
            raise RelationalSubstrateMlxCacheError(
                "relational substrate tensor specification is malformed"
            )
        if raw.get("shape") != list(expected_shape):
            raise RelationalSubstrateMlxCacheError("relational substrate tensor shape drifted")
        path = self.root / str(raw.get("file"))
        if path.parent != self.root or not path.is_file():
            raise RelationalSubstrateMlxCacheError(
                "relational substrate tensor path escapes or is absent"
            )
        dtype = _DTYPES[str(raw["dtype"])]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        if raw.get("bytes") != expected_bytes or path.stat().st_size != expected_bytes:
            raise RelationalSubstrateMlxCacheError("relational substrate tensor byte count drifted")
        digest = raw.get("blake3")
        _require_blake3(digest, "relational substrate tensor")
        if verify_checksum and _checksum(path) != digest:
            raise RelationalSubstrateMlxCacheError("relational substrate tensor checksum mismatch")
        return _Tensor(
            path=path,
            dtype=dtype,
            shape=expected_shape,
            blake3=str(digest),
        )

    def _verify_split_semantics(self, split: str) -> None:
        source = self.splits[split]
        tensors = source.tensors
        r3 = self.r3_cache.splits[split]
        if not np.array_equal(
            tensors["group_ids"],
            r3.tensors["group_ids"][: source.groups],
        ):
            raise RelationalSubstrateMlxCacheError(f"{split} group IDs differ from R3")
        if not np.array_equal(
            tensors["public_state_hashes"],
            r3.tensors["public_state_hashes"][: source.groups],
        ):
            raise RelationalSubstrateMlxCacheError(f"{split} public hashes differ from R3")
        candidate_offsets = np.asarray(tensors["candidate_offsets"], dtype=np.uint64)
        expected_offsets = np.asarray(
            r3.tensors["candidate_offsets"][: source.groups + 1],
            dtype=np.uint64,
        )
        if not np.array_equal(candidate_offsets, expected_offsets):
            raise RelationalSubstrateMlxCacheError(f"{split} candidate offsets differ from R3")
        if not np.array_equal(
            tensors["source_candidate_indices"],
            r3.tensors["source_candidate_indices"][: source.retained_candidates],
        ):
            raise RelationalSubstrateMlxCacheError(
                f"{split} source candidate indices differ from R3"
            )
        if not np.array_equal(
            tensors["action_hashes"],
            r3.tensors["action_hashes"][: source.retained_candidates],
        ):
            raise RelationalSubstrateMlxCacheError(f"{split} action hashes differ from R3")
        if not np.array_equal(
            tensors["candidate_identity_hashes"],
            r3.tensors["candidate_identity_hashes"][: source.groups],
        ):
            raise RelationalSubstrateMlxCacheError(f"{split} candidate identities differ from R3")
        raw = self.manifest["splits"][split]
        r3_raw = self.r3_cache.manifest["splits"][split]
        if raw.get("dataset_id") != r3_raw.get("dataset_id") or raw.get(
            "dataset_manifest_blake3"
        ) != r3_raw.get("dataset_manifest_blake3"):
            raise RelationalSubstrateMlxCacheError(f"{split} dataset binding differs from R3")

        view_offsets = np.asarray(tensors["parent_view_offsets"], dtype=np.uint64)
        view_counts = np.asarray(tensors["parent_view_counts"], dtype=np.int64).reshape(-1)
        classes = np.asarray(tensors["parent_token_classes"], dtype=np.int64)
        seats = np.asarray(tensors["parent_token_seats"], dtype=np.int64)
        flags = np.asarray(tensors["opportunity_flags"], dtype=np.uint8)
        if (
            view_offsets[0] != 0
            or view_offsets[-1] != source.parent_tokens
            or np.any(np.diff(view_offsets) != view_counts)
            or np.any(view_counts <= 0)
            or np.any((classes < 1) | (classes > RELATIONAL_CLASS_COUNT))
            or np.any((seats < 0) | (seats >= BOARD_SLOTS))
            or np.any((flags != 0) & (flags != 1))
            or np.any(np.all(tensors["parent_view_hashes"] == 0, axis=-1))
        ):
            raise RelationalSubstrateMlxCacheError(f"{split} relational parent accounting drifted")
        for view in range(source.parent_views):
            start = int(view_offsets[view])
            end = int(view_offsets[view + 1])
            prior_by_seat = np.zeros(BOARD_SLOTS, dtype=np.int64)
            for token_class, seat in zip(
                classes[start:end],
                seats[start:end],
                strict=True,
            ):
                if token_class < prior_by_seat[seat]:
                    raise RelationalSubstrateMlxCacheError(
                        f"{split} relational classes are noncanonical at view {view}"
                    )
                prior_by_seat[seat] = token_class

        checks = raw.get("checks")
        if (
            not isinstance(checks, dict)
            or checks.get("groups_replayed") != source.groups
            or checks.get("r3_group_id_checks") != source.groups
            or checks.get("r3_public_state_hash_checks") != source.groups
            or checks.get("r3_candidate_count_checks") != source.groups
            or checks.get("r3_candidate_identity_checks") != source.groups
            or checks.get("position_record_checks") != source.groups
            or checks.get("public_state_hash_checks") != source.groups
            or checks.get("public_supply_checks") != source.groups
            or checks.get("current_score_checks") != source.groups * BOARD_SLOTS
            or checks.get("current_score_failures") != 0
            or checks.get("parent_views") != source.parent_views
            or checks.get("parent_tokens") != source.parent_tokens
            or checks.get("minimum_parent_tokens") != int(view_counts.min())
            or checks.get("maximum_parent_tokens") != int(view_counts.max())
            or checks.get("retained_action_hash_checks") != source.retained_candidates
            or checks.get("grouped_action_matches") != source.retained_candidates
            or checks.get("afterstate_hash_checks") != source.retained_candidates
            or checks.get("score_delta_checks") != source.retained_candidates
            or checks.get("s5_width_checks") != source.retained_candidates
            or checks.get("i16_storage_checks") != source.retained_candidates * S5_FEATURES
        ):
            raise RelationalSubstrateMlxCacheError(
                f"{split} relational mechanical checks are incomplete"
            )

    def _verify_normalization(self) -> None:
        values = self.splits["train"].tensors["s5_values"]
        candidates = self.splits["train"].retained_candidates
        for index, spec in enumerate(self.normalization):
            column = np.asarray(values[:, index], dtype=np.int16)
            absolute = np.abs(column.astype(np.int32))
            p99 = int(
                np.partition(absolute, (candidates - 1) * 99 // 100)[(candidates - 1) * 99 // 100]
            )
            maximum_absolute = int(absolute.max())
            transform = (
                "identity"
                if maximum_absolute == 0
                else (
                    "signed_log1p_robust_divide"
                    if p99 > 0 and maximum_absolute > p99 * 16
                    else "robust_divide"
                )
            )
            if (
                spec.get("count") != candidates
                or spec.get("nonzero_count") != int(np.count_nonzero(column))
                or spec.get("minimum") != int(column.min())
                or spec.get("maximum") != int(column.max())
                or spec.get("p99_absolute") != p99
                or spec.get("maximum_absolute") != maximum_absolute
                or spec.get("transform") != transform
                or spec.get("divisor") != max(p99, 1)
            ):
                raise RelationalSubstrateMlxCacheError(
                    f"S5 normalization field {index} was not fitted from train only"
                )

    def _parent_capacities(self) -> dict[str, int]:
        capacities = {CONTROL_ARM: BOARD_TOKEN_CAPACITY}
        for arm in ARMS[1:]:
            maximum = 0
            minimal = arm == R5_ARM
            for split in EXPECTED_SPLIT_COUNTS:
                source = self.splits[split]
                tensors = source.tensors
                offsets = tensors["parent_view_offsets"]
                classes = tensors["parent_token_classes"]
                seats = tensors["parent_token_seats"]
                for view in range(source.parent_views):
                    start = int(offsets[view])
                    end = int(offsets[view + 1])
                    selected_classes = np.asarray(classes[start:end], dtype=np.uint8)
                    selected_seats = np.asarray(seats[start:end], dtype=np.uint8)
                    if minimal:
                        keep = selected_classes <= R5_MINIMAL_CLASS_COUNT
                        selected_seats = selected_seats[keep]
                    counts = np.bincount(
                        selected_seats,
                        minlength=BOARD_SLOTS,
                    )
                    maximum = max(maximum, int(counts.max()))
            if maximum <= 0:
                raise RelationalSubstrateMlxCacheError(
                    f"relational parent capacity is empty for {arm}"
                )
            capacities[arm] = maximum
        return capacities

    def _relational_sequences(
        self,
        split: str,
        rows: np.ndarray,
        transforms: np.ndarray,
        *,
        minimal: bool,
    ) -> list[list[list[tuple[int, np.ndarray]]]]:
        source = self.splits[split]
        tensors = source.tensors
        offsets = tensors["parent_view_offsets"]
        classes = tensors["parent_token_classes"]
        seats = tensors["parent_token_seats"]
        values = tensors["parent_token_values"]
        output: list[list[list[tuple[int, np.ndarray]]]] = []
        for row, transform in zip(rows, transforms, strict=True):
            view = int(row) * D6_TRANSFORMS + int(transform)
            start = int(offsets[view])
            end = int(offsets[view + 1])
            boards: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(BOARD_SLOTS)]
            for token in range(start, end):
                token_class = int(classes[token])
                if minimal and token_class > R5_MINIMAL_CLASS_COUNT:
                    continue
                payload = np.asarray(values[token], dtype=np.int16).copy()
                if minimal:
                    payload = _r5_minimal_values(token_class, payload)
                boards[int(seats[token])].append((token_class, payload))
            output.append(boards)
        return output


class RelationalSubstrateMlxDataset:
    """One R3 dataset whose parent/candidate facts are routed by ADR 0161."""

    def __init__(
        self,
        *,
        cache: RelationalSubstrateMlxCache,
        r3: R3ActionEditMlxDataset,
    ):
        self.cache = cache
        self.r3 = r3
        self.base = r3.base
        self.split = r3.split
        self._group_count = cache.splits[self.split].groups
        self._candidate_count = cache.splits[self.split].retained_candidates
        self.low_supply_rows = r3.low_supply_rows[r3.low_supply_rows < self._group_count]
        self.independent_winner_rows = r3.independent_winner_rows[
            r3.independent_winner_rows < self._group_count
        ]
        self.opportunity_rows = {
            name: np.flatnonzero(
                np.asarray(
                    cache.splits[self.split].tensors["opportunity_flags"][:, index],
                    dtype=np.bool_,
                )
            )
            for index, name in enumerate(OPPORTUNITY_NAMES)
        }

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
        control_materialization: str = CONTROL_MATERIALIZATION_VERIFIED,
    ) -> RelationalSubstrateBatch:
        if arm not in ARMS:
            raise ValueError(f"unknown relational substrate arm: {arm}")
        selected, transforms = self._normalize_rows_and_transforms(
            rows,
            transform_ids,
        )
        common = self.r3.batch(
            selected,
            arm=ARM_R3_CANDIDATE[arm],
            transform_ids=transforms,
            verify_control_hashes=verify_control_hashes,
            include_parent_tokens=arm == CONTROL_ARM,
            control_materialization=control_materialization,
        )
        parent = self.cache.parent_batch(
            self.split,
            selected,
            transforms,
            arm=arm,
            r3_parent=common.parent,
        )
        derivative = self._derivative_batch(
            selected,
            common,
            enabled=arm == S5_ARM,
        )
        flags = np.asarray(
            self.cache.splits[self.split].tensors["opportunity_flags"][selected],
            dtype=np.bool_,
        ).copy()
        return RelationalSubstrateBatch(
            r3=common,
            parent=parent,
            derivative_features=mx.array(derivative),
            opportunity_flags=mx.array(flags),
            arm=arm,
        )

    def parent_batch(
        self,
        rows: Sequence[int] | np.ndarray,
        *,
        arm: str,
        transform_ids: Sequence[int] | np.ndarray | None = None,
    ) -> RelationalParentBatch:
        selected, transforms = self._normalize_rows_and_transforms(
            rows,
            transform_ids,
        )
        r3_parent = self.r3.parent_batch(
            selected,
            transform_ids=transforms,
            include_tokens=arm == CONTROL_ARM,
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
        control_materialization: str = CONTROL_MATERIALIZATION_VERIFIED,
    ) -> RelationalSubstrateBatch:
        if self.split != "train":
            raise ValueError("deterministic relational substrate batches require train")
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

    def parent_token_statistics(self, arm: str) -> dict[str, Any]:
        return self.cache.parent_token_statistics(self.split, arm)

    def derivative_statistics(self, arm: str) -> dict[str, Any]:
        if arm != S5_ARM:
            return {
                "enabled": False,
                "features": S5_FEATURES,
                "nonzero_values": 0,
            }
        values = self.cache.splits[self.split].tensors["s5_values"]
        return {
            "enabled": True,
            "features": S5_FEATURES,
            "candidates": self.candidate_count,
            "nonzero_values": int(np.count_nonzero(values)),
            "normalization": {
                transform: sum(spec["transform"] == transform for spec in self.cache.normalization)
                for transform in (
                    "identity",
                    "robust_divide",
                    "signed_log1p_robust_divide",
                )
            },
        }

    def _normalize_rows_and_transforms(
        self,
        rows: Sequence[int] | np.ndarray,
        transform_ids: Sequence[int] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(rows, dtype=np.int64)
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= self.group_count)
        ):
            raise IndexError("relational substrate rows must be a nonempty in-range vector")
        if transform_ids is None:
            transforms = np.zeros(len(selected), dtype=np.int64)
        else:
            transforms = np.asarray(transform_ids, dtype=np.int64)
            if transforms.shape != selected.shape:
                raise ValueError("relational substrate transform IDs must align with rows")
            if np.any((transforms < 0) | (transforms >= D6_TRANSFORMS)):
                raise ValueError("relational substrate transform IDs must be in [0, 11]")
        return selected, transforms

    def _derivative_batch(
        self,
        rows: np.ndarray,
        common: R3ActionEditBatch,
        *,
        enabled: bool,
    ) -> np.ndarray:
        mask = np.asarray(common.base.candidate_mask, dtype=np.bool_)
        output = np.zeros((*mask.shape, S5_FEATURES), dtype=np.float32)
        source = self.cache.splits[self.split]
        offsets = source.tensors["candidate_offsets"]
        for batch_row, row in enumerate(rows):
            start = int(offsets[row])
            end = int(offsets[row + 1])
            count = end - start
            if count != int(mask[batch_row].sum()):
                raise RelationalSubstrateMlxCacheError(
                    "relational derivative candidate count differs from R3"
                )
            expected_sources = np.asarray(
                source.tensors["source_candidate_indices"][start:end],
                dtype=np.int32,
            )
            expected_hashes = np.asarray(
                source.tensors["action_hashes"][start:end],
                dtype=np.uint8,
            )
            if not np.array_equal(
                expected_sources,
                np.asarray(common.source_candidate_indices)[batch_row, :count],
            ) or not np.array_equal(
                expected_hashes,
                np.asarray(common.base.action_hash)[batch_row, :count],
            ):
                raise RelationalSubstrateMlxCacheError(
                    "relational derivative candidate identity differs from R3"
                )
            if enabled:
                output[batch_row, :count] = np.asarray(
                    source.tensors["s5_values"][start:end],
                    dtype=np.float32,
                )
        if enabled:
            log_fields = np.asarray(
                [
                    spec["transform"] == "signed_log1p_robust_divide"
                    for spec in self.cache.normalization
                ],
                dtype=np.bool_,
            )
            if np.any(log_fields):
                selected = output[..., log_fields]
                output[..., log_fields] = np.sign(selected) * np.log1p(np.abs(selected))
            divisors = np.asarray(
                [spec["divisor"] for spec in self.cache.normalization],
                dtype=np.float32,
            )
            output /= divisors
        return output


def open_data_verification_identity(
    *,
    cache: RelationalSubstrateMlxCache,
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
        "relational_cache_id": cache.cache_id,
        "relational_cache_manifest_blake3": _checksum(cache.root / "cache.json"),
    }


def open_data_verification_id(identity: dict[str, Any]) -> str:
    return _canonical_blake3(identity)


def _r5_minimal_values(token_class: int, values: np.ndarray) -> np.ndarray:
    if values.shape != (RELATIONAL_VALUE_WIDTH,):
        raise RelationalSubstrateMlxCacheError("relational token payload width differs")
    if token_class == 1:
        values[42:] = 0
    elif token_class in (2, 3):
        pass
    elif token_class == 4:
        member_count = int(values[0])
        continuation_start = 6 + 2 * member_count
        if member_count < 0 or continuation_start > RELATIONAL_VALUE_WIDTH:
            raise RelationalSubstrateMlxCacheError(
                "Salmon member count cannot define an R5 projection"
            )
        values[1:4] = 0
        values[5] = 0
        if continuation_start < RELATIONAL_VALUE_WIDTH:
            values[continuation_start:] = 0
    elif token_class == 5:
        values[2] = 0
    elif token_class == 6:
        values[3:] = 0
    else:
        raise RelationalSubstrateMlxCacheError("R5 projection received a nonminimal token class")
    return values


def _materialize_relational_sequences(
    sequences: list[list[list[tuple[int, np.ndarray]]]],
    *,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sequences or any(len(boards) != BOARD_SLOTS for boards in sequences):
        raise ValueError("relational parent sequences must contain four boards per group")
    observed = max(len(tokens) for boards in sequences for tokens in boards)
    if capacity <= 0 or observed > capacity:
        raise RelationalSubstrateMlxCacheError(
            "relational parent sequence exceeds its exact fixed capacity"
        )
    classes = np.zeros((len(sequences), BOARD_SLOTS, capacity), dtype=np.uint8)
    values = np.zeros(
        (
            len(sequences),
            BOARD_SLOTS,
            capacity,
            RELATIONAL_VALUE_WIDTH,
        ),
        dtype=np.int16,
    )
    mask = np.zeros((len(sequences), BOARD_SLOTS, capacity), dtype=np.bool_)
    counts = np.zeros((len(sequences), BOARD_SLOTS), dtype=np.int32)
    for group, boards in enumerate(sequences):
        for board, tokens in enumerate(boards):
            prior = 0
            for slot, (token_class, payload) in enumerate(tokens):
                if not 1 <= token_class <= RELATIONAL_CLASS_COUNT or token_class < prior:
                    raise RelationalSubstrateMlxCacheError(
                        "relational parent classes are invalid or noncanonical"
                    )
                prior = token_class
                payload = np.asarray(payload, dtype=np.int16)
                if payload.shape != (RELATIONAL_VALUE_WIDTH,):
                    raise RelationalSubstrateMlxCacheError(
                        "relational parent payload width differs"
                    )
                classes[group, board, slot] = token_class
                values[group, board, slot] = payload
                mask[group, board, slot] = True
            counts[group, board] = len(tokens)
    return classes, values, mask, counts


def _parent_class_name(token_class: int) -> str:
    names = (
        "rel_habitat_component",
        "rel_bear_component",
        "rel_elk_line",
        "rel_salmon_component",
        "rel_hawk_position",
        "rel_fox_center",
        "rel_frontier_summary",
        "rel_opportunity_summary",
        "r2_occupied",
        "r2_frontier",
        "r2_habitat_component",
        "r2_wildlife_motif",
    )
    return names[token_class - 1]


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "count": len(numeric),
        "minimum": int(numeric.min()),
        "mean": float(numeric.mean()),
        "p50": float(np.quantile(numeric, 0.50)),
        "p90": float(np.quantile(numeric, 0.90)),
        "p99": float(np.quantile(numeric, 0.99)),
        "maximum": int(numeric.max()),
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
        raise RelationalSubstrateMlxCacheError(f"{label} is not a lowercase BLAKE3 digest")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RelationalSubstrateMlxCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RelationalSubstrateMlxCacheError(f"{label} must be a JSON object")
    return value


__all__ = [
    "ADR_ID",
    "ARMS",
    "CONTROL_ARM",
    "EXPERIMENT_ID",
    "OPPORTUNITY_NAMES",
    "PROTOCOL_ID",
    "R5_ARM",
    "RELATIONAL_CLASS_COUNT",
    "RELATIONAL_VALUE_WIDTH",
    "S3_ARM",
    "S5_ARM",
    "S5_FEATURES",
    "RelationalParentBatch",
    "RelationalSubstrateBatch",
    "RelationalSubstrateMlxCache",
    "RelationalSubstrateMlxCacheError",
    "RelationalSubstrateMlxDataset",
    "open_data_verification_id",
    "open_data_verification_identity",
]
