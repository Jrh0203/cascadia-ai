"""Verified Rust-sidecar adapter for the S1 exact-supply MLX comparison."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.d6_contract import D6_CONTRACT_ID, D6_SCIENTIFIC_BLAKE3
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ACTION_ROTATION_SLICE,
    GRADED_ACTION_TILE_Q_INDEX,
    GRADED_ACTION_TILE_R_INDEX,
    GRADED_ACTION_WILDLIFE_Q_INDEX,
    GRADED_ACTION_WILDLIFE_R_INDEX,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleBatch,
    GradedOracleDataset,
)
from cascadia_mlx.hex_symmetry import (
    d6_transform_ids,
    transform_axial,
    transform_orientation_one_hot,
)

CACHE_SCHEMA_VERSION = 1
CACHE_SCHEMA = "s1-exact-supply-mlx-cache-v1"
EXPERIMENT_ID = "exact-semantic-supply-learned-comparison-v1"
PROTOCOL_ID = "s1-exact-semantic-supply-mlx-comparison-v1"
ADR_ID = "0147"
CATALOG_BLAKE3 = "362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19"

ARMS = (
    "c0-legacy-marginals",
    "t1-exact-counts",
    "t2-relational-supply",
)
EXACT_ARMS = frozenset(ARMS[1:])
RELATIONAL_ARM = ARMS[2]
LEGACY_SUPPLY_DIM = 30
EXACT_SUPPLY_DIM = 83
ARCHETYPE_COUNT = 75
SUPPLY_TOKEN_DIM = 32
EXACT_TOKEN_COUNT = 80
FRONTIER_FEATURE_DIM = 17
FRONTIER_NONE = 5

EXPECTED_SPLIT_COUNTS = {
    "train": (560, 2_135_111),
    "validation": (240, 860_203),
}
_DTYPES = {
    "|u1": np.dtype("u1"),
    "<u8": np.dtype("<u8"),
}
_TERRAIN_INDEX = {
    "Mountain": 0,
    "Forest": 1,
    "Prairie": 2,
    "Wetland": 3,
    "River": 4,
}
_EXACT_VECTOR_SCALES = np.asarray(
    [20.0] * 5 + [2.0] * ARCHETYPE_COUNT + [81.0, 79.0, 2.0],
    dtype=np.float32,
)
NORMALIZATION_CONTRACT = {
    "legacy_wildlife_divisor": 20.0,
    "legacy_tile_marginal_divisor": 81.0,
    "exact_wildlife_divisor": 20.0,
    "exact_archetype_count_divisor": 2.0,
    "exact_unseen_divisor": 81.0,
    "exact_drawable_divisor": 79.0,
    "exact_excluded_divisor": 2.0,
    "refill_target": "archetype_count_divided_by_unseen_count",
    "unused_c0_exact_slots": "zero",
}
ARM_INPUT_CONTRACTS = {
    ARMS[0]: {
        "state_dependent_supply_scalars": LEGACY_SUPPLY_DIM,
        "supply_vector_width": EXACT_SUPPLY_DIM,
        "supply_token_count": EXACT_TOKEN_COUNT,
        "exact_archetype_counts_visible": False,
        "drawable_and_exclusion_state_visible": False,
        "candidate_archetype_visible": False,
        "frontier_relation_visible": False,
        "unused_exact_inputs": "zero-or-state-independent-placeholder",
    },
    ARMS[1]: {
        "state_dependent_supply_scalars": EXACT_SUPPLY_DIM,
        "supply_vector_width": EXACT_SUPPLY_DIM,
        "supply_token_count": EXACT_TOKEN_COUNT,
        "exact_archetype_counts_visible": True,
        "drawable_and_exclusion_state_visible": True,
        "candidate_archetype_visible": False,
        "frontier_relation_visible": False,
        "unused_exact_inputs": "none",
    },
    ARMS[2]: {
        "state_dependent_supply_scalars": EXACT_SUPPLY_DIM,
        "supply_vector_width": EXACT_SUPPLY_DIM,
        "supply_token_count": EXACT_TOKEN_COUNT,
        "exact_archetype_counts_visible": True,
        "drawable_and_exclusion_state_visible": True,
        "candidate_archetype_visible": True,
        "frontier_relation_visible": True,
        "unused_exact_inputs": "none",
    },
}


class S1ExactSupplyCacheError(ValueError):
    """The exact-supply sidecar cannot prove the frozen S1 contract."""


@dataclass(frozen=True)
class S1ExactSupplyBatch:
    """A graded-oracle batch plus arm-routed public supply facts."""

    base: GradedOracleBatch
    supply_vector: mx.array
    staged_supply_vector: mx.array
    supply_tokens: mx.array
    supply_mask: mx.array
    refill_target: mx.array
    selected_archetype: mx.array
    frontier_features: mx.array

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
    candidates: int
    tensors: dict[str, np.memmap]
    group_rows: dict[int, int]


class S1ExactSupplyCache:
    """Content-addressed, checksum-verified exact semantic supply sidecar."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.manifest_path = self.root / "cache.json"
        self.manifest = _read_json(self.manifest_path, "S1 cache manifest")
        self._validate_envelope(require_complete=require_complete)
        self.catalog_features = _catalog_features(self.manifest["catalog"])
        self.splits = {
            split: self._load_split(
                split,
                verify_checksums=verify_checksums,
                verify_semantics=verify_semantics,
            )
            for split in EXPECTED_SPLIT_COUNTS
        }
        self._verified_candidate_groups: dict[str, set[int]] = {
            split: set() for split in EXPECTED_SPLIT_COUNTS
        }
        self._preverified_candidate_identity_proofs: dict[str, str] = {}

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def bind_dataset(
        self,
        root: str | Path,
        *,
        arm: str,
        verify_dataset_checksums: bool = True,
    ) -> S1ExactSupplyDataset:
        return S1ExactSupplyDataset(
            root,
            cache=self,
            arm=arm,
            verify_dataset_checksums=verify_dataset_checksums,
        )

    def verify_group_candidate_identity(
        self,
        split: str,
        group_id: int,
        action_hashes: np.ndarray,
    ) -> None:
        """Bind one full graded-oracle action order to the S1 sidecar."""
        source = self._split(split)
        try:
            cache_row = source.group_rows[group_id]
        except KeyError as error:
            raise S1ExactSupplyCacheError(
                f"graded-oracle group {group_id} is absent from the S1 cache"
            ) from error
        offsets = source.tensors["candidate_offsets"]
        start = int(offsets[cache_row])
        end = int(offsets[cache_row + 1])
        hashes = np.asarray(action_hashes, dtype=np.uint8)
        if hashes.shape != (end - start, 32):
            raise S1ExactSupplyCacheError(
                "graded-oracle action count disagrees with S1 candidate identity"
            )
        observed = _candidate_identity_hash(
            group_id,
            hashes,
            np.asarray(source.tensors["staged_wildlife_counts"][start:end]),
            np.asarray(source.tensors["selected_archetype_ids"][start:end]),
            np.asarray(source.tensors["frontier_requirements"][start:end]),
            np.asarray(source.tensors["selected_compatibility"][start:end]),
        )
        expected = bytes(source.tensors["candidate_identity_hashes"][cache_row])
        if observed != expected:
            raise S1ExactSupplyCacheError("graded-oracle action identity disagrees with S1 cache")
        self._verified_group_set(split).add(group_id)

    def register_preverified_group_candidate_identities(
        self,
        split: str,
        group_ids: set[int],
        *,
        proof_id: str,
    ) -> None:
        """Accept an exact candidate-order proof produced by exhaustive preflight."""
        _require_blake3(proof_id, "open-data verification proof")
        source = self._split(split)
        expected = set(source.group_rows)
        if not group_ids or not group_ids.issubset(expected):
            raise S1ExactSupplyCacheError(
                "preverified S1 candidate identities contain absent or empty groups"
            )
        existing = self._preverified_candidate_identity_proofs.get(split)
        if existing is not None and existing != proof_id:
            raise S1ExactSupplyCacheError(
                "S1 split was already bound to a different verification proof"
            )
        self._preverified_candidate_identity_proofs[split] = proof_id
        self._verified_group_set(split).update(expected)

    def materialize(
        self,
        split: str,
        arm: str,
        batch: GradedOracleBatch,
        *,
        source_candidate_indices: np.ndarray | None = None,
    ) -> S1ExactSupplyBatch:
        if arm not in ARMS:
            raise ValueError(f"unknown S1 arm: {arm}")
        source = self._split(split)
        group_ids = np.asarray(batch.group_id, dtype=np.uint64)
        try:
            rows = np.asarray(
                [source.group_rows[int(group_id)] for group_id in group_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise S1ExactSupplyCacheError(
                f"graded-oracle group {error.args[0]} is absent from the S1 cache"
            ) from error

        public_hashes = np.asarray(batch.public_state_hash, dtype=np.uint8)
        cached_public_hashes = np.asarray(source.tensors["public_state_hashes"][rows])
        if not np.array_equal(public_hashes, cached_public_hashes):
            raise S1ExactSupplyCacheError("graded-oracle public-state hash disagrees with S1 cache")

        candidate_mask = np.asarray(batch.candidate_mask, dtype=np.bool_)
        candidate_counts = candidate_mask.sum(axis=1).astype(np.int64)
        max_candidates = candidate_mask.shape[1]
        exact_values = np.asarray(source.tensors["exact_supply_values"][rows]).copy()
        staged_wildlife = np.zeros((len(rows), max_candidates, 5), dtype=np.uint8)
        selected_archetype = np.zeros((len(rows), max_candidates), dtype=np.int32)
        frontier = np.zeros(
            (len(rows), max_candidates, FRONTIER_FEATURE_DIM),
            dtype=np.float32,
        )

        offsets = source.tensors["candidate_offsets"]
        action_hashes = np.asarray(batch.action_hash, dtype=np.uint8)
        action_features = np.asarray(batch.action_features, dtype=np.float32)
        selected_sources = (
            None
            if source_candidate_indices is None
            else np.asarray(source_candidate_indices, dtype=np.int64)
        )
        if selected_sources is not None and selected_sources.shape != candidate_mask.shape:
            raise ValueError("source candidate indices must match the padded candidate mask")
        for batch_row, (cache_row, candidate_count) in enumerate(
            zip(rows, candidate_counts, strict=True)
        ):
            start = int(offsets[cache_row])
            end = int(offsets[cache_row + 1])
            if selected_sources is None:
                if end - start != candidate_count:
                    raise S1ExactSupplyCacheError(
                        "graded-oracle candidate width disagrees with S1 cache"
                    )
                source_rows = np.arange(start, end, dtype=np.int64)
            else:
                group_id = int(group_ids[batch_row])
                if group_id not in self._verified_group_set(split):
                    raise S1ExactSupplyCacheError(
                        "selected S1 materialization requires a verified full-group identity"
                    )
                indices = selected_sources[batch_row, :candidate_count]
                if (
                    np.any(indices < 0)
                    or np.any(indices >= end - start)
                    or np.any(np.diff(indices) <= 0)
                ):
                    raise S1ExactSupplyCacheError(
                        "selected S1 source indices are unordered, duplicated, or out of range"
                    )
                source_rows = start + indices
            staged = np.asarray(source.tensors["staged_wildlife_counts"][source_rows])
            archetypes = np.asarray(source.tensors["selected_archetype_ids"][source_rows])
            requirements = np.asarray(source.tensors["frontier_requirements"][source_rows])
            compatibility = np.asarray(source.tensors["selected_compatibility"][source_rows])
            staged_wildlife[batch_row, :candidate_count] = staged
            selected_archetype[batch_row, :candidate_count] = archetypes
            frontier[batch_row, :candidate_count] = _frontier_features(
                requirements,
                compatibility,
                action_features[batch_row, :candidate_count, GRADED_ACTION_ROTATION_SLICE],
            )
            if selected_sources is None:
                observed_hash = _candidate_identity_hash(
                    int(group_ids[batch_row]),
                    action_hashes[batch_row, :candidate_count],
                    staged,
                    archetypes,
                    requirements,
                    compatibility,
                )
                expected_hash = bytes(source.tensors["candidate_identity_hashes"][cache_row])
                if observed_hash != expected_hash:
                    raise S1ExactSupplyCacheError(
                        "graded-oracle action identity disagrees with S1 cache"
                    )
                self._verified_group_set(split).add(int(group_ids[batch_row]))

        return _route_arm_inputs(
            arm=arm,
            batch=batch,
            exact_values=exact_values,
            staged_wildlife=staged_wildlife,
            selected_archetype=selected_archetype,
            frontier_features=frontier,
            catalog_features=self.catalog_features,
        )

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
            or manifest.get("catalog_blake3") != CATALOG_BLAKE3
        ):
            raise S1ExactSupplyCacheError("unsupported S1 exact-supply cache envelope")
        if self.root.name != manifest.get("cache_id"):
            raise S1ExactSupplyCacheError("S1 cache directory is not its content address")
        identity = manifest.get("scientific_identity")
        if not isinstance(identity, dict) or _canonical_blake3(identity) != manifest.get(
            "cache_id"
        ):
            raise S1ExactSupplyCacheError("S1 cache content address is invalid")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise S1ExactSupplyCacheError("production S1 cache must cover the full open corpus")
        if require_complete:
            for split, (groups, candidates) in EXPECTED_SPLIT_COUNTS.items():
                raw = manifest.get("splits", {}).get(split)
                if (
                    not isinstance(raw, dict)
                    or raw.get("complete_open_split") is not True
                    or raw.get("groups") != groups
                    or raw.get("candidates") != candidates
                ):
                    raise S1ExactSupplyCacheError(
                        f"production S1 cache does not fully cover {split}"
                    )
        if not isinstance(manifest.get("catalog"), list) or len(manifest["catalog"]) != 75:
            raise S1ExactSupplyCacheError("S1 cache catalog must contain 75 archetypes")
        _validate_collision_witness(manifest.get("collision_witness"))
        hidden = manifest.get("hidden_information")
        if (
            not isinstance(hidden, dict)
            or hidden.get("public_position_records_only") is not True
            or hidden.get("public_supply_only") is not True
            or any(
                hidden.get(field) is not False
                for field in (
                    "hidden_stack_order_read",
                    "hidden_wildlife_order_read",
                    "excluded_tile_identities_read",
                    "future_refills_read",
                    "sealed_test_opened",
                    "gameplay_opened",
                )
            )
        ):
            raise S1ExactSupplyCacheError("S1 cache violates the hidden-information boundary")
        exporter = manifest.get("exporter")
        if not isinstance(exporter, dict) or not isinstance(exporter.get("source"), dict):
            raise S1ExactSupplyCacheError("S1 cache exporter provenance is missing")
        _require_blake3(exporter.get("executable_blake3"), "exporter executable")
        _require_blake3(
            exporter["source"].get("v2_source_blake3"),
            "exporter source",
        )

    def _load_split(
        self,
        split: str,
        *,
        verify_checksums: bool,
        verify_semantics: bool,
    ) -> _Split:
        raw = self.manifest.get("splits", {}).get(split)
        if not isinstance(raw, dict):
            raise S1ExactSupplyCacheError(f"S1 cache is missing the {split} split")
        groups = _positive_integer(raw.get("groups"), f"{split} groups")
        candidates = _positive_integer(raw.get("candidates"), f"{split} candidates")
        expected_groups, expected_candidates = EXPECTED_SPLIT_COUNTS[split]
        complete = raw.get("complete_open_split") is True
        if complete and (groups, candidates) != (expected_groups, expected_candidates):
            raise S1ExactSupplyCacheError(f"complete {split} S1 cache has wrong coverage")
        files = raw.get("files")
        if not isinstance(files, dict):
            raise S1ExactSupplyCacheError(f"{split} S1 tensor manifest is missing")
        expected_shapes = _expected_shapes(groups, candidates)
        if set(files) != set(expected_shapes):
            raise S1ExactSupplyCacheError(f"{split} S1 tensor set drifted")
        tensors: dict[str, np.memmap] = {}
        for name, shape in expected_shapes.items():
            tensor = self._tensor(files[name], shape)
            if verify_checksums and _checksum(tensor.path) != tensor.blake3:
                raise S1ExactSupplyCacheError(f"{split} S1 tensor checksum failed: {name}")
            tensors[name] = tensor.memmap()

        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        if len(np.unique(group_ids)) != groups:
            raise S1ExactSupplyCacheError(f"{split} S1 group IDs are not unique")
        offsets = np.asarray(tensors["candidate_offsets"], dtype=np.uint64)
        if offsets[0] != 0 or offsets[-1] != candidates or np.any(offsets[1:] <= offsets[:-1]):
            raise S1ExactSupplyCacheError(f"{split} S1 candidate offsets are invalid")
        if verify_semantics:
            exact = np.asarray(tensors["exact_supply_values"], dtype=np.uint8)
            if (
                np.any(exact[:, 80] != exact[:, 5:80].sum(axis=1))
                or np.any(exact[:, 80] != exact[:, 81] + exact[:, 82])
                or np.any(exact[:, 82] != 2)
            ):
                raise S1ExactSupplyCacheError(f"{split} S1 exact-supply conservation failed")
            archetypes = np.asarray(tensors["selected_archetype_ids"], dtype=np.uint8)
            requirements = np.asarray(tensors["frontier_requirements"], dtype=np.uint8)
            compatibility = np.asarray(tensors["selected_compatibility"], dtype=np.uint8)
            if (
                np.any(archetypes >= ARCHETYPE_COUNT)
                or np.any(requirements > FRONTIER_NONE)
                or np.any(compatibility[:, :6] > 6)
                or np.any(compatibility[:, 6] >= 64)
                or np.any(compatibility[:, 7] > 6)
            ):
                raise S1ExactSupplyCacheError(f"{split} S1 candidate facts are out of range")
            checks = raw.get("checks")
            if not isinstance(checks, dict) or not _checks_cover_split(checks, groups, candidates):
                raise S1ExactSupplyCacheError(f"{split} S1 semantic audit counts are incomplete")
        return _Split(
            groups=groups,
            candidates=candidates,
            tensors=tensors,
            group_rows={int(group_id): row for row, group_id in enumerate(group_ids)},
        )

    def _tensor(self, raw: object, expected_shape: tuple[int, ...]) -> _Tensor:
        if not isinstance(raw, dict) or raw.get("dtype") not in _DTYPES:
            raise S1ExactSupplyCacheError("S1 tensor specification is malformed")
        if raw.get("shape") != list(expected_shape):
            raise S1ExactSupplyCacheError("S1 tensor shape drifted")
        path = self.root / str(raw.get("file"))
        if path.parent != self.root or not path.is_file():
            raise S1ExactSupplyCacheError("S1 tensor path escapes or is absent")
        dtype = _DTYPES[str(raw["dtype"])]
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        if raw.get("bytes") != expected_bytes or path.stat().st_size != expected_bytes:
            raise S1ExactSupplyCacheError("S1 tensor byte count drifted")
        digest = raw.get("blake3")
        _require_blake3(digest, "S1 tensor")
        return _Tensor(
            path=path,
            dtype=dtype,
            shape=expected_shape,
            blake3=str(digest),
        )

    def _split(self, split: str) -> _Split:
        try:
            return self.splits[split]
        except KeyError as error:
            raise ValueError("S1 cache split must be train or validation") from error

    def _verified_group_set(self, split: str) -> set[int]:
        registry = getattr(self, "_verified_candidate_groups", None)
        if registry is None:
            registry = {name: set() for name in EXPECTED_SPLIT_COUNTS}
            self._verified_candidate_groups = registry
        return registry[split]


class S1ExactSupplyDataset:
    """The same open graded-oracle rows augmented from one verified sidecar."""

    def __init__(
        self,
        root: str | Path,
        *,
        cache: S1ExactSupplyCache,
        arm: str,
        verify_dataset_checksums: bool = True,
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown S1 arm: {arm}")
        self.base = GradedOracleDataset(root, verify_checksums=verify_dataset_checksums)
        if self.base.split not in EXPECTED_SPLIT_COUNTS:
            raise S1ExactSupplyCacheError("S1 accepts only open train and validation splits")
        self.root = self.base.root
        self.manifest = self.base.manifest
        self.split = self.base.split
        self.group_count = self.base.group_count
        self.candidate_count = self.base.candidate_count
        self.cache = cache
        self.arm = arm
        sidecar = cache.manifest["splits"][self.split]
        manifest_hash = _checksum(self.root / "dataset.json")
        if (
            sidecar.get("dataset_id") != self.manifest.get("dataset_id")
            or sidecar.get("dataset_manifest_blake3") != manifest_hash
            or sidecar.get("groups") != self.group_count
            or sidecar.get("candidates") != self.candidate_count
        ):
            raise S1ExactSupplyCacheError("graded-oracle dataset identity disagrees with S1 cache")

    def batches(
        self,
        group_batch_size: int,
        *,
        maximum_actions_per_batch: int | None = GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions: int | None = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[S1ExactSupplyBatch]:
        for batch in self.base.batches(
            group_batch_size,
            maximum_actions_per_batch=maximum_actions_per_batch,
            maximum_group_actions=maximum_group_actions,
            shuffle=shuffle,
            seed=seed,
        ):
            yield self.cache.materialize(self.split, self.arm, batch)


def transform_s1_exact_supply_batch(
    batch: S1ExactSupplyBatch,
    transforms: int | np.ndarray,
) -> S1ExactSupplyBatch:
    """Apply the full Rust-owned D6 action to geometry; supply facts stay invariant."""
    groups = batch.base.action_features.shape[0]
    transform_ids = d6_transform_ids(transforms, groups)
    base = batch.base

    boards = base.board_entities
    board_q, board_r = transform_axial(boards[..., 0], boards[..., 1], transform_ids)
    board_dual = boards[..., 12] < 0.5
    board_rotations = transform_orientation_one_hot(
        boards[..., 13:19],
        transform_ids,
        base.board_mask,
        is_dual_terrain=board_dual,
    )
    transformed_boards = mx.concatenate(
        [
            board_q[..., None],
            board_r[..., None],
            boards[..., 2:13],
            board_rotations,
            boards[..., 19:],
        ],
        axis=-1,
    )

    actions = base.action_features
    tile_q, tile_r = transform_axial(
        actions[..., GRADED_ACTION_TILE_Q_INDEX],
        actions[..., GRADED_ACTION_TILE_R_INDEX],
        transform_ids,
    )
    wildlife_q, wildlife_r = transform_axial(
        actions[..., GRADED_ACTION_WILDLIFE_Q_INDEX],
        actions[..., GRADED_ACTION_WILDLIFE_R_INDEX],
        transform_ids,
    )
    action_dual = actions[..., 22] < 0.5
    action_rotations = transform_orientation_one_hot(
        actions[..., GRADED_ACTION_ROTATION_SLICE],
        transform_ids,
        base.candidate_mask,
        is_dual_terrain=action_dual,
    )
    transformed_actions = mx.concatenate(
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
        base=replace(
            base,
            board_entities=transformed_boards,
            action_features=transformed_actions,
        ),
    )


def randomly_transform_s1_exact_supply_batch(
    batch: S1ExactSupplyBatch,
    seed: int,
) -> S1ExactSupplyBatch:
    """Sample one uniform D6 transform per complete decision group."""
    rng = np.random.default_rng(seed)
    transforms = rng.integers(0, 12, size=batch.base.action_features.shape[0])
    return transform_s1_exact_supply_batch(batch, transforms)


def _route_arm_inputs(
    *,
    arm: str,
    batch: GradedOracleBatch,
    exact_values: np.ndarray,
    staged_wildlife: np.ndarray,
    selected_archetype: np.ndarray,
    frontier_features: np.ndarray,
    catalog_features: np.ndarray,
) -> S1ExactSupplyBatch:
    groups, candidates = staged_wildlife.shape[:2]
    candidate_mask = np.asarray(batch.candidate_mask, dtype=np.bool_)
    exact_vector = exact_values.astype(np.float32) / _EXACT_VECTOR_SCALES
    refill_denominator = np.maximum(exact_values[:, 80:81].astype(np.float32), 1.0)
    refill_target = exact_values[:, 5:80].astype(np.float32) / refill_denominator

    if arm == ARMS[0]:
        supply_vector = np.zeros((groups, EXACT_SUPPLY_DIM), dtype=np.float32)
        supply_vector[:, :LEGACY_SUPPLY_DIM] = np.asarray(batch.public_supply)
        staged_supply = np.zeros((groups, candidates, EXACT_SUPPLY_DIM), dtype=np.float32)
        staged_supply[:, :, :LEGACY_SUPPLY_DIM] = np.asarray(batch.staged_public_supply)
        supply_tokens = np.zeros(
            (groups, EXACT_TOKEN_COUNT, SUPPLY_TOKEN_DIM),
            dtype=np.float32,
        )
        supply_tokens[:, :LEGACY_SUPPLY_DIM, 0] = np.asarray(batch.public_supply)
        supply_tokens[:, :LEGACY_SUPPLY_DIM, 5] = 1.0
    else:
        supply_vector = exact_vector
        staged_raw = np.broadcast_to(
            exact_values[:, None, :],
            (groups, candidates, EXACT_SUPPLY_DIM),
        ).copy()
        staged_raw[:, :, :5] = staged_wildlife
        staged_supply = staged_raw.astype(np.float32) / _EXACT_VECTOR_SCALES
        supply_tokens = np.zeros(
            (groups, EXACT_TOKEN_COUNT, SUPPLY_TOKEN_DIM),
            dtype=np.float32,
        )
        supply_tokens[:, :5, 0] = exact_values[:, :5].astype(np.float32) / 20.0
        supply_tokens[:, :5, 3] = 1.0
        supply_tokens[:, :5, 6:11] = np.eye(5, dtype=np.float32)[None, :, :]
        supply_tokens[:, 5:, 0] = exact_values[:, 5:80].astype(np.float32) / 2.0
        supply_tokens[:, 5:, 1] = refill_target
        supply_tokens[:, 5:, 4] = 1.0
        supply_tokens[:, 5:, 6:29] = catalog_features[None, :, :]
        supply_tokens[:, :, 29] = exact_values[:, None, 80] / 81.0
        supply_tokens[:, :, 30] = exact_values[:, None, 81] / 79.0
        supply_tokens[:, :, 31] = exact_values[:, None, 82] / 2.0

    if arm != RELATIONAL_ARM:
        selected_archetype = np.zeros_like(selected_archetype)
        frontier_features = np.zeros_like(frontier_features)
    supply_mask = np.ones((groups, EXACT_TOKEN_COUNT), dtype=np.bool_)
    staged_supply *= candidate_mask[..., None]
    selected_archetype = selected_archetype * candidate_mask
    frontier_features *= candidate_mask[..., None]
    return S1ExactSupplyBatch(
        base=batch,
        supply_vector=mx.array(supply_vector),
        staged_supply_vector=mx.array(staged_supply),
        supply_tokens=mx.array(supply_tokens),
        supply_mask=mx.array(supply_mask),
        refill_target=mx.array(refill_target),
        selected_archetype=mx.array(selected_archetype),
        frontier_features=mx.array(frontier_features),
    )


def _frontier_features(
    requirements: np.ndarray,
    compatibility: np.ndarray,
    action_rotation: np.ndarray,
) -> np.ndarray:
    count = len(requirements)
    features = np.zeros((count, FRONTIER_FEATURE_DIM), dtype=np.float32)
    for terrain in range(5):
        features[:, terrain] = np.sum(requirements == terrain, axis=1) / 6.0
    present = np.sum(requirements != FRONTIER_NONE, axis=1)
    rotations = np.argmax(action_rotation, axis=1)
    matching = compatibility[:, :6]
    masks = compatibility[:, 6]
    features[:, 5] = present / 6.0
    features[:, 6] = matching[np.arange(count), rotations] / 6.0
    features[:, 7] = ((masks >> rotations) & 1).astype(np.float32)
    features[:, 8] = compatibility[:, 7] / 6.0
    features[:, 9] = np.asarray([int(value).bit_count() for value in masks]) / 6.0
    for matched_edges in range(7):
        features[:, 10 + matched_edges] = (
            np.sum(
                matching == matched_edges,
                axis=1,
            )
            / 6.0
        )
    return features


def _catalog_features(raw_catalog: object) -> np.ndarray:
    if not isinstance(raw_catalog, list) or len(raw_catalog) != ARCHETYPE_COUNT:
        raise S1ExactSupplyCacheError("S1 semantic catalog is malformed")
    features = np.zeros((ARCHETYPE_COUNT, 23), dtype=np.float32)
    for expected_id, definition in enumerate(raw_catalog):
        if not isinstance(definition, dict) or definition.get("id") != expected_id:
            raise S1ExactSupplyCacheError("S1 semantic catalog IDs are not contiguous")
        archetype = definition.get("archetype")
        if not isinstance(archetype, dict):
            raise S1ExactSupplyCacheError("S1 semantic archetype is malformed")
        try:
            primary = _TERRAIN_INDEX[str(archetype["primary_terrain"])]
            secondary_raw = archetype["secondary_terrain"]
            secondary = 5 if secondary_raw is None else _TERRAIN_INDEX[str(secondary_raw)]
            edges = [_TERRAIN_INDEX[str(value)] for value in archetype["directed_edges"]]
            wildlife = int(archetype["wildlife"])
            keystone = bool(archetype["keystone"])
            multiplicity = int(definition["standard_tile_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise S1ExactSupplyCacheError("S1 semantic catalog field is invalid") from error
        if len(edges) != 6 or wildlife < 0 or wildlife >= 32 or multiplicity not in {1, 2}:
            raise S1ExactSupplyCacheError("S1 semantic catalog value is out of range")
        features[expected_id, primary] = 1.0
        features[expected_id, 5 + secondary] = 1.0
        for bit in range(5):
            features[expected_id, 11 + bit] = float((wildlife >> bit) & 1)
        features[expected_id, 16] = float(keystone)
        for terrain in range(5):
            features[expected_id, 17 + terrain] = edges.count(terrain) / 6.0
        features[expected_id, 22] = multiplicity / 2.0
    return features


def collision_witness_arm_inputs(
    cache: S1ExactSupplyCache,
    arm: str,
) -> dict[str, np.ndarray]:
    """Return the frozen factual collision in the selected arm's input coordinates."""
    if arm not in ARMS:
        raise ValueError(f"unknown S1 arm: {arm}")
    identity = cache.manifest["collision_witness"]["identity"]
    legacy = np.asarray(identity["legacy_supply_values"], dtype=np.float32)
    legacy /= np.asarray([20.0] * 5 + [81.0] * 25, dtype=np.float32)
    left_counts = np.asarray(identity["left_refill_numerators"], dtype=np.float32)
    right_counts = np.asarray(identity["right_refill_numerators"], dtype=np.float32)
    denominator = float(identity["refill_denominator"])

    def vector(counts: np.ndarray) -> np.ndarray:
        if arm == ARMS[0]:
            values = np.zeros(EXACT_SUPPLY_DIM, dtype=np.float32)
            values[:LEGACY_SUPPLY_DIM] = legacy
            return values
        raw = np.concatenate(
            [
                np.zeros(5, dtype=np.float32),
                counts,
                np.asarray([denominator, denominator, 0.0], dtype=np.float32),
            ]
        )
        return raw / _EXACT_VECTOR_SCALES

    return {
        "left_supply_vector": vector(left_counts),
        "right_supply_vector": vector(right_counts),
        "left_refill_target": left_counts / denominator,
        "right_refill_target": right_counts / denominator,
    }


def _candidate_identity_hash(
    group_id: int,
    action_hashes: np.ndarray,
    staged_wildlife: np.ndarray,
    archetypes: np.ndarray,
    requirements: np.ndarray,
    compatibility: np.ndarray,
) -> bytes:
    digest = blake3.blake3()
    digest.update(b"S1MLXCAND1\0")
    digest.update(group_id.to_bytes(8, "little", signed=False))
    digest.update(len(action_hashes).to_bytes(8, "little", signed=False))
    for values in zip(
        action_hashes,
        staged_wildlife,
        archetypes,
        requirements,
        compatibility,
        strict=True,
    ):
        action_hash, wildlife, archetype, frontier, compatible = values
        digest.update(np.asarray(action_hash, dtype=np.uint8).tobytes())
        digest.update(np.asarray(wildlife, dtype=np.uint8).tobytes())
        digest.update(bytes([int(archetype)]))
        digest.update(np.asarray(frontier, dtype=np.uint8).tobytes())
        digest.update(np.asarray(compatible, dtype=np.uint8).tobytes())
    return digest.digest()


def _expected_shapes(groups: int, candidates: int) -> dict[str, tuple[int, ...]]:
    return {
        "group_ids": (groups,),
        "public_state_hashes": (groups, 32),
        "exact_supply_values": (groups, EXACT_SUPPLY_DIM),
        "exact_supply_hashes": (groups, 32),
        "candidate_offsets": (groups + 1,),
        "staged_wildlife_counts": (candidates, 5),
        "selected_archetype_ids": (candidates,),
        "frontier_requirements": (candidates, 6),
        "selected_compatibility": (candidates, 8),
        "candidate_identity_hashes": (groups, 32),
    }


def _checks_cover_split(checks: dict[str, object], groups: int, candidates: int) -> bool:
    return (
        all(
            checks.get(field) == groups
            for field in (
                "csssup_round_trips",
                "legacy_parity_groups",
                "wildlife_parity_groups",
                "tile_count_conservation_groups",
                "drawable_conservation_groups",
                "hidden_exclusion_count_groups",
            )
        )
        and all(
            checks.get(field) == candidates
            for field in (
                "staged_legacy_tile_parity_candidates",
                "staged_wildlife_parity_candidates",
                "market_tile_identity_candidates",
                "frontier_compatibility_candidates",
            )
        )
        and all(
            checks.get(field) == 0
            for field in (
                "hidden_order_fields_read",
                "excluded_tile_identity_fields_read",
                "future_refill_fields_read",
            )
        )
    )


def _validate_collision_witness(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"witness_id", "identity"}:
        raise S1ExactSupplyCacheError("S1 cache lacks the ADR 0143 collision witness")
    identity = value["identity"]
    if (
        not isinstance(identity, dict)
        or _canonical_blake3(identity) != value["witness_id"]
        or identity.get("schema") != "adr-0143-factual-legacy-collision-v1"
        or identity.get("left_physical_tile_ids") != [0, 23]
        or identity.get("right_physical_tile_ids") != [2, 20]
        or identity.get("left_archetype_ids") != [26, 72]
        or identity.get("right_archetype_ids") != [24, 74]
        or identity.get("legacy_marginals_equal") is not True
        or identity.get("refill_laws_differ") is not True
        or identity.get("refill_denominator") != 2
    ):
        raise S1ExactSupplyCacheError("S1 collision witness identity drifted")
    legacy = identity.get("legacy_supply_values")
    left = identity.get("left_refill_numerators")
    right = identity.get("right_refill_numerators")
    if (
        not isinstance(legacy, list)
        or len(legacy) != LEGACY_SUPPLY_DIM
        or not isinstance(left, list)
        or len(left) != ARCHETYPE_COUNT
        or not isinstance(right, list)
        or len(right) != ARCHETYPE_COUNT
        or sum(left) != 2
        or sum(right) != 2
        or left == right
    ):
        raise S1ExactSupplyCacheError("S1 collision witness values are invalid")
    _require_blake3(identity.get("left_supply_blake3"), "left collision supply")
    _require_blake3(identity.get("right_supply_blake3"), "right collision supply")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise S1ExactSupplyCacheError(f"{label} must be a positive integer")
    return value


def _require_blake3(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise S1ExactSupplyCacheError(f"{label} BLAKE3 is invalid")


def _canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise S1ExactSupplyCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise S1ExactSupplyCacheError(f"{label} must be a JSON object")
    return value


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


S1_D6_CONTRACT = {
    "contract_id": D6_CONTRACT_ID,
    "scientific_blake3": D6_SCIENTIFIC_BLAKE3,
    "transform_ids": list(range(12)),
}
