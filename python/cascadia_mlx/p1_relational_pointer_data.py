"""Streaming selected-prefix pointer batches over frozen factor and R2 caches."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.d6_contract import D6_CONTRACT
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    DRAFT_FACTOR_DIM,
    STAGED_PUBLIC_DIM,
    STAGES,
    TILE_FACTOR_DIM,
    HierarchicalFactorCache,
)
from cascadia_mlx.p1_relational_pointer_model import (
    DESTINATION_EXISTING_TILE,
    DESTINATION_NEW_TILE,
    DESTINATION_NONE,
    DRAFT_OBSERVABLE_DIM,
    STAGE_ITEM_DIMS,
    STAGE_QUERY_DIMS,
    WILDLIFE_QUERY_DIM,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    TOKEN_FEATURES,
    TOKEN_PAYLOAD_WIDTH,
    materialize_token_features,
    transform_token_payload_in_place,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_cache import RELATIONAL_VALUE_WIDTH

COORDINATE_SCALE = 24.0
R2_OCCUPIED_TOKEN = 1
R2_FRONTIER_TOKEN = 2
WILDLIFE_QUERY_TILE_OFFSET = DRAFT_FACTOR_DIM + STAGED_PUBLIC_DIM

DEFAULT_FACTOR_CACHE = Path(
    "artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache"
)
DEFAULT_R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)

_DUAL_ROTATION_TABLES = np.asarray(
    D6_CONTRACT.dual_tile_rotation_tables,
    dtype=np.int16,
)
_SINGLE_ROTATION_TABLES = np.asarray(
    D6_CONTRACT.single_tile_rotation_tables,
    dtype=np.int16,
)


class RelationalPointerDataError(ValueError):
    """The selected-prefix pointer inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class PointerParentBatch:
    """Exact-R2 parent tensors accepted by the warm-start-compatible encoder."""

    r2_token_features: mx.array
    r2_token_types: mx.array
    r2_token_mask: mx.array
    relational_values: mx.array
    relational_classes: mx.array
    relational_mask: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array


@dataclass(frozen=True)
class PointerStageBatch:
    """One padded batch of conditional legal-factor pointer choices."""

    shard_index: int
    parent: PointerParentBatch
    query_parent_indices: mx.array
    query_features: mx.array
    item_features: mx.array
    item_pointer_indices: mx.array
    item_rotations: mx.array
    item_kinds: mx.array
    query_tile_pointer_indices: mx.array
    query_tile_rotations: mx.array
    item_mask: mx.array
    expected_rank: mx.array
    expected_rank_mask: mx.array
    target: mx.array
    parent_group_ids: np.ndarray
    parent_transform_ids: np.ndarray
    source_query_indices: np.ndarray
    source_item_indices: np.ndarray
    transform_ids: np.ndarray


@dataclass(frozen=True)
class PointerShardMetadata:
    """Lossless item-to-token maps for one immutable factor shard."""

    group_r3_rows: np.ndarray
    tile_pointer_indices: np.ndarray
    tile_rotations: np.ndarray
    tile_dual_terrain: np.ndarray
    wildlife_pointer_indices: np.ndarray
    wildlife_kinds: np.ndarray
    wildlife_query_tile_pointer_indices: np.ndarray
    wildlife_query_tile_rotations: np.ndarray
    wildlife_query_tile_dual_terrain: np.ndarray


class RelationalPointerCorpus:
    """Verified streaming join of historical labels and exact-R2 parents."""

    def __init__(
        self,
        *,
        split: str,
        factor_cache: str | Path = DEFAULT_FACTOR_CACHE,
        r3_cache: str | Path = DEFAULT_R3_CACHE,
        verify_r3_checksums: bool = True,
        verify_r3_semantics: bool = True,
    ):
        if split not in {"train", "validation"}:
            raise RelationalPointerDataError("pointer corpus split must be train or validation")
        self.split = split
        self.factor = HierarchicalFactorCache(Path(factor_cache) / split)
        self.r3 = R3ActionEditMlxCache(
            r3_cache,
            verify_checksums=verify_r3_checksums,
            verify_semantics=verify_r3_semantics,
        )
        self.source = self.r3.splits[split]
        self._metadata: dict[tuple[int, str], PointerShardMetadata] = {}
        if (
            self.factor.split != split
            or self.factor.group_count != self.source.groups
            or self.factor.candidate_count != self.source.source_candidates
            or self.factor.manifest.get("dataset_manifest_blake3")
            != self.r3.manifest["splits"][split].get("dataset_manifest_blake3")
        ):
            raise RelationalPointerDataError(
                "factor and exact-R2 cache identities do not align"
            )

    @property
    def group_count(self) -> int:
        return self.factor.group_count

    @property
    def candidate_count(self) -> int:
        return self.factor.candidate_count

    def iter_stage_batches(
        self,
        *,
        stage: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        epoch: int,
        d6_augment: bool = True,
    ) -> Iterator[PointerStageBatch]:
        """Stream every query exactly once from each immutable source shard."""
        if stage not in STAGES:
            raise RelationalPointerDataError("unknown pointer stage")
        if batch_size <= 0 or seed < 0 or epoch < 0:
            raise RelationalPointerDataError("pointer batch schedule is invalid")
        for shard_index, arrays in enumerate(self.factor.iter_shards()):
            metadata_key = (shard_index, stage)
            metadata = self._metadata.get(metadata_key)
            if metadata is None:
                metadata = build_pointer_metadata(
                    arrays,
                    r3_tensors=self.source.tensors,
                    r3_group_rows=self.source.group_rows,
                    stages=(stage,),
                )
                self._metadata[metadata_key] = metadata
            offsets = np.asarray(
                arrays[f"{stage}_query_offsets"],
                dtype=np.int64,
            )
            query_count = len(offsets) - 1
            order = np.arange(query_count, dtype=np.int64)
            if shuffle:
                np.random.default_rng(
                    _schedule_seed(seed, epoch, shard_index)
                ).shuffle(order)
            for start in range(0, query_count, batch_size):
                selected = order[start : start + batch_size]
                yield materialize_pointer_batch(
                    arrays,
                    metadata,
                    r3_tensors=self.source.tensors,
                    stage=stage,
                    selected_queries=selected,
                    seed=seed,
                    epoch=epoch,
                    shard_index=shard_index,
                    d6_augment=d6_augment,
                )


def build_pointer_metadata(
    arrays: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    r3_tensors: dict[str, np.ndarray],
    r3_group_rows: dict[int, int],
    stages: Sequence[str] | None = None,
) -> PointerShardMetadata:
    """Resolve every legal item to one stable padded active-board token index."""
    requested = set(stages or ("tile", "wildlife"))
    if not requested.issubset(STAGES):
        raise RelationalPointerDataError("pointer metadata names an unknown stage")
    group_ids = np.asarray(arrays["group_id"], dtype=np.uint64)
    try:
        group_r3_rows = np.asarray(
            [r3_group_rows[int(group_id)] for group_id in group_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RelationalPointerDataError(
            "factor shard names a group absent from exact R2"
        ) from error

    tile_pointer_indices = np.empty(0, dtype=np.int32)
    tile_rotations = np.empty(0, dtype=np.int16)
    tile_dual_terrain = np.empty(0, dtype=np.bool_)
    if "tile" in requested:
        tile_features = np.asarray(
            arrays["tile_item_features"],
            dtype=np.float32,
        )
        tile_context = np.asarray(
            arrays["tile_query_context"],
            dtype=np.float32,
        )
        tile_offsets = np.asarray(
            arrays["tile_query_offsets"],
            dtype=np.int64,
        )
        tile_query_groups = np.asarray(
            arrays["tile_query_group"],
            dtype=np.int64,
        )
        tile_coordinates = _decode_coordinates(tile_features[:, :2], "tile")
        tile_rotations = np.argmax(
            tile_features[:, 2:8],
            axis=-1,
        ).astype(np.int16)
        tile_dual_query = np.argmax(tile_context[:, 17:23], axis=-1) != 5
        tile_query_index = np.repeat(
            np.arange(len(tile_context), dtype=np.int64),
            np.diff(tile_offsets),
        )
        if len(tile_query_index) != len(tile_features):
            raise RelationalPointerDataError(
                "tile query-to-item expansion drifted"
            )
        tile_dual_terrain = tile_dual_query[tile_query_index]
        tile_pointer_indices = np.empty(len(tile_features), dtype=np.int32)
        for query_index in range(len(tile_context)):
            local_group = int(tile_query_groups[query_index])
            token_types, token_payload = _active_parent(
                r3_tensors,
                int(group_r3_rows[local_group]),
            )
            frontier_indices = np.flatnonzero(
                token_types == R2_FRONTIER_TOKEN
            )
            frontier_coordinates = token_payload[
                frontier_indices,
                :2,
            ].astype(np.int16)
            for item_index in range(
                int(tile_offsets[query_index]),
                int(tile_offsets[query_index + 1]),
            ):
                matches = frontier_indices[
                    np.all(
                        frontier_coordinates == tile_coordinates[item_index],
                        axis=1,
                    )
                ]
                if len(matches) != 1:
                    raise RelationalPointerDataError(
                        "tile item does not point to exactly one frontier token"
                    )
                tile_pointer_indices[item_index] = int(matches[0])

    wildlife_pointer_indices = np.empty(0, dtype=np.int32)
    wildlife_kinds = np.empty(0, dtype=np.int32)
    query_tile_pointer_indices = np.empty(0, dtype=np.int32)
    query_tile_rotations = np.empty(0, dtype=np.int16)
    query_tile_dual = np.empty(0, dtype=np.bool_)
    if "wildlife" in requested:
        wildlife_context = np.asarray(
            arrays["wildlife_query_context"],
            dtype=np.float32,
        )
        wildlife_features = np.asarray(
            arrays["wildlife_item_features"],
            dtype=np.float32,
        )
        wildlife_offsets = np.asarray(
            arrays["wildlife_query_offsets"],
            dtype=np.int64,
        )
        wildlife_query_groups = np.asarray(
            arrays["wildlife_query_group"],
            dtype=np.int64,
        )
        if len(wildlife_context) != len(arrays["tile_item_features"]):
            raise RelationalPointerDataError(
                "wildlife selected-prefix ordering differs from tile items"
            )
        wildlife_coordinates = _decode_coordinates(
            wildlife_features[:, 1:3],
            "wildlife",
        )
        wildlife_present = wildlife_features[:, 0] > 0.5
        wildlife_pointer_indices = np.zeros(
            len(wildlife_features),
            dtype=np.int32,
        )
        wildlife_kinds = np.full(
            len(wildlife_features),
            DESTINATION_NONE,
            dtype=np.int32,
        )
        query_tile_coordinates = _decode_coordinates(
            wildlife_context[
                :,
                WILDLIFE_QUERY_TILE_OFFSET : WILDLIFE_QUERY_TILE_OFFSET + 2,
            ],
            "wildlife selected tile",
        )
        query_tile_rotations = np.argmax(
            wildlife_context[
                :,
                WILDLIFE_QUERY_TILE_OFFSET + 2 : WILDLIFE_QUERY_TILE_OFFSET
                + TILE_FACTOR_DIM
            ],
            axis=-1,
        ).astype(np.int16)
        query_tile_dual = np.argmax(
            wildlife_context[:, 17:23],
            axis=-1,
        ) != 5
        query_tile_pointer_indices = np.empty(
            len(wildlife_context),
            dtype=np.int32,
        )
        for query_index in range(len(wildlife_context)):
            local_group = int(wildlife_query_groups[query_index])
            token_types, token_payload = _active_parent(
                r3_tensors,
                int(group_r3_rows[local_group]),
            )
            frontier_indices = np.flatnonzero(
                token_types == R2_FRONTIER_TOKEN
            )
            frontier_coordinates = token_payload[
                frontier_indices,
                :2,
            ].astype(np.int16)
            selected_matches = frontier_indices[
                np.all(
                    frontier_coordinates == query_tile_coordinates[query_index],
                    axis=1,
                )
            ]
            if len(selected_matches) != 1:
                raise RelationalPointerDataError(
                    "wildlife prefix does not point to exactly one frontier token"
                )
            query_tile_pointer_indices[query_index] = int(selected_matches[0])
            occupied_indices = np.flatnonzero(
                token_types == R2_OCCUPIED_TOKEN
            )
            occupied_coordinates = token_payload[
                occupied_indices,
                :2,
            ].astype(np.int16)
            for item_index in range(
                int(wildlife_offsets[query_index]),
                int(wildlife_offsets[query_index + 1]),
            ):
                if not wildlife_present[item_index]:
                    continue
                if np.array_equal(
                    wildlife_coordinates[item_index],
                    query_tile_coordinates[query_index],
                ):
                    wildlife_kinds[item_index] = DESTINATION_NEW_TILE
                    wildlife_pointer_indices[item_index] = int(
                        selected_matches[0]
                    )
                    continue
                matches = occupied_indices[
                    np.all(
                        occupied_coordinates
                        == wildlife_coordinates[item_index],
                        axis=1,
                    )
                ]
                if len(matches) != 1:
                    raise RelationalPointerDataError(
                        "wildlife item does not point to exactly one occupied token"
                    )
                wildlife_kinds[item_index] = DESTINATION_EXISTING_TILE
                wildlife_pointer_indices[item_index] = int(matches[0])

    return PointerShardMetadata(
        group_r3_rows=group_r3_rows,
        tile_pointer_indices=tile_pointer_indices,
        tile_rotations=tile_rotations,
        tile_dual_terrain=tile_dual_terrain,
        wildlife_pointer_indices=wildlife_pointer_indices,
        wildlife_kinds=wildlife_kinds,
        wildlife_query_tile_pointer_indices=query_tile_pointer_indices,
        wildlife_query_tile_rotations=query_tile_rotations,
        wildlife_query_tile_dual_terrain=query_tile_dual,
    )


def materialize_pointer_batch(
    arrays: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    metadata: PointerShardMetadata,
    *,
    r3_tensors: dict[str, np.ndarray],
    stage: str,
    selected_queries: Sequence[int] | np.ndarray,
    seed: int,
    epoch: int,
    shard_index: int,
    d6_augment: bool = True,
) -> PointerStageBatch:
    """Materialize one query batch with one exact parent encode per unique group."""
    if stage not in STAGES:
        raise RelationalPointerDataError("unknown pointer stage")
    selected = np.asarray(selected_queries, dtype=np.int64)
    offsets = np.asarray(arrays[f"{stage}_query_offsets"], dtype=np.int64)
    if (
        selected.ndim != 1
        or not len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(offsets) - 1)
    ):
        raise RelationalPointerDataError("selected pointer queries are invalid")
    query_groups = np.asarray(
        arrays[f"{stage}_query_group"],
        dtype=np.int64,
    )[selected]
    unique_groups, query_parent = np.unique(query_groups, return_inverse=True)
    r3_rows = metadata.group_r3_rows[unique_groups]
    group_ids = np.asarray(arrays["group_id"], dtype=np.uint64)[unique_groups]
    transforms = (
        np.asarray(
            [
                deterministic_transform_id(
                    seed=seed,
                    epoch=epoch,
                    shard_index=shard_index,
                    group_id=int(group_id),
                )
                for group_id in group_ids
            ],
            dtype=np.int64,
        )
        if d6_augment
        else np.zeros(len(group_ids), dtype=np.int64)
    )
    parent = _materialize_parent(
        r3_tensors,
        r3_rows,
        transforms,
    )
    query_transforms = transforms[query_parent]
    widths = offsets[selected + 1] - offsets[selected]
    maximum = int(widths.max())
    item_features = np.zeros(
        (len(selected), maximum, STAGE_ITEM_DIMS[stage]),
        dtype=np.float32,
    )
    query_features = np.zeros(
        (len(selected), STAGE_QUERY_DIMS[stage]),
        dtype=np.float32,
    )
    item_pointers = np.zeros((len(selected), maximum), dtype=np.int32)
    item_rotations = np.zeros((len(selected), maximum), dtype=np.int32)
    item_kinds = np.full(
        (len(selected), maximum),
        DESTINATION_EXISTING_TILE,
        dtype=np.int32,
    )
    query_tile_pointers = np.zeros(len(selected), dtype=np.int32)
    query_tile_rotations = np.zeros(len(selected), dtype=np.int32)
    item_mask = np.zeros((len(selected), maximum), dtype=np.bool_)
    ranks = np.zeros((len(selected), maximum), dtype=np.float32)
    rank_mask = np.zeros((len(selected), maximum), dtype=np.bool_)
    target = np.zeros((len(selected), maximum), dtype=np.bool_)
    source_items = np.full((len(selected), maximum), -1, dtype=np.int64)

    all_item_features = np.asarray(
        arrays[f"{stage}_item_features"],
        dtype=np.float32,
    )
    all_ranks = np.asarray(arrays[f"{stage}_item_rank"], dtype=np.float32)
    all_rank_mask = np.asarray(
        arrays[f"{stage}_item_rank_mask"],
        dtype=np.bool_,
    )
    all_target = np.asarray(arrays[f"{stage}_item_target"], dtype=np.bool_)
    all_query_context = np.asarray(
        arrays[f"{stage}_query_context"],
        dtype=np.float32,
    )
    for batch_row, query_index in enumerate(selected):
        left = int(offsets[query_index])
        right = int(offsets[query_index + 1])
        width = right - left
        source_slice = np.arange(left, right, dtype=np.int64)
        item_mask[batch_row, :width] = True
        ranks[batch_row, :width] = all_ranks[left:right]
        rank_mask[batch_row, :width] = all_rank_mask[left:right]
        target[batch_row, :width] = all_target[left:right]
        source_items[batch_row, :width] = source_slice
        if stage == "draft":
            item_features[batch_row, :width] = all_item_features[
                left:right,
                :DRAFT_OBSERVABLE_DIM,
            ]
        elif stage == "tile":
            query_features[batch_row] = all_query_context[
                query_index,
                :DRAFT_OBSERVABLE_DIM,
            ]
            item_pointers[batch_row, :width] = metadata.tile_pointer_indices[
                left:right
            ]
            item_rotations[batch_row, :width] = _transform_rotations(
                metadata.tile_rotations[left:right],
                metadata.tile_dual_terrain[left:right],
                int(query_transforms[batch_row]),
            )
        else:
            query_features[batch_row] = all_query_context[
                query_index,
                :WILDLIFE_QUERY_DIM,
            ]
            item_pointers[batch_row, :width] = (
                metadata.wildlife_pointer_indices[left:right]
            )
            item_kinds[batch_row, :width] = metadata.wildlife_kinds[left:right]
            query_tile_pointers[batch_row] = (
                metadata.wildlife_query_tile_pointer_indices[query_index]
            )
            query_tile_rotations[batch_row] = _transform_rotations(
                metadata.wildlife_query_tile_rotations[query_index : query_index + 1],
                metadata.wildlife_query_tile_dual_terrain[
                    query_index : query_index + 1
                ],
                int(query_transforms[batch_row]),
            )[0]

    return PointerStageBatch(
        shard_index=shard_index,
        parent=parent,
        query_parent_indices=mx.array(query_parent.astype(np.int32)),
        query_features=mx.array(query_features),
        item_features=mx.array(item_features),
        item_pointer_indices=mx.array(item_pointers),
        item_rotations=mx.array(item_rotations),
        item_kinds=mx.array(item_kinds),
        query_tile_pointer_indices=mx.array(query_tile_pointers),
        query_tile_rotations=mx.array(query_tile_rotations),
        item_mask=mx.array(item_mask),
        expected_rank=mx.array(ranks),
        expected_rank_mask=mx.array(rank_mask),
        target=mx.array(target),
        parent_group_ids=group_ids.copy(),
        parent_transform_ids=transforms.copy(),
        source_query_indices=selected.copy(),
        source_item_indices=source_items,
        transform_ids=query_transforms.copy(),
    )


def deterministic_transform_id(
    *,
    seed: int,
    epoch: int,
    shard_index: int,
    group_id: int,
) -> int:
    """Map the frozen schedule tuple to one exact D6 transform."""
    if min(seed, epoch, shard_index, group_id) < 0:
        raise RelationalPointerDataError("D6 schedule inputs must be nonnegative")
    digest = blake3.blake3()
    digest.update(b"p1-relational-pointer-d6-v1")
    for value in (seed, epoch, shard_index, group_id):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(length=8), "little") % 12


def _materialize_parent(
    r3_tensors: dict[str, np.ndarray],
    rows: np.ndarray,
    transforms: np.ndarray,
) -> PointerParentBatch:
    token_types = np.asarray(r3_tensors["parent_token_types"][rows]).copy()
    token_seats = np.asarray(r3_tensors["parent_token_seats"][rows]).copy()
    payload = np.asarray(r3_tensors["parent_token_payload"][rows]).copy()
    token_mask = token_types != 0
    transform_token_payload_in_place(
        payload.reshape(len(rows), -1, TOKEN_PAYLOAD_WIDTH),
        token_types.reshape(len(rows), -1),
        transforms,
    )
    features = materialize_token_features(
        token_types,
        token_seats,
        payload,
        token_mask,
    )
    groups = len(rows)
    return PointerParentBatch(
        r2_token_features=mx.array(features),
        r2_token_types=mx.array(token_types.astype(np.int32)),
        r2_token_mask=mx.array(token_mask),
        relational_values=mx.zeros(
            (groups, BOARD_SLOTS, 0, RELATIONAL_VALUE_WIDTH),
            dtype=mx.int8,
        ),
        relational_classes=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.int32,
        ),
        relational_mask=mx.zeros(
            (groups, BOARD_SLOTS, 0),
            dtype=mx.bool_,
        ),
        market_features=mx.array(
            np.asarray(r3_tensors["parent_market_features"][rows]).copy()
        ),
        market_mask=mx.array(
            np.asarray(
                r3_tensors["parent_market_mask"][rows],
                dtype=np.bool_,
            ).copy()
        ),
        player_features=mx.array(
            np.asarray(r3_tensors["parent_player_features"][rows]).copy()
        ),
        player_mask=mx.array(
            np.asarray(
                r3_tensors["parent_player_mask"][rows],
                dtype=np.bool_,
            ).copy()
        ),
        global_features=mx.array(
            np.asarray(r3_tensors["parent_global_features"][rows]).copy()
        ),
    )


def _active_parent(
    r3_tensors: dict[str, np.ndarray],
    row: int,
) -> tuple[np.ndarray, np.ndarray]:
    types = np.asarray(
        r3_tensors["parent_token_types"][row, 0],
        dtype=np.uint8,
    )
    payload = np.asarray(
        r3_tensors["parent_token_payload"][row, 0],
        dtype=np.int8,
    )
    if (
        types.shape != (BOARD_TOKEN_CAPACITY,)
        or payload.shape != (BOARD_TOKEN_CAPACITY, TOKEN_PAYLOAD_WIDTH)
    ):
        raise RelationalPointerDataError("exact-R2 active-board tensor shape drifted")
    return types, payload


def _decode_coordinates(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise RelationalPointerDataError(f"{label} coordinates have the wrong shape")
    scaled = array * COORDINATE_SCALE
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-5, rtol=0.0):
        raise RelationalPointerDataError(f"{label} coordinates are not integral")
    return rounded.astype(np.int16)


def _transform_rotations(
    rotations: np.ndarray,
    dual_terrain: np.ndarray,
    transform_id: int,
) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.int16)
    dual = np.asarray(dual_terrain, dtype=np.bool_)
    if (
        rotations.shape != dual.shape
        or np.any((rotations < 0) | (rotations >= 6))
        or not 0 <= transform_id < 12
    ):
        raise RelationalPointerDataError("tile rotation transform inputs are invalid")
    return np.where(
        dual,
        _DUAL_ROTATION_TABLES[transform_id, rotations],
        _SINGLE_ROTATION_TABLES[transform_id, rotations],
    ).astype(np.int32)


def _schedule_seed(seed: int, epoch: int, shard_index: int) -> int:
    digest = blake3.blake3()
    digest.update(b"p1-relational-pointer-query-order-v1")
    for value in (seed, epoch, shard_index):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest(length=8), "little")


def validate_pointer_batch(batch: PointerStageBatch, *, stage: str) -> None:
    """Fail closed on padded pointer, parent, and target tensor drift."""
    if stage not in STAGES:
        raise RelationalPointerDataError("unknown pointer stage")
    queries, items = batch.item_mask.shape
    if (
        batch.shard_index < 0
        or batch.query_parent_indices.shape != (queries,)
        or batch.query_features.shape != (queries, STAGE_QUERY_DIMS[stage])
        or batch.item_features.shape != (queries, items, STAGE_ITEM_DIMS[stage])
        or batch.item_pointer_indices.shape != (queries, items)
        or batch.item_rotations.shape != (queries, items)
        or batch.item_kinds.shape != (queries, items)
        or batch.query_tile_pointer_indices.shape != (queries,)
        or batch.query_tile_rotations.shape != (queries,)
        or batch.expected_rank.shape != (queries, items)
        or batch.expected_rank_mask.shape != (queries, items)
        or batch.target.shape != (queries, items)
        or batch.parent_group_ids.shape
        != (batch.parent.r2_token_features.shape[0],)
        or batch.parent_transform_ids.shape != batch.parent_group_ids.shape
        or batch.source_item_indices.shape != (queries, items)
        or batch.transform_ids.shape != (queries,)
    ):
        raise RelationalPointerDataError("pointer batch tensor contract drifted")
    if batch.parent.r2_token_features.shape[-3:] != (
        BOARD_SLOTS,
        BOARD_TOKEN_CAPACITY,
        TOKEN_FEATURES,
    ):
        raise RelationalPointerDataError("pointer parent token contract drifted")
