"""Exact, policy-blind decoder for the O1 opponent-intent corpus."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.dataset import (
    POSITION_RECORD_DTYPE,
    decode_records,
)
from cascadia_mlx.dataset import (
    Batch as PositionBatch,
)

DATASET_SCHEMA_VERSION = 1
FEATURE_SCHEMA = "compact-public-state-recent-actions-no-policy-id-v1"
TARGET_SCHEMA = "three-opponent-next-action-four-tile-survival-v1"
SHARD_MAGIC = b"CSD2O1I\0"
SHARD_HEADER_SIZE = 64
RECORD_SIZE = 1_312
HISTORY_LENGTH = 12
OPPONENT_COUNT = 3
MARKET_SLOTS = 4
ACTION_RECORD_SIZE = 24
HISTORY_ENTRY_SIZE = 27
OPPONENT_TARGET_SIZE = 27
SURVIVAL_TARGET_SIZE = 5
HISTORY_FEATURE_DIM = 55

_HEADER = struct.Struct("<8sHIBQI Q".replace(" ", ""))
_EXPECTED_SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
    "final": 3,
}

PUBLIC_ACTION_DTYPE = np.dtype(
    {
        "names": [
            "draft_kind",
            "tile_slot",
            "wildlife_slot",
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
            "paid_wipe_count",
            "paid_wipe_slot_mask",
            "paid_wipe_total_slots",
            "reserved",
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
            "i1",
            "i1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            "u1",
            "u1",
            ("u1", (6,)),
        ],
        "offsets": [*range(18), 18],
        "itemsize": ACTION_RECORD_SIZE,
    }
)

HISTORY_ENTRY_DTYPE = np.dtype(
    {
        "names": ["valid", "age", "relative_seat", "action"],
        "formats": ["u1", "u1", "u1", PUBLIC_ACTION_DTYPE],
        "offsets": [0, 1, 2, 3],
        "itemsize": HISTORY_ENTRY_SIZE,
    }
)

OPPONENT_TARGET_DTYPE = np.dtype(
    {
        "names": ["relative_seat", "policy_code", "selected_tile_id", "action"],
        "formats": ["u1", "u1", "u1", PUBLIC_ACTION_DTYPE],
        "offsets": [0, 1, 2, 3],
        "itemsize": OPPONENT_TARGET_SIZE,
    }
)

SURVIVAL_TARGET_DTYPE = np.dtype(
    {
        "names": [
            "initial_tile_id",
            "initial_wildlife",
            "disposition",
            "pair_survives",
            "final_slot",
        ],
        "formats": ["u1", "u1", "u1", "u1", "u1"],
        "offsets": [0, 1, 2, 3, 4],
        "itemsize": SURVIVAL_TARGET_SIZE,
    }
)

OPPONENT_INTENT_RECORD_DTYPE = np.dtype(
    {
        "names": [
            "game_index",
            "focal_turn",
            "focal_seat",
            "seat_policy_codes",
            "position",
            "history_count",
            "history",
            "opponent_targets",
            "survival_targets",
            "final_scores",
        ],
        "formats": [
            "<u8",
            "u1",
            "u1",
            ("u1", (4,)),
            POSITION_RECORD_DTYPE,
            "u1",
            (HISTORY_ENTRY_DTYPE, (HISTORY_LENGTH,)),
            (OPPONENT_TARGET_DTYPE, (OPPONENT_COUNT,)),
            (SURVIVAL_TARGET_DTYPE, (MARKET_SLOTS,)),
            ("<u2", (4,)),
        ],
        "offsets": [0, 8, 9, 10, 14, 878, 879, 1_203, 1_284, 1_304],
        "itemsize": RECORD_SIZE,
    }
)


class OpponentIntentDatasetError(ValueError):
    """Raised when an O1 corpus is incompatible or fails integrity checks."""


@dataclass(frozen=True)
class OpponentIntentBatch:
    """One policy-blind model batch plus evaluation-only target metadata."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    history_features: mx.array
    history_mask: mx.array
    disposition_targets: mx.array
    pair_survival_targets: mx.array
    final_slot_targets: mx.array
    tile_slot_targets: mx.array
    wildlife_slot_targets: mx.array
    draft_kind_targets: mx.array
    drafted_wildlife_targets: mx.array
    replace_three_targets: mx.array
    game_index: mx.array
    focal_turn: mx.array


@dataclass(frozen=True)
class OpponentIntentInputs:
    """The exact policy-blind surface accepted by an O1 model."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    history_features: mx.array
    history_mask: mx.array


@dataclass(frozen=True)
class OpponentIntentEvaluationMetadata:
    """Provenance-only fields that are never accepted by the model."""

    seat_policy_codes: np.ndarray
    opponent_policy_codes: np.ndarray


@dataclass(frozen=True)
class OpponentIntentShard:
    """Validated fixed-width O1 shard."""

    path: Path
    record_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=OPPONENT_INTENT_RECORD_DTYPE,
            offset=SHARD_HEADER_SIZE,
            shape=(self.record_count,),
        )


class OpponentIntentDataset:
    """One immutable O1 role with lazy, checksum-verified shard access."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise OpponentIntentDatasetError(f"cannot read O1 dataset manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def sample_count(self) -> int:
        return int(self.manifest["total_records"])

    def batches(self, batch_size: int) -> Iterator[OpponentIntentBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for shard in self.shards:
            records = shard.records()
            for start in range(0, len(records), batch_size):
                yield decode_opponent_intent_records(records[start : start + batch_size])

    def raw_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for shard in self.shards:
            records = shard.records()
            for start in range(0, len(records), batch_size):
                yield np.asarray(records[start : start + batch_size])

    def evaluation_metadata(
        self,
        records: np.ndarray,
    ) -> OpponentIntentEvaluationMetadata:
        records = np.asarray(records)
        return OpponentIntentEvaluationMetadata(
            seat_policy_codes=records["seat_policy_codes"].copy(),
            opponent_policy_codes=records["opponent_targets"]["policy_code"].copy(),
        )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != DATASET_SCHEMA_VERSION
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("target_schema") != TARGET_SCHEMA
            or manifest.get("record_size") != RECORD_SIZE
            or manifest.get("windows_per_game") != 76
            or manifest.get("history_length") != HISTORY_LENGTH
            or manifest.get("policy_identity_observable") is not False
            or manifest.get("game_index_observable") is not False
            or manifest.get("strategy_switch_targets_available") is not False
        ):
            raise OpponentIntentDatasetError(
                "O1 manifest does not match the frozen public-input contract"
            )
        split = manifest.get("split")
        if split not in _EXPECTED_SPLIT_CODES:
            raise OpponentIntentDatasetError("O1 manifest has an unknown split")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise OpponentIntentDatasetError("O1 manifest shards must be a list")
        if sum(int(item["record_count"]) for item in shards) != int(
            manifest.get("total_records", -1)
        ):
            raise OpponentIntentDatasetError("O1 manifest record total disagrees with its shards")
        if sum(int(item["game_count"]) for item in shards) != int(
            manifest.get("completed_games", -1)
        ):
            raise OpponentIntentDatasetError("O1 manifest game total disagrees with its shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> OpponentIntentShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            with path.open("rb") as handle:
                header = handle.read(SHARD_HEADER_SIZE)
        except OSError as error:
            raise OpponentIntentDatasetError(f"cannot read O1 shard {path}: {error}") from error
        if stat.st_size != int(entry["byte_count"]):
            raise OpponentIntentDatasetError(f"O1 shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise OpponentIntentDatasetError(f"O1 shard checksum mismatch: {path}")
        if len(header) != SHARD_HEADER_SIZE:
            raise OpponentIntentDatasetError(f"truncated O1 shard: {path}")
        (
            magic,
            schema,
            record_size,
            split_code,
            first_game_index,
            game_count,
            record_count,
        ) = _HEADER.unpack_from(header)
        if (
            magic != SHARD_MAGIC
            or schema != DATASET_SCHEMA_VERSION
            or record_size != RECORD_SIZE
            or split_code != _EXPECTED_SPLIT_CODES[self.split]
            or first_game_index != int(entry["first_game_index"])
            or game_count != int(entry["game_count"])
            or record_count != int(entry["record_count"])
        ):
            raise OpponentIntentDatasetError(f"O1 shard header disagrees with manifest: {path}")
        expected_size = SHARD_HEADER_SIZE + record_count * RECORD_SIZE
        if stat.st_size != expected_size:
            raise OpponentIntentDatasetError(
                f"O1 shard record count disagrees with file size: {path}"
            )
        return OpponentIntentShard(
            path=path,
            record_count=record_count,
            game_count=game_count,
            first_game_index=first_game_index,
        )


class CombinedOpponentIntentDataset:
    """A deterministic shard-first view over multiple same-role datasets."""

    def __init__(self, datasets: Sequence[OpponentIntentDataset]):
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("combined O1 dataset requires at least one role")
        if any(dataset.split != self.datasets[0].split for dataset in self.datasets):
            raise OpponentIntentDatasetError("combined O1 datasets must share one split")
        self.shards = tuple(shard for dataset in self.datasets for shard in dataset.shards)

    @property
    def split(self) -> str:
        return self.datasets[0].split

    @property
    def sample_count(self) -> int:
        return sum(dataset.sample_count for dataset in self.datasets)

    @property
    def batches_per_epoch(self) -> int:
        return sum((shard.record_count + 127) // 128 for shard in self.shards)

    def training_examples_for_steps(
        self,
        *,
        steps: int,
        seed: int,
        batch_size: int,
    ) -> int:
        """Count examples consumed by the deterministic shard-first schedule."""
        if steps < 0 or batch_size <= 0:
            raise ValueError("invalid O1 training-example count request")
        batch_counts = np.asarray(
            [(shard.record_count + batch_size - 1) // batch_size for shard in self.shards],
            dtype=np.int64,
        )
        batches_per_epoch = int(np.sum(batch_counts))
        full_epochs, remaining = divmod(steps, batches_per_epoch)
        total = full_epochs * self.sample_count
        if remaining == 0:
            return total
        shard_order = np.random.default_rng(seed + full_epochs).permutation(len(self.shards))
        for shard_index in shard_order:
            shard = self.shards[int(shard_index)]
            count = int(batch_counts[int(shard_index)])
            take = min(remaining, count)
            full_batches = min(take, shard.record_count // batch_size)
            total += full_batches * batch_size
            if take > full_batches:
                total += shard.record_count - full_batches * batch_size
            remaining -= take
            if remaining == 0:
                break
        return total

    def deterministic_training_batch(
        self,
        *,
        step: int,
        seed: int,
        batch_size: int,
    ) -> OpponentIntentBatch:
        if step < 0 or batch_size <= 0:
            raise ValueError("invalid deterministic O1 batch request")
        batch_counts = np.asarray(
            [(shard.record_count + batch_size - 1) // batch_size for shard in self.shards],
            dtype=np.int64,
        )
        batches_per_epoch = int(np.sum(batch_counts))
        epoch, batch_in_epoch = divmod(step, batches_per_epoch)
        shard_order = np.random.default_rng(seed + epoch).permutation(len(self.shards))
        cursor = batch_in_epoch
        selected_shard = -1
        selected_local_batch = -1
        for shard_index in shard_order:
            count = int(batch_counts[int(shard_index)])
            if cursor < count:
                selected_shard = int(shard_index)
                selected_local_batch = cursor
                break
            cursor -= count
        if selected_shard < 0:
            raise AssertionError("deterministic O1 batch schedule drifted")
        shard = self.shards[selected_shard]
        local_order = np.random.default_rng(
            seed + epoch * 65_537 + selected_shard * 257
        ).permutation(shard.record_count)
        start = selected_local_batch * batch_size
        indices = local_order[start : start + batch_size]
        return decode_opponent_intent_records(shard.records()[indices])


def decode_opponent_intent_records(
    records: np.ndarray,
) -> OpponentIntentBatch:
    """Decode O1 rows while excluding every provenance and future-only field."""
    records = np.asarray(records)
    if records.dtype != OPPONENT_INTENT_RECORD_DTYPE:
        records = records.astype(OPPONENT_INTENT_RECORD_DTYPE, copy=False)
    inputs = decode_opponent_intent_inputs(records)
    targets = records["opponent_targets"]["action"]
    survival = records["survival_targets"]
    dispositions = survival["disposition"].astype(np.int32) - 1
    if np.any((dispositions < 0) | (dispositions >= 4)):
        raise OpponentIntentDatasetError(
            "O1 survival disposition is outside the frozen four classes"
        )
    final_slots = survival["final_slot"].astype(np.int32)
    final_slots = np.where(dispositions == 3, final_slots, -1)
    return OpponentIntentBatch(
        board_entities=inputs.board_entities,
        board_mask=inputs.board_mask,
        market_entities=inputs.market_entities,
        market_mask=inputs.market_mask,
        global_features=inputs.global_features,
        history_features=inputs.history_features,
        history_mask=inputs.history_mask,
        disposition_targets=mx.array(dispositions),
        pair_survival_targets=mx.array(survival["pair_survives"].astype(np.int32)),
        final_slot_targets=mx.array(final_slots),
        tile_slot_targets=mx.array(targets["tile_slot"].astype(np.int32)),
        wildlife_slot_targets=mx.array(targets["wildlife_slot"].astype(np.int32)),
        draft_kind_targets=mx.array(targets["draft_kind"].astype(np.int32)),
        drafted_wildlife_targets=mx.array(targets["drafted_wildlife"].astype(np.int32)),
        replace_three_targets=mx.array(targets["replace_three_of_a_kind"].astype(np.int32)),
        game_index=mx.array(records["game_index"].astype(np.int64)),
        focal_turn=mx.array(records["focal_turn"].astype(np.int32)),
    )


def decode_opponent_intent_inputs(
    records: np.ndarray,
) -> OpponentIntentInputs:
    """Decode only public state and public action history for target-free inference."""
    records = np.asarray(records)
    if records.dtype != OPPONENT_INTENT_RECORD_DTYPE:
        records = records.astype(OPPONENT_INTENT_RECORD_DTYPE, copy=False)
    focal_seats = records["focal_seat"].astype(np.int64)
    decoded = decode_records(records["position"])
    rotated = _rotate_position_batch(decoded, focal_seats)
    history = records["history"]
    history_mask = history["valid"] != 0
    history_features = _decode_action_features(
        history["action"],
        age=history["age"],
        relative_seat=history["relative_seat"],
    )
    history_features *= history_mask[..., None]
    return OpponentIntentInputs(
        board_entities=rotated.board_entities,
        board_mask=rotated.board_mask,
        market_entities=rotated.market_entities,
        market_mask=rotated.market_mask,
        global_features=rotated.global_features,
        history_features=mx.array(history_features),
        history_mask=mx.array(history_mask),
    )


def model_input_arrays(records: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return only arrays that can flow into an O1 model."""
    batch = decode_opponent_intent_inputs(records)
    return tuple(
        np.asarray(value)
        for value in (
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.history_features,
            batch.history_mask,
        )
    )


def _rotate_position_batch(
    batch: PositionBatch,
    focal_seats: np.ndarray,
) -> PositionBatch:
    board_entities = np.asarray(batch.board_entities)
    board_mask = np.asarray(batch.board_mask)
    global_features = np.asarray(batch.global_features)
    seat_order = (focal_seats[:, None] + np.arange(4, dtype=np.int64)[None, :]) % 4
    board_entities = np.take_along_axis(
        board_entities,
        seat_order[:, :, None, None],
        axis=1,
    )
    board_mask = np.take_along_axis(
        board_mask,
        seat_order[:, :, None],
        axis=1,
    )
    rotated_global = global_features.copy()
    for start, width in ((6, 1), (10, 1), (14, 5), (34, 5)):
        values = global_features[:, start : start + 4 * width].reshape(
            len(global_features), 4, width
        )
        rotated = np.take_along_axis(
            values,
            seat_order[:, :, None],
            axis=1,
        )
        rotated_global[:, start : start + 4 * width] = rotated.reshape(len(global_features), -1)
    return PositionBatch(
        board_entities=mx.array(board_entities),
        board_mask=mx.array(board_mask),
        market_entities=batch.market_entities,
        market_mask=batch.market_mask,
        global_features=mx.array(rotated_global),
        targets=batch.targets,
        game_index=batch.game_index,
        turn=batch.turn,
    )


def _decode_action_features(
    actions: np.ndarray,
    *,
    age: np.ndarray,
    relative_seat: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(actions)
    features = np.concatenate(
        [
            np.asarray(age, dtype=np.float32)[..., None] / 11.0,
            _one_hot(relative_seat, 4),
            _one_hot(actions["draft_kind"], 2),
            _one_hot(actions["tile_slot"], 4),
            _one_hot(actions["wildlife_slot"], 4),
            _one_hot(actions["tile_terrain_a"], 5),
            _one_hot_with_none(actions["tile_terrain_b"], 5),
            _mask_bits(actions["tile_wildlife_mask"], 5),
            actions["tile_keystone"].astype(np.float32)[..., None],
            _one_hot(actions["drafted_wildlife"], 5),
            actions["tile_q"].astype(np.float32)[..., None] / 24.0,
            actions["tile_r"].astype(np.float32)[..., None] / 24.0,
            _one_hot(actions["rotation"], 6),
            actions["wildlife_present"].astype(np.float32)[..., None],
            actions["wildlife_q"].astype(np.float32)[..., None] / 24.0,
            actions["wildlife_r"].astype(np.float32)[..., None] / 24.0,
            actions["replace_three_of_a_kind"].astype(np.float32)[..., None],
            actions["paid_wipe_count"].astype(np.float32)[..., None] / 4.0,
            _mask_bits(actions["paid_wipe_slot_mask"], 4),
            actions["paid_wipe_total_slots"].astype(np.float32)[..., None] / 4.0,
        ],
        axis=-1,
    )
    if features.shape[-1] != HISTORY_FEATURE_DIM:
        raise AssertionError("O1 history feature width drifted")
    return features


def _one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(values)
    valid = values < classes
    clipped = np.where(valid, values, 0)
    return np.eye(classes, dtype=np.float32)[clipped] * valid[..., None]


def _one_hot_with_none(values: np.ndarray, classes: int) -> np.ndarray:
    mapped = np.where(np.asarray(values) < classes, values, classes)
    return np.eye(classes + 1, dtype=np.float32)[mapped]


def _mask_bits(values: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits, dtype=np.uint8)
    return ((np.asarray(values)[..., None] >> shifts) & 1).astype(np.float32)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
