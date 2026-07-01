"""Strict MLX decoder for qualified R12 counterfactual-advantage data."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.action_ranking_dataset import (
    _ACTION_POSITION_DTYPE,
    ACTION_POSITION_RECORD_SIZE,
    decode_action_positions,
)
from cascadia_mlx.dataset import (
    _RECORD_DTYPE as _POSITION_DTYPE,
)
from cascadia_mlx.dataset import (
    TARGET_DIM,
    DatasetError,
)

COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION = 1
COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA = (
    "grouped-observable-action-afterstates-with-public-supply-v1"
)
COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA = (
    "shared-public-redetermined-centered-terminal-components-v1"
)
COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC = b"CSD2CFA\0"
COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE = 160
COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE = 6676
COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE = 1308
COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES = 4
COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES = 16
COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES = 12
COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE = 30
COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SELECTION = "selected-high-median-low-v1"
COUNTERFACTUAL_ADVANTAGE_TEACHER = "habitat-candidate-lookahead-v1-k8-h6-r4-d4"
COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING = "reject-unstable-market-trajectories-v1"

_HEADER = struct.Struct("<8sHHHHHHHHIIBBBBIQ32s32s32s16s")
_SPLIT_CODES = {"train": 0, "validation": 1, "test": 2, "final": 3}
_PUBLIC_SUPPLY_SCALES = np.array([20.0] * 5 + [81.0] * 25, dtype=np.float32)

_CANDIDATE_DTYPE = np.dtype(
    {
        "names": [
            "action_hash",
            "shallow_mean",
            "shallow_stddev",
            "input",
            "sample_finals",
        ],
        "formats": [
            ("u1", (32,)),
            "<f4",
            "<f4",
            _ACTION_POSITION_DTYPE,
            ("<u2", (COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES, TARGET_DIM)),
        ],
        "offsets": [0, 32, 36, 40, 956],
        "itemsize": COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE,
    }
)
_COUNTERFACTUAL_ADVANTAGE_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "selected_index",
            "candidate_count",
            "sample_count",
            "reserved",
            "current",
            "public_supply",
            "parent",
            "sample_seeds",
            "candidates",
        ],
        "formats": [
            "<u8",
            "u1",
            "u1",
            "u1",
            ("u1", (5,)),
            ("<u2", (TARGET_DIM,)),
            ("u1", (COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE,)),
            _POSITION_DTYPE,
            ("u1", (COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES, 32)),
            (_CANDIDATE_DTYPE, (COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES,)),
        ],
        "offsets": [0, 8, 9, 10, 11, 16, 38, 68, 932, 1444],
        "itemsize": COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
    }
)


@dataclass(frozen=True)
class CounterfactualAdvantageBatch:
    """One batch of complete four-candidate R12 decision groups."""

    board_entities: mx.array
    board_mask: mx.array
    market_entities: mx.array
    market_mask: mx.array
    global_features: mx.array
    action_features: mx.array
    public_supply: mx.array
    candidate_mask: mx.array
    target_mean: mx.array
    target_stddev: mx.array
    target_standard_error: mx.array
    immediate_score: mx.array
    shallow_mean: mx.array
    shallow_stddev: mx.array
    selected_index: mx.array
    group_id: mx.array
    game_index: mx.array
    turn: mx.array


@dataclass(frozen=True)
class CounterfactualAdvantageShard:
    """A validated, fixed-width R12 counterfactual shard."""

    path: Path
    record_count: int
    game_count: int
    first_game_index: int

    def records(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=_COUNTERFACTUAL_ADVANTAGE_DTYPE,
            offset=COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE,
            shape=(self.record_count,),
        )


class CounterfactualAdvantageDataset:
    """Manifest-backed selected/high/median/low R12 candidate groups."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        try:
            self.manifest: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read counterfactual-advantage manifest: {error}") from error
        self._validate_manifest()
        self.shards = tuple(
            self._load_shard(entry, verify_checksums) for entry in self.manifest["shards"]
        )
        self._validate_dataset_records()

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def group_count(self) -> int:
        return int(self.manifest["total_groups"])

    @property
    def candidate_count(self) -> int:
        return int(self.manifest["total_candidates"])

    def batches(
        self,
        group_batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[CounterfactualAdvantageBatch]:
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        rng = np.random.default_rng(seed)
        shard_indices = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            records = self.shards[int(shard_index)].records()
            record_indices = np.arange(len(records))
            if shuffle:
                rng.shuffle(record_indices)
            for start in range(0, len(record_indices), group_batch_size):
                yield decode_counterfactual_advantage_records(
                    records[record_indices[start : start + group_batch_size]]
                )

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if (
            manifest.get("schema_version") != COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION
            or manifest.get("feature_schema") != COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA
            or manifest.get("target_schema") != COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA
            or manifest.get("record_size") != COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
            or manifest.get("action_position_record_size") != ACTION_POSITION_RECORD_SIZE
            or manifest.get("target_dim") != TARGET_DIM
            or manifest.get("maximum_candidates") != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or manifest.get("maximum_samples") != COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
        ):
            raise DatasetError("unsupported counterfactual-advantage dataset schema")
        game = manifest.get("game")
        cards = game.get("scoring_cards", {}) if isinstance(game, dict) else {}
        if (
            not isinstance(game, dict)
            or game.get("player_count") != 4
            or game.get("mode") != "Standard"
            or game.get("habitat_bonuses") is not False
            or set(cards) != {"bear", "elk", "salmon", "hawk", "fox"}
            or any(value != "A" for value in cards.values())
        ):
            raise DatasetError("counterfactual-advantage game is not canonical AAAAA")
        teacher = manifest.get("teacher")
        if (
            not isinstance(teacher, dict)
            or teacher.get("strategy_id") != COUNTERFACTUAL_ADVANTAGE_TEACHER
            or teacher.get("immediate_candidates") != 8
            or teacher.get("habitat_candidates") != 6
            or teacher.get("determinizations") != 4
            or teacher.get("greedy_plies") != 4
            or teacher.get("candidate_count") != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or teacher.get("samples_per_candidate") != COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES
            or teacher.get("sample_seed_domain") != "cascadia-v2-counterfactual-advantage-v1"
            or teacher.get("candidate_selection") != COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SELECTION
            or teacher.get("stabilization_conditioning")
            != COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING
            or not isinstance(teacher.get("groups_per_game"), int)
            or teacher["groups_per_game"] <= 0
            or 80 % teacher["groups_per_game"] != 0
        ):
            raise DatasetError("dataset is not the frozen R12 rank-stratified teacher")
        if manifest.get("split") not in _SPLIT_CODES:
            raise DatasetError("counterfactual-advantage split is invalid")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise DatasetError("counterfactual-advantage shards must be a list")
        groups = sum(int(shard["record_count"]) for shard in shards)
        games = sum(int(shard["game_count"]) for shard in shards)
        if (
            groups != int(manifest.get("total_groups", -1))
            or games != int(manifest.get("completed_games", -1))
            or int(manifest.get("total_candidates", -1))
            != groups * COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or int(manifest.get("total_continuations", -1))
            != groups
            * COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            * COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES
        ):
            raise DatasetError("counterfactual-advantage manifest totals do not match shards")

    def _load_shard(
        self,
        entry: dict[str, Any],
        verify_checksum: bool,
    ) -> CounterfactualAdvantageShard:
        path = self.root / str(entry["file"])
        try:
            stat = path.stat()
            with path.open("rb") as handle:
                header = handle.read(COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE)
        except OSError as error:
            raise DatasetError(
                f"cannot read counterfactual-advantage shard {path}: {error}"
            ) from error
        if stat.st_size != int(entry["byte_count"]):
            raise DatasetError(f"counterfactual-advantage shard size mismatch: {path}")
        if verify_checksum and _checksum(path) != entry["blake3"]:
            raise DatasetError(f"counterfactual-advantage shard checksum mismatch: {path}")
        if len(header) != COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE:
            raise DatasetError(f"truncated counterfactual-advantage shard header: {path}")
        (
            magic,
            schema,
            header_size,
            record_size,
            action_record_size,
            target_dim,
            maximum_candidates,
            maximum_samples,
            public_supply_size,
            record_count,
            game_count,
            split_code,
            players,
            candidate_count,
            sample_count,
            groups_per_game,
            first_game_index,
            feature_hash,
            target_hash,
            teacher_hash,
            reserved,
        ) = _HEADER.unpack(header)
        teacher_bytes = json.dumps(
            self.manifest["teacher"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if (
            magic != COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC
            or schema != COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION
            or header_size != COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE
            or record_size != COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
            or action_record_size != ACTION_POSITION_RECORD_SIZE
            or target_dim != TARGET_DIM
            or maximum_candidates != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or maximum_samples != COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
            or public_supply_size != COUNTERFACTUAL_ADVANTAGE_PUBLIC_SUPPLY_SIZE
            or split_code != _SPLIT_CODES[self.split]
            or players != 4
            or candidate_count != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or sample_count != COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES
            or groups_per_game != self.manifest["teacher"]["groups_per_game"]
            or feature_hash
            != blake3.blake3(COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA.encode()).digest()
            or target_hash
            != blake3.blake3(COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA.encode()).digest()
            or teacher_hash != blake3.blake3(teacher_bytes).digest()
            or reserved != bytes(16)
        ):
            raise DatasetError(f"incompatible counterfactual-advantage shard header: {path}")
        if (
            record_count != int(entry["record_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise DatasetError(f"counterfactual-advantage header disagrees with manifest: {path}")
        expected_size = (
            COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE
            + record_count * COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
        )
        if stat.st_size != expected_size:
            raise DatasetError(f"counterfactual-advantage record count disagrees with size: {path}")
        return CounterfactualAdvantageShard(
            path=path,
            record_count=record_count,
            game_count=game_count,
            first_game_index=first_game_index,
        )

    def _validate_dataset_records(self) -> None:
        teacher = self.manifest["teacher"]
        groups_per_game = int(teacher["groups_per_game"])
        stride = 80 // groups_per_game
        expected_game = int(self.manifest["first_game_index"])
        seen_group_ids: set[int] = set()
        for shard in self.shards:
            if (
                shard.game_count != 1
                or shard.record_count != groups_per_game
                or shard.first_game_index != expected_game
            ):
                raise DatasetError(
                    f"counterfactual-advantage shard game range is invalid: {shard.path}"
                )
            records = shard.records()
            for group_index, record in enumerate(records):
                self._validate_record(
                    record,
                    expected_game=expected_game,
                    expected_turn=group_index * stride,
                    seen_group_ids=seen_group_ids,
                    path=shard.path,
                )
            expected_game += 1

    @staticmethod
    def _validate_record(
        record: np.void,
        *,
        expected_game: int,
        expected_turn: int,
        seen_group_ids: set[int],
        path: Path,
    ) -> None:
        group_id = int(record["group_id"])
        parent = record["parent"]
        candidates = record["candidates"]
        sample_seeds = record["sample_seeds"]
        if (
            group_id == 0
            or group_id in seen_group_ids
            or int(record["selected_index"]) >= COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or int(record["candidate_count"]) != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or int(record["sample_count"]) != COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES
            or np.any(record["reserved"] != 0)
            or int(parent["game_index"]) != expected_game
            or int(parent["turn"]) != expected_turn
            or int(parent["active_seat"]) != expected_turn % 4
            or int(parent["player_count"]) != 4
            or int(parent["total_turns"]) != 80
            or np.any(parent["targets"] != 0)
        ):
            raise DatasetError(f"invalid counterfactual-advantage group metadata: {path}")
        seen_group_ids.add(group_id)

        supply = record["public_supply"]
        if int(np.sum(supply[15:30], dtype=np.int64)) != 81 - expected_turn:
            raise DatasetError(f"invalid counterfactual-advantage public supply: {path}")
        active_seeds = [bytes(seed) for seed in sample_seeds[:12]]
        if (
            any(seed == bytes(32) for seed in active_seeds)
            or len(set(active_seeds)) != COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES
            or np.any(sample_seeds[12:] != 0)
        ):
            raise DatasetError(f"invalid counterfactual-advantage shared seeds: {path}")

        action_hashes = [bytes(candidate["action_hash"]) for candidate in candidates]
        current_total = int(np.sum(record["current"], dtype=np.int64))
        immediate_scores = candidates["input"]["action"]["immediate_score"].astype(np.int64)
        immediate_deltas = np.sum(
            candidates["input"]["action"]["immediate_deltas"].astype(np.int64),
            axis=-1,
        )
        candidate_positions = candidates["input"]["position"]
        active_finals = candidates["sample_finals"][:, :12]
        if (
            any(value == bytes(32) for value in action_hashes)
            or len(set(action_hashes)) != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            or np.any(~np.isfinite(candidates["shallow_mean"]))
            or np.any(~np.isfinite(candidates["shallow_stddev"]))
            or np.any(candidates["shallow_stddev"] < 0)
            or np.any(candidates["input"]["action"]["immediate_rank"] == 0)
            or np.any(candidate_positions["game_index"] != expected_game)
            or np.any(candidate_positions["turn"] != expected_turn + 1)
            or np.any(candidate_positions["active_seat"] != expected_turn % 4)
            or np.any(candidate_positions["player_count"] != 4)
            or np.any(candidate_positions["total_turns"] != 80)
            or np.any(candidate_positions["targets"] != 0)
            or np.any(immediate_scores - immediate_deltas != current_total)
            or np.any(np.sum(active_finals, axis=-1, dtype=np.int64) <= 0)
            or np.any(candidates["sample_finals"][:, 12:] != 0)
        ):
            raise DatasetError(f"invalid counterfactual-advantage candidates: {path}")


def decode_counterfactual_advantage_records(
    records: np.ndarray,
) -> CounterfactualAdvantageBatch:
    """Decode complete R12 groups into MLX tensors."""
    records = np.asarray(records)
    if not len(records):
        raise ValueError("counterfactual-advantage batch must contain at least one group")
    groups = len(records)
    candidates = COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
    candidate_records = records["candidates"]
    inputs = candidate_records["input"].reshape(groups * candidates)
    decoded, action_features = decode_action_positions(inputs)

    sample_components = candidate_records["sample_finals"][:, :, :12].astype(np.float32)
    sample_totals = np.sum(sample_components, axis=-1)
    target_mean = np.mean(sample_totals, axis=-1, dtype=np.float32)
    target_stddev = np.std(sample_totals, axis=-1, ddof=1, dtype=np.float32)
    target_standard_error = target_stddev / np.sqrt(
        np.float32(COUNTERFACTUAL_ADVANTAGE_ESTIMATOR_SAMPLES)
    )
    shape_prefix = (groups, candidates)
    candidate_mask = np.ones(shape_prefix, dtype=np.bool_)
    public_supply = records["public_supply"].astype(np.float32) / _PUBLIC_SUPPLY_SCALES

    return CounterfactualAdvantageBatch(
        board_entities=decoded.board_entities.reshape(*shape_prefix, 4, 23, -1),
        board_mask=decoded.board_mask.reshape(*shape_prefix, 4, 23),
        market_entities=decoded.market_entities.reshape(*shape_prefix, 4, -1),
        market_mask=decoded.market_mask.reshape(*shape_prefix, 4),
        global_features=decoded.global_features.reshape(*shape_prefix, -1),
        action_features=action_features.reshape(*shape_prefix, -1),
        public_supply=mx.array(public_supply),
        candidate_mask=mx.array(candidate_mask),
        target_mean=mx.array(target_mean),
        target_stddev=mx.array(target_stddev),
        target_standard_error=mx.array(target_standard_error),
        immediate_score=mx.array(
            candidate_records["input"]["action"]["immediate_score"].astype(np.float32)
        ),
        shallow_mean=mx.array(candidate_records["shallow_mean"].astype(np.float32)),
        shallow_stddev=mx.array(candidate_records["shallow_stddev"].astype(np.float32)),
        selected_index=mx.array(records["selected_index"].astype(np.int32)),
        group_id=mx.array(records["group_id"].astype(np.int64)),
        game_index=decoded.game_index.reshape(*shape_prefix),
        turn=decoded.turn.reshape(*shape_prefix),
    )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
