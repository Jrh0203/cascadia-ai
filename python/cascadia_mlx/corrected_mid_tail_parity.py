"""Frozen 200,000-state C0/T1 parity campaign for the corrected NNUE schema."""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx import legacy_nnue as legacy_nnue_module
from cascadia_mlx.legacy_nnue import (
    CORRECTED_NNUE_BASE_END,
    CORRECTED_NNUE_MAGIC,
    CORRECTED_NNUE_OPPONENT_END,
    CORRECTED_NNUE_OPPONENT_START,
    CORRECTED_NNUE_OVERFLOW_END,
    CORRECTED_NNUE_TERRAIN_START,
    LEGACY_NNUE_FEATURES,
    LEGACY_NNUE_MAGIC,
    LegacyNnueError,
    LegacyNnueWeights,
    LegacyRustExactSparseNnue,
    pack_sparse_csr,
    parse_legacy_nnue,
    remap_historical_features_to_corrected,
)

SCHEMA_VERSION = 1
EXPERIMENT_ID = "corrected-mid-tail-frozen-parity-v1"
DATASET_ID = "legacy-mid-v4opp-activation-v1"
FEATURE_SCHEMA = "legacy-mid-v4opp-11231"
CORRECTED_SCHEMA = "legacy-mid-v4-fixed-v1"
HISTORICAL_DEFECT_START = 10_561
HISTORICAL_DEFECT_END = 10_862
HISTORICAL_OPPONENT_START = 10_862
HISTORICAL_OPPONENT_END = 11_231

CORPUS_MANIFEST_FILE_BLAKE3 = "7ade2ca310c976c5db9a0e5a840399e226ad8c650e6a4342da845fbb501e0996"
CORPUS_MANIFEST_SCIENTIFIC_BLAKE3 = (
    "193da520e3ccf3f440dd0f657996d486c1144737abcef1f8399b12ee8b34be92"
)
CORPUS_PAYLOAD_BLAKE3 = "433ebf13b88f6133efa41f42f3225e13278052b82e3f23a7735401427b5019d8"

HISTORICAL_CHECKPOINT_BLAKE3 = "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
HISTORICAL_CHECKPOINT_BYTES = 23_134_992
CORRECTED_CHECKPOINT_BLAKE3 = "a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0"
CORRECTED_CHECKPOINT_BYTES = 23_135_024

DEFAULT_CORPUS_ROOT = Path("artifacts/datasets/legacy-mid-v4opp-activation-v1")
DEFAULT_HISTORICAL_CHECKPOINT = Path("nnue_weights_v4opp_modal_iter3.bin")
DEFAULT_CORRECTED_CHECKPOINT = Path(
    "artifacts/experiments/corrected-mid-tail-v1/models/blake3/"
    f"{CORRECTED_CHECKPOINT_BLAKE3}/"
    "nnue_weights_legacy_mid_v4_fixed_v1_init.bin"
)

EXPECTED_ROW_KEYS = frozenset(
    {
        "game_index",
        "decision_index",
        "features",
        "raw_feature_count",
        "focal_seat",
        "personal_turn",
        "phase",
        "policy",
        "free_overflow_applied",
    }
)
POLICIES = ("greedy", "random_draft", "scarcity_draft", "preference_draft")
PHASES = ("opening", "early", "middle", "late")
DOWNSTREAM_TENSOR_NAMES = (
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
    "w3_policy",
    "b3_policy",
    "w3_wildlife",
    "b3_wildlife",
    "w3_habitat",
    "b3_habitat",
    "w3_heads",
    "b3_heads",
    "w3_var",
    "b3_var",
)
SCIENTIFIC_FORBIDDEN_KEY_PARTS = (
    "path",
    "wall",
    "second",
    "throughput",
    "device",
    "host",
    "timestamp",
    "generated_at",
    "started_at",
    "completed_at",
)


class ParityCampaignError(RuntimeError):
    """Raised when frozen parity evidence is incomplete, ambiguous, or corrupt."""


@dataclass(frozen=True)
class CorpusContract:
    dataset_id: str
    feature_schema: str
    feature_count: int
    split: str
    manifest_file_blake3: str
    manifest_scientific_blake3: str
    payload_blake3: str
    shard_count: int
    rows_per_shard: int
    games_per_shard: int
    rows_per_game: int
    first_game_index: int
    expected_statistics: Mapping[str, Any] | None = None

    @property
    def rows(self) -> int:
        return self.shard_count * self.rows_per_shard


@dataclass(frozen=True)
class CheckpointContract:
    arm: str
    schema: str
    expected_bytes: int
    expected_blake3: str
    expected_magic: bytes
    expected_head_version: int


@dataclass(frozen=True)
class ShardDeclaration:
    shard_index: int
    file: str
    row_count: int
    byte_count: int
    blake3: str
    first_game_index: int
    games: int


@dataclass(frozen=True)
class FrozenCorpus:
    contract: CorpusContract
    manifest: Mapping[str, Any]
    shards: tuple[ShardDeclaration, ...]


@dataclass(frozen=True)
class PreparedModels:
    historical_weights: LegacyNnueWeights
    corrected_weights: LegacyNnueWeights
    historical_model: LegacyRustExactSparseNnue
    corrected_model: LegacyRustExactSparseNnue
    checkpoint_receipts: Mapping[str, Any]
    mapping_receipt: Mapping[str, Any]


PRODUCTION_CORPUS_CONTRACT = CorpusContract(
    dataset_id=DATASET_ID,
    feature_schema=FEATURE_SCHEMA,
    feature_count=LEGACY_NNUE_FEATURES,
    split="train",
    manifest_file_blake3=CORPUS_MANIFEST_FILE_BLAKE3,
    manifest_scientific_blake3=CORPUS_MANIFEST_SCIENTIFIC_BLAKE3,
    payload_blake3=CORPUS_PAYLOAD_BLAKE3,
    shard_count=10,
    rows_per_shard=20_000,
    games_per_shard=250,
    rows_per_game=80,
    first_game_index=0,
    expected_statistics={
        "games": 2_500,
        "rows": 200_000,
        "rows_by_phase": {
            "opening": 10_000,
            "early": 40_000,
            "middle": 80_000,
            "late": 70_000,
        },
        "rows_by_focal_seat": {
            "0": 50_000,
            "1": 50_000,
            "2": 50_000,
            "3": 50_000,
        },
        "rows_by_policy": {
            "greedy": 50_000,
            "random_draft": 50_000,
            "scarcity_draft": 50_000,
            "preference_draft": 50_000,
        },
        "free_overflow_preludes": 61_006,
        "raw_feature_emissions": 49_849_871,
        "unique_feature_activations": 46_833_451,
        "duplicate_feature_emissions_removed": 3_016_420,
        "minimum_unique_features_per_row": 165,
        "maximum_unique_features_per_row": 317,
    },
)
HISTORICAL_CHECKPOINT_CONTRACT = CheckpointContract(
    arm="C0",
    schema=FEATURE_SCHEMA,
    expected_bytes=HISTORICAL_CHECKPOINT_BYTES,
    expected_blake3=HISTORICAL_CHECKPOINT_BLAKE3,
    expected_magic=LEGACY_NNUE_MAGIC,
    expected_head_version=1,
)
CORRECTED_CHECKPOINT_CONTRACT = CheckpointContract(
    arm="T1",
    schema=CORRECTED_SCHEMA,
    expected_bytes=CORRECTED_CHECKPOINT_BYTES,
    expected_blake3=CORRECTED_CHECKPOINT_BLAKE3,
    expected_magic=CORRECTED_NNUE_MAGIC,
    expected_head_version=1,
)


def canonical_json(value: object) -> bytes:
    """Return the single canonical byte encoding used by scientific receipts."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ParityCampaignError(f"value is not canonical JSON: {error}") from error


def scientific_blake3(value: object) -> str:
    return blake3.blake3(canonical_json(value)).hexdigest()


def checksum_file(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ParityCampaignError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def strict_json_loads(encoded: str, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ParityCampaignError(f"{label} contains non-finite JSON constant {value}")

    try:
        return json.loads(
            encoded,
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except ParityCampaignError:
        raise
    except json.JSONDecodeError as error:
        raise ParityCampaignError(f"{label} is malformed JSON: {error}") from error


def read_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = strict_json_loads(path.read_text(), label=str(path))
    except OSError as error:
        raise ParityCampaignError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ParityCampaignError(f"JSON root must be an object: {path}")
    return value


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    if path.is_symlink():
        raise ParityCampaignError(f"{label} must be a non-symlink regular file")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ParityCampaignError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ParityCampaignError(f"{label} must be a regular file")
    return metadata


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ParityCampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityCampaignError(f"{label} must be a nonempty string")
    return value


def _require_hex_digest(value: object, label: str) -> str:
    text = _require_string(value, label)
    try:
        decoded = bytes.fromhex(text)
    except ValueError as error:
        raise ParityCampaignError(f"{label} must be lowercase hexadecimal") from error
    if len(decoded) != 32 or text != decoded.hex():
        raise ParityCampaignError(f"{label} must be a canonical 32-byte BLAKE3 digest")
    return text


def payload_blake3(shards: Sequence[ShardDeclaration]) -> str:
    digest = blake3.blake3()
    digest.update(b"legacy-mid-v4opp-activation-v1/payload/v1")
    for shard in shards:
        encoded_file = shard.file.encode()
        digest.update(struct.pack("<Q", len(encoded_file)))
        digest.update(encoded_file)
        digest.update(struct.pack("<Q", shard.row_count))
        digest.update(struct.pack("<Q", shard.byte_count))
        digest.update(shard.blake3.encode())
    return digest.hexdigest()


def _parse_shard_declaration(
    value: object,
    *,
    shard_index: int,
    contract: CorpusContract,
) -> ShardDeclaration:
    if not isinstance(value, dict):
        raise ParityCampaignError(f"manifest shard {shard_index} must be an object")
    file = _require_string(value.get("file"), f"shards[{shard_index}].file")
    expected_file = f"part-{shard_index:05d}.jsonl"
    if file != expected_file:
        raise ParityCampaignError(
            f"manifest shard {shard_index} file is {file!r}, expected {expected_file!r}"
        )
    declaration = ShardDeclaration(
        shard_index=shard_index,
        file=file,
        row_count=_require_int(value.get("row_count"), f"shards[{shard_index}].row_count"),
        byte_count=_require_int(value.get("bytes"), f"shards[{shard_index}].bytes"),
        blake3=_require_hex_digest(value.get("blake3"), f"shards[{shard_index}].blake3"),
        first_game_index=_require_int(
            value.get("first_game_index"),
            f"shards[{shard_index}].first_game_index",
        ),
        games=_require_int(value.get("games"), f"shards[{shard_index}].games", minimum=1),
    )
    expected_first_game = contract.first_game_index + shard_index * contract.games_per_shard
    if declaration.row_count != contract.rows_per_shard:
        raise ParityCampaignError(f"manifest shard {shard_index} row count drifted")
    if declaration.games != contract.games_per_shard:
        raise ParityCampaignError(f"manifest shard {shard_index} game count drifted")
    if declaration.first_game_index != expected_first_game:
        raise ParityCampaignError(f"manifest shard {shard_index} game interval drifted")
    if declaration.row_count != declaration.games * contract.rows_per_game:
        raise ParityCampaignError(f"manifest shard {shard_index} rows-per-game drifted")
    return declaration


def validate_corpus_manifest(
    root: Path,
    contract: CorpusContract = PRODUCTION_CORPUS_CONTRACT,
) -> FrozenCorpus:
    if root.is_symlink() or not root.is_dir():
        raise ParityCampaignError("corpus root must be a non-symlink directory")
    manifest_path = root / "manifest.json"
    _require_regular_file(manifest_path, "corpus manifest")
    manifest_digest = checksum_file(manifest_path)
    if manifest_digest != contract.manifest_file_blake3:
        raise ParityCampaignError(
            "corpus manifest identity drifted: "
            f"expected {contract.manifest_file_blake3}, found {manifest_digest}"
        )
    manifest = read_strict_json(manifest_path)
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": contract.dataset_id,
        "feature_schema": contract.feature_schema,
        "feature_count": contract.feature_count,
        "split": contract.split,
        "rows": contract.rows,
        "payload_blake3": contract.payload_blake3,
        "scientific_blake3": contract.manifest_scientific_blake3,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ParityCampaignError(
                f"corpus manifest {key} is {manifest.get(key)!r}, expected {expected!r}"
            )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != contract.shard_count:
        raise ParityCampaignError(
            f"corpus manifest must declare exactly {contract.shard_count} shards"
        )
    shards = tuple(
        _parse_shard_declaration(value, shard_index=index, contract=contract)
        for index, value in enumerate(raw_shards)
    )
    if payload_blake3(shards) != contract.payload_blake3:
        raise ParityCampaignError("corpus payload receipt drifted")
    actual_parts = sorted(
        path.name
        for path in root.iterdir()
        if path.name.startswith("part-") and path.name.endswith(".jsonl")
    )
    expected_parts = [shard.file for shard in shards]
    if actual_parts != expected_parts:
        raise ParityCampaignError("corpus payload shard set differs from the frozen manifest")
    return FrozenCorpus(contract=contract, manifest=manifest, shards=shards)


def validate_shard_file_identity(root: Path, shard: ShardDeclaration) -> Path:
    path = root / shard.file
    metadata = _require_regular_file(path, f"corpus shard {shard.shard_index}")
    if metadata.st_size != shard.byte_count:
        raise ParityCampaignError(
            f"corpus shard {shard.shard_index} byte count drifted: "
            f"expected {shard.byte_count}, found {metadata.st_size}"
        )
    digest = checksum_file(path)
    if digest != shard.blake3:
        raise ParityCampaignError(
            f"corpus shard {shard.shard_index} BLAKE3 drifted: "
            f"expected {shard.blake3}, found {digest}"
        )
    return path


def validate_all_corpus_payload_identities(
    root: Path,
    contract: CorpusContract = PRODUCTION_CORPUS_CONTRACT,
) -> FrozenCorpus:
    corpus = validate_corpus_manifest(root, contract)
    for shard in corpus.shards:
        validate_shard_file_identity(root, shard)
    return corpus


def _phase(personal_turn: int) -> str:
    if personal_turn == 1:
        return "opening"
    if personal_turn <= 5:
        return "early"
    if personal_turn <= 13:
        return "middle"
    return "late"


def validate_activation_row(
    value: object,
    *,
    expected_game_index: int,
    expected_decision_index: int,
    feature_count: int = LEGACY_NNUE_FEATURES,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityCampaignError("activation row must be an object")
    if frozenset(value) != EXPECTED_ROW_KEYS:
        missing = sorted(EXPECTED_ROW_KEYS - frozenset(value))
        extra = sorted(frozenset(value) - EXPECTED_ROW_KEYS)
        raise ParityCampaignError(f"activation row keys drifted: missing={missing}, extra={extra}")
    game_index = _require_int(value["game_index"], "row.game_index")
    decision_index = _require_int(value["decision_index"], "row.decision_index")
    if game_index != expected_game_index or decision_index != expected_decision_index:
        raise ParityCampaignError(
            "activation row identity drifted: "
            f"found ({game_index}, {decision_index}), "
            f"expected ({expected_game_index}, {expected_decision_index})"
        )
    focal_seat = _require_int(value["focal_seat"], "row.focal_seat")
    personal_turn = _require_int(value["personal_turn"], "row.personal_turn", minimum=1)
    expected_seat = decision_index % 4
    expected_personal_turn = decision_index // 4 + 1
    if focal_seat != expected_seat or personal_turn != expected_personal_turn:
        raise ParityCampaignError("activation row seat or personal turn drifted")
    phase = _require_string(value["phase"], "row.phase")
    if phase != _phase(personal_turn):
        raise ParityCampaignError("activation row phase drifted")
    policy = _require_string(value["policy"], "row.policy")
    expected_policy = POLICIES[(game_index + focal_seat) % len(POLICIES)]
    if policy != expected_policy:
        raise ParityCampaignError("activation row policy assignment drifted")
    overflow = value["free_overflow_applied"]
    if not isinstance(overflow, bool):
        raise ParityCampaignError("activation row overflow flag must be boolean")
    raw_feature_count = _require_int(value["raw_feature_count"], "row.raw_feature_count", minimum=1)
    raw_features = value["features"]
    if not isinstance(raw_features, list) or not raw_features:
        raise ParityCampaignError("activation row features must be a nonempty array")
    features = [
        _require_int(feature, f"row.features[{index}]")
        for index, feature in enumerate(raw_features)
    ]
    if any(feature >= feature_count for feature in features):
        raise ParityCampaignError("activation row contains an out-of-range feature")
    if any(left >= right for left, right in pairwise(features)):
        raise ParityCampaignError("activation row features must be strictly increasing")
    discarded = [
        feature
        for feature in features
        if HISTORICAL_DEFECT_START <= feature < HISTORICAL_DEFECT_END
    ]
    if discarded:
        raise ParityCampaignError(
            "activation row uses discarded historical schema rows: "
            f"first={discarded[0]}, count={len(discarded)}"
        )
    if raw_feature_count < len(features):
        raise ParityCampaignError("activation row raw feature count is below the unique count")
    return {
        "game_index": game_index,
        "decision_index": decision_index,
        "features": features,
        "raw_feature_count": raw_feature_count,
        "focal_seat": focal_seat,
        "personal_turn": personal_turn,
        "phase": phase,
        "policy": policy,
        "free_overflow_applied": overflow,
    }


def iter_validated_shard_rows(
    path: Path,
    shard: ShardDeclaration,
    contract: CorpusContract,
    *,
    row_limit: int | None,
) -> Iterator[dict[str, Any]]:
    if row_limit is not None and not 1 <= row_limit <= shard.row_count:
        raise ParityCampaignError(
            f"row limit must be within 1..{shard.row_count}, found {row_limit}"
        )
    requested_rows = shard.row_count if row_limit is None else row_limit
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ParityCampaignError(f"cannot open corpus shard {path}: {error}") from error
    with handle:
        observed_rows = 0
        for row_index, line in enumerate(handle):
            if row_index >= requested_rows:
                if row_limit is None:
                    raise ParityCampaignError(
                        f"corpus shard {shard.shard_index} contains more than "
                        f"{shard.row_count} rows"
                    )
                break
            expected_game = shard.first_game_index + row_index // contract.rows_per_game
            expected_decision = row_index % contract.rows_per_game
            value = strict_json_loads(
                line,
                label=f"{shard.file} row {row_index}",
            )
            yield validate_activation_row(
                value,
                expected_game_index=expected_game,
                expected_decision_index=expected_decision,
                feature_count=contract.feature_count,
            )
            observed_rows += 1
        if observed_rows != requested_rows:
            raise ParityCampaignError(
                f"corpus shard {shard.shard_index} ended after {observed_rows} rows, "
                f"expected {requested_rows}"
            )


def _array_blake3(array: np.ndarray) -> str:
    return blake3.blake3(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tensor_bundle_blake3(weights: LegacyNnueWeights) -> str:
    digest = blake3.blake3()
    digest.update(b"cascadia-corrected-mid-tail-downstream-tensors-v1\0")
    for name in DOWNSTREAM_TENSOR_NAMES:
        tensor = getattr(weights, name)
        if tensor is None:
            continue
        encoded_name = name.encode()
        encoded = np.ascontiguousarray(tensor).tobytes(order="C")
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _checkpoint_receipt(
    path: Path,
    contract: CheckpointContract,
) -> tuple[LegacyNnueWeights, dict[str, Any]]:
    metadata = _require_regular_file(path, f"{contract.arm} checkpoint")
    if metadata.st_size != contract.expected_bytes:
        raise ParityCampaignError(
            f"{contract.arm} checkpoint byte count drifted: "
            f"expected {contract.expected_bytes}, found {metadata.st_size}"
        )
    digest = checksum_file(path)
    if digest != contract.expected_blake3:
        raise ParityCampaignError(
            f"{contract.arm} checkpoint BLAKE3 drifted: "
            f"expected {contract.expected_blake3}, found {digest}"
        )
    try:
        weights = parse_legacy_nnue(path)
    except (OSError, LegacyNnueError) as error:
        raise ParityCampaignError(f"cannot parse {contract.arm} checkpoint: {error}") from error
    if weights.container_magic != contract.expected_magic:
        raise ParityCampaignError(f"{contract.arm} checkpoint container magic drifted")
    if weights.version != contract.expected_head_version:
        raise ParityCampaignError(f"{contract.arm} checkpoint head version drifted")
    if (
        weights.feature_count != LEGACY_NNUE_FEATURES
        or weights.hidden1 != 512
        or weights.hidden2 != 64
    ):
        raise ParityCampaignError(f"{contract.arm} checkpoint dimensions drifted")
    if contract.arm == "T1" and not weights.is_corrected:
        raise ParityCampaignError("T1 checkpoint does not identify the corrected schema")
    if contract.arm == "C0" and weights.is_corrected:
        raise ParityCampaignError("C0 checkpoint must retain the historical schema")
    receipt = {
        "arm": contract.arm,
        "schema": contract.schema,
        "bytes": metadata.st_size,
        "blake3": digest,
        "container_magic": weights.container_magic.decode("ascii"),
        "container_version": weights.container_version,
        "head_version": weights.version,
        "feature_count": weights.feature_count,
        "hidden1": weights.hidden1,
        "hidden2": weights.hidden2,
    }
    return weights, receipt


def validate_checkpoint_identity(
    path: Path,
    contract: CheckpointContract,
) -> Mapping[str, Any]:
    _, receipt = _checkpoint_receipt(path, contract)
    return receipt


def audit_checkpoint_mapping(
    historical: LegacyNnueWeights,
    corrected: LegacyNnueWeights,
) -> dict[str, Any]:
    historical_base = historical.w1[:CORRECTED_NNUE_BASE_END]
    corrected_base = corrected.w1[:CORRECTED_NNUE_BASE_END]
    historical_opponent = historical.w1[HISTORICAL_OPPONENT_START:HISTORICAL_OPPONENT_END]
    corrected_opponent = corrected.w1[CORRECTED_NNUE_OPPONENT_START:CORRECTED_NNUE_OPPONENT_END]
    corrected_tail = corrected.w1[CORRECTED_NNUE_TERRAIN_START:CORRECTED_NNUE_OVERFLOW_END]
    tail_bits = np.ascontiguousarray(corrected_tail).view(np.uint32)

    historical_downstream = _tensor_bundle_blake3(historical)
    corrected_downstream = _tensor_bundle_blake3(corrected)
    gates = {
        "base_rows_byte_identical": (
            historical_base.tobytes(order="C") == corrected_base.tobytes(order="C")
        ),
        "opponent_rows_byte_identical_after_remap": (
            historical_opponent.tobytes(order="C") == corrected_opponent.tobytes(order="C")
        ),
        "corrected_tail_all_ieee754_signed_zero": bool(
            np.all((tail_bits & np.uint32(0x7FFF_FFFF)) == 0)
        ),
        "all_downstream_tensors_byte_identical": (historical_downstream == corrected_downstream),
    }
    if not all(gates.values()):
        failed = sorted(key for key, passed in gates.items() if not passed)
        raise ParityCampaignError(f"checkpoint migration mapping failed: {failed}")
    return {
        "mapping_id": "legacy-mid-v4opp-to-legacy-mid-v4-fixed-v1",
        "base_rows": {
            "source_range": [0, CORRECTED_NNUE_BASE_END],
            "destination_range": [0, CORRECTED_NNUE_BASE_END],
            "rows": CORRECTED_NNUE_BASE_END,
            "source_blake3": _array_blake3(historical_base),
            "destination_blake3": _array_blake3(corrected_base),
        },
        "discarded_rows": {
            "source_range": [HISTORICAL_DEFECT_START, HISTORICAL_DEFECT_END],
            "destination_range": None,
            "rows": HISTORICAL_DEFECT_END - HISTORICAL_DEFECT_START,
        },
        "opponent_rows": {
            "source_range": [HISTORICAL_OPPONENT_START, HISTORICAL_OPPONENT_END],
            "destination_range": [
                CORRECTED_NNUE_OPPONENT_START,
                CORRECTED_NNUE_OPPONENT_END,
            ],
            "rows": CORRECTED_NNUE_OPPONENT_END - CORRECTED_NNUE_OPPONENT_START,
            "source_blake3": _array_blake3(historical_opponent),
            "destination_blake3": _array_blake3(corrected_opponent),
        },
        "corrected_zero_tail": {
            "destination_range": [
                CORRECTED_NNUE_TERRAIN_START,
                CORRECTED_NNUE_OVERFLOW_END,
            ],
            "rows": CORRECTED_NNUE_OVERFLOW_END - CORRECTED_NNUE_TERRAIN_START,
            "blake3": _array_blake3(corrected_tail),
            "positive_zero_values": int(np.count_nonzero(tail_bits == 0)),
            "negative_zero_values": int(np.count_nonzero(tail_bits == np.uint32(0x8000_0000))),
        },
        "downstream_tensors": {
            "historical_blake3": historical_downstream,
            "corrected_blake3": corrected_downstream,
        },
        "gates": gates,
    }


def implementation_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    files = (
        (
            "cascadia_mlx_init.py",
            repository / "python/cascadia_mlx/__init__.py",
            "python/cascadia_mlx/__init__.py",
        ),
        (
            "corrected_mid_tail_parity.py",
            Path(__file__),
            "python/cascadia_mlx/corrected_mid_tail_parity.py",
        ),
        (
            "legacy_nnue.py",
            Path(legacy_nnue_module.__file__),
            "python/cascadia_mlx/legacy_nnue.py",
        ),
        (
            "corrected_mid_tail_parity_cli.py",
            repository / "tools/corrected_mid_tail_parity.py",
            "tools/corrected_mid_tail_parity.py",
        ),
    )
    entries = []
    digest = blake3.blake3()
    digest.update(b"cascadia-corrected-mid-tail-parity-implementation-v1\0")
    for label, path, relative_path in files:
        metadata = _require_regular_file(path, f"implementation source {label}")
        file_digest = checksum_file(path)
        encoded_label = label.encode()
        encoded_path = relative_path.encode()
        digest.update(struct.pack("<Q", len(encoded_label)))
        digest.update(encoded_label)
        digest.update(struct.pack("<Q", len(encoded_path)))
        digest.update(encoded_path)
        digest.update(struct.pack("<Q", metadata.st_size))
        digest.update(bytes.fromhex(file_digest))
        entries.append(
            {
                "label": label,
                "relative_file": relative_path,
                "bytes": metadata.st_size,
                "blake3": file_digest,
            }
        )
    return {
        "identity_kind": "corrected-mid-tail-parity-python-v1",
        "bundle_blake3": digest.hexdigest(),
        "files": entries,
    }


def prepare_models(
    historical_checkpoint: Path,
    corrected_checkpoint: Path,
) -> PreparedModels:
    historical, historical_receipt = _checkpoint_receipt(
        historical_checkpoint,
        HISTORICAL_CHECKPOINT_CONTRACT,
    )
    corrected, corrected_receipt = _checkpoint_receipt(
        corrected_checkpoint,
        CORRECTED_CHECKPOINT_CONTRACT,
    )
    mapping_receipt = audit_checkpoint_mapping(historical, corrected)
    return PreparedModels(
        historical_weights=historical,
        corrected_weights=corrected,
        historical_model=LegacyRustExactSparseNnue(historical.tensors()),
        corrected_model=LegacyRustExactSparseNnue(corrected.tensors()),
        checkpoint_receipts={
            "C0": historical_receipt,
            "T1": corrected_receipt,
        },
        mapping_receipt=mapping_receipt,
    )


def _update_feature_digest(
    digest: Any,
    *,
    game_index: int,
    decision_index: int,
    features: Sequence[int],
) -> None:
    digest.update(struct.pack("<QH", game_index, decision_index))
    digest.update(struct.pack("<H", len(features)))
    digest.update(np.asarray(features, dtype="<u2").tobytes(order="C"))


def _predict_exact(
    model: LegacyRustExactSparseNnue,
    feature_sets: list[list[int]],
) -> tuple[np.ndarray, float]:
    offsets, indices = pack_sparse_csr(feature_sets)
    started = time.perf_counter()
    predictions = model(offsets, indices)
    mx.eval(predictions)
    elapsed = time.perf_counter() - started
    values = np.asarray(predictions, dtype=np.float32)
    if values.shape != (len(feature_sets),):
        raise ParityCampaignError(
            f"MLX evaluator returned shape {values.shape}, expected {(len(feature_sets),)}"
        )
    return values, elapsed


def compare_prediction_bytes(
    historical: np.ndarray,
    corrected: np.ndarray,
    row_identities: Sequence[tuple[int, int]],
) -> tuple[bytes, bytes]:
    historical_values = np.asarray(historical, dtype="<f4")
    corrected_values = np.asarray(corrected, dtype="<f4")
    if historical_values.shape != corrected_values.shape:
        raise ParityCampaignError("C0 and T1 prediction shapes differ")
    if historical_values.shape != (len(row_identities),):
        raise ParityCampaignError("prediction count differs from row identity count")
    if not np.all(np.isfinite(historical_values)):
        raise ParityCampaignError("C0 produced a non-finite prediction")
    if not np.all(np.isfinite(corrected_values)):
        raise ParityCampaignError("T1 produced a non-finite prediction")
    historical_bytes = historical_values.tobytes(order="C")
    corrected_bytes = corrected_values.tobytes(order="C")
    if historical_bytes != corrected_bytes:
        historical_bits = historical_values.view("<u4")
        corrected_bits = corrected_values.view("<u4")
        mismatch = int(np.flatnonzero(historical_bits != corrected_bits)[0])
        game_index, decision_index = row_identities[mismatch]
        raise ParityCampaignError(
            "C0/T1 float32 prediction mismatch at "
            f"game={game_index}, decision={decision_index}: "
            f"C0=0x{int(historical_bits[mismatch]):08x}, "
            f"T1=0x{int(corrected_bits[mismatch]):08x}"
        )
    return historical_bytes, corrected_bytes


def _empty_statistics() -> dict[str, Any]:
    return {
        "rows_by_phase": Counter(),
        "rows_by_focal_seat": Counter(),
        "rows_by_policy": Counter(),
        "free_overflow_preludes": 0,
        "raw_feature_emissions": 0,
        "unique_feature_activations": 0,
        "duplicate_feature_emissions_removed": 0,
        "minimum_unique_features_per_row": None,
        "maximum_unique_features_per_row": None,
    }


def _observe_statistics(statistics: dict[str, Any], row: Mapping[str, Any]) -> None:
    feature_count = len(row["features"])
    statistics["rows_by_phase"][row["phase"]] += 1
    statistics["rows_by_focal_seat"][str(row["focal_seat"])] += 1
    statistics["rows_by_policy"][row["policy"]] += 1
    statistics["free_overflow_preludes"] += int(row["free_overflow_applied"])
    statistics["raw_feature_emissions"] += row["raw_feature_count"]
    statistics["unique_feature_activations"] += feature_count
    statistics["duplicate_feature_emissions_removed"] += row["raw_feature_count"] - feature_count
    minimum = statistics["minimum_unique_features_per_row"]
    maximum = statistics["maximum_unique_features_per_row"]
    statistics["minimum_unique_features_per_row"] = (
        feature_count if minimum is None else min(minimum, feature_count)
    )
    statistics["maximum_unique_features_per_row"] = (
        feature_count if maximum is None else max(maximum, feature_count)
    )


def _finish_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_by_phase": {
            phase: int(statistics["rows_by_phase"].get(phase, 0)) for phase in PHASES
        },
        "rows_by_focal_seat": {
            str(seat): int(statistics["rows_by_focal_seat"].get(str(seat), 0)) for seat in range(4)
        },
        "rows_by_policy": {
            policy: int(statistics["rows_by_policy"].get(policy, 0)) for policy in POLICIES
        },
        "free_overflow_preludes": int(statistics["free_overflow_preludes"]),
        "raw_feature_emissions": int(statistics["raw_feature_emissions"]),
        "unique_feature_activations": int(statistics["unique_feature_activations"]),
        "duplicate_feature_emissions_removed": int(
            statistics["duplicate_feature_emissions_removed"]
        ),
        "minimum_unique_features_per_row": int(statistics["minimum_unique_features_per_row"]),
        "maximum_unique_features_per_row": int(statistics["maximum_unique_features_per_row"]),
    }


def _scientific_has_forbidden_keys(value: object, *, prefix: str = "scientific") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if any(part in lowered for part in SCIENTIFIC_FORBIDDEN_KEY_PARTS):
                failures.append(f"{prefix}.{key}")
            failures.extend(_scientific_has_forbidden_keys(item, prefix=f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_scientific_has_forbidden_keys(item, prefix=f"{prefix}[{index}]"))
    return failures


def assert_scientific_section_is_portable(scientific: Mapping[str, Any]) -> None:
    failures = _scientific_has_forbidden_keys(scientific)
    if failures:
        raise ParityCampaignError(
            f"scientific section contains operational keys: {sorted(failures)}"
        )


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_shard(
    *,
    corpus_root: Path,
    shard_index: int,
    historical_checkpoint: Path,
    corrected_checkpoint: Path,
    output: Path | None,
    batch_rows: int = 512,
    row_limit: int | None = None,
    expected_implementation_blake3: str | None = None,
    contract: CorpusContract = PRODUCTION_CORPUS_CONTRACT,
    prepared_models: PreparedModels | None = None,
) -> dict[str, Any]:
    if batch_rows <= 0:
        raise ParityCampaignError("batch rows must be positive")
    if not 0 <= shard_index < contract.shard_count:
        raise ParityCampaignError(f"shard index must be within 0..{contract.shard_count - 1}")
    wall_started = time.perf_counter()
    corpus = validate_corpus_manifest(corpus_root, contract)
    shard = corpus.shards[shard_index]
    shard_path = validate_shard_file_identity(corpus_root, shard)
    implementation = implementation_identity()
    if (
        expected_implementation_blake3 is not None
        and implementation["bundle_blake3"] != expected_implementation_blake3
    ):
        raise ParityCampaignError(
            "parity implementation identity drifted: "
            f"expected {expected_implementation_blake3}, "
            f"found {implementation['bundle_blake3']}"
        )
    models = prepared_models or prepare_models(
        historical_checkpoint,
        corrected_checkpoint,
    )
    if prepared_models is not None:
        expected_receipts = {
            "C0": validate_checkpoint_identity(
                historical_checkpoint,
                HISTORICAL_CHECKPOINT_CONTRACT,
            ),
            "T1": validate_checkpoint_identity(
                corrected_checkpoint,
                CORRECTED_CHECKPOINT_CONTRACT,
            ),
        }
        if models.checkpoint_receipts != expected_receipts:
            raise ParityCampaignError("prepared model checkpoint receipts drifted")

    requested_rows = shard.row_count if row_limit is None else row_limit
    if requested_rows is None:
        raise AssertionError("requested row count must be resolved")
    mode = "production" if requested_rows == shard.row_count else "bounded_smoke"
    historical_feature_digest = blake3.blake3()
    historical_feature_digest.update(b"cascadia-f5-c0-sparse-features-v1\0")
    corrected_feature_digest = blake3.blake3()
    corrected_feature_digest.update(b"cascadia-f5-t1-sparse-features-v1\0")
    historical_prediction_digest = blake3.blake3()
    historical_prediction_digest.update(b"cascadia-f5-float32-predictions-v1\0")
    corrected_prediction_digest = blake3.blake3()
    corrected_prediction_digest.update(b"cascadia-f5-float32-predictions-v1\0")

    statistics = _empty_statistics()
    activation_counts = {
        "historical_base": 0,
        "historical_discarded": 0,
        "historical_opponent": 0,
        "corrected_base": 0,
        "corrected_opponent": 0,
        "corrected_tail": 0,
    }
    historical_batches: list[list[int]] = []
    corrected_batches: list[list[int]] = []
    row_identities: list[tuple[int, int]] = []
    evaluated_rows = 0
    historical_inference_seconds = 0.0
    corrected_inference_seconds = 0.0
    first_row_identity: list[int] | None = None
    last_row_identity: list[int] | None = None

    def evaluate_pending() -> None:
        nonlocal evaluated_rows
        nonlocal historical_inference_seconds
        nonlocal corrected_inference_seconds
        if not historical_batches:
            return
        historical_values, historical_elapsed = _predict_exact(
            models.historical_model,
            historical_batches,
        )
        corrected_values, corrected_elapsed = _predict_exact(
            models.corrected_model,
            corrected_batches,
        )
        historical_bytes, corrected_bytes = compare_prediction_bytes(
            historical_values,
            corrected_values,
            row_identities,
        )
        historical_prediction_digest.update(historical_bytes)
        corrected_prediction_digest.update(corrected_bytes)
        historical_inference_seconds += historical_elapsed
        corrected_inference_seconds += corrected_elapsed
        evaluated_rows += len(historical_batches)
        historical_batches.clear()
        corrected_batches.clear()
        row_identities.clear()

    for row in iter_validated_shard_rows(
        shard_path,
        shard,
        contract,
        row_limit=row_limit,
    ):
        features = row["features"]
        try:
            corrected_features = remap_historical_features_to_corrected(features)
        except LegacyNnueError as error:
            raise ParityCampaignError(f"cannot remap activation row: {error}") from error
        if any(feature < 0 or feature >= LEGACY_NNUE_FEATURES for feature in corrected_features):
            raise ParityCampaignError("corrected sparse remap produced an out-of-range row")
        if any(left >= right for left, right in pairwise(corrected_features)):
            raise ParityCampaignError("corrected sparse remap is not strictly increasing")

        identity = (row["game_index"], row["decision_index"])
        if first_row_identity is None:
            first_row_identity = list(identity)
        last_row_identity = list(identity)
        _observe_statistics(statistics, row)
        activation_counts["historical_base"] += sum(
            feature < HISTORICAL_DEFECT_START for feature in features
        )
        activation_counts["historical_discarded"] += sum(
            HISTORICAL_DEFECT_START <= feature < HISTORICAL_DEFECT_END for feature in features
        )
        activation_counts["historical_opponent"] += sum(
            HISTORICAL_OPPONENT_START <= feature < HISTORICAL_OPPONENT_END for feature in features
        )
        activation_counts["corrected_base"] += sum(
            feature < CORRECTED_NNUE_BASE_END for feature in corrected_features
        )
        activation_counts["corrected_opponent"] += sum(
            CORRECTED_NNUE_OPPONENT_START <= feature < CORRECTED_NNUE_OPPONENT_END
            for feature in corrected_features
        )
        activation_counts["corrected_tail"] += sum(
            CORRECTED_NNUE_TERRAIN_START <= feature < CORRECTED_NNUE_OVERFLOW_END
            for feature in corrected_features
        )
        _update_feature_digest(
            historical_feature_digest,
            game_index=identity[0],
            decision_index=identity[1],
            features=features,
        )
        _update_feature_digest(
            corrected_feature_digest,
            game_index=identity[0],
            decision_index=identity[1],
            features=corrected_features,
        )
        historical_batches.append(features)
        corrected_batches.append(corrected_features)
        row_identities.append(identity)
        if len(historical_batches) >= batch_rows:
            evaluate_pending()
    evaluate_pending()

    if evaluated_rows != requested_rows:
        raise ParityCampaignError(f"evaluated {evaluated_rows} rows, expected {requested_rows}")
    if activation_counts["historical_discarded"] != 0:
        raise ParityCampaignError("discarded historical rows were activated")
    if activation_counts["corrected_tail"] != 0:
        raise ParityCampaignError("zero-initialized corrected tail was activated by C0 corpus")
    historical_prediction_blake3 = historical_prediction_digest.hexdigest()
    corrected_prediction_blake3 = corrected_prediction_digest.hexdigest()
    if historical_prediction_blake3 != corrected_prediction_blake3:
        raise ParityCampaignError("C0/T1 prediction stream receipts differ")

    complete_shard = evaluated_rows == shard.row_count
    gates = {
        "corpus_manifest_identity_exact": True,
        "corpus_shard_identity_exact": True,
        "checkpoint_identities_exact": True,
        "base_and_opponent_weight_mapping_exact": True,
        "requested_rows_evaluated_exactly_once": evaluated_rows == requested_rows,
        "row_identities_contiguous": first_row_identity is not None
        and last_row_identity is not None,
        "historical_discarded_rows_absent": (activation_counts["historical_discarded"] == 0),
        "corrected_tail_rows_absent": activation_counts["corrected_tail"] == 0,
        "all_predictions_finite": True,
        "all_float32_prediction_bytes_identical": (
            historical_prediction_blake3 == corrected_prediction_blake3
        ),
    }
    if not all(gates.values()):
        failed = sorted(key for key, passed in gates.items() if not passed)
        raise ParityCampaignError(f"parity shard gates failed: {failed}")

    scientific = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "classification": (
            "corrected_mid_tail_frozen_parity_shard_complete"
            if complete_shard
            else "corrected_mid_tail_frozen_parity_shard_smoke_complete"
        ),
        "mode": mode,
        "implementation": implementation,
        "corpus": {
            "dataset_id": contract.dataset_id,
            "feature_schema": contract.feature_schema,
            "feature_count": contract.feature_count,
            "manifest_file_blake3": contract.manifest_file_blake3,
            "manifest_scientific_blake3": contract.manifest_scientific_blake3,
            "payload_blake3": contract.payload_blake3,
            "shard_index": shard.shard_index,
            "shard_blake3": shard.blake3,
            "shard_bytes": shard.byte_count,
            "declared_rows": shard.row_count,
            "first_game_index": shard.first_game_index,
            "games": shard.games,
        },
        "checkpoints": dict(models.checkpoint_receipts),
        "mapping": dict(models.mapping_receipt),
        "coverage": {
            "requested_rows": requested_rows,
            "evaluated_rows": evaluated_rows,
            "complete_shard": complete_shard,
            "first_row_identity": first_row_identity,
            "last_row_identity": last_row_identity,
        },
        "statistics": _finish_statistics(statistics),
        "activations": activation_counts,
        "feature_streams": {
            "C0_blake3": historical_feature_digest.hexdigest(),
            "T1_blake3": corrected_feature_digest.hexdigest(),
        },
        "predictions": {
            "dtype": "float32-little-endian",
            "C0_blake3": historical_prediction_blake3,
            "T1_blake3": corrected_prediction_blake3,
            "bit_identical_rows": evaluated_rows,
            "mismatched_rows": 0,
            "nonfinite_C0": 0,
            "nonfinite_T1": 0,
        },
        "gates": gates,
    }
    assert_scientific_section_is_portable(scientific)
    wall_seconds = time.perf_counter() - wall_started
    paired_inference_seconds = historical_inference_seconds + corrected_inference_seconds
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific": scientific,
        "scientific_blake3": scientific_blake3(scientific),
        "operational": {
            "host": socket.gethostname(),
            "device": str(mx.default_device()),
            "batch_rows": batch_rows,
            "paths": {
                "corpus_root": str(corpus_root.resolve()),
                "shard": str(shard_path.resolve()),
                "historical_checkpoint": str(historical_checkpoint.resolve()),
                "corrected_checkpoint": str(corrected_checkpoint.resolve()),
                "output": str(output.resolve()) if output is not None else None,
            },
            "timing": {
                "wall_seconds": wall_seconds,
                "C0_inference_seconds": historical_inference_seconds,
                "T1_inference_seconds": corrected_inference_seconds,
                "paired_inference_seconds": paired_inference_seconds,
                "C0_rows_per_second": (
                    evaluated_rows / historical_inference_seconds
                    if historical_inference_seconds > 0
                    else None
                ),
                "T1_rows_per_second": (
                    evaluated_rows / corrected_inference_seconds
                    if corrected_inference_seconds > 0
                    else None
                ),
                "paired_rows_per_second": (
                    evaluated_rows / paired_inference_seconds
                    if paired_inference_seconds > 0
                    else None
                ),
            },
        },
        "passed": True,
        "aggregate_eligible": complete_shard,
    }
    if output is not None:
        write_json_atomic(output, report)
    return report


def _load_shard_report(path: Path) -> Mapping[str, Any]:
    _require_regular_file(path, "parity shard report")
    report = read_strict_json(path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ParityCampaignError(f"unsupported parity shard report schema: {path}")
    scientific = report.get("scientific")
    if not isinstance(scientific, dict):
        raise ParityCampaignError(f"parity shard report lacks scientific section: {path}")
    assert_scientific_section_is_portable(scientific)
    if report.get("scientific_blake3") != scientific_blake3(scientific):
        raise ParityCampaignError(f"parity shard report scientific hash drifted: {path}")
    if report.get("passed") is not True or report.get("aggregate_eligible") is not True:
        raise ParityCampaignError(f"parity shard report is not a complete passing shard: {path}")
    return report


def _combined_receipt(domain: bytes, receipts: Sequence[tuple[int, str]]) -> str:
    digest = blake3.blake3()
    digest.update(domain)
    for shard_index, receipt in sorted(receipts):
        digest.update(struct.pack("<I", shard_index))
        digest.update(bytes.fromhex(receipt))
    return digest.hexdigest()


def _sum_statistics(
    shard_scientific: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_phase = Counter()
    rows_by_focal_seat = Counter()
    rows_by_policy = Counter()
    totals = Counter()
    minimums = []
    maximums = []
    for scientific in shard_scientific:
        statistics = scientific["statistics"]
        rows_by_phase.update(statistics["rows_by_phase"])
        rows_by_focal_seat.update(statistics["rows_by_focal_seat"])
        rows_by_policy.update(statistics["rows_by_policy"])
        for key in (
            "free_overflow_preludes",
            "raw_feature_emissions",
            "unique_feature_activations",
            "duplicate_feature_emissions_removed",
        ):
            totals[key] += statistics[key]
        minimums.append(statistics["minimum_unique_features_per_row"])
        maximums.append(statistics["maximum_unique_features_per_row"])
    return {
        "rows_by_phase": {phase: int(rows_by_phase[phase]) for phase in PHASES},
        "rows_by_focal_seat": {str(seat): int(rows_by_focal_seat[str(seat)]) for seat in range(4)},
        "rows_by_policy": {policy: int(rows_by_policy[policy]) for policy in POLICIES},
        **{key: int(value) for key, value in totals.items()},
        "minimum_unique_features_per_row": min(minimums),
        "maximum_unique_features_per_row": max(maximums),
    }


def aggregate_reports(
    report_paths: Sequence[Path],
    *,
    output: Path | None,
    contract: CorpusContract = PRODUCTION_CORPUS_CONTRACT,
) -> dict[str, Any]:
    if len(report_paths) != contract.shard_count:
        raise ParityCampaignError(
            f"aggregate requires exactly {contract.shard_count} shard reports"
        )
    loaded = [(path, _load_shard_report(path)) for path in report_paths]
    by_index: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for path, report in loaded:
        scientific = report["scientific"]
        if scientific.get("experiment_id") != EXPERIMENT_ID:
            raise ParityCampaignError(f"unexpected experiment ID in {path}")
        if scientific.get("classification") != ("corrected_mid_tail_frozen_parity_shard_complete"):
            raise ParityCampaignError(f"non-production shard classification in {path}")
        corpus = scientific.get("corpus")
        if not isinstance(corpus, dict):
            raise ParityCampaignError(f"missing corpus receipt in {path}")
        shard_index = _require_int(corpus.get("shard_index"), "corpus.shard_index")
        if shard_index in by_index:
            raise ParityCampaignError(f"duplicate parity shard index {shard_index}")
        by_index[shard_index] = (path, report)
    if sorted(by_index) != list(range(contract.shard_count)):
        raise ParityCampaignError("parity shard reports contain a gap or out-of-range index")

    ordered = [by_index[index] for index in range(contract.shard_count)]
    scientific_shards = [report["scientific"] for _path, report in ordered]
    expected_corpus = {
        "dataset_id": contract.dataset_id,
        "feature_schema": contract.feature_schema,
        "feature_count": contract.feature_count,
        "manifest_file_blake3": contract.manifest_file_blake3,
        "manifest_scientific_blake3": contract.manifest_scientific_blake3,
        "payload_blake3": contract.payload_blake3,
    }
    common_implementation = scientific_shards[0]["implementation"]
    common_checkpoints = scientific_shards[0]["checkpoints"]
    common_mapping = scientific_shards[0]["mapping"]
    shard_receipts = []
    previous_last_game = contract.first_game_index - 1
    for expected_index, scientific in enumerate(scientific_shards):
        corpus = scientific["corpus"]
        for key, expected in expected_corpus.items():
            if corpus.get(key) != expected:
                raise ParityCampaignError(
                    f"shard {expected_index} corpus {key} differs from the frozen contract"
                )
        if corpus.get("shard_index") != expected_index:
            raise ParityCampaignError("parity shard order or identity drifted")
        if corpus.get("declared_rows") != contract.rows_per_shard:
            raise ParityCampaignError(f"shard {expected_index} declared row count drifted")
        expected_first_game = contract.first_game_index + expected_index * contract.games_per_shard
        if (
            corpus.get("first_game_index") != expected_first_game
            or corpus.get("games") != contract.games_per_shard
        ):
            raise ParityCampaignError(f"shard {expected_index} game interval drifted")
        if expected_first_game != previous_last_game + 1:
            raise ParityCampaignError("parity shard game intervals overlap or contain a gap")
        previous_last_game = expected_first_game + contract.games_per_shard - 1
        coverage = scientific["coverage"]
        if (
            coverage.get("complete_shard") is not True
            or coverage.get("requested_rows") != contract.rows_per_shard
            or coverage.get("evaluated_rows") != contract.rows_per_shard
        ):
            raise ParityCampaignError(f"shard {expected_index} coverage is incomplete")
        expected_first_identity = [expected_first_game, 0]
        expected_last_identity = [
            expected_first_game + contract.games_per_shard - 1,
            contract.rows_per_game - 1,
        ]
        if (
            coverage.get("first_row_identity") != expected_first_identity
            or coverage.get("last_row_identity") != expected_last_identity
        ):
            raise ParityCampaignError(f"shard {expected_index} row interval drifted")
        if scientific["implementation"] != common_implementation:
            raise ParityCampaignError("parity shards disagree on implementation identity")
        if scientific["checkpoints"] != common_checkpoints:
            raise ParityCampaignError("parity shards disagree on checkpoint identities")
        if scientific["mapping"] != common_mapping:
            raise ParityCampaignError("parity shards disagree on checkpoint mapping")
        if not all(scientific["gates"].values()):
            raise ParityCampaignError(f"shard {expected_index} contains a failed gate")
        predictions = scientific["predictions"]
        if (
            predictions.get("C0_blake3") != predictions.get("T1_blake3")
            or predictions.get("bit_identical_rows") != contract.rows_per_shard
            or predictions.get("mismatched_rows") != 0
            or predictions.get("nonfinite_C0") != 0
            or predictions.get("nonfinite_T1") != 0
        ):
            raise ParityCampaignError(f"shard {expected_index} prediction parity drifted")
        if scientific["activations"].get("historical_discarded") != 0:
            raise ParityCampaignError(f"shard {expected_index} activated discarded rows")
        if scientific["activations"].get("corrected_tail") != 0:
            raise ParityCampaignError(f"shard {expected_index} activated corrected tail rows")
        shard_receipts.append(
            {
                "shard_index": expected_index,
                "scientific_blake3": scientific_blake3(scientific),
                "rows": contract.rows_per_shard,
                "C0_prediction_blake3": predictions["C0_blake3"],
                "T1_prediction_blake3": predictions["T1_blake3"],
                "C0_feature_stream_blake3": scientific["feature_streams"]["C0_blake3"],
                "T1_feature_stream_blake3": scientific["feature_streams"]["T1_blake3"],
            }
        )

    statistics = _sum_statistics(scientific_shards)
    if contract.expected_statistics is not None:
        for key, expected in contract.expected_statistics.items():
            observed = contract.rows // contract.rows_per_game if key == "games" else None
            if key == "rows":
                observed = contract.rows
            elif key not in {"games", "rows"}:
                observed = statistics.get(key)
            if observed != expected:
                raise ParityCampaignError(
                    f"aggregate corpus statistic {key} is {observed!r}, expected {expected!r}"
                )

    c0_prediction_receipt = _combined_receipt(
        b"cascadia-f5-aggregate-predictions-v1\0",
        [(receipt["shard_index"], receipt["C0_prediction_blake3"]) for receipt in shard_receipts],
    )
    t1_prediction_receipt = _combined_receipt(
        b"cascadia-f5-aggregate-predictions-v1\0",
        [(receipt["shard_index"], receipt["T1_prediction_blake3"]) for receipt in shard_receipts],
    )
    gates = {
        "all_ten_shards_present_once": len(shard_receipts) == contract.shard_count,
        "complete_contiguous_row_coverage": (
            sum(receipt["rows"] for receipt in shard_receipts) == contract.rows
        ),
        "corpus_statistics_match_manifest": (
            contract.expected_statistics is None
            or all(
                (
                    contract.rows // contract.rows_per_game
                    if key == "games"
                    else contract.rows
                    if key == "rows"
                    else statistics.get(key)
                )
                == expected
                for key, expected in contract.expected_statistics.items()
            )
        ),
        "all_discarded_row_activations_zero": all(
            scientific["activations"]["historical_discarded"] == 0
            for scientific in scientific_shards
        ),
        "all_predictions_finite": all(
            scientific["predictions"]["nonfinite_C0"] == 0
            and scientific["predictions"]["nonfinite_T1"] == 0
            for scientific in scientific_shards
        ),
        "all_200000_prediction_bytes_identical": all(
            receipt["C0_prediction_blake3"] == receipt["T1_prediction_blake3"]
            for receipt in shard_receipts
        ),
    }
    if not all(gates.values()):
        failed = sorted(key for key, passed in gates.items() if not passed)
        raise ParityCampaignError(f"aggregate parity gates failed: {failed}")
    scientific = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "classification": "corrected_mid_tail_frozen_parity_complete",
        "implementation": common_implementation,
        "corpus": {
            **expected_corpus,
            "shards": contract.shard_count,
            "rows": contract.rows,
            "first_game_index": contract.first_game_index,
            "games": contract.rows // contract.rows_per_game,
        },
        "checkpoints": common_checkpoints,
        "mapping": common_mapping,
        "coverage": {
            "shards": contract.shard_count,
            "rows": contract.rows,
            "complete_game_interval": True,
            "first_row_identity": [contract.first_game_index, 0],
            "last_row_identity": [
                contract.first_game_index + contract.shard_count * contract.games_per_shard - 1,
                contract.rows_per_game - 1,
            ],
        },
        "statistics": statistics,
        "shard_receipts": shard_receipts,
        "predictions": {
            "dtype": "float32-little-endian",
            "C0_aggregate_receipt_blake3": c0_prediction_receipt,
            "T1_aggregate_receipt_blake3": t1_prediction_receipt,
            "bit_identical_rows": contract.rows,
            "mismatched_rows": 0,
            "nonfinite_C0": 0,
            "nonfinite_T1": 0,
        },
        "gates": gates,
    }
    assert_scientific_section_is_portable(scientific)

    timing_fields = (
        "wall_seconds",
        "C0_inference_seconds",
        "T1_inference_seconds",
        "paired_inference_seconds",
    )
    summed_timing = {
        field: float(
            sum(
                report.get("operational", {}).get("timing", {}).get(field, 0.0)
                for _path, report in ordered
            )
        )
        for field in timing_fields
    }
    paired_seconds = summed_timing["paired_inference_seconds"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific": scientific,
        "scientific_blake3": scientific_blake3(scientific),
        "operational": {
            "input_reports": [str(path.resolve()) for path, _report in ordered],
            "hosts_by_shard": [
                {
                    "shard_index": index,
                    "host": report.get("operational", {}).get("host"),
                }
                for index, (_path, report) in enumerate(ordered)
            ],
            "summed_timing": {
                **summed_timing,
                "paired_rows_per_accelerator_second": (
                    contract.rows / paired_seconds if paired_seconds > 0 else None
                ),
            },
        },
        "passed": True,
    }
    if output is not None:
        write_json_atomic(output, report)
    return report
