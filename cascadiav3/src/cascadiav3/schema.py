"""Executable schema contracts for Cascadia v3 replay and tensor artifacts.

The original CPU scaffold shipped with one JSONL schema,
``cascadiav3.pre_gpu.v0``.  The expert-search pipeline is additive: legacy
fixtures and greedy tensor shards keep their identifiers, while richer
expert-root and expert-tensor contracts register beside them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .hex import RADIUS6_CELL_COUNT
from .rival.schema import RIVAL_SCHEMA_DEFINITIONS

PRE_GPU_SCHEMA_ID = "cascadiav3.pre_gpu.v0"
EXPERT_ROOT_SCHEMA_ID = "cascadiav3.expert_root.v1"
GREEDY_TENSOR_SHARD_SCHEMA_ID = "greedy_policy_tensor_shard_v1"
EXPERT_TENSOR_SHARD_SCHEMA_ID = "cascadiav3.expert_tensor_shard.v1"
EXPERT_TENSOR_SHARD_SCHEMA_ID_V2 = "cascadiav3.expert_tensor_shard.v2"
EXPERT_TENSOR_SHARD_SCHEMA_ID_V3 = "cascadiav3.expert_tensor_shard.v3"
EXPERT_TENSOR_SHARD_SCHEMA_ID_V4 = "cascadiav3.expert_tensor_shard.v4"

# Backward-compatible name used by the original scaffold and tests.
SCHEMA_ID = PRE_GPU_SCHEMA_ID


@dataclass(frozen=True)
class SchemaDefinition:
    schema_id: str
    artifact_kind: str
    version: int
    status: str
    description: str
    compatible_readers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["compatible_readers"] = list(self.compatible_readers)
        return out


SCHEMA_REGISTRY: dict[str, SchemaDefinition] = {
    PRE_GPU_SCHEMA_ID: SchemaDefinition(
        schema_id=PRE_GPU_SCHEMA_ID,
        artifact_kind="search_root_jsonl",
        version=0,
        status="legacy-active",
        description="CPU pre-GPU search-root fixture and dry-run replay schema.",
        compatible_readers=("cascadiav3.validate", "cascadiav3.torch_replay"),
    ),
    GREEDY_TENSOR_SHARD_SCHEMA_ID: SchemaDefinition(
        schema_id=GREEDY_TENSOR_SHARD_SCHEMA_ID,
        artifact_kind="greedy_policy_tensor_shard",
        version=1,
        status="active",
        description="Compact public-token/semantic-action greedy behavior cloning shard.",
        compatible_readers=(
            "cascadiav3.greedy_tensor_shards",
            "cascadiav3.torch_greedy_policy_pretrain",
        ),
    ),
    EXPERT_ROOT_SCHEMA_ID: SchemaDefinition(
        schema_id=EXPERT_ROOT_SCHEMA_ID,
        artifact_kind="expert_root_jsonl",
        version=1,
        status="active",
        description="Full legal-action expert root with reconstruction, chance, and Q-valid audit fields.",
        compatible_readers=(
            "cascadiav3.validate_public_boundary",
            "cascadiav3.validate_d6_roundtrip",
            "cascadiav3.validate_category_targets",
            "cascadiav3.torch_train_cascadiaformer",
        ),
    ),
    EXPERT_TENSOR_SHARD_SCHEMA_ID: SchemaDefinition(
        schema_id=EXPERT_TENSOR_SHARD_SCHEMA_ID,
        artifact_kind="expert_tensor_shard",
        version=1,
        status="active",
        description="Packed NPZ trainer shard for expert roots with legal mask, sparse relations, Q-valid mask, and decomposition labels.",
        compatible_readers=(
            "cascadiav3.expert_tensor_shards",
            "cascadiav3.torch_train_cascadiaformer",
        ),
    ),
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V2: SchemaDefinition(
        schema_id=EXPERT_TENSOR_SHARD_SCHEMA_ID_V2,
        artifact_kind="expert_tensor_shard",
        version=2,
        status="active",
        description="v1 expert tensor shard plus Gumbel self-play targets: action-aligned improved_policy soft targets and per-record search_root_value, with real terminal-outcome value labels.",
        compatible_readers=(
            "cascadiav3.expert_tensor_shards",
            "cascadiav3.torch_train_cascadiaformer",
        ),
    ),
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V3: SchemaDefinition(
        schema_id=EXPERT_TENSOR_SHARD_SCHEMA_ID_V3,
        artifact_kind="expert_tensor_shard",
        version=3,
        status="active",
        description="v2 Gumbel self-play shard plus an explicit per-record exact_endgame flag and complete generation provenance in metadata.",
        compatible_readers=(
            "cascadiav3.expert_tensor_shards",
            "cascadiav3.torch_train_cascadiaformer",
        ),
    ),
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V4: SchemaDefinition(
        schema_id=EXPERT_TENSOR_SHARD_SCHEMA_ID_V4,
        artifact_kind="expert_tensor_shard",
        version=4,
        status="active",
        description=(
            "v3 Gumbel self-play shard plus explicit active_seat and action-aligned exact "
            "wildlife/habitat/Nature afterstate components for grounded structured Q."
        ),
        compatible_readers=(
            "cascadiav3.expert_tensor_shards",
            "cascadiav3.torch_train_cascadiaformer",
        ),
    ),
}

# Rival contracts are additive and intentionally remain separate from the
# legacy replay/tensor validator families below.  Registering them here makes
# the repository-wide schema gate fail closed on an unknown Rival artifact
# without pretending a preference sidecar is an ExpertTensorShard.
for _rival_definition in RIVAL_SCHEMA_DEFINITIONS.values():
    SCHEMA_REGISTRY[_rival_definition.schema_id] = SchemaDefinition(
        schema_id=_rival_definition.schema_id,
        artifact_kind=_rival_definition.artifact_kind,
        version=_rival_definition.version,
        status="pre-gpu-cpu-reference",
        description=_rival_definition.description,
        compatible_readers=("cascadiav3.rival",),
    )

RIVAL_SCHEMA_IDS = frozenset(RIVAL_SCHEMA_DEFINITIONS)

REPLAY_JSONL_SCHEMA_IDS = {PRE_GPU_SCHEMA_ID, EXPERT_ROOT_SCHEMA_ID}
TENSOR_SCHEMA_IDS = {
    GREEDY_TENSOR_SHARD_SCHEMA_ID,
    EXPERT_TENSOR_SHARD_SCHEMA_ID,
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V2,
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V3,
    EXPERT_TENSOR_SHARD_SCHEMA_ID_V4,
}


class SchemaError(ValueError):
    """Raised when a fixture violates a Cascadia v3 schema contract."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def checksum(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _payload_without_checksum(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "checksum"}


def attach_checksum(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["checksum"] = checksum(_payload_without_checksum(out))
    return out


def registry_report(*, include_legacy: bool = True, include_expert: bool = True) -> dict[str, Any]:
    schema_ids: list[str] = []
    for schema_id in SCHEMA_REGISTRY:
        if schema_id == PRE_GPU_SCHEMA_ID and not include_legacy:
            continue
        if (
            schema_id
            in {
                EXPERT_ROOT_SCHEMA_ID,
                EXPERT_TENSOR_SHARD_SCHEMA_ID,
                EXPERT_TENSOR_SHARD_SCHEMA_ID_V2,
                EXPERT_TENSOR_SHARD_SCHEMA_ID_V3,
            }
            and not include_expert
        ):
            continue
        schema_ids.append(schema_id)
    missing = []
    if include_legacy:
        missing.extend(
            schema_id
            for schema_id in (PRE_GPU_SCHEMA_ID, GREEDY_TENSOR_SHARD_SCHEMA_ID)
            if schema_id not in schema_ids
        )
    if include_expert:
        missing.extend(
            schema_id
            for schema_id in (
                EXPERT_ROOT_SCHEMA_ID,
                EXPERT_TENSOR_SHARD_SCHEMA_ID,
                EXPERT_TENSOR_SHARD_SCHEMA_ID_V2,
                EXPERT_TENSOR_SHARD_SCHEMA_ID_V3,
            )
            if schema_id not in schema_ids
        )
    return {
        "status": "pass" if not missing else "fail",
        "schema_count": len(schema_ids),
        "schemas": [SCHEMA_REGISTRY[schema_id].to_dict() for schema_id in schema_ids],
        "missing_required": missing,
    }


def validate_schema_id(record: dict[str, Any], expected: str | None = None) -> None:
    schema_id = record.get("schema_id")
    if expected is not None:
        if schema_id != expected:
            raise SchemaError(f"schema_id mismatch: {schema_id!r}; expected {expected!r}")
        return
    if schema_id not in SCHEMA_REGISTRY:
        raise SchemaError(f"unknown schema_id: {schema_id!r}")


def validate_coord_ref(coord: dict[str, Any]) -> None:
    kind = coord.get("kind")
    q = coord.get("q")
    r = coord.get("r")
    s = coord.get("s")
    if not all(isinstance(v, int) for v in (q, r, s)):
        raise SchemaError(f"coordinate requires integer q/r/s: {coord!r}")
    if s != -q - r:
        raise SchemaError(f"coordinate violates s = -q - r: {coord!r}")

    radius6_member = max(abs(q), abs(r), abs(s)) <= 6
    if coord.get("radius6_member") is not radius6_member:
        raise SchemaError(f"radius6_member mismatch: {coord!r}")

    if kind == "canonical":
        index = coord.get("cell_index")
        if not radius6_member:
            raise SchemaError(f"canonical coord outside radius 6: {coord!r}")
        if not isinstance(index, int) or index < 0 or index >= RADIUS6_CELL_COUNT:
            raise SchemaError(f"canonical coord has invalid cell_index: {coord!r}")
        return

    if kind == "overflow":
        if radius6_member:
            raise SchemaError(f"overflow coord is inside radius 6: {coord!r}")
        if not isinstance(coord.get("owner_seat"), int):
            raise SchemaError(f"overflow coord requires owner_seat: {coord!r}")
        if not isinstance(coord.get("placement_id"), int):
            raise SchemaError(f"overflow coord requires placement_id: {coord!r}")
        if "cell_index" in coord and coord.get("cell_index") is not None:
            raise SchemaError(f"overflow coord cannot have cell_index: {coord!r}")
        return

    raise SchemaError(f"unknown coord_ref kind: {kind!r}")


def validate_action_token(action: dict[str, Any]) -> None:
    required = [
        "action_id",
        "active_seat",
        "cleanup_choice",
        "nature_spend",
        "draft_slot",
        "tile_ref",
        "wildlife_ref",
        "target_coord_ref",
        "rotation",
        "wildlife_coord_ref",
    ]
    missing = [field for field in required if field not in action]
    if missing:
        raise SchemaError(f"ActionToken missing fields: {missing}")
    if not isinstance(action["action_id"], str):
        raise SchemaError("ActionToken action_id must be a string")
    if not isinstance(action["draft_slot"], int):
        raise SchemaError("ActionToken draft_slot must be an int")
    validate_coord_ref(action["target_coord_ref"])
    validate_coord_ref(action["wildlife_coord_ref"])


def validate_score_decomposition(record: dict[str, Any]) -> None:
    final_score_vector = record.get("final_score_vector")
    decomp = record.get("score_decomposition")
    if not isinstance(final_score_vector, list) or len(final_score_vector) != 4:
        raise SchemaError("final_score_vector must have length 4")
    if not isinstance(decomp, dict):
        raise SchemaError("score_decomposition must be a map")
    for seat in range(4):
        key = str(seat)
        if key not in decomp:
            raise SchemaError(f"missing score decomposition for seat {seat}")
        parts = decomp[key]
        expected_keys = {"wildlife", "habitat", "nature_tokens", "total"}
        if set(parts) != expected_keys:
            raise SchemaError(f"seat {seat} score parts mismatch: {parts!r}")
        subtotal = parts["wildlife"] + parts["habitat"] + parts["nature_tokens"]
        if abs(subtotal - parts["total"]) > 1e-6:
            raise SchemaError(f"seat {seat} score parts do not sum to total")
        if abs(parts["total"] - final_score_vector[seat]) > 1e-6:
            raise SchemaError(f"seat {seat} total does not match final_score_vector")


def _validate_common_search_root_fields(record: dict[str, Any]) -> int:
    required = [
        "state_hash",
        "active_seat",
        "legal_actions",
        "priors",
        "visits",
        "per_action_Q",
        "selected_action",
        "chance_samples",
        "final_score_vector",
        "score_decomposition",
        "rank_vector",
        "checksum",
    ]
    missing = [field for field in required if field not in record]
    if missing:
        raise SchemaError(f"SearchRootRecord missing fields: {missing}")

    legal_actions = record["legal_actions"]
    if not isinstance(legal_actions, list) or not legal_actions:
        raise SchemaError("legal_actions must be a non-empty list")
    for action in legal_actions:
        validate_action_token(action)

    action_count = len(legal_actions)
    for field in ("priors", "visits", "per_action_Q"):
        values = record[field]
        if not isinstance(values, list) or len(values) != action_count:
            raise SchemaError(f"{field} must align one-to-one with legal_actions")
    for field in ("per_action_Q_variance", "per_action_Q_count", "per_action_truncated_count"):
        if field in record:
            values = record[field]
            if not isinstance(values, list) or len(values) != action_count:
                raise SchemaError(f"{field} must align one-to-one with legal_actions")

    prior_sum = sum(record["priors"])
    if abs(prior_sum - 1.0) > 1e-6:
        raise SchemaError(f"priors must sum to 1.0, got {prior_sum}")

    action_ids = [action["action_id"] for action in legal_actions]
    if len(action_ids) != len(set(action_ids)):
        raise SchemaError("legal action ids must be unique")
    if record["selected_action"] not in action_ids:
        raise SchemaError("selected_action must reference a legal action id")

    if not isinstance(record["rank_vector"], list) or len(record["rank_vector"]) != 4:
        raise SchemaError("rank_vector must have length 4")
    validate_score_decomposition(record)
    return action_count


def _validate_checksum(record: dict[str, Any], label: str) -> None:
    expected_checksum = checksum(_payload_without_checksum(record))
    if record["checksum"] != expected_checksum:
        raise SchemaError(f"{label} checksum mismatch")


def validate_pre_gpu_search_root_record(record: dict[str, Any]) -> None:
    validate_schema_id(record, PRE_GPU_SCHEMA_ID)
    _validate_common_search_root_fields(record)
    _validate_checksum(record, "SearchRootRecord")


def validate_expert_root_record(record: dict[str, Any]) -> None:
    validate_schema_id(record, EXPERT_ROOT_SCHEMA_ID)
    required = [
        "ruleset_id",
        "seed",
        "ply",
        "public_hash",
        "source_hash",
        "binary_hash",
        "root_replay",
        "actor_identity",
        "opponent_identities",
        "model_identity",
        "search_identity",
        "rng_identity",
        "action_ids",
        "afterstate_hashes",
        "exact_afterstate_score_active",
        "per_action_score_to_go",
        "per_action_Q_valid",
    ]
    missing = [field for field in required if field not in record]
    if missing:
        raise SchemaError(f"ExpertRootRecord missing fields: {missing}")

    action_count = _validate_common_search_root_fields(record)
    action_ids = [action["action_id"] for action in record["legal_actions"]]
    if record["action_ids"] != action_ids:
        raise SchemaError("action_ids must exactly match legal_actions order")

    for field in (
        "afterstate_hashes",
        "exact_afterstate_score_active",
        "per_action_score_to_go",
        "per_action_Q_valid",
    ):
        values = record[field]
        if not isinstance(values, list) or len(values) != action_count:
            raise SchemaError(f"{field} must align one-to-one with legal_actions")

    if not all(isinstance(value, bool) for value in record["per_action_Q_valid"]):
        raise SchemaError("per_action_Q_valid must be a boolean mask")

    for index, valid in enumerate(record["per_action_Q_valid"]):
        if not valid:
            continue
        q = float(record["per_action_Q"][index])
        afterstate_score = float(record["exact_afterstate_score_active"][index])
        score_to_go = float(record["per_action_score_to_go"][index])
        if abs((afterstate_score + score_to_go) - q) > 1.0e-5:
            raise SchemaError(f"Q target semantics mismatch at action {index}")

    root_replay = record["root_replay"]
    if not isinstance(root_replay, dict):
        raise SchemaError("root_replay must be a map")
    for field in ("seed_u64", "replay_prefix", "market_prelude", "root_public_hash"):
        if field not in root_replay:
            raise SchemaError(f"root_replay missing {field}")
    if not isinstance(root_replay["replay_prefix"], list):
        raise SchemaError("root_replay.replay_prefix must be a list")

    if int(record["seed"]) != int(root_replay["seed_u64"]):
        raise SchemaError("seed must match root_replay.seed_u64")
    if record["public_hash"] != root_replay["root_public_hash"]:
        raise SchemaError("public_hash must match root_replay.root_public_hash")

    if not isinstance(record["opponent_identities"], list) or len(record["opponent_identities"]) != 3:
        raise SchemaError("opponent_identities must contain the three non-active seats")

    for sample in record["chance_samples"]:
        if not isinstance(sample, dict):
            raise SchemaError("chance_samples entries must be maps")
        for field in (
            "sample_id",
            "action_id",
            "seed",
            "probability",
            "logprob",
            "before_hash",
            "after_hash",
            "before_public_hash",
            "after_public_hash",
            "public_delta",
            "private_audit_hash",
        ):
            if field not in sample:
                raise SchemaError(f"chance sample missing {field}")
        if sample["action_id"] not in action_ids:
            raise SchemaError("chance sample action_id must reference a legal action")

    metadata = record.get("metadata", {})
    if metadata:
        retained = metadata.get("full_legal_action_count")
        if retained is not None and int(retained) != action_count:
            raise SchemaError("full_legal_action_count must equal exported legal action count")
        coverage = metadata.get("legal_action_coverage")
        if coverage is not None and abs(float(coverage) - 1.0) > 1.0e-9:
            raise SchemaError("expert roots must report 100% legal action coverage")

    _validate_checksum(record, "ExpertRootRecord")


def validate_search_root_record(record: dict[str, Any]) -> None:
    schema_id = record.get("schema_id")
    if schema_id == PRE_GPU_SCHEMA_ID:
        validate_pre_gpu_search_root_record(record)
    elif schema_id == EXPERT_ROOT_SCHEMA_ID:
        validate_expert_root_record(record)
    else:
        raise SchemaError(f"unsupported search root schema_id: {schema_id!r}")


def validate_replay_manifest(manifest: dict[str, Any]) -> None:
    validate_schema_id(manifest)
    schema_id = manifest.get("schema_id")
    if schema_id not in REPLAY_JSONL_SCHEMA_IDS and schema_id not in TENSOR_SCHEMA_IDS:
        raise SchemaError(f"unsupported manifest schema_id: {schema_id!r}")
    required = [
        "source_generator",
        "seed_domain",
        "record_count",
        "checksum",
        "scientific_eligibility",
        "created_at_utc",
        "format",
    ]
    missing = [field for field in required if field not in manifest]
    if missing:
        raise SchemaError(f"ReplayShardManifest missing fields: {missing}")
    if manifest["scientific_eligibility"] not in {
        "dry_run",
        "debug",
        "behavior_clone_pretraining",
        "training_candidate",
        "evaluation_locked",
    }:
        raise SchemaError("unknown scientific_eligibility")
    if manifest["format"] not in {"jsonl", "binary", "npz"}:
        raise SchemaError("unknown replay format")
