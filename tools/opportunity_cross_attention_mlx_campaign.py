#!/usr/bin/env python3
"""Authorize and preflight the four-host ADR 0166 MLX campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.opportunity_cross_attention_mlx_model import (
    ARMS,
    OpportunityCrossAttentionModelConfig,
    OpportunityCrossAttentionRanker,
)
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ADR_ID,
    ARM_HOSTS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    RELATIONAL_DATA_ARM,
    TRAINING_SEED,
    OpportunityCrossAttentionTrainingProtocol,
    normalize_host,
)
from cascadia_mlx.opportunity_cross_attention_mlx_train import (
    cross_arm_initialization,
    load_verified_warm_start,
    verify_zero_init_prediction_parity,
)
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_cache import (
    RelationalSubstrateMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.relational_substrate_mlx_train import (
    scientific_batch_blake3,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from opportunity_cross_attention_mlx_smoke_compare import (
    EXPECTED_HOSTS,
)
from opportunity_cross_attention_mlx_smoke_compare import (
    PASS as SMOKE_PASS,
)

HOSTS = EXPECTED_HOSTS


class CampaignError(RuntimeError):
    """The ADR 0166 campaign cannot proceed with identity drift."""


def create_authorization(
    *,
    repository: Path,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache: Path,
    relational_cache: Path,
    s1_cache: Path,
    r6_binary: Path,
    warm_start_run_dir: Path,
    warm_start_report: Path,
    smoke_proof: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Create launch authorization after the four-host smoke passes."""
    if not approved_by.strip():
        raise CampaignError("ADR 0166 authorization requires an approver")
    (
        r3,
        relational,
        supply,
        train,
        _validation,
    ) = _bind_open_data(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        r3_cache=r3_cache,
        relational_cache=relational_cache,
        s1_cache=s1_cache,
        verify_all=True,
    )
    warm_model, warm_start, checkpoint = load_verified_warm_start(
        warm_start_run_dir,
        warm_start_report,
    )
    initialization = cross_arm_initialization(
        warm_model,
        warm_start_checkpoint=checkpoint,
    )
    source = source_provenance(repository)
    open_data = open_data_verification_identity(
        cache=relational,
        s1_cache=supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    smoke = validate_smoke_proof(
        smoke_proof,
        source_blake3=source["v2_source_blake3"],
        warm_start_id=warm_start["warm_start_id"],
        r3_cache_id=r3.cache_id,
        relational_cache_id=relational.cache_id,
        s1_cache_id=supply.cache_id,
        r6_binary_blake3=_checksum(r6_binary),
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "source_blake3": source["v2_source_blake3"],
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational.cache_id,
        "s1_cache_id": supply.cache_id,
        "r6_binary_blake3": _checksum(r6_binary),
        "authorized_arms": list(ARMS),
        "arm_hosts": ARM_HOSTS,
        "protocol": OpportunityCrossAttentionTrainingProtocol().to_dict(),
        "open_data_verification": open_data,
        "open_data_verification_id": open_data_verification_id(open_data),
        "warm_start": warm_start,
        "cross_arm_initialization": initialization,
        "first_scientific_batch": _first_batch_identity(train),
        "smoke_proof_id": smoke["proof_id"],
        "approved_by": approved_by.strip(),
        "approved_unix_ms": (
            time.time_ns() // 1_000_000
            if approved_unix_ms is None
            else approved_unix_ms
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": canonical_blake3(identity),
        "identity": identity,
        "launch_effect": {
            "training_started": False,
            "queue_modified": False,
            "gameplay_authorized": False,
            "sealed_test_authorized": False,
        },
    }


def validate_authorization(
    authorization: dict[str, Any],
    *,
    repository: Path,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache: R3ActionEditMlxCache,
    relational_cache: RelationalSubstrateMlxCache,
    s1_cache: S1ExactSupplyCache,
    r6_binary: Path,
    warm_start_run_dir: Path,
    warm_start_report: Path,
    smoke_proof: Path,
    train: object,
) -> dict[str, Any]:
    """Reconstruct every authorization input and require exact equality."""
    identity = authorization.get("identity")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("protocol_id") != PROTOCOL_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(identity, dict)
        or canonical_blake3(identity)
        != authorization.get("authorization_id")
    ):
        raise CampaignError("ADR 0166 authorization is malformed")
    warm_model, warm_start, checkpoint = load_verified_warm_start(
        warm_start_run_dir,
        warm_start_report,
    )
    initialization = cross_arm_initialization(
        warm_model,
        warm_start_checkpoint=checkpoint,
    )
    source = source_provenance(repository)
    open_data = open_data_verification_identity(
        cache=relational_cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    smoke = validate_smoke_proof(
        smoke_proof,
        source_blake3=source["v2_source_blake3"],
        warm_start_id=warm_start["warm_start_id"],
        r3_cache_id=r3_cache.cache_id,
        relational_cache_id=relational_cache.cache_id,
        s1_cache_id=s1_cache.cache_id,
        r6_binary_blake3=_checksum(r6_binary),
    )
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "source_blake3": source["v2_source_blake3"],
        "r3_cache_id": r3_cache.cache_id,
        "relational_cache_id": relational_cache.cache_id,
        "s1_cache_id": s1_cache.cache_id,
        "r6_binary_blake3": _checksum(r6_binary),
        "authorized_arms": list(ARMS),
        "arm_hosts": ARM_HOSTS,
        "protocol": OpportunityCrossAttentionTrainingProtocol().to_dict(),
        "open_data_verification": open_data,
        "open_data_verification_id": open_data_verification_id(open_data),
        "warm_start": warm_start,
        "cross_arm_initialization": initialization,
        "first_scientific_batch": _first_batch_identity(train),
        "smoke_proof_id": smoke["proof_id"],
        "approved_by": identity.get("approved_by"),
        "approved_unix_ms": identity.get("approved_unix_ms"),
    }
    if identity != expected:
        raise CampaignError(
            "ADR 0166 authorization is stale for current inputs"
        )
    return authorization


def validate_smoke_proof(
    path: Path,
    *,
    source_blake3: str,
    warm_start_id: str,
    r3_cache_id: str,
    relational_cache_id: str,
    s1_cache_id: str,
    r6_binary_blake3: str,
) -> dict[str, Any]:
    """Require the order-invariant common-arm smoke proof."""
    proof = _read_json(path, "ADR 0166 cross-host smoke proof")
    identity = proof.get("scientific_identity")
    checks = identity.get("checks") if isinstance(identity, dict) else None
    if (
        proof.get("schema_version") != 1
        or proof.get("experiment_id") != EXPERIMENT_ID
        or proof.get("protocol_id") != PROTOCOL_ID
        or proof.get("adr") != ADR_ID
        or proof.get("classification") != SMOKE_PASS
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != proof.get("proof_id")
        or identity.get("hosts") != list(HOSTS)
        or identity.get("source_blake3") != source_blake3
        or identity.get("warm_start_id") != warm_start_id
        or identity.get("r3_cache_id") != r3_cache_id
        or identity.get("relational_cache_id") != relational_cache_id
        or identity.get("s1_cache_id") != s1_cache_id
        or identity.get("r6_binary_blake3") != r6_binary_blake3
        or not isinstance(checks, dict)
        or set(checks) != set(HOSTS)
        or any(value is not True for value in checks.values())
    ):
        raise CampaignError("ADR 0166 cross-host smoke proof is invalid")
    return proof


def run_preflight(
    *,
    host: str,
    arm: str,
    repository: Path,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache_path: Path,
    relational_cache_path: Path,
    s1_cache_path: Path,
    r6_binary: Path,
    warm_start_run_dir: Path,
    warm_start_report: Path,
    authorization_path: Path,
    smoke_proof: Path,
) -> dict[str, Any]:
    """Validate one assigned host without starting production."""
    actual_host = normalize_host(socket.gethostname().split(".")[0])
    if host != actual_host or ARM_HOSTS.get(arm) != host:
        raise CampaignError("ADR 0166 preflight host/arm assignment is invalid")
    (
        r3,
        relational,
        supply,
        train,
        validation,
    ) = _bind_open_data(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        r3_cache=r3_cache_path,
        relational_cache=relational_cache_path,
        s1_cache=s1_cache_path,
        verify_all=False,
    )
    authorization = validate_authorization(
        _read_json(authorization_path, "ADR 0166 authorization"),
        repository=repository,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        r3_cache=r3,
        relational_cache=relational,
        s1_cache=supply,
        r6_binary=r6_binary,
        warm_start_run_dir=warm_start_run_dir,
        warm_start_report=warm_start_report,
        smoke_proof=smoke_proof,
        train=train,
    )
    runtime = _runtime_identity()
    warm_model, warm_start, checkpoint = load_verified_warm_start(
        warm_start_run_dir,
        warm_start_report,
    )
    initialization = cross_arm_initialization(
        warm_model,
        warm_start_checkpoint=checkpoint,
    )
    mx.random.seed(TRAINING_SEED)
    candidate = OpportunityCrossAttentionRanker(
        OpportunityCrossAttentionModelConfig(arm=arm)
    )
    candidate.load_weights(str(checkpoint / "model.safetensors"), strict=False)
    candidate.freeze_base_for_adapter_training()
    zero_parity = verify_zero_init_prediction_parity(
        warm_model,
        candidate,
        validation,
    )
    first_batch = _first_batch_identity(train)
    expected = authorization["identity"]
    smoke = validate_smoke_proof(
        smoke_proof,
        source_blake3=expected["source_blake3"],
        warm_start_id=warm_start["warm_start_id"],
        r3_cache_id=r3.cache_id,
        relational_cache_id=relational.cache_id,
        s1_cache_id=supply.cache_id,
        r6_binary_blake3=_checksum(r6_binary),
    )
    checks = {
        "authorization_verified": True,
        "source_verified": (
            source_provenance(repository)["v2_source_blake3"]
            == expected["source_blake3"]
        ),
        "open_data_verified": (
            open_data_verification_identity(
                cache=relational,
                s1_cache=supply,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
            )
            == expected["open_data_verification"]
        ),
        "warm_start_verified": warm_start == expected["warm_start"],
        "initialization_parity_verified": (
            initialization == expected["cross_arm_initialization"]
        ),
        "zero_init_prediction_parity_verified": (
            zero_parity.get("exact_array_equal") is True
        ),
        "first_scientific_batch_verified": (
            first_batch == expected["first_scientific_batch"]
        ),
        "smoke_replay_verified": smoke["proof_id"]
        == expected["smoke_proof_id"],
        "r6_binary_verified": (
            _checksum(r6_binary) == expected["r6_binary_blake3"]
        ),
        "apple_silicon_verified": runtime["machine"] == "arm64",
        "mlx_gpu_verified": "gpu" in runtime["default_device"].lower(),
        "host_assignment_verified": ARM_HOSTS[arm] == host,
        "open_data_only_verified": True,
        "production_training_started": False,
    }
    if any(
        value is not True
        for key, value in checks.items()
        if key != "production_training_started"
    ) or checks["production_training_started"] is not False:
        raise CampaignError(f"ADR 0166 host preflight failed: {checks}")
    identity = {
        "authorization_id": authorization["authorization_id"],
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational.cache_id,
        "s1_cache_id": supply.cache_id,
        "r6_binary_blake3": _checksum(r6_binary),
        "arm": arm,
        "host": host,
        "runtime": runtime,
        "source_blake3": expected["source_blake3"],
        "open_data_verification_id": expected[
            "open_data_verification_id"
        ],
        "warm_start_id": warm_start["warm_start_id"],
        "mlx_gpu_verified": checks["mlx_gpu_verified"],
        "open_data_only_verified": checks["open_data_only_verified"],
        "warm_start_verified": checks["warm_start_verified"],
        "initialization_parity_verified": checks[
            "initialization_parity_verified"
        ],
        "zero_init_prediction_parity_verified": checks[
            "zero_init_prediction_parity_verified"
        ],
        "smoke_replay_verified": checks["smoke_replay_verified"],
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": arm,
        "preflight_id": canonical_blake3(identity),
        "identity": identity,
        "checks": checks,
        "zero_init_prediction_parity": zero_parity,
        "claims": {
            "preflight_complete": True,
            "production_training_started": False,
            "gameplay_strength_measured": False,
        },
    }


def _bind_open_data(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache: Path,
    relational_cache: Path,
    s1_cache: Path,
    verify_all: bool,
) -> tuple[
    R3ActionEditMlxCache,
    RelationalSubstrateMlxCache,
    S1ExactSupplyCache,
    object,
    object,
]:
    r3 = R3ActionEditMlxCache(
        r3_cache,
        verify_checksums=verify_all,
        verify_semantics=verify_all,
        require_complete=True,
    )
    relational = RelationalSubstrateMlxCache(
        relational_cache,
        r3_cache=r3,
        verify_checksums=verify_all,
        verify_semantics=verify_all,
        require_complete=True,
    )
    supply = S1ExactSupplyCache(
        s1_cache,
        verify_checksums=verify_all,
        verify_semantics=verify_all,
        require_complete=True,
    )
    train = relational.bind_dataset(
        train_dataset,
        s1_cache=supply,
        verify_dataset_checksums=verify_all,
    )
    validation = relational.bind_dataset(
        validation_dataset,
        s1_cache=supply,
        verify_dataset_checksums=verify_all,
    )
    return r3, relational, supply, train, validation


def _first_batch_identity(train: object) -> dict[str, Any]:
    batch = train.deterministic_training_batch(
        step=0,
        seed=TRAINING_SEED,
        arm=RELATIONAL_DATA_ARM,
    )
    mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)
    return {
        "step": 0,
        "seed": TRAINING_SEED,
        "data_arm": RELATIONAL_DATA_ARM,
        "scientific_batch_blake3": scientific_batch_blake3(batch),
        "groups": int(mask.shape[0]),
        "candidates": int(mask.sum()),
        "r2_tokens": int(
            np.asarray(batch.parent.r2_token_mask, dtype=np.bool_).sum()
        ),
        "supply_tokens": int(
            np.asarray(batch.supply_mask, dtype=np.bool_).sum()
        ),
        "relational_tokens": int(
            np.asarray(batch.parent.relational_mask, dtype=np.bool_).sum()
        ),
        "derivative_nonzero_values": int(
            np.count_nonzero(
                np.asarray(batch.derivative_features)[mask]
            )
        ),
    }


def _runtime_identity() -> dict[str, Any]:
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": version("mlx"),
        "numpy": version("numpy"),
        "default_device": str(mx.default_device()),
        "host": normalize_host(socket.gethostname().split(".")[0]),
    }


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument("--relational-cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--r6-binary", type=Path, required=True)
    parser.add_argument("--warm-start-run-dir", type=Path, required=True)
    parser.add_argument("--warm-start-report", type=Path, required=True)
    parser.add_argument("--smoke-proof", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    _common_arguments(authorize)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--approved-unix-ms", type=int)
    authorize.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    _common_arguments(preflight)
    preflight.add_argument("--host", choices=HOSTS, required=True)
    preflight.add_argument("--arm", choices=ARMS, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "authorize":
        result = create_authorization(
            repository=args.repository,
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache=args.r3_cache,
            relational_cache=args.relational_cache,
            s1_cache=args.s1_cache,
            r6_binary=args.r6_binary,
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_report=args.warm_start_report,
            smoke_proof=args.smoke_proof,
            approved_by=args.approved_by,
            approved_unix_ms=args.approved_unix_ms,
        )
    else:
        result = run_preflight(
            host=args.host,
            arm=args.arm,
            repository=args.repository,
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache_path=args.r3_cache,
            relational_cache_path=args.relational_cache,
            s1_cache_path=args.s1_cache,
            r6_binary=args.r6_binary,
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_report=args.warm_start_report,
            authorization_path=args.authorization,
            smoke_proof=args.smoke_proof,
        )
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
