#!/usr/bin/env python3
"""Authorize, preflight, and describe the four-host ADR 0161 campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.r3_action_edit_mlx_cache import R3ActionEditMlxCache
from cascadia_mlx.relational_substrate_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    S5_ARM,
    RelationalSubstrateMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.relational_substrate_mlx_train import (
    ARM_HOSTS,
    TRAINING_SEED,
    RelationalSubstrateTrainingProtocol,
    cross_arm_initialization,
    runtime_identity,
    scientific_batch_blake3,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from relational_substrate_mlx_smoke_compare import (
    PASS as SMOKE_PASS,
)
from relational_substrate_mlx_smoke_compare import (
    SMOKE_ARM,
    SMOKE_STEPS,
)

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
FROZEN_SOURCE_RELATIVE = EXPERIMENT_ROOT / "frozen-source"
DEFAULT_TRAIN_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-train"
)
DEFAULT_VALIDATION_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-validation"
)
DEFAULT_R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
DEFAULT_S1_CACHE = Path(
    "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/"
    "cache/2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
DEFAULT_R6_BINARY = Path(
    "tools/relational_feature_census/target/release/"
    "relational-substrate-r6-replay"
)


class CampaignError(RuntimeError):
    """The ADR 0161 campaign cannot proceed without identity drift."""


def create_authorization(
    *,
    repository: Path,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache: Path,
    relational_cache: Path,
    s1_cache: Path,
    r6_binary: Path,
    smoke_proof: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Create immutable launch authorization without starting training."""
    if not approved_by.strip():
        raise CampaignError(
            "ADR 0161 authorization requires a nonempty approver"
        )
    repository = repository.resolve()
    (
        r3,
        relational,
        exact_supply,
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
    smoke = validate_smoke_proof(
        smoke_proof,
        r3_cache_id=r3.cache_id,
        relational_cache_id=relational.cache_id,
        s1_cache_id=exact_supply.cache_id,
        r6_binary_blake3=_checksum(r6_binary),
    )
    source = source_provenance(repository)
    initialization = cross_arm_initialization()
    first_batch = _cross_arm_first_batch_identity(train)
    open_data = open_data_verification_identity(
        cache=relational,
        s1_cache=exact_supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "source_blake3": source["v2_source_blake3"],
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational.cache_id,
        "s1_cache_id": exact_supply.cache_id,
        "r6_binary_blake3": _checksum(r6_binary),
        "authorized_arms": list(ARMS),
        "arm_hosts": ARM_HOSTS,
        "protocol": RelationalSubstrateTrainingProtocol().to_dict(),
        "open_data_verification": open_data,
        "open_data_verification_id": open_data_verification_id(
            open_data
        ),
        "cross_arm_initialization": initialization,
        "cross_arm_first_batch": first_batch,
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
    smoke_proof: Path,
    train: Any,
) -> dict[str, Any]:
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
        raise CampaignError("ADR 0161 authorization is malformed")
    smoke = validate_smoke_proof(
        smoke_proof,
        r3_cache_id=r3_cache.cache_id,
        relational_cache_id=relational_cache.cache_id,
        s1_cache_id=s1_cache.cache_id,
        r6_binary_blake3=_checksum(r6_binary),
    )
    source = source_provenance(repository)
    open_data = open_data_verification_identity(
        cache=relational_cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
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
        "protocol": RelationalSubstrateTrainingProtocol().to_dict(),
        "open_data_verification": open_data,
        "open_data_verification_id": open_data_verification_id(
            open_data
        ),
        "cross_arm_initialization": cross_arm_initialization(),
        "cross_arm_first_batch": _cross_arm_first_batch_identity(train),
        "smoke_proof_id": smoke["proof_id"],
        "approved_by": identity.get("approved_by"),
        "approved_unix_ms": identity.get("approved_unix_ms"),
    }
    if identity != expected:
        raise CampaignError(
            "ADR 0161 authorization is stale for current inputs"
        )
    return authorization


def validate_smoke_proof(
    path: Path,
    *,
    r3_cache_id: str,
    relational_cache_id: str,
    s1_cache_id: str,
    r6_binary_blake3: str,
) -> dict[str, Any]:
    proof = _read_json(path, "ADR 0161 cross-host smoke proof")
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
        or identity.get("arm") != SMOKE_ARM
        or identity.get("steps") != SMOKE_STEPS
        or identity.get("hosts") != ["john1", "john4"]
        or identity.get("r3_cache_id") != r3_cache_id
        or identity.get("relational_cache_id")
        != relational_cache_id
        or identity.get("s1_cache_id") != s1_cache_id
        or identity.get("r6_binary_blake3")
        != r6_binary_blake3
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or proof.get("claims", {}).get("production_training_started")
        is not False
    ):
        raise CampaignError(
            "ADR 0161 cross-host smoke proof is invalid"
        )
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
    authorization_path: Path,
    smoke_proof: Path,
) -> dict[str, Any]:
    """Reverify one assigned host without starting an optimizer."""
    if host not in HOSTS or arm not in ARMS or ARM_HOSTS[arm] != host:
        raise CampaignError(
            "ADR 0161 preflight host/arm assignment is invalid"
        )
    repository = repository.resolve()
    (
        r3,
        relational,
        exact_supply,
        train,
        validation,
    ) = _bind_open_data(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        r3_cache=r3_cache_path,
        relational_cache=relational_cache_path,
        s1_cache=s1_cache_path,
        verify_all=True,
    )
    authorization = validate_authorization(
        _read_json(authorization_path, "ADR 0161 authorization"),
        repository=repository,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        r3_cache=r3,
        relational_cache=relational,
        s1_cache=exact_supply,
        r6_binary=r6_binary,
        smoke_proof=smoke_proof,
        train=train,
    )
    source = source_provenance(repository)
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    runtime = runtime_identity()
    actual_host = _normalize_host(socket.gethostname().split(".")[0])
    open_data = open_data_verification_identity(
        cache=relational,
        s1_cache=exact_supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    first_batch = _first_batch_identity(train, arm)
    expected_first_batch = authorization["identity"][
        "cross_arm_first_batch"
    ]
    parent_surface = _parent_surface_verified(arm, first_batch)
    derivative_surface = _derivative_surface_verified(arm, first_batch)
    hidden = relational.manifest["hidden_information"]
    checks = {
        "authorization_verified": True,
        "r3_cache_checksums_and_semantics_verified": True,
        "relational_cache_checksums_and_semantics_verified": True,
        "s1_cache_checksums_and_semantics_verified": True,
        "train_dataset_verified": train.group_count == 560,
        "validation_dataset_verified": validation.group_count == 240,
        "apple_silicon_verified": (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
        ),
        "mlx_gpu_verified": "gpu" in str(mx.default_device()).lower(),
        "python_bytecode_disabled": sys.dont_write_bytecode,
        "host_assignment_verified": (
            actual_host == host == runtime["host"]
        ),
        "source_identity_verified": (
            source["v2_source_blake3"]
            == authorization["identity"]["source_blake3"]
        ),
        "open_data_verification_identity_verified": (
            open_data
            == authorization["identity"]["open_data_verification"]
            and open_data_verification_id(open_data)
            == authorization["identity"][
                "open_data_verification_id"
            ]
        ),
        "initialization_parity_verified": (
            cross_arm_initialization()
            == authorization["identity"][
                "cross_arm_initialization"
            ]
        ),
        "smoke_replay_verified": (
            validate_smoke_proof(
                smoke_proof,
                r3_cache_id=r3.cache_id,
                relational_cache_id=relational.cache_id,
                s1_cache_id=exact_supply.cache_id,
                r6_binary_blake3=_checksum(r6_binary),
            )["proof_id"]
            == authorization["identity"]["smoke_proof_id"]
        ),
        "candidate_identity_verified": (
            first_batch["scientific_batch_blake3"]
            == expected_first_batch["common_scientific_batch_blake3"]
            and first_batch
            == expected_first_batch["arms"][arm]
        ),
        "parent_surface_verified": parent_surface,
        "derivative_surface_verified": derivative_surface,
        "r6_binary_verified": (
            _checksum(r6_binary)
            == authorization["identity"]["r6_binary_blake3"]
        ),
        "open_data_only_verified": all(
            hidden[field] is expected
            for field, expected in (
                ("open_train_and_validation_only", True),
                ("hidden_order_exported", False),
                ("excluded_tile_identity_exported", False),
                ("future_refill_exported", False),
                ("sealed_test_opened", False),
                ("gameplay_opened", False),
                ("teacher_values_used_for_features", False),
            )
        ),
        "production_training_started": False,
    }
    if not all(
        value
        for key, value in checks.items()
        if key != "production_training_started"
    ):
        raise CampaignError(
            f"ADR 0161 host preflight failed: {checks}"
        )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "authorization_id": authorization["authorization_id"],
        "r3_cache_id": r3.cache_id,
        "relational_cache_id": relational.cache_id,
        "s1_cache_id": exact_supply.cache_id,
        "r6_binary_blake3": _checksum(r6_binary),
        "arm": arm,
        "host": host,
        "runtime": runtime,
        "source_blake3": source["v2_source_blake3"],
        "open_data_verification_id": open_data_verification_id(
            open_data
        ),
        "mlx_gpu_verified": checks["mlx_gpu_verified"],
        "open_data_only_verified": checks[
            "open_data_only_verified"
        ],
        "initialization_parity_verified": checks[
            "initialization_parity_verified"
        ],
        "smoke_replay_verified": checks["smoke_replay_verified"],
        "candidate_identity_verified": checks[
            "candidate_identity_verified"
        ],
        "parent_surface_verified": checks[
            "parent_surface_verified"
        ],
        "derivative_surface_verified": checks[
            "derivative_surface_verified"
        ],
        "first_batch": first_batch,
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
        "claims": {
            "preflight_complete": True,
            "production_training_started": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
        },
    }


def build_task_specification(
    *,
    relational_cache_relative: Path,
) -> dict[str, Any]:
    """Describe the nonduplicative four-host production graph."""
    tasks: list[dict[str, Any]] = [
        {
            "id": "cache-and-source-fanout",
            "host": "john1",
            "kind": "shared-prerequisite",
            "dependencies": [],
            "uses_mlx": False,
            "purpose": (
                "Checksum-fanout source, relational cache, smoke proof, "
                "authorization, and R6 binary to treatment hosts"
            ),
        }
    ]
    preflight_ids = [
        f"relmlx-preflight-{host}" for host in ARM_HOSTS.values()
    ]
    train_ids = [
        f"relmlx-train-{_slug(arm)}" for arm in ARM_HOSTS
    ]
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        preflight_id = f"relmlx-preflight-{host}"
        root = REMOTE_ROOTS[host]
        source = root / FROZEN_SOURCE_RELATIVE
        preflight = EXPERIMENT_ROOT / f"reports/preflight-{host}.json"
        tasks.append(
            {
                "id": preflight_id,
                "host": host,
                "kind": "preflight",
                "arm": arm,
                "dependencies": ["cache-and-source-fanout"],
                "uses_mlx": True,
                "command": [
                    "/usr/bin/env",
                    f"PYTHONPATH={source / 'python'}",
                    str(root / ".venv/bin/python"),
                    "-B",
                    str(source / "tools/relational_substrate_mlx_campaign.py"),
                    "preflight",
                    "--host",
                    host,
                    "--arm",
                    arm,
                    "--repository",
                    str(source),
                    "--train-dataset",
                    str(root / DEFAULT_TRAIN_DATASET),
                    "--validation-dataset",
                    str(root / DEFAULT_VALIDATION_DATASET),
                    "--r3-cache",
                    str(root / DEFAULT_R3_CACHE),
                    "--relational-cache",
                    str(root / relational_cache_relative),
                    "--s1-cache",
                    str(root / DEFAULT_S1_CACHE),
                    "--r6-binary",
                    str(root / DEFAULT_R6_BINARY),
                    "--authorization",
                    str(
                        root
                        / EXPERIMENT_ROOT
                        / "control/authorization.json"
                    ),
                    "--smoke-proof",
                    str(
                        root
                        / EXPERIMENT_ROOT
                        / "control/cross-host-smoke-proof.json"
                    ),
                    "--output",
                    str(root / preflight),
                ],
            }
        )
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        train_id = f"relmlx-train-{slug}"
        root = REMOTE_ROOTS[host]
        source = root / FROZEN_SOURCE_RELATIVE
        preflight = EXPERIMENT_ROOT / f"reports/preflight-{host}.json"
        tasks.append(
            {
                "id": train_id,
                "host": host,
                "kind": "independent-experiment",
                "arm": arm,
                "dependencies": preflight_ids,
                "uses_mlx": True,
                "command": [
                    "/usr/bin/env",
                    f"PYTHONPATH={source / 'python'}",
                    str(root / ".venv/bin/python"),
                    "-B",
                    "-m",
                    "cascadia_mlx.relational_substrate_mlx_train",
                    "--train-dataset",
                    str(root / DEFAULT_TRAIN_DATASET),
                    "--validation-dataset",
                    str(root / DEFAULT_VALIDATION_DATASET),
                    "--r3-cache",
                    str(root / DEFAULT_R3_CACHE),
                    "--relational-cache",
                    str(root / relational_cache_relative),
                    "--s1-cache",
                    str(root / DEFAULT_S1_CACHE),
                    "--r6-binary",
                    str(root / DEFAULT_R6_BINARY),
                    "--run-dir",
                    str(root / EXPERIMENT_ROOT / f"runs/{slug}"),
                    "--output",
                    str(
                        root / EXPERIMENT_ROOT / f"reports/{slug}.json"
                    ),
                    "--authorization",
                    str(
                        root
                        / EXPERIMENT_ROOT
                        / "control/authorization.json"
                    ),
                    "--preflight",
                    str(root / preflight),
                    "--arm",
                    arm,
                ],
            }
        )
    tasks.extend(
        [
            {
                "id": "relmlx-fanout-control-run",
                "host": "john1",
                "kind": "shared-prerequisite",
                "dependencies": train_ids,
                "uses_mlx": False,
                "purpose": (
                    "Fan the exact C0 run to john2/john3/john4 for "
                    "same-host serving controls"
                ),
            },
            {
                "id": "relmlx-paired-controls",
                "hosts": ["john2", "john3", "john4"],
                "kind": "three-independent-replays",
                "dependencies": ["relmlx-fanout-control-run"],
                "uses_mlx": True,
            },
            {
                "id": "relmlx-classify",
                "host": "john1",
                "kind": "decision-terminal",
                "dependencies": ["relmlx-paired-controls"],
                "uses_mlx": False,
                "tool": "tools/relational_substrate_mlx_report.py",
            },
        ]
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "maximum_concurrent_primary_experiments": 4,
        "duplicate_primary_work": False,
        "tasks": tasks,
    }
    return {
        "schema_version": 1,
        "specification_id": canonical_blake3(identity),
        "scientific_identity": identity,
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
    Any,
    Any,
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
    exact_supply = S1ExactSupplyCache(
        s1_cache,
        verify_checksums=verify_all,
        verify_semantics=verify_all,
        require_complete=True,
    )
    train = relational.bind_dataset(
        train_dataset,
        s1_cache=exact_supply,
        verify_dataset_checksums=verify_all,
    )
    validation = relational.bind_dataset(
        validation_dataset,
        s1_cache=exact_supply,
        verify_dataset_checksums=verify_all,
    )
    return r3, relational, exact_supply, train, validation


def _cross_arm_first_batch_identity(train: Any) -> dict[str, Any]:
    arms = {arm: _first_batch_identity(train, arm) for arm in ARMS}
    hashes = {
        value["scientific_batch_blake3"] for value in arms.values()
    }
    candidates = {value["candidates"] for value in arms.values()}
    if len(hashes) != 1 or len(candidates) != 1:
        raise CampaignError(
            "ADR 0161 first scientific batch differs across arms"
        )
    return {
        "step": 0,
        "seed": TRAINING_SEED,
        "common_scientific_batch_blake3": next(iter(hashes)),
        "common_candidates": next(iter(candidates)),
        "arms": arms,
    }


def _first_batch_identity(train: Any, arm: str) -> dict[str, Any]:
    batch = train.deterministic_training_batch(
        step=0,
        seed=TRAINING_SEED,
        arm=arm,
    )
    candidate_mask = np.asarray(
        batch.base.candidate_mask,
        dtype=np.bool_,
    )
    r2_mask = np.asarray(batch.parent.r2_token_mask, dtype=np.bool_)
    relational_mask = np.asarray(
        batch.parent.relational_mask,
        dtype=np.bool_,
    )
    derivatives = np.asarray(batch.derivative_features)
    return {
        "scientific_batch_blake3": scientific_batch_blake3(batch),
        "groups": int(candidate_mask.shape[0]),
        "candidates": int(candidate_mask.sum()),
        "r2_tokens": int(r2_mask.sum()),
        "relational_tokens": int(relational_mask.sum()),
        "derivative_nonzero_values": int(
            np.count_nonzero(derivatives[candidate_mask])
        ),
        "derivative_width": int(derivatives.shape[-1]),
    }


def _parent_surface_verified(
    arm: str,
    first_batch: dict[str, Any],
) -> bool:
    if arm == CONTROL_ARM:
        return (
            first_batch["r2_tokens"] > 0
            and first_batch["relational_tokens"] == 0
        )
    return (
        first_batch["r2_tokens"] == 0
        and first_batch["relational_tokens"] > 0
    )


def _derivative_surface_verified(
    arm: str,
    first_batch: dict[str, Any],
) -> bool:
    if first_batch["derivative_width"] != 154:
        return False
    if arm == S5_ARM:
        return first_batch["derivative_nonzero_values"] > 0
    return first_batch["derivative_nonzero_values"] == 0


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


def _normalize_host(host: str) -> str:
    lowered = host.lower()
    for known in HOSTS:
        if known in lowered:
            return known
    if host == "Johns-Mac-mini":
        return "john1"
    return host.removesuffix(".local")


def _slug(value: str) -> str:
    return value.replace("-", "_")


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


def _common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument(
        "--relational-cache",
        type=Path,
        required=True,
    )
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--r6-binary", type=Path, required=True)
    parser.add_argument("--smoke-proof", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    _common_data_arguments(authorize)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--approved-unix-ms", type=int)
    authorize.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    _common_data_arguments(preflight)
    preflight.add_argument("--host", choices=HOSTS, required=True)
    preflight.add_argument("--arm", choices=ARMS, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    task_spec = subparsers.add_parser("task-spec")
    task_spec.add_argument(
        "--relational-cache-relative",
        type=Path,
        required=True,
    )
    task_spec.add_argument("--output", type=Path, required=True)
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
            smoke_proof=args.smoke_proof,
            approved_by=args.approved_by,
            approved_unix_ms=args.approved_unix_ms,
        )
    elif args.command == "preflight":
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
            authorization_path=args.authorization,
            smoke_proof=args.smoke_proof,
        )
    else:
        result = build_task_specification(
            relational_cache_relative=args.relational_cache_relative
        )
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
