#!/usr/bin/env python3
"""Authorize and describe the inert four-host ADR 0153 MLX campaign."""

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
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from cascadia_mlx.s4_candidate_context_cache import S4CandidateContextCache
from cascadia_mlx.s4_candidate_set_mlx_data import S4CandidateSetDataset
from cascadia_mlx.s4_candidate_set_mlx_model import S4_ARMS
from cascadia_mlx.s4_candidate_set_mlx_train import (
    ADR_ID,
    ARM_HOSTS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    S4CandidateSetTrainingProtocol,
    _initial_prediction_parity,
    _runtime_identity,
    _warm_start_identity,
    cross_arm_initialization,
)
from cluster_research_queue import add_task, empty_queue, validate_queue
from rust_experiment_bundle import BundleError, file_blake3, validate_bundle
from s4_candidate_set_mlx_report import (
    validate_r3_rescue_evidence,
)
from s4_candidate_set_mlx_smoke_compare import PASS as SMOKE_PASS

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
HOST_ALIASES = {"Johns-Mac-mini": "john1"}
TASK_PREFIX = "s4ctxmlx"
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
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
    "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache/"
    "2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
DEFAULT_CONTEXT_CACHE = Path(
    "artifacts/experiments/s4-candidate-context-cache-v1/cache/"
    "fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def"
)
DEFAULT_CONTROL = DEFAULT_EXPERIMENT_ROOT / "control"
DEFAULT_AUTHORIZATION = DEFAULT_CONTROL / "authorization.json"
DEFAULT_SMOKE_PROOF = DEFAULT_CONTROL / "cross-host-smoke-proof.json"
DEFAULT_R3_CLASSIFICATION = DEFAULT_CONTROL / "r3-classification.json"
DEFAULT_R3_CONTROL = DEFAULT_CONTROL / "r3-control-report.json"
DEFAULT_R3_SUBSTRATE = DEFAULT_CONTROL / "r3-radius1-report.json"
DEFAULT_WARM_START = DEFAULT_CONTROL / "r3-radius1-checkpoint"
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "docs/v2/decisions/0153-s4-candidate-context-mlx-rescue.md",
    "docs/v2/reports/s4-candidate-context-mlx-comparison-v1-preregistration.md",
    "tools/cluster_artifact_collect.py",
    "tools/cluster_artifact_fanout.py",
    "tools/cluster_research_queue.py",
    "tools/rust_experiment_bundle.py",
    "tools/s4_candidate_set_mlx_campaign.py",
    "tools/s4_candidate_set_mlx_report.py",
    "tools/s4_candidate_set_mlx_smoke_compare.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "python/cascadia_mlx/",
    "apps/web/src/",
    "crates/cascadia-game/",
    "crates/cascadia-sim/",
    "crates/cascadia-data/",
    "crates/cascadia-model/",
    "crates/cascadia-eval/",
    "crates/cascadia-search/",
    "crates/cascadia-api/",
    "crates/cascadia-cli-v2/",
    "crates/cascadia-differential/",
    "crates/cascadia-provenance/",
)


class CampaignError(RuntimeError):
    """The S4 campaign cannot proceed without changing scientific identity."""


def validate_bundle_for_campaign(bundle: Path) -> dict[str, Any]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    identity = manifest.get("identity", {})
    if identity.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("S4 immutable bundle names the wrong experiment")
    entries = identity.get("source_files", [])
    paths = {
        entry.get("path")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing_files = sorted(REQUIRED_SOURCE_FILES - paths)
    missing_prefixes = sorted(
        prefix
        for prefix in REQUIRED_SOURCE_PREFIXES
        if not any(path.startswith(prefix) for path in paths)
    )
    if missing_files or missing_prefixes or identity.get("binaries") != []:
        raise CampaignError(
            "S4 immutable source bundle is incomplete: "
            f"missing_files={missing_files}, "
            f"missing_prefixes={missing_prefixes}, "
            f"unexpected_binaries={identity.get('binaries')}"
        )
    return manifest


def create_authorization(
    *,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    context_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    warm_start_checkpoint: Path,
    r3_classification: Path,
    r3_control: Path,
    r3_substrate: Path,
    smoke_proof: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Create explicit production authorization without starting training."""
    if not approved_by.strip():
        raise CampaignError("S4 authorization requires a nonempty approver")
    bundle = bundle.resolve()
    manifest = validate_bundle_for_campaign(bundle)
    sidecar, exact_supply, context, train, validation = _bind_open_data(
        cache=cache,
        s1_cache=s1_cache,
        context_cache=context_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    source = source_provenance(bundle / "source")
    warm_start = _warm_start_identity(
        warm_start_checkpoint,
        require_production=True,
    )
    r3 = validate_r3_rescue_evidence(
        classification_path=r3_classification,
        control_path=r3_control,
        substrate_path=r3_substrate,
    )
    _validate_warm_start_binding(warm_start, r3)
    smoke = validate_smoke_proof(
        smoke_proof,
        cache_id=sidecar.cache_id,
        s1_cache_id=exact_supply.cache_id,
        context_cache_id=context.cache_id,
        warm_start=warm_start,
    )
    open_data = open_data_verification_identity(
        cache=sidecar,
        s1_cache=exact_supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    initialization = cross_arm_initialization(warm_start_checkpoint)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "bundle_id": manifest["bundle_id"],
        "source_blake3": source["v2_source_blake3"],
        "cache_id": sidecar.cache_id,
        "cache_manifest_blake3": file_blake3(sidecar.root / "cache.json"),
        "s1_cache_id": exact_supply.cache_id,
        "s1_cache_manifest_blake3": file_blake3(
            exact_supply.manifest_path
        ),
        "context_cache_id": context.cache_id,
        "context_cache_manifest_blake3": file_blake3(
            context.root / "cache.json"
        ),
        "train_dataset_id": train.manifest["dataset_id"],
        "train_manifest_blake3": file_blake3(
            train.root / "dataset.json"
        ),
        "validation_dataset_id": validation.manifest["dataset_id"],
        "validation_manifest_blake3": file_blake3(
            validation.root / "dataset.json"
        ),
        "open_data_verification_id": open_data_verification_id(open_data),
        "open_data_verification": open_data,
        "protocol": S4CandidateSetTrainingProtocol().to_dict(),
        "authorized_arms": list(S4_ARMS),
        "arm_hosts": ARM_HOSTS,
        "cross_arm_initialization": initialization,
        "warm_start": warm_start,
        "r3_rescue_evidence_id": r3["evidence_id"],
        "r3_rescue_evidence": r3["identity"],
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
    path: Path,
    **inputs: Any,
) -> dict[str, Any]:
    authorization = _read_json(path, "S4 authorization")
    identity = authorization.get("identity")
    approved_unix_ms = (
        identity.get("approved_unix_ms")
        if isinstance(identity, dict)
        else None
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("protocol_id") != PROTOCOL_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(identity, dict)
        or not isinstance(approved_unix_ms, int)
        or isinstance(approved_unix_ms, bool)
        or approved_unix_ms < 0
        or canonical_blake3(identity)
        != authorization.get("authorization_id")
    ):
        raise CampaignError("S4 authorization is malformed")
    expected = create_authorization(
        **inputs,
        approved_by=str(identity.get("approved_by", "")),
        approved_unix_ms=approved_unix_ms,
    )
    if expected != authorization:
        raise CampaignError("S4 authorization is stale for its immutable inputs")
    return authorization


def validate_smoke_proof(
    path: Path,
    *,
    cache_id: str,
    s1_cache_id: str,
    context_cache_id: str,
    warm_start: dict[str, Any],
) -> dict[str, Any]:
    proof = _read_json(path, "S4 cross-host smoke proof")
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
        or identity.get("cache_id") != cache_id
        or identity.get("s1_cache_id") != s1_cache_id
        or identity.get("context_cache_id") != context_cache_id
        or identity.get("warm_start") != warm_start
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or proof.get("claims", {}).get("production_training_started")
        is not False
    ):
        raise CampaignError("S4 cross-host smoke proof is invalid")
    return proof


def run_preflight(
    *,
    host: str,
    arm: str,
    repository: Path,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    context_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    warm_start_checkpoint: Path,
    r3_classification: Path,
    r3_control: Path,
    r3_substrate: Path,
    authorization: Path,
    smoke_proof: Path,
) -> dict[str, Any]:
    """Verify one assigned host without starting an optimizer."""
    if host not in HOSTS or arm not in S4_ARMS or ARM_HOSTS[arm] != host:
        raise CampaignError("S4 preflight host/arm assignment is invalid")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("S4 preflight must run from immutable bundle source")
    manifest = validate_bundle_for_campaign(bundle)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        cache=cache,
        s1_cache=s1_cache,
        context_cache=context_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        warm_start_checkpoint=warm_start_checkpoint,
        r3_classification=r3_classification,
        r3_control=r3_control,
        r3_substrate=r3_substrate,
        smoke_proof=smoke_proof,
    )
    sidecar, exact_supply, context, _train, validation = _bind_open_data(
        cache=cache,
        s1_cache=s1_cache,
        context_cache=context_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    warm_start = _warm_start_identity(
        warm_start_checkpoint,
        require_production=True,
    )
    r3 = validate_r3_rescue_evidence(
        classification_path=r3_classification,
        control_path=r3_control,
        substrate_path=r3_substrate,
    )
    _validate_warm_start_binding(warm_start, r3)
    smoke = validate_smoke_proof(
        smoke_proof,
        cache_id=sidecar.cache_id,
        s1_cache_id=exact_supply.cache_id,
        context_cache_id=context.cache_id,
        warm_start=warm_start,
    )
    source = source_provenance(repository)
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    runtime = _runtime_identity()
    actual_host = _normalize_host(socket.gethostname().split(".")[0])
    initialization = cross_arm_initialization(warm_start_checkpoint)
    parity = _initial_prediction_parity(
        validation,
        warm_start_checkpoint=warm_start_checkpoint,
        arm=arm,
    )
    open_data = open_data_verification_identity(
        cache=sidecar,
        s1_cache=exact_supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    checks = {
        "immutable_bundle_verified": True,
        "authorization_verified": True,
        "cache_verified": True,
        "s1_cache_verified": True,
        "context_cache_verified": True,
        "dataset_manifests_verified": True,
        "apple_silicon_verified": (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ),
        "mlx_gpu_verified": "gpu" in str(mx.default_device()).lower(),
        "python_bytecode_disabled": sys.dont_write_bytecode,
        "host_assignment_verified": (
            actual_host == host == runtime["host"]
        ),
        "source_identity_verified": (
            source["v2_source_blake3"]
            == approval["identity"]["source_blake3"]
        ),
        "open_data_verification_identity_verified": (
            open_data == approval["identity"]["open_data_verification"]
            and open_data_verification_id(open_data)
            == approval["identity"]["open_data_verification_id"]
        ),
        "initialization_parity_verified": (
            initialization
            == approval["identity"]["cross_arm_initialization"]
        ),
        "prediction_parity_verified": (
            parity["scores_byte_identical"] is True
            and parity["standard_errors_byte_identical"] is True
        ),
        "r3_rescue_evidence_verified": (
            r3["evidence_id"]
            == approval["identity"]["r3_rescue_evidence_id"]
        ),
        "smoke_replay_verified": (
            smoke["proof_id"] == approval["identity"]["smoke_proof_id"]
        ),
        "open_data_only_verified": all(
            sidecar.manifest["hidden_information"][field] is expected
            for field, expected in (
                ("open_train_and_validation_only", True),
                ("hidden_order_exported", False),
                ("excluded_tile_identity_exported", False),
                ("future_refill_exported", False),
                ("sealed_test_opened", False),
                ("gameplay_opened", False),
            )
        ),
        "production_training_started": False,
    }
    if not all(
        value
        for key, value in checks.items()
        if key != "production_training_started"
    ):
        raise CampaignError(f"S4 host preflight failed: {checks}")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "bundle_id": manifest["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "cache_id": sidecar.cache_id,
        "s1_cache_id": exact_supply.cache_id,
        "context_cache_id": context.cache_id,
        "warm_start": warm_start,
        "r3_rescue_evidence_id": r3["evidence_id"],
        "arm": arm,
        "host": host,
        "runtime": runtime,
        "source_blake3": source["v2_source_blake3"],
        "open_data_verification_id": open_data_verification_id(open_data),
        "mlx_gpu_verified": checks["mlx_gpu_verified"],
        "context_cache_verified": checks["context_cache_verified"],
        "initialization_parity_verified": (
            checks["initialization_parity_verified"]
        ),
        "prediction_parity_verified": checks["prediction_parity_verified"],
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
        "claims": {
            "preflight_complete": True,
            "production_training_started": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
        },
    }


def build_task_specs(
    *,
    repository: Path,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    context_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    control: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> list[dict[str, Any]]:
    """Build the generated-only four-arm concurrent execution graph."""
    repository = repository.resolve()
    bundle_relative = _relative(repository, bundle, "bundle")
    cache_relative = _relative(repository, cache, "R3 cache")
    s1_relative = _relative(repository, s1_cache, "S1 cache")
    context_relative = _relative(
        repository,
        context_cache,
        "S4 context cache",
    )
    train_relative = _relative(repository, train_dataset, "train dataset")
    validation_relative = _relative(
        repository,
        validation_dataset,
        "validation dataset",
    )
    control_relative = _relative(repository, control, "control")
    experiment_relative = _relative(
        repository,
        experiment_root,
        "experiment root",
    )
    authorization_relative = control_relative / DEFAULT_AUTHORIZATION.name
    smoke_relative = control_relative / DEFAULT_SMOKE_PROOF.name
    warm_relative = control_relative / DEFAULT_WARM_START.name
    r3_classification_relative = (
        control_relative / DEFAULT_R3_CLASSIFICATION.name
    )
    r3_control_relative = control_relative / DEFAULT_R3_CONTROL.name
    r3_substrate_relative = control_relative / DEFAULT_R3_SUBSTRATE.name
    specs: list[dict[str, Any]] = []

    fanout_inputs = (
        ("bundle", bundle_relative, ("bundle.json",), []),
        (
            "r3-cache",
            cache_relative,
            ("cache.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "s1-cache",
            s1_relative,
            ("cache.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "context-cache",
            context_relative,
            ("cache.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "train",
            train_relative,
            ("dataset.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "validation",
            validation_relative,
            ("dataset.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "control",
            control_relative,
            (
                authorization_relative.name,
                smoke_relative.name,
                r3_classification_relative.name,
                r3_control_relative.name,
                r3_substrate_relative.name,
                f"{warm_relative.name}/checkpoint.json",
                f"{warm_relative.name}/state.json",
                f"{warm_relative.name}/model.safetensors",
            ),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
    )
    fanout_ids: list[str] = []
    for name, relative, required, dependencies in fanout_inputs:
        task_id = f"{TASK_PREFIX}-fanout-{name}"
        fanout_ids.append(task_id)
        command = [
            ".venv/bin/python",
            "-B",
            "tools/cluster_artifact_fanout.py",
            "--source",
            f"{relative}/",
            "--local-root",
            str(relative),
        ]
        for remote_host in HOSTS[1:]:
            command.extend(
                [
                    "--destination",
                    f"{remote_host}:{REMOTE_ROOTS[remote_host] / relative}/",
                ]
            )
        for required_file in required:
            command.extend(["--required-file", required_file])
        command.extend(
            [
                "--verify-tree",
                "--output",
                str(experiment_relative / f"reports/fanout-{name}.json"),
            ]
        )
        specs.append(
            _task(
                task_id=task_id,
                title=f"Fan out frozen S4 {name}",
                decision=f"Make every {name} byte identical on all hosts",
                workload_class="shared-prerequisite",
                priority=1,
                expected_runtime_seconds=900,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=list(dependencies),
                command=command,
                artifact_path=str(
                    experiment_relative / f"reports/fanout-{name}.json"
                ),
                stop_rule=f"Every {name} byte must match before preflight.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=False,
            )
        )

    preflight_ids: list[str] = []
    for arm, host in ARM_HOSTS.items():
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(task_id)
        output = experiment_relative / f"reports/preflight-{host}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Preflight S4 {arm} on {host}",
                decision=(
                    "Verify source, caches, failed-substrate evidence, "
                    "warm start, smoke proof, and MLX GPU"
                ),
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=900,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=fanout_ids,
                command=[
                    *_frozen_campaign_command(host, bundle_relative),
                    "preflight",
                    "--host",
                    host,
                    "--arm",
                    arm,
                    "--repository",
                    str(REMOTE_ROOTS[host] / bundle_relative / "source"),
                    "--bundle",
                    str(REMOTE_ROOTS[host] / bundle_relative),
                    "--cache",
                    str(REMOTE_ROOTS[host] / cache_relative),
                    "--s1-cache",
                    str(REMOTE_ROOTS[host] / s1_relative),
                    "--context-cache",
                    str(REMOTE_ROOTS[host] / context_relative),
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--warm-start-checkpoint",
                    str(REMOTE_ROOTS[host] / warm_relative),
                    "--r3-classification",
                    str(REMOTE_ROOTS[host] / r3_classification_relative),
                    "--r3-control",
                    str(REMOTE_ROOTS[host] / r3_control_relative),
                    "--r3-substrate",
                    str(REMOTE_ROOTS[host] / r3_substrate_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--smoke-proof",
                    str(REMOTE_ROOTS[host] / smoke_relative),
                    "--output",
                    str(REMOTE_ROOTS[host] / output),
                ],
                artifact_path=str(output),
                stop_rule="No optimizer starts unless every preflight is true.",
                cpu_cores=1,
                memory_gib=10.0,
                uses_mlx=True,
            )
        )

    arm_task_ids: list[str] = []
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        task_id = f"{TASK_PREFIX}-train-{slug}"
        arm_task_ids.append(task_id)
        report = experiment_relative / f"reports/{slug}.json"
        run_dir = experiment_relative / f"runs/{slug}"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Train S4 {arm}",
                decision=(
                    "Measure one frozen context treatment on the failed "
                    "radius-one substrate"
                ),
                workload_class="independent-experiment",
                priority=10,
                expected_runtime_seconds=10_800,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=preflight_ids,
                command=[
                    *_frozen_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.s4_candidate_set_mlx_train",
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--cache",
                    str(REMOTE_ROOTS[host] / cache_relative),
                    "--s1-cache",
                    str(REMOTE_ROOTS[host] / s1_relative),
                    "--context-cache",
                    str(REMOTE_ROOTS[host] / context_relative),
                    "--warm-start-checkpoint",
                    str(REMOTE_ROOTS[host] / warm_relative),
                    "--run-dir",
                    str(REMOTE_ROOTS[host] / run_dir),
                    "--output",
                    str(REMOTE_ROOTS[host] / report),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--preflight",
                    str(
                        REMOTE_ROOTS[host]
                        / experiment_relative
                        / f"reports/preflight-{host}.json"
                    ),
                    "--arm",
                    arm,
                ],
                artifact_path=str(report),
                stop_rule=(
                    "Finish exactly 3,000 steps, full validation, and an "
                    "isolated serving benchmark."
                ),
                cpu_cores=10,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    collected = experiment_relative / "reports/collected"
    collection_report = experiment_relative / "reports/collection.json"
    collect_command = [
        ".venv/bin/python",
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        remote_report = (
            REMOTE_ROOTS[host]
            / experiment_relative
            / f"reports/{slug}.json"
        )
        collect_command.extend(
            [
                "--artifact",
                f"{host}:{remote_report}",
                str(collected / f"{slug}.json"),
            ]
        )
    collect_command.extend(["--output", str(collection_report)])
    collection_id = f"{TASK_PREFIX}-collect"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect four S4 arm reports",
            decision="Checksum-copy one production report from each host",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=arm_task_ids,
            command=collect_command,
            artifact_path=str(collection_report),
            stop_rule="All four reports must match their producing hosts.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classifier_ids: list[str] = []
    for order in ("forward", "reverse"):
        task_id = f"{TASK_PREFIX}-classify-{order}"
        classifier_ids.append(task_id)
        ordered = S4_ARMS if order == "forward" else tuple(reversed(S4_ARMS))
        output = (
            experiment_relative / f"reports/classification-{order}.json"
        )
        command = [
            *_frozen_report_command("john1", bundle_relative),
            "classify",
        ]
        for arm in ordered:
            command.extend(
                [
                    "--report",
                    str(
                        REMOTE_ROOTS["john1"]
                        / collected
                        / f"{_slug(arm)}.json"
                    ),
                ]
            )
        command.extend(
            [
                "--r3-classification",
                str(REMOTE_ROOTS["john1"] / r3_classification_relative),
                "--r3-control",
                str(REMOTE_ROOTS["john1"] / r3_control_relative),
                "--r3-substrate",
                str(REMOTE_ROOTS["john1"] / r3_substrate_relative),
                "--output",
                str(REMOTE_ROOTS["john1"] / output),
            ]
        )
        specs.append(
            _task(
                task_id=task_id,
                title=f"Classify S4 in {order} order",
                decision=(
                    "Apply context-effect, full-R2 rescue, and serving gates"
                ),
                workload_class="replica",
                priority=40,
                expected_runtime_seconds=60,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=command,
                artifact_path=str(output),
                stop_rule="Emit one deterministic offline classification.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )

    proof = (
        experiment_relative / "reports/classification-order-proof.json"
    )
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-classification-order-proof",
            title="Prove S4 classifier order invariance",
            decision="Require byte-identical forward and reverse results",
            workload_class="replica",
            priority=50,
            expected_runtime_seconds=30,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=classifier_ids,
            command=[
                *_frozen_report_command("john1", bundle_relative),
                "compare",
                "--forward",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/classification-forward.json"
                ),
                "--reverse",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/classification-reverse.json"
                ),
                "--output",
                str(REMOTE_ROOTS["john1"] / proof),
            ],
            artifact_path=str(proof),
            stop_rule="Classification bytes must not depend on report order.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    _validate_task_specs(specs)
    return specs


def queue_specification(specs: list[dict[str, Any]]) -> dict[str, Any]:
    state = empty_queue(EXPERIMENT_ID, now_ms=0)
    for index, specification in enumerate(specs, start=1):
        add_task(state, specification, now_ms=index)
    validate_queue(state)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "task_count": len(specs),
        "task_spec_blake3": canonical_blake3(specs),
        "applied": False,
        "installation_supported_by_this_tool": False,
        "live_queue_path": None,
        "tasks": specs,
        "validated_queue_preview": state,
    }


def _bind_open_data(
    *,
    cache: Path,
    s1_cache: Path,
    context_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> tuple[
    R3ActionEditMlxCache,
    S1ExactSupplyCache,
    S4CandidateContextCache,
    S4CandidateSetDataset,
    S4CandidateSetDataset,
]:
    sidecar = R3ActionEditMlxCache(cache, require_complete=True)
    exact_supply = S1ExactSupplyCache(s1_cache)
    context = S4CandidateContextCache(context_cache)
    train_r3 = sidecar.bind_dataset(
        train_dataset,
        s1_cache=exact_supply,
    )
    validation_r3 = sidecar.bind_dataset(
        validation_dataset,
        s1_cache=exact_supply,
    )
    train = S4CandidateSetDataset(train_r3, context_cache=context)
    validation = S4CandidateSetDataset(
        validation_r3,
        context_cache=context,
    )
    if (
        train.group_count != 560
        or train.candidate_count != 280_012
        or validation.group_count != 240
        or validation.candidate_count != 860_203
        or context.manifest["scientific_identity"].get("r3_cache_id")
        != sidecar.cache_id
    ):
        raise CampaignError("S4 complete open-corpus coverage drifted")
    return sidecar, exact_supply, context, train, validation


def _validate_warm_start_binding(
    warm_start: dict[str, Any],
    r3: dict[str, Any],
) -> None:
    substrate = r3["substrate"]
    if (
        warm_start.get("global_step") != 3000
        or warm_start.get("model_blake3")
        != substrate["checkpoint"].get("model_blake3")
        or warm_start.get("manifest_blake3")
        != substrate["checkpoint"].get("manifest_blake3")
        or warm_start.get("model_config", {}).get("arm")
        != "t3-r3-radius1-global"
    ):
        raise CampaignError(
            "S4 warm start is not the failed R3 radius-one checkpoint"
        )


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("S4 queue graph contains duplicate task IDs")
    known = set(identifiers)
    preflight_ids = {
        f"{TASK_PREFIX}-preflight-{host}" for host in HOSTS
    }
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(
                f"S4 task {spec['id']} has unknown dependencies: "
                f"{sorted(unknown)}"
            )
        if (
            any("python" in item for item in spec["command"])
            and "-B" not in spec["command"]
        ):
            raise CampaignError(
                f"frozen Python task omits -B: {spec['id']}"
            )
    for arm, host in ARM_HOSTS.items():
        task = next(
            spec
            for spec in specs
            if spec["id"] == f"{TASK_PREFIX}-train-{_slug(arm)}"
        )
        if task["compatible_hosts"] != [host]:
            raise CampaignError(f"S4 arm {arm} is not pinned to {host}")
        if set(task["dependencies"]) != preflight_ids:
            raise CampaignError(
                f"S4 arm {arm} can launch before every host preflight"
            )


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: float,
    critical_path: bool,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: str,
    stop_rule: str,
    cpu_cores: int,
    memory_gib: float,
    uses_mlx: bool,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": critical_path,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": artifact_path,
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": uses_mlx,
        },
    }


def _frozen_python_prefix(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    return [
        "/usr/bin/env",
        "-C",
        str(root / bundle_relative / "source"),
        "PYTHONPATH=python:tools",
        "PYTHONDONTWRITEBYTECODE=1",
        str(root / ".venv/bin/python"),
        "-B",
    ]


def _frozen_campaign_command(
    host: str,
    bundle_relative: Path,
) -> list[str]:
    return [
        *_frozen_python_prefix(host, bundle_relative),
        "tools/s4_candidate_set_mlx_campaign.py",
    ]


def _frozen_report_command(
    host: str,
    bundle_relative: Path,
) -> list[str]:
    return [
        *_frozen_python_prefix(host, bundle_relative),
        "tools/s4_candidate_set_mlx_report.py",
    ]


def _relative(repository: Path, path: Path, label: str) -> Path:
    try:
        relative = path.resolve().relative_to(repository)
    except ValueError as error:
        raise CampaignError(
            f"S4 {label} must remain beneath the repository"
        ) from error
    if not relative.parts:
        raise CampaignError(f"S4 {label} cannot be the repository root")
    return relative


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _normalize_host(value: str) -> str:
    value = value.removesuffix(".local")
    return HOST_ALIASES.get(value, value)


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _write_once(path: Path, value: object, label: str) -> None:
    if path.exists():
        if _read_json(path, label) != value:
            raise CampaignError(
                f"existing {label} differs from requested identity"
            )
        return
    _write_json_atomic(path, value)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_R3_CACHE)
    parser.add_argument("--s1-cache", type=Path, default=DEFAULT_S1_CACHE)
    parser.add_argument(
        "--context-cache",
        type=Path,
        default=DEFAULT_CONTEXT_CACHE,
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=DEFAULT_WARM_START,
    )
    parser.add_argument(
        "--r3-classification",
        type=Path,
        default=DEFAULT_R3_CLASSIFICATION,
    )
    parser.add_argument(
        "--r3-control",
        type=Path,
        default=DEFAULT_R3_CONTROL,
    )
    parser.add_argument(
        "--r3-substrate",
        type=Path,
        default=DEFAULT_R3_SUBSTRATE,
    )
    parser.add_argument(
        "--smoke-proof",
        type=Path,
        default=DEFAULT_SMOKE_PROOF,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    _common_inputs(authorize)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AUTHORIZATION,
    )

    preflight = subparsers.add_parser("preflight")
    _common_inputs(preflight)
    preflight.add_argument("--host", choices=HOSTS, required=True)
    preflight.add_argument("--arm", choices=S4_ARMS, required=True)
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument(
        "--authorization",
        type=Path,
        required=True,
    )
    preflight.add_argument("--output", type=Path, required=True)

    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--cache", type=Path, default=DEFAULT_R3_CACHE)
    queue.add_argument("--s1-cache", type=Path, default=DEFAULT_S1_CACHE)
    queue.add_argument(
        "--context-cache",
        type=Path,
        default=DEFAULT_CONTEXT_CACHE,
    )
    queue.add_argument(
        "--train-dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )
    queue.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    queue.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    queue.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    queue.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "authorize":
            report = create_authorization(
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                context_cache=args.context_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                warm_start_checkpoint=args.warm_start_checkpoint,
                r3_classification=args.r3_classification,
                r3_control=args.r3_control,
                r3_substrate=args.r3_substrate,
                smoke_proof=args.smoke_proof,
                approved_by=args.approved_by,
            )
            _write_once(args.output, report, "S4 authorization")
        elif args.command == "preflight":
            report = run_preflight(
                host=args.host,
                arm=args.arm,
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                context_cache=args.context_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                warm_start_checkpoint=args.warm_start_checkpoint,
                r3_classification=args.r3_classification,
                r3_control=args.r3_control,
                r3_substrate=args.r3_substrate,
                authorization=args.authorization,
                smoke_proof=args.smoke_proof,
            )
            _write_json_atomic(args.output, report)
        else:
            validate_bundle_for_campaign(args.bundle)
            control = args.control
            validate_authorization(
                control / DEFAULT_AUTHORIZATION.name,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                context_cache=args.context_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                warm_start_checkpoint=(
                    control / DEFAULT_WARM_START.name
                ),
                r3_classification=(
                    control / DEFAULT_R3_CLASSIFICATION.name
                ),
                r3_control=control / DEFAULT_R3_CONTROL.name,
                r3_substrate=control / DEFAULT_R3_SUBSTRATE.name,
                smoke_proof=control / DEFAULT_SMOKE_PROOF.name,
            )
            specs = build_task_specs(
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                context_cache=args.context_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                control=control,
                experiment_root=args.experiment_root,
            )
            report = queue_specification(specs)
            _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (CampaignError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
