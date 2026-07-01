#!/usr/bin/env python3
"""Authorize and describe the inert four-host ADR 0150 MLX campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ARMS,
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_train import (
    ADR_ID,
    ARM_HOSTS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    R3ActionEditTrainingProtocol,
    cross_arm_initialization,
    runtime_identity,
)
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache
from cluster_research_queue import add_task, empty_queue, validate_queue
from rust_experiment_bundle import BundleError, file_blake3, validate_bundle

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
HOST_ALIASES = {"Johns-Mac-mini": "john1"}
EXPORTER_BINARY = "r3-action-edit-mlx-exporter"
TASK_PREFIX = "r3aemlx"
SMOKE_PASS = "r3_action_edit_mlx_cross_host_smoke_pass"
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_TRAIN_DATASET = Path("artifacts/datasets/complete-action-graded-oracle-v1-train")
DEFAULT_VALIDATION_DATASET = Path("artifacts/datasets/complete-action-graded-oracle-v1-validation")
DEFAULT_S1_CACHE = Path(
    "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache/"
    "2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
DEFAULT_AUTHORIZATION = DEFAULT_EXPERIMENT_ROOT / "control/authorization.json"
DEFAULT_SMOKE_PROOF = DEFAULT_EXPERIMENT_ROOT / "control/cross-host-smoke-proof.json"
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "docs/v2/decisions/0150-r3-action-edit-mlx-matched-comparison.md",
    "docs/v2/reports/r3-action-edit-mlx-comparison-v1-preregistration.md",
    "docs/v2/reports/r3-action-edit-mlx-cross-host-smoke-amendment-2026-06-17.md",
    "docs/v2/reports/r3-action-edit-mlx-serving-rss-amendment-2026-06-17.md",
    "tools/cluster_artifact_collect.py",
    "tools/cluster_artifact_fanout.py",
    "tools/cluster_research_queue.py",
    "tools/r3_action_edit_mlx_campaign.py",
    "tools/r3_action_edit_mlx_report.py",
    "tools/r3_action_edit_mlx_smoke_compare.py",
    "tools/rust_experiment_bundle.py",
    "python/cascadia_mlx/r3_action_edit_mlx_benchmark.py",
    "tools/r3_action_edit_mlx_exporter/Cargo.lock",
    "tools/r3_action_edit_mlx_exporter/Cargo.toml",
    "tools/r3_action_edit_mlx_exporter/README.md",
}
REQUIRED_SOURCE_PREFIXES = (
    "python/cascadia_mlx/",
    "apps/web/src/",
    "legacy/crates/cascadia-core/",
    "legacy/crates/cascadia-ai/",
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
    "tools/r2_sparse_entity_census/src/",
    "tools/r3_action_edit_census/src/",
    "tools/r3_action_edit_mlx_exporter/src/",
)


class CampaignError(RuntimeError):
    """The R3 MLX campaign cannot proceed without changing scientific identity."""


def validate_bundle_for_campaign(bundle: Path) -> dict[str, Any]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    if manifest.get("identity", {}).get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("R3 MLX immutable bundle names the wrong experiment")
    entries = manifest.get("identity", {}).get("source_files", [])
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
    binaries = {
        entry.get("name"): entry
        for entry in manifest.get("identity", {}).get("binaries", [])
        if isinstance(entry, dict)
    }
    executable = bundle / "bin" / EXPORTER_BINARY
    if (
        missing_files
        or missing_prefixes
        or EXPORTER_BINARY not in binaries
        or file_blake3(executable) != binaries.get(EXPORTER_BINARY, {}).get("blake3")
    ):
        raise CampaignError(
            "R3 MLX immutable bundle is incomplete: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}, "
            f"missing_exporter={EXPORTER_BINARY not in binaries}"
        )
    return manifest


def create_authorization(
    *,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    smoke_proof: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Create an explicit production authorization without starting training."""
    if not approved_by.strip():
        raise CampaignError("R3 MLX authorization requires a nonempty approver")
    bundle = bundle.resolve()
    manifest = validate_bundle_for_campaign(bundle)
    sidecar, exact_supply, train, validation = _bind_open_data(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    source = source_provenance(bundle / "source")
    exporter_blake3 = file_blake3(bundle / "bin" / EXPORTER_BINARY)
    _validate_cache_exporter_binding(sidecar, exporter_blake3)
    initialization = cross_arm_initialization()
    smoke = validate_smoke_proof(smoke_proof, s1_cache_id=exact_supply.cache_id)
    open_data = open_data_verification_identity(
        cache=sidecar,
        s1_cache=exact_supply,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "bundle_id": manifest["bundle_id"],
        "source_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": exporter_blake3,
        "cache_id": sidecar.cache_id,
        "cache_manifest_blake3": file_blake3(sidecar.root / "cache.json"),
        "s1_cache_id": exact_supply.cache_id,
        "s1_cache_manifest_blake3": file_blake3(exact_supply.manifest_path),
        "train_dataset_id": train.base.manifest["dataset_id"],
        "train_manifest_blake3": file_blake3(train.base.root / "dataset.json"),
        "validation_dataset_id": validation.base.manifest["dataset_id"],
        "validation_manifest_blake3": file_blake3(validation.base.root / "dataset.json"),
        "open_data_verification_id": open_data_verification_id(open_data),
        "open_data_verification": open_data,
        "protocol": R3ActionEditTrainingProtocol().to_dict(),
        "authorized_arms": list(ARMS),
        "arm_hosts": ARM_HOSTS,
        "cross_arm_initialization": initialization,
        "smoke_proof_id": smoke["proof_id"],
        "smoke_cache_id": smoke["scientific_identity"]["cache_id"],
        "approved_by": approved_by.strip(),
        "approved_unix_ms": (
            time.time_ns() // 1_000_000 if approved_unix_ms is None else approved_unix_ms
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
    *,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    smoke_proof: Path,
) -> dict[str, Any]:
    authorization = _read_json(path, "R3 MLX authorization")
    identity = authorization.get("identity")
    approved_unix_ms = identity.get("approved_unix_ms") if isinstance(identity, dict) else None
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
        or canonical_blake3(identity) != authorization.get("authorization_id")
    ):
        raise CampaignError("R3 MLX authorization is malformed")
    expected = create_authorization(
        bundle=bundle,
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        smoke_proof=smoke_proof,
        approved_by=str(identity.get("approved_by", "")),
        approved_unix_ms=approved_unix_ms,
    )
    if expected != authorization:
        raise CampaignError("R3 MLX authorization is stale for its immutable inputs")
    return authorization


def validate_smoke_proof(path: Path, *, s1_cache_id: str) -> dict[str, Any]:
    proof = _read_json(path, "R3 MLX cross-host smoke proof")
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
        or identity.get("s1_cache_id") != s1_cache_id
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or proof.get("claims", {}).get("production_training_started") is not False
    ):
        raise CampaignError("R3 MLX cross-host smoke proof is invalid")
    return proof


def export_cache(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    s1_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    output_root: Path,
    receipt: Path,
    maximum_groups_per_split: int | None = None,
) -> dict[str, Any]:
    """Run the immutable Rust exporter once on john1."""
    if host != "john1":
        raise CampaignError("the shared R3 MLX cache is exported once on john1")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("R3 MLX cache export must run from immutable bundle source")
    manifest = validate_bundle_for_campaign(bundle)
    exporter = bundle / "bin" / EXPORTER_BINARY
    command = [
        str(exporter),
        "--train-dataset",
        str(Path(train_dataset).resolve()),
        "--validation-dataset",
        str(Path(validation_dataset).resolve()),
        "--output-root",
        str(Path(output_root).resolve()),
        "--receipt",
        str(Path(receipt).resolve()),
    ]
    if maximum_groups_per_split is not None:
        if maximum_groups_per_split <= 0:
            raise CampaignError("R3 MLX cache smoke bound must be positive")
        command.extend(["--max-groups-per-split", str(maximum_groups_per_split)])
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CampaignError(
            f"R3 MLX cache export failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    export = _read_json(receipt, "R3 MLX cache export receipt")
    cache_id = export.get("cache_id")
    if not _is_blake3(cache_id):
        raise CampaignError("R3 MLX cache export receipt is malformed")
    cache_root = Path(output_root).resolve() / str(cache_id)
    complete = maximum_groups_per_split is None
    sidecar = R3ActionEditMlxCache(cache_root, require_complete=complete)
    _validate_cache_exporter_binding(sidecar, file_blake3(exporter))
    if bool(export.get("complete_open_corpus")) is not complete:
        raise CampaignError("R3 MLX cache completeness disagrees with the command")
    if complete:
        exact_supply = S1ExactSupplyCache(s1_cache)
        sidecar.bind_dataset(
            train_dataset,
            s1_cache=exact_supply,
        )
        sidecar.bind_dataset(
            validation_dataset,
            s1_cache=exact_supply,
        )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "host": host,
        "bundle_id": manifest["bundle_id"],
        "exporter_executable_blake3": file_blake3(exporter),
        "cache_id": sidecar.cache_id,
        "cache_manifest_blake3": file_blake3(sidecar.root / "cache.json"),
        "complete_open_corpus": complete,
        "train_groups": sidecar.splits["train"].groups,
        "train_candidates": sidecar.splits["train"].retained_candidates,
        "validation_groups": sidecar.splits["validation"].groups,
        "validation_candidates": sidecar.splits["validation"].retained_candidates,
        "hidden_information": sidecar.manifest["hidden_information"],
        "production_training_started": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "export_id": canonical_blake3(identity),
        "scientific_identity": identity,
        "cache_root": str(cache_root),
        "claims": {
            "cache_export_complete": complete,
            "production_training_started": False,
            "gameplay_run": False,
            "sealed_test_opened": False,
        },
    }


def run_preflight(
    *,
    host: str,
    arm: str,
    repository: Path,
    bundle: Path,
    cache: Path,
    s1_cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    authorization: Path,
    smoke_proof: Path,
) -> dict[str, Any]:
    """Verify one assigned host without starting an optimizer."""
    if host not in HOSTS or arm not in ARMS or ARM_HOSTS[arm] != host:
        raise CampaignError("R3 MLX preflight host/arm assignment is invalid")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("R3 MLX preflight must run from immutable bundle source")
    manifest = validate_bundle_for_campaign(bundle)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        smoke_proof=smoke_proof,
    )
    sidecar, exact_supply, _train, _validation = _bind_open_data(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    smoke = validate_smoke_proof(smoke_proof, s1_cache_id=exact_supply.cache_id)
    source = source_provenance(repository)
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    runtime = runtime_identity()
    actual_host = _normalize_host(socket.gethostname().split(".")[0])
    initialization = cross_arm_initialization()
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
        "dataset_manifests_verified": True,
        "apple_silicon_verified": (platform.system() == "Darwin" and platform.machine() == "arm64"),
        "mlx_gpu_verified": "gpu" in str(mx.default_device()).lower(),
        "python_bytecode_disabled": sys.dont_write_bytecode,
        "host_assignment_verified": actual_host == host == runtime["host"],
        "source_identity_verified": (
            source["v2_source_blake3"] == approval["identity"]["source_blake3"]
        ),
        "open_data_verification_identity_verified": (
            open_data == approval["identity"]["open_data_verification"]
            and open_data_verification_id(open_data)
            == approval["identity"]["open_data_verification_id"]
        ),
        "initialization_parity_verified": (
            initialization == approval["identity"]["cross_arm_initialization"]
        ),
        "smoke_replay_verified": smoke["proof_id"] == approval["identity"]["smoke_proof_id"],
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
    if not all(value for key, value in checks.items() if key != "production_training_started"):
        raise CampaignError(f"R3 MLX host preflight failed: {checks}")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "bundle_id": manifest["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "cache_id": sidecar.cache_id,
        "s1_cache_id": exact_supply.cache_id,
        "arm": arm,
        "host": host,
        "runtime": runtime,
        "source_blake3": source["v2_source_blake3"],
        "open_data_verification_id": open_data_verification_id(open_data),
        "mlx_gpu_verified": checks["mlx_gpu_verified"],
        "open_data_only_verified": checks["open_data_only_verified"],
        "initialization_parity_verified": checks["initialization_parity_verified"],
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
    train_dataset: Path,
    validation_dataset: Path,
    authorization: Path,
    smoke_proof: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> list[dict[str, Any]]:
    """Build the generated-only, four-arm concurrent execution graph."""
    repository = repository.resolve()
    bundle_relative = _relative(repository, bundle, "bundle")
    cache_relative = _relative(repository, cache, "cache")
    s1_relative = _relative(repository, s1_cache, "S1 cache")
    train_relative = _relative(repository, train_dataset, "train dataset")
    validation_relative = _relative(repository, validation_dataset, "validation dataset")
    authorization_relative = _relative(repository, authorization, "authorization")
    smoke_relative = _relative(repository, smoke_proof, "smoke proof")
    if authorization_relative.parent != smoke_relative.parent:
        raise CampaignError("R3 MLX authorization and smoke proof must share control/")
    experiment_relative = _relative(repository, experiment_root, "experiment root")
    specs: list[dict[str, Any]] = []

    fanout_inputs = (
        ("bundle", bundle_relative, ("bundle.json",), []),
        ("cache", cache_relative, ("cache.json",), [f"{TASK_PREFIX}-fanout-bundle"]),
        ("s1-cache", s1_relative, ("cache.json",), [f"{TASK_PREFIX}-fanout-bundle"]),
        ("train", train_relative, ("dataset.json",), [f"{TASK_PREFIX}-fanout-bundle"]),
        (
            "validation",
            validation_relative,
            ("dataset.json",),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
        (
            "control",
            authorization_relative.parent,
            (authorization_relative.name, smoke_relative.name),
            [f"{TASK_PREFIX}-fanout-bundle"],
        ),
    )
    fanout_ids = []
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
                title=f"Fan out frozen R3 MLX {name}",
                decision=f"Make every {name} byte identical on all four hosts",
                workload_class="shared-prerequisite",
                priority=1,
                expected_runtime_seconds=900 if name == "cache" else 300,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=dependencies,
                command=command,
                artifact_path=str(experiment_relative / f"reports/fanout-{name}.json"),
                stop_rule=f"Every {name} byte must match before preflight.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=False,
            )
        )

    preflight_ids = []
    for arm, host in ARM_HOSTS.items():
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(task_id)
        output = experiment_relative / f"reports/preflight-{host}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Preflight R3 MLX {arm} on {host}",
                decision="Verify source, caches, open data, MLX GPU, and launch controls",
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=600,
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
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--smoke-proof",
                    str(REMOTE_ROOTS[host] / smoke_relative),
                    "--output",
                    str(REMOTE_ROOTS[host] / output),
                ],
                artifact_path=str(output),
                stop_rule="No optimizer starts unless every preflight check is true.",
                cpu_cores=1,
                memory_gib=10.0,
                uses_mlx=True,
            )
        )

    arm_task_ids = []
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        task_id = f"{TASK_PREFIX}-train-{slug}"
        arm_task_ids.append(task_id)
        report = experiment_relative / f"reports/{slug}.json"
        run_dir = experiment_relative / f"runs/{slug}"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Train R3 MLX {arm}",
                decision="Measure one frozen representation on the matched open corpus",
                workload_class="independent-experiment",
                priority=10,
                expected_runtime_seconds=7_200,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=preflight_ids,
                command=[
                    *_frozen_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.r3_action_edit_mlx_train",
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--cache",
                    str(REMOTE_ROOTS[host] / cache_relative),
                    "--s1-cache",
                    str(REMOTE_ROOTS[host] / s1_relative),
                    "--run-dir",
                    str(REMOTE_ROOTS[host] / run_dir),
                    "--output",
                    str(REMOTE_ROOTS[host] / report),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--preflight",
                    str(
                        REMOTE_ROOTS[host] / experiment_relative / f"reports/preflight-{host}.json"
                    ),
                    "--arm",
                    arm,
                ],
                artifact_path=str(report),
                stop_rule="Finish exactly 3,000 steps and one complete validation pass.",
                cpu_cores=10,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    collected = experiment_relative / "reports/collected"
    collection_report = experiment_relative / "reports/collection.json"
    collect_command = [".venv/bin/python", "-B", "tools/cluster_artifact_collect.py"]
    for arm, host in ARM_HOSTS.items():
        slug = _slug(arm)
        remote_report = REMOTE_ROOTS[host] / experiment_relative / f"reports/{slug}.json"
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
            title="Collect four R3 MLX arm reports",
            decision="Checksum-copy exactly one production report from each host",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=arm_task_ids,
            command=collect_command,
            artifact_path=str(collection_report),
            stop_rule="All four report files must match their producing hosts.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classifier_ids = []
    for order in ("forward", "reverse"):
        task_id = f"{TASK_PREFIX}-classify-{order}"
        classifier_ids.append(task_id)
        ordered = ARMS if order == "forward" else tuple(reversed(ARMS))
        output = experiment_relative / f"reports/classification-{order}.json"
        command = [*_frozen_report_command("john1", bundle_relative), "classify"]
        for arm in ordered:
            command.extend(
                [
                    "--report",
                    str(REMOTE_ROOTS["john1"] / collected / f"{_slug(arm)}.json"),
                ]
            )
        command.extend(["--output", str(REMOTE_ROOTS["john1"] / output)])
        specs.append(
            _task(
                task_id=task_id,
                title=f"Classify R3 MLX in {order} order",
                decision="Apply ADR 0150 gates without gameplay or promotion",
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

    proof = experiment_relative / "reports/classification-order-proof.json"
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-classification-order-proof",
            title="Prove R3 MLX classifier order invariance",
            decision="Require byte-identical forward and reverse classifications",
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
    train_dataset: Path,
    validation_dataset: Path,
) -> tuple[R3ActionEditMlxCache, S1ExactSupplyCache, Any, Any]:
    sidecar = R3ActionEditMlxCache(cache, require_complete=True)
    exact_supply = S1ExactSupplyCache(s1_cache)
    train = sidecar.bind_dataset(train_dataset, s1_cache=exact_supply)
    validation = sidecar.bind_dataset(validation_dataset, s1_cache=exact_supply)
    if (
        train.group_count != 560
        or train.candidate_count <= 0
        or train.candidate_count > 560 * 512
        or sidecar.splits["train"].source_candidates != 2_135_111
        or validation.group_count != 240
        or validation.candidate_count != 860_203
    ):
        raise CampaignError("R3 MLX complete open-corpus coverage drifted")
    return sidecar, exact_supply, train, validation


def _validate_cache_exporter_binding(
    cache: R3ActionEditMlxCache,
    exporter_blake3: str,
) -> None:
    exporter = cache.manifest.get("exporter")
    if (
        not _is_blake3(exporter_blake3)
        or not isinstance(exporter, dict)
        or exporter.get("executable_blake3") != exporter_blake3
    ):
        raise CampaignError("R3 MLX cache was not produced by the immutable exporter")


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("R3 MLX queue graph contains duplicate task IDs")
    known = set(identifiers)
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(
                f"R3 MLX task {spec['id']} has unknown dependencies: {sorted(unknown)}"
            )
        if any("python" in item for item in spec["command"]) and "-B" not in spec["command"]:
            raise CampaignError(f"frozen Python task omits -B: {spec['id']}")
    for arm, host in ARM_HOSTS.items():
        task = next(spec for spec in specs if spec["id"] == f"{TASK_PREFIX}-train-{_slug(arm)}")
        if task["compatible_hosts"] != [host]:
            raise CampaignError(f"R3 MLX arm {arm} is not pinned to {host}")


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


def _frozen_campaign_command(host: str, bundle_relative: Path) -> list[str]:
    return [
        *_frozen_python_prefix(host, bundle_relative),
        "tools/r3_action_edit_mlx_campaign.py",
    ]


def _frozen_report_command(host: str, bundle_relative: Path) -> list[str]:
    return [
        *_frozen_python_prefix(host, bundle_relative),
        "tools/r3_action_edit_mlx_report.py",
    ]


def _relative(repository: Path, path: Path, label: str) -> Path:
    try:
        relative = path.resolve().relative_to(repository)
    except ValueError as error:
        raise CampaignError(f"R3 MLX {label} must remain beneath the repository") from error
    if not relative.parts:
        raise CampaignError(f"R3 MLX {label} cannot be the repository root")
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


def _is_blake3(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
            raise CampaignError(f"existing {label} differs from requested identity")
        return
    _write_json_atomic(path, value)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--bundle", type=Path, required=True)
    authorize.add_argument("--cache", type=Path, required=True)
    authorize.add_argument("--s1-cache", type=Path, default=DEFAULT_S1_CACHE)
    authorize.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    authorize.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    authorize.add_argument("--smoke-proof", type=Path, default=DEFAULT_SMOKE_PROOF)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--output", type=Path, default=DEFAULT_AUTHORIZATION)

    export = subparsers.add_parser("export-cache")
    export.add_argument("--host", choices=HOSTS, required=True)
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--s1-cache", type=Path, required=True)
    export.add_argument("--train-dataset", type=Path, required=True)
    export.add_argument("--validation-dataset", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--receipt", type=Path, required=True)
    export.add_argument("--max-groups-per-split", type=int)
    export.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--host", choices=HOSTS, required=True)
    preflight.add_argument("--arm", choices=ARMS, required=True)
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--cache", type=Path, required=True)
    preflight.add_argument("--s1-cache", type=Path, required=True)
    preflight.add_argument("--train-dataset", type=Path, required=True)
    preflight.add_argument("--validation-dataset", type=Path, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--smoke-proof", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--cache", type=Path, required=True)
    queue.add_argument("--s1-cache", type=Path, default=DEFAULT_S1_CACHE)
    queue.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    queue.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    queue.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    queue.add_argument("--smoke-proof", type=Path, default=DEFAULT_SMOKE_PROOF)
    queue.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
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
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                smoke_proof=args.smoke_proof,
                approved_by=args.approved_by,
            )
            _write_once(args.output, report, "R3 MLX authorization")
        elif args.command == "export-cache":
            report = export_cache(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                s1_cache=args.s1_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                output_root=args.output_root,
                receipt=args.receipt,
                maximum_groups_per_split=args.max_groups_per_split,
            )
            _write_json_atomic(args.output, report)
        elif args.command == "preflight":
            report = run_preflight(
                host=args.host,
                arm=args.arm,
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
                smoke_proof=args.smoke_proof,
            )
            _write_json_atomic(args.output, report)
        else:
            validate_bundle_for_campaign(args.bundle)
            validate_authorization(
                args.authorization,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                smoke_proof=args.smoke_proof,
            )
            specs = build_task_specs(
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                s1_cache=args.s1_cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
                smoke_proof=args.smoke_proof,
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
