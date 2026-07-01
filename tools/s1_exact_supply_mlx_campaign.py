#!/usr/bin/env python3
"""Authorize and describe the inert four-host ADR 0147 MLX campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.run_manifest import source_provenance
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ADR_ID,
    ARM_INPUT_CONTRACTS,
    ARMS,
    CATALOG_BLAKE3,
    EXPERIMENT_ID,
    NORMALIZATION_CONTRACT,
    PROTOCOL_ID,
    S1_D6_CONTRACT,
    S1ExactSupplyCache,
    transform_s1_exact_supply_batch,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
    S1ExactSupplyRanker,
    parameter_count,
    parameter_layout_blake3,
)
from cascadia_mlx.s1_exact_supply_mlx_train import (
    TRAINING_SEED,
    S1ExactSupplyTrainingProtocol,
)
from cluster_research_queue import add_task, empty_queue, validate_queue
from mlx.utils import tree_flatten
from rust_experiment_bundle import BundleError, file_blake3, validate_bundle

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
}
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
}
REPLAY_ROLE = "independent-replay-control"
EXPORTER_BINARY = "s1_exact_supply_mlx_exporter"
TASK_PREFIX = "s1esmlx"
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_TRAIN_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-train"
)
DEFAULT_VALIDATION_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-validation"
)
DEFAULT_AUTHORIZATION = DEFAULT_EXPERIMENT_ROOT / "control/authorization.json"
REQUIRED_SOURCE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "python/cascadia_mlx/s1_exact_supply_mlx_cache.py",
    "python/cascadia_mlx/s1_exact_supply_mlx_metrics.py",
    "python/cascadia_mlx/s1_exact_supply_mlx_model.py",
    "python/cascadia_mlx/s1_exact_supply_mlx_train.py",
    "docs/v2/decisions/0147-exact-semantic-supply-learned-comparison.md",
    (
        "docs/v2/reports/"
        "exact-semantic-supply-learned-comparison-v1-preregistration.md"
    ),
    "tools/s1_exact_supply_mlx_campaign.py",
    "tools/s1_exact_supply_mlx_exporter/Cargo.lock",
    "tools/s1_exact_supply_mlx_exporter/Cargo.toml",
    "tools/s1_exact_supply_mlx_exporter/README.md",
    "tools/s1_exact_supply_mlx_exporter/src/main.rs",
    "tools/s1_exact_supply_mlx_report.py",
    "tools/cluster_artifact_collect.py",
    "tools/cluster_artifact_fanout.py",
    "tools/cluster_research_queue.py",
    "tools/rust_experiment_bundle.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "python/cascadia_mlx/",
    "crates/cascadia-data/",
    "crates/cascadia-game/",
    "crates/cascadia-provenance/",
)


class CampaignError(RuntimeError):
    """The S1 campaign cannot proceed without changing scientific identity."""


def validate_bundle_for_campaign(bundle: Path) -> dict[str, Any]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    if manifest.get("identity", {}).get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("immutable bundle names the wrong experiment")
    source_entries = manifest.get("identity", {}).get("source_files", [])
    paths = {
        entry.get("path")
        for entry in source_entries
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
    if missing_files or missing_prefixes or EXPORTER_BINARY not in binaries:
        raise CampaignError(
            "S1 immutable bundle is incomplete: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}, "
            f"missing_exporter={EXPORTER_BINARY not in binaries}"
        )
    return manifest


def create_authorization(
    *,
    bundle: Path,
    cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    """Create an explicit parent authorization without starting training."""
    if not approved_by.strip():
        raise CampaignError("S1 authorization requires a nonempty approver")
    bundle = bundle.resolve()
    cache = cache.resolve()
    train_dataset = train_dataset.resolve()
    validation_dataset = validation_dataset.resolve()
    bundle_manifest = validate_bundle_for_campaign(bundle)
    sidecar = S1ExactSupplyCache(cache, require_complete=True)
    train = sidecar.bind_dataset(train_dataset, arm=ARMS[0])
    validation = sidecar.bind_dataset(validation_dataset, arm=ARMS[0])
    if train.split != "train" or validation.split != "validation":
        raise CampaignError("S1 authorization accepts only open train and validation splits")
    source = source_provenance(bundle / "source")
    exporter = bundle / "bin" / EXPORTER_BINARY
    exporter_blake3 = file_blake3(exporter)
    _validate_cache_exporter_binding(sidecar, exporter_blake3)
    parameter_counts = _parameter_counts()
    parameter_layouts = _parameter_layouts()
    if (
        set(parameter_counts.values()) != {FROZEN_PARAMETER_COUNT}
        or len(set(parameter_layouts.values())) != 1
    ):
        raise CampaignError("S1 model capacity does not match the frozen iso-parameter contract")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "bundle_id": bundle_manifest["bundle_id"],
        "bundle_source_v2_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": exporter_blake3,
        "cache_id": sidecar.cache_id,
        "cache_manifest_blake3": file_blake3(sidecar.manifest_path),
        "catalog_blake3": CATALOG_BLAKE3,
        "train_dataset_id": train.manifest["dataset_id"],
        "train_manifest_blake3": file_blake3(train.root / "dataset.json"),
        "validation_dataset_id": validation.manifest["dataset_id"],
        "validation_manifest_blake3": file_blake3(
            validation.root / "dataset.json"
        ),
        "protocol": S1ExactSupplyTrainingProtocol().to_dict(),
        "protocol_blake3": canonical_blake3(
            S1ExactSupplyTrainingProtocol().to_dict()
        ),
        "d6_contract": S1_D6_CONTRACT,
        "normalization": NORMALIZATION_CONTRACT,
        "arm_input_contracts": ARM_INPUT_CONTRACTS,
        "collision_witness_id": sidecar.manifest["collision_witness"]["witness_id"],
        "authorized_arms": list(ARMS),
        "independent_replay_role": REPLAY_ROLE,
        "cross_arm_parameter_counts": parameter_counts,
        "cross_arm_parameter_layout_blake3": parameter_layouts,
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


def export_cache(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    train_dataset: Path,
    validation_dataset: Path,
    output_root: Path,
    receipt: Path,
    maximum_groups_per_split: int | None = None,
) -> dict[str, Any]:
    """Run the reviewed Rust exporter once without authorizing training."""
    if host != "john1":
        raise CampaignError("the shared S1 exact-supply cache is exported once on john1")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("S1 cache export repository must be immutable bundle source")
    bundle_manifest = validate_bundle_for_campaign(bundle)
    exporter = bundle / "bin" / EXPORTER_BINARY
    exporter_blake3 = file_blake3(exporter)
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
            raise CampaignError("S1 cache smoke bound must be positive")
        command.extend(
            [
                "--max-groups-per-split",
                str(maximum_groups_per_split),
            ]
        )
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CampaignError(
            "S1 exact-supply cache export failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    export = _read_json(receipt, "S1 cache export receipt")
    cache_id = export.get("cache_id")
    if (
        export.get("schema_version") != 1
        or export.get("experiment_id") != EXPERIMENT_ID
        or export.get("protocol_id") != PROTOCOL_ID
        or not _is_blake3(cache_id)
    ):
        raise CampaignError("S1 cache export receipt is malformed")
    cache_root = Path(output_root).resolve() / str(cache_id)
    complete = maximum_groups_per_split is None
    sidecar = S1ExactSupplyCache(cache_root, require_complete=complete)
    _validate_cache_exporter_binding(sidecar, exporter_blake3)
    if bool(export.get("complete_open_corpus")) is not complete:
        raise CampaignError("S1 cache export completeness disagrees with the command")
    if complete:
        sidecar.bind_dataset(train_dataset, arm=ARMS[0])
        sidecar.bind_dataset(validation_dataset, arm=ARMS[0])
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "host": host,
        "bundle_id": bundle_manifest["bundle_id"],
        "exporter_executable_blake3": exporter_blake3,
        "cache_id": sidecar.cache_id,
        "cache_manifest_blake3": file_blake3(sidecar.manifest_path),
        "complete_open_corpus": complete,
        "train_groups": sidecar.splits["train"].groups,
        "train_candidates": sidecar.splits["train"].candidates,
        "validation_groups": sidecar.splits["validation"].groups,
        "validation_candidates": sidecar.splits["validation"].candidates,
        "hidden_information": sidecar.manifest["hidden_information"],
        "production_training_started": False,
        "queue_modified": False,
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
            "queue_modified": False,
            "gameplay_run": False,
            "sealed_test_opened": False,
        },
    }


def validate_authorization(
    path: Path,
    *,
    bundle: Path,
    cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    authorization = _read_json(path, "S1 authorization")
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
        raise CampaignError("S1 authorization is malformed")
    expected = create_authorization(
        bundle=bundle,
        cache=cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        approved_by=str(identity.get("approved_by", "")),
        approved_unix_ms=approved_unix_ms,
    )
    if expected != authorization:
        raise CampaignError("S1 authorization is stale for the bundle, cache, or datasets")
    return authorization


def run_preflight(
    *,
    host: str,
    role: str,
    repository: Path,
    bundle: Path,
    cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    authorization: Path,
) -> dict[str, Any]:
    """Verify one host without starting an optimizer or opening sealed data."""
    if host not in HOSTS:
        raise CampaignError(f"unknown S1 host: {host}")
    if role not in {*ARMS, REPLAY_ROLE}:
        raise CampaignError(f"unknown S1 host role: {role}")
    expected_host = "john4" if role == REPLAY_ROLE else ARM_HOSTS[role]
    if host != expected_host:
        raise CampaignError(f"S1 role {role} is pinned to {expected_host}")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("S1 preflight repository must be immutable bundle source")
    bundle_manifest = validate_bundle_for_campaign(bundle)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        cache=cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    sidecar = S1ExactSupplyCache(cache, require_complete=True)
    sidecar.bind_dataset(train_dataset, arm=ARMS[0])
    sidecar.bind_dataset(validation_dataset, arm=ARMS[0])
    source = source_provenance(repository)

    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    device = str(mx.default_device())
    actual_host = _normalize_host(socket.gethostname().split(".")[0])
    checks = {
        "immutable_bundle_verified": True,
        "authorization_verified": True,
        "cache_verified": True,
        "dataset_manifests_verified": True,
        "apple_silicon_verified": (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ),
        "mlx_gpu_verified": "gpu" in device.lower(),
        "python_bytecode_disabled": sys.dont_write_bytecode,
        "host_assignment_verified": actual_host == host,
        "production_training_started": False,
    }
    if not all(value for key, value in checks.items() if key != "production_training_started"):
        raise CampaignError(f"S1 host preflight failed: {checks}")
    if source["v2_source_blake3"] != approval["identity"]["bundle_source_v2_blake3"]:
        raise CampaignError("S1 preflight source differs from authorization")
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "host": host,
        "arm": role,
        "bundle_id": bundle_manifest["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "cache_id": sidecar.cache_id,
        "source_v2_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": file_blake3(
            bundle / "bin" / EXPORTER_BINARY
        ),
        "train_manifest_blake3": file_blake3(
            Path(train_dataset) / "dataset.json"
        ),
        "validation_manifest_blake3": file_blake3(
            Path(validation_dataset) / "dataset.json"
        ),
        "cross_arm_parameter_counts": _parameter_counts(),
        "cross_arm_parameter_layout_blake3": _parameter_layouts(),
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "device": device,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "preflight_id": canonical_blake3(identity),
        "scientific_identity": identity,
        "checks": checks,
        "operational": {
            "hostname": actual_host,
            "repository": str(repository),
            "bundle": str(bundle),
            "cache": str(Path(cache).resolve()),
        },
    }


def run_replay_control(
    *,
    bundle: Path,
    cache: Path,
    train_dataset: Path,
    validation_dataset: Path,
    authorization: Path,
    preflight: Path,
) -> dict[str, Any]:
    """Independently replay all public bindings and model/D6 fairness checks."""
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        cache=cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    host_preflight = _read_json(preflight, "S1 replay preflight")
    if (
        host_preflight.get("scientific_identity", {}).get("arm") != REPLAY_ROLE
        or host_preflight.get("scientific_identity", {}).get("host") != "john4"
        or host_preflight.get("scientific_identity", {}).get("authorization_id")
        != approval["authorization_id"]
    ):
        raise CampaignError("S1 replay preflight is stale or assigned to the wrong host")
    sidecar = S1ExactSupplyCache(cache, require_complete=True)
    coverage: dict[str, dict[str, int]] = {}
    first_batch = None
    for split, root in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        dataset = sidecar.bind_dataset(root, arm=ARMS[0])
        groups = 0
        candidates = 0
        for batch in dataset.batches(64):
            if first_batch is None:
                first_batch = batch
            groups += int(batch.candidate_mask.shape[0])
            candidates += int(np.asarray(batch.candidate_mask).sum())
        coverage[split] = {"groups": groups, "candidates": candidates}

    if first_batch is None:
        raise CampaignError("S1 replay did not observe any open training batch")
    d6_round_trips = 0
    invariant_supply = True
    original_supply = np.asarray(first_batch.supply_vector)
    original_tokens = np.asarray(first_batch.supply_tokens)
    original_frontier = np.asarray(first_batch.frontier_features)
    original_boards = np.asarray(first_batch.board_entities)
    original_actions = np.asarray(first_batch.action_features)
    for transform_id in range(12):
        transformed = transform_s1_exact_supply_batch(first_batch, transform_id)
        inverse = S1_D6_CONTRACT["transform_ids"][
            _d6_inverse(transform_id)
        ]
        restored = transform_s1_exact_supply_batch(transformed, inverse)
        invariant_supply &= (
            np.array_equal(np.asarray(transformed.supply_vector), original_supply)
            and np.array_equal(np.asarray(transformed.supply_tokens), original_tokens)
            and np.array_equal(
                np.asarray(transformed.frontier_features),
                original_frontier,
            )
        )
        if np.allclose(np.asarray(restored.board_entities), original_boards) and np.allclose(
            np.asarray(restored.action_features),
            original_actions,
        ):
            d6_round_trips += 1

    fingerprints = {}
    parameter_counts = {}
    parameter_layouts = {}
    for arm in ARMS:
        mx.random.seed(TRAINING_SEED)
        model = S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        parameter_counts[arm] = parameter_count(model)
        parameter_layouts[arm] = parameter_layout_blake3(model)
        fingerprints[arm] = _parameter_fingerprint(model)
    checks = {
        "full_train_group_coverage": coverage["train"]["groups"] == 560,
        "full_train_candidate_coverage": (
            coverage["train"]["candidates"] == 2_135_111
        ),
        "full_validation_group_coverage": (
            coverage["validation"]["groups"] == 240
        ),
        "full_validation_candidate_coverage": (
            coverage["validation"]["candidates"] == 860_203
        ),
        "all_12_d6_inverse_round_trips": d6_round_trips == 12,
        "supply_and_frontier_features_d6_invariant": invariant_supply,
        "cross_arm_parameter_counts_equal": len(set(parameter_counts.values())) == 1,
        "cross_arm_parameter_count_frozen": (
            set(parameter_counts.values()) == {FROZEN_PARAMETER_COUNT}
        ),
        "cross_arm_parameter_layouts_identical": (
            len(set(parameter_layouts.values())) == 1
        ),
        "cross_arm_initial_weights_identical": len(set(fingerprints.values())) == 1,
        "cache_hidden_information_boundary_clean": all(
            sidecar.manifest["hidden_information"][field] is False
            for field in (
                "hidden_stack_order_read",
                "hidden_wildlife_order_read",
                "excluded_tile_identities_read",
                "future_refills_read",
                "sealed_test_opened",
                "gameplay_opened",
            )
        ),
    }
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "host": "john4",
        "role": REPLAY_ROLE,
        "bundle_id": approval["identity"]["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "cache_id": sidecar.cache_id,
        "coverage": coverage,
        "d6_contract": S1_D6_CONTRACT,
        "d6_round_trips": d6_round_trips,
        "parameter_counts": parameter_counts,
        "parameter_layout_blake3": parameter_layouts,
        "initial_weight_fingerprints": fingerprints,
        "checks": checks,
        "sealed_test_opened": False,
        "gameplay_run": False,
        "training_run": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "cache_id": sidecar.cache_id,
        "authorization_id": approval["authorization_id"],
        "scientific_identity": identity,
        "replay_id": canonical_blake3(identity),
        "checks": checks,
        "passed": all(checks.values()),
        "claims": {
            "independent_replay_complete": all(checks.values()),
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
    train_dataset: Path,
    validation_dataset: Path,
    authorization: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
) -> list[dict[str, Any]]:
    """Build the generated-only, nonduplicative four-host execution graph."""
    repository = repository.resolve()
    bundle_relative = _relative(repository, bundle, "bundle")
    cache_relative = _relative(repository, cache, "cache")
    train_relative = _relative(repository, train_dataset, "train dataset")
    validation_relative = _relative(
        repository,
        validation_dataset,
        "validation dataset",
    )
    authorization_relative = _relative(
        repository,
        authorization,
        "authorization",
    )
    experiment_relative = _relative(
        repository,
        experiment_root,
        "experiment root",
    )
    specs: list[dict[str, Any]] = []
    fanouts = []
    for name, relative, required in (
        ("bundle", bundle_relative, "bundle.json"),
        ("cache", cache_relative, "cache.json"),
        ("train", train_relative, "dataset.json"),
        ("validation", validation_relative, "dataset.json"),
        ("authorization", authorization_relative.parent, authorization_relative.name),
    ):
        task_id = f"{TASK_PREFIX}-fanout-{name}"
        fanouts.append(task_id)
        specs_task = _task(
            task_id=task_id,
            title=f"Fan out S1 {name}",
            decision=f"Make the frozen {name} byte-identical on all four hosts",
            workload_class="shared-prerequisite",
            priority=1,
            expected_runtime_seconds=180 if name in {"cache", "train", "validation"} else 60,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[] if name == "bundle" else [f"{TASK_PREFIX}-fanout-bundle"],
            command=[
                ".venv/bin/python",
                "-B",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{relative}/",
                "--local-root",
                str(relative),
                *[
                    item
                    for host in HOSTS[1:]
                    for item in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / relative}/",
                    )
                ],
                "--required-file",
                required,
                "--verify-tree",
                "--output",
                str(experiment_relative / f"reports/fanout-{name}.json"),
            ],
            artifact_path=str(experiment_relative / f"reports/fanout-{name}.json"),
            stop_rule=f"Every {name} byte must match before host preflight.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
        specs.append(specs_task)

    roles = {
        "john1": ARMS[0],
        "john2": ARMS[1],
        "john3": ARMS[2],
        "john4": REPLAY_ROLE,
    }
    preflight_ids = {}
    for host, role in roles.items():
        task_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids[host] = task_id
        output = experiment_relative / f"reports/preflight-{host}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Preflight S1 {role} on {host}",
                decision="Prove immutable source, public data, MLX GPU, and launch controls",
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=240,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=fanouts,
                command=[
                    *_frozen_campaign_command(host, bundle_relative),
                    "preflight",
                    "--host",
                    host,
                    "--role",
                    role,
                    "--repository",
                    str(REMOTE_ROOTS[host] / bundle_relative / "source"),
                    "--bundle",
                    str(REMOTE_ROOTS[host] / bundle_relative),
                    "--cache",
                    str(REMOTE_ROOTS[host] / cache_relative),
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--output",
                    str(REMOTE_ROOTS[host] / output),
                ],
                artifact_path=str(output),
                stop_rule="No training or replay begins unless every preflight check is true.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=True,
            )
        )

    all_preflights = list(preflight_ids.values())
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
                title=f"Train S1 {arm}",
                decision=f"Measure the frozen {arm} representation on identical rows",
                workload_class="independent-experiment",
                priority=10,
                expected_runtime_seconds=14_400,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=all_preflights,
                command=[
                    *_frozen_python_prefix(host, bundle_relative),
                    "-m",
                    "cascadia_mlx.s1_exact_supply_mlx_train",
                    "--train-dataset",
                    str(REMOTE_ROOTS[host] / train_relative),
                    "--validation-dataset",
                    str(REMOTE_ROOTS[host] / validation_relative),
                    "--cache",
                    str(REMOTE_ROOTS[host] / cache_relative),
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
                    "--resume-if-present",
                ],
                artifact_path=str(report),
                stop_rule="Finish the fixed epoch/patience protocol and validation benchmark.",
                cpu_cores=10,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    replay_id = f"{TASK_PREFIX}-replay-control"
    replay_report = experiment_relative / "reports/independent-replay-control.json"
    specs.append(
        _task(
            task_id=replay_id,
            title="Run independent S1 replay/control on john4",
            decision=(
                "Rebind every public row and verify D6, initialization, "
                "and parameter fairness"
            ),
            workload_class="replica",
            priority=10,
            expected_runtime_seconds=3_600,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john4"],
            dependencies=all_preflights,
            command=[
                *_frozen_campaign_command("john4", bundle_relative),
                "replay-control",
                "--bundle",
                str(REMOTE_ROOTS["john4"] / bundle_relative),
                "--cache",
                str(REMOTE_ROOTS["john4"] / cache_relative),
                "--train-dataset",
                str(REMOTE_ROOTS["john4"] / train_relative),
                "--validation-dataset",
                str(REMOTE_ROOTS["john4"] / validation_relative),
                "--authorization",
                str(REMOTE_ROOTS["john4"] / authorization_relative),
                "--preflight",
                str(
                    REMOTE_ROOTS["john4"]
                    / experiment_relative
                    / "reports/preflight-john4.json"
                ),
                "--output",
                str(REMOTE_ROOTS["john4"] / replay_report),
            ],
            artifact_path=str(replay_report),
            stop_rule="All open rows and all 12 D6 transforms must replay without drift.",
            cpu_cores=10,
            memory_gib=8.0,
            uses_mlx=True,
        )
    )

    collection_id = f"{TASK_PREFIX}-collect"
    collected = experiment_relative / "reports/collected"
    collection_report = experiment_relative / "reports/collection.json"
    collect_command = [
        ".venv/bin/python",
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
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
    collect_command.extend(
        [
            "--artifact",
            f"john4:{REMOTE_ROOTS['john4'] / replay_report}",
            str(collected / "independent-replay-control.json"),
            "--output",
            str(collection_report),
        ]
    )
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect S1 arm and replay reports",
            decision="Checksum-copy exactly three arm reports and one independent replay",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[*arm_task_ids, replay_id],
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
        ordered_arms = ARMS if order == "forward" else tuple(reversed(ARMS))
        output = experiment_relative / f"reports/classification-{order}.json"
        command = [
            *_frozen_report_command("john1", bundle_relative),
            "classify",
        ]
        for arm in ordered_arms:
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
                "--replay",
                str(
                    REMOTE_ROOTS["john1"]
                    / collected
                    / "independent-replay-control.json"
                ),
                "--output",
                str(REMOTE_ROOTS["john1"] / output),
            ]
        )
        specs.append(
            _task(
                task_id=task_id,
                title=f"Classify S1 in {order} order",
                decision="Apply ADR 0147 gates without promotion or gameplay",
                workload_class="replica",
                priority=40,
                expected_runtime_seconds=30,
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
            title="Prove S1 classifier order invariance",
            decision="Require byte-identical forward and reverse classifications",
            workload_class="replica",
            priority=50,
            expected_runtime_seconds=10,
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
            stop_rule="Classification bytes must be independent of report input order.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    _validate_task_specs(specs)
    return specs


def queue_specification(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a deterministic preview without touching the live queue ledger."""
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


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("S1 queue graph contains duplicate task IDs")
    known = set(identifiers)
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(
                f"S1 task {spec['id']} has unknown dependencies: {sorted(unknown)}"
            )
        command = spec["command"]
        if any("python" in item for item in command) and "-B" not in command:
            raise CampaignError(f"frozen Python task omits -B: {spec['id']}")
    for arm, host in ARM_HOSTS.items():
        task = next(
            spec
            for spec in specs
            if spec["id"] == f"{TASK_PREFIX}-train-{_slug(arm)}"
        )
        if task["compatible_hosts"] != [host]:
            raise CampaignError(f"S1 arm {arm} is not pinned to {host}")
    replay = next(spec for spec in specs if spec["id"] == f"{TASK_PREFIX}-replay-control")
    if replay["compatible_hosts"] != ["john4"]:
        raise CampaignError("S1 independent replay is not pinned to john4")


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
        "tools/s1_exact_supply_mlx_campaign.py",
    ]


def _frozen_report_command(host: str, bundle_relative: Path) -> list[str]:
    return [
        *_frozen_python_prefix(host, bundle_relative),
        "tools/s1_exact_supply_mlx_report.py",
    ]


def _relative(repository: Path, path: Path, label: str) -> Path:
    try:
        relative = path.resolve().relative_to(repository)
    except ValueError as error:
        raise CampaignError(f"S1 {label} must remain beneath the repository") from error
    if not relative.parts:
        raise CampaignError(f"S1 {label} cannot be the repository root")
    return relative


def _parameter_counts() -> dict[str, int]:
    return {
        arm: parameter_count(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        )
        for arm in ARMS
    }


def _validate_cache_exporter_binding(
    cache: S1ExactSupplyCache,
    exporter_blake3: str,
) -> None:
    exporter = cache.manifest.get("exporter", {})
    if (
        not _is_blake3(exporter_blake3)
        or not isinstance(exporter, dict)
        or exporter.get("executable_blake3") != exporter_blake3
    ):
        raise CampaignError("S1 cache was not produced by the immutable bundle exporter")


def _parameter_layouts() -> dict[str, str]:
    return {
        arm: parameter_layout_blake3(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        )
        for arm in ARMS
    }


def _parameter_fingerprint(model: S1ExactSupplyRanker) -> str:
    parameters = tree_flatten(model.parameters())
    mx.eval(*(value for _, value in parameters))
    digest = blake3.blake3()
    for name, value in parameters:
        encoded_name = name.encode()
        array = np.asarray(value)
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _d6_inverse(transform_id: int) -> int:
    from cascadia_mlx.d6_contract import D6_CONTRACT

    return D6_CONTRACT.inverse_table[transform_id]


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _normalize_host(value: str) -> str:
    return HOST_ALIASES.get(value.removesuffix(".local"), value.removesuffix(".local"))


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


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
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--bundle", type=Path, required=True)
    authorize.add_argument("--cache", type=Path, required=True)
    authorize.add_argument(
        "--train-dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )
    authorize.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--output", type=Path, default=DEFAULT_AUTHORIZATION)

    export = subparsers.add_parser("export-cache")
    export.add_argument("--host", choices=HOSTS, required=True)
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--train-dataset", type=Path, required=True)
    export.add_argument("--validation-dataset", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--receipt", type=Path, required=True)
    export.add_argument("--max-groups-per-split", type=int)
    export.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--host", choices=HOSTS, required=True)
    preflight.add_argument("--role", choices=(*ARMS, REPLAY_ROLE), required=True)
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--cache", type=Path, required=True)
    preflight.add_argument("--train-dataset", type=Path, required=True)
    preflight.add_argument("--validation-dataset", type=Path, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser("replay-control")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--cache", type=Path, required=True)
    replay.add_argument("--train-dataset", type=Path, required=True)
    replay.add_argument("--validation-dataset", type=Path, required=True)
    replay.add_argument("--authorization", type=Path, required=True)
    replay.add_argument("--preflight", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--cache", type=Path, required=True)
    queue.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    queue.add_argument(
        "--validation-dataset",
        type=Path,
        default=DEFAULT_VALIDATION_DATASET,
    )
    queue.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    queue.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    queue.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "authorize":
            report = create_authorization(
                bundle=args.bundle,
                cache=args.cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                approved_by=args.approved_by,
            )
            _write_once(args.output, report, "S1 authorization")
        elif args.command == "export-cache":
            report = export_cache(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
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
                role=args.role,
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
            )
            _write_json_atomic(args.output, report)
        elif args.command == "replay-control":
            report = run_replay_control(
                bundle=args.bundle,
                cache=args.cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
                preflight=args.preflight,
            )
            _write_json_atomic(args.output, report)
        else:
            validate_bundle_for_campaign(args.bundle)
            validate_authorization(
                args.authorization,
                bundle=args.bundle,
                cache=args.cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
            )
            specs = build_task_specs(
                repository=args.repository,
                bundle=args.bundle,
                cache=args.cache,
                train_dataset=args.train_dataset,
                validation_dataset=args.validation_dataset,
                authorization=args.authorization,
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
