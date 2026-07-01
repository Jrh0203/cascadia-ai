#!/usr/bin/env python3
"""Freeze, authorize, preflight, and specify the inert ADR 0146 campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.dataset import Dataset
from cascadia_mlx.r2_sparse_mlx_cache import (
    CORPUS_LOCK_CONTRACT,
    CORPUS_LOCK_SCHEMA_VERSION,
    EXPECTED_ACTIVE_TOKENS,
    EXPECTED_LAYER_MAXIMA,
    EXPECTED_TYPE_TOKEN_TOTALS,
    EXPERIMENT_ID,
    FOUNDATION_EXPERIMENT_ID,
    FOUNDATION_PACKED_STATE_BLAKE3,
    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS,
    FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
    FOUNDATION_PUBLIC_POSITION_BLAKE3,
    FOUNDATION_SCIENTIFIC_BLAKE3,
    R2SparseMlxCache,
)
from cascadia_mlx.r2_sparse_mlx_model import architecture_parameter_counts
from cascadia_mlx.r2_sparse_mlx_tournament import (
    ADR_ID,
    AUTHORIZED_RUNS,
    PROTOCOL_ID,
    RUN_ARCHITECTURES,
    R2SparseMlxTournamentConfig,
    R2SparseMlxTournamentProtocol,
    report_scientific_identity,
    run_tournament,
)
from cascadia_mlx.run_manifest import source_provenance
from r2_sparse_mlx_report import (
    R0_BINDING_CONTRACT,
    R0_COMPLETE_CLASSIFICATION,
    R0_EXACT_CONTROL,
)
from rust_experiment_bundle import BundleError, file_blake3, validate_bundle

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
RUN_HOSTS = {
    "set-primary": "john1",
    "graph-primary": "john2",
    "perceiver-primary": "john3",
    "set-replay": "john4",
}
TASK_PREFIX = "r2smlx"
FEATURE_SCHEMA = "compact-entity-v2"
TARGET_SCHEMA = "base-score-components-v1"
STRATEGY_ID = "pattern-aware-v1-k8-h6-b8-m4"
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_DATASET_ROOT = Path(
    "artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen"
)
DEFAULT_CORPUS_LOCK = DEFAULT_EXPERIMENT_ROOT / "control/corpus-lock.json"
DEFAULT_R0_BINDING = DEFAULT_EXPERIMENT_ROOT / "control/r0-control-binding.json"
DEFAULT_AUTHORIZATION = DEFAULT_EXPERIMENT_ROOT / "control/authorization.json"
DEFAULT_CACHE_PARENT = DEFAULT_EXPERIMENT_ROOT / "caches"
DEFAULT_REPORT_ROOT = DEFAULT_EXPERIMENT_ROOT / "reports"
DEFAULT_R0_ROOT = Path("artifacts/experiments/r0-spatial-mlx-tournament-v1")
DEFAULT_R0_CLASSIFICATION_FORWARD = (
    DEFAULT_R0_ROOT / "reports/classification-forward.json"
)
DEFAULT_R0_CLASSIFICATION_REVERSE = (
    DEFAULT_R0_ROOT / "reports/classification-reverse.json"
)
DEFAULT_R0_ORDER_PROOF = DEFAULT_R0_ROOT / "reports/classification-order-proof.json"
DEFAULT_R0_COLLECTION = DEFAULT_R0_ROOT / "reports/collection.json"
EXPORTER_BINARY = "r2-sparse-entity-census"

TRAIN_GAMES = (157, 156, 156, 156)
TRAIN_FIRST_GAME_INDEX = (200_000, 200_157, 200_313, 200_469)
VALIDATION_GAMES = (32, 31, 31, 31)
VALIDATION_FIRST_GAME_INDEX = (210_000, 210_032, 210_063, 210_094)
EXPECTED_MANIFEST_BLAKE3 = (
    "57f86b3f6ae06bee782974995aa6b8d3cad6f637e68d5ef8aac7ffd8112d4244",
    "79bcceebd52144f8c39130de15404f0f2820b695111f2f1e9004dcac5f33c555",
    "fbddc7aa1794b753fcbd3d8f030b51dcc4456051f61f7914eab541e9658db666",
    "8ab6d2a9229f3cfe8bf1567c3a9d110b9268e322a0c96cf30ba131c937435849",
    "a991d05962965d61a31d40fe0b8572c743cff04a12d1e948be9e2fa3e6a871d4",
    "adf3903a59d9d522fbb9fab2bb3c8a9370c7f2d46c3aa74ac85b6879b80efddc",
    "9bfeed300489ac6610313dd2bf032c809197be92cfeac43b357a4cb8aca14803",
    "7491212c5a524f954414402661a6aa064161a16cfe23755e051d80886b257186",
)
REQUIRED_SOURCE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "pyproject.toml",
    "uv.lock",
    "tools/cluster_artifact_fanout.py",
    "tools/r2_sparse_mlx_campaign.py",
    "tools/r2_sparse_mlx_report.py",
    "tools/rust_experiment_bundle.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "crates/cascadia-data/",
    "crates/cascadia-game/",
    "python/cascadia_mlx/",
    "tools/r2_sparse_entity_census/",
)


class CampaignError(RuntimeError):
    """Raised when the R2 campaign cannot proceed without ambiguity."""


@dataclass(frozen=True)
class DatasetPart:
    split: str
    part_index: int
    games: int
    first_game_index: int
    root: Path
    manifest_blake3: str

    @property
    def records(self) -> int:
        return self.games * 80

    @property
    def dataset_id(self) -> str:
        return (
            f"{STRATEGY_ID}-{self.split}-"
            f"{self.first_game_index}"
        )


def dataset_parts(dataset_root: Path = DEFAULT_DATASET_ROOT) -> tuple[DatasetPart, ...]:
    parts: list[DatasetPart] = []
    digest_index = 0
    for split, games, first_indexes in (
        ("train", TRAIN_GAMES, TRAIN_FIRST_GAME_INDEX),
        ("validation", VALIDATION_GAMES, VALIDATION_FIRST_GAME_INDEX),
    ):
        for part_index in range(len(HOSTS)):
            parts.append(
                DatasetPart(
                    split=split,
                    part_index=part_index,
                    games=games[part_index],
                    first_game_index=first_indexes[part_index],
                    root=Path(f"{dataset_root}-{split}-part-{part_index}"),
                    manifest_blake3=EXPECTED_MANIFEST_BLAKE3[digest_index],
                )
            )
            digest_index += 1
    return tuple(parts)


def freeze_corpus(
    roots: list[Path] | None = None,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Validate every dataset byte and bind the accepted ADR 0145 foundation."""
    parts = dataset_parts(dataset_root)
    selected_roots = list(roots) if roots is not None else [part.root for part in parts]
    if len(selected_roots) != len(parts):
        raise CampaignError("the R2 MLX corpus requires exactly eight dataset roots")

    identities: list[dict[str, Any]] = []
    for order, (part, root) in enumerate(zip(parts, selected_roots, strict=True)):
        root = root.resolve()
        manifest_path = root / "dataset.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(
                f"cannot read dataset manifest {manifest_path}: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise CampaignError(f"dataset manifest must be an object: {manifest_path}")
        manifest_digest = blake3.blake3(manifest_bytes).hexdigest()
        _validate_dataset_part(part, root, manifest, manifest_digest)
        try:
            Dataset(root, verify_checksums=True)
        except ValueError as error:
            raise CampaignError(f"dataset validation failed at {root}: {error}") from error
        identities.append(
            {
                "order": order,
                "split": part.split,
                "root_name": root.name,
                "dataset_id": part.dataset_id,
                "total_records": part.records,
                "manifest_blake3": manifest_digest,
            }
        )

    identity = {
        "foundation_experiment_id": FOUNDATION_EXPERIMENT_ID,
        "foundation_scientific_blake3": FOUNDATION_SCIENTIFIC_BLAKE3,
        "foundation_public_position_blake3": FOUNDATION_PUBLIC_POSITION_BLAKE3,
        "foundation_packed_state_blake3": FOUNDATION_PACKED_STATE_BLAKE3,
        "feature_schema": FEATURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "total_records": 60_000,
        "train_records": 50_000,
        "validation_records": 10_000,
        "layer_maxima": EXPECTED_LAYER_MAXIMA,
        "type_token_totals": EXPECTED_TYPE_TOKEN_TOTALS,
        "active_tokens": EXPECTED_ACTIVE_TOKENS,
        "per_board_p99_active_tokens": FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
        "per_board_max_active_tokens": FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS,
        "datasets": identities,
    }
    return {
        "schema_version": CORPUS_LOCK_SCHEMA_VERSION,
        "contract_id": CORPUS_LOCK_CONTRACT,
        "lock_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_corpus_lock(
    path: Path,
    *,
    roots: list[Path] | None = None,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    lock = _read_json(path, "corpus lock")
    if (
        lock.get("schema_version") != CORPUS_LOCK_SCHEMA_VERSION
        or lock.get("contract_id") != CORPUS_LOCK_CONTRACT
        or not isinstance(lock.get("identity"), dict)
        or canonical_blake3(lock["identity"]) != lock.get("lock_id")
    ):
        raise CampaignError("R2 MLX corpus lock is malformed or content-address drifted")
    identity = lock["identity"]
    if (
        identity.get("foundation_experiment_id") != FOUNDATION_EXPERIMENT_ID
        or identity.get("foundation_scientific_blake3")
        != FOUNDATION_SCIENTIFIC_BLAKE3
        or identity.get("foundation_public_position_blake3")
        != FOUNDATION_PUBLIC_POSITION_BLAKE3
        or identity.get("foundation_packed_state_blake3")
        != FOUNDATION_PACKED_STATE_BLAKE3
        or identity.get("total_records") != 60_000
        or identity.get("train_records") != 50_000
        or identity.get("validation_records") != 10_000
        or identity.get("layer_maxima") != EXPECTED_LAYER_MAXIMA
        or identity.get("type_token_totals") != EXPECTED_TYPE_TOKEN_TOTALS
        or identity.get("active_tokens") != EXPECTED_ACTIVE_TOKENS
        or identity.get("per_board_p99_active_tokens")
        != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
        or identity.get("per_board_max_active_tokens")
        != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
        or len(identity.get("datasets", [])) != 8
    ):
        raise CampaignError("R2 MLX corpus lock does not bind the accepted foundation")
    if roots is not None:
        observed = freeze_corpus(roots, dataset_root=dataset_root)
        if observed != lock:
            raise CampaignError("local corpus bytes do not match the supplied R2 lock")
    return lock


def bind_r0_control(
    *,
    classification_forward: Path,
    classification_reverse: Path,
    order_proof: Path,
    collection: Path,
) -> dict[str, Any]:
    """Bind the completed R0 selection, falling back to exact only on explicit null."""
    try:
        forward_bytes = classification_forward.read_bytes()
        reverse_bytes = classification_reverse.read_bytes()
    except OSError as error:
        raise CampaignError(f"cannot read R0 classification: {error}") from error
    if forward_bytes != reverse_bytes:
        raise CampaignError("R0 forward and reverse classifications are not byte-identical")
    try:
        classification = json.loads(forward_bytes)
    except json.JSONDecodeError as error:
        raise CampaignError("R0 classification JSON is malformed") from error
    proof = _read_json(order_proof, "R0 classification order proof")
    reports = _read_json(collection, "R0 report collection")
    if (
        not isinstance(classification, dict)
        or classification.get("classification") != R0_COMPLETE_CLASSIFICATION
        or not _is_digest(classification.get("aggregate_id"))
        or proof.get("byte_identical") is not True
        or proof.get("classification_blake3")
        != blake3.blake3(forward_bytes).hexdigest()
        or not isinstance(reports.get("reports"), list)
    ):
        raise CampaignError("R0 result is unavailable, incomplete, or lacks order proof")

    selected = classification.get("selected_stage2_candidate")
    selected_control = selected if selected is not None else R0_EXACT_CONTROL
    eligible = {
        R0_EXACT_CONTROL,
        "hex-radius-6-127",
        "hex-radius-5-91",
        "hex-radius-4-61",
    }
    if selected_control not in eligible:
        raise CampaignError("R0 selected an ineligible control arm")
    entry = next(
        (
            item
            for item in reports["reports"]
            if isinstance(item, dict) and item.get("arm") == selected_control
        ),
        None,
    )
    if entry is None:
        raise CampaignError(f"R0 collection lacks selected control {selected_control}")
    report_path = Path(str(entry.get("file", "")))
    if not report_path.is_absolute():
        report_path = collection.parent / report_path
    report = _read_json(report_path, "R0 selected control report")
    if (
        file_blake3(report_path) != entry.get("blake3")
        or report.get("report_id") != entry.get("report_id")
        or report.get("arm") != selected_control
        or report.get("integrity", {}).get("all_metrics_finite") is not True
        or report.get("claims", {}).get("promotion_authorized") is not False
    ):
        raise CampaignError("R0 selected control report failed integrity validation")
    validation = report.get("metrics", {}).get("validation")
    if not isinstance(validation, dict) or not _all_finite(validation):
        raise CampaignError("R0 selected control lacks finite validation metrics")

    identity = {
        "r0_experiment_id": classification.get("experiment_id"),
        "r0_adr": classification.get("adr"),
        "r0_classification": classification["classification"],
        "r0_classification_aggregate_id": classification["aggregate_id"],
        "r0_classification_file_blake3": file_blake3(classification_forward),
        "r0_order_proof_file_blake3": file_blake3(order_proof),
        "classification_order_byte_identical": True,
        "r0_selected_stage2_candidate": selected,
        "selected_control_arm": selected_control,
        "r0_control_report_id": report["report_id"],
        "r0_control_report_file_blake3": file_blake3(report_path),
        "validation": validation,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "contract_id": R0_BINDING_CONTRACT,
        "binding_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_r0_control_binding(path: Path) -> dict[str, Any]:
    binding = _read_json(path, "R0 control binding")
    identity = binding.get("identity")
    if (
        binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("adr") != ADR_ID
        or binding.get("contract_id") != R0_BINDING_CONTRACT
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != binding.get("binding_id")
        or identity.get("r0_classification") != R0_COMPLETE_CLASSIFICATION
        or identity.get("classification_order_byte_identical") is not True
        or not _is_digest(identity.get("r0_control_report_id"))
        or not isinstance(identity.get("validation"), dict)
        or not _all_finite(identity["validation"])
    ):
        raise CampaignError("R0 selected control binding is invalid or incomplete")
    selected = identity.get("r0_selected_stage2_candidate")
    expected = selected if selected is not None else R0_EXACT_CONTROL
    if identity.get("selected_control_arm") != expected:
        raise CampaignError("R0 control fallback is not explicit and fail-closed")
    return binding


def validate_bundle_for_campaign(bundle: Path) -> dict[str, Any]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    if manifest.get("identity", {}).get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("immutable bundle names a different experiment ID")
    source_entries = manifest.get("identity", {}).get("source_files", [])
    source_paths = {
        entry.get("path")
        for entry in source_entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing_files = sorted(REQUIRED_SOURCE_FILES - source_paths)
    missing_prefixes = sorted(
        prefix
        for prefix in REQUIRED_SOURCE_PREFIXES
        if not any(path.startswith(prefix) for path in source_paths)
    )
    binaries = {
        entry.get("name"): entry
        for entry in manifest.get("identity", {}).get("binaries", [])
        if isinstance(entry, dict)
    }
    if missing_files or missing_prefixes or EXPORTER_BINARY not in binaries:
        raise CampaignError(
            "R2 MLX bundle is incomplete: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}, "
            f"missing_exporter={EXPORTER_BINARY not in binaries}"
        )
    return manifest


def create_authorization(
    *,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    if not approved_by.strip():
        raise CampaignError("authorization requires a nonempty approver")
    bundle = bundle.resolve()
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock)
    control = validate_r0_control_binding(r0_control)
    source = source_provenance(bundle / "source")
    exporter = bundle / "bin" / EXPORTER_BINARY
    exporter_digest = file_blake3(exporter)
    binary_entry = next(
        entry
        for entry in bundle_manifest["identity"]["binaries"]
        if entry["name"] == EXPORTER_BINARY
    )
    if binary_entry["blake3"] != exporter_digest:
        raise CampaignError("exporter binary differs from the immutable bundle manifest")
    protocol = R2SparseMlxTournamentProtocol().to_dict()
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": canonical_blake3(protocol),
        "corpus_lock_id": lock["lock_id"],
        "r0_control_binding_id": control["binding_id"],
        "mlx_source_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": exporter_digest,
        "bundle_id": bundle_manifest["bundle_id"],
        "authorized_runs": list(AUTHORIZED_RUNS),
        "run_architectures": RUN_ARCHITECTURES,
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
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_authorization(
    path: Path,
    *,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
) -> dict[str, Any]:
    authorization = _read_json(path, "authorization")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization.get("identity"), dict)
        or canonical_blake3(authorization["identity"])
        != authorization.get("authorization_id")
    ):
        raise CampaignError("R2 MLX production authorization is invalid")
    approved_unix_ms = authorization["identity"].get("approved_unix_ms")
    if (
        not isinstance(approved_unix_ms, int)
        or isinstance(approved_unix_ms, bool)
        or approved_unix_ms < 0
    ):
        raise CampaignError("R2 MLX authorization has an invalid approval timestamp")
    expected = create_authorization(
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
        approved_by=str(authorization["identity"].get("approved_by", "")),
        approved_unix_ms=approved_unix_ms,
    )
    if expected != authorization:
        raise CampaignError(
            "R2 MLX authorization does not match the bundle, corpus, or R0 control"
        )
    return authorization


def run_preflight(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    authorization: Path,
    dataset_roots: list[Path],
) -> dict[str, Any]:
    if host not in HOSTS:
        raise CampaignError(f"unknown cluster host: {host}")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("preflight repository must be immutable bundle source")
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock, roots=dataset_roots)
    control = validate_r0_control_binding(r0_control)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    source = source_provenance(repository)

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    device = str(mx.default_device())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise CampaignError("R2 MLX production requires Apple Silicon macOS")
    if "gpu" not in device.lower():
        raise CampaignError("R2 MLX production requires the MLX GPU device")
    if source["v2_source_blake3"] != approval["identity"]["mlx_source_blake3"]:
        raise CampaignError("preflight source differs from authorized immutable source")

    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "host": host,
        "bundle_id": bundle_manifest["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "corpus_lock_id": lock["lock_id"],
        "r0_control_binding_id": control["binding_id"],
        "source_v2_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": file_blake3(
            bundle / "bin" / EXPORTER_BINARY
        ),
        "protocol_blake3": approval["identity"]["protocol_blake3"],
        "model_parameter_counts": architecture_parameter_counts(),
        "dataset_manifest_blake3": [
            entry["manifest_blake3"] for entry in lock["identity"]["datasets"]
        ],
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "device": device,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "preflight_id": canonical_blake3(scientific_identity),
        "scientific_identity": scientific_identity,
        "operational": {
            "hostname": socket.gethostname().split(".")[0],
            "repository": str(repository),
            "bundle": str(bundle),
            "dataset_roots": [str(path.resolve()) for path in dataset_roots],
        },
        "checks": {
            "bundle_verified": True,
            "authorization_verified": True,
            "corpus_verified": True,
            "r0_control_verified": True,
            "all_shards_checksummed": True,
            "apple_silicon_verified": True,
            "mlx_gpu_verified": True,
            "production_training_started": False,
        },
    }


def validate_preflight(
    path: Path,
    *,
    host: str,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    authorization: Path,
) -> dict[str, Any]:
    report = _read_json(path, "preflight")
    identity = report.get("scientific_identity")
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock)
    control = validate_r0_control_binding(r0_control)
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != report.get("preflight_id")
        or identity.get("host") != host
        or identity.get("bundle_id") != bundle_manifest["bundle_id"]
        or identity.get("authorization_id") != approval["authorization_id"]
        or identity.get("corpus_lock_id") != lock["lock_id"]
        or identity.get("r0_control_binding_id") != control["binding_id"]
        or identity.get("protocol_blake3") != approval["identity"]["protocol_blake3"]
        or identity.get("exporter_executable_blake3")
        != approval["identity"]["exporter_executable_blake3"]
        or not all(
            report.get("checks", {}).get(field) is True
            for field in (
                "bundle_verified",
                "authorization_verified",
                "corpus_verified",
                "r0_control_verified",
                "all_shards_checksummed",
                "apple_silicon_verified",
                "mlx_gpu_verified",
            )
        )
        or report.get("checks", {}).get("production_training_started") is not False
    ):
        raise CampaignError("R2 MLX host preflight is absent, stale, or incomplete")
    return report


def export_cache(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    authorization: Path,
    preflight: Path,
    dataset_roots: list[Path],
    cache_parent: Path,
) -> dict[str, Any]:
    if host != "john1":
        raise CampaignError("the shared R2 cache is exported once on john1")
    if repository.resolve() != bundle.resolve() / "source":
        raise CampaignError("cache export repository must be immutable bundle source")
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    validate_preflight(
        preflight,
        host=host,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
        authorization=authorization,
    )
    cache_parent.mkdir(parents=True, exist_ok=True)
    binding_path = cache_parent / "cache-binding.json"
    if binding_path.exists():
        binding = _read_json(binding_path, "existing cache binding")
        cache = _validate_cache_binding(
            binding,
            cache_parent=cache_parent,
            corpus_lock=corpus_lock,
            authorization=approval,
        )
        return _cache_export_report(binding, cache, reused=True)

    receipt = cache_parent / "cache-export-receipt.json"
    command = [
        str(bundle.resolve() / "bin" / EXPORTER_BINARY),
        "export-mlx",
        "--corpus-lock",
        str(corpus_lock.resolve()),
        *[
            value
            for root in dataset_roots
            for value in ("--dataset-root", str(root.resolve()))
        ],
        "--output-root",
        str(cache_parent.resolve()),
        "--receipt",
        str(receipt.resolve()),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CampaignError(
            "R2 MLX cache export failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    export = _read_json(receipt, "cache export receipt")
    cache_id = export.get("cache_id")
    if export.get("experiment_id") != EXPERIMENT_ID or not _is_digest(cache_id):
        raise CampaignError("R2 MLX cache export receipt is invalid")
    cache = R2SparseMlxCache(
        cache_parent / str(cache_id),
        corpus_lock=corpus_lock,
    )
    identity = {
        "cache_id": cache.cache_id,
        "cache_manifest_blake3": file_blake3(cache.manifest_path),
        "corpus_lock_id": cache.corpus_lock_id,
        "authorization_id": approval["authorization_id"],
        "exporter_executable_blake3": cache.exporter_executable_blake3,
        "identity_semantic_blake3": cache.identity_semantic_blake3,
        "d6_semantic_blake3": cache.d6_semantic_blake3,
        "target_blake3": cache.target_blake3,
    }
    binding = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "binding_id": canonical_blake3(identity),
        "identity": identity,
    }
    _write_once(binding_path, binding, "cache binding")
    return _cache_export_report(binding, cache, reused=False)


def run_role(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    authorization: Path,
    preflight: Path,
    cache_parent: Path,
    run_role_id: str,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if run_role_id not in AUTHORIZED_RUNS:
        raise CampaignError(f"unknown R2 MLX run role: {run_role_id}")
    expected_host = RUN_HOSTS[run_role_id]
    resolved_host = detect_host(repository) if host == "auto" else host
    if resolved_host != expected_host:
        raise CampaignError(
            f"{run_role_id} is preregistered on {expected_host}, not {resolved_host}"
        )
    if repository.resolve() != bundle.resolve() / "source":
        raise CampaignError("run repository must be immutable bundle source")
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    validate_preflight(
        preflight,
        host=resolved_host,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
        authorization=authorization,
    )
    binding = _read_json(cache_parent / "cache-binding.json", "cache binding")
    cache = _validate_cache_binding(
        binding,
        cache_parent=cache_parent,
        corpus_lock=corpus_lock,
        authorization=approval,
    )
    if output.exists():
        existing = _read_json(output, "existing run report")
        _validate_run_report(existing, run_role_id, approval)
        return existing
    return run_tournament(
        R2SparseMlxTournamentConfig(
            cache=cache.root,
            corpus_lock=corpus_lock,
            run_dir=run_dir / "training",
            output=output,
            authorization=authorization,
            r0_control=r0_control,
            run_role=run_role_id,
            resume=(run_dir / "training/latest.json").exists(),
        )
    )


def build_task_specs(
    *,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    authorization: Path,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> list[dict[str, Any]]:
    """Return a scheduler-compatible graph without installing or mutating it."""
    repository = repository.resolve()
    bundle = bundle.resolve()
    validate_bundle_for_campaign(bundle)
    validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    bundle_relative = _relative_to_repository(repository, bundle, "bundle")
    lock_relative = _relative_to_repository(repository, corpus_lock, "corpus lock")
    r0_relative = _relative_to_repository(repository, r0_control, "R0 control")
    authorization_relative = _relative_to_repository(
        repository,
        authorization,
        "authorization",
    )
    experiment_relative = _relative_to_repository(
        repository,
        experiment_root,
        "experiment root",
    )
    if len(
        {
            corpus_lock.resolve().parent,
            r0_control.resolve().parent,
            authorization.resolve().parent,
        }
    ) != 1:
        raise CampaignError("corpus, R0 control, and authorization must share control/")
    control_relative = lock_relative.parent

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    control_fanout_id = f"{TASK_PREFIX}-control-fanout"
    specs = [
        _task(
            task_id=bundle_fanout_id,
            title="Fan out immutable R2 MLX bundle",
            decision="Install byte-identical source and exporter on all four hosts",
            workload_class="shared-prerequisite",
            priority=1,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=[
                ".venv/bin/python",
                "-B",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{bundle_relative}/",
                "--local-root",
                str(bundle_relative),
                *[
                    value
                    for host in HOSTS[1:]
                    for value in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / bundle_relative}/",
                    )
                ],
                "--required-file",
                "bundle.json",
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/bundle-fanout.json"),
            ],
            artifact_path=str(experiment_relative / "reports/bundle-fanout.json"),
            stop_rule="Every bundle byte must match the immutable manifest.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
        _task(
            task_id=control_fanout_id,
            title="Fan out R2 MLX frozen controls",
            decision="Install corpus, R0 binding, and authorization on every host",
            workload_class="shared-prerequisite",
            priority=2,
            expected_runtime_seconds=30,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[bundle_fanout_id],
            command=[
                ".venv/bin/python",
                "-B",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{control_relative}/",
                "--local-root",
                str(control_relative),
                *[
                    value
                    for host in HOSTS[1:]
                    for value in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / control_relative}/",
                    )
                ],
                "--required-file",
                lock_relative.name,
                "--required-file",
                r0_relative.name,
                "--required-file",
                authorization_relative.name,
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/control-fanout.json"),
            ],
            artifact_path=str(experiment_relative / "reports/control-fanout.json"),
            stop_rule="Every host must receive the same three immutable controls.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
    ]

    preflight_ids: list[str] = []
    for host in HOSTS:
        preflight_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(preflight_id)
        local_preflight = experiment_relative / "reports/preflight-local.json"
        specs.append(
            _task(
                task_id=preflight_id,
                title=f"Preflight R2 MLX on {host}",
                decision="Prove source, corpus, R0 control, and MLX GPU readiness",
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[bundle_fanout_id, control_fanout_id],
                command=[
                    *_frozen_python_command(host, bundle_relative),
                    "preflight",
                    "--host",
                    host,
                    "--repository",
                    str(REMOTE_ROOTS[host] / bundle_relative / "source"),
                    "--bundle",
                    str(REMOTE_ROOTS[host] / bundle_relative),
                    "--corpus-lock",
                    str(REMOTE_ROOTS[host] / lock_relative),
                    "--r0-control",
                    str(REMOTE_ROOTS[host] / r0_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    *_dataset_flags(host, dataset_root),
                    "--output",
                    str(REMOTE_ROOTS[host] / local_preflight),
                ],
                artifact_path=str(local_preflight),
                stop_rule="No cache export or optimizer step may start before preflight.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=True,
            )
        )

    cache_export_id = f"{TASK_PREFIX}-export-cache"
    cache_report = experiment_relative / "reports/cache-export.json"
    specs.append(
        _task(
            task_id=cache_export_id,
            title="Export one exact shared R2 MLX cache",
            decision="Materialize the Rust-authored cache once for all four runs",
            workload_class="shared-prerequisite",
            priority=8,
            expected_runtime_seconds=2400,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=preflight_ids,
            command=[
                *_frozen_python_command("john1", bundle_relative),
                "export-cache",
                "--host",
                "john1",
                "--repository",
                str(REMOTE_ROOTS["john1"] / bundle_relative / "source"),
                "--bundle",
                str(REMOTE_ROOTS["john1"] / bundle_relative),
                "--corpus-lock",
                str(REMOTE_ROOTS["john1"] / lock_relative),
                "--r0-control",
                str(REMOTE_ROOTS["john1"] / r0_relative),
                "--authorization",
                str(REMOTE_ROOTS["john1"] / authorization_relative),
                "--preflight",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/preflight-local.json"
                ),
                *_dataset_flags("john1", dataset_root),
                "--cache-parent",
                str(REMOTE_ROOTS["john1"] / experiment_relative / "caches"),
                "--output",
                str(REMOTE_ROOTS["john1"] / cache_report),
            ],
            artifact_path=str(cache_report),
            stop_rule="Cache export must prove exact no-truncation and all D6 semantics.",
            cpu_cores=8,
            memory_gib=12.0,
            uses_mlx=False,
        )
    )

    cache_fanout_id = f"{TASK_PREFIX}-cache-fanout"
    specs.append(
        _task(
            task_id=cache_fanout_id,
            title="Fan out exact R2 MLX cache",
            decision="Install the one content-addressed cache on all training hosts",
            workload_class="shared-prerequisite",
            priority=9,
            expected_runtime_seconds=300,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[cache_export_id],
            command=[
                ".venv/bin/python",
                "-B",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{experiment_relative / 'caches'}/",
                "--local-root",
                str(experiment_relative / "caches"),
                *[
                    value
                    for host in HOSTS[1:]
                    for value in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / experiment_relative / 'caches'}/",
                    )
                ],
                "--required-file",
                "cache-binding.json",
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/cache-fanout.json"),
            ],
            artifact_path=str(experiment_relative / "reports/cache-fanout.json"),
            stop_rule="All hosts must receive the byte-identical content-addressed cache.",
            cpu_cores=2,
            memory_gib=2.0,
            uses_mlx=False,
        )
    )

    run_task_ids: list[str] = []
    for role in AUTHORIZED_RUNS:
        host = RUN_HOSTS[role]
        task_id = f"{TASK_PREFIX}-run-{role}"
        run_task_ids.append(task_id)
        report = experiment_relative / "runs" / role / "report.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Run R2 MLX {role}",
                decision=(
                    f"Measure {RUN_ARCHITECTURES[role]}"
                    if role != "set-replay"
                    else "Independently replay the Set Transformer protocol"
                ),
                workload_class=(
                    "replica" if role == "set-replay" else "independent-experiment"
                ),
                priority=10,
                expected_runtime_seconds=3600,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[
                    cache_fanout_id,
                    f"{TASK_PREFIX}-preflight-{host}",
                ],
                command=[
                    *_frozen_python_command(host, bundle_relative),
                    "run-role",
                    "--host",
                    "auto",
                    "--repository",
                    str(REMOTE_ROOTS[host] / bundle_relative / "source"),
                    "--bundle",
                    str(REMOTE_ROOTS[host] / bundle_relative),
                    "--corpus-lock",
                    str(REMOTE_ROOTS[host] / lock_relative),
                    "--r0-control",
                    str(REMOTE_ROOTS[host] / r0_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    "--preflight",
                    str(
                        REMOTE_ROOTS[host]
                        / experiment_relative
                        / "reports/preflight-local.json"
                    ),
                    "--cache-parent",
                    str(REMOTE_ROOTS[host] / experiment_relative / "caches"),
                    "--run-role",
                    role,
                    "--run-dir",
                    str(REMOTE_ROOTS[host] / experiment_relative / "runs" / role),
                    "--output",
                    str(REMOTE_ROOTS[host] / report),
                ],
                artifact_path=str(report),
                stop_rule="Complete 500 matched steps and all metric/performance evidence.",
                cpu_cores=8,
                memory_gib=12.0,
                uses_mlx=True,
            )
        )

    collection_id = f"{TASK_PREFIX}-collect-reports"
    collection = experiment_relative / "reports/collection.json"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect four R2 MLX run reports",
            decision="Retrieve the three architectures and independent replay",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=120,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=run_task_ids,
            command=[
                *_frozen_python_command("john1", bundle_relative),
                "collect-reports",
                "--bundle",
                str(REMOTE_ROOTS["john1"] / bundle_relative),
                "--corpus-lock",
                str(REMOTE_ROOTS["john1"] / lock_relative),
                "--r0-control",
                str(REMOTE_ROOTS["john1"] / r0_relative),
                "--authorization",
                str(REMOTE_ROOTS["john1"] / authorization_relative),
                "--experiment-root",
                str(REMOTE_ROOTS["john1"] / experiment_relative),
                "--destination",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/collected"
                ),
                "--output",
                str(REMOTE_ROOTS["john1"] / collection),
            ],
            artifact_path=str(collection),
            stop_rule="Exactly four authorized reports must be retrieved and verified.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classifier_ids: list[str] = []
    for order in ("forward", "reverse"):
        task_id = f"{TASK_PREFIX}-classify-{order}"
        classifier_ids.append(task_id)
        output = experiment_relative / f"reports/classification-{order}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Classify R2 MLX in {order} order",
                decision="Apply matched-capacity, R0-reference, and replay gates",
                workload_class="replica",
                priority=40,
                expected_runtime_seconds=30,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    *_frozen_report_command("john1", bundle_relative),
                    "--collection",
                    str(REMOTE_ROOTS["john1"] / collection),
                    "--r0-control",
                    str(REMOTE_ROOTS["john1"] / r0_relative),
                    "--order",
                    order,
                    "--output",
                    str(REMOTE_ROOTS["john1"] / output),
                ],
                artifact_path=str(output),
                stop_rule="Classification must be deterministic and non-promotional.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )

    comparison = experiment_relative / "reports/classification-order-proof.json"
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-classification-order-proof",
            title="Prove R2 MLX classification order invariance",
            decision="Require byte-identical forward and reverse outputs",
            workload_class="replica",
            priority=50,
            expected_runtime_seconds=10,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=classifier_ids,
            command=[
                *_frozen_python_command("john1", bundle_relative),
                "compare-classifications",
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
                str(REMOTE_ROOTS["john1"] / comparison),
            ],
            artifact_path=str(comparison),
            stop_rule="Forward and reverse classifier bytes must match exactly.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    _validate_task_specs(specs)
    return specs


def collect_reports(
    *,
    authorization: Path,
    bundle: Path,
    corpus_lock: Path,
    r0_control: Path,
    experiment_root: Path,
    destination: Path,
) -> dict[str, Any]:
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
        r0_control=r0_control,
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        experiment_relative = experiment_root.resolve().relative_to(
            REMOTE_ROOTS["john1"]
        )
    except ValueError as error:
        raise CampaignError(
            "collection experiment root must be beneath the john1 repository"
        ) from error
    collected = []
    for role in AUTHORIZED_RUNS:
        host = RUN_HOSTS[role]
        artifact = Path("runs") / role / "report.json"
        target = destination / f"{role}.json"
        _retrieve_artifact(
            host,
            REMOTE_ROOTS[host] / experiment_relative / artifact,
            target,
        )
        report = _read_json(target, f"{role} report")
        _validate_run_report(report, role, approval)
        collected.append(
            {
                "run_role": role,
                "architecture": RUN_ARCHITECTURES[role],
                "host": host,
                "source_artifact": str(artifact),
                "file": str(target.resolve()),
                "blake3": file_blake3(target),
                "report_id": report["report_id"],
            }
        )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "authorization_id": approval["authorization_id"],
        "reports": [
            {
                "run_role": entry["run_role"],
                "blake3": entry["blake3"],
                "report_id": entry["report_id"],
            }
            for entry in collected
        ],
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "collection_id": canonical_blake3(identity),
        "scientific_identity": identity,
        "reports": collected,
        "claims": {
            "all_reports_collected": True,
            "promotion_authorized": False,
        },
    }


def compare_classifications(forward: Path, reverse: Path) -> dict[str, Any]:
    try:
        forward_bytes = forward.read_bytes()
        reverse_bytes = reverse.read_bytes()
    except OSError as error:
        raise CampaignError(f"cannot read classifier output: {error}") from error
    if forward_bytes != reverse_bytes:
        raise CampaignError("forward and reverse R2 MLX classifications differ")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "classification_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "byte_identical": True,
        "promotion_authorized": False,
    }


def detect_host(repository: Path) -> str:
    resolved = repository.resolve()
    matches = []
    for host, root in REMOTE_ROOTS.items():
        try:
            resolved.relative_to(root)
            matches.append(host)
        except ValueError:
            pass
    if len(matches) != 1:
        raise CampaignError(f"cannot map repository path to one cluster host: {resolved}")
    return matches[0]


def _validate_dataset_part(
    part: DatasetPart,
    root: Path,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> None:
    expected_cards = {
        wildlife: "A" for wildlife in ("bear", "elk", "salmon", "hawk", "fox")
    }
    game = manifest.get("game")
    provenance = manifest.get("provenance")
    if (
        manifest_digest != part.manifest_blake3
        or manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != part.dataset_id
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("target_schema") != TARGET_SCHEMA
        or manifest.get("record_size") != 864
        or manifest.get("split") != part.split
        or manifest.get("strategy") != STRATEGY_ID
        or manifest.get("first_game_index") != part.first_game_index
        or manifest.get("requested_games") != part.games
        or manifest.get("completed_games") != part.games
        or manifest.get("total_records") != part.records
        or not isinstance(game, dict)
        or game.get("player_count") != 4
        or game.get("mode") != "Standard"
        or game.get("scoring_cards") != expected_cards
        or game.get("habitat_bonuses") is not False
        or not isinstance(provenance, dict)
    ):
        raise CampaignError(f"dataset part violates the frozen foundation: {root}")
    _require_digest(provenance.get("v2_source_blake3"), "dataset source digest")


def _validate_cache_binding(
    binding: dict[str, Any],
    *,
    cache_parent: Path,
    corpus_lock: Path,
    authorization: dict[str, Any],
) -> R2SparseMlxCache:
    identity = binding.get("identity")
    if (
        binding.get("schema_version") != 1
        or binding.get("experiment_id") != EXPERIMENT_ID
        or binding.get("adr") != ADR_ID
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != binding.get("binding_id")
        or identity.get("authorization_id") != authorization["authorization_id"]
        or identity.get("corpus_lock_id")
        != authorization["identity"]["corpus_lock_id"]
        or identity.get("exporter_executable_blake3")
        != authorization["identity"]["exporter_executable_blake3"]
        or not _is_digest(identity.get("cache_id"))
    ):
        raise CampaignError("shared R2 cache binding is invalid")
    cache = R2SparseMlxCache(
        cache_parent / identity["cache_id"],
        corpus_lock=corpus_lock,
    )
    if (
        file_blake3(cache.manifest_path) != identity.get("cache_manifest_blake3")
        or cache.identity_semantic_blake3
        != identity.get("identity_semantic_blake3")
        or cache.d6_semantic_blake3 != identity.get("d6_semantic_blake3")
        or cache.target_blake3 != identity.get("target_blake3")
    ):
        raise CampaignError("shared R2 cache bytes differ from their binding")
    return cache


def _cache_export_report(
    binding: dict[str, Any],
    cache: R2SparseMlxCache,
    *,
    reused: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "cache_binding_id": binding["binding_id"],
        "cache_id": cache.cache_id,
        "cache_root": str(cache.root.resolve()),
        "cache_manifest_blake3": file_blake3(cache.manifest_path),
        "reused": reused,
        "production_training_started": False,
    }


def _validate_run_report(
    report: dict[str, Any],
    role: str,
    authorization: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("run_role") != role
        or report.get("architecture") != RUN_ARCHITECTURES[role]
        or not isinstance(report.get("scientific_identity"), dict)
        or report["scientific_identity"] != report_scientific_identity(report)
        or canonical_blake3(report["scientific_identity"]) != report.get("report_id")
        or report.get("authorization", {}).get("authorization_id")
        != authorization["authorization_id"]
        or report.get("integrity", {}).get("all_metrics_finite") is not True
        or report.get("claims", {}).get("promotion_authorized") is not False
    ):
        raise CampaignError(f"collected R2 MLX report is invalid: {role}")


def _frozen_python_command(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    return [
        "/usr/bin/env",
        "-C",
        str(root / bundle_relative / "source"),
        "PYTHONPATH=python:tools",
        str(root / ".venv/bin/python"),
        "-B",
        "tools/r2_sparse_mlx_campaign.py",
    ]


def _frozen_report_command(host: str, bundle_relative: Path) -> list[str]:
    command = _frozen_python_command(host, bundle_relative)
    command[-1] = "tools/r2_sparse_mlx_report.py"
    return command


def _dataset_flags(host: str, dataset_root: Path) -> list[str]:
    return [
        value
        for part in dataset_parts(dataset_root)
        for value in ("--dataset-root", str(REMOTE_ROOTS[host] / part.root))
    ]


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


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("R2 MLX task graph contains duplicate IDs")
    known = set(identifiers)
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(
                f"task {spec['id']} has unknown dependencies: {sorted(unknown)}"
            )
        command_text = "\0".join(spec["command"])
        if "cluster_research_queue" in command_text:
            raise CampaignError("inert R2 task graph may not mutate the live queue")
        if "tools/r2_sparse_mlx_" in command_text and "-B" not in spec["command"]:
            raise CampaignError("every frozen R2 Python command must use -B")
    for role, host in RUN_HOSTS.items():
        task = next(spec for spec in specs if spec["id"] == f"{TASK_PREFIX}-run-{role}")
        if task["compatible_hosts"] != [host]:
            raise CampaignError(f"{role} host assignment drifted")


def _relative_to_repository(repository: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as error:
        raise CampaignError(f"{label} must remain beneath the repository") from error
    if not relative.parts:
        raise CampaignError(f"{label} cannot be the repository root")
    return relative


def _retrieve_artifact(host: str, source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if host == "john1":
        try:
            shutil.copyfile(source, temporary)
        except OSError as error:
            raise CampaignError(f"cannot collect local report {source}: {error}") from error
    else:
        completed = subprocess.run(
            ["scp", f"{host}:{source}", str(temporary)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"cannot collect report from {host}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
    os.replace(temporary, destination)


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    os.replace(temporary, path)


def _write_once(path: Path, value: object, label: str) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists():
        try:
            existing = path.read_text()
        except OSError as error:
            raise CampaignError(f"cannot read existing {label}: {error}") from error
        if existing != encoded:
            raise CampaignError(f"refusing to overwrite a different {label}: {path}")
        return
    _write_json_atomic(path, value)


def _require_digest(value: object, label: str) -> str:
    if not _is_digest(value):
        raise CampaignError(f"{label} must be a lowercase 64-character digest")
    return str(value)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-corpus")
    freeze.add_argument("--dataset-root", type=Path, action="append")
    freeze.add_argument("--dataset-prefix", type=Path, default=DEFAULT_DATASET_ROOT)
    freeze.add_argument("--output", type=Path, default=DEFAULT_CORPUS_LOCK)

    bind = subparsers.add_parser("bind-r0-control")
    bind.add_argument(
        "--classification-forward",
        type=Path,
        default=DEFAULT_R0_CLASSIFICATION_FORWARD,
    )
    bind.add_argument(
        "--classification-reverse",
        type=Path,
        default=DEFAULT_R0_CLASSIFICATION_REVERSE,
    )
    bind.add_argument("--order-proof", type=Path, default=DEFAULT_R0_ORDER_PROOF)
    bind.add_argument("--collection", type=Path, default=DEFAULT_R0_COLLECTION)
    bind.add_argument("--output", type=Path, default=DEFAULT_R0_BINDING)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--bundle", type=Path, required=True)
    authorize.add_argument("--corpus-lock", type=Path, default=DEFAULT_CORPUS_LOCK)
    authorize.add_argument("--r0-control", type=Path, default=DEFAULT_R0_BINDING)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--output", type=Path, default=DEFAULT_AUTHORIZATION)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--host", required=True)
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--corpus-lock", type=Path, required=True)
    preflight.add_argument("--r0-control", type=Path, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--dataset-root", type=Path, action="append", required=True)
    preflight.add_argument("--output", type=Path, required=True)

    export = subparsers.add_parser("export-cache")
    export.add_argument("--host", required=True)
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--bundle", type=Path, required=True)
    export.add_argument("--corpus-lock", type=Path, required=True)
    export.add_argument("--r0-control", type=Path, required=True)
    export.add_argument("--authorization", type=Path, required=True)
    export.add_argument("--preflight", type=Path, required=True)
    export.add_argument("--dataset-root", type=Path, action="append", required=True)
    export.add_argument("--cache-parent", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run-role")
    run.add_argument("--host", required=True)
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--corpus-lock", type=Path, required=True)
    run.add_argument("--r0-control", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--cache-parent", type=Path, required=True)
    run.add_argument("--run-role", choices=AUTHORIZED_RUNS, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--corpus-lock", type=Path, default=DEFAULT_CORPUS_LOCK)
    queue.add_argument("--r0-control", type=Path, default=DEFAULT_R0_BINDING)
    queue.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    queue.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    queue.add_argument("--dataset-prefix", type=Path, default=DEFAULT_DATASET_ROOT)
    queue.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect-reports")
    collect.add_argument("--bundle", type=Path, required=True)
    collect.add_argument("--corpus-lock", type=Path, required=True)
    collect.add_argument("--r0-control", type=Path, required=True)
    collect.add_argument("--authorization", type=Path, required=True)
    collect.add_argument("--experiment-root", type=Path, required=True)
    collect.add_argument("--destination", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-classifications")
    compare.add_argument("--forward", type=Path, required=True)
    compare.add_argument("--reverse", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze-corpus":
            report = freeze_corpus(args.dataset_root, dataset_root=args.dataset_prefix)
            _write_once(args.output, report, "corpus lock")
        elif args.command == "bind-r0-control":
            report = bind_r0_control(
                classification_forward=args.classification_forward,
                classification_reverse=args.classification_reverse,
                order_proof=args.order_proof,
                collection=args.collection,
            )
            _write_once(args.output, report, "R0 control binding")
        elif args.command == "authorize":
            report = create_authorization(
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                approved_by=args.approved_by,
            )
            _write_once(args.output, report, "authorization")
        elif args.command == "preflight":
            report = run_preflight(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                authorization=args.authorization,
                dataset_roots=args.dataset_root,
            )
            _write_json_atomic(args.output, report)
        elif args.command == "export-cache":
            report = export_cache(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                authorization=args.authorization,
                preflight=args.preflight,
                dataset_roots=args.dataset_root,
                cache_parent=args.cache_parent,
            )
            _write_json_atomic(args.output, report)
        elif args.command == "run-role":
            report = run_role(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                authorization=args.authorization,
                preflight=args.preflight,
                cache_parent=args.cache_parent,
                run_role_id=args.run_role,
                run_dir=args.run_dir,
                output=args.output,
            )
        elif args.command == "queue-spec":
            specs = build_task_specs(
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                authorization=args.authorization,
                experiment_root=args.experiment_root,
                dataset_root=args.dataset_prefix,
            )
            report = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "adr": ADR_ID,
                "inert": True,
                "live_queue_modified": False,
                "task_count": len(specs),
                "task_spec_blake3": canonical_blake3(specs),
                "tasks": specs,
            }
            _write_json_atomic(args.output, report)
        elif args.command == "collect-reports":
            report = collect_reports(
                authorization=args.authorization,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                r0_control=args.r0_control,
                experiment_root=args.experiment_root,
                destination=args.destination,
            )
            _write_json_atomic(args.output, report)
        else:
            report = compare_classifications(args.forward, args.reverse)
            _write_json_atomic(args.output, report)
    except CampaignError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
