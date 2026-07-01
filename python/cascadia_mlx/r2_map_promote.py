"""Fail-closed, crash-recoverable R2-MAP campaign promotion.

Promotion consumes exactly two centrally registered objects: one independently
verified checkpoint and one immutable fixed-250 focal aggregate.  It never
scans a run directory or selects among checkpoints.  The campaign's incumbent
and promoted pointers live in one atomic pointer bundle so readers can never
observe those roles disagreeing.

The transaction deliberately commits campaign state before the pointer bundle.
Generation requires both to agree, so a crash at that boundary fails closed.
Re-running the same transaction repairs the bundle from the immutable journal.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.checkpoint import (
    CheckpointError,
    verify_loss_stream_prefix,
    verify_r2_map_checkpoint_files,
)
from cascadia_mlx.r2_map_contracts import (
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    DEFAULT_STORAGE_CONTRACT,
    ContractError,
    Phase,
    StorageContract,
    TransitionError,
    canonical_campaign_path,
    canonical_json_bytes,
    preflight_storage,
    read_state,
    reject_frozen_campaign_path,
    require_local_storage_authority,
    transition_state,
    utc_now,
    write_state,
)
from cascadia_mlx.r2_map_verify import validate_verification_receipt

CANDIDATE_REGISTRATION_SCHEMA = "cascadia.r2-map.candidate-registration.v1"
GATE_REGISTRATION_SCHEMA = "cascadia.r2-map.promotion-gate-registration.v1"
MODEL_POINTERS_SCHEMA = "cascadia.r2-map.model-pointers.v1"
PROMOTION_HISTORY_SCHEMA = "cascadia.r2-map.promotion-history.v1"
OPPONENT_POOL_SCHEMA = "cascadia.r2-map.opponent-pool.v1"
PROMOTION_TRANSACTION_SCHEMA = "cascadia.r2-map.promotion-transaction.v1"
PROMOTION_OUTCOME_SCHEMA = "cascadia.r2-map.promotion-outcome.v1"
FOCAL_REPORT_SCHEMA = "cascadia.r2-map.focal-report.v4"
FOCAL_CONTRACT_SCHEMA = "cascadia.r2-map.focal-contract.v4"
OPPONENT_FIELD_SCHEMA = "cascadia.r2-map.opponent-field.v4"
FOCAL_PROTOCOL = "r2-map-focal-paired-v1"
FIXED_PAIR_COUNT = 250
FIXED_PHYSICAL_GAMES = 500
GREEDY_POLICY_ID = "greedy-v1"

HARD_GATE_NAMES = (
    "checkpoint",
    "fixed_sample",
    "identity",
    "replay",
    "legality",
    "score_accounting",
    "pinecone_conservation",
    "resource",
    "memory",
    "zero_swap",
    "clean_shutdown",
)


class PromotionError(ContractError):
    """Promotion evidence or durable state failed closed."""


CheckpointVerifier = Callable[..., tuple[dict[str, Any], Any, Any]]
ReceiptVerifier = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class PromotionValidators:
    checkpoint: CheckpointVerifier = verify_r2_map_checkpoint_files
    receipt: ReceiptVerifier = validate_verification_receipt


DEFAULT_VALIDATORS = PromotionValidators()


@dataclass(frozen=True)
class PromotionPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "control/campaign-state.json"

    @property
    def model_pointers(self) -> Path:
        return self.root / "control/incumbent-promoted-pointers.json"

    @property
    def promotion_lock(self) -> Path:
        return self.root / "control/.promotion.lock"

    def candidate_registration(self, logical_id: str) -> Path:
        return self.root / "control/candidate-registry" / f"{logical_id}.json"

    def gate_registration(self, round_id: str) -> Path:
        return self.root / "control/promotion-gates" / f"{round_id}.json"

    def transaction(self, round_id: str) -> Path:
        return self.root / "control/promotion-transactions" / f"{round_id}.json"

    def outcome(self, round_id: str) -> Path:
        return self.root / "control/promotion-outcomes" / f"{round_id}.json"

    def history(self, promotion_id: str) -> Path:
        return self.root / "opponent-pool/promotions" / f"{promotion_id}.json"

    def pool(self, pool_id: str) -> Path:
        return self.root / "opponent-pool" / f"{pool_id}.json"


def register_verified_candidate(
    *,
    campaign_root: str | Path,
    checkpoint_path: str | Path,
    run_dir: str | Path,
    verification_receipt_path: str | Path,
    logical_candidate_id: str,
    round_index: int | None,
    benchmark_id: str,
    validators: PromotionValidators = DEFAULT_VALIDATORS,
    now: str | None = None,
) -> Path:
    """Register one explicit verified checkpoint without checkpoint discovery."""
    paths = _paths(campaign_root)
    checkpoint_path = _contained(Path(checkpoint_path), paths.root, "candidate checkpoint")
    run_dir = _contained(Path(run_dir), paths.root, "candidate run")
    verification_path = _contained(
        Path(verification_receipt_path), paths.root, "candidate verification receipt"
    )
    manifest, checkpoint_state, _ = validators.checkpoint(checkpoint_path)
    receipt = validators.receipt(verification_path, checkpoint_path=checkpoint_path)
    expected_logical_id = "bootstrap-candidate" if round_index is None else f"T[{round_index}]"
    if round_index is not None and (
        not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0
    ):
        raise PromotionError("candidate round index must be null or a nonnegative integer")
    if logical_candidate_id != expected_logical_id or not benchmark_id:
        raise PromotionError("candidate logical, round, or benchmark identity is malformed")
    if checkpoint_path.parent != run_dir / "checkpoints":
        raise PromotionError("registered checkpoint is not inside the registered run/checkpoints")
    if manifest.get("checkpoint_id") != checkpoint_path.name:
        raise PromotionError("registered checkpoint manifest names another checkpoint")
    loss_binding = _verify_checkpoint_loss_binding(run_dir, checkpoint_state)
    registration: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": CANDIDATE_REGISTRATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "logical_candidate_id": logical_candidate_id,
        "round_index": round_index,
        "benchmark_id": benchmark_id,
        "run_dir": _relative(run_dir, paths.root),
        "checkpoint_path": _relative(checkpoint_path, paths.root),
        "checkpoint_id": checkpoint_path.name,
        "checkpoint_manifest_sha256": _file_sha256(checkpoint_path / "checkpoint.json"),
        "checkpoint_manifest_blake3": _file_blake3(checkpoint_path / "checkpoint.json"),
        "checkpoint_identity_blake3": manifest.get("manifest_identity_blake3"),
        "loss_stream_path": loss_binding["path"],
        "loss_stream_offset_bytes": loss_binding["offset_bytes"],
        "loss_stream_prefix_blake3": loss_binding["prefix_blake3"],
        "verification_receipt_path": _relative(verification_path, paths.root),
        "verification_receipt_blake3": _file_blake3(verification_path),
        "verification_id": receipt.get("verification_id"),
        "registered_at": now or utc_now(),
    }
    _require_blake3(registration["checkpoint_identity_blake3"], "checkpoint identity")
    _require_blake3(registration["verification_id"], "verification id")
    registration["registration_blake3"] = _content_blake3(registration, "registration_blake3")
    destination = paths.candidate_registration(logical_candidate_id)
    _write_immutable_json(destination, registration)
    return destination


def register_fixed_250_gate(
    *,
    campaign_root: str | Path,
    candidate_registration_path: str | Path,
    focal_report_path: str | Path,
    focal_contract_path: str | Path,
    opponent_field_path: str | Path,
    gate_results: Mapping[str, bool],
    resource_limits: Mapping[str, int | bool],
    opponent_pool_manifest_blake3: str | None,
    validators: PromotionValidators = DEFAULT_VALIDATORS,
    now: str | None = None,
) -> Path:
    """Bind the single fixed analysis and all preregistered hard gates."""
    paths = _paths(campaign_root)
    candidate_path = _contained(
        Path(candidate_registration_path), paths.root, "candidate registration"
    )
    candidate = _load_candidate_registration(candidate_path, paths, validators)
    report_path = _contained(Path(focal_report_path), paths.root, "fixed-250 focal report")
    report = _read_json(report_path, "fixed-250 focal report")
    contract_path = _contained(Path(focal_contract_path), paths.root, "focal contract")
    opponent_field_path = _contained(Path(opponent_field_path), paths.root, "focal opponent field")
    contract = _read_json(contract_path, "focal contract")
    opponent_field = _read_json(opponent_field_path, "focal opponent field")
    state = read_state(paths.state)
    expected_control = (
        GREEDY_POLICY_ID
        if state["phase"] == Phase.BOOTSTRAP_CANDIDATE_GATE.value
        else state.get("incumbent_checkpoint_id")
    )
    _validate_state_binding(
        state,
        candidate,
        {"control_checkpoint_id": expected_control},
    )
    pointers = _load_current_pointers(paths, state)
    observed_pool_hash = None if pointers is None else pointers["opponent_pool"]["manifest_blake3"]
    if opponent_pool_manifest_blake3 != observed_pool_hash:
        raise PromotionError("gate registration does not bind the current opponent pool")
    allowed_opponents = (
        {GREEDY_POLICY_ID}
        if pointers is None
        else {entry["policy_id"] for entry in _load_pool_from_pointers(paths, pointers)["entries"]}
    )
    _validate_focal_contract_and_field(
        report,
        contract,
        opponent_field,
        contract_sha256=_file_sha256(contract_path),
        opponent_field_sha256=_file_sha256(opponent_field_path),
        benchmark_id=candidate["benchmark_id"],
        candidate_id=candidate["logical_candidate_id"],
        control_id=expected_control,
        allowed_opponents=allowed_opponents,
    )
    statistics = _validate_focal_report(
        report,
        candidate_id=candidate["logical_candidate_id"],
        control_id=expected_control,
        benchmark_id=candidate["benchmark_id"],
        gate_results=gate_results,
        resource_limits=resource_limits,
    )
    if set(gate_results) != set(HARD_GATE_NAMES) or not all(
        isinstance(value, bool) for value in gate_results.values()
    ):
        raise PromotionError("promotion gate results must contain the exact boolean gate set")
    if opponent_pool_manifest_blake3 is not None:
        _require_blake3(opponent_pool_manifest_blake3, "opponent pool manifest")
    round_id = _round_id(candidate["round_index"])
    registration: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": GATE_REGISTRATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "round_id": round_id,
        "round_index": candidate["round_index"],
        "benchmark_id": candidate["benchmark_id"],
        "candidate_registration_path": _relative(candidate_path, paths.root),
        "candidate_registration_blake3": candidate["registration_blake3"],
        "candidate_checkpoint_id": candidate["logical_candidate_id"],
        "candidate_checkpoint_manifest_sha256": candidate["checkpoint_manifest_sha256"],
        "control_checkpoint_id": expected_control,
        "opponent_pool_manifest_blake3": opponent_pool_manifest_blake3,
        "opponent_pool_path": (None if pointers is None else pointers["opponent_pool"]["path"]),
        "focal_report_path": _relative(report_path, paths.root),
        "focal_report_blake3": _file_blake3(report_path),
        "focal_contract_path": _relative(contract_path, paths.root),
        "focal_contract_file_blake3": _file_blake3(contract_path),
        "focal_contract_file_sha256": _file_sha256(contract_path),
        "opponent_field_path": _relative(opponent_field_path, paths.root),
        "opponent_field_file_blake3": _file_blake3(opponent_field_path),
        "opponent_field_file_sha256": _file_sha256(opponent_field_path),
        "focal_report_classification": statistics["classification"],
        "gate_results": dict(sorted(gate_results.items())),
        "resource_limits": dict(sorted(resource_limits.items())),
        "registered_at": now or utc_now(),
    }
    registration["registration_blake3"] = _content_blake3(registration, "registration_blake3")
    destination = paths.gate_registration(round_id)
    _write_immutable_json(destination, registration)
    return destination


def apply_registered_gate(
    *,
    campaign_root: str | Path = CAMPAIGN_ROOT,
    candidate_registration_path: str | Path,
    gate_registration_path: str | Path,
    storage_contract: StorageContract | None = None,
    validators: PromotionValidators = DEFAULT_VALIDATORS,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Apply or recover one registered outcome under an exclusive transaction lock."""
    paths = _paths(campaign_root)
    if storage_contract is not None:
        preflight_storage(
            contract=storage_contract,
            configured_paths={
                "campaign_state": paths.state,
                "candidate_registration": Path(candidate_registration_path),
                "gate_registration": Path(gate_registration_path),
                "model_pointers": paths.model_pointers,
            },
        )
    elif paths.root == CAMPAIGN_ROOT.resolve():
        preflight_storage(
            contract=DEFAULT_STORAGE_CONTRACT,
            configured_paths={
                "campaign_state": paths.state,
                "candidate_registration": Path(candidate_registration_path),
                "gate_registration": Path(gate_registration_path),
                "model_pointers": paths.model_pointers,
            },
        )
    candidate_path = _contained(
        Path(candidate_registration_path), paths.root, "candidate registration"
    )
    gate_path = _contained(Path(gate_registration_path), paths.root, "gate registration")
    paths.promotion_lock.parent.mkdir(parents=True, exist_ok=True)
    with paths.promotion_lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _apply_locked(
                paths,
                candidate_path,
                gate_path,
                validators=validators,
                fault_injector=fault_injector,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _apply_locked(
    paths: PromotionPaths,
    candidate_path: Path,
    gate_path: Path,
    *,
    validators: PromotionValidators,
    fault_injector: Callable[[str], None] | None,
) -> dict[str, Any]:
    candidate = _load_candidate_registration(candidate_path, paths, validators)
    gate, report = _load_gate_registration(gate_path, candidate, paths)
    round_id = gate["round_id"]
    transaction_path = paths.transaction(round_id)
    if transaction_path.exists():
        transaction = _read_self_hashed(
            transaction_path,
            PROMOTION_TRANSACTION_SCHEMA,
            "transaction_blake3",
        )
        if (
            transaction["candidate_registration_blake3"] != candidate["registration_blake3"]
            or transaction["gate_registration_blake3"] != gate["registration_blake3"]
        ):
            raise PromotionError("existing promotion transaction binds different evidence")
    else:
        current = read_state(paths.state)
        _validate_state_binding(current, candidate, gate)
        transaction = _prepare_transaction(paths, current, candidate, gate, report)
        _write_immutable_json(transaction_path, transaction)
    _inject(fault_injector, "transaction-durable")
    return _commit_or_recover(paths, transaction, fault_injector=fault_injector)


def _prepare_transaction(
    paths: PromotionPaths,
    current: dict[str, Any],
    candidate: dict[str, Any],
    gate: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    classification = gate["focal_report_classification"]
    target = Phase.INCUMBENT_PROMOTED if classification == "promote" else Phase.CANDIDATE_REJECTED
    if (
        target is Phase.CANDIDATE_REJECTED
        and Phase(current["phase"]) is not Phase.PAIRED_CANDIDATE_GATE
    ):
        raise PromotionError(
            "a bootstrap candidate may only advance after a promote classification"
        )
    timestamp = utc_now()
    proposed = transition_state(
        current,
        target,
        reason=f"registered fixed-250 gate classified {classification}",
        now=timestamp,
    )
    previous_pointers = _load_current_pointers(paths, current)
    observed_pool_hash = (
        None if previous_pointers is None else previous_pointers["opponent_pool"]["manifest_blake3"]
    )
    if gate["opponent_pool_manifest_blake3"] != observed_pool_hash:
        raise PromotionError("gate registration binds a different current opponent pool")
    desired_pointers: dict[str, Any] | None = None
    prepared_artifacts: list[dict[str, Any]] = []
    if classification == "promote":
        promotion_id = proposed["incumbent_checkpoint_id"]
        promotion_index = proposed["promotion_index"]
        checkpoint_record = _promoted_checkpoint_record(promotion_id, candidate)
        history = {
            "schema_version": 1,
            "schema_id": PROMOTION_HISTORY_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "promotion_id": promotion_id,
            "promotion_index": promotion_index,
            "source_candidate_id": candidate["logical_candidate_id"],
            "source_round_index": candidate["round_index"],
            "checkpoint": checkpoint_record,
            "gate_registration_blake3": gate["registration_blake3"],
            "focal_report_blake3": gate["focal_report_blake3"],
            "promoted_at": timestamp,
        }
        history["manifest_blake3"] = _content_blake3(history, "manifest_blake3")
        history_path = paths.history(promotion_id)
        _write_immutable_json(history_path, history)

        entries = [_greedy_pool_entry()]
        if previous_pointers is not None:
            prior_pool = _load_pool_from_pointers(paths, previous_pointers)
            entries = list(prior_pool["entries"])
            entries.append(previous_pointers["incumbent"])
        _validate_pool_entries(entries, excluded_checkpoint_id=promotion_id)
        pool_id = f"P[{promotion_index}]"
        pool = {
            "schema_version": 1,
            "schema_id": OPPONENT_POOL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "pool_id": pool_id,
            "promotion_index": promotion_index,
            "excluded_newest_checkpoint_id": promotion_id,
            "entries": entries,
            "created_at": timestamp,
        }
        pool["manifest_blake3"] = _content_blake3(pool, "manifest_blake3")
        pool_path = paths.pool(pool_id)
        _write_immutable_json(pool_path, pool)
        pool_reference = {
            "pool_id": pool_id,
            "path": _relative(pool_path, paths.root),
            "manifest_blake3": pool["manifest_blake3"],
        }
        desired_pointers = {
            "schema_version": 1,
            "schema_id": MODEL_POINTERS_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "revision": promotion_index,
            "campaign_state_sha256": proposed["state_sha256"],
            "incumbent": checkpoint_record,
            "promoted": checkpoint_record,
            "opponent_pool": pool_reference,
            "updated_at": timestamp,
        }
        desired_pointers["pointers_blake3"] = _content_blake3(desired_pointers, "pointers_blake3")
        prepared_artifacts = [
            _artifact_reference(history_path, paths.root),
            _artifact_reference(pool_path, paths.root),
        ]
    transaction: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": PROMOTION_TRANSACTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "round_id": gate["round_id"],
        "classification": classification,
        "candidate_registration_blake3": candidate["registration_blake3"],
        "gate_registration_blake3": gate["registration_blake3"],
        "focal_report_blake3": gate["focal_report_blake3"],
        "expected_state": current,
        "desired_state": proposed,
        "expected_model_pointers_blake3": (
            None if previous_pointers is None else previous_pointers["pointers_blake3"]
        ),
        "desired_model_pointers": desired_pointers,
        "prepared_artifacts": prepared_artifacts,
        "created_at": timestamp,
    }
    transaction["transaction_blake3"] = _content_blake3(transaction, "transaction_blake3")
    return transaction


def _commit_or_recover(
    paths: PromotionPaths,
    transaction: dict[str, Any],
    *,
    fault_injector: Callable[[str], None] | None,
) -> dict[str, Any]:
    for artifact in transaction["prepared_artifacts"]:
        path = _registered_path(artifact.get("path"), paths, "prepared promotion artifact")
        if _file_blake3(path) != artifact.get("blake3"):
            raise PromotionError("prepared promotion artifact changed before commit")
    expected = transaction["expected_state"]
    desired = transaction["desired_state"]
    current = read_state(paths.state)
    if current["state_sha256"] == expected["state_sha256"]:
        _validate_pointer_cas(paths, transaction)
        _inject(fault_injector, "before-state-cas")
        write_state(paths.state, desired, expected_current=expected)
        _inject(fault_injector, "state-committed")
    elif current["state_sha256"] != desired["state_sha256"]:
        raise PromotionError("campaign state revision conflicts with promotion transaction")

    desired_pointers = transaction["desired_model_pointers"]
    if desired_pointers is not None:
        _commit_pointer_bundle(paths, transaction, desired_pointers)
        _inject(fault_injector, "pointers-committed")
    else:
        _validate_pointer_cas(paths, transaction)

    outcome: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": PROMOTION_OUTCOME_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "round_id": transaction["round_id"],
        "classification": transaction["classification"],
        "transaction_blake3": transaction["transaction_blake3"],
        "campaign_state_revision": desired["revision"],
        "campaign_state_sha256": desired["state_sha256"],
        "incumbent_checkpoint_id": desired["incumbent_checkpoint_id"],
        "opponent_pool_changed": desired_pointers is not None,
        "completed_at": utc_now(),
    }
    outcome["outcome_blake3"] = _content_blake3(outcome, "outcome_blake3")
    destination = paths.outcome(transaction["round_id"])
    if destination.exists():
        existing = _read_self_hashed(destination, PROMOTION_OUTCOME_SCHEMA, "outcome_blake3")
        # Completion time is not scientific identity; an earlier successful receipt wins.
        if any(
            existing[key] != outcome[key]
            for key in outcome
            if key not in {"completed_at", "outcome_blake3"}
        ):
            raise PromotionError("existing promotion outcome binds a different transaction")
        return existing
    _write_immutable_json(destination, outcome)
    _inject(fault_injector, "outcome-durable")
    return outcome


def _load_candidate_registration(
    path: Path,
    paths: PromotionPaths,
    validators: PromotionValidators,
) -> dict[str, Any]:
    value = _read_self_hashed(path, CANDIDATE_REGISTRATION_SCHEMA, "registration_blake3")
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "logical_candidate_id",
        "round_index",
        "benchmark_id",
        "run_dir",
        "checkpoint_path",
        "checkpoint_id",
        "checkpoint_manifest_sha256",
        "checkpoint_manifest_blake3",
        "checkpoint_identity_blake3",
        "loss_stream_path",
        "loss_stream_offset_bytes",
        "loss_stream_prefix_blake3",
        "verification_receipt_path",
        "verification_receipt_blake3",
        "verification_id",
        "registered_at",
        "registration_blake3",
    }
    _require_exact_keys(value, required, "candidate registration")
    checkpoint = _registered_path(value["checkpoint_path"], paths, "candidate checkpoint")
    run_dir = _registered_path(value["run_dir"], paths, "candidate run")
    receipt_path = _registered_path(
        value["verification_receipt_path"], paths, "verification receipt"
    )
    if checkpoint.parent != run_dir / "checkpoints" or checkpoint.name != value["checkpoint_id"]:
        raise PromotionError("candidate checkpoint path identity drifted")
    if (
        _file_sha256(checkpoint / "checkpoint.json") != value["checkpoint_manifest_sha256"]
        or _file_blake3(checkpoint / "checkpoint.json") != value["checkpoint_manifest_blake3"]
        or _file_blake3(receipt_path) != value["verification_receipt_blake3"]
    ):
        raise PromotionError("registered candidate bytes changed after registration")
    manifest, checkpoint_state, _ = validators.checkpoint(checkpoint)
    receipt = validators.receipt(receipt_path, checkpoint_path=checkpoint)
    loss_binding = _verify_checkpoint_loss_binding(run_dir, checkpoint_state)
    if (
        manifest.get("checkpoint_id") != value["checkpoint_id"]
        or manifest.get("manifest_identity_blake3") != value["checkpoint_identity_blake3"]
        or receipt.get("verification_id") != value["verification_id"]
        or loss_binding["path"] != value["loss_stream_path"]
        or loss_binding["offset_bytes"] != value["loss_stream_offset_bytes"]
        or loss_binding["prefix_blake3"] != value["loss_stream_prefix_blake3"]
    ):
        raise PromotionError("registered checkpoint or verification identity drifted")
    return value


def _load_gate_registration(
    path: Path,
    candidate: dict[str, Any],
    paths: PromotionPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read_self_hashed(path, GATE_REGISTRATION_SCHEMA, "registration_blake3")
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "round_id",
        "round_index",
        "benchmark_id",
        "candidate_registration_path",
        "candidate_registration_blake3",
        "candidate_checkpoint_id",
        "candidate_checkpoint_manifest_sha256",
        "control_checkpoint_id",
        "opponent_pool_manifest_blake3",
        "opponent_pool_path",
        "focal_report_path",
        "focal_report_blake3",
        "focal_contract_path",
        "focal_contract_file_blake3",
        "focal_contract_file_sha256",
        "opponent_field_path",
        "opponent_field_file_blake3",
        "opponent_field_file_sha256",
        "focal_report_classification",
        "gate_results",
        "resource_limits",
        "registered_at",
        "registration_blake3",
    }
    _require_exact_keys(value, required, "promotion gate registration")
    candidate_path = _registered_path(
        value["candidate_registration_path"], paths, "candidate registration"
    )
    if (
        candidate_path != paths.candidate_registration(candidate["logical_candidate_id"])
        or value["candidate_registration_blake3"] != candidate["registration_blake3"]
        or value["candidate_checkpoint_id"] != candidate["logical_candidate_id"]
        or value["candidate_checkpoint_manifest_sha256"] != candidate["checkpoint_manifest_sha256"]
        or value["round_index"] != candidate["round_index"]
        or value["benchmark_id"] != candidate["benchmark_id"]
    ):
        raise PromotionError("gate registration and candidate registration differ")
    if set(value["gate_results"]) != set(HARD_GATE_NAMES) or not all(
        isinstance(item, bool) for item in value["gate_results"].values()
    ):
        raise PromotionError("gate registration has an invalid hard-gate set")
    report_path = _registered_path(value["focal_report_path"], paths, "focal report")
    if _file_blake3(report_path) != value["focal_report_blake3"]:
        raise PromotionError("fixed-250 focal report changed after registration")
    report = _read_json(report_path, "fixed-250 focal report")
    contract_path = _registered_path(value["focal_contract_path"], paths, "focal contract")
    opponent_field_path = _registered_path(
        value["opponent_field_path"], paths, "focal opponent field"
    )
    if (
        _file_blake3(contract_path) != value["focal_contract_file_blake3"]
        or _file_sha256(contract_path) != value["focal_contract_file_sha256"]
        or _file_blake3(opponent_field_path) != value["opponent_field_file_blake3"]
        or _file_sha256(opponent_field_path) != value["opponent_field_file_sha256"]
    ):
        raise PromotionError("focal contract or opponent field changed after registration")
    contract = _read_json(contract_path, "focal contract")
    opponent_field = _read_json(opponent_field_path, "focal opponent field")
    if value["opponent_pool_path"] is None:
        if value["opponent_pool_manifest_blake3"] is not None:
            raise PromotionError("gate has a pool hash without a pool path")
        allowed_opponents = {GREEDY_POLICY_ID}
    else:
        gate_pool_path = _registered_path(value["opponent_pool_path"], paths, "gate opponent pool")
        gate_pool = _read_self_hashed(gate_pool_path, OPPONENT_POOL_SCHEMA, "manifest_blake3")
        if gate_pool["manifest_blake3"] != value["opponent_pool_manifest_blake3"]:
            raise PromotionError("registered gate opponent pool identity drifted")
        _validate_pool_entries(
            gate_pool["entries"],
            excluded_checkpoint_id=gate_pool["excluded_newest_checkpoint_id"],
        )
        allowed_opponents = {entry["policy_id"] for entry in gate_pool["entries"]}
    _validate_focal_contract_and_field(
        report,
        contract,
        opponent_field,
        contract_sha256=value["focal_contract_file_sha256"],
        opponent_field_sha256=value["opponent_field_file_sha256"],
        benchmark_id=value["benchmark_id"],
        candidate_id=value["candidate_checkpoint_id"],
        control_id=value["control_checkpoint_id"],
        allowed_opponents=allowed_opponents,
    )
    statistics = _validate_focal_report(
        report,
        candidate_id=value["candidate_checkpoint_id"],
        control_id=value["control_checkpoint_id"],
        benchmark_id=value["benchmark_id"],
        gate_results=value["gate_results"],
        resource_limits=value["resource_limits"],
    )
    if statistics["classification"] != value["focal_report_classification"]:
        raise PromotionError("registered and reported promotion classifications differ")
    return value, report


def _validate_focal_report(
    report: Mapping[str, Any],
    *,
    candidate_id: str,
    control_id: str | None,
    benchmark_id: str,
    gate_results: Mapping[str, bool],
    resource_limits: Mapping[str, int | bool],
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 4
        or report.get("schema_id") != FOCAL_REPORT_SCHEMA
        or report.get("benchmark_id") != benchmark_id
    ):
        raise PromotionError("fixed-250 focal report has the wrong schema")
    result = report.get("result")
    if not isinstance(result, dict) or result.get("kind") != "development":
        raise PromotionError("promotion requires the unblinded development aggregate")
    statistics = result.get("statistics")
    if not isinstance(statistics, dict):
        raise PromotionError("fixed-250 focal statistics are absent")
    if (
        statistics.get("schema_version") != 1
        or statistics.get("protocol_id") != FOCAL_PROTOCOL
        or statistics.get("stage") != "development"
        or statistics.get("strength_outputs_blinded") is not False
        or statistics.get("pairs") != FIXED_PAIR_COUNT
        or statistics.get("physical_games") != FIXED_PHYSICAL_GAMES
        or statistics.get("candidate_checkpoint_id") != candidate_id
        or statistics.get("control_checkpoint_id") != control_id
        or statistics.get("classification") not in {"promote", "reject", "inconclusive"}
    ):
        raise PromotionError("fixed-250 focal aggregate identity or sample size drifted")
    work_items = report.get("work_items")
    if not isinstance(work_items, list) or len(work_items) != FIXED_PAIR_COUNT:
        raise PromotionError("fixed-250 promotion requires one work item per pair")
    by_work_item = {
        item.get("work_item_id"): item for item in work_items if isinstance(item, dict)
    }
    expected_work_items = {f"pair-{pair_index:04}" for pair_index in range(FIXED_PAIR_COUNT)}
    if set(by_work_item) != expected_work_items:
        raise PromotionError("promotion must cover every registered pair work item")
    _require_sha256(report.get("contract_sha256"), "focal report contract")
    _require_sha256(report.get("opponent_field_sha256"), "focal report opponent field")
    _require_blake3(report.get("contract_blake3"), "focal report contract")
    _require_blake3(report.get("opponent_field_blake3"), "focal report opponent field")
    seen_pairs: set[int] = set()
    max_peak = 0
    all_clean_shutdowns = True
    all_pinecone_conservation = True
    all_zero_swap = True
    for pair_index in range(FIXED_PAIR_COUNT):
        work_item_id = f"pair-{pair_index:04}"
        item = by_work_item[work_item_id]
        references = item.get("pair_receipts")
        actual_indices = {
            reference.get("pair_index")
            for reference in references or []
            if isinstance(reference, dict)
        }
        if (
            item.get("schema_version") != 4
            or item.get("schema_id") != "cascadia.r2-map.focal-work-item.v4"
            or item.get("contract_sha256") != report["contract_sha256"]
            or item.get("opponent_field_sha256") != report["opponent_field_sha256"]
            or item.get("contract_blake3") != report["contract_blake3"]
            or item.get("opponent_field_blake3") != report["opponent_field_blake3"]
            or item.get("work_item_id") != work_item_id
            or item.get("stage") != "development"
            or item.get("pairs") != 1
            or item.get("physical_games") != 2
            or not isinstance(references, list)
            or len(references) != 1
            or actual_indices != {pair_index}
            or seen_pairs.intersection(actual_indices)
        ):
            raise PromotionError("fixed-250 work-item coverage or identity drifted")
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {
                "pair_index",
                "receipt_blake3",
            }:
                raise PromotionError("fixed-250 work-item receipt reference is malformed")
            _require_blake3(reference["receipt_blake3"], "focal pair receipt")
        seen_pairs.update(actual_indices)
        if (
            not isinstance(item.get("all_clean_shutdowns"), bool)
            or not isinstance(item.get("all_pinecone_conservation_checks_passed"), bool)
            or not isinstance(item.get("peak_rss_bytes"), int)
            or not isinstance(item.get("maximum_swap_delta_bytes"), int)
        ):
            raise PromotionError("runtime or Pinecone evidence is incomplete")
        max_peak = max(max_peak, item["peak_rss_bytes"])
        all_clean_shutdowns &= item["all_clean_shutdowns"]
        all_pinecone_conservation &= item["all_pinecone_conservation_checks_passed"]
        all_zero_swap &= item["maximum_swap_delta_bytes"] <= 0
    if seen_pairs != set(range(FIXED_PAIR_COUNT)):
        raise PromotionError("fixed-250 pair coverage is incomplete")
    required_limits = {
        "max_peak_rss_bytes",
        "max_swap_delta_bytes",
        "require_clean_shutdown",
    }
    if set(resource_limits) != required_limits:
        raise PromotionError("resource limits are not the preregistered exact set")
    if (
        not isinstance(resource_limits["max_peak_rss_bytes"], int)
        or resource_limits["max_peak_rss_bytes"] <= 0
        or resource_limits["max_swap_delta_bytes"] != 0
        or resource_limits["require_clean_shutdown"] is not True
    ):
        raise PromotionError("resource limits are malformed or permit swap")
    actual_memory_pass = max_peak <= resource_limits["max_peak_rss_bytes"]
    actual_zero_swap = all(
        item["maximum_swap_delta_bytes"] <= resource_limits["max_swap_delta_bytes"]
        for item in work_items
    )
    actual_resource_pass = actual_memory_pass and actual_zero_swap and all_clean_shutdowns
    observed_gates = {
        "resource": actual_resource_pass,
        "memory": actual_memory_pass,
        "zero_swap": actual_zero_swap and all_zero_swap,
        "clean_shutdown": all_clean_shutdowns,
        "pinecone_conservation": all_pinecone_conservation,
    }
    for gate_name, observed in observed_gates.items():
        if gate_results.get(gate_name) is not observed:
            raise PromotionError(
                f"registered {gate_name} gate disagrees with aggregate measurements"
            )
    classification = statistics["classification"]
    all_gates_pass = set(gate_results) == set(HARD_GATE_NAMES) and all(gate_results.values())
    if classification == "promote" and not all_gates_pass:
        raise PromotionError("a promote classification has a failed hard gate")
    if not all_gates_pass and classification != "reject":
        raise PromotionError("a failed hard gate must classify the candidate as reject")
    delta = statistics.get("paired_delta", {}).get("base_total", {})
    if delta.get("count") != FIXED_PAIR_COUNT:
        raise PromotionError("paired score delta does not contain exactly 250 observations")
    for key in ("mean",):
        if not isinstance(delta.get(key), int | float) or not math.isfinite(delta[key]):
            raise PromotionError("paired score delta is not finite")
    interval = delta.get("confidence_95")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(isinstance(item, int | float) and math.isfinite(item) for item in interval)
    ):
        raise PromotionError("paired score interval is malformed")
    # Recompute the mechanical strength classification when hard gates pass.
    expected_strength = (
        "promote"
        if delta["mean"] > 0 and interval[0] > 0
        else "reject"
        if interval[1] < 0
        else "inconclusive"
    )
    if all_gates_pass and classification != expected_strength:
        raise PromotionError("reported classification disagrees with the frozen paired rule")
    for arm in ("candidate", "control"):
        arm_statistics = statistics.get(arm, {})
        if arm_statistics.get("base_total", {}).get("count") != FIXED_PAIR_COUNT:
            raise PromotionError(f"{arm} score distribution has the wrong count")
        pinecones = arm_statistics.get("pinecones", {})
        if pinecones.get("conservation_valid_games") != FIXED_PAIR_COUNT:
            raise PromotionError(f"{arm} Pinecone conservation coverage is incomplete")
    return statistics


def _validate_focal_contract_and_field(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    opponent_field: Mapping[str, Any],
    *,
    contract_sha256: str,
    opponent_field_sha256: str,
    benchmark_id: str,
    candidate_id: str,
    control_id: str,
    allowed_opponents: set[str],
) -> None:
    """Recheck the immutable W5 identities without trusting aggregate labels."""
    if (
        contract.get("schema_version") != 4
        or contract.get("schema_id") != FOCAL_CONTRACT_SCHEMA
        or contract.get("benchmark_id") != benchmark_id
        or contract.get("stage") != "development"
        or contract.get("pair_count") != FIXED_PAIR_COUNT
        or contract.get("execution_partition") != {"kind": "scheduler-managed-pairs"}
        or contract.get("candidate_checkpoint_id") != candidate_id
        or contract.get("control_checkpoint_id") != control_id
    ):
        raise PromotionError("focal contract identity or fixed protocol drifted")
    if (
        opponent_field.get("schema_version") != 4
        or opponent_field.get("schema_id") != OPPONENT_FIELD_SCHEMA
        or opponent_field.get("manifest_id") != contract.get("opponent_field_manifest_id")
    ):
        raise PromotionError("focal opponent field identity drifted")
    if report.get("contract_blake3") != _rust_struct_blake3(contract) or report.get(
        "opponent_field_blake3"
    ) != _rust_struct_blake3(opponent_field):
        raise PromotionError("focal report does not bind its contract and opponent field")
    _require_sha256(contract_sha256, "focal contract file")
    _require_sha256(opponent_field_sha256, "focal opponent field file")
    if (
        report.get("contract_sha256") != contract_sha256
        or report.get("opponent_field_sha256") != opponent_field_sha256
    ):
        raise PromotionError("focal report does not bind exact contract and opponent-field files")
    assignments = opponent_field.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != FIXED_PAIR_COUNT:
        raise PromotionError("focal opponent field does not contain exactly 250 assignments")
    seen: set[int] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise PromotionError("focal opponent assignment is not an object")
        pair_index = assignment.get("pair_index")
        if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index in seen:
            raise PromotionError("focal opponent field has duplicate or invalid pair identity")
        seen.add(pair_index)
        focal_seat = pair_index % 4
        opponents = assignment.get("opponents")
        if (
            assignment.get("focal_seat") != focal_seat
            or not isinstance(assignment.get("seed_domain_id"), str)
            or not assignment["seed_domain_id"]
            or not isinstance(opponents, list)
            or len(opponents) != 3
        ):
            raise PromotionError("focal opponent assignment seat or seed domain drifted")
        if {"executor_shard", "host", "node", "compatible_hosts"}.intersection(assignment):
            raise PromotionError("focal opponent assignment contains topology-bearing fields")
        expected_seats = {seat for seat in range(4) if seat != focal_seat}
        actual_seats = {
            opponent.get("seat") for opponent in opponents if isinstance(opponent, dict)
        }
        opponent_ids = {
            opponent.get("checkpoint_id") for opponent in opponents if isinstance(opponent, dict)
        }
        if (
            actual_seats != expected_seats
            or not opponent_ids.issubset(allowed_opponents)
            or candidate_id in opponent_ids
            or (control_id != GREEDY_POLICY_ID and control_id in opponent_ids)
        ):
            raise PromotionError("focal opponent assignment escaped the registered historical pool")
    if seen != set(range(FIXED_PAIR_COUNT)):
        raise PromotionError("focal opponent field pair coverage drifted")


def _rust_struct_blake3(value: Mapping[str, Any]) -> str:
    """Match serde_json::to_vec on the insertion-ordered W5 structs."""
    try:
        encoded = json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise PromotionError(f"focal identity is not canonical JSON: {error}") from error
    return blake3.blake3(encoded).hexdigest()


def _validate_state_binding(
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    phase = Phase(state["phase"])
    if phase not in {Phase.BOOTSTRAP_CANDIDATE_GATE, Phase.PAIRED_CANDIDATE_GATE}:
        raise PromotionError("campaign is not at a legal candidate-gate phase")
    if (
        state["candidate_checkpoint_id"] != candidate["logical_candidate_id"]
        or state["candidate_checkpoint_sha256"] != candidate["checkpoint_manifest_sha256"]
        or state["round_index"] != candidate["round_index"]
    ):
        raise PromotionError("campaign state and registered candidate differ")
    expected_control = (
        GREEDY_POLICY_ID
        if phase is Phase.BOOTSTRAP_CANDIDATE_GATE
        else state["incumbent_checkpoint_id"]
    )
    if gate["control_checkpoint_id"] != expected_control:
        raise PromotionError("gate control differs from the durable incumbent")


def _load_current_pointers(
    paths: PromotionPaths, state: Mapping[str, Any]
) -> dict[str, Any] | None:
    if state["promotion_index"] is None:
        if paths.model_pointers.exists():
            raise PromotionError("model pointers exist before bootstrap promotion")
        return None
    if not paths.model_pointers.exists():
        raise PromotionError("post-bootstrap campaign has no model pointer bundle")
    value = _read_self_hashed(paths.model_pointers, MODEL_POINTERS_SCHEMA, "pointers_blake3")
    if (
        value.get("revision") != state["promotion_index"]
        or value.get("incumbent") != value.get("promoted")
        or value.get("incumbent", {}).get("policy_id") != state["incumbent_checkpoint_id"]
        or value.get("incumbent", {}).get("checkpoint_manifest_sha256")
        != state["incumbent_checkpoint_sha256"]
    ):
        raise PromotionError("current incumbent/promoted pointers drifted from campaign state")
    pool = _load_pool_from_pointers(paths, value)
    if pool["promotion_index"] != state["promotion_index"]:
        raise PromotionError("current opponent pool promotion index drifted")
    return value


def _load_pool_from_pointers(paths: PromotionPaths, pointers: Mapping[str, Any]) -> dict[str, Any]:
    reference = pointers.get("opponent_pool")
    if not isinstance(reference, dict):
        raise PromotionError("model pointers omit the opponent pool")
    pool_path = _registered_path(reference.get("path"), paths, "opponent pool")
    pool = _read_self_hashed(pool_path, OPPONENT_POOL_SCHEMA, "manifest_blake3")
    if pool.get("pool_id") != reference.get("pool_id") or pool.get(
        "manifest_blake3"
    ) != reference.get("manifest_blake3"):
        raise PromotionError("opponent pool pointer differs from immutable pool")
    _validate_pool_entries(
        pool.get("entries"), excluded_checkpoint_id=pool.get("excluded_newest_checkpoint_id")
    )
    return pool


def _validate_pool_entries(entries: Any, *, excluded_checkpoint_id: str) -> None:
    if not isinstance(entries, list) or not entries or entries[0] != _greedy_pool_entry():
        raise PromotionError("opponent pool must begin with the unique greedy policy")
    policy_ids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("policy_id") in policy_ids:
            raise PromotionError("opponent pool policy identities are not unique")
        policy_id = entry["policy_id"]
        if policy_id == excluded_checkpoint_id:
            raise PromotionError("opponent pool contains the newest promoted checkpoint")
        policy_ids.add(policy_id)
        if entry.get("kind") == "checkpoint":
            digest = entry.get("checkpoint_manifest_blake3")
            _require_blake3(digest, "opponent checkpoint manifest")
            if digest in checkpoint_hashes:
                raise PromotionError("opponent pool contains duplicate checkpoint bytes")
            checkpoint_hashes.add(digest)
        elif entry != _greedy_pool_entry():
            raise PromotionError("opponent pool contains an unknown policy kind")


def _validate_pointer_cas(paths: PromotionPaths, transaction: Mapping[str, Any]) -> None:
    expected = transaction["expected_model_pointers_blake3"]
    if paths.model_pointers.exists():
        current = _read_self_hashed(paths.model_pointers, MODEL_POINTERS_SCHEMA, "pointers_blake3")
        observed = current["pointers_blake3"]
    else:
        observed = None
    desired = transaction["desired_model_pointers"]
    if desired is not None and observed == desired["pointers_blake3"]:
        return
    if observed != expected:
        raise PromotionError("incumbent/promoted pointer revision conflicts with transaction")


def _commit_pointer_bundle(
    paths: PromotionPaths,
    transaction: Mapping[str, Any],
    desired: dict[str, Any],
) -> None:
    _validate_pointer_cas(paths, transaction)
    if paths.model_pointers.exists():
        existing = _read_self_hashed(paths.model_pointers, MODEL_POINTERS_SCHEMA, "pointers_blake3")
        if existing["pointers_blake3"] == desired["pointers_blake3"]:
            return
    _write_json_atomic(paths.model_pointers, desired)


def _promoted_checkpoint_record(promotion_id: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "checkpoint",
        "policy_id": promotion_id,
        "source_candidate_id": candidate["logical_candidate_id"],
        "checkpoint_id": candidate["checkpoint_id"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_manifest_sha256": candidate["checkpoint_manifest_sha256"],
        "checkpoint_manifest_blake3": candidate["checkpoint_manifest_blake3"],
        "checkpoint_identity_blake3": candidate["checkpoint_identity_blake3"],
        "loss_stream_path": candidate["loss_stream_path"],
        "loss_stream_offset_bytes": candidate["loss_stream_offset_bytes"],
        "loss_stream_prefix_blake3": candidate["loss_stream_prefix_blake3"],
        "verification_id": candidate["verification_id"],
    }


def _verify_checkpoint_loss_binding(run_dir: Path, state: Any) -> dict[str, Any]:
    """Verify the externally stored loss prefix captured by a production checkpoint."""
    if state is None:
        # Synthetic promotion tests inject a verifier for metadata-only checkpoints.
        return {"path": None, "offset_bytes": None, "prefix_blake3": None}
    try:
        loss_stream = state.loss_stream
        relative_path = loss_stream["relative_path"]
        loss_path = run_dir / relative_path
    except (AttributeError, KeyError, TypeError) as error:
        raise PromotionError("verified checkpoint has no loss-stream binding") from error
    try:
        loss_path.relative_to(run_dir)
    except ValueError as error:
        raise PromotionError("checkpoint loss stream escapes its registered run") from error
    verify_loss_stream_prefix(loss_path, loss_stream)
    return {
        "path": relative_path,
        "offset_bytes": loss_stream["offset_bytes"],
        "prefix_blake3": loss_stream["prefix_blake3"],
    }


def _greedy_pool_entry() -> dict[str, Any]:
    return {"kind": "greedy", "policy_id": GREEDY_POLICY_ID}


def _paths(root: str | Path) -> PromotionPaths:
    root = Path(root)
    reject_frozen_campaign_path(root, label="promotion campaign root")
    if root == CAMPAIGN_ROOT:
        require_local_storage_authority()
    if not root.is_absolute() or not root.exists() or not root.is_dir():
        raise PromotionError("campaign root must be an existing absolute directory")
    return PromotionPaths(root.resolve())


def _contained(path: Path, root: Path, label: str) -> Path:
    try:
        return canonical_campaign_path(path, root=root, label=label)
    except ContractError as error:
        raise PromotionError(str(error)) from error


def _registered_path(value: Any, paths: PromotionPaths, label: str) -> Path:
    if not isinstance(value, str):
        raise PromotionError(f"{label} path is not a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PromotionError(f"{label} path is not campaign-relative")
    return _contained(paths.root / relative, paths.root, label)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise PromotionError(f"artifact escapes campaign root: {path}") from error


def _round_id(round_index: int | None) -> str:
    return "bootstrap" if round_index is None else f"R[{round_index}]"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return value


def _read_self_hashed(path: Path, schema: str, hash_field: str) -> dict[str, Any]:
    value = _read_json(path, schema)
    if (
        value.get("schema_version") != 1
        or value.get("schema_id") != schema
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get(hash_field) != _content_blake3(value, hash_field)
    ):
        raise PromotionError(f"{schema} schema, campaign, or content hash differs")
    return value


def _require_exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    if set(value) != required:
        raise PromotionError(
            f"{label} keys differ: missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )


def _require_blake3(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromotionError(f"{label} must be a lowercase BLAKE3 digest")


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromotionError(f"{label} must be a lowercase SHA-256 digest")


def _content_blake3(value: Mapping[str, Any], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return blake3.blake3(canonical_json_bytes(payload)).hexdigest()


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PromotionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PromotionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _artifact_reference(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _relative(path, root), "blake3": _file_blake3(path)}


def _write_immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".immutable-write.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            existing = _read_json(path, "immutable artifact")
            if existing != value:
                raise PromotionError(
                    f"immutable artifact already exists with different bytes: {path}"
                )
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_dev != path.parent.stat().st_dev:
                raise PromotionError("immutable promotion artifact temporary crossed volumes")
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_dev != path.parent.stat().st_dev:
            raise PromotionError("pointer bundle temporary crossed volumes")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inject(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-registration", type=Path, required=True)
    parser.add_argument("--gate-registration", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = apply_registered_gate(
            candidate_registration_path=arguments.candidate_registration,
            gate_registration_path=arguments.gate_registration,
        )
    except (PromotionError, CheckpointError, TransitionError) as error:
        print(f"R2-MAP promotion refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
