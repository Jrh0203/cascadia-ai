"""Deterministic R2-MAP adapter over the existing research queue and ledger.

This module is deliberately not a scheduler. It materializes immutable work
packets, validates host receipts, reconciles those artifacts with the existing
queue/ledger schemas, and advances the hash-chained campaign state only after
the registered phase barrier is proven.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cascadia_mlx.r2_map_contracts import (
    ALLOWED_HOSTS,
    CAMPAIGN_BUDGET_BYTES,
    CAMPAIGN_ID,
    CAMPAIGN_ROOT,
    FORBIDDEN_HOST,
    PHASE_HOST_INTENTS,
    STORAGE_HOST,
    ContractError,
    Phase,
    canonical_json_bytes,
    content_sha256,
    new_campaign_state,
    read_state,
    transition_state,
    validate_state,
    write_state,
)

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import cluster_experiment_ledger as experiment_ledger  # noqa: E402
import cluster_research_queue as research_queue  # noqa: E402

PACKET_SCHEMA = "cascadia.r2-map.work-packet.v2"
RECEIPT_SCHEMA = "cascadia.r2-map.work-receipt.v2"
REMOTE_STORAGE_TRANSPORT = "cascadia-r2-map-ssh-framed-v1"
REMOTE_RECEIPT_SCHEMA = "cascadia.r2-map.remote-receipt.v1"
REMOTE_TRANSACTION_SCHEMA = "cascadia.r2-map.remote-transaction-manifest.v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REMOTE_ARTIFACT_TOP_LEVELS = frozenset(
    {
        "benchmarks",
        "bundles",
        "checkpoints",
        "control",
        "datasets",
        "logs",
        "opponent-pool",
        "reports",
        "runs",
    }
)
CONTROLLER_EXPERIMENT_ID = CAMPAIGN_ID
MAX_ATTEMPTS = 3
MAX_RSS_BYTES = 4 * (1 << 30)
GENERATION_LEASE_SIZE = 1_000_000

WORK_PACKET_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": PACKET_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "schema_id",
        "campaign_id",
        "controller_revision",
        "controller_state_sha256",
        "phase",
        "task_id",
        "operation",
        "task_kind",
        "aggregate_kind",
        "host",
        "required_host_intent",
        "dependencies",
        "command",
        "storage",
        "artifact_root",
        "receipt_name",
        "seed_lease",
        "retry",
        "stop_gates",
        "packet_sha256",
    ],
    "properties": {
        "schema_version": {"const": 2},
        "schema_id": {"const": PACKET_SCHEMA},
        "campaign_id": {"const": CAMPAIGN_ID},
        "controller_revision": {"type": "integer", "minimum": 0},
        "controller_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "phase": {"enum": [phase.value for phase in Phase]},
        "task_id": {"type": "string", "minLength": 1},
        "operation": {"type": "string", "minLength": 1},
        "task_kind": {
            "enum": ["generate", "train", "longitudinal-benchmark", "candidate-gate", "aggregate"]
        },
        "aggregate_kind": {
            "type": ["string", "null"],
            "enum": ["generation", "longitudinal-benchmark", "candidate-gate", None],
        },
        "host": {"enum": list(ALLOWED_HOSTS)},
        "required_host_intent": {
            "enum": sorted(
                {
                    str(intent)
                    for intents in PHASE_HOST_INTENTS.values()
                    for intent in intents.values()
                }
            )
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "command": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "storage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["host", "root", "transport"],
            "properties": {
                "host": {"const": STORAGE_HOST},
                "root": {"const": str(CAMPAIGN_ROOT)},
                "transport": {"const": REMOTE_STORAGE_TRANSPORT},
            },
        },
        "artifact_root": {"type": "string", "minLength": 1},
        "receipt_name": {"type": "string", "minLength": 1},
        "seed_lease": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "purpose",
                        "round_index",
                        "host",
                        "first_index",
                        "count",
                        "stride",
                        "lease_sha256",
                    ],
                    "properties": {
                        "purpose": {
                            "enum": [
                                "bootstrap",
                                "generation",
                                "longitudinal-benchmark",
                                "candidate-gate",
                            ]
                        },
                        "round_index": {"type": "integer", "minimum": 0},
                        "host": {"enum": list(ALLOWED_HOSTS)},
                        "first_index": {"type": "integer", "minimum": 0},
                        "count": {"type": "integer", "minimum": 1},
                        "stride": {"type": "integer", "minimum": 1},
                        "lease_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                },
            ]
        },
        "retry": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "maximum_attempts",
                "same_host_only",
                "live_seed_reassignment",
                "idempotency_key",
            ],
            "properties": {
                "maximum_attempts": {"const": MAX_ATTEMPTS},
                "same_host_only": {"const": True},
                "live_seed_reassignment": {"const": False},
                "idempotency_key": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "stop_gates": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "maximum_rss_bytes",
                "maximum_process_swaps",
                "maximum_system_swap_delta_bytes",
                "require_identity",
                "require_replay",
                "require_finite_training",
            ],
            "properties": {
                "maximum_rss_bytes": {"const": MAX_RSS_BYTES},
                "maximum_process_swaps": {"const": 0},
                "maximum_system_swap_delta_bytes": {"const": 0},
                "require_identity": {"const": True},
                "require_replay": {"type": "boolean"},
                "require_finite_training": {"type": "boolean"},
            },
        },
        "packet_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}

WORK_RECEIPT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": RECEIPT_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "schema_id",
        "campaign_id",
        "packet_sha256",
        "controller_state_sha256",
        "task_id",
        "task_kind",
        "host",
        "storage",
        "outcome",
        "completed_unix_ms",
        "used_seed_prefix",
        "artifacts",
        "metrics",
        "gates",
        "receipt_sha256",
    ],
    "properties": {
        "schema_version": {"const": 2},
        "schema_id": {"const": RECEIPT_SCHEMA},
        "campaign_id": {"const": CAMPAIGN_ID},
        "packet_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "controller_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "task_id": {"type": "string", "minLength": 1},
        "task_kind": {
            "enum": ["generate", "train", "longitudinal-benchmark", "candidate-gate", "aggregate"]
        },
        "host": {"enum": list(ALLOWED_HOSTS)},
        "storage": WORK_PACKET_JSON_SCHEMA["properties"]["storage"],
        "outcome": {"const": "completed"},
        "completed_unix_ms": {"type": "integer", "minimum": 0},
        "used_seed_prefix": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["lease_sha256", "used_count", "unused_count", "last_index"],
                    "properties": {
                        "lease_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "used_count": {"type": "integer", "minimum": 0},
                        "unused_count": {"type": "integer", "minimum": 0},
                        "last_index": {"type": ["integer", "null"]},
                    },
                },
            ]
        },
        "artifacts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "label",
                    "path",
                    "bytes",
                    "sha256",
                    "storage_receipt_relative",
                    "storage_receipt_sha256",
                ],
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "bytes": {"type": "integer", "minimum": 0},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "storage_receipt_relative": {
                        "type": "string",
                        "pattern": "^control/receipts/req-[A-Za-z0-9._-]+\\.json$",
                    },
                    "storage_receipt_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    },
                },
            },
        },
        "metrics": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "maximum_rss_bytes",
                "process_swaps",
                "system_swap_delta_bytes",
                "games",
                "classification",
            ],
            "properties": {
                "maximum_rss_bytes": {"type": "integer", "minimum": 0, "maximum": MAX_RSS_BYTES},
                "process_swaps": {"const": 0},
                "system_swap_delta_bytes": {"const": 0},
                "games": {"type": "integer", "minimum": 0},
                "classification": {
                    "type": ["string", "null"],
                    "enum": ["promote", "reject", "inconclusive", None],
                },
            },
        },
        "gates": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "identity",
                "replay",
                "finite_training",
                "checkpoint_verified",
                "score_accounting",
            ],
            "properties": {
                "identity": {"const": True},
                "replay": {"const": True},
                "finite_training": {"const": True},
                "checkpoint_verified": {"const": True},
                "score_accounting": {"const": True},
            },
        },
        "receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}


class CampaignControllerError(ContractError):
    """The queue, ledger, packet, receipt, or phase barrier is inconsistent."""


@dataclass(frozen=True)
class ControllerPaths:
    root: Path
    state: Path
    queue: Path
    ledger: Path
    packets: Path
    incoming: Path
    receipts: Path
    dashboard_inputs: Path
    history: Path
    stop: Path

    @classmethod
    def under(cls, root: str | Path) -> ControllerPaths:
        root = Path(root)
        control = root / "control"
        return cls(
            root=root,
            state=control / "campaign-state.json",
            queue=control / "research-queue-v1.json",
            ledger=control / "research-experiments-v1.json",
            packets=control / "work-packets",
            incoming=control / "incoming-receipts",
            receipts=control / "receipts",
            dashboard_inputs=control / "dashboard-inputs",
            history=control / "controller-history.jsonl",
            stop=control / "controller-stop.json",
        )

    @classmethod
    def with_existing_queue_and_ledger(
        cls,
        root: str | Path,
        *,
        queue: str | Path,
        ledger: str | Path,
    ) -> ControllerPaths:
        paths = cls.under(root)
        return cls(
            root=paths.root,
            state=paths.state,
            queue=Path(queue),
            ledger=Path(ledger),
            packets=paths.packets,
            incoming=paths.incoming,
            receipts=paths.receipts,
            dashboard_inputs=paths.dashboard_inputs,
            history=paths.history,
            stop=paths.stop,
        )


def write_controller_schemas(paths: ControllerPaths) -> dict[str, str]:
    destinations = {
        "work_packet": paths.root / "control/contracts/r2-map-work-packet-v2.schema.json",
        "work_receipt": paths.root / "control/contracts/r2-map-work-receipt-v2.schema.json",
    }
    _write_atomic_json(destinations["work_packet"], WORK_PACKET_JSON_SCHEMA)
    _write_atomic_json(destinations["work_receipt"], WORK_RECEIPT_JSON_SCHEMA)
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in destinations.items()
    }


def initialize_controller(paths: ControllerPaths, *, now_ms: int) -> dict[str, Any]:
    """Create an isolated controller projection without touching another queue."""
    paths.root.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_controller_schemas(paths)
    if not paths.state.exists():
        write_state(paths.state, new_campaign_state(now="1970-01-01T00:00:00.000Z"))
    elif read_state(paths.state)["campaign_id"] != CAMPAIGN_ID:
        raise CampaignControllerError("controller state names another campaign")
    if not paths.queue.exists():
        queue = research_queue.empty_queue("r2-map-existing-queue-adapter-v1", now_ms=now_ms)
        research_queue._atomic_write(paths.queue, queue)
    else:
        research_queue.load_queue(paths.queue)
    if not paths.ledger.exists():
        experiment_ledger.write_ledger(paths.ledger, experiment_ledger.empty_ledger(now_ms=now_ms))
    else:
        experiment_ledger.read_ledger(paths.ledger)
    return reconcile(paths, now_ms=now_ms)


@dataclass(frozen=True)
class TaskTemplate:
    operation: str
    kind: str
    host: str
    dependencies: tuple[str, ...] = ()
    aggregate_kind: str | None = None


def phase_templates(phase: Phase | str) -> tuple[TaskTemplate, ...]:
    phase = Phase(phase)
    if phase is Phase.BOOTSTRAP_GENERATING:
        generation = tuple(
            TaskTemplate(f"bootstrap-generate-{host}", "generate", host) for host in ALLOWED_HOSTS
        )
        return (
            *generation,
            TaskTemplate(
                "bootstrap-generation-aggregate",
                "aggregate",
                "john1",
                tuple(item.operation for item in generation),
                "generation",
            ),
        )
    if phase is Phase.BOOTSTRAP_TRAINING:
        return (TaskTemplate("bootstrap-train", "train", "john1"),)
    if phase is Phase.BOOTSTRAP_CANDIDATE_GATE:
        gates = (
            TaskTemplate("bootstrap-candidate-gate-john2", "candidate-gate", "john2"),
            TaskTemplate("bootstrap-candidate-gate-john3", "candidate-gate", "john3"),
        )
        return (
            *gates,
            TaskTemplate(
                "bootstrap-candidate-gate-aggregate",
                "aggregate",
                "john1",
                tuple(item.operation for item in gates),
                "candidate-gate",
            ),
        )
    if phase is Phase.GENERATING:
        return tuple(
            TaskTemplate(f"round-generate-{host}", "generate", host) for host in ALLOWED_HOSTS
        )
    if phase is Phase.LOCAL_SHARDS_COMPLETE:
        return (
            TaskTemplate(
                "round-generation-aggregate", "aggregate", "john1", aggregate_kind="generation"
            ),
        )
    if phase is Phase.TRAINING_AND_BENCHMARKING:
        benchmarks = (
            TaskTemplate("longitudinal-benchmark-john2", "longitudinal-benchmark", "john2"),
            TaskTemplate("longitudinal-benchmark-john3", "longitudinal-benchmark", "john3"),
        )
        return (
            TaskTemplate("round-train", "train", "john1"),
            *benchmarks,
            TaskTemplate(
                "longitudinal-benchmark-aggregate",
                "aggregate",
                "john1",
                tuple(item.operation for item in benchmarks),
                "longitudinal-benchmark",
            ),
        )
    if phase is Phase.PAIRED_CANDIDATE_GATE:
        gates = (
            TaskTemplate("round-candidate-gate-john2", "candidate-gate", "john2"),
            TaskTemplate("round-candidate-gate-john3", "candidate-gate", "john3"),
        )
        return (
            *gates,
            TaskTemplate(
                "round-candidate-gate-aggregate",
                "aggregate",
                "john1",
                tuple(item.operation for item in gates),
                "candidate-gate",
            ),
        )
    return ()


def build_phase_packets(
    state: Mapping[str, Any],
    *,
    commands: Mapping[str, Sequence[str]],
    artifact_root: str,
    synthetic: bool = False,
) -> tuple[dict[str, Any], ...]:
    state = validate_state(state)
    phase = Phase(state["phase"])
    templates = phase_templates(phase)
    operation_ids = {
        template.operation: _task_id(state, template.operation) for template in templates
    }
    artifact_root = _validate_remote_relative_path(artifact_root, "artifact root")
    packets = []
    for template in templates:
        command = commands.get(template.operation)
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
            raise CampaignControllerError(f"missing command for operation {template.operation}")
        command = [str(item) for item in command]
        if any(not item for item in command):
            raise CampaignControllerError(f"operation {template.operation} has an empty command")
        if not synthetic and command == ["/usr/bin/true"]:
            raise CampaignControllerError("synthetic commands are forbidden outside dry-run mode")
        task_id = operation_ids[template.operation]
        packet: dict[str, Any] = {
            "schema_version": 2,
            "schema_id": PACKET_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "controller_revision": state["revision"],
            "controller_state_sha256": state["state_sha256"],
            "phase": phase.value,
            "task_id": task_id,
            "operation": template.operation,
            "task_kind": template.kind,
            "aggregate_kind": template.aggregate_kind,
            "host": template.host,
            "required_host_intent": str(PHASE_HOST_INTENTS[phase][template.host]),
            "dependencies": [operation_ids[item] for item in template.dependencies],
            "command": command,
            "storage": _storage_binding(),
            "artifact_root": artifact_root,
            "receipt_name": f"{task_id}.json",
            "seed_lease": _seed_lease(state, template),
            "retry": {
                "maximum_attempts": MAX_ATTEMPTS,
                "same_host_only": True,
                "live_seed_reassignment": False,
                "idempotency_key": _sha256_text(
                    f"{state['state_sha256']}:{task_id}:{template.host}"
                ),
            },
            "stop_gates": {
                "maximum_rss_bytes": MAX_RSS_BYTES,
                "maximum_process_swaps": 0,
                "maximum_system_swap_delta_bytes": 0,
                "require_identity": True,
                "require_replay": template.kind == "generate",
                "require_finite_training": template.kind == "train",
            },
        }
        _reject_john4(packet)
        packet["packet_sha256"] = content_sha256(packet, hash_field="packet_sha256")
        packets.append(validate_work_packet(packet))
    return tuple(packets)


def validate_work_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "controller_revision",
        "controller_state_sha256",
        "phase",
        "task_id",
        "operation",
        "task_kind",
        "aggregate_kind",
        "host",
        "required_host_intent",
        "dependencies",
        "command",
        "storage",
        "artifact_root",
        "receipt_name",
        "seed_lease",
        "retry",
        "stop_gates",
        "packet_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CampaignControllerError("work packet keys differ from schema v2")
    if value["schema_version"] != 2 or value["schema_id"] != PACKET_SCHEMA:
        raise CampaignControllerError("work packet schema differs")
    if value["campaign_id"] != CAMPAIGN_ID:
        raise CampaignControllerError("work packet names another campaign")
    if value["storage"] != _storage_binding():
        raise CampaignControllerError("work packet storage authority differs")
    _validate_remote_relative_path(value["artifact_root"], "artifact root")
    _reject_john4(value)
    if value["host"] not in ALLOWED_HOSTS:
        raise CampaignControllerError("work packet host is unauthorized")
    try:
        phase = Phase(value["phase"])
    except ValueError as error:
        raise CampaignControllerError("work packet phase is unknown") from error
    if value["required_host_intent"] != str(PHASE_HOST_INTENTS[phase][value["host"]]):
        raise CampaignControllerError("work packet host intent differs from phase")
    if value["task_kind"] not in {
        "generate",
        "train",
        "longitudinal-benchmark",
        "candidate-gate",
        "aggregate",
    }:
        raise CampaignControllerError("work packet task kind is unknown")
    _validate_packet_template(value, phase)
    if not isinstance(value["dependencies"], list) or len(value["dependencies"]) != len(
        set(value["dependencies"])
    ):
        raise CampaignControllerError("work packet dependencies are invalid")
    if (
        not isinstance(value["command"], list)
        or not value["command"]
        or any(not isinstance(item, str) or not item for item in value["command"])
    ):
        raise CampaignControllerError("work packet command is invalid")
    retry = value["retry"]
    if (
        not isinstance(retry, Mapping)
        or retry.get("maximum_attempts") != MAX_ATTEMPTS
        or retry.get("same_host_only") is not True
        or retry.get("live_seed_reassignment") is not False
        or not _is_sha256(retry.get("idempotency_key"))
    ):
        raise CampaignControllerError("work packet retry contract differs")
    _validate_seed_lease(
        value["seed_lease"],
        kind=value["task_kind"],
        host=value["host"],
        phase=phase,
    )
    if value["packet_sha256"] != content_sha256(value, hash_field="packet_sha256"):
        raise CampaignControllerError("work packet hash differs")
    return dict(value)


def _validate_packet_template(value: Mapping[str, Any], phase: Phase) -> None:
    revision = value["controller_revision"]
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 0
        or not _is_sha256(value["controller_state_sha256"])
    ):
        raise CampaignControllerError("work packet controller identity is invalid")
    matching = [
        template for template in phase_templates(phase) if template.operation == value["operation"]
    ]
    if len(matching) != 1:
        raise CampaignControllerError("work packet operation is not registered for its phase")
    template = matching[0]
    expected_task_id = f"r2map-r{revision:04d}-{template.operation}"
    expected_dependencies = [
        f"r2map-r{revision:04d}-{operation}" for operation in template.dependencies
    ]
    if (
        value["task_id"] != expected_task_id
        or value["task_kind"] != template.kind
        or value["aggregate_kind"] != template.aggregate_kind
        or value["host"] != template.host
        or value["dependencies"] != expected_dependencies
        or value["receipt_name"] != f"{expected_task_id}.json"
    ):
        raise CampaignControllerError("work packet differs from its registered phase template")


def queue_task_for_packet(packet: Mapping[str, Any], *, created_unix_ms: int) -> dict[str, Any]:
    packet = validate_work_packet(packet)
    task = {
        "id": packet["task_id"],
        "title": f"R2-MAP {packet['operation']}",
        "experiment_id": CONTROLLER_EXPERIMENT_ID,
        "decision": f"Complete immutable packet {packet['packet_sha256']}",
        "workload_class": "shared-prerequisite"
        if packet["task_kind"] == "aggregate"
        else "independent-experiment",
        "priority": 0,
        "decision_value": 1.0,
        "expected_runtime_seconds": 2700.0 if packet["task_kind"] == "generate" else 900.0,
        "critical_path": True,
        "decision_terminal": packet["task_kind"] == "aggregate",
        "compatible_hosts": [packet["host"]],
        "dependencies": list(packet["dependencies"]),
        "command": list(packet["command"]),
        "artifact_path": _remote_artifact_uri(
            _join_remote_relative(packet["artifact_root"], packet["receipt_name"])
        ),
        "stop_rule": "Stop at the packet receipt; retries retain the same host and seed lease.",
        "resources": {
            "cpu_cores": 1,
            "memory_gib": 4.0,
            "uses_mlx": packet["task_kind"] == "train",
        },
    }
    # Reuse the queue's authoritative schema constructor. Full DAG validation
    # occurs atomically when the phase packet set is installed.
    probe = research_queue.empty_queue("r2-map-packet-probe", now_ms=created_unix_ms)
    return research_queue._task_from_specification(probe, task, now_ms=created_unix_ms)


def make_synthetic_receipt(
    packet: Mapping[str, Any],
    *,
    completed_unix_ms: int,
    outcome: str = "completed",
    classification: str | None = None,
) -> dict[str, Any]:
    packet = validate_work_packet(packet)
    synthetic_payload = _synthetic_artifact_payload(packet)
    synthetic_artifact = {
        "label": _artifact_label(packet),
        "path": _join_remote_relative(
            packet["artifact_root"], f"{packet['task_id']}.artifact"
        ),
        "bytes": len(synthetic_payload),
        "sha256": hashlib.sha256(synthetic_payload).hexdigest(),
        "storage_receipt_relative": (
            "control/receipts/req-synthetic-"
            f"{packet['packet_sha256'][:32]}.json"
        ),
    }
    synthetic_storage_receipt = _synthetic_remote_storage_receipt(
        packet,
        synthetic_artifact,
        completed_unix_ms=completed_unix_ms,
    )
    synthetic_artifact["storage_receipt_sha256"] = synthetic_storage_receipt[
        "receipt_sha256"
    ]
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "schema_id": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "packet_sha256": packet["packet_sha256"],
        "controller_state_sha256": packet["controller_state_sha256"],
        "task_id": packet["task_id"],
        "task_kind": packet["task_kind"],
        "host": packet["host"],
        "storage": dict(packet["storage"]),
        "outcome": outcome,
        "completed_unix_ms": completed_unix_ms,
        "used_seed_prefix": _synthetic_seed_prefix(packet["seed_lease"]),
        "artifacts": [synthetic_artifact],
        "metrics": {
            "maximum_rss_bytes": 64 * (1 << 20),
            "process_swaps": 0,
            "system_swap_delta_bytes": 0,
            "games": 1 if packet["task_kind"] != "aggregate" else 0,
            "classification": classification,
        },
        "gates": {
            "identity": True,
            "replay": True,
            "finite_training": True,
            "checkpoint_verified": True,
            "score_accounting": True,
        },
    }
    receipt["receipt_sha256"] = content_sha256(receipt, hash_field="receipt_sha256")
    return validate_receipt(receipt, packet=packet)


def validate_receipt(value: Mapping[str, Any], *, packet: Mapping[str, Any]) -> dict[str, Any]:
    packet = validate_work_packet(packet)
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "packet_sha256",
        "controller_state_sha256",
        "task_id",
        "task_kind",
        "host",
        "storage",
        "outcome",
        "completed_unix_ms",
        "used_seed_prefix",
        "artifacts",
        "metrics",
        "gates",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CampaignControllerError("work receipt keys differ from schema v2")
    if value["schema_version"] != 2 or value["schema_id"] != RECEIPT_SCHEMA:
        raise CampaignControllerError("work receipt schema differs")
    _reject_john4(value)
    for field in (
        "campaign_id",
        "packet_sha256",
        "controller_state_sha256",
        "task_id",
        "task_kind",
        "host",
        "storage",
    ):
        expected = packet[field]
        if value[field] != expected:
            raise CampaignControllerError(f"work receipt {field} differs from packet")
    if value["outcome"] != "completed":
        raise CampaignControllerError("only completed work receipts satisfy a phase barrier")
    if not isinstance(value["completed_unix_ms"], int) or value["completed_unix_ms"] < 0:
        raise CampaignControllerError("work receipt completion time is invalid")
    _validate_used_seed_prefix(value["used_seed_prefix"], packet["seed_lease"])
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise CampaignControllerError("work receipt requires immutable artifacts")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {
                "label",
                "path",
                "bytes",
                "sha256",
                "storage_receipt_relative",
                "storage_receipt_sha256",
            }
            or not isinstance(artifact["label"], str)
            or not artifact["label"]
            or not isinstance(artifact["bytes"], int)
            or isinstance(artifact["bytes"], bool)
            or artifact["bytes"] < 0
            or not _is_sha256(artifact["sha256"])
            or not _is_sha256(artifact["storage_receipt_sha256"])
        ):
            raise CampaignControllerError("work receipt artifact identity is invalid")
        relative = _validate_remote_relative_path(artifact["path"], "receipt artifact")
        root = _validate_remote_relative_path(packet["artifact_root"], "artifact root")
        if relative != root and not relative.startswith(f"{root}/"):
            raise CampaignControllerError("work receipt artifact escapes its remote artifact root")
        _validate_storage_receipt_relative(artifact["storage_receipt_relative"])
    metrics = value["metrics"]
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(metrics.get("maximum_rss_bytes"), int)
        or metrics["maximum_rss_bytes"] < 0
        or metrics["maximum_rss_bytes"] > MAX_RSS_BYTES
        or metrics.get("process_swaps") != 0
        or metrics.get("system_swap_delta_bytes") != 0
    ):
        raise CampaignControllerError("work receipt violates memory or zero-swap gates")
    classification = metrics.get("classification")
    if packet["aggregate_kind"] == "candidate-gate":
        if classification not in {"promote", "reject", "inconclusive"}:
            raise CampaignControllerError("candidate-gate aggregate lacks a valid classification")
    elif classification is not None:
        raise CampaignControllerError("non-gate receipt cannot report a gate classification")
    gates = value["gates"]
    if not isinstance(gates, Mapping) or not all(
        gates.get(name) is True
        for name in (
            "identity",
            "replay",
            "finite_training",
            "checkpoint_verified",
            "score_accounting",
        )
    ):
        raise CampaignControllerError("work receipt has a failed scientific or recovery gate")
    if value["receipt_sha256"] != content_sha256(value, hash_field="receipt_sha256"):
        raise CampaignControllerError("work receipt hash differs")
    return dict(value)


def verify_receipt_storage_evidence(
    paths: ControllerPaths,
    receipt: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve every persisted John2 worker receipt and reverify its artifact."""
    receipt = validate_receipt(receipt, packet=packet)
    evidence = []
    for artifact in receipt["artifacts"]:
        locator = _validate_storage_receipt_relative(
            artifact["storage_receipt_relative"]
        )
        receipt_path, receipt_payload = _read_immutable_contained_file(
            paths,
            locator,
            label="persisted storage receipt",
            allowed_modes={0o400},
            maximum_bytes=2 << 20,
        )
        try:
            storage_receipt = json.loads(receipt_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CampaignControllerError("persisted storage receipt is invalid JSON") from error
        if not isinstance(storage_receipt, Mapping):
            raise CampaignControllerError("persisted storage receipt must be an object")
        storage_receipt = dict(storage_receipt)
        if canonical_json_bytes(storage_receipt) != receipt_payload:
            raise CampaignControllerError("persisted storage receipt bytes are not canonical")
        expected_request_id = receipt_path.name.removesuffix(".json")
        if (
            set(storage_receipt)
            != {
                "schema_version",
                "schema_id",
                "request_id",
                "command_sha256",
                "operation",
                "status",
                "host",
                "host_identity_sha256",
                "root",
                "completed_unix_ms",
                "result",
                "receipt_sha256",
            }
            or storage_receipt["schema_version"] != 1
            or storage_receipt["schema_id"] != REMOTE_RECEIPT_SCHEMA
            or storage_receipt["request_id"] != expected_request_id
            or storage_receipt["status"] != "ok"
            or storage_receipt["host"] != STORAGE_HOST
            or storage_receipt["root"] != str(CAMPAIGN_ROOT)
            or not _is_sha256(storage_receipt["command_sha256"])
            or not _is_sha256(storage_receipt["host_identity_sha256"])
            or not isinstance(storage_receipt["completed_unix_ms"], int)
            or isinstance(storage_receipt["completed_unix_ms"], bool)
            or storage_receipt["completed_unix_ms"] < 0
            or not isinstance(storage_receipt["result"], Mapping)
            or storage_receipt["receipt_sha256"]
            != content_sha256(storage_receipt, hash_field="receipt_sha256")
            or storage_receipt["receipt_sha256"]
            != artifact["storage_receipt_sha256"]
        ):
            raise CampaignControllerError(
                "persisted storage receipt identity or self-hash differs"
            )
        result = dict(storage_receipt["result"])
        if result.get("payload_size") != 0 or result.get("payload_sha256") != EMPTY_SHA256:
            raise CampaignControllerError("artifact publication receipt has a payload")
        operation = storage_receipt["operation"]
        if operation in {"put-file", "put-stream"}:
            _verify_direct_publication(paths, artifact, result, operation=operation)
        elif operation == "transaction-commit":
            _verify_transaction_publication(paths, artifact, result)
        else:
            raise CampaignControllerError(
                "storage receipt operation did not publish an immutable work artifact: "
                f"{operation}"
            )
        evidence.append(
            {
                "artifact_path": artifact["path"],
                "storage_receipt_relative": locator,
                "storage_receipt_sha256": storage_receipt["receipt_sha256"],
                "operation": operation,
            }
        )
    return {
        "artifacts_verified": len(evidence),
        "evidence": evidence,
    }


def _verify_direct_publication(
    paths: ControllerPaths,
    artifact: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    common = {
        "relative",
        "sha256",
        "size",
        "mode",
        "previous_sha256",
        "payload_size",
        "payload_sha256",
    }
    operation_fields = common if operation == "put-file" else common | {"max_bytes"}
    if (
        set(result) != operation_fields
        or result.get("relative") != artifact["path"]
        or result.get("size") != artifact["bytes"]
        or isinstance(result.get("size"), bool)
        or result.get("sha256") != artifact["sha256"]
        or result.get("mode") != "0o400"
        or (
            result.get("previous_sha256") is not None
            and not _is_sha256(result.get("previous_sha256"))
        )
        or (
            "max_bytes" in result
            and (
                not isinstance(result["max_bytes"], int)
                or isinstance(result["max_bytes"], bool)
                or result["max_bytes"] < artifact["bytes"]
            )
        )
    ):
        raise CampaignControllerError("direct publication receipt differs from artifact")
    _verify_artifact_file(paths, artifact, expected_modes={0o400})


def _verify_transaction_publication(
    paths: ControllerPaths,
    artifact: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    required_result = {
        "transaction_id",
        "target_relative",
        "manifest_sha256",
        "object_count",
        "committed",
        "payload_size",
        "payload_sha256",
    }
    if set(result) != required_result:
        raise CampaignControllerError("transaction receipt result schema differs")
    target = _validate_remote_relative_path(
        result.get("target_relative"), "transaction receipt target"
    )
    artifact_path = _validate_remote_relative_path(artifact["path"], "receipt artifact")
    transaction_id = result.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or not transaction_id
        or "/" in transaction_id
        or transaction_id in {".", ".."}
        or not artifact_path.startswith(f"{target}/")
        or result.get("committed") is not True
        or not _is_sha256(result.get("manifest_sha256"))
        or not isinstance(result.get("object_count"), int)
        or isinstance(result.get("object_count"), bool)
        or result["object_count"] < 1
    ):
        raise CampaignControllerError("transaction receipt target does not contain artifact")
    _verify_immutable_directory(paths, target, expected_mode=0o500)
    manifest_relative = f"{target}/.r2-map-transaction.json"
    _, payload = _read_immutable_contained_file(
        paths,
        manifest_relative,
        label="transaction provenance manifest",
        allowed_modes={0o400},
        maximum_bytes=2 << 20,
    )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignControllerError("transaction provenance manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise CampaignControllerError("transaction provenance manifest must be an object")
    manifest = dict(manifest)
    if (
        set(manifest)
        != {
            "schema_version",
            "schema_id",
            "transaction_id",
            "target_relative",
            "objects",
            "manifest_sha256",
        }
        or canonical_json_bytes(manifest) != payload
        or manifest.get("schema_version") != 1
        or manifest.get("schema_id") != REMOTE_TRANSACTION_SCHEMA
        or manifest.get("transaction_id") != transaction_id
        or manifest.get("target_relative") != target
        or manifest.get("manifest_sha256")
        != content_sha256(manifest, hash_field="manifest_sha256")
        or result.get("manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise CampaignControllerError("transaction provenance identity or hash differs")
    relative = artifact_path[len(target) + 1 :]
    objects = manifest.get("objects")
    if (
        not isinstance(objects, list)
        or len(objects) != result["object_count"]
        or not objects
    ):
        raise CampaignControllerError("transaction provenance object list is invalid")
    normalized_objects = []
    seen = set()
    for item in objects:
        if (
            not isinstance(item, Mapping)
            or set(item)
            not in (
                {"relative", "sha256", "size"},
                {"relative", "sha256", "size", "mode"},
            )
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] < 0
            or not _is_sha256(item.get("sha256"))
            or item.get("mode", "0400") not in {"0400", "0500"}
        ):
            raise CampaignControllerError("transaction object descriptor is invalid")
        object_relative = _validate_remote_relative_path(
            item.get("relative"),
            "transaction object",
            require_authorized_top_level=False,
        )
        if object_relative in seen or object_relative == ".r2-map-transaction.json":
            raise CampaignControllerError("transaction object identity is duplicated")
        seen.add(object_relative)
        normalized_objects.append(dict(item))
    if normalized_objects != sorted(normalized_objects, key=lambda item: item["relative"]):
        raise CampaignControllerError("transaction objects are not canonically ordered")
    matching = [
        item
        for item in normalized_objects
        if isinstance(item, Mapping) and item.get("relative") == relative
    ]
    if len(matching) != 1:
        raise CampaignControllerError("transaction provenance does not uniquely bind artifact")
    descriptor = matching[0]
    expected_mode = 0o500 if descriptor.get("mode") == "0500" else 0o400
    if (
        descriptor.get("size") != artifact["bytes"]
        or descriptor.get("sha256") != artifact["sha256"]
    ):
        raise CampaignControllerError("transaction object descriptor differs from artifact")
    _verify_artifact_file(paths, artifact, expected_modes={expected_mode})


def _verify_artifact_file(
    paths: ControllerPaths,
    artifact: Mapping[str, Any],
    *,
    expected_modes: set[int],
) -> None:
    if (
        not isinstance(artifact["bytes"], int)
        or isinstance(artifact["bytes"], bool)
        or not 0 <= artifact["bytes"] <= CAMPAIGN_BUDGET_BYTES
    ):
        raise CampaignControllerError("receipt artifact size exceeds the campaign bound")
    descriptor, metadata, _ = _open_immutable_contained_file(
        paths,
        artifact["path"],
        label="receipt artifact",
        allowed_modes=expected_modes,
        maximum_bytes=CAMPAIGN_BUDGET_BYTES,
    )
    digest = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed += len(chunk)
            if observed > artifact["bytes"]:
                raise CampaignControllerError("receipt artifact size differs")
            digest.update(chunk)
        _verify_open_file_unchanged(descriptor, metadata, "receipt artifact")
    finally:
        os.close(descriptor)
    if observed != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
        raise CampaignControllerError("receipt artifact bytes or SHA-256 differ")


def _read_immutable_contained_file(
    paths: ControllerPaths,
    relative: str,
    *,
    label: str,
    allowed_modes: set[int],
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    descriptor, metadata, current = _open_immutable_contained_file(
        paths,
        relative,
        label=label,
        allowed_modes=allowed_modes,
        maximum_bytes=maximum_bytes,
    )
    try:
        payload = b""
        remaining = metadata.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise CampaignControllerError(f"{label} ended while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CampaignControllerError(f"{label} grew while being read")
        payload = b"".join(chunks)
        _verify_open_file_unchanged(descriptor, metadata, label)
    finally:
        os.close(descriptor)
    return current, payload


def _open_immutable_contained_file(
    paths: ControllerPaths,
    relative: str,
    *,
    label: str,
    allowed_modes: set[int],
    maximum_bytes: int,
) -> tuple[int, os.stat_result, Path]:
    relative = _validate_remote_relative_path(relative, label)
    root = paths.root.resolve(strict=True)
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    root_metadata = os.fstat(root_descriptor)
    current_descriptor = root_descriptor
    current = root
    parts = PurePosixPath(relative).parts
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if index < len(parts) - 1:
                flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                next_descriptor = os.open(part, flags, dir_fd=current_descriptor)
            except OSError as error:
                raise CampaignControllerError(f"{label} cannot be resolved safely") from error
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
            current = current / part
            metadata = os.fstat(current_descriptor)
            if (
                metadata.st_dev != root_metadata.st_dev
                or metadata.st_uid != root_metadata.st_uid
                or metadata.st_gid != root_metadata.st_gid
            ):
                raise CampaignControllerError(f"{label} crosses an unsafe ownership boundary")
            if index < len(parts) - 1 and (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise CampaignControllerError(f"{label} ancestor is mutable or not a directory")
        metadata = os.fstat(current_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) not in allowed_modes
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_bytes
        ):
            raise CampaignControllerError(
                f"{label} is mutable, oversized, or has unsafe metadata"
            )
        if root_descriptor != current_descriptor:
            os.close(root_descriptor)
        return current_descriptor, metadata, current
    except BaseException:
        if current_descriptor != root_descriptor:
            os.close(current_descriptor)
        os.close(root_descriptor)
        raise


def _verify_open_file_unchanged(
    descriptor: int, before: os.stat_result, label: str
) -> None:
    after = os.fstat(descriptor)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise CampaignControllerError(f"{label} changed while being verified")


def _verify_immutable_directory(
    paths: ControllerPaths, relative: str, *, expected_mode: int
) -> None:
    relative = _validate_remote_relative_path(relative, "immutable transaction target")
    root = paths.root.resolve(strict=True)
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    root_metadata = os.fstat(root_descriptor)
    current_descriptor = root_descriptor
    try:
        for part in PurePosixPath(relative).parts:
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_descriptor,
            )
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
            metadata = os.fstat(current_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_dev != root_metadata.st_dev
                or metadata.st_uid != root_metadata.st_uid
                or metadata.st_gid != root_metadata.st_gid
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise CampaignControllerError("transaction target crosses an unsafe directory")
        metadata = os.fstat(current_descriptor)
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise CampaignControllerError("transaction target directory is mutable")
    except OSError as error:
        raise CampaignControllerError("transaction target cannot be resolved safely") from error
    finally:
        if current_descriptor != root_descriptor:
            os.close(current_descriptor)
        os.close(root_descriptor)


def write_packets(paths: ControllerPaths, packets: Sequence[Mapping[str, Any]]) -> None:
    paths.packets.mkdir(mode=0o700, parents=True, exist_ok=True)
    for packet in packets:
        packet = validate_work_packet(packet)
        _write_immutable_json(paths.packets / f"{packet['task_id']}.json", packet)


def install_packets(
    paths: ControllerPaths,
    packets: Sequence[Mapping[str, Any]],
    *,
    now_ms: int,
) -> dict[str, Any]:
    packets = tuple(validate_work_packet(packet) for packet in packets)
    with research_queue.locked_queue(paths.queue) as queue:
        existing = {task["id"]: task for task in queue["tasks"]}
        specifications = []
        for packet in packets:
            expected = queue_task_for_packet(packet, created_unix_ms=now_ms)
            observed = existing.get(packet["task_id"])
            if observed is None:
                specifications.append(
                    {
                        key: expected[key]
                        for key in (
                            "id",
                            "title",
                            "experiment_id",
                            "decision",
                            "workload_class",
                            "priority",
                            "decision_value",
                            "expected_runtime_seconds",
                            "critical_path",
                            "decision_terminal",
                            "compatible_hosts",
                            "dependencies",
                            "command",
                            "artifact_path",
                            "stop_rule",
                            "resources",
                        )
                    }
                )
            elif _queue_static_identity(observed) != _queue_static_identity(expected):
                raise CampaignControllerError(f"queue task {packet['task_id']} differs from packet")
        if specifications:
            research_queue.add_tasks(queue, specifications, now_ms=now_ms)
        _set_exact_queue_intents(
            queue,
            Phase(packets[0]["phase"]) if packets else Phase(read_state(paths.state)["phase"]),
            now_ms,
        )
    return reconcile(paths, now_ms=now_ms)


def import_receipt(
    paths: ControllerPaths,
    *,
    source: Path,
) -> dict[str, Any]:
    try:
        relative = source.resolve(strict=True).relative_to(paths.incoming.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CampaignControllerError("incoming receipt escapes the registered inbox") from error
    if len(relative.parts) != 2 or relative.parts[0] not in ALLOWED_HOSTS:
        raise CampaignControllerError("incoming receipt must be directly beneath its host inbox")
    try:
        raw = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignControllerError(f"cannot read incoming receipt: {error}") from error
    task_id = raw.get("task_id") if isinstance(raw, Mapping) else None
    if not isinstance(task_id, str):
        raise CampaignControllerError("incoming receipt has no task identity")
    packet = _read_packet(paths, task_id)
    try:
        receipt = validate_receipt(raw, packet=packet)
        if relative.parts[0] != receipt["host"]:
            raise CampaignControllerError(
                "incoming receipt directory differs from receipt host"
            )
        verify_receipt_storage_evidence(paths, receipt, packet=packet)
    except CampaignControllerError as error:
        _write_stop(paths, f"invalid receipt for {task_id}: {error}", 0)
        raise
    with research_queue.locked_queue(paths.queue) as queue:
        task = _queue_task(queue, task_id)
        result = task.get("result")
        if task["status"] != "completed" or not isinstance(result, Mapping):
            raise CampaignControllerError("queue task must complete before receipt import")
        if result.get("host") != receipt["host"]:
            raise CampaignControllerError("queue completion host differs from receipt")
        if len(task["attempts"]) > MAX_ATTEMPTS:
            raise CampaignControllerError("task exceeded the registered retry ceiling")
    destination = paths.receipts / f"{task_id}.json"
    _write_immutable_json(destination, receipt)
    reconcile(paths, now_ms=receipt["completed_unix_ms"])
    return receipt


def import_benchmark_feed(
    paths: ControllerPaths,
    *,
    feed_path: Path,
    aggregate_task_id: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Stamp and upsert one receipt-bound deterministic benchmark ledger feed.

    Host shards, aggregate reports, and the feed itself remain byte-stable. The
    only time-bearing mutation happens in the canonical john2 control-plane
    state, using the imported John1 aggregate receipt's completion time.
    """
    state_lock_path = paths.state.with_name(f".{paths.state.name}.lock")
    state_lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_lock_descriptor = os.open(
        state_lock_path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600
    )
    with os.fdopen(state_lock_descriptor, "a+", closefd=True) as state_lock:
        fcntl.flock(state_lock.fileno(), fcntl.LOCK_EX)
        state = read_state(paths.state)
        if state["state_sha256"] != expected_state_sha256:
            raise CampaignControllerError("benchmark feed campaign-state CAS differs")

        packet = _read_packet(paths, aggregate_task_id)
        if (
            packet["controller_state_sha256"] != state["state_sha256"]
            or packet["phase"] != state["phase"]
            or packet["task_kind"] != "aggregate"
            or packet["aggregate_kind"] not in {"longitudinal-benchmark", "candidate-gate"}
            or packet["host"] != "john1"
        ):
            raise CampaignControllerError(
                "benchmark feed requires the current John1 benchmark aggregate packet"
            )
        receipt_path = paths.receipts / f"{aggregate_task_id}.json"
        try:
            receipt = validate_receipt(json.loads(receipt_path.read_text()), packet=packet)
            verify_receipt_storage_evidence(paths, receipt, packet=packet)
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignControllerError(
                f"cannot read imported benchmark aggregate receipt: {error}"
            ) from error
        with research_queue.locked_queue(paths.queue) as queue:
            task = _queue_task(queue, aggregate_task_id)
            result = task.get("result")
            if (
                task["status"] != "completed"
                or not isinstance(result, Mapping)
                or result.get("host") != "john1"
            ):
                raise CampaignControllerError(
                    "benchmark aggregate queue task must complete on John1 before feed import"
                )

        try:
            canonical_root = paths.root.resolve(strict=True)
            canonical_feed = feed_path.resolve(strict=True)
            canonical_feed.relative_to(canonical_root)
            metadata = canonical_feed.stat()
            if not canonical_feed.is_file() or metadata.st_size > 1 << 20:
                raise CampaignControllerError(
                    "benchmark ledger feed must be one regular compact file at most 1 MiB"
                )
            feed_bytes = canonical_feed.read_bytes()
        except CampaignControllerError:
            raise
        except (OSError, ValueError) as error:
            raise CampaignControllerError(
                "benchmark ledger feed escapes the campaign root or cannot be read"
            ) from error
        feed_sha256 = hashlib.sha256(feed_bytes).hexdigest()
        matching_artifacts = []
        for artifact in receipt["artifacts"]:
            if artifact["label"] != "benchmark-ledger-feed":
                continue
            claimed = Path(artifact["path"])
            if not claimed.is_absolute():
                claimed = paths.root / claimed
            try:
                claimed = claimed.resolve(strict=True)
            except OSError as error:
                raise CampaignControllerError(
                    "benchmark ledger feed receipt path cannot be resolved"
                ) from error
            if claimed == canonical_feed and artifact["sha256"] == feed_sha256:
                matching_artifacts.append(artifact)
        if len(matching_artifacts) != 1:
            raise CampaignControllerError(
                "benchmark ledger feed is not uniquely bound by the imported aggregate receipt"
            )

        try:
            raw_feed = json.loads(feed_bytes)
        except json.JSONDecodeError as error:
            raise CampaignControllerError("benchmark ledger feed is invalid JSON") from error
        if not isinstance(raw_feed, Mapping):
            raise CampaignControllerError("benchmark ledger feed must be an object")
        _reject_john4(raw_feed)
        try:
            experiment_ledger.validate_experiment(raw_feed)
        except experiment_ledger.LedgerError as error:
            raise CampaignControllerError(f"benchmark ledger feed is invalid: {error}") from error
        if (
            raw_feed["id"] == CONTROLLER_EXPERIMENT_ID
            or raw_feed["status"] != "completed"
            or raw_feed["started_unix_ms"] != 0
            or raw_feed["completed_unix_ms"] != 0
            or raw_feed["updated_unix_ms"] != 0
        ):
            raise CampaignControllerError(
                "deterministic benchmark ledger feed requires zero placeholder times "
                "and a distinct completed experiment"
            )

        completion = receipt["completed_unix_ms"]
        stamped = copy.deepcopy(dict(raw_feed))
        stamped["started_unix_ms"] = completion
        stamped["completed_unix_ms"] = completion
        stamped["updated_unix_ms"] = completion
        binding_note = (
            f"Imported by John1 from deterministic feed SHA-256 {feed_sha256} "
            f"bound by receipt {receipt['receipt_sha256']}."
        )
        if binding_note not in stamped["notes"]:
            stamped["notes"].append(binding_note)
        experiment_ledger.validate_experiment(stamped)

        # Recheck the state while holding its CAS lock, then take the existing
        # ledger lock and use its established idempotent upsert operation.
        if read_state(paths.state)["state_sha256"] != expected_state_sha256:
            raise CampaignControllerError("benchmark feed campaign state changed before upsert")
        with experiment_ledger.locked_ledger(paths.ledger) as ledger:
            experiment_ledger.upsert(ledger, stamped)
        fcntl.flock(state_lock.fileno(), fcntl.LOCK_UN)
    return {
        "experiment": stamped,
        "feed_path": str(canonical_feed),
        "feed_sha256": feed_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "state_sha256": expected_state_sha256,
        "completion_unix_ms": completion,
    }


def reconcile(paths: ControllerPaths, *, now_ms: int) -> dict[str, Any]:
    state = read_state(paths.state)
    packets = _phase_packets_on_disk(paths, state)
    with research_queue.locked_queue(paths.queue) as queue:
        research_queue.expire_leases(queue, now_ms=now_ms)
        for packet in packets:
            task = _queue_task(queue, packet["task_id"])
            if len(task["attempts"]) >= MAX_ATTEMPTS and task["status"] in {
                "ready",
                "failed",
            }:
                _write_stop(paths, f"task {task['id']} exhausted its retry ceiling", now_ms)
                raise CampaignControllerError(f"task {task['id']} exhausted its retry ceiling")
            expected = queue_task_for_packet(packet, created_unix_ms=task["created_unix_ms"])
            if _queue_static_identity(task) != _queue_static_identity(expected):
                _write_stop(paths, f"queue task {task['id']} drifted", now_ms)
                raise CampaignControllerError(f"queue task {task['id']} drifted")
            receipt_path = paths.receipts / f"{task['id']}.json"
            if task["status"] == "completed" and receipt_path.exists():
                validate_receipt(json.loads(receipt_path.read_text()), packet=packet)
        _set_exact_queue_intents(queue, Phase(state["phase"]), now_ms)
        queue_snapshot = copy.deepcopy(queue)
    experiment = _experiment_from_state(state, packets, queue_snapshot, now_ms=now_ms)
    with experiment_ledger.locked_ledger(paths.ledger) as ledger:
        experiment_ledger.upsert(ledger, experiment)
    dashboard = publish_dashboard_inputs(paths, state=state, packets=packets, queue=queue_snapshot)
    return {
        "state_sha256": state["state_sha256"],
        "phase": state["phase"],
        "packet_count": len(packets),
        "queue_task_count": len(queue_snapshot["tasks"]),
        "ledger_experiment_id": CONTROLLER_EXPERIMENT_ID,
        "dashboard_inputs_sha256": dashboard["dashboard_inputs_sha256"],
    }


def phase_barrier(paths: ControllerPaths) -> dict[str, Any]:
    state = read_state(paths.state)
    packets = _phase_packets_on_disk(paths, state)
    with research_queue.locked_queue(paths.queue) as queue:
        required = []
        for packet in packets:
            task = _queue_task(queue, packet["task_id"])
            if task["status"] != "completed":
                raise CampaignControllerError(f"phase task {task['id']} is not complete")
            receipt_path = paths.receipts / f"{task['id']}.json"
            if not receipt_path.exists():
                raise CampaignControllerError(f"phase task {task['id']} has no imported receipt")
            receipt = validate_receipt(json.loads(receipt_path.read_text()), packet=packet)
            verify_receipt_storage_evidence(paths, receipt, packet=packet)
            required.append(receipt)
    return {
        "phase": state["phase"],
        "state_sha256": state["state_sha256"],
        "receipts": required,
        "receipt_count": len(required),
    }


def advance_campaign(
    paths: ControllerPaths,
    *,
    commands: Mapping[str, Sequence[str]],
    artifact_root: str,
    reason: str,
    now: str,
    now_ms: int,
    synthetic: bool = False,
    fault_injector: Any | None = None,
) -> dict[str, Any]:
    """Cross exactly one barrier, CAS the state, then install the next packet DAG."""
    if paths.stop.exists():
        raise CampaignControllerError(
            "campaign controller is stopped; inspect controller-stop.json"
        )
    current = read_state(paths.state)
    source = Phase(current["phase"])
    barrier = (
        phase_barrier(paths)
        if phase_templates(source)
        else {
            "phase": source.value,
            "state_sha256": current["state_sha256"],
            "receipts": [],
            "receipt_count": 0,
        }
    )
    next_phase, classification = _next_phase(source, barrier["receipts"])
    transition_arguments = _transition_artifacts(source, barrier["receipts"])
    proposed = transition_state(
        current,
        next_phase,
        reason=reason,
        generation_manifest_sha256=transition_arguments.get("generation_manifest_sha256"),
        candidate_checkpoint_sha256=transition_arguments.get("candidate_checkpoint_sha256"),
        completed_shard_hosts=transition_arguments.get("completed_shard_hosts"),
        now=now,
    )
    write_state(paths.state, proposed, expected_current=current)
    if fault_injector is not None:
        fault_injector("after-state-cas")
    _append_history(
        paths,
        {
            "schema_version": 1,
            "from_state_sha256": current["state_sha256"],
            "to_state_sha256": proposed["state_sha256"],
            "from_phase": source.value,
            "to_phase": proposed["phase"],
            "classification": classification,
            "at_unix_ms": now_ms,
        },
    )
    if fault_injector is not None:
        fault_injector("after-history")
    apply_stop_rules(paths, now_ms=now_ms)
    templates = phase_templates(Phase(proposed["phase"]))
    packets: tuple[dict[str, Any], ...] = ()
    if templates:
        packets = build_phase_packets(
            proposed,
            commands=commands,
            artifact_root=artifact_root,
            synthetic=synthetic,
        )
        write_packets(paths, packets)
        install_packets(paths, packets, now_ms=now_ms)
    reconciliation = reconcile(paths, now_ms=now_ms)
    return {
        "from_phase": source.value,
        "to_phase": proposed["phase"],
        "state_sha256": proposed["state_sha256"],
        "packet_count": len(packets),
        "classification": classification,
        "reconciliation": reconciliation,
    }


def recover_current_phase(
    paths: ControllerPaths,
    *,
    commands: Mapping[str, Sequence[str]],
    artifact_root: str,
    now_ms: int,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Idempotently repair a crash between state CAS, packet install, and ledger projection."""
    state = read_state(paths.state)
    history = _read_history(paths)
    if not history or history[-1]["to_state_sha256"] != state["state_sha256"]:
        transition = state.get("last_transition")
        if not isinstance(transition, Mapping):
            raise CampaignControllerError("cannot recover an initial state without a transition")
        classification = None
        if Phase(state["phase"]) is Phase.CANDIDATE_REJECTED:
            classification = "inconclusive"
        elif Phase(state["phase"]) is Phase.INCUMBENT_PROMOTED:
            classification = "promote"
        _append_history(
            paths,
            {
                "schema_version": 1,
                "from_state_sha256": state["previous_state_sha256"],
                "to_state_sha256": state["state_sha256"],
                "from_phase": transition["from"],
                "to_phase": transition["to"],
                "classification": classification,
                "at_unix_ms": now_ms,
                "recovered": True,
            },
        )
    templates = phase_templates(Phase(state["phase"]))
    if templates:
        packets = build_phase_packets(
            state,
            commands=commands,
            artifact_root=artifact_root,
            synthetic=synthetic,
        )
        write_packets(paths, packets)
        install_packets(paths, packets, now_ms=now_ms)
    return reconcile(paths, now_ms=now_ms)


def apply_stop_rules(paths: ControllerPaths, *, now_ms: int) -> dict[str, Any]:
    consecutive = _consecutive_rejections(paths)
    stopped = consecutive >= 3
    if stopped:
        _write_stop(paths, "three consecutive candidates were rejected or inconclusive", now_ms)
    return {"consecutive_rejections": consecutive, "stopped": stopped}


def complete_synthetic_phase(
    paths: ControllerPaths,
    *,
    now_ms: int,
    gate_classification: str = "promote",
) -> list[dict[str, Any]]:
    """Exercise the real queue/import path with deterministic synthetic receipts."""
    state = read_state(paths.state)
    packets = _phase_packets_on_disk(paths, state)
    imported = []
    packet_by_id = {packet["task_id"]: packet for packet in packets}
    while True:
        progress = False
        with research_queue.locked_queue(paths.queue) as queue:
            research_queue.refresh_dependencies(queue)
            for host in ALLOWED_HOSTS:
                claim = research_queue.claim_next(
                    queue,
                    host=host,
                    lease_seconds=60,
                    now_ms=now_ms,
                )
                if claim is None:
                    continue
                packet = packet_by_id.get(claim["id"])
                if packet is None:
                    raise CampaignControllerError("existing queue selected a non-phase task")
                research_queue.finish_task(
                    queue,
                    task_id=packet["task_id"],
                    host=host,
                    token=claim["claim"]["token"],
                    outcome="completed",
                    artifact=claim["artifact_path"],
                    now_ms=now_ms + 1,
                )
                progress = True
        queue_snapshot = research_queue.load_queue(paths.queue)
        for packet in packets:
            destination = paths.receipts / f"{packet['task_id']}.json"
            if destination.exists():
                continue
            if _queue_task(queue_snapshot, packet["task_id"])["status"] != "completed":
                continue
            classification = None
            if packet["aggregate_kind"] == "candidate-gate":
                classification = gate_classification
            receipt = make_synthetic_receipt(
                packet,
                completed_unix_ms=now_ms + 1,
                classification=classification,
            )
            receipt = _install_synthetic_storage_evidence(paths, packet, receipt)
            incoming = paths.incoming / packet["host"] / f"{packet['task_id']}.json"
            _write_immutable_json(incoming, receipt)
            imported.append(import_receipt(paths, source=incoming))
        if all((paths.receipts / f"{packet['task_id']}.json").exists() for packet in packets):
            return imported
        if not progress:
            raise CampaignControllerError("synthetic phase made no queue progress")


def run_isolated_dry_run(paths: ControllerPaths, *, now_ms: int = 1_000) -> dict[str, Any]:
    """Traverse bootstrap plus rejection and promotion round shapes without W7 work."""
    initialize_controller(paths, now_ms=now_ms)
    operations = {template.operation for phase in Phase for template in phase_templates(phase)}
    commands = {operation: ["/usr/bin/true"] for operation in operations}
    artifact_root = "reports/synthetic-artifacts"
    steps: list[dict[str, Any]] = []

    def advance() -> None:
        index = len(steps) + 1
        steps.append(
            advance_campaign(
                paths,
                commands=commands,
                artifact_root=artifact_root,
                reason="W6 isolated synthetic transition proof",
                now=f"1970-01-01T00:{index:02d}:00.000Z",
                now_ms=now_ms + index * 10,
                synthetic=True,
            )
        )

    def complete(classification: str = "promote") -> None:
        complete_synthetic_phase(
            paths,
            now_ms=now_ms + len(steps) * 10 + 1,
            gate_classification=classification,
        )

    # Bootstrap through C[0].
    advance()  # contracts-ready -> bootstrap-generating
    complete()
    advance()  # -> bootstrap-validated
    advance()  # -> bootstrap-training
    complete()
    advance()  # -> bootstrap-candidate-gate
    complete()
    advance()  # -> incumbent-promoted

    # Round zero follows the rejection branch.
    advance()  # -> round-allocated
    advance()  # -> generating
    complete()
    advance()  # -> local-shards-complete
    complete()
    advance()  # -> collected-and-validated
    advance()  # -> training-and-benchmarking
    complete()
    advance()  # -> candidate-verified-benchmark-complete
    advance()  # -> paired-candidate-gate
    complete("reject")
    advance()  # -> candidate-rejected

    # Round one follows the promotion branch.
    advance()  # -> round-allocated
    advance()  # -> generating
    complete()
    advance()  # -> local-shards-complete
    complete()
    advance()  # -> collected-and-validated
    advance()  # -> training-and-benchmarking
    complete()
    advance()  # -> candidate-verified-benchmark-complete
    advance()  # -> paired-candidate-gate
    complete("promote")
    advance()  # -> incumbent-promoted C[1]

    final = read_state(paths.state)
    queue = research_queue.load_queue(paths.queue)
    ledger = experiment_ledger.read_ledger(paths.ledger)
    result: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.w6-isolated-dry-run.v1",
        "campaign_id": CAMPAIGN_ID,
        "root": str(paths.root),
        "transition_count": len(steps),
        "history_count": len(_read_history(paths)),
        "final_phase": final["phase"],
        "final_promotion_index": final["promotion_index"],
        "final_round_index": final["round_index"],
        "final_state_sha256": final["state_sha256"],
        "queue_task_count": len(queue["tasks"]),
        "completed_queue_tasks": sum(task["status"] == "completed" for task in queue["tasks"]),
        "work_receipt_count": len(tuple(paths.receipts.glob("r2map-*.json"))),
        "storage_receipt_count": len(tuple(paths.receipts.glob("req-*.json"))),
        "receipt_count": len(tuple(paths.receipts.glob("*.json"))),
        "ledger_experiment_count": len(ledger["experiments"]),
        "stop_file_present": paths.stop.exists(),
    }
    result["dry_run_sha256"] = content_sha256(result, hash_field="dry_run_sha256")
    _write_immutable_json(paths.root / "dry-run-report.json", result)
    return result


def publish_dashboard_inputs(
    paths: ControllerPaths,
    *,
    state: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = []
    for packet in packets:
        path = paths.receipts / f"{packet['task_id']}.json"
        if path.exists():
            receipt = validate_receipt(json.loads(path.read_text()), packet=packet)
            receipts.append(
                {
                    "task_id": receipt["task_id"],
                    "host": receipt["host"],
                    "task_kind": receipt["task_kind"],
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
    value: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.controller-dashboard-inputs.v1",
        "campaign_id": CAMPAIGN_ID,
        "state_sha256": state["state_sha256"],
        "phase": state["phase"],
        "host_intents": state["host_intents"],
        "queue_summary": research_queue.queue_summary(dict(queue)),
        "receipts": sorted(receipts, key=lambda item: item["task_id"]),
    }
    value["dashboard_inputs_sha256"] = content_sha256(value, hash_field="dashboard_inputs_sha256")
    _write_atomic_json(paths.dashboard_inputs / "controller.json", value)
    _write_atomic_json(
        paths.dashboard_inputs / "host-receipts.json",
        _dashboard_host_receipts(state, packets, queue, paths),
    )
    _write_atomic_json(
        paths.dashboard_inputs / "training-progress.json",
        {
            "active": Phase(state["phase"])
            in {Phase.BOOTSTRAP_TRAINING, Phase.TRAINING_AND_BENCHMARKING},
            "latest_verified_checkpoint": None,
            "current_step": None,
            "total_steps": None,
            "examples_per_second": None,
            "loss_samples": [],
        },
    )
    _write_atomic_json(
        paths.dashboard_inputs / "benchmark-aggregate.json",
        {
            "active": Phase(state["phase"])
            in {Phase.BOOTSTRAP_CANDIDATE_GATE, Phase.TRAINING_AND_BENCHMARKING},
            "stage": None,
            "pairs_completed": 0,
            "pairs_total": None,
            "eta_seconds": None,
            "throughput_games_per_second": None,
            "peak_rss_bytes": None,
            "swap_delta_bytes": None,
            "focal": None,
            "paired_delta": None,
            "classification": "pending",
        },
    )
    return value


def _dashboard_host_receipts(
    state: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    queue: Mapping[str, Any],
    paths: ControllerPaths,
) -> dict[str, Any]:
    result = {}
    for host in ALLOWED_HOSTS:
        host_packets = [packet for packet in packets if packet["host"] == host]
        completed = sum(
            _queue_task(queue, packet["task_id"])["status"] == "completed"
            for packet in host_packets
        )
        generation_target = sum(
            int(packet["seed_lease"]["count"])
            for packet in host_packets
            if packet["task_kind"] == "generate"
        )
        benchmark_total = sum(
            int(packet["seed_lease"]["count"])
            for packet in host_packets
            if packet["task_kind"] in {"longitudinal-benchmark", "candidate-gate"}
        )
        imported = []
        for packet in host_packets:
            receipt_path = paths.receipts / f"{packet['task_id']}.json"
            if receipt_path.exists():
                imported.append(json.loads(receipt_path.read_text()))
        latest = imported[-1] if imported else None
        gameplay_receipts = [
            receipt for receipt in imported if receipt["used_seed_prefix"] is not None
        ]
        used = None if not gameplay_receipts else gameplay_receipts[-1]["used_seed_prefix"]
        result[host] = {
            "intent": state["host_intents"][host],
            "detail": None if not host_packets else f"{completed}/{len(host_packets)} tasks",
            "generation_games_completed": sum(
                int(receipt["used_seed_prefix"]["used_count"])
                for receipt in imported
                if receipt["task_kind"] == "generate"
            ),
            "generation_games_target": generation_target or None,
            "generation_seed_prefix": (
                None if used is None else f"{used['used_count']}:{used['last_index']}"
            ),
            "benchmark_pairs_completed": sum(
                int(receipt["used_seed_prefix"]["used_count"])
                for receipt in imported
                if receipt["task_kind"] in {"longitudinal-benchmark", "candidate-gate"}
            ),
            "benchmark_pairs_total": benchmark_total or None,
            "eta_seconds": None,
            "throughput_games_per_second": None,
            "rss_bytes": None if latest is None else latest["metrics"]["maximum_rss_bytes"],
            "swap_delta_bytes": (
                None if latest is None else latest["metrics"]["system_swap_delta_bytes"]
            ),
        }
    safety_path = paths.root / "control/host-safety.json"
    if safety_path.exists():
        from cascadia_mlx.r2_map_apfs_lifecycle import host_dashboard_receipt

        result["john1"] = host_dashboard_receipt(
            state["host_intents"]["john1"], json.loads(safety_path.read_text())
        )
    return result


def _next_phase(
    source: Phase,
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[Phase, str | None]:
    direct = {
        Phase.CONTRACTS_READY: Phase.BOOTSTRAP_GENERATING,
        Phase.BOOTSTRAP_GENERATING: Phase.BOOTSTRAP_VALIDATED,
        Phase.BOOTSTRAP_VALIDATED: Phase.BOOTSTRAP_TRAINING,
        Phase.BOOTSTRAP_TRAINING: Phase.BOOTSTRAP_CANDIDATE_GATE,
        Phase.INCUMBENT_PROMOTED: Phase.ROUND_ALLOCATED,
        Phase.ROUND_ALLOCATED: Phase.GENERATING,
        Phase.GENERATING: Phase.LOCAL_SHARDS_COMPLETE,
        Phase.LOCAL_SHARDS_COMPLETE: Phase.COLLECTED_AND_VALIDATED,
        Phase.COLLECTED_AND_VALIDATED: Phase.TRAINING_AND_BENCHMARKING,
        Phase.TRAINING_AND_BENCHMARKING: Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE: Phase.PAIRED_CANDIDATE_GATE,
        Phase.CANDIDATE_REJECTED: Phase.ROUND_ALLOCATED,
    }
    if source in direct:
        return direct[source], None
    if source in {Phase.BOOTSTRAP_CANDIDATE_GATE, Phase.PAIRED_CANDIDATE_GATE}:
        classification = _candidate_gate_classification(receipts)
        if source is Phase.BOOTSTRAP_CANDIDATE_GATE and classification != "promote":
            raise CampaignControllerError("bootstrap candidate must pass before C[0] exists")
        target = (
            Phase.INCUMBENT_PROMOTED if classification == "promote" else Phase.CANDIDATE_REJECTED
        )
        return target, classification
    raise CampaignControllerError(f"phase {source.value} has no legal controller successor")


def _transition_artifacts(
    source: Phase,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if source in {Phase.BOOTSTRAP_GENERATING, Phase.GENERATING}:
        result["completed_shard_hosts"] = list(ALLOWED_HOSTS)
    if source in {Phase.BOOTSTRAP_GENERATING, Phase.LOCAL_SHARDS_COMPLETE}:
        result["generation_manifest_sha256"] = _artifact_sha256(receipts, "generation-manifest")
    if source in {Phase.BOOTSTRAP_TRAINING, Phase.TRAINING_AND_BENCHMARKING}:
        result["candidate_checkpoint_sha256"] = _artifact_sha256(receipts, "candidate-checkpoint")
    return result


def _artifact_sha256(receipts: Sequence[Mapping[str, Any]], label: str) -> str:
    matches = [
        artifact["sha256"]
        for receipt in receipts
        for artifact in receipt["artifacts"]
        if artifact["label"] == label
    ]
    if len(matches) != 1:
        raise CampaignControllerError(f"phase receipts require exactly one {label}")
    return matches[0]


def _candidate_gate_classification(receipts: Sequence[Mapping[str, Any]]) -> str:
    values = [
        receipt["metrics"].get("classification")
        for receipt in receipts
        if any(artifact["label"] == "candidate-gate-report" for artifact in receipt["artifacts"])
    ]
    if len(values) != 1 or values[0] not in {"promote", "reject", "inconclusive"}:
        raise CampaignControllerError("candidate gate aggregate classification is invalid")
    return str(values[0])


def _append_history(paths: ControllerPaths, entry: Mapping[str, Any]) -> None:
    paths.history.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = _read_history(paths)
    record = dict(entry)
    record["sequence"] = len(existing)
    record["previous_entry_sha256"] = None if not existing else existing[-1]["entry_sha256"]
    if existing and record.get("from_state_sha256") != existing[-1].get("to_state_sha256"):
        raise CampaignControllerError("controller history does not extend the prior state")
    record["entry_sha256"] = content_sha256(record, hash_field="entry_sha256")
    descriptor = os.open(paths.history, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab", closefd=True) as handle:
        handle.write(canonical_json_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync(paths.history.parent)


def _read_history(paths: ControllerPaths) -> list[dict[str, Any]]:
    if not paths.history.exists():
        return []
    text = paths.history.read_text()
    if text and not text.endswith("\n"):
        raise CampaignControllerError("controller history has an incomplete record")
    records = []
    for line in text.splitlines():
        record = json.loads(line)
        expected_previous = None if not records else records[-1]["entry_sha256"]
        if (
            record.get("sequence") != len(records)
            or record.get("previous_entry_sha256") != expected_previous
            or record.get("entry_sha256") != content_sha256(record, hash_field="entry_sha256")
            or (records and record.get("from_state_sha256") != records[-1].get("to_state_sha256"))
        ):
            raise CampaignControllerError("controller history hash chain differs")
        records.append(record)
    return records


def _consecutive_rejections(paths: ControllerPaths) -> int:
    count = 0
    for record in reversed(_read_history(paths)):
        classification = record.get("classification")
        if classification in {"reject", "inconclusive"}:
            count += 1
        elif classification == "promote":
            break
    return count


def _write_stop(paths: ControllerPaths, reason: str, now_ms: int) -> None:
    value = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.controller-stop.v1",
        "campaign_id": CAMPAIGN_ID,
        "reason": reason,
        "stopped_unix_ms": now_ms,
    }
    value["stop_sha256"] = content_sha256(value, hash_field="stop_sha256")
    _write_immutable_json(paths.stop, value)


def _task_id(state: Mapping[str, Any], operation: str) -> str:
    return f"r2map-r{int(state['revision']):04d}-{operation}"


def _seed_lease(state: Mapping[str, Any], template: TaskTemplate) -> dict[str, Any] | None:
    if template.kind not in {"generate", "longitudinal-benchmark", "candidate-gate"}:
        return None
    host_index = ALLOWED_HOSTS.index(template.host)
    phase = Phase(state["phase"])
    round_index = state.get("round_index") or 0
    if template.kind == "generate":
        if phase is Phase.BOOTSTRAP_GENERATING:
            counts = (33_334, 33_333, 33_333)
            first = sum(counts[:host_index])
            count = counts[host_index]
        else:
            first = (round_index + 1) * (1 << 48) + host_index * GENERATION_LEASE_SIZE
            count = GENERATION_LEASE_SIZE
        purpose = "bootstrap" if phase is Phase.BOOTSTRAP_GENERATING else "generation"
        stride = 1
    elif template.kind == "longitudinal-benchmark":
        first = 0 if template.host == "john2" else 1
        count = 50
        stride = 2
        purpose = "longitudinal-benchmark"
    else:
        first = 0 if template.host == "john2" else 1
        count = 125
        stride = 2
        purpose = "candidate-gate"
    lease = {
        "purpose": purpose,
        "round_index": round_index,
        "host": template.host,
        "first_index": first,
        "count": count,
        "stride": stride,
        "lease_sha256": "",
    }
    lease["lease_sha256"] = content_sha256(lease, hash_field="lease_sha256")
    return lease


def _validate_seed_lease(value: Any, *, kind: str, host: str, phase: Phase) -> None:
    if kind not in {"generate", "longitudinal-benchmark", "candidate-gate"}:
        if value is not None:
            raise CampaignControllerError("non-gameplay packet cannot carry a seed lease")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "purpose",
        "round_index",
        "host",
        "first_index",
        "count",
        "stride",
        "lease_sha256",
    }:
        raise CampaignControllerError("gameplay packet seed lease is invalid")
    integer_fields = ("round_index", "first_index", "count", "stride")
    if any(
        not isinstance(value[field], int) or isinstance(value[field], bool)
        for field in integer_fields
    ):
        raise CampaignControllerError("gameplay packet seed lease integers are invalid")
    if (
        value["host"] != host
        or value["round_index"] < 0
        or value["first_index"] < 0
        or value["count"] <= 0
        or value["stride"] <= 0
    ):
        raise CampaignControllerError("gameplay packet seed lease host or range is invalid")
    host_index = ALLOWED_HOSTS.index(host)
    round_index = value["round_index"]
    if kind == "generate" and phase is Phase.BOOTSTRAP_GENERATING:
        counts = (33_334, 33_333, 33_333)
        expected = ("bootstrap", 0, sum(counts[:host_index]), counts[host_index], 1)
    elif kind == "generate" and phase is Phase.GENERATING:
        expected = (
            "generation",
            round_index,
            (round_index + 1) * (1 << 48) + host_index * GENERATION_LEASE_SIZE,
            GENERATION_LEASE_SIZE,
            1,
        )
    elif kind == "longitudinal-benchmark" and phase is Phase.TRAINING_AND_BENCHMARKING:
        expected = ("longitudinal-benchmark", round_index, 0 if host == "john2" else 1, 50, 2)
    elif kind == "candidate-gate" and phase in {
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        Phase.PAIRED_CANDIDATE_GATE,
    }:
        expected = ("candidate-gate", round_index, 0 if host == "john2" else 1, 125, 2)
    else:
        raise CampaignControllerError("gameplay packet seed lease is illegal for its phase")
    observed = (
        value["purpose"],
        round_index,
        value["first_index"],
        value["count"],
        value["stride"],
    )
    if observed != expected:
        raise CampaignControllerError("gameplay packet seed lease differs from its phase contract")
    if value["lease_sha256"] != content_sha256(value, hash_field="lease_sha256"):
        raise CampaignControllerError("gameplay packet seed lease hash differs")


def _synthetic_seed_prefix(lease: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if lease is None:
        return None
    return {
        "lease_sha256": lease["lease_sha256"],
        "used_count": lease["count"],
        "unused_count": 0,
        "last_index": lease["first_index"] + (lease["count"] - 1) * lease["stride"],
    }


def _synthetic_artifact_payload(packet: Mapping[str, Any]) -> bytes:
    return f"artifact:{packet['packet_sha256']}".encode("ascii")


def _synthetic_remote_storage_receipt(
    packet: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    completed_unix_ms: int,
) -> dict[str, Any]:
    locator = _validate_storage_receipt_relative(artifact["storage_receipt_relative"])
    request_id = PurePosixPath(locator).name.removesuffix(".json")
    value: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": REMOTE_RECEIPT_SCHEMA,
        "request_id": request_id,
        "command_sha256": _sha256_text(
            f"synthetic-put-file:{packet['packet_sha256']}:{artifact['path']}"
        ),
        "operation": "put-file",
        "status": "ok",
        "host": STORAGE_HOST,
        "host_identity_sha256": _sha256_text("synthetic-john2-storage-identity"),
        "root": str(CAMPAIGN_ROOT),
        "completed_unix_ms": completed_unix_ms,
        "result": {
            "relative": artifact["path"],
            "sha256": artifact["sha256"],
            "size": artifact["bytes"],
            "mode": "0o400",
            "previous_sha256": None,
            "payload_size": 0,
            "payload_sha256": EMPTY_SHA256,
        },
    }
    value["receipt_sha256"] = content_sha256(value, hash_field="receipt_sha256")
    return value


def _install_synthetic_storage_evidence(
    paths: ControllerPaths,
    packet: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize dry-run-only immutable artifacts and worker receipt evidence."""
    packet = validate_work_packet(packet)
    value = copy.deepcopy(dict(receipt))
    for artifact in value["artifacts"]:
        relative = _validate_remote_relative_path(artifact["path"], "synthetic artifact")
        artifact_path = paths.root / relative
        if artifact_path.exists():
            payload = artifact_path.read_bytes()
        else:
            expected_relative = _join_remote_relative(
                packet["artifact_root"], f"{packet['task_id']}.artifact"
            )
            if relative != expected_relative:
                raise CampaignControllerError(
                    "synthetic fixture omitted a declared artifact payload"
                )
            payload = _synthetic_artifact_payload(packet)
        if (
            len(payload) != artifact["bytes"]
            or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
        ):
            raise CampaignControllerError("synthetic artifact differs from work receipt")
        _write_immutable_bytes(artifact_path, payload)
        storage_receipt = _synthetic_remote_storage_receipt(
            packet,
            artifact,
            completed_unix_ms=value["completed_unix_ms"],
        )
        artifact["storage_receipt_sha256"] = storage_receipt["receipt_sha256"]
        storage_path = paths.root / _validate_storage_receipt_relative(
            artifact["storage_receipt_relative"]
        )
        _write_immutable_bytes(storage_path, canonical_json_bytes(storage_receipt))
    value["receipt_sha256"] = content_sha256(value, hash_field="receipt_sha256")
    return validate_receipt(value, packet=packet)


def _validate_used_seed_prefix(value: Any, lease: Mapping[str, Any] | None) -> None:
    if lease is None:
        if value is not None:
            raise CampaignControllerError("non-gameplay receipt cannot report seed use")
        return
    if not isinstance(value, Mapping) or set(value) != {
        "lease_sha256",
        "used_count",
        "unused_count",
        "last_index",
    }:
        raise CampaignControllerError("receipt used-seed prefix is invalid")
    if value["lease_sha256"] != lease["lease_sha256"]:
        raise CampaignControllerError("receipt used another seed lease")
    used = value["used_count"]
    unused = value["unused_count"]
    if (
        not isinstance(used, int)
        or not isinstance(unused, int)
        or used < 0
        or unused < 0
        or used + unused != lease["count"]
    ):
        raise CampaignControllerError("receipt seed-prefix accounting differs")
    expected_last = None if used == 0 else lease["first_index"] + (used - 1) * lease["stride"]
    if value["last_index"] != expected_last:
        raise CampaignControllerError("receipt seed use is not a contiguous lease prefix")


def _artifact_label(packet: Mapping[str, Any]) -> str:
    if packet["task_kind"] == "train":
        return "candidate-checkpoint"
    if packet["task_kind"] == "aggregate" and packet["aggregate_kind"] == "candidate-gate":
        return "candidate-gate-report"
    if packet["task_kind"] == "aggregate" and packet["aggregate_kind"] == "generation":
        return "generation-manifest"
    return f"{packet['task_kind']}-artifact"


def _phase_packets_on_disk(
    paths: ControllerPaths, state: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    prefix = f"r2map-r{int(state['revision']):04d}-"
    if not paths.packets.exists():
        return ()
    packets = []
    for path in sorted(paths.packets.glob(f"{prefix}*.json")):
        packet = validate_work_packet(json.loads(path.read_text()))
        if (
            packet["controller_state_sha256"] != state["state_sha256"]
            or packet["phase"] != state["phase"]
        ):
            raise CampaignControllerError("phase packet identity differs from campaign state")
        packets.append(packet)
    expected = phase_templates(Phase(state["phase"]))
    if len(packets) != len(expected):
        raise CampaignControllerError("phase packet set is incomplete")
    return tuple(packets)


def _read_packet(paths: ControllerPaths, task_id: str) -> dict[str, Any]:
    path = paths.packets / f"{task_id}.json"
    try:
        return validate_work_packet(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignControllerError(f"cannot read work packet {task_id}: {error}") from error


def _queue_task(queue: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    for task in queue["tasks"]:
        if task["id"] == task_id:
            return task
    raise CampaignControllerError(f"queue lacks registered task {task_id}")


def _queue_static_identity(task: Mapping[str, Any]) -> str:
    fields = (
        "id",
        "title",
        "experiment_id",
        "decision",
        "workload_class",
        "priority",
        "decision_value",
        "expected_runtime_seconds",
        "critical_path",
        "decision_terminal",
        "compatible_hosts",
        "dependencies",
        "command",
        "artifact_path",
        "stop_rule",
        "resources",
    )
    return hashlib.sha256(
        canonical_json_bytes({field: task[field] for field in fields})
    ).hexdigest()


def _set_exact_queue_intents(queue: dict[str, Any], phase: Phase, now_ms: int) -> None:
    mapping = {
        "control": "reserved",
        "validate": "reserved",
        "train": "reserved",
        "generate": "reserved",
        "benchmark": "reserved",
        "candidate-gate": "reserved",
        "idle": "intentionally-idle",
    }
    for host in ALLOWED_HOSTS:
        if any(
            task["status"] == "running" and (task.get("claim") or {}).get("host") == host
            for task in queue["tasks"]
        ):
            continue
        intent = mapping[str(PHASE_HOST_INTENTS[phase][host])]
        research_queue.set_host_intent(
            queue,
            host=host,
            intent=intent,
            reason=f"R2-MAP phase {phase.value}",
            now_ms=now_ms,
        )


def _experiment_from_state(
    state: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    queue: Mapping[str, Any],
    *,
    now_ms: int,
) -> dict[str, Any]:
    packet_ids = [
        task["id"] for task in queue["tasks"] if task["experiment_id"] == CONTROLLER_EXPERIMENT_ID
    ]
    tasks = [_queue_task(queue, task_id) for task_id in packet_ids]
    terminal = bool(tasks) and all(
        task["status"] in {"completed", "failed", "cancelled"} for task in tasks
    )
    failed = any(task["status"] in {"failed", "cancelled"} for task in tasks)
    return {
        "id": CONTROLLER_EXPERIMENT_ID,
        "title": "R2-MAP expert iteration",
        "hypothesis": "Exact-R2 generalized policy iteration can reach a 100-point focal mean.",
        "summary": f"Controller phase {state['phase']} at revision {state['revision']}.",
        "status": "completed" if terminal else "running",
        "outcome": "failed" if terminal and failed else ("inconclusive" if terminal else "pending"),
        "verdict": None,
        "plan_section": "W6",
        "started_unix_ms": 0,
        "completed_unix_ms": now_ms if terminal else None,
        "updated_unix_ms": now_ms,
        "hosts": list(ALLOWED_HOSTS),
        "tags": ["r2-map", "expert-iteration", "orchestration"],
        "task_ids": packet_ids,
        "metrics": [
            {"label": "Phase", "value": state["phase"], "tone": "neutral"},
            {
                "label": "Completed tasks",
                "value": str(sum(task["status"] == "completed" for task in tasks)),
                "tone": "neutral",
            },
        ],
        "criteria": [
            {"label": "All phase tasks terminal", "passed": terminal, "observed": str(terminal)}
        ],
        "notes": ["Queue and ledger are projections of the hash-chained campaign state."],
        "artifacts": [
            {"label": "R2-MAP plan", "path": "docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md"}
        ],
    }


def _storage_binding() -> dict[str, str]:
    return {
        "host": STORAGE_HOST,
        "root": str(CAMPAIGN_ROOT),
        "transport": REMOTE_STORAGE_TRANSPORT,
    }


def _validate_remote_relative_path(
    value: Any, label: str, *, require_authorized_top_level: bool = True
) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CampaignControllerError(f"{label} must be a canonical remote-relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise CampaignControllerError(f"{label} must be a canonical remote-relative path")
    if require_authorized_top_level and candidate.parts[0] not in REMOTE_ARTIFACT_TOP_LEVELS:
        raise CampaignControllerError(f"{label} uses an unauthorized remote top-level")
    serialized = value.casefold()
    if "/volumes/john_1" in serialized or "/users/johnherrick" in serialized:
        raise CampaignControllerError(f"{label} names forbidden John1 storage")
    return value


def _join_remote_relative(root: str, name: str) -> str:
    root = _validate_remote_relative_path(root, "artifact root")
    name = _validate_remote_relative_path(
        name, "artifact name", require_authorized_top_level=False
    )
    if "/" in name:
        raise CampaignControllerError("artifact name must be one path component")
    return f"{root}/{name}"


def _remote_artifact_uri(relative: str) -> str:
    relative = _validate_remote_relative_path(relative, "remote artifact URI path")
    return f"r2map+ssh://{STORAGE_HOST}/{relative}"


def _validate_storage_receipt_relative(value: Any) -> str:
    relative = _validate_remote_relative_path(value, "storage receipt")
    parts = PurePosixPath(relative).parts
    request_id = parts[2][:-5] if len(parts) == 3 and parts[2].endswith(".json") else ""
    if (
        len(parts) != 3
        or parts[:2] != ("control", "receipts")
        or not request_id.startswith("req-")
        or len(request_id) <= len("req-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in request_id
        )
    ):
        raise CampaignControllerError(
            "storage receipt must resolve to control/receipts/<request-id>.json"
        )
    return relative


def _reject_john4(value: Any) -> None:
    if _contains_text(value, FORBIDDEN_HOST):
        raise CampaignControllerError("R2-MAP controller payload may not name john4")


def _contains_text(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle.casefold() in value.casefold()
    if isinstance(value, Mapping):
        return any(
            _contains_text(key, needle) or _contains_text(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_text(item, needle) for item in value)
    return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise CampaignControllerError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CampaignControllerError(f"immutable artifact differs: {path}")
        path.chmod(0o400)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
