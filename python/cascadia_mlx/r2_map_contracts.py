"""Fail-closed storage and orchestration contracts for the R2-MAP campaign.

This module deliberately has no MLX dependency.  The campaign controller, data
collectors, trainers, and verifiers can all import it before allocating large
objects or initializing an accelerator runtime.
"""

from __future__ import annotations

import errno
import fcntl
import getpass
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # Apple Python 3.9 on the John2 control plane.
    UTC = timezone.utc  # noqa: UP017

try:
    from enum import StrEnum
except ImportError:  # Apple Python 3.9 on the John2 control plane.

    class StrEnum(str, Enum):  # noqa: UP042
        """Python 3.9-compatible subset used by explicit-value campaign enums."""

CAMPAIGN_ID = "r2-map-expert-iteration-v1"
CAMPAIGN_STATE_SCHEMA_ID = "cascadia.r2-map.campaign-state.v1"
DECISION_LOG_SCHEMA_ID = "cascadia.r2-map.decision-log-entry.v1"
STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID = (
    "cascadia.r2-map.storage-supersession-genesis.v1"
)
SCHEMA_VERSION = 1

STORAGE_HOST = "john1"
EXPECTED_VOLUME = Path("/")
CAMPAIGN_RELATIVE_PATH = Path("Users/johnherrick/cascadia-bench/r2-map-v1")
CAMPAIGN_ROOT = EXPECTED_VOLUME / CAMPAIGN_RELATIVE_PATH
FROZEN_EXTERNAL_SSD_ROOT = Path("/Volumes/John_1/cascadia-cluster/r2-map-v1")
FROZEN_LEGACY_JOHN2_ROOT = Path("/Users/john2/cascadia-bench/r2-map-v1")
ALLOWED_HOSTS = ("john1", "john2", "john3")
FORBIDDEN_HOST = "john4"
GIB = 1 << 30
MIN_FREE_BYTES = 64 * GIB
CAMPAIGN_BUDGET_BYTES = 64 * GIB
PER_RUN_BUDGET_BYTES = 32 * GIB

LAYOUT_DIRECTORIES = (
    "control",
    "control/contracts",
    "control/bin",
    "control/locks",
    "control/receipts",
    "control/transactions",
    "source",
    "build",
    "toolchains",
    "home",
    "cache",
    "cache/uv",
    "cache/runs",
    "bundles",
    "datasets/bootstrap",
    "datasets/iterations",
    "opponent-pool",
    "checkpoints",
    "runs",
    "benchmarks",
    "reports",
    "logs",
    "tmp",
    "tmp/cargo-target",
    "tmp/pytest-cache",
    "tmp/uv-cache",
)


class ContractError(RuntimeError):
    """Base class for an R2-MAP contract violation."""


class StoragePreflightError(ContractError):
    """The authoritative John1 storage endpoint is unsafe for campaign work."""


class StateValidationError(ContractError):
    """A campaign state document is malformed or internally inconsistent."""


class TransitionError(ContractError):
    """A requested campaign phase transition is illegal."""


class DecisionLogError(ContractError):
    """The append-only operational decision log is malformed."""


class Phase(StrEnum):
    CONTRACTS_READY = "contracts-ready"
    BOOTSTRAP_GENERATING = "bootstrap-generating"
    BOOTSTRAP_VALIDATED = "bootstrap-validated"
    BOOTSTRAP_TRAINING = "bootstrap-training"
    BOOTSTRAP_CANDIDATE_GATE = "bootstrap-candidate-gate"
    INCUMBENT_PROMOTED = "incumbent-promoted"
    ROUND_ALLOCATED = "round-allocated"
    GENERATING = "generating-45m"
    LOCAL_SHARDS_COMPLETE = "local-shards-complete"
    COLLECTED_AND_VALIDATED = "collected-and-validated-on-john1"
    TRAINING_AND_BENCHMARKING = "training-on-john1-benchmarking-on-john2-john3"
    CANDIDATE_VERIFIED_BENCHMARK_COMPLETE = "candidate-verified-benchmark-complete"
    PAIRED_CANDIDATE_GATE = "paired-candidate-gate-on-john2-john3"
    CANDIDATE_REJECTED = "candidate-rejected"


class HostIntent(StrEnum):
    CONTROL = "control"
    GENERATE = "generate"
    VALIDATE = "validate"
    TRAIN = "train"
    BENCHMARK = "benchmark"
    CANDIDATE_GATE = "candidate-gate"
    IDLE = "idle"


PHASE_HOST_INTENTS: dict[Phase, dict[str, str]] = {
    Phase.CONTRACTS_READY: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.BOOTSTRAP_GENERATING: {host: HostIntent.GENERATE for host in ALLOWED_HOSTS},
    Phase.BOOTSTRAP_VALIDATED: {
        "john1": HostIntent.VALIDATE,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.BOOTSTRAP_TRAINING: {
        "john1": HostIntent.TRAIN,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.BOOTSTRAP_CANDIDATE_GATE: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.CANDIDATE_GATE,
        "john3": HostIntent.CANDIDATE_GATE,
    },
    Phase.INCUMBENT_PROMOTED: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.ROUND_ALLOCATED: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.GENERATING: {host: HostIntent.GENERATE for host in ALLOWED_HOSTS},
    Phase.LOCAL_SHARDS_COMPLETE: {
        "john1": HostIntent.VALIDATE,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.COLLECTED_AND_VALIDATED: {
        "john1": HostIntent.VALIDATE,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.TRAINING_AND_BENCHMARKING: {
        "john1": HostIntent.TRAIN,
        "john2": HostIntent.BENCHMARK,
        "john3": HostIntent.BENCHMARK,
    },
    Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
    Phase.PAIRED_CANDIDATE_GATE: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.CANDIDATE_GATE,
        "john3": HostIntent.CANDIDATE_GATE,
    },
    Phase.CANDIDATE_REJECTED: {
        "john1": HostIntent.CONTROL,
        "john2": HostIntent.IDLE,
        "john3": HostIntent.IDLE,
    },
}

LEGAL_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.CONTRACTS_READY: frozenset({Phase.BOOTSTRAP_GENERATING}),
    Phase.BOOTSTRAP_GENERATING: frozenset({Phase.BOOTSTRAP_VALIDATED}),
    Phase.BOOTSTRAP_VALIDATED: frozenset({Phase.BOOTSTRAP_TRAINING}),
    Phase.BOOTSTRAP_TRAINING: frozenset({Phase.BOOTSTRAP_CANDIDATE_GATE}),
    Phase.BOOTSTRAP_CANDIDATE_GATE: frozenset({Phase.INCUMBENT_PROMOTED}),
    Phase.INCUMBENT_PROMOTED: frozenset({Phase.ROUND_ALLOCATED}),
    Phase.ROUND_ALLOCATED: frozenset({Phase.GENERATING}),
    Phase.GENERATING: frozenset({Phase.LOCAL_SHARDS_COMPLETE}),
    Phase.LOCAL_SHARDS_COMPLETE: frozenset({Phase.COLLECTED_AND_VALIDATED}),
    Phase.COLLECTED_AND_VALIDATED: frozenset({Phase.TRAINING_AND_BENCHMARKING}),
    Phase.TRAINING_AND_BENCHMARKING: frozenset({Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE}),
    Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE: frozenset({Phase.PAIRED_CANDIDATE_GATE}),
    Phase.PAIRED_CANDIDATE_GATE: frozenset({Phase.INCUMBENT_PROMOTED, Phase.CANDIDATE_REJECTED}),
    Phase.CANDIDATE_REJECTED: frozenset({Phase.ROUND_ALLOCATED}),
}


@dataclass(frozen=True)
class StorageContract:
    expected_host: str = STORAGE_HOST
    expected_volume: Path = EXPECTED_VOLUME
    campaign_relative_path: Path = CAMPAIGN_RELATIVE_PATH
    campaign_root: Path = CAMPAIGN_ROOT
    expected_uid: int = 501
    expected_gid: int = 20
    required_mode: int = 0o700
    min_free_bytes: int = MIN_FREE_BYTES
    campaign_budget_bytes: int = CAMPAIGN_BUDGET_BYTES
    per_run_budget_bytes: int = PER_RUN_BUDGET_BYTES


DEFAULT_STORAGE_CONTRACT = StorageContract()


def local_campaign_host_id(*, username: str | None = None) -> str | None:
    """Map the fixed cluster login identity to its orchestration host id.

    This deliberately does not trust an environment variable: a caller cannot
    turn a john1 process into a john2 storage process by changing its env.
    Tests and remote executors may pass ``current_host_id`` directly to the
    preflight boundary.
    """

    login = getpass.getuser() if username is None else username
    return {"johnherrick": "john1", "john1": "john1", "john2": "john2", "john3": "john3"}.get(
        login
    )


def require_local_storage_authority(
    contract: StorageContract = DEFAULT_STORAGE_CONTRACT,
    *,
    current_host_id: str | None = None,
) -> str:
    """Fail unless this process is executing on the contract's storage host."""

    observed = local_campaign_host_id() if current_host_id is None else current_host_id
    if observed != contract.expected_host:
        rendered = "unrecognized" if observed is None else observed
        raise StoragePreflightError(
            "authoritative campaign storage is remote: "
            f"expected host {contract.expected_host}, observed {rendered}; "
            "use the strict R2-MAP remote-storage transport instead of a local Path"
        )
    return observed


def reject_frozen_campaign_path(path: str | Path, *, label: str) -> None:
    """Reject excluded legacy roots at a path-taking component boundary.

    The controller's storage preflight proves the exact John1 root. Components
    retain path injection for isolated unit tests, but may never target either
    frozen legacy tree.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        raise StoragePreflightError(f"{label} must be absolute")
    _reject_parent_segments(candidate, label=label)
    resolved = candidate.resolve(strict=False)
    for frozen in (FROZEN_EXTERNAL_SSD_ROOT, FROZEN_LEGACY_JOHN2_ROOT):
        if resolved == frozen or frozen in resolved.parents:
            raise StoragePreflightError(f"{label} targets frozen legacy evidence")


CAMPAIGN_STATE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": CAMPAIGN_STATE_SCHEMA_ID,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "schema_id",
        "campaign_id",
        "revision",
        "phase",
        "promotion_index",
        "round_index",
        "incumbent_checkpoint_id",
        "incumbent_checkpoint_sha256",
        "generation_dataset_id",
        "generation_manifest_sha256",
        "candidate_checkpoint_id",
        "candidate_checkpoint_sha256",
        "completed_shard_hosts",
        "host_intents",
        "previous_state_sha256",
        "last_transition",
        "updated_at",
        "state_sha256",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "schema_id": {"const": CAMPAIGN_STATE_SCHEMA_ID},
        "campaign_id": {"const": CAMPAIGN_ID},
        "revision": {"type": "integer", "minimum": 0},
        "phase": {"enum": [phase.value for phase in Phase]},
        "promotion_index": {"type": ["integer", "null"], "minimum": 0},
        "round_index": {"type": ["integer", "null"], "minimum": 0},
        "incumbent_checkpoint_id": {"type": ["string", "null"]},
        "incumbent_checkpoint_sha256": {"type": ["string", "null"]},
        "generation_dataset_id": {"type": ["string", "null"]},
        "generation_manifest_sha256": {"type": ["string", "null"]},
        "candidate_checkpoint_id": {"type": ["string", "null"]},
        "candidate_checkpoint_sha256": {"type": ["string", "null"]},
        "completed_shard_hosts": {
            "type": "array",
            "uniqueItems": True,
            "items": {"enum": list(ALLOWED_HOSTS)},
        },
        "host_intents": {
            "type": "object",
            "additionalProperties": False,
            "required": list(ALLOWED_HOSTS),
        },
        "previous_state_sha256": {"type": ["string", "null"]},
        "last_transition": {"type": ["object", "null"]},
        "updated_at": {"type": "string"},
        "state_sha256": {"type": "string"},
    },
}

DECISION_LOG_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": DECISION_LOG_SCHEMA_ID,
    "$comment": (
        "Exactly one validated pre-contract or storage-supersession genesis may precede v1 "
        "entries. Its decision SHA-256 anchors the first v1 entry; no genesis is accepted later."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "schema_id",
        "campaign_id",
        "sequence",
        "decision_kind",
        "actor",
        "recorded_at",
        "campaign_state_revision",
        "campaign_state_sha256",
        "triggering_evidence",
        "alternatives_considered",
        "chosen_action",
        "affected_artifacts",
        "rollback_path",
        "authorization_sha256",
        "previous_decision_sha256",
        "decision_sha256",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "schema_id": {"const": DECISION_LOG_SCHEMA_ID},
        "campaign_id": {"const": CAMPAIGN_ID},
        "sequence": {"type": "integer", "minimum": 0},
        "decision_kind": {
            "enum": ["bounded-operational-adaptation", "scientific-contract-amendment"]
        },
        "actor": {"type": "string", "minLength": 1},
        "recorded_at": {"type": "string"},
        "campaign_state_revision": {"type": ["integer", "null"], "minimum": 0},
        "campaign_state_sha256": {"type": ["string", "null"]},
        "triggering_evidence": {"type": "array", "minItems": 1},
        "alternatives_considered": {"type": "array", "minItems": 1},
        "chosen_action": {"type": "string", "minLength": 1},
        "affected_artifacts": {"type": "array", "minItems": 1},
        "rollback_path": {"type": "string", "minLength": 1},
        "authorization_sha256": {"type": ["string", "null"]},
        "previous_decision_sha256": {"type": ["string", "null"]},
        "decision_sha256": {"type": "string"},
    },
}

STORAGE_SUPERSESSION_GENESIS_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "schema_id",
        "campaign_id",
        "sequence",
        "decision_kind",
        "recorded_at",
        "legacy_storage_host",
        "legacy_campaign_root",
        "legacy_campaign_state_sha256",
        "legacy_decision_head_sha256",
        "legacy_evidence_immutable",
        "canonical_storage_host",
        "canonical_campaign_root",
        "canonical_campaign_state_sha256",
        "authorization_sha256",
        "previous_decision_sha256",
        "decision_sha256",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "schema_id": {"const": STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID},
        "campaign_id": {"const": CAMPAIGN_ID},
        "sequence": {"const": 0},
        "decision_kind": {"const": "storage-supersession-genesis"},
        "recorded_at": {"type": "string", "minLength": 1},
        "legacy_storage_host": {"const": "john2"},
        "legacy_campaign_root": {"const": str(FROZEN_LEGACY_JOHN2_ROOT)},
        "legacy_campaign_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "legacy_decision_head_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "legacy_evidence_immutable": {"const": True},
        "canonical_storage_host": {"const": STORAGE_HOST},
        "canonical_campaign_root": {"const": str(CAMPAIGN_ROOT)},
        "canonical_campaign_state_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "authorization_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "previous_decision_sha256": {"const": None},
        "decision_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_sha256(value: Mapping[str, Any], *, hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_forbidden_host(value: Any) -> bool:
    if isinstance(value, str):
        return FORBIDDEN_HOST in value.casefold()
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_host(key) or _contains_forbidden_host(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_host(item) for item in value)
    return False


def _reject_parent_segments(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise StoragePreflightError(f"{label} contains a forbidden '..' path segment: {path}")


def _reject_existing_symlinks(path: Path, *, stop: Path | None = None, label: str) -> None:
    absolute = path.absolute()
    components: list[Path] = []
    cursor = absolute
    while True:
        components.append(cursor)
        if cursor == cursor.parent or (stop is not None and cursor == stop.absolute()):
            break
        cursor = cursor.parent
    for component in reversed(components):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise StoragePreflightError(f"{label} traverses symlink {component}")


def canonical_campaign_path(path: Path, *, root: Path, label: str) -> Path:
    if not path.is_absolute():
        raise StoragePreflightError(f"{label} must be an absolute path: {path}")
    _reject_parent_segments(path, label=label)
    _reject_existing_symlinks(root, label="campaign root")
    _reject_existing_symlinks(path, stop=root, label=label)
    canonical_root = root.resolve(strict=True)
    canonical = path.resolve(strict=False)
    if canonical != canonical_root and canonical_root not in canonical.parents:
        raise StoragePreflightError(f"{label} escapes campaign root {canonical_root}: {canonical}")
    return canonical


def canonical_rooted_path(path: Path, *, root: Path, label: str) -> Path:
    """Resolve one exact nonsymlink child of a trusted mounted root."""
    if not path.is_absolute():
        raise StoragePreflightError(f"{label} must be an absolute path: {path}")
    _reject_parent_segments(path, label=label)
    _reject_existing_symlinks(root, label=f"{label} root")
    _reject_existing_symlinks(path, stop=root, label=label)
    canonical_root = root.resolve(strict=True)
    canonical = path.resolve(strict=False)
    if canonical != canonical_root and canonical_root not in canonical.parents:
        raise StoragePreflightError(f"{label} escapes trusted root {canonical_root}: {canonical}")
    return canonical


def _tree_apparent_size(root: Path) -> int:
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in (*names, *files):
            path = directory_path / name
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                # Cargo and MLX may atomically remove completed temporary objects
                # after os.walk has enumerated a directory. A vanished descendant
                # contributes zero current bytes; configured roots are validated
                # independently and continue to fail closed.
                continue
            if stat.S_ISLNK(metadata.st_mode):
                # The size walk must never follow a convenience symlink (pytest, for
                # example, maintains ``pytest-current``). Configured artifact paths
                # are independently rejected if any component is a symlink.
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def _atomic_rename_fsync_probe(root: Path) -> None:
    payload = os.urandom(32)
    source: Path | None = None
    destination: Path | None = None
    try:
        descriptor, source_name = tempfile.mkstemp(prefix=".r2map-preflight-", dir=root)
        source = Path(source_name)
        destination = source.with_suffix(".verified")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(source, destination)
        source = None
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if destination.read_bytes() != payload:
            raise StoragePreflightError("atomic rename/fsync probe read back different bytes")
    except OSError as error:
        raise StoragePreflightError(
            f"atomic rename/fsync probe failed under {root}: {error}"
        ) from error
    finally:
        for path in (source, destination):
            if path is not None:
                with suppress(FileNotFoundError):
                    path.unlink()


def preflight_storage(
    *,
    contract: StorageContract = DEFAULT_STORAGE_CONTRACT,
    configured_paths: Mapping[str, Path] | None = None,
    expected_run_bytes: int = 0,
    environ: Mapping[str, str] | None = None,
    mount_checker: Callable[[Path], bool] = os.path.ismount,
    writable_checker: Callable[[Path], bool] = lambda path: os.access(path, os.W_OK),
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    atomic_probe: Callable[[Path], None] = _atomic_rename_fsync_probe,
    measure_campaign_bytes: bool = True,
    apfs_workspace_spec: Any | None = None,
    apfs_mount_observation: Mapping[str, Any] | None = None,
    path_stat: Callable[[Path], Any] = lambda path: path.stat(),
    current_uid: Callable[[], int] = os.getuid,
    current_host_id: str | None = None,
) -> dict[str, Any]:
    """Prove the local John1 active-storage contract before filesystem work.

    Remote hosts never treat this path as mounted storage. They return immutable
    bundles through the reviewed ingress protocol for John1 to validate and
    atomically install.
    """
    observed_host = require_local_storage_authority(
        contract, current_host_id=current_host_id
    )
    if apfs_workspace_spec is not None or apfs_mount_observation is not None:
        raise StoragePreflightError(
            "nested or external APFS workspaces are forbidden; active storage is "
            "John1's owner-private internal-APFS campaign root"
        )
    volume = contract.expected_volume
    root = contract.campaign_root
    for frozen in (FROZEN_EXTERNAL_SSD_ROOT, FROZEN_LEGACY_JOHN2_ROOT):
        if root == frozen or frozen in root.parents:
            raise StoragePreflightError("campaign root targets frozen legacy evidence")
    _reject_parent_segments(volume, label="expected volume")
    _reject_parent_segments(root, label="campaign root")
    if not volume.is_absolute() or not root.is_absolute():
        raise StoragePreflightError("expected volume and campaign root must be absolute")
    if not volume.exists() or not volume.is_dir() or not mount_checker(volume):
        raise StoragePreflightError(f"expected John1 filesystem root is not mounted: {volume}")
    _reject_existing_symlinks(volume, label="expected volume")
    canonical_volume = volume.resolve(strict=True)
    if not root.exists() or not root.is_dir():
        raise StoragePreflightError(f"campaign root does not exist: {root}")
    _reject_existing_symlinks(root, label="campaign root")
    canonical_root = root.resolve(strict=True)
    expected_root = canonical_volume / contract.campaign_relative_path
    if canonical_root != expected_root:
        raise StoragePreflightError(
            "campaign root is not the exact required path on the storage host: "
            f"{canonical_root}"
        )
    root_metadata = path_stat(canonical_root)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != contract.expected_uid
        or root_metadata.st_gid != contract.expected_gid
        or stat.S_IMODE(root_metadata.st_mode) != contract.required_mode
    ):
        raise StoragePreflightError(
            "campaign root owner or mode differs from the John1 0700 contract"
        )

    environment = os.environ if environ is None else environ
    tmp_value = environment.get("TMPDIR")
    cargo_value = environment.get("CARGO_TARGET_DIR")
    if not tmp_value:
        raise StoragePreflightError("TMPDIR is required and must point inside campaign tmp/")
    if not cargo_value:
        raise StoragePreflightError(
            "CARGO_TARGET_DIR is required and must point inside campaign tmp/cargo-target/"
        )
    workspace_proof: dict[str, Any] | None = None
    tmp_root_path = canonical_root / "tmp"
    execution_root = canonical_root
    canonicalizer = canonical_campaign_path
    canonical_tmp = canonicalizer(Path(tmp_value), root=execution_root, label="TMPDIR")
    tmp_root = canonicalizer(tmp_root_path, root=execution_root, label="campaign tmp root")
    if canonical_tmp != tmp_root and tmp_root not in canonical_tmp.parents:
        raise StoragePreflightError(f"TMPDIR is outside campaign tmp/: {canonical_tmp}")
    canonical_control = canonicalizer(
        canonical_root / "control", root=execution_root, label="campaign control root"
    )
    # A provisioned campaign writes control-plane state below ``control/``.
    # During first-party initialization that directory does not exist yet, so
    # the storage owner must instead prove that it can create it from the
    # campaign root.  Controller sandboxes never receive root write access:
    # their already-provisioned ``control/`` directory is the writable target.
    control_write_target = canonical_control if canonical_control.exists() else canonical_root
    if not writable_checker(canonical_tmp) or not writable_checker(control_write_target):
        raise StoragePreflightError(
            "registered TMPDIR or campaign control root is not writable"
        )
    canonical_cargo = canonicalizer(
        Path(cargo_value), root=execution_root, label="CARGO_TARGET_DIR"
    )
    cargo_roots = tuple(
        canonicalizer(path, root=execution_root, label="cargo target root")
        for path in (canonical_root / "tmp/cargo-target", canonical_root / "build")
    )
    if not any(
        canonical_cargo == cargo_root or cargo_root in canonical_cargo.parents
        for cargo_root in cargo_roots
    ):
        raise StoragePreflightError(
            "CARGO_TARGET_DIR is outside campaign tmp/cargo-target/ or build/: "
            f"{canonical_cargo}"
        )
    canonical_cache: Path | None = None
    if apfs_workspace_spec is not None:
        cache_value = environment.get("UV_CACHE_DIR")
        if not cache_value:
            raise StoragePreflightError("UV_CACHE_DIR is required inside the APFS workspace cache/")
        canonical_cache = canonical_rooted_path(
            Path(cache_value), root=apfs_workspace_spec.mountpoint, label="UV_CACHE_DIR"
        )
        cache_root = canonical_rooted_path(
            apfs_workspace_spec.cache,
            root=apfs_workspace_spec.mountpoint,
            label="APFS workspace cache root",
        )
        if canonical_cache != cache_root and cache_root not in canonical_cache.parents:
            raise StoragePreflightError(
                f"UV_CACHE_DIR is outside APFS workspace cache/: {canonical_cache}"
            )

    validated_paths: dict[str, str] = {}
    for name, path in sorted((configured_paths or {}).items()):
        candidate = Path(path)
        try:
            canonical = canonical_campaign_path(
                candidate, root=canonical_root, label=f"configured path {name}"
            )
        except StoragePreflightError:
            raise
        validated_paths[name] = str(canonical)

    usage = disk_usage(canonical_root)
    if usage.free < contract.min_free_bytes:
        raise StoragePreflightError(
            f"John1 disk has {usage.free} free bytes; {contract.min_free_bytes} required"
        )
    campaign_bytes = _tree_apparent_size(canonical_root) if measure_campaign_bytes else None
    if campaign_bytes is not None and campaign_bytes > contract.campaign_budget_bytes:
        raise StoragePreflightError(
            f"campaign uses {campaign_bytes} bytes; budget is {contract.campaign_budget_bytes}"
        )
    if expected_run_bytes < 0 or expected_run_bytes > contract.per_run_budget_bytes:
        raise StoragePreflightError(
            f"requested run budget {expected_run_bytes} exceeds {contract.per_run_budget_bytes}"
        )
    if usage.free - expected_run_bytes < contract.min_free_bytes:
        raise StoragePreflightError(
            "requested run would cross the 64 GiB free-space floor: "
            f"free={usage.free}, requested={expected_run_bytes}"
        )

    # Probe in the registered same-volume temporary namespace. This exercises
    # the filesystem semantics used by every atomic publication without
    # authorizing a preflight process to create an ad-hoc root-level object.
    atomic_probe(canonical_tmp)
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "storage_host": observed_host,
        "storage_uid": root_metadata.st_uid,
        "storage_gid": root_metadata.st_gid,
        "storage_mode": oct(stat.S_IMODE(root_metadata.st_mode)),
        "expected_volume": str(canonical_volume),
        "campaign_root": str(canonical_root),
        "free_bytes": usage.free,
        "campaign_bytes": campaign_bytes,
        "campaign_bytes_measured": measure_campaign_bytes,
        "min_free_bytes": contract.min_free_bytes,
        "campaign_budget_bytes": contract.campaign_budget_bytes,
        "per_run_budget_bytes": contract.per_run_budget_bytes,
        "expected_run_bytes": expected_run_bytes,
        "tmpdir": str(canonical_tmp),
        "cargo_target_dir": str(canonical_cargo),
        "uv_cache_dir": None if canonical_cache is None else str(canonical_cache),
        "configured_paths": validated_paths,
        "atomic_rename_fsync": True,
        "atomic_probe_directory": str(canonical_tmp),
        "apfs_workspace": workspace_proof,
    }


def _validate_apfs_preflight(
    spec: Any,
    observation: Mapping[str, Any] | None,
    *,
    canonical_root: Path,
    planned_bytes: int,
    mount_checker: Callable[[Path], bool],
    path_stat: Callable[[Path], Any],
    current_uid: int,
) -> dict[str, Any]:
    """Bind the pure APFS identity contract to observed mounted filesystem facts."""
    from cascadia_mlx.r2_map_apfs_workspace import (
        ApfsWorkspaceContractError,
        ApfsWorkspaceSpec,
        validate_marker_and_mount_observation,
    )

    if not isinstance(spec, ApfsWorkspaceSpec):
        raise StoragePreflightError("APFS workspace specification type differs")
    if spec.campaign_root != canonical_root:
        raise StoragePreflightError("APFS workspace names another campaign root")
    try:
        spec.validate()
        if observation is None:
            raise ApfsWorkspaceContractError("APFS mount observation is required")
        canonical_campaign_path(
            spec.backing_bundle, root=canonical_root, label="APFS backing bundle"
        )
        for label, path in (
            ("APFS mountpoint", spec.mountpoint),
            ("APFS marker", spec.marker),
            ("APFS Cargo target", spec.cargo_target),
            ("APFS temporary", spec.temporary),
            ("APFS cache", spec.cache),
        ):
            canonical_rooted_path(path, root=spec.mountpoint, label=label)
        if not spec.backing_bundle.is_dir():
            raise ApfsWorkspaceContractError("APFS backing sparsebundle is absent")
        if not spec.mountpoint.is_dir() or not mount_checker(spec.mountpoint):
            raise ApfsWorkspaceContractError("APFS workspace is not mounted at the exact path")
        try:
            marker = json.loads(spec.marker.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ApfsWorkspaceContractError("APFS workspace marker cannot be read") from error
        validate_marker_and_mount_observation(
            spec,
            marker,
            observation,
            planned_bytes=planned_bytes,
        )
        if marker["owner_uid"] != current_uid:
            raise ApfsWorkspaceContractError("APFS workspace owner is not the campaign owner")
        root_stat = path_stat(canonical_root)
        mount_stat = path_stat(spec.mountpoint)
        if root_stat.st_uid != current_uid or mount_stat.st_uid != current_uid:
            raise ApfsWorkspaceContractError("campaign root and APFS mount owner differ")
        if root_stat.st_dev == mount_stat.st_dev:
            raise ApfsWorkspaceContractError("APFS workspace is not a distinct mounted filesystem")
        for path in (spec.mountpoint, *spec.work_paths()):
            if not path.is_dir():
                raise ApfsWorkspaceContractError("APFS workspace directory is absent")
            metadata = path_stat(path)
            if (
                metadata.st_dev != mount_stat.st_dev
                or metadata.st_uid != marker["owner_uid"]
                or metadata.st_gid != marker["owner_gid"]
                or stat.S_IMODE(metadata.st_mode) != spec.required_mode
            ):
                raise ApfsWorkspaceContractError(
                    "APFS workspace ownership, device, or 0700 mode differs"
                )
    except ApfsWorkspaceContractError as error:
        raise StoragePreflightError(f"APFS workspace preflight failed: {error}") from error
    return {
        "schema_version": 1,
        "schema_id": marker["schema_id"],
        "backing_bundle": str(spec.backing_bundle),
        "mountpoint": str(spec.mountpoint),
        "physical_backing_root": marker["physical_backing_root"],
        "mount_namespace_only": marker["mount_namespace_only"],
        "volume_name": marker["volume_name"],
        "volume_uuid": marker["volume_uuid"],
        "filesystem": marker["filesystem"],
        "capacity_bytes": marker["capacity_bytes"],
        "budget_bytes": marker["budget_bytes"],
        "free_bytes": observation["free_bytes"],
        "backing_free_bytes": observation["backing_free_bytes"],
        "owner_uid": marker["owner_uid"],
        "owner_gid": marker["owner_gid"],
        "mode": marker["mode"],
    }


def initialize_layout(*, contract: StorageContract = DEFAULT_STORAGE_CONTRACT) -> None:
    """Create only the registered campaign directories after a successful preflight."""
    preflight_storage(contract=contract)
    for relative in LAYOUT_DIRECTORIES:
        destination = contract.campaign_root / relative
        canonical_campaign_path(destination, root=contract.campaign_root, label=relative)
        destination.mkdir(mode=contract.required_mode, parents=True, exist_ok=True)
        metadata = destination.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != contract.expected_uid
            or metadata.st_gid != contract.expected_gid
            or stat.S_IMODE(metadata.st_mode) != contract.required_mode
        ):
            raise StoragePreflightError(
                f"registered john2 directory owner or mode differs: {destination}"
            )
    _fsync_directory(contract.campaign_root)


def write_contract_schemas(root: Path) -> dict[str, str]:
    destinations = {
        "campaign_state": root / "control/contracts/campaign-state-v1.schema.json",
        "decision_log": root / "control/contracts/decision-log-entry-v1.schema.json",
        "storage_supersession_genesis": (
            root / "control/contracts/storage-supersession-genesis-v1.schema.json"
        ),
    }
    _atomic_write_json(destinations["campaign_state"], CAMPAIGN_STATE_JSON_SCHEMA)
    _atomic_write_json(destinations["decision_log"], DECISION_LOG_JSON_SCHEMA)
    _atomic_write_json(
        destinations["storage_supersession_genesis"],
        STORAGE_SUPERSESSION_GENESIS_JSON_SCHEMA,
    )
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in destinations.items()
    }


def new_campaign_state(*, now: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": CAMPAIGN_STATE_SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "revision": 0,
        "phase": Phase.CONTRACTS_READY.value,
        "promotion_index": None,
        "round_index": None,
        "incumbent_checkpoint_id": None,
        "incumbent_checkpoint_sha256": None,
        "generation_dataset_id": None,
        "generation_manifest_sha256": None,
        "candidate_checkpoint_id": None,
        "candidate_checkpoint_sha256": None,
        "completed_shard_hosts": [],
        "host_intents": dict(PHASE_HOST_INTENTS[Phase.CONTRACTS_READY]),
        "previous_state_sha256": None,
        "last_transition": None,
        "updated_at": now or utc_now(),
    }
    state["state_sha256"] = content_sha256(state, hash_field="state_sha256")
    validate_state(state)
    return state


def validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    required = set(CAMPAIGN_STATE_JSON_SCHEMA["required"])
    if set(state) != required:
        raise StateValidationError(
            f"campaign state keys differ: missing={sorted(required - set(state))}, "
            f"extra={sorted(set(state) - required)}"
        )
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("schema_id") != CAMPAIGN_STATE_SCHEMA_ID
    ):
        raise StateValidationError("unsupported campaign state schema")
    if state.get("campaign_id") != CAMPAIGN_ID:
        raise StateValidationError("campaign state names the wrong campaign")
    if _contains_forbidden_host(state):
        raise StateValidationError("campaign state may not name john4")
    revision = state.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise StateValidationError("campaign revision must be a nonnegative integer")
    try:
        phase = Phase(state.get("phase"))
    except ValueError as error:
        raise StateValidationError(f"unknown campaign phase {state.get('phase')!r}") from error
    if state.get("host_intents") != PHASE_HOST_INTENTS[phase]:
        raise StateValidationError(f"host intents do not match phase {phase.value}")
    if state.get("completed_shard_hosts") != sorted(set(state.get("completed_shard_hosts", []))):
        raise StateValidationError("completed shard hosts must be sorted and unique")
    if any(host not in ALLOWED_HOSTS for host in state.get("completed_shard_hosts", [])):
        raise StateValidationError("completed shard list names an unauthorized host")
    for field in (
        "incumbent_checkpoint_sha256",
        "generation_manifest_sha256",
        "candidate_checkpoint_sha256",
        "previous_state_sha256",
    ):
        if state.get(field) is not None and not _is_sha256(state[field]):
            raise StateValidationError(f"{field} must be null or a lowercase SHA-256 digest")
    if not _is_sha256(state.get("state_sha256")):
        raise StateValidationError("state_sha256 must be a lowercase SHA-256 digest")
    expected_hash = content_sha256(state, hash_field="state_sha256")
    if state["state_sha256"] != expected_hash:
        raise StateValidationError("campaign state content hash does not match its payload")
    if revision == 0:
        if (
            state.get("previous_state_sha256") is not None
            or state.get("last_transition") is not None
        ):
            raise StateValidationError(
                "initial campaign state cannot name a prior state or transition"
            )
    else:
        if not _is_sha256(state.get("previous_state_sha256")):
            raise StateValidationError("advanced campaign state requires a previous-state hash")
        transition = state.get("last_transition")
        if not isinstance(transition, dict) or set(transition) != {"from", "to", "reason", "at"}:
            raise StateValidationError("advanced campaign state requires exact transition metadata")
        try:
            Phase(transition["from"])
            Phase(transition["to"])
        except (KeyError, ValueError) as error:
            raise StateValidationError("transition metadata names an unknown phase") from error
        if transition["to"] != phase.value:
            raise StateValidationError("transition metadata target disagrees with campaign phase")
        if transition["at"] != state.get("updated_at"):
            raise StateValidationError("transition timestamp disagrees with state timestamp")
        if not isinstance(transition["reason"], str) or not transition["reason"].strip():
            raise StateValidationError("transition reason must be nonempty")

    pre_incumbent = {
        Phase.CONTRACTS_READY,
        Phase.BOOTSTRAP_GENERATING,
        Phase.BOOTSTRAP_VALIDATED,
        Phase.BOOTSTRAP_TRAINING,
        Phase.BOOTSTRAP_CANDIDATE_GATE,
    }
    if phase in pre_incumbent:
        if (
            state.get("promotion_index") is not None
            or state.get("incumbent_checkpoint_id") is not None
        ):
            raise StateValidationError("bootstrap state cannot name a promoted incumbent")
    else:
        promotion_index = state.get("promotion_index")
        if not isinstance(promotion_index, int) or promotion_index < 0:
            raise StateValidationError("post-bootstrap state requires a promotion index")
        if state.get("incumbent_checkpoint_id") != f"C[{promotion_index}]":
            raise StateValidationError("incumbent checkpoint id disagrees with promotion index")
        if not _is_sha256(state.get("incumbent_checkpoint_sha256")):
            raise StateValidationError("post-bootstrap state requires an incumbent checkpoint hash")

    iterative = {
        Phase.ROUND_ALLOCATED,
        Phase.GENERATING,
        Phase.LOCAL_SHARDS_COMPLETE,
        Phase.COLLECTED_AND_VALIDATED,
        Phase.TRAINING_AND_BENCHMARKING,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        Phase.PAIRED_CANDIDATE_GATE,
        Phase.CANDIDATE_REJECTED,
    }
    if phase in iterative:
        round_index = state.get("round_index")
        if not isinstance(round_index, int) or round_index < 0:
            raise StateValidationError("iterative phase requires a round index")
        if state.get("generation_dataset_id") != f"G[{round_index}]":
            raise StateValidationError("generation dataset id disagrees with round index")
        if state.get("candidate_checkpoint_id") != f"T[{round_index}]":
            raise StateValidationError("candidate checkpoint id disagrees with round index")

    shards_required = {
        Phase.BOOTSTRAP_VALIDATED,
        Phase.BOOTSTRAP_TRAINING,
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        Phase.LOCAL_SHARDS_COMPLETE,
        Phase.COLLECTED_AND_VALIDATED,
        Phase.TRAINING_AND_BENCHMARKING,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        Phase.PAIRED_CANDIDATE_GATE,
        Phase.CANDIDATE_REJECTED,
    }
    if phase in shards_required and state.get("completed_shard_hosts") != list(ALLOWED_HOSTS):
        raise StateValidationError(
            f"phase {phase.value} requires complete john1/john2/john3 shards"
        )
    manifest_required = {
        Phase.BOOTSTRAP_VALIDATED,
        Phase.BOOTSTRAP_TRAINING,
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        Phase.COLLECTED_AND_VALIDATED,
        Phase.TRAINING_AND_BENCHMARKING,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        Phase.PAIRED_CANDIDATE_GATE,
        Phase.CANDIDATE_REJECTED,
    }
    if phase in manifest_required and not _is_sha256(state.get("generation_manifest_sha256")):
        raise StateValidationError(f"phase {phase.value} requires a validated generation manifest")
    candidate_required = {
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        Phase.PAIRED_CANDIDATE_GATE,
        Phase.CANDIDATE_REJECTED,
    }
    if phase in candidate_required and not _is_sha256(state.get("candidate_checkpoint_sha256")):
        raise StateValidationError(f"phase {phase.value} requires a verified candidate checkpoint")
    return dict(state)


def transition_state(
    current: Mapping[str, Any],
    next_phase: Phase | str,
    *,
    reason: str,
    generation_manifest_sha256: str | None = None,
    candidate_checkpoint_sha256: str | None = None,
    completed_shard_hosts: Sequence[str] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Return the next hash-chained state, rejecting every out-of-order transition."""
    current = validate_state(current)
    source = Phase(current["phase"])
    try:
        target = Phase(next_phase)
    except ValueError as error:
        raise TransitionError(f"unknown target phase {next_phase!r}") from error
    if target not in LEGAL_TRANSITIONS[source]:
        raise TransitionError(f"illegal campaign transition {source.value} -> {target.value}")
    if not reason.strip():
        raise TransitionError("a durable transition reason is required")
    proposed = dict(current)
    proposed["revision"] = current["revision"] + 1
    proposed["phase"] = target.value
    proposed["host_intents"] = dict(PHASE_HOST_INTENTS[target])
    proposed["previous_state_sha256"] = current["state_sha256"]
    proposed["updated_at"] = now or utc_now()
    proposed["last_transition"] = {
        "from": source.value,
        "to": target.value,
        "reason": reason.strip(),
        "at": proposed["updated_at"],
    }

    if generation_manifest_sha256 is not None:
        proposed["generation_manifest_sha256"] = generation_manifest_sha256
    if candidate_checkpoint_sha256 is not None:
        proposed["candidate_checkpoint_sha256"] = candidate_checkpoint_sha256
    if completed_shard_hosts is not None:
        proposed["completed_shard_hosts"] = sorted(set(completed_shard_hosts))

    if target is Phase.ROUND_ALLOCATED:
        prior_round = current.get("round_index")
        round_index = 0 if prior_round is None else prior_round + 1
        proposed.update(
            {
                "round_index": round_index,
                "generation_dataset_id": f"G[{round_index}]",
                "generation_manifest_sha256": None,
                "candidate_checkpoint_id": f"T[{round_index}]",
                "candidate_checkpoint_sha256": None,
                "completed_shard_hosts": [],
            }
        )
    elif target is Phase.BOOTSTRAP_GENERATING:
        proposed.update(
            {
                "generation_dataset_id": "bootstrap-100000",
                "generation_manifest_sha256": None,
                "candidate_checkpoint_id": "bootstrap-candidate",
                "candidate_checkpoint_sha256": None,
                "completed_shard_hosts": [],
            }
        )
    elif target is Phase.INCUMBENT_PROMOTED:
        if not _is_sha256(current.get("candidate_checkpoint_sha256")):
            raise TransitionError("only a verified candidate checkpoint can be promoted")
        promotion_index = (
            0 if current.get("promotion_index") is None else current["promotion_index"] + 1
        )
        proposed["promotion_index"] = promotion_index
        proposed["incumbent_checkpoint_id"] = f"C[{promotion_index}]"
        proposed["incumbent_checkpoint_sha256"] = current["candidate_checkpoint_sha256"]

    proposed["state_sha256"] = content_sha256(proposed, hash_field="state_sha256")
    try:
        return validate_state(proposed)
    except StateValidationError as error:
        raise TransitionError(str(error)) from error


def validate_transition(current: Mapping[str, Any], proposed: Mapping[str, Any]) -> None:
    current = validate_state(current)
    proposed = validate_state(proposed)
    if proposed["revision"] != current["revision"] + 1:
        raise TransitionError("proposed state revision must increment by exactly one")
    if proposed["previous_state_sha256"] != current["state_sha256"]:
        raise TransitionError("proposed state does not extend the current state hash")
    source = Phase(current["phase"])
    target = Phase(proposed["phase"])
    if target not in LEGAL_TRANSITIONS[source]:
        raise TransitionError(f"illegal campaign transition {source.value} -> {target.value}")
    transition = proposed["last_transition"]
    if transition["from"] != source.value:
        raise TransitionError("proposed transition metadata names the wrong source phase")
    expected = transition_state(
        current,
        target,
        reason=transition["reason"],
        generation_manifest_sha256=proposed.get("generation_manifest_sha256"),
        candidate_checkpoint_sha256=proposed.get("candidate_checkpoint_sha256"),
        completed_shard_hosts=proposed.get("completed_shard_hosts"),
        now=proposed["updated_at"],
    )
    if proposed != expected:
        raise TransitionError("proposed state is not the canonical result of its transition")


def read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise StateValidationError(f"cannot read campaign state {path}: {error}") from error
    if not isinstance(value, dict):
        raise StateValidationError("campaign state must be a JSON object")
    return validate_state(value)


def write_state(
    path: Path,
    state: Mapping[str, Any],
    *,
    expected_current: Mapping[str, Any] | None = None,
) -> None:
    state = validate_state(state)
    if expected_current is not None:
        validate_transition(expected_current, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if expected_current is not None:
            if (
                path.exists()
                and read_state(path)["state_sha256"] != expected_current["state_sha256"]
            ):
                raise TransitionError(
                    "durable campaign state changed before compare-and-swap write"
                )
        elif path.exists():
            raise TransitionError(
                f"refusing to overwrite existing campaign state without CAS: {path}"
            )
        _atomic_write_json(path, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_decision_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = set(DECISION_LOG_JSON_SCHEMA["required"])
    if set(entry) != required:
        raise DecisionLogError(
            f"decision entry keys differ: missing={sorted(required - set(entry))}, "
            f"extra={sorted(set(entry) - required)}"
        )
    if (
        entry.get("schema_version") != SCHEMA_VERSION
        or entry.get("schema_id") != DECISION_LOG_SCHEMA_ID
    ):
        raise DecisionLogError("unsupported decision-log schema")
    if entry.get("campaign_id") != CAMPAIGN_ID or _contains_forbidden_host(entry):
        raise DecisionLogError("decision entry has the wrong campaign or names john4")
    if not isinstance(entry.get("sequence"), int) or entry["sequence"] < 0:
        raise DecisionLogError("decision sequence must be a nonnegative integer")
    if entry.get("decision_kind") not in {
        "bounded-operational-adaptation",
        "scientific-contract-amendment",
    }:
        raise DecisionLogError("unknown decision kind")
    for field in (
        "actor",
        "chosen_action",
        "rollback_path",
        "recorded_at",
    ):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise DecisionLogError(f"{field} must be a nonempty string")
    for field in ("triggering_evidence", "alternatives_considered", "affected_artifacts"):
        value = entry.get(field)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise DecisionLogError(f"{field} must be a nonempty string array")
    for field in (
        "campaign_state_sha256",
        "authorization_sha256",
        "previous_decision_sha256",
    ):
        if entry.get(field) is not None and not _is_sha256(entry[field]):
            raise DecisionLogError(f"{field} must be null or a SHA-256 digest")
    if entry["decision_kind"] == "scientific-contract-amendment" and not _is_sha256(
        entry.get("authorization_sha256")
    ):
        raise DecisionLogError("scientific contract changes require an amended authorization hash")
    if not _is_sha256(entry.get("decision_sha256")):
        raise DecisionLogError("decision_sha256 must be a SHA-256 digest")
    if entry["decision_sha256"] != content_sha256(entry, hash_field="decision_sha256"):
        raise DecisionLogError("decision entry hash does not match its payload")
    return dict(entry)


def new_storage_supersession_genesis(
    *,
    legacy_campaign_state_sha256: str,
    legacy_decision_head_sha256: str,
    canonical_state: Mapping[str, Any],
    authorization_sha256: str,
    now: str | None = None,
) -> dict[str, Any]:
    """Anchor the fresh John1 chain to immutable legacy identities, not copied bytes."""
    state = validate_state(canonical_state)
    value: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "sequence": 0,
        "decision_kind": "storage-supersession-genesis",
        "recorded_at": now or utc_now(),
        "legacy_storage_host": "john2",
        "legacy_campaign_root": str(FROZEN_LEGACY_JOHN2_ROOT),
        "legacy_campaign_state_sha256": legacy_campaign_state_sha256,
        "legacy_decision_head_sha256": legacy_decision_head_sha256,
        "legacy_evidence_immutable": True,
        "canonical_storage_host": STORAGE_HOST,
        "canonical_campaign_root": str(CAMPAIGN_ROOT),
        "canonical_campaign_state_sha256": state["state_sha256"],
        "authorization_sha256": authorization_sha256,
        "previous_decision_sha256": None,
    }
    value["decision_sha256"] = content_sha256(value, hash_field="decision_sha256")
    return validate_storage_supersession_genesis(value)


def validate_storage_supersession_genesis(value: Mapping[str, Any]) -> dict[str, Any]:
    required = set(STORAGE_SUPERSESSION_GENESIS_JSON_SCHEMA["required"])
    if not isinstance(value, Mapping) or set(value) != required:
        raise DecisionLogError("storage-supersession genesis field set differs")
    fixed = {
        "schema_version": 1,
        "schema_id": STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "sequence": 0,
        "decision_kind": "storage-supersession-genesis",
        "legacy_storage_host": "john2",
        "legacy_campaign_root": str(FROZEN_LEGACY_JOHN2_ROOT),
        "legacy_evidence_immutable": True,
        "canonical_storage_host": STORAGE_HOST,
        "canonical_campaign_root": str(CAMPAIGN_ROOT),
        "previous_decision_sha256": None,
    }
    if any(value.get(name) != expected for name, expected in fixed.items()):
        raise DecisionLogError("storage-supersession genesis identity differs")
    if not isinstance(value.get("recorded_at"), str) or not value["recorded_at"].strip():
        raise DecisionLogError("storage-supersession genesis timestamp is empty")
    for name in (
        "legacy_campaign_state_sha256",
        "legacy_decision_head_sha256",
        "canonical_campaign_state_sha256",
        "authorization_sha256",
        "decision_sha256",
    ):
        if not _is_sha256(value.get(name)):
            raise DecisionLogError(f"storage-supersession genesis {name} is not SHA-256")
    if _contains_forbidden_host(value):
        raise DecisionLogError("storage-supersession genesis names john4")
    if value["decision_sha256"] != content_sha256(value, hash_field="decision_sha256"):
        raise DecisionLogError("storage-supersession genesis hash differs")
    return dict(value)


def encode_storage_supersession_genesis(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(validate_storage_supersession_genesis(value)) + b"\n"


def write_storage_supersession_genesis(path: Path, value: Mapping[str, Any]) -> None:
    """Create the canonical first decision line once; never replace or append here."""
    encoded = encode_storage_supersession_genesis(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise DecisionLogError("refusing to replace an existing decision log genesis") from error
    _fsync_directory(path.parent)


def read_decision_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError as error:
        raise DecisionLogError(f"cannot read decision log {path}: {error}") from error
    return _parse_decision_log_text(text)


def append_decision(
    path: Path,
    *,
    actor: str,
    triggering_evidence: Sequence[str],
    alternatives_considered: Sequence[str],
    chosen_action: str,
    affected_artifacts: Sequence[str],
    rollback_path: str,
    state: Mapping[str, Any] | None = None,
    decision_kind: str = "bounded-operational-adaptation",
    authorization_sha256: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Append and fsync one hash-chained decision while holding an exclusive lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_text = handle.read()
        existing = _parse_decision_log_text(existing_text)
        if state is not None:
            state = validate_state(state)
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "schema_id": DECISION_LOG_SCHEMA_ID,
            "campaign_id": CAMPAIGN_ID,
            "sequence": len(existing),
            "decision_kind": decision_kind,
            "actor": actor.strip(),
            "recorded_at": now or utc_now(),
            "campaign_state_revision": None if state is None else state["revision"],
            "campaign_state_sha256": None if state is None else state["state_sha256"],
            "triggering_evidence": [item.strip() for item in triggering_evidence],
            "alternatives_considered": [item.strip() for item in alternatives_considered],
            "chosen_action": chosen_action.strip(),
            "affected_artifacts": [item.strip() for item in affected_artifacts],
            "rollback_path": rollback_path.strip(),
            "authorization_sha256": authorization_sha256,
            "previous_decision_sha256": None if not existing else existing[-1]["decision_sha256"],
        }
        entry["decision_sha256"] = content_sha256(entry, hash_field="decision_sha256")
        entry = validate_decision_entry(entry)
        encoded = canonical_json_bytes(entry) + b"\n"
        handle.seek(0, os.SEEK_END)
        os.write(handle.fileno(), encoded)
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    _fsync_directory(path.parent)
    return entry


def _parse_decision_log_text(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    if not text.endswith("\n"):
        raise DecisionLogError("decision log has an incomplete trailing record")
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise DecisionLogError(f"invalid decision log line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise DecisionLogError(f"decision log line {line_number} is not an object")
        if value.get("schema_id") == STORAGE_SUPERSESSION_GENESIS_SCHEMA_ID:
            if entries:
                raise DecisionLogError(
                    "the storage-supersession genesis is allowed only as the first line"
                )
            entry = validate_storage_supersession_genesis(value)
            entry["storage_supersession_genesis"] = True
        elif "schema_id" not in value:
            if entries:
                raise DecisionLogError(
                    "the pre-contract decision record is allowed only as the genesis line"
                )
            entry = _validate_legacy_genesis_decision(value, raw_line=line)
        else:
            entry = validate_decision_entry(value)
        expected_previous = None if not entries else entries[-1]["decision_sha256"]
        if (
            entry.get("legacy_genesis") is not True
            and entry.get("storage_supersession_genesis") is not True
            and (
            entry["sequence"] != len(entries)
            or entry["previous_decision_sha256"] != expected_previous
            )
        ):
            raise DecisionLogError(f"decision log chain mismatch on line {line_number}")
        entries.append(entry)
    return entries


def _validate_legacy_genesis_decision(value: Mapping[str, Any], *, raw_line: str) -> dict[str, Any]:
    """Validate and anchor the one decision written before schemas were frozen."""
    required = {
        "schema_version",
        "timestamp",
        "campaign_id",
        "decision_id",
        "trigger",
        "alternatives",
        "chosen_action",
        "affected_artifacts",
        "rollback",
        "scientific_contract_changed",
    }
    if set(value) != required:
        raise DecisionLogError("pre-contract genesis decision has an unknown field layout")
    if value.get("schema_version") != 1 or value.get("campaign_id") != CAMPAIGN_ID:
        raise DecisionLogError("pre-contract genesis decision has the wrong identity")
    if value.get("scientific_contract_changed") is not False:
        raise DecisionLogError("pre-contract genesis cannot authorize a scientific contract change")
    if _contains_forbidden_host(value):
        raise DecisionLogError("pre-contract genesis decision may not name john4")
    for field in ("timestamp", "decision_id", "trigger", "chosen_action", "rollback"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise DecisionLogError(f"pre-contract genesis {field} must be nonempty")
    for field in ("alternatives", "affected_artifacts"):
        items = value.get(field)
        if (
            not isinstance(items, list)
            or not items
            or not all(isinstance(item, str) and item.strip() for item in items)
        ):
            raise DecisionLogError(f"pre-contract genesis {field} must be a string array")
    normalized = dict(value)
    normalized.update(
        {
            "legacy_genesis": True,
            "sequence": 0,
            "previous_decision_sha256": None,
            "decision_sha256": hashlib.sha256(raw_line.encode()).hexdigest(),
        }
    )
    return normalized


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)
