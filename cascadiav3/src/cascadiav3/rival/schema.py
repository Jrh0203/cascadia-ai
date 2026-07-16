"""Versioned, fail-closed schemas shared by the Rival CPU tools.

The schemas in this module are additive.  They do not overload existing v1-v4
expert tensors or replay Q/value fields.  Validators reject unknown fields so
an artifact cannot silently acquire new scientific semantics without a schema
revision.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RIVAL_POLICY_IDENTITY_SCHEMA_ID = "cascadiav3.rival_policy_identity.v1"
RIVAL_ROOT_MANIFEST_SCHEMA_ID = "cascadiav3.rival_root_manifest.v1"
RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID = "cascadiav3.rival_terminal_pair_ledger.v1"
RIVAL_BOUND_CERTIFICATE_SCHEMA_ID = "cascadiav3.rival_bound_certificate.v1"
RIVAL_POWER_ENVELOPE_SCHEMA_ID = "cascadiav3.rival_power_envelope.v1"
RIVAL_GPU_PERMIT_SCHEMA_ID = "cascadiav3.rival_gpu_permit.v1"
RIVAL_PREFERENCE_SHARD_SCHEMA_ID = "cascadiav3.rival_preference_shard.v1"
RIVAL_TRAINING_VIEW_SCHEMA_ID = "cascadiav3.rival_training_view.v1"
RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID = "cascadiav3.rival_terminal_panel_plan.v1"
RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID = "cascadiav3.rival_coefficient_calibration.v1"
RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID = "cascadiav3.rival_potential_root_census.v1"
RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID = "cascadiav3.rival_error_family_ledger.v1"
RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID = "cascadiav3.rival_allocation_registry.v1"
MAX_RIVAL_JSON_BYTES = 64 * 1024 * 1024
_JSON_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RivalSchemaDefinition:
    schema_id: str
    artifact_kind: str
    version: int
    description: str


RIVAL_SCHEMA_DEFINITIONS: dict[str, RivalSchemaDefinition] = {
    RIVAL_POLICY_IDENTITY_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_POLICY_IDENTITY_SCHEMA_ID,
        "rival_policy_identity",
        1,
        "Non-substitutable B_k, pi_L, W_k, or M_(k+1) policy identity.",
    ),
    RIVAL_ROOT_MANIFEST_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_ROOT_MANIFEST_SCHEMA_ID,
        "rival_root_manifest",
        1,
        "Frozen root, candidate, panel, coefficient, bound, and error identities.",
    ),
    RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID,
        "rival_terminal_pair_ledger",
        1,
        "Immutable terminal action-pair outcomes and operational failures.",
    ),
    RIVAL_BOUND_CERTIFICATE_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_BOUND_CERTIFICATE_SCHEMA_ID,
        "rival_bound_certificate",
        1,
        "Rust-authored certified high/low score-difference ranges.",
    ),
    RIVAL_POWER_ENVELOPE_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_POWER_ENVELOPE_SCHEMA_ID,
        "rival_power_envelope",
        1,
        "Symbolic, explicitly non-funding pre-measurement work envelope.",
    ),
    RIVAL_GPU_PERMIT_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_GPU_PERMIT_SCHEMA_ID,
        "rival_gpu_permit",
        1,
        "Future hash-pinned accelerator authorization; denied in pre-GPU phases.",
    ),
    RIVAL_PREFERENCE_SHARD_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
        "rival_preference_shard",
        1,
        "Categorical preference sidecar with complete provenance.",
    ),
    RIVAL_TRAINING_VIEW_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_TRAINING_VIEW_SCHEMA_ID,
        "rival_training_view",
        1,
        "Hash-checked join of an unchanged expert shard and preference sidecar.",
    ),
    RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
        "rival_terminal_panel_plan",
        1,
        "Exact preregistered terminal units and random-key commitments for one panel.",
    ),
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
        "rival_coefficient_calibration",
        1,
        "Immutable control-variate coefficient and exact calibration provenance.",
    ),
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
        "rival_potential_root_census",
        1,
        "Externally registry-derived census of every root eligible for one error family.",
    ),
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
        "rival_error_family_ledger",
        1,
        "Census-complete immutable family-wise error allocation.",
    ),
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID: RivalSchemaDefinition(
        RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
        "rival_allocation_registry",
        1,
        "Exact root-cohort and complete-game seed-commitment allocation registry.",
    ),
}


class RivalSchemaError(ValueError):
    """Raised when a Rival artifact fails a versioned contract."""


_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_CAPABILITY_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True)
class _ValidationCapability:
    """Process-local proof that a specific content identity passed its validator.

    The capability is never serialized and is intentionally unavailable through
    the package public surface.  Its keyed authenticator prevents a copied
    sentinel or ``dataclasses.replace`` from blessing substituted content.
    """

    artifact_kind: str
    content_sha256: str
    authenticator: bytes


def _issue_validation_capability(artifact_kind: str, content_sha256: str) -> object:
    kind = require_nonempty_string(artifact_kind, "artifact_kind")
    digest = require_sha256(content_sha256, "content_sha256")
    message = f"{kind}\0{digest}".encode()
    return _ValidationCapability(
        kind,
        digest,
        hmac.new(_CAPABILITY_SECRET, message, hashlib.sha256).digest(),
    )


def _require_validation_capability(
    capability: object,
    *,
    artifact_kind: str,
    content_sha256: str,
) -> None:
    kind = require_nonempty_string(artifact_kind, "artifact_kind")
    digest = require_sha256(content_sha256, "content_sha256")
    if not isinstance(capability, _ValidationCapability):
        raise RivalSchemaError(f"{kind} must be produced by its artifact validator")
    message = f"{kind}\0{digest}".encode()
    expected = hmac.new(_CAPABILITY_SECRET, message, hashlib.sha256).digest()
    if (
        capability.artifact_kind != kind
        or capability.content_sha256 != digest
        or not hmac.compare_digest(capability.authenticator, expected)
    ):
        raise RivalSchemaError(f"{kind} validation capability does not match its content")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical JSON encoding used by Rival Python artifacts."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RivalSchemaError(f"value is not canonical-JSON encodable: {exc}") from exc


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_without_hash(
    record: Mapping[str, Any], hash_field: str = "content_sha256"
) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != hash_field}


def attach_content_hash(
    record: Mapping[str, Any], hash_field: str = "content_sha256"
) -> dict[str, Any]:
    if hash_field in record:
        raise RivalSchemaError(f"refusing to overwrite existing {hash_field!r}")
    result = dict(record)
    result[hash_field] = sha256_hex(result)
    return result


def write_new_bytes(path: str | Path, payload: bytes) -> Path:
    """Durably publish ``payload`` exactly once without replacing evidence.

    The temporary file lives beside the destination, is fsynced before an
    atomic hard-link publication, and is always removed by this process.  A
    hard link is intentional: unlike ``rename``, it fails when the destination
    already exists, including under a concurrent writer.  Rival artifacts are
    immutable evidence, so replacement is never an acceptable default.
    """

    if not isinstance(payload, bytes):
        raise RivalSchemaError("immutable artifact payload must be bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RivalSchemaError(
                f"refusing to replace immutable Rival artifact {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_new_canonical_json(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Durably publish one canonical-JSON object with a trailing newline."""

    return write_new_bytes(path, canonical_json_bytes(record) + b"\n")


def _stable_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bounded_stable_file(
    path: str | Path,
    *,
    field: str,
    require_single_link: bool,
    expected_file_sha256: str | None = None,
) -> bytearray:
    """Read one stable regular file into one bounded buffer.

    The size metadata is checked before the first read, and a one-byte growth
    probe makes the 64 MiB ceiling inclusive.  When a byte pin is supplied,
    SHA-256 is updated per chunk so no second joined byte buffer is needed.
    """

    source = Path(path)
    expected = (
        require_sha256(expected_file_sha256, "expected_file_sha256")
        if expected_file_sha256 is not None
        else None
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise RivalSchemaError(f"could not safely open {field} {source}: {exc}") from exc
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RivalSchemaError(f"{field} must be a regular file")
            if require_single_link and before.st_nlink != 1:
                raise RivalSchemaError(f"{field} must be a single-link regular file")
            if before.st_size > MAX_RIVAL_JSON_BYTES:
                raise RivalSchemaError(
                    f"{field} exceeds maximum JSON artifact size of {MAX_RIVAL_JSON_BYTES} bytes"
                )

            data = bytearray()
            digest = hashlib.sha256() if expected is not None else None
            while True:
                # Read one byte past the remaining allowance to detect growth
                # without allocating an unbounded final chunk.
                read_size = min(
                    _JSON_READ_CHUNK_BYTES,
                    MAX_RIVAL_JSON_BYTES - len(data) + 1,
                )
                chunk = os.read(descriptor, read_size)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_RIVAL_JSON_BYTES:
                    raise RivalSchemaError(
                        f"{field} exceeds maximum JSON artifact size of "
                        f"{MAX_RIVAL_JSON_BYTES} bytes"
                    )
                if digest is not None:
                    digest.update(chunk)
            after = os.fstat(descriptor)
        except OSError as exc:
            raise RivalSchemaError(f"could not safely read {field} {source}: {exc}") from exc
    finally:
        os.close(descriptor)

    if _stable_file_signature(before) != _stable_file_signature(after):
        raise RivalSchemaError(f"{field} changed while being read")
    if expected is not None:
        assert digest is not None
        observed = digest.hexdigest()
        if observed != expected:
            raise RivalSchemaError(
                f"{field} file SHA-256 mismatch: observed {observed}; expected {expected}"
            )
    return data


def read_strict_json_value(path: str | Path, *, field: str = "JSON artifact") -> Any:
    """Read one bounded, stable UTF-8 JSON value with strict token rules."""

    source = Path(path)
    data = _read_bounded_stable_file(
        source,
        field=field,
        require_single_link=False,
    )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RivalSchemaError(f"{field} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RivalSchemaError(f"{field} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RivalSchemaError(f"{field} {source} is not valid UTF-8 JSON: {exc}") from exc
    return value


def read_strict_json_object(path: str | Path, *, field: str = "JSON artifact") -> dict[str, Any]:
    """Read one strict JSON value and require an object at the top level."""

    value = read_strict_json_value(path, field=field)
    if not isinstance(value, dict):
        raise RivalSchemaError(f"{field} {Path(path)} must contain one JSON object")
    return value


def read_pinned_canonical_json_object(
    path: str | Path,
    *,
    expected_file_sha256: str,
    field: str = "JSON artifact",
) -> dict[str, Any]:
    """Read one single-link canonical artifact through a stable descriptor.

    Scientific inputs are byte-pinned in addition to carrying a semantic
    content hash.  Requiring the sole canonical encoding prevents two byte
    representations from sharing one decoded meaning, while the descriptor
    checks reject symlinks, hard links, and in-place mutation during the read.
    """

    data = _read_bounded_stable_file(
        path,
        field=field,
        require_single_link=True,
        expected_file_sha256=expected_file_sha256,
    )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RivalSchemaError(f"{field} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RivalSchemaError(f"{field} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RivalSchemaError(f"{field} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RivalSchemaError(f"{field} must contain one JSON object")
    if data != canonical_json_bytes(value) + b"\n":
        raise RivalSchemaError(f"{field} must use canonical JSON with one trailing newline")
    return value


def verify_content_hash(record: Mapping[str, Any], hash_field: str = "content_sha256") -> str:
    observed = record.get(hash_field)
    require_sha256(observed, hash_field)
    assert isinstance(observed, str)
    normalized = observed.removeprefix("sha256:")
    expected = sha256_hex(content_without_hash(record, hash_field))
    if normalized != expected:
        raise RivalSchemaError(
            f"{hash_field} mismatch: observed {normalized}; recomputed {expected}"
        )
    return normalized


def require_exact_keys(
    record: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    where: str = "record",
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    keys = set(record)
    missing = sorted(required_set - keys)
    unknown = sorted(keys - allowed)
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise RivalSchemaError(f"{where} keys fail closed: " + ", ".join(parts))


def require_schema(record: Mapping[str, Any], expected: str) -> None:
    if expected not in RIVAL_SCHEMA_DEFINITIONS:
        raise RivalSchemaError(f"validator requested unknown Rival schema {expected!r}")
    observed = record.get("schema_id")
    if observed != expected:
        raise RivalSchemaError(f"schema_id mismatch: observed {observed!r}; expected {expected!r}")


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise RivalSchemaError(f"{field} must be a non-empty, whitespace-trimmed string")
    return value


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RivalSchemaError(f"{field} must be a lowercase SHA-256 hex digest")
    return value.removeprefix("sha256:")


def require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RivalSchemaError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RivalSchemaError(f"{field} must be finite")
    return result


def require_probability(value: Any, field: str, *, allow_zero: bool = False) -> float:
    result = require_finite(value, field)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not lower_ok or result >= 1.0:
        interval = "[0, 1)" if allow_zero else "(0, 1)"
        raise RivalSchemaError(f"{field} must be in {interval}")
    return result


def require_positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RivalSchemaError(f"{field} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RivalSchemaError(f"{field} must be {qualifier}")
    return value


def assert_no_accelerator_strings(value: Any, *, where: str = "artifact") -> None:
    """Guard CPU contracts against accidentally embedded execution requests.

    Artifact prose may discuss unresolved GPU hours, so only exact execution
    selector keys are inspected; ordinary descriptive strings are untouched.
    """
    execution_keys = {"device", "accelerator", "execution_device"}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}"
                if key in execution_keys and child not in (None, "cpu", "UNRESOLVED"):
                    raise RivalSchemaError(
                        f"{child_path} requests a non-CPU execution target: {child!r}"
                    )
                visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, where)
