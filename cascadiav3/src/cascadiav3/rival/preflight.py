"""Standard-library-only, default-deny pre-GPU Rival phase checker."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import (
    RIVAL_GPU_PERMIT_SCHEMA_ID,
    RivalSchemaError,
    read_strict_json_object,
    require_exact_keys,
    require_finite,
    require_nonempty_string,
    require_schema,
    require_sha256,
    verify_content_hash,
)

# This constant is deliberately not configurable through an environment
# variable.  Enabling an accelerator is a reviewed source transition at P2,
# not a runtime convenience flag.
ACCELERATOR_PHASE_ENABLED = False
PREFLIGHT_EXPECTATION_SCHEMA_ID = "cascadiav3.rival_preflight_expectation.v1"


class PreflightError(ValueError):
    """Raised when a phase-readiness or permit contract is denied."""


@dataclass(frozen=True)
class PermitExpectation:
    phase: str
    requested_accelerator: str
    source_revision: str
    source_digest: str
    command_sha256: str
    preregistration_sha256: str
    requested_gpu_hours: float

    def __post_init__(self) -> None:
        require_nonempty_string(self.phase, "expectation.phase")
        if self.requested_accelerator not in {"cuda", "mps"}:
            raise RivalSchemaError("expectation accelerator must be cuda or mps")
        require_nonempty_string(self.source_revision, "expectation.source_revision")
        for field in ("source_digest", "command_sha256", "preregistration_sha256"):
            require_sha256(getattr(self, field), f"expectation.{field}")
        hours = require_finite(self.requested_gpu_hours, "expectation.requested_gpu_hours")
        if hours <= 0.0:
            raise RivalSchemaError("expectation requested_gpu_hours must be positive")


@dataclass(frozen=True)
class ValidatedFuturePermit:
    permit_id: str
    phase: str
    allowed_device: str
    max_gpu_hours: float
    expires_at: datetime
    content_sha256: str


_PERMIT_FIELDS = (
    "schema_id",
    "permit_id",
    "phase",
    "authority",
    "source_revision",
    "source_digest",
    "command_sha256",
    "preregistration_sha256",
    "allowed_device",
    "max_gpu_hours",
    "issued_at",
    "expires_at",
    "content_sha256",
)

_EXPECTATION_FIELDS = (
    "schema_id",
    "phase",
    "requested_accelerator",
    "source_revision",
    "source_digest",
    "command_sha256",
    "preregistration_sha256",
    "requested_gpu_hours",
    "content_sha256",
)


def validate_expectation_fixture(record: Mapping[str, Any]) -> PermitExpectation:
    """Validate the exact, hash-pinned preflight request fixture."""

    require_exact_keys(record, required=_EXPECTATION_FIELDS, where="preflight expectation")
    if record["schema_id"] != PREFLIGHT_EXPECTATION_SCHEMA_ID:
        raise RivalSchemaError("preflight expectation schema_id mismatch")
    verify_content_hash(record)
    return PermitExpectation(
        phase=require_nonempty_string(record["phase"], "phase"),
        requested_accelerator=require_nonempty_string(
            record["requested_accelerator"], "requested_accelerator"
        ),
        source_revision=require_nonempty_string(record["source_revision"], "source_revision"),
        source_digest="sha256:" + require_sha256(record["source_digest"], "source_digest"),
        command_sha256="sha256:" + require_sha256(record["command_sha256"], "command_sha256"),
        preregistration_sha256="sha256:"
        + require_sha256(record["preregistration_sha256"], "preregistration_sha256"),
        requested_gpu_hours=require_finite(record["requested_gpu_hours"], "requested_gpu_hours"),
    )


def _timestamp(value: Any, field: str) -> datetime:
    text = require_nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RivalSchemaError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RivalSchemaError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def validate_future_gpu_permit(
    record: Mapping[str, Any],
    *,
    expectation: PermitExpectation,
    now: datetime,
) -> ValidatedFuturePermit:
    """Validate the future wire without enabling it in the current phase."""
    if not isinstance(expectation, PermitExpectation):
        raise RivalSchemaError("future permit requires a typed PermitExpectation")
    if not isinstance(now, datetime):
        raise RivalSchemaError("current preflight time must be a datetime")
    require_schema(record, RIVAL_GPU_PERMIT_SCHEMA_ID)
    require_exact_keys(record, required=_PERMIT_FIELDS, where="GPU permit")
    permit_id = require_nonempty_string(record["permit_id"], "permit_id")
    phase = require_nonempty_string(record["phase"], "phase")
    authority = require_nonempty_string(record["authority"], "authority")
    if authority != "john_explicit_gpu_authorization":
        raise RivalSchemaError("permit authority is not the reserved explicit authorization")
    allowed_device = require_nonempty_string(record["allowed_device"], "allowed_device")
    if allowed_device not in {"cuda", "mps"}:
        raise RivalSchemaError("future GPU permit allowed_device must be cuda or mps")
    if phase != expectation.phase:
        raise RivalSchemaError("permit phase mismatch")
    if allowed_device != expectation.requested_accelerator:
        raise RivalSchemaError("permit allowed_device mismatch")
    issued = _timestamp(record["issued_at"], "issued_at")
    expires = _timestamp(record["expires_at"], "expires_at")
    if now.tzinfo is None:
        raise RivalSchemaError("current preflight time must include a timezone")
    now = now.astimezone(UTC)
    if expires <= issued:
        raise RivalSchemaError("permit expires_at must be later than issued_at")
    if now < issued or now >= expires:
        raise RivalSchemaError("permit is not currently valid (not-yet-valid or expired)")
    max_hours = require_finite(record["max_gpu_hours"], "max_gpu_hours")
    if max_hours <= 0.0:
        raise RivalSchemaError("max_gpu_hours must be positive")
    if (
        not isinstance(expectation.requested_gpu_hours, (int, float))
        or isinstance(expectation.requested_gpu_hours, bool)
        or not math.isfinite(expectation.requested_gpu_hours)
        or expectation.requested_gpu_hours <= 0.0
        or expectation.requested_gpu_hours > max_hours
    ):
        raise RivalSchemaError("requested GPU-hours are invalid or exceed the permit")
    exact_fields = {
        "source_revision": expectation.source_revision,
        "source_digest": expectation.source_digest,
        "command_sha256": expectation.command_sha256,
        "preregistration_sha256": expectation.preregistration_sha256,
    }
    for field, expected in exact_fields.items():
        observed = record[field]
        if field.endswith("sha256") or field.endswith("digest"):
            observed = "sha256:" + require_sha256(observed, field)
            expected = "sha256:" + require_sha256(expected, f"expected {field}")
        else:
            observed = require_nonempty_string(observed, field)
        if observed != expected:
            raise RivalSchemaError(f"permit {field} mismatch")
    content_hash = verify_content_hash(record)
    return ValidatedFuturePermit(
        permit_id=permit_id,
        phase=phase,
        allowed_device=allowed_device,
        max_gpu_hours=max_hours,
        expires_at=expires,
        content_sha256=content_hash,
    )


def validate_cpu_device(device: str) -> None:
    """Accept only explicit CPU; never probe availability of another device."""
    if device != "cpu":
        raise PreflightError(
            f"pre-GPU phases accept only explicit --device cpu; rejected {device!r}"
        )


def preflight_validate_only(
    *,
    device: str,
    permit: Mapping[str, Any] | None,
    expectation: PermitExpectation | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a stable DENIED report without importing a device library."""
    validate_cpu_device(device)
    if permit is None:
        return {
            "status": "DENIED",
            "reason_code": "PERMIT_MISSING",
            "device": "cpu",
            "accelerator_phase_enabled": False,
        }
    if expectation is None:
        raise PreflightError("permit validation requires a complete caller expectation")
    validated = validate_future_gpu_permit(
        permit, expectation=expectation, now=now or datetime.now(UTC)
    )
    if ACCELERATOR_PHASE_ENABLED:  # pragma: no cover - source-locked false in P0/P1
        raise AssertionError("accelerator source gate changed without preflight redesign")
    return {
        "status": "DENIED",
        "reason_code": "PRE_GPU_PHASE_LOCKED",
        "device": "cpu",
        "validated_future_permit_id": validated.permit_id,
        "accelerator_phase_enabled": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return read_strict_json_object(path, field="preflight input")
    except RivalSchemaError as exc:
        raise PreflightError(f"could not read {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    torch_loaded_before = "torch" in sys.modules
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("pre-GPU preflight supports only --validate-only")
    try:
        fixture = _load_json(args.fixture)
        expectation = validate_expectation_fixture(fixture)
        permit = _load_json(args.permit) if args.permit else None
        report = preflight_validate_only(device=args.device, permit=permit, expectation=expectation)
    except (KeyError, TypeError, ValueError, RivalSchemaError, PreflightError) as exc:
        report = {
            "status": "DENIED",
            "reason_code": "INVALID_REQUEST",
            "reason": str(exc),
            "accelerator_phase_enabled": False,
        }
    if not torch_loaded_before and "torch" in sys.modules:
        raise RuntimeError("preflight imported torch")
    print(json.dumps(report, sort_keys=True))
    expected_denials = {"PERMIT_MISSING", "PRE_GPU_PHASE_LOCKED"}
    return 0 if report.get("reason_code") in expected_denials else 2


if __name__ == "__main__":
    raise SystemExit(main())
