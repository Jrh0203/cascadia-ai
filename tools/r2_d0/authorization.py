"""Signed work-packet authorization shared by every mutating D0 command."""

from __future__ import annotations

import os
import pwd
import time
from typing import Any

from .bootstrap import HOST_USERS
from .canonical import (
    D0Error,
    load_canonical_json,
    sha256_bytes,
    validate_signature_bundle,
    validate_work_packet,
)
from .signing import public_key_fingerprint, verify_stdin

MAX_PACKET_BYTES = 1024 * 1024
MAX_SIGNATURE_BUNDLE_BYTES = 1024 * 1024
REPORT_PERSISTENCE_GRACE_MS = 30_000


def authorize_work_packet(
    packet_bytes: bytes,
    signature_bytes: bytes,
    public_key: bytes,
    *,
    expected_phase: str,
    required_operation: str,
    helper_sha256: str,
    current_user: str | None = None,
    now_unix_ms: int | None = None,
    require_full_execution_window: bool = False,
) -> dict[str, Any]:
    """Verify bytes, signature, host owner, validity window, phase, and operation."""

    packet = load_canonical_json(
        packet_bytes,
        maximum=MAX_PACKET_BYTES,
        label="work packet",
    )
    validate_work_packet(packet)
    signature = load_canonical_json(
        signature_bytes,
        maximum=MAX_SIGNATURE_BUNDLE_BYTES,
        label="work packet signature",
    )
    payload_sha256 = sha256_bytes(packet_bytes)
    validate_signature_bundle(signature, payload_sha256=payload_sha256)
    verify_stdin(public_key, packet_bytes, signature)
    kernel_user = pwd.getpwuid(os.getuid()).pw_name
    if current_user is not None and current_user != kernel_user:
        raise D0Error("caller-supplied owner differs from the kernel UID")
    observed_user = kernel_user
    if HOST_USERS[packet["host"]] != observed_user:
        raise D0Error("work packet is addressed to a different host owner")
    now = now_unix_ms if now_unix_ms is not None else time.time_ns() // 1_000_000
    if not packet["issued_unix_ms"] <= now <= packet["expires_unix_ms"]:
        raise D0Error("work packet is outside its validity window")
    if require_full_execution_window and (
        now + packet["limits"]["timeout_seconds"] * 1000 + REPORT_PERSISTENCE_GRACE_MS
        > packet["expires_unix_ms"]
    ):
        raise D0Error("work packet lacks a full execution and receipt-persistence window")
    if packet["phase"] != expected_phase:
        raise D0Error("work packet phase differs from the invoked command")
    if required_operation not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize the invoked operation")
    if packet["helper_sha256"] != helper_sha256:
        raise D0Error("work packet targets a different helper source identity")
    if packet["public_key_fingerprint"] != public_key_fingerprint(public_key):
        raise D0Error("work packet public-key fingerprint differs")
    return packet
