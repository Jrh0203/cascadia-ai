"""Fail-closed proof that a John1 R2-MAP process cannot write local files."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
from typing import Any

SANDBOX_PROFILE = "\n".join(
    (
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        '(allow file-write* (literal "/dev/null"))',
    )
)
SANDBOX_PROFILE_SHA256 = hashlib.sha256(SANDBOX_PROFILE.encode("ascii")).hexdigest()
PROFILE_ENVIRONMENT = "R2_MAP_NO_LOCAL_WRITE_PROFILE_SHA256"
ATTESTATION_ENVIRONMENT = "R2_MAP_LOCAL_WRITE_ATTESTATION_RELATIVE"
JOHN1_ATTESTATION_REQUEST_PREFIX = "req-john1-attestation-"
JOHN1_MLX_INTERPRETER = (
    "/Users/johnherrick/.local/share/uv/python/"
    "cpython-3.12.13-macos-aarch64-none/bin/python3.12"
)


def john1_attestation_publication_receipt_relative(attestation_sha256: str) -> str:
    if (
        not isinstance(attestation_sha256, str)
        or len(attestation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in attestation_sha256)
    ):
        raise ValueError("John1 attestation digest is invalid")
    return (
        f"control/receipts/{JOHN1_ATTESTATION_REQUEST_PREFIX}"
        f"{attestation_sha256[:32]}.json"
    )


def require_no_local_write_sandbox(probe_path: Path | None = None) -> dict[str, Any]:
    """Prove the current process is subject to the frozen file-write denial.

    Opening an existing regular source file for write without truncation has no
    side effect when unsandboxed, while the frozen sandbox must reject the open.
    This avoids creating a probe file even if a caller forgot the sandbox.
    """

    if os.environ.get(PROFILE_ENVIRONMENT) != SANDBOX_PROFILE_SHA256:
        raise RuntimeError("R2-MAP local-write sandbox profile identity is absent")
    attestation_relative = os.environ.get(ATTESTATION_ENVIRONMENT)
    if (
        not attestation_relative
        or attestation_relative.startswith("/")
        or ".." in Path(attestation_relative).parts
    ):
        raise RuntimeError("R2-MAP local-write attestation path is absent or unsafe")
    probe = (probe_path or Path(__file__)).resolve(strict=True)
    try:
        descriptor = os.open(probe, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM}:
            raise RuntimeError("R2-MAP local-write denial probe failed unexpectedly") from error
        probe_errno = error.errno
    else:
        os.close(descriptor)
        raise RuntimeError("R2-MAP process is not protected by local file-write denial")
    return {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.john1-local-write-sandbox.v1",
        "profile_sha256": SANDBOX_PROFILE_SHA256,
        "probe": str(probe),
        "probe_errno": probe_errno,
        "all_local_file_writes_denied": True,
        "allowed_write_path": "/dev/null",
        "attestation_relative": attestation_relative,
    }
