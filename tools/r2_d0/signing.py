"""OpenSSH Ed25519 stdin signing and verification without temporary files."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from .canonical import (
    PUBLIC_KEY_NAMESPACE,
    SIGNATURE_SCHEMA,
    D0Error,
    canonical_json,
    document_sha256,
    sha256_bytes,
    validate_signature_bundle,
)

SSH_KEYGEN = "/usr/bin/ssh-keygen"
SIGNER_IDENTITY = "cascadia-r2-map-d0"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_SIGNATURE_BYTES = 16 * 1024
MAX_PUBLIC_KEY_BYTES = 16 * 1024


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
        or observed.st_mode & 0o022
        or observed.st_size > maximum
    ):
        raise D0Error(f"{label} metadata or size is unsafe")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            value = b"".join(chunks)
            if len(value) != observed.st_size:
                raise D0Error(f"{label} changed while reading")
        finally:
            os.close(descriptor)
    except OSError as error:
        raise D0Error(f"cannot read {label}") from error
    return value


def _private_key_preflight(path: Path) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise D0Error("private key cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise D0Error("private key is not a single-link regular file")
    if value.st_uid != os.getuid() or value.st_mode & 0o077:
        raise D0Error("private key owner or mode is unsafe")
    if value.st_size > 64 * 1024:
        raise D0Error("private key exceeds its byte limit")


def normalize_public_key(value: bytes) -> bytes:
    if len(value) > MAX_PUBLIC_KEY_BYTES:
        raise D0Error("public key exceeds its byte limit")
    try:
        line = value.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise D0Error("public key is not ASCII") from error
    fields = line.split()
    if len(fields) not in {2, 3} or fields[0] != "ssh-ed25519":
        raise D0Error("public key is not one OpenSSH Ed25519 key")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise D0Error("public key blob is not canonical base64") from error
    if base64.b64encode(blob).decode("ascii") != fields[1]:
        raise D0Error("public key base64 is not canonical")
    return f"ssh-ed25519 {fields[1]}\n".encode("ascii")


def public_key_fingerprint(value: bytes) -> str:
    normalized = normalize_public_key(value)
    blob = base64.b64decode(normalized.decode("ascii").split()[1], validate=True)
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def public_key_from_private(private_key: Path) -> bytes:
    _private_key_preflight(private_key)
    process = subprocess.run(
        [SSH_KEYGEN, "-y", "-f", str(private_key)],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if process.returncode != 0:
        raise D0Error("ssh-keygen could not derive the public key")
    return normalize_public_key(process.stdout)


def sign_stdin(private_key: Path, payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise D0Error("signed payload exceeds its byte limit")
    _private_key_preflight(private_key)
    public_key = public_key_from_private(private_key)
    process = subprocess.run(
        [
            SSH_KEYGEN,
            "-Y",
            "sign",
            "-f",
            str(private_key),
            "-n",
            PUBLIC_KEY_NAMESPACE,
            "-O",
            "hashalg=sha512",
        ],
        input=payload,
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if process.returncode != 0:
        raise D0Error("OpenSSH Ed25519 signing failed")
    signature = process.stdout
    if len(signature) > MAX_SIGNATURE_BYTES:
        raise D0Error("OpenSSH signature exceeds its byte limit")
    try:
        armored = signature.decode("ascii")
    except UnicodeDecodeError as error:
        raise D0Error("OpenSSH signature is not ASCII armored") from error
    bundle: dict[str, Any] = {
        "schema_id": SIGNATURE_SCHEMA,
        "schema_version": 1,
        "algorithm": "openssh-ed25519",
        "namespace": PUBLIC_KEY_NAMESPACE,
        "signer_identity": SIGNER_IDENTITY,
        "public_key_fingerprint": public_key_fingerprint(public_key),
        "public_key_sha256": sha256_bytes(public_key),
        "payload_sha256": sha256_bytes(payload),
        "signature_armored": armored,
        "signature_sha256": sha256_bytes(signature),
    }
    bundle["bundle_sha256"] = document_sha256(bundle, "bundle_sha256")
    validate_signature_bundle(bundle, payload_sha256=bundle["payload_sha256"])
    return bundle


def _pipe_bytes(value: bytes) -> int:
    read_descriptor, write_descriptor = os.pipe()
    try:
        written = 0
        while written < len(value):
            written += os.write(write_descriptor, value[written:])
    finally:
        os.close(write_descriptor)
    return read_descriptor


def verify_stdin(public_key: bytes, payload: bytes, bundle: dict[str, Any]) -> None:
    normalized = normalize_public_key(public_key)
    validate_signature_bundle(bundle, payload_sha256=sha256_bytes(payload))
    if bundle["public_key_sha256"] != sha256_bytes(normalized) or bundle[
        "public_key_fingerprint"
    ] != public_key_fingerprint(normalized):
        raise D0Error("signature bundle public key differs")
    allowed = SIGNER_IDENTITY.encode("ascii") + b" " + normalized
    signature = bundle["signature_armored"].encode("ascii")
    allowed_descriptor = _pipe_bytes(allowed)
    signature_descriptor = _pipe_bytes(signature)
    try:
        process = subprocess.run(
            [
                SSH_KEYGEN,
                "-Y",
                "verify",
                "-f",
                f"/dev/fd/{allowed_descriptor}",
                "-I",
                SIGNER_IDENTITY,
                "-n",
                PUBLIC_KEY_NAMESPACE,
                "-s",
                f"/dev/fd/{signature_descriptor}",
            ],
            input=payload,
            check=False,
            capture_output=True,
            pass_fds=(allowed_descriptor, signature_descriptor),
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    finally:
        os.close(allowed_descriptor)
        os.close(signature_descriptor)
    if process.returncode != 0:
        raise D0Error("OpenSSH Ed25519 verification failed")


def signature_bytes(bundle: dict[str, Any]) -> bytes:
    validate_signature_bundle(bundle, payload_sha256=bundle.get("payload_sha256", ""))
    return canonical_json(bundle)


def load_public_key(path: Path) -> bytes:
    return normalize_public_key(_read_bounded(path, MAX_PUBLIC_KEY_BYTES, "public key"))
