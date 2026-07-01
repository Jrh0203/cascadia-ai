"""Validate published output manifests and atomically import accepted artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from .errors import ArtifactValidationError
from .models import ArtifactFile, ArtifactManifest
from .object_store import ObjectStoreClient

MANIFEST_NAME = "manifest.json"
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_directory(root: Path) -> ArtifactManifest:
    root = root.resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"cannot read output manifest: {error}") from error
    required = {"schema_id", "protocol_version", "command", "files", "application_metadata"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ArtifactValidationError("output manifest schema differs")
    if raw["schema_id"] != "cascadia.cluster.output-manifest.v1":
        raise ArtifactValidationError("output manifest protocol differs")
    if (
        not isinstance(raw["command"], list)
        or any(not isinstance(value, str) for value in raw["command"])
        or not isinstance(raw["files"], list)
        or not isinstance(raw["application_metadata"], dict)
    ):
        raise ArtifactValidationError("output manifest fields are malformed")
    files = []
    seen: set[str] = set()
    for descriptor in raw["files"]:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
            raise ArtifactValidationError("output file descriptor differs")
        artifact = ArtifactFile(**descriptor)
        if artifact.path == MANIFEST_NAME or artifact.path in seen:
            raise ArtifactValidationError("output manifest duplicates or includes itself")
        seen.add(artifact.path)
        path = (root / artifact.path).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ArtifactValidationError("output artifact escapes root") from error
        if not path.is_file() or path.stat().st_size != artifact.bytes:
            raise ArtifactValidationError(f"output size differs: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ArtifactValidationError(f"output checksum differs: {artifact.path}")
        files.append(artifact)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if actual != seen:
        raise ArtifactValidationError("published output set differs from manifest")
    return ArtifactManifest(
        protocol_version=str(raw["protocol_version"]),
        command=tuple(raw["command"]),
        files=tuple(files),
        application_metadata=raw["application_metadata"],
    )


def atomic_import(source: Path, destination: Path) -> ArtifactManifest:
    manifest = validate_output_directory(source)
    if destination.exists():
        existing = validate_output_directory(destination)
        if existing != manifest:
            raise ArtifactValidationError(f"accepted artifact already differs: {destination}")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source, temporary / "payload")
        copied = validate_output_directory(temporary / "payload")
        if copied != manifest:
            raise ArtifactValidationError("artifact changed during canonical import")
        os.replace(temporary / "payload", destination)
        return copied
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a Bacalhau result archive without accepting link or traversal entries."""

    total = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ArtifactValidationError("published result archive has too many members")
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
            ):
                raise ArtifactValidationError(
                    f"published result archive contains unsafe member: {member.name}"
                )
            total += max(0, member.size)
            if total > MAX_ARCHIVE_BYTES:
                raise ArtifactValidationError("published result archive exceeds extraction limit")
        bundle.extractall(destination, members=members, filter="data")


def import_execution_result(
    *,
    object_store: ObjectStoreClient,
    job_id: str,
    execution_id: str,
    output_name: str,
    destination: Path,
) -> ArtifactManifest:
    """Download, validate, and atomically accept one execution-specific output."""

    workspace = Path(tempfile.mkdtemp(prefix=f"cascadia-{execution_id}."))
    try:
        archive = workspace / "result.tar.gz"
        object_store.download(
            object_store.config.result_bucket,
            object_store.result_key(job_id, execution_id),
            archive,
        )
        extracted = workspace / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        output = extracted / output_name
        if not output.is_dir():
            raise ArtifactValidationError(
                f"published result omits expected output directory: {output_name}"
            )
        return atomic_import(output, destination)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
