from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from cascadia_cluster import ArtifactValidationError
from cascadia_cluster.results import (
    _safe_extract,
    atomic_import,
    import_execution_result,
    validate_output_directory,
)


def _output(root: Path, payload: bytes = b"result") -> Path:
    root.mkdir()
    (root / "result.json").write_bytes(payload)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "cascadia.cluster.output-manifest.v1",
                "protocol_version": "test-v1",
                "command": ["worker", "--seed", "1"],
                "files": [
                    {
                        "path": "result.json",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "application_metadata": {"score": 95},
            }
        )
    )
    return root


def test_manifest_validation_and_atomic_import_are_idempotent(tmp_path: Path) -> None:
    source = _output(tmp_path / "source")
    destination = tmp_path / "canonical/item-0"
    expected = validate_output_directory(source)
    assert atomic_import(source, destination) == expected
    assert atomic_import(source, destination) == expected


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    source = _output(tmp_path / "source")
    (source / "result.json").write_bytes(b"tampered")
    with pytest.raises(ArtifactValidationError, match=r"size|checksum"):
        validate_output_directory(source)


def test_unmanifested_output_is_rejected(tmp_path: Path) -> None:
    source = _output(tmp_path / "source")
    (source / "surprise.txt").write_text("not declared")
    with pytest.raises(ArtifactValidationError, match="output set"):
        validate_output_directory(source)


class _FakeObjectStore:
    class _Config:
        result_bucket = "results"

    config = _Config()

    @staticmethod
    def result_key(job_id: str, execution_id: str) -> str:
        return f"executions/{job_id}/{execution_id}.tar.gz"

    def __init__(self, archive: Path) -> None:
        self.archive = archive

    def download(self, bucket: str, key: str, destination: Path) -> str:
        value = self.archive.read_bytes()
        destination.write_bytes(value)
        return hashlib.sha256(value).hexdigest()


def test_execution_archive_is_validated_and_imported(tmp_path: Path) -> None:
    payload = _output(tmp_path / "payload")
    archive = tmp_path / "result.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(payload, arcname="output-0")
    destination = tmp_path / "accepted/item"
    manifest = import_execution_result(
        object_store=_FakeObjectStore(archive),  # type: ignore[arg-type]
        job_id="job",
        execution_id="execution",
        output_name="output-0",
        destination=destination,
    )
    assert manifest.application_metadata["score"] == 95
    assert (destination / "result.json").read_bytes() == b"result"


def test_result_archive_traversal_and_links_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("unsafe")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="../escape")
    with pytest.raises(ArtifactValidationError, match="unsafe member"):
        _safe_extract(archive, tmp_path / "out")
