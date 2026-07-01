from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import blake3
import pytest

MODULE_PATH = Path(__file__).with_name("corrected_mid_tail_champion_migration.py")
SPEC = importlib.util.spec_from_file_location("corrected_mid_tail_champion_migration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)


def _repository(tmp_path: Path) -> tuple[Path, Path, Path, tuple[str, ...]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    identity_paths = (
        "Cargo.toml",
        "Cargo.lock",
        "legacy/crates/cascadia-ai/Cargo.toml",
        "legacy/crates/cascadia-ai/src/nnue.rs",
        "legacy/crates/cascadia-ai/examples/migrate_legacy_mid_v4_weights.rs",
        "tools/corrected_mid_tail_champion_migration.py",
        "tools/corrected_mid_tail_champion_audit.py",
        "docs/v2/decisions/0137-corrected-mid-tail-schema.md",
        "docs/v2/reports/corrected-mid-tail-v1-preregistration.md",
    )
    for index, relative_text in enumerate(identity_paths):
        path = repository / relative_text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n")
    source = repository / migration.SOURCE_FILENAME
    source.write_bytes(b"historical-production-champion")
    output_root = repository / "artifacts/experiments/corrected-mid-tail-v1"
    return repository, source, output_root, identity_paths


def _expected(source: Path) -> migration.ExpectedSource:
    payload = source.read_bytes()
    return migration.ExpectedSource(
        bytes=len(payload),
        blake3=blake3.blake3(payload).hexdigest(),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class FakePipeline:
    def __init__(self, *, audit_verdict: str = "pass") -> None:
        self.migrations = 0
        self.audits = 0
        self.audit_verdict = audit_verdict

    def migrate(self, repository: Path, source: Path, output: Path) -> None:
        assert repository.is_dir()
        self.migrations += 1
        output.write_bytes(b"NNUC-corrected-" + source.read_bytes())

    def audit(
        self,
        repository: Path,
        source: Path,
        corrected: Path,
        output: Path,
        source_label: str,
        corrected_label: str,
    ) -> dict[str, object]:
        assert repository.is_dir()
        self.audits += 1
        source_identity = migration.file_identity(source, source_label)
        corrected_identity = migration.file_identity(corrected, corrected_label)
        report = {
            "schema_version": 1,
            "audit_id": "corrected-mid-tail-champion-audit-v1",
            "verdict": self.audit_verdict,
            "failure_reasons": [] if self.audit_verdict == "pass" else ["synthetic_failure"],
            "contract": {
                "feature_count": 11_231,
                "hidden1": 512,
                "hidden2": 64,
                "head_format_version": 1,
            },
            "source": source_identity,
            "corrected": {
                **corrected_identity,
                "container_version": 1,
                "head_format_version": 1,
                "schema_tag_hex": b"MIDTAIL-CORR-V1\0".hex(),
                "feature_count": 11_231,
                "hidden1": 512,
                "hidden2": 64,
            },
            "checks": {"synthetic_exact_audit": self.audit_verdict == "pass"},
        }
        migration.write_json_atomic(output, report)
        return report


def _run(
    repository: Path,
    source: Path,
    output_root: Path,
    identity_paths: tuple[str, ...],
    pipeline: FakePipeline,
) -> migration.MigrationResult:
    return migration.migrate_champion(
        repository=repository,
        source=source,
        output_root=output_root,
        expected_source=_expected(source),
        migration_runner=pipeline.migrate,
        audit_runner=pipeline.audit,
        source_identity_paths=identity_paths,
    )


def test_publish_is_content_addressed_read_only_and_idempotent(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    pipeline = FakePipeline()
    first = _run(repository, source, output_root, identity_paths, pipeline)
    second = _run(repository, source, output_root, identity_paths, pipeline)

    assert first.reused is False
    assert second.reused is True
    assert first.artifact_directory == second.artifact_directory
    assert first.artifact_directory.name == first.manifest["output"]["blake3"]
    assert pipeline.migrations == 1
    assert pipeline.audits == 3
    assert {path.name for path in first.artifact_directory.iterdir()} == {
        migration.MODEL_FILENAME,
        migration.AUDIT_FILENAME,
        migration.MANIFEST_FILENAME,
        migration.MANIFEST_DIGEST_FILENAME,
    }
    for path in [first.artifact_directory, *first.artifact_directory.iterdir()]:
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


def test_source_drift_fails_before_migration(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    expected = _expected(source)
    source.write_bytes(source.read_bytes() + b"-drift")
    pipeline = FakePipeline()
    with pytest.raises(migration.MigrationError, match="identity mismatch"):
        migration.migrate_champion(
            repository=repository,
            source=source,
            output_root=output_root,
            expected_source=expected,
            migration_runner=pipeline.migrate,
            audit_runner=pipeline.audit,
            source_identity_paths=identity_paths,
        )
    assert pipeline.migrations == 0


def test_failed_independent_audit_publishes_nothing(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    pipeline = FakePipeline(audit_verdict="fail")
    with pytest.raises(migration.MigrationError, match="passing verdict"):
        _run(repository, source, output_root, identity_paths, pipeline)
    models_root = output_root / "models/blake3"
    assert not models_root.exists() or not list(models_root.glob("[0-9a-f]" * 64))


def test_existing_model_tampering_fails_closed(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    pipeline = FakePipeline()
    result = _run(repository, source, output_root, identity_paths, pipeline)
    model = result.artifact_directory / migration.MODEL_FILENAME
    model.chmod(0o644)
    model.write_bytes(model.read_bytes() + b"tampered")
    model.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="mismatch"):
        _run(repository, source, output_root, identity_paths, pipeline)
    assert pipeline.migrations == 1


def test_unexpected_file_in_immutable_directory_fails_closed(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    pipeline = FakePipeline()
    result = _run(repository, source, output_root, identity_paths, pipeline)
    result.artifact_directory.chmod(0o755)
    (result.artifact_directory / "extra.txt").write_text("unexpected\n")
    result.artifact_directory.chmod(0o555)
    with pytest.raises(migration.MigrationError, match="file set mismatch"):
        _run(repository, source, output_root, identity_paths, pipeline)


def test_rust_command_is_exact_and_uses_locked_corrected_feature(tmp_path: Path) -> None:
    command = migration.rust_migration_command(
        tmp_path / "source.bin",
        tmp_path / "output.bin",
    )
    assert command[:4] == ["cargo", "run", "--locked", "--quiet"]
    assert command[command.index("--features") + 1] == "legacy-mid-v4-fixed-v1"
    assert command[-3] == "--"


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    pipeline = FakePipeline()
    result = _run(repository, source, output_root, identity_paths, pipeline)
    manifest_path = result.artifact_directory / migration.MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["content_address"]["digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="manifest digest"):
        _run(repository, source, output_root, identity_paths, pipeline)


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    repository, source, output_root, identity_paths = _repository(tmp_path)
    linked = repository / "linked.bin"
    os.symlink(source, linked)
    pipeline = FakePipeline()
    with pytest.raises(migration.MigrationError, match="non-symlink"):
        migration.migrate_champion(
            repository=repository,
            source=linked,
            output_root=output_root,
            expected_source=_expected(source),
            migration_runner=pipeline.migrate,
            audit_runner=pipeline.audit,
            source_identity_paths=identity_paths,
        )
