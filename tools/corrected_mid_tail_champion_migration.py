#!/usr/bin/env python3
"""Publish the F5 Gate 12 champion migration as an immutable artifact."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_KIND = "corrected-mid-tail-champion-migration-v1"
EXPERIMENT_ID = "corrected-mid-tail-v1"
GATE = 12
SCHEMA_ID = "legacy-mid-v4-fixed-v1"
HISTORICAL_SCHEMA_ID = "historical-legacy-mid-v4opp-11231"
CARGO_FEATURE = "legacy-mid-v4-fixed-v1"
SOURCE_FILENAME = "nnue_weights_v4opp_modal_iter3.bin"
MODEL_FILENAME = "nnue_weights_legacy_mid_v4_fixed_v1_init.bin"
AUDIT_FILENAME = "audit.json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_DIGEST_FILENAME = "manifest.blake3"
PRODUCTION_SOURCE_BYTES = 23_134_992
PRODUCTION_SOURCE_BLAKE3 = "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
PRODUCTION_SOURCE_SHA256 = "f40627623d3686d7d2d6a2f8f109445f54e449f0d7045552ebe831f955a58f48"
SOURCE_IDENTITY_PATHS = (
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


class MigrationError(RuntimeError):
    """Raised when the immutable migration cannot be created or validated."""


@dataclass(frozen=True)
class ExpectedSource:
    bytes: int
    blake3: str
    sha256: str


@dataclass(frozen=True)
class MigrationResult:
    artifact_directory: Path
    manifest: dict[str, Any]
    reused: bool


MigrationRunner = Callable[[Path, Path, Path], None]
AuditRunner = Callable[[Path, Path, Path, str, str, Path], dict[str, Any]]


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "blake3":
        digest: Any = blake3.blake3()
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "bytes": path.stat().st_size,
        "blake3": _hash_file(path, "blake3"),
        "sha256": _hash_file(path, "sha256"),
    }


def _relative_label(path: Path, repository: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise MigrationError(f"{label} must remain inside the repository: {path}") from error


def _require_regular_non_symlink(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def validate_source(
    source: Path,
    repository: Path,
    expected: ExpectedSource,
) -> tuple[Path, dict[str, Any]]:
    source = _require_regular_non_symlink(source, "production champion")
    source_label = _relative_label(source, repository, "production champion")
    identity = file_identity(source, source_label)
    failures = []
    if identity["bytes"] != expected.bytes:
        failures.append(f"bytes expected={expected.bytes} found={identity['bytes']}")
    if identity["blake3"] != expected.blake3:
        failures.append(f"blake3 expected={expected.blake3} found={identity['blake3']}")
    if identity["sha256"] != expected.sha256:
        failures.append(f"sha256 expected={expected.sha256} found={identity['sha256']}")
    if failures:
        raise MigrationError("production champion identity mismatch: " + "; ".join(failures))
    return source, identity


def collect_source_identity(
    repository: Path,
    paths: Iterable[str] = SOURCE_IDENTITY_PATHS,
) -> dict[str, Any]:
    entries = []
    for relative_text in sorted(paths):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise MigrationError(f"invalid source identity path: {relative}")
        path = _require_regular_non_symlink(repository / relative, "migration source")
        entries.append(file_identity(path, relative.as_posix()))
    return {
        "identity_kind": "corrected-mid-tail-gate12-source-v1",
        "bundle_blake3": blake3.blake3(canonical_json(entries)).hexdigest(),
        "files": len(entries),
        "entries": entries,
    }


def rust_migration_command(source: Path, output: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "-p",
        "cascadia-ai",
        "--example",
        "migrate_legacy_mid_v4_weights",
        "--features",
        CARGO_FEATURE,
        "--",
        str(source),
        str(output),
    ]


def run_rust_migration(repository: Path, source: Path, output: Path) -> None:
    if output.exists():
        raise MigrationError(f"refusing to overwrite migration staging output: {output}")
    command = rust_migration_command(source, output)
    environment = {**os.environ, "CARGO_TERM_COLOR": "never"}
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise MigrationError(
            f"Rust migration command exited {completed.returncode}: {detail[-4000:]}"
        )
    _require_regular_non_symlink(output, "Rust migration output")


def run_independent_audit(
    repository: Path,
    source: Path,
    corrected: Path,
    output: Path,
    source_label: str,
    corrected_label: str,
) -> dict[str, Any]:
    audit_tool = repository / "tools/corrected_mid_tail_champion_audit.py"
    command = [
        sys.executable,
        str(audit_tool),
        "--source",
        str(source),
        "--corrected",
        str(corrected),
        "--output",
        str(output),
        "--source-label",
        source_label,
        "--corrected-label",
        corrected_label,
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if not output.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise MigrationError(f"independent audit emitted no report: {detail[-4000:]}")
    try:
        report = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationError(f"independent audit report is unreadable: {error}") from error
    if completed.returncode != 0 or report.get("verdict") != "pass":
        raise MigrationError(
            "independent audit failed: "
            + json.dumps(report.get("failure_reasons", []), sort_keys=True)
        )
    return report


def _command_contract() -> dict[str, Any]:
    return {
        "program": "cargo",
        "argv_template": [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "-p",
            "cascadia-ai",
            "--example",
            "migrate_legacy_mid_v4_weights",
            "--features",
            CARGO_FEATURE,
            "--",
            "<production-champion>",
            "<staging-output>",
        ],
        "cargo_feature": CARGO_FEATURE,
        "rust_example": "migrate_legacy_mid_v4_weights",
    }


def build_manifest(
    *,
    repository: Path,
    source_identity: dict[str, Any],
    model_path: Path,
    model_label: str,
    audit_path: Path,
    audit_report: dict[str, Any],
    source_code_identity: dict[str, Any],
) -> dict[str, Any]:
    model_identity = file_identity(model_path, model_label)
    audit_identity = file_identity(audit_path, f"{model_label.rsplit('/', 1)[0]}/{AUDIT_FILENAME}")
    output_header = audit_report.get("corrected", {})
    contract = audit_report.get("contract", {})
    if audit_report.get("verdict") != "pass":
        raise MigrationError("cannot publish an artifact with a non-passing audit")
    if output_header.get("blake3") != model_identity["blake3"]:
        raise MigrationError("audit output hash does not match the migration output")
    if audit_report.get("source", {}).get("blake3") != source_identity["blake3"]:
        raise MigrationError("audit source hash does not match the production champion")
    relative_directory = Path(model_label).parent.as_posix()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "experiment_id": EXPERIMENT_ID,
        "gate": GATE,
        "verdict": "pass",
        "model_id": f"blake3:{model_identity['blake3']}",
        "content_address": {
            "algorithm": "blake3",
            "digest": model_identity["blake3"],
            "relative_directory": relative_directory,
        },
        "source": {
            **source_identity,
            "logical_name": "production-v4opp-modal-iter3-champion",
            "container_magic": "NNUE",
            "schema_id": HISTORICAL_SCHEMA_ID,
            "feature_count": contract.get("feature_count"),
            "hidden1": contract.get("hidden1"),
            "hidden2": contract.get("hidden2"),
            "head_format_version": contract.get("head_format_version"),
        },
        "output": {
            **model_identity,
            "logical_name": "corrected-mid-tail-v1-untrained-initialization",
            "container_magic": "NNUC",
            "container_version": output_header.get("container_version"),
            "schema_id": SCHEMA_ID,
            "schema_tag_hex": output_header.get("schema_tag_hex"),
            "feature_count": output_header.get("feature_count"),
            "hidden1": output_header.get("hidden1"),
            "hidden2": output_header.get("hidden2"),
            "head_format_version": output_header.get("head_format_version"),
        },
        "migration": {
            "deterministic": True,
            "command": _command_contract(),
            "source_code_identity": source_code_identity,
            "mapping": [
                {
                    "source_range": [0, 10_561],
                    "destination_range": [0, 10_561],
                    "operation": "byte-exact-copy",
                },
                {
                    "source_range": [10_561, 10_862],
                    "destination_range": None,
                    "operation": "discard",
                },
                {
                    "source_range": [10_862, 11_231],
                    "destination_range": [10_561, 10_930],
                    "operation": "byte-exact-remap",
                },
                {
                    "source_range": None,
                    "destination_range": [10_930, 11_231],
                    "operation": "ieee754-signed-zero-initialize",
                },
            ],
        },
        "audit": {
            **audit_identity,
            "audit_id": audit_report.get("audit_id"),
            "verdict": audit_report.get("verdict"),
            "failure_reasons": audit_report.get("failure_reasons"),
            "independent_of_rust_loader": True,
            "all_checks": audit_report.get("checks"),
        },
        "immutability": {
            "content_address_is_output_blake3": True,
            "expected_files": [
                AUDIT_FILENAME,
                MANIFEST_DIGEST_FILENAME,
                MANIFEST_FILENAME,
                MODEL_FILENAME,
            ],
            "files_read_only": True,
            "directory_read_only": True,
        },
        "repository": {
            "root_label": repository.name,
        },
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise MigrationError(f"{label} root must be an object: {path}")
    return value


def _assert_read_only(path: Path, label: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise MigrationError(f"{label} is writable: {path}")


def _validate_identity(path: Path, expected: dict[str, Any], label: str) -> None:
    actual = file_identity(path, str(expected.get("label", path.name)))
    for key in ("bytes", "blake3", "sha256"):
        if actual[key] != expected.get(key):
            raise MigrationError(
                f"{label} {key} mismatch: expected={expected.get(key)} found={actual[key]}"
            )


def validate_artifact(
    *,
    repository: Path,
    artifact_directory: Path,
    source: Path,
    source_identity: dict[str, Any],
    audit_runner: AuditRunner = run_independent_audit,
    require_read_only: bool = True,
) -> dict[str, Any]:
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise MigrationError(f"artifact directory is invalid: {artifact_directory}")
    digest = artifact_directory.name
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MigrationError(f"artifact directory is not a BLAKE3 digest: {artifact_directory}")

    expected_names = {
        AUDIT_FILENAME,
        MANIFEST_DIGEST_FILENAME,
        MANIFEST_FILENAME,
        MODEL_FILENAME,
    }
    actual_names = {path.name for path in artifact_directory.iterdir()}
    if actual_names != expected_names:
        raise MigrationError(
            f"immutable artifact file set mismatch: expected={sorted(expected_names)} "
            f"found={sorted(actual_names)}"
        )

    model_path = _require_regular_non_symlink(
        artifact_directory / MODEL_FILENAME,
        "corrected model",
    )
    audit_path = _require_regular_non_symlink(
        artifact_directory / AUDIT_FILENAME,
        "stored independent audit",
    )
    manifest_path = _require_regular_non_symlink(
        artifact_directory / MANIFEST_FILENAME,
        "artifact manifest",
    )
    manifest_digest_path = _require_regular_non_symlink(
        artifact_directory / MANIFEST_DIGEST_FILENAME,
        "artifact manifest digest",
    )
    expected_manifest_digest = f"{_hash_file(manifest_path, 'blake3')}  {MANIFEST_FILENAME}\n"
    if manifest_digest_path.read_text() != expected_manifest_digest:
        raise MigrationError("artifact manifest digest mismatch")
    manifest = _load_json(manifest_path, "artifact manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise MigrationError("artifact manifest schema version mismatch")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise MigrationError("artifact kind mismatch")
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("gate") != GATE:
        raise MigrationError("artifact experiment or gate mismatch")
    if manifest.get("verdict") != "pass":
        raise MigrationError("artifact manifest verdict is not pass")
    if manifest.get("content_address", {}).get("digest") != digest:
        raise MigrationError("manifest content address does not match its directory")
    if manifest.get("model_id") != f"blake3:{digest}":
        raise MigrationError("manifest model ID does not match its directory")
    if source_identity["blake3"] != manifest.get("source", {}).get("blake3"):
        raise MigrationError("artifact source does not match the current production champion")

    _validate_identity(model_path, manifest.get("output", {}), "corrected model")
    if _hash_file(model_path, "blake3") != digest:
        raise MigrationError("corrected model bytes do not match the content address")
    _validate_identity(audit_path, manifest.get("audit", {}), "stored audit")
    stored_audit = _load_json(audit_path, "stored independent audit")
    if stored_audit.get("verdict") != "pass":
        raise MigrationError("stored independent audit verdict is not pass")
    if stored_audit.get("source", {}).get("blake3") != source_identity["blake3"]:
        raise MigrationError("stored audit source hash mismatch")
    if stored_audit.get("corrected", {}).get("blake3") != digest:
        raise MigrationError("stored audit output hash mismatch")

    if require_read_only:
        _assert_read_only(artifact_directory, "artifact directory")
        for path in (model_path, audit_path, manifest_path, manifest_digest_path):
            _assert_read_only(path, "artifact file")

    source_label = str(manifest.get("source", {}).get("label", ""))
    corrected_label = str(manifest.get("output", {}).get("label", ""))
    with tempfile.TemporaryDirectory(prefix="corrected-mid-tail-reaudit-") as temporary:
        live_audit_path = Path(temporary) / AUDIT_FILENAME
        live_audit = audit_runner(
            repository,
            source,
            model_path,
            live_audit_path,
            source_label,
            corrected_label,
        )
    if canonical_json(live_audit) != canonical_json(stored_audit):
        raise MigrationError("live independent re-audit differs from the immutable audit")
    return manifest


def _chmod_immutable(artifact_directory: Path) -> None:
    for path in artifact_directory.iterdir():
        path.chmod(0o444)
    artifact_directory.chmod(0o555)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for directory, directory_names, file_names in os.walk(path, topdown=False):
        current = Path(directory)
        current.chmod(0o755)
        for name in file_names:
            (current / name).chmod(0o644)
        for name in directory_names:
            (current / name).chmod(0o755)
    shutil.rmtree(path)


@contextmanager
def _migration_lock(repository: Path) -> Iterable[None]:
    lock_path = repository / "target/corrected-mid-tail-v1-champion-migration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _matching_existing_artifacts(
    models_root: Path,
    source_blake3: str,
) -> list[Path]:
    if not models_root.exists():
        return []
    matches = []
    for manifest_path in sorted(models_root.glob("*/manifest.json")):
        manifest = _load_json(manifest_path, "candidate artifact manifest")
        if (
            manifest.get("artifact_kind") == ARTIFACT_KIND
            and manifest.get("source", {}).get("blake3") == source_blake3
        ):
            matches.append(manifest_path.parent)
    return matches


def migrate_champion(
    *,
    repository: Path,
    source: Path,
    output_root: Path,
    expected_source: ExpectedSource,
    migration_runner: MigrationRunner = run_rust_migration,
    audit_runner: AuditRunner = run_independent_audit,
    source_identity_paths: Iterable[str] = SOURCE_IDENTITY_PATHS,
) -> MigrationResult:
    repository = repository.resolve()
    if not repository.is_dir():
        raise MigrationError(f"repository is not a directory: {repository}")
    source, source_identity = validate_source(source, repository, expected_source)
    output_root = output_root.resolve()
    _relative_label(output_root, repository, "artifact output root")
    models_root = output_root / "models/blake3"
    source_code_identity = collect_source_identity(repository, source_identity_paths)

    with _migration_lock(repository):
        matches = _matching_existing_artifacts(models_root, source_identity["blake3"])
        if len(matches) > 1:
            raise MigrationError(
                "multiple immutable artifacts claim the production champion: "
                + ", ".join(str(path) for path in matches)
            )
        if matches:
            manifest = validate_artifact(
                repository=repository,
                artifact_directory=matches[0],
                source=source,
                source_identity=source_identity,
                audit_runner=audit_runner,
            )
            return MigrationResult(matches[0], manifest, True)

        models_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=models_root))
        published: Path | None = None
        try:
            staged_model = staging / MODEL_FILENAME
            migration_runner(repository, source, staged_model)
            staged_model = _require_regular_non_symlink(staged_model, "staged corrected model")
            output_blake3 = _hash_file(staged_model, "blake3")
            final_directory = models_root / output_blake3
            final_model_label = _relative_label(
                final_directory / MODEL_FILENAME,
                repository,
                "corrected model",
            )
            staged_audit = staging / AUDIT_FILENAME
            source_label = str(source_identity["label"])
            audit_report = audit_runner(
                repository,
                source,
                staged_model,
                staged_audit,
                source_label,
                final_model_label,
            )
            if audit_report.get("verdict") != "pass":
                raise MigrationError("independent audit did not return a passing verdict")
            manifest = build_manifest(
                repository=repository,
                source_identity=source_identity,
                model_path=staged_model,
                model_label=final_model_label,
                audit_path=staged_audit,
                audit_report=audit_report,
                source_code_identity=source_code_identity,
            )
            write_json_atomic(staging / MANIFEST_FILENAME, manifest)
            manifest_digest = _hash_file(staging / MANIFEST_FILENAME, "blake3")
            (staging / MANIFEST_DIGEST_FILENAME).write_text(
                f"{manifest_digest}  {MANIFEST_FILENAME}\n"
            )
            if final_directory.exists():
                raise MigrationError(
                    f"content-addressed destination already exists unexpectedly: {final_directory}"
                )
            staging.rename(final_directory)
            published = final_directory
            _chmod_immutable(final_directory)
            final_manifest = validate_artifact(
                repository=repository,
                artifact_directory=final_directory,
                source=source,
                source_identity=source_identity,
                audit_runner=audit_runner,
            )
            return MigrationResult(final_directory, final_manifest, False)
        except Exception:
            _remove_tree(published or staging)
            raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=repository_default,
        help="repository root used for Cargo execution and provenance labels",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=f"production champion path (default: <repository>/{SOURCE_FILENAME})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "experiment artifact root; the immutable model is published under "
            "<output-root>/models/blake3/<digest> "
            "(default: <repository>/artifacts/experiments/corrected-mid-tail-v1)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repository = args.repository.resolve()
    source = (args.source or repository / SOURCE_FILENAME).resolve()
    output_root = (
        args.output_root or repository / "artifacts/experiments/corrected-mid-tail-v1"
    ).resolve()
    expected = ExpectedSource(
        bytes=PRODUCTION_SOURCE_BYTES,
        blake3=PRODUCTION_SOURCE_BLAKE3,
        sha256=PRODUCTION_SOURCE_SHA256,
    )
    try:
        result = migrate_champion(
            repository=repository,
            source=source,
            output_root=output_root,
            expected_source=expected,
        )
    except (MigrationError, OSError) as error:
        print(f"migration failed: {error}", file=sys.stderr)
        return 1
    summary = {
        "artifact_directory": result.artifact_directory.as_posix(),
        "model_id": result.manifest["model_id"],
        "source_blake3": result.manifest["source"]["blake3"],
        "output_blake3": result.manifest["output"]["blake3"],
        "audit_verdict": result.manifest["audit"]["verdict"],
        "reused": result.reused,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
