#!/usr/bin/env python3
"""Validate one immutable W0 v1.1 and W5 source transaction on John2."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import r2_map_w0_w5_source_manifest as source_manifest

SCHEMA_ID = "cascadia.r2-map.w0-w5-source-manifest.v1"
SUMMARY_SCHEMA_ID = "cascadia.r2-map.w0-w5-validation-summary.v1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
CANONICAL_ROOT = Path("/Users/john2/cascadia-bench/r2-map-v1")
OPEN_DOMAIN = "r2-map-open-reference-performance-100-v1"
REGISTRATION = CANONICAL_ROOT / "control/w0-preregistration/registration-v1.1.json"
REFERENCE_MANIFEST = (
    CANONICAL_ROOT
    / "control/w0-preregistration/reference-panel-manifest-v1.1.json"
)


class ValidationError(RuntimeError):
    """The frozen source, registration, or validation command is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(value: dict[str, Any], field: str) -> str:
    content = dict(value)
    content.pop(field, None)
    return sha256_bytes(canonical_json(content))


def load_and_verify_source(root: Path) -> dict[str, Any]:
    manifest_path = root / "source-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read source manifest: {error}") from error
    if (
        manifest.get("schema_id") != SCHEMA_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("protected_seed_values_opened") is not False
        or manifest.get("open_seed_domain") != OPEN_DOMAIN
        or manifest.get("open_seed_count") != 100
        or manifest.get("document_sha256")
        != document_sha256(manifest, "document_sha256")
    ):
        raise ValidationError("source manifest identity or seed boundary differs")
    try:
        expected_manifest = source_manifest.build_manifest(root)
    except source_manifest.SourceManifestError as error:
        raise ValidationError(f"deterministic source selection failed: {error}") from error
    if manifest != expected_manifest:
        raise ValidationError("source manifest differs from deterministic selection")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("source manifest has no files")
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "relative",
            "size",
            "sha256",
            "mode",
        }:
            raise ValidationError("source manifest file entry has the wrong shape")
        relative = entry["relative"]
        candidate = Path(relative)
        if (
            not isinstance(relative, str)
            or not relative
            or candidate.is_absolute()
            or candidate.as_posix() != relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or relative in expected
        ):
            raise ValidationError("source manifest contains an unsafe or duplicate path")
        if entry["mode"] not in {"0400", "0500"}:
            raise ValidationError("source manifest contains an invalid mode")
        expected[relative] = entry

    observed: set[str] = set()
    ignored = {"source-manifest.json", ".r2-map-transaction.json"}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        details = current_path.lstat()
        if not stat.S_ISDIR(details.st_mode) or current_path.is_symlink():
            raise ValidationError("source transaction contains an unsafe directory")
        for name in directories:
            child = current_path / name
            details = child.lstat()
            if not stat.S_ISDIR(details.st_mode) or child.is_symlink():
                raise ValidationError("source transaction contains an unsafe directory entry")
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative in ignored:
                continue
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or path.is_symlink():
                raise ValidationError("source transaction contains a non-regular file")
            entry = expected.get(relative)
            if entry is None:
                raise ValidationError(f"unregistered source file: {relative}")
            if (
                details.st_size != entry["size"]
                or sha256_file(path) != entry["sha256"]
                or f"{stat.S_IMODE(details.st_mode):04o}" != entry["mode"]
            ):
                raise ValidationError(f"source file identity differs: {relative}")
            observed.add(relative)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise ValidationError(f"source transaction omits registered files: {missing[:3]}")
    return manifest


def make_writable_tree(path: Path) -> None:
    try:
        root_details = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or not stat.S_ISDIR(root_details.st_mode):
        raise ValidationError("cleanup root is not a regular directory")
    for current, directories, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        current_details = current_path.lstat()
        if current_path.is_symlink() or not stat.S_ISDIR(current_details.st_mode):
            raise ValidationError("cleanup tree contains an unsafe directory")
        os.chmod(current_path, 0o700)
        retained_directories: list[str] = []
        for name in directories:
            child = current_path / name
            details = child.lstat()
            if child.is_symlink():
                continue
            if not stat.S_ISDIR(details.st_mode):
                raise ValidationError("cleanup tree contains a non-directory entry")
            os.chmod(child, 0o700)
            retained_directories.append(name)
        directories[:] = retained_directories
        for name in files:
            child = current_path / name
            details = child.lstat()
            if child.is_symlink():
                continue
            if stat.S_ISREG(details.st_mode):
                os.chmod(child, 0o600)


def path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def remove_tree(path: Path) -> None:
    if not path_entry_exists(path):
        return
    make_writable_tree(path)
    shutil.rmtree(path)


def run_command(
    command_id: str,
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    print("VALIDATION_COMMAND", json.dumps({"id": command_id, "argv": argv}), flush=True)
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    maximum_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = {
        "id": command_id,
        "argv": argv,
        "returncode": completed.returncode,
        "maximum_child_rss": maximum_rss,
    }
    print("VALIDATION_RESULT", json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(CANONICAL_ROOT / "source"):
        raise ValidationError("validator is not running from immutable John2 source")
    manifest = load_and_verify_source(root)
    if not REGISTRATION.is_file() or not REFERENCE_MANIFEST.is_file():
        raise ValidationError("canonical W0 v1.1 registration is absent")

    temporary = Path(os.environ["TMPDIR"])
    pytest_temp = temporary / "w0-w5-pytest"
    web_workspace = temporary / "w0-w5-web"
    if path_entry_exists(pytest_temp) or path_entry_exists(web_workspace):
        raise ValidationError("validation temporary path already exists")

    python = sys.executable
    cargo = shutil.which("cargo")
    npm = shutil.which("npm")
    if cargo is None or npm is None:
        raise ValidationError("John2 validation toolchain is incomplete")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(root / "python"), str(root / "tools"))),
            "CARGO_INCREMENTAL": "0",
            "CARGO_TERM_COLOR": "never",
            "NO_COLOR": "1",
            "NPM_CONFIG_AUDIT": "false",
            "NPM_CONFIG_FUND": "false",
            "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        }
    )
    results: list[dict[str, Any]] = []
    try:
        shutil.copytree(root / "apps/web", web_workspace, symlinks=False)
        make_writable_tree(web_workspace)
        commands = [
            (
                "w0-manifest-verify",
                [
                    python,
                    "tools/r2_map_reference_panels.py",
                    "--revision",
                    "v1.1",
                    "verify",
                    "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json",
                ],
                root,
            ),
            (
                "w0-registration-verify",
                [
                    python,
                    "tools/r2_map_reference_panels.py",
                    "--revision",
                    "v1.1",
                    "verify-registration",
                    str(REGISTRATION),
                ],
                root,
            ),
            (
                "w0-python-tests",
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tools/test_r2_map_reference_panels.py",
                    "tools/test_r2_map_paired_gate_power.py",
                    "tools/test_r2_map_w0_w5_source_manifest.py",
                    "tools/test_r2_map_w0_w5_validate.py",
                    "python/tests/test_d6_contract.py",
                    "python/tests/test_r2_map_dataset.py",
                    "python/tests/test_r2_map_market_decision.py",
                    "python/tests/test_r2_map_model.py",
                    "python/tests/test_r2_map_checkpoint_train.py",
                    "python/tests/test_r2_map_remote_training.py",
                    "python/tests/test_r2_map_serve.py",
                    "tools/test_r2_map_john1_packing_sweep.py",
                    "--basetemp",
                    str(pytest_temp / "w0"),
                ],
                root,
            ),
            (
                "w5-python-tests",
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "python/tests/test_r2_map_dashboard_status.py",
                    "python/tests/test_r2_map_campaign_controller.py",
                    "tools/test_r2_map_dashboard_fetch.py",
                    "tools/test_r2_map_w5_dashboard_live.py",
                    "--basetemp",
                    str(pytest_temp / "w5"),
                ],
                root,
            ),
            (
                "w0-w5-ruff",
                [
                    python,
                    "-m",
                    "ruff",
                    "check",
                    "--no-cache",
                    "tools/r2_map_reference_panels.py",
                    "tools/r2_map_paired_gate_power.py",
                    "tools/test_r2_map_reference_panels.py",
                    "tools/test_r2_map_paired_gate_power.py",
                    "tools/r2_map_w0_w5_validate.py",
                    "tools/r2_map_w0_w5_source_manifest.py",
                    "tools/test_r2_map_w0_w5_source_manifest.py",
                    "tools/test_r2_map_w0_w5_validate.py",
                    "tools/r2_map_w5_dashboard_live.py",
                    "tools/test_r2_map_w5_dashboard_live.py",
                    "tools/r2_map_dashboard_fetch.py",
                    "tools/test_r2_map_dashboard_fetch.py",
                    "python/cascadia_mlx/checkpoint.py",
                    "python/cascadia_mlx/r2_map_dataset.py",
                    "python/cascadia_mlx/r2_map_market_decision.py",
                    "python/cascadia_mlx/r2_map_model.py",
                    "python/cascadia_mlx/r2_map_protocol_fixture.py",
                    "python/cascadia_mlx/r2_map_remote_training.py",
                    "python/cascadia_mlx/r2_map_serve.py",
                    "python/cascadia_mlx/r2_map_train.py",
                    "python/cascadia_mlx/r2_map_training_contract.py",
                    "python/cascadia_mlx/r2_map_verify.py",
                    "python/tests/test_d6_contract.py",
                    "python/tests/test_r2_map_checkpoint_train.py",
                    "python/tests/test_r2_map_dataset.py",
                    "python/tests/test_r2_map_market_decision.py",
                    "python/tests/test_r2_map_model.py",
                    "python/tests/test_r2_map_remote_training.py",
                    "python/tests/test_r2_map_serve.py",
                    "tools/r2_map_john1_train.py",
                    "tools/r2_map_john1_packing_sweep.py",
                    "tools/r2_map_market_protocol_fixture.py",
                    "tools/test_r2_map_john1_packing_sweep.py",
                    "python/cascadia_mlx/r2_map_dashboard_status.py",
                    "python/cascadia_mlx/r2_map_campaign_controller.py",
                    "python/tests/test_r2_map_dashboard_status.py",
                    "python/tests/test_r2_map_campaign_controller.py",
                ],
                root,
            ),
            (
                "w0-rust-market-tests",
                [
                    cargo,
                    "test",
                    "--locked",
                    "-p",
                    "cascadia-game",
                    "-p",
                    "cascadia-r2",
                    "-p",
                    "cascadia-model",
                    "-p",
                    "cascadia-search",
                    "-p",
                    "cascadia-data",
                    "--lib",
                ],
                root,
            ),
            (
                "w0-w5-rustfmt",
                [cargo, "fmt", "--all", "--", "--check"],
                root,
            ),
            (
                "w5-eval-tests",
                [cargo, "test", "--locked", "-p", "cascadia-eval", "--lib"],
                root,
            ),
            (
                "w5-api-tests",
                [
                    cargo,
                    "test",
                    "--locked",
                    "-p",
                    "cascadia-api",
                    "cluster_r2_map",
                    "--lib",
                ],
                root,
            ),
            (
                "w0-w5-cli-tests",
                [
                    cargo,
                    "test",
                    "--locked",
                    "-p",
                    "cascadia-cli-v2",
                    "r2_map_commands",
                ],
                root,
            ),
            (
                "w0-w5-clippy",
                [
                    cargo,
                    "clippy",
                    "--locked",
                    "-p",
                    "cascadia-game",
                    "-p",
                    "cascadia-r2",
                    "-p",
                    "cascadia-model",
                    "-p",
                    "cascadia-search",
                    "-p",
                    "cascadia-data",
                    "-p",
                    "cascadia-eval",
                    "-p",
                    "cascadia-api",
                    "-p",
                    "cascadia-cli-v2",
                    "--no-deps",
                    "--",
                    "-D",
                    "warnings",
                ],
                root,
            ),
            ("w5-web-install", [npm, "ci", "--ignore-scripts"], web_workspace),
            (
                "w5-web-tests",
                [
                    npm,
                    "test",
                    "--",
                    "src/R2MapCampaignPanel.test.tsx",
                    "src/cluster.test.ts",
                ],
                web_workspace,
            ),
            ("w5-web-build", [npm, "run", "build"], web_workspace),
            ("w5-web-lint", [npm, "run", "lint"], web_workspace),
        ]
        for command_id, argv, cwd in commands:
            result = run_command(command_id, argv, cwd=cwd, environment=environment)
            results.append(result)
            if result["returncode"] != 0:
                break
    finally:
        remove_tree(pytest_temp)
        remove_tree(web_workspace)

    summary = {
        "schema_id": SUMMARY_SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "source_document_sha256": manifest["document_sha256"],
        "source_file_count": len(manifest["files"]),
        "open_seed_domain": manifest["open_seed_domain"],
        "open_seed_count": manifest["open_seed_count"],
        "protected_seed_values_opened": False,
        "results": results,
        "pytest_temp_removed": not path_entry_exists(pytest_temp),
        "web_workspace_removed": not path_entry_exists(web_workspace),
        "registration_sha256": sha256_file(REGISTRATION),
        "reference_manifest_sha256": sha256_file(REFERENCE_MANIFEST),
    }
    summary["summary_sha256"] = document_sha256(summary, "summary_sha256")
    print("VALIDATION_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    return 0 if (
        results
        and len(results) == 15
        and all(result["returncode"] == 0 for result in results)
        and summary["pytest_temp_removed"]
        and summary["web_workspace_removed"]
    ) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValidationError, ValueError) as error:
        print(f"R2-MAP W0/W5 validation refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
