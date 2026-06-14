"""Shared provenance and exact-resume validation for MLX training runs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import blake3


class RunManifestError(ValueError):
    """Raised when a run cannot be resumed exactly."""


def validate_resume_manifest(
    run_dir: Path,
    *,
    training: dict[str, Any],
    datasets: dict[str, Any],
    runtime: dict[str, Any],
    source: dict[str, Any],
) -> None:
    """Reject any resume that changes data, optimization, runtime, or source."""
    try:
        existing = json.loads((run_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RunManifestError(f"cannot read run manifest: {error}") from error
    if existing.get("schema_version") != 1:
        raise RunManifestError("unsupported run manifest schema")

    existing_training = dict(existing.get("training", {}))
    expected_training = dict(training)
    for mutable in ("epochs", "resume"):
        existing_training.pop(mutable, None)
        expected_training.pop(mutable, None)
    _require_equal("training configuration", existing_training, expected_training)
    _require_equal("dataset identity", existing.get("datasets"), datasets)
    _require_equal("runtime environment", existing.get("runtime"), runtime)

    existing_source = existing.get("source", {})
    existing_digest = existing_source.get("v2_source_blake3")
    expected_digest = source.get("v2_source_blake3")
    if not existing_digest or existing_digest != expected_digest:
        raise RunManifestError("v2 source digest changed; start a new run instead of resuming")


def source_provenance(repository: Path) -> dict[str, Any]:
    """Capture Git identity and a content digest of every v2 source boundary."""
    repository = repository.resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
        dirty = True

    roots = [
        repository / "Cargo.toml",
        repository / "Cargo.lock",
        repository / "Makefile",
        repository / "pyproject.toml",
        repository / "uv.lock",
        repository / "python" / "cascadia_mlx",
        repository / "apps" / "web" / "src",
        repository / "crates" / "cascadia-game",
        repository / "crates" / "cascadia-sim",
        repository / "crates" / "cascadia-data",
        repository / "crates" / "cascadia-model",
        repository / "crates" / "cascadia-eval",
        repository / "crates" / "cascadia-search",
        repository / "crates" / "cascadia-api",
        repository / "crates" / "cascadia-cli-v2",
        repository / "crates" / "cascadia-differential",
        repository / "crates" / "cascadia-provenance",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )

    digest = blake3.blake3()
    for path in sorted(files):
        relative = path.relative_to(repository).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "v2_source_blake3": digest.hexdigest(),
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RunManifestError(f"resume {label} does not match the original run")
