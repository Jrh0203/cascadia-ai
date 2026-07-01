#!/usr/bin/env python3
"""Run the isolated W6 controller proof without mutating canonical state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import r2_map_w0_w5_source_manifest as source_manifest
from r2_map_w0_w5_validate import load_and_verify_source

SCHEMA_ID = "cascadia.r2-map.w6-isolated-validation-summary.v1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
CANONICAL_ROOT = Path("/Users/john2/cascadia-bench/r2-map-v1")
ISOLATED_NAME = "w6-isolated-controller-v1"
SENSITIVE_RELATIVES = (
    "control/campaign-state.json",
    "control/decision-log.jsonl",
    "control/research-queue-v1.json",
    "control/research-experiments-v1.json",
    "control/controller-history.jsonl",
    "control/controller-stop.json",
    "control/headless-terminal.json",
    "control/headless-STOP",
    "control/work-packets",
    "control/incoming-receipts",
    "control/contracts/r2-map-work-packet-v2.schema.json",
    "control/contracts/r2-map-work-receipt-v2.schema.json",
    "control/dashboard-inputs/controller.json",
    "control/dashboard-inputs/training-progress.json",
    "control/dashboard-inputs/benchmark-aggregate.json",
    "control/dashboard-inputs/model-manifest.json",
    "control/dashboard-inputs/pool-manifest.json",
    "control/dashboard-inputs/host-receipts.json",
)


class W6ValidationError(RuntimeError):
    """The isolated proof, canonical boundary, or cleanup is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(value: dict[str, Any]) -> str:
    content = dict(value)
    content.pop("summary_sha256", None)
    return hashlib.sha256(canonical_json(content)).hexdigest()


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        current_details = current_path.lstat()
        if current_path.is_symlink() or not stat.S_ISDIR(current_details.st_mode):
            raise W6ValidationError("validated tree contains an unsafe directory")
        for name in sorted(directory_names):
            child = current_path / name
            details = child.lstat()
            if child.is_symlink() or not stat.S_ISDIR(details.st_mode):
                raise W6ValidationError("validated tree contains an unsafe directory entry")
        for name in sorted(file_names):
            path = current_path / name
            details = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(details.st_mode):
                raise W6ValidationError("validated tree contains a non-regular file")
            entries.append(
                {
                    "relative": path.relative_to(root).as_posix(),
                    "size": details.st_size,
                    "mode": f"{stat.S_IMODE(details.st_mode):04o}",
                    "sha256": sha256_file(path),
                }
            )
    return sorted(entries, key=lambda entry: entry["relative"])


def snapshot_sensitive(root: Path = CANONICAL_ROOT) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for relative in SENSITIVE_RELATIVES:
        path = root / relative
        try:
            details = path.lstat()
        except FileNotFoundError:
            snapshot[relative] = {"kind": "absent"}
            continue
        if path.is_symlink():
            raise W6ValidationError(f"canonical sensitive path is a symlink: {relative}")
        if stat.S_ISREG(details.st_mode):
            snapshot[relative] = {
                "kind": "file",
                "size": details.st_size,
                "mode": f"{stat.S_IMODE(details.st_mode):04o}",
                "sha256": sha256_file(path),
            }
        elif stat.S_ISDIR(details.st_mode):
            entries = _tree_entries(path)
            snapshot[relative] = {
                "kind": "directory",
                "entry_count": len(entries),
                "tree_sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
            }
        else:
            raise W6ValidationError(f"canonical sensitive path is special: {relative}")
    return snapshot


def validate_dry_run_report(report: dict[str, Any], isolated: Path) -> None:
    expected = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.w6-isolated-dry-run.v1",
        "campaign_id": CAMPAIGN_ID,
        "root": str(isolated),
        "transition_count": 21,
        "history_count": 21,
        "final_phase": "incumbent-promoted",
        "final_promotion_index": 1,
        "final_round_index": 1,
        "queue_task_count": 30,
        "completed_queue_tasks": 30,
        "work_receipt_count": 30,
        "storage_receipt_count": 30,
        "receipt_count": 60,
        "ledger_experiment_count": 1,
        "stop_file_present": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise W6ValidationError(f"W6 dry-run report differs at {key}")
    observed_hash = report.get("dry_run_sha256")
    payload = dict(report)
    payload.pop("dry_run_sha256", None)
    if observed_hash != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise W6ValidationError("W6 dry-run report hash differs")
    if not isinstance(report.get("final_state_sha256"), str) or len(
        report["final_state_sha256"]
    ) != 64:
        raise W6ValidationError("W6 dry-run final state identity is invalid")


def run_command(
    command_id: str,
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    capture: bool = False,
    timeout_seconds: int = 600,
) -> tuple[dict[str, Any], str, str]:
    print("W6_COMMAND", json.dumps({"id": command_id, "argv": argv}), flush=True)
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=capture,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise W6ValidationError(f"W6 command timed out: {command_id}") from error
    stdout = completed.stdout if capture else ""
    stderr = completed.stderr if capture else ""
    result = {
        "id": command_id,
        "argv": argv,
        "returncode": completed.returncode,
        "captured_stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "captured_stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }
    print("W6_RESULT", json.dumps(result, sort_keys=True), flush=True)
    if completed.returncode != 0:
        if capture:
            print(stdout, end="", file=sys.stdout)
            print(stderr, end="", file=sys.stderr)
        raise W6ValidationError(f"W6 command failed: {command_id}")
    return result, stdout, stderr


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _remove_tree(path: Path) -> None:
    if not _entry_exists(path):
        return
    details = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(details.st_mode):
        raise W6ValidationError("W6 cleanup root is not a regular directory")
    # rmtree unlinks read-only files through their writable parent directories
    # and never follows directory symlinks.
    shutil.rmtree(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(CANONICAL_ROOT / "source"):
        raise W6ValidationError("W6 validator is not running from immutable John2 source")
    source = load_and_verify_source(root)
    if source != source_manifest.build_manifest(root):
        raise W6ValidationError("W6 source selection differs")

    temporary = Path(os.environ["TMPDIR"]).resolve(strict=True)
    isolated = temporary / ISOLATED_NAME
    pytest_temp = temporary / "w6-controller-pytest"
    if _entry_exists(isolated) or _entry_exists(pytest_temp) or temporary == CANONICAL_ROOT:
        raise W6ValidationError("W6 isolated path is unsafe or already exists")
    if isolated == CANONICAL_ROOT or CANONICAL_ROOT not in isolated.parents:
        raise W6ValidationError("W6 isolated path is outside John2 canonical storage")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(root / "python"), str(root / "tools"))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
        }
    )
    before = snapshot_sensitive()
    results: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None
    tree_entries: list[dict[str, Any]] = []
    try:
        result, _, _ = run_command(
            "w6-controller-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "python/tests/test_r2_map_campaign_controller.py",
                "tools/test_r2_map_w6_validate.py",
                "--basetemp",
                str(pytest_temp),
            ],
            cwd=root,
            environment=environment,
        )
        results.append(result)
        result, _, _ = run_command(
            "w6-ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--no-cache",
                "python/cascadia_mlx/r2_map_campaign_controller.py",
                "python/cascadia_mlx/r2_map_contracts.py",
                "python/cascadia_mlx/r2_map_dashboard_status.py",
                "python/tests/test_r2_map_campaign_controller.py",
                "tools/cluster_experiment_ledger.py",
                "tools/cluster_research_queue.py",
                "tools/r2_map_expert_iteration.py",
                "tools/r2_map_w6_validate.py",
                "tools/test_r2_map_w6_validate.py",
            ],
            cwd=root,
            environment=environment,
        )
        results.append(result)
        result, stdout, stderr = run_command(
            "w6-isolated-dry-run",
            [
                sys.executable,
                "tools/r2_map_expert_iteration.py",
                "w6-dry-run",
                "--campaign-root",
                str(isolated),
            ],
            cwd=root,
            environment=environment,
            capture=True,
        )
        results.append(result)
        if stderr:
            raise W6ValidationError("W6 isolated dry run emitted stderr")
        report = json.loads(stdout)
        validate_dry_run_report(report, isolated)
        disk_report = json.loads((isolated / "dry-run-report.json").read_text(encoding="ascii"))
        if disk_report != report:
            raise W6ValidationError("W6 stdout and immutable report differ")
        history = (isolated / "control/controller-history.jsonl").read_text(encoding="ascii")
        if '"classification":"reject"' not in history or history.count(
            '"classification":"promote"'
        ) != 2:
            raise W6ValidationError("W6 history omitted rejection or promotion evidence")
        tree_entries = _tree_entries(isolated)
    finally:
        _remove_tree(isolated)
        _remove_tree(pytest_temp)

    after = snapshot_sensitive()
    if before != after:
        raise W6ValidationError("canonical controller state changed during isolated W6 proof")
    if _entry_exists(isolated) or _entry_exists(pytest_temp) or report is None:
        raise W6ValidationError("W6 isolated validation cleanup is incomplete")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "source_document_sha256": source["document_sha256"],
        "protected_seed_values_opened": False,
        "canonical_snapshot_sha256": hashlib.sha256(canonical_json(before)).hexdigest(),
        "canonical_state_unchanged": True,
        "canonical_initialization_performed": False,
        "isolated_root": str(isolated),
        "isolated_tree_entry_count": len(tree_entries),
        "isolated_tree_sha256": hashlib.sha256(canonical_json(tree_entries)).hexdigest(),
        "isolated_root_removed": True,
        "pytest_root_removed": True,
        "results": results,
        "dry_run_report": report,
        "john4_used": False,
    }
    summary["summary_sha256"] = document_sha256(summary)
    print("W6_VALIDATION_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        json.JSONDecodeError,
        source_manifest.SourceManifestError,
        W6ValidationError,
    ) as error:
        print(f"R2-MAP W6 validation refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
