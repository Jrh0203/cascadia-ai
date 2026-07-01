from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import r2_map_w6_validate as validate


def test_validate_dry_run_report_requires_exact_registered_counts(tmp_path: Path) -> None:
    isolated = tmp_path / validate.ISOLATED_NAME
    report = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.w6-isolated-dry-run.v1",
        "campaign_id": validate.CAMPAIGN_ID,
        "root": str(isolated),
        "transition_count": 21,
        "history_count": 21,
        "final_phase": "incumbent-promoted",
        "final_promotion_index": 1,
        "final_round_index": 1,
        "final_state_sha256": "a" * 64,
        "queue_task_count": 30,
        "completed_queue_tasks": 30,
        "work_receipt_count": 30,
        "storage_receipt_count": 30,
        "receipt_count": 60,
        "ledger_experiment_count": 1,
        "stop_file_present": False,
    }
    report["dry_run_sha256"] = hashlib.sha256(validate.canonical_json(report)).hexdigest()
    validate.validate_dry_run_report(report, isolated)
    report["queue_task_count"] = 29
    with pytest.raises(validate.W6ValidationError, match="queue_task_count"):
        validate.validate_dry_run_report(report, isolated)


def test_sensitive_snapshot_is_content_bound_and_rejects_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate, "SENSITIVE_RELATIVES", ("control/state.json",))
    state = tmp_path / "control/state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"revision": 0}), encoding="ascii")
    first = validate.snapshot_sensitive(tmp_path)
    assert first["control/state.json"]["kind"] == "file"
    state.write_text(json.dumps({"revision": 1}), encoding="ascii")
    assert validate.snapshot_sensitive(tmp_path) != first
    state.unlink()
    state.symlink_to(tmp_path / "target")
    with pytest.raises(validate.W6ValidationError, match="symlink"):
        validate.snapshot_sensitive(tmp_path)


def test_tree_entries_reject_special_or_linked_entries(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "file").write_bytes(b"evidence")
    entries = validate._tree_entries(root)
    assert entries[0]["sha256"] == hashlib.sha256(b"evidence").hexdigest()
    (root / "link").symlink_to(root / "file")
    with pytest.raises(validate.W6ValidationError, match="non-regular"):
        validate._tree_entries(root)


def test_remove_tree_handles_read_only_files_and_rejects_dangling_root_link(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    payload = root / "immutable.json"
    payload.write_text("evidence\n", encoding="ascii")
    payload.chmod(0o400)
    validate._remove_tree(root)
    assert not validate._entry_exists(root)

    root.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(validate.W6ValidationError, match="cleanup root"):
        validate._remove_tree(root)
    assert root.is_symlink()
