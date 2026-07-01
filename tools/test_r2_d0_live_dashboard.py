from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from r2_d0 import live_dashboard as subject
from r2_d0.canonical import canonical_json


def facts(*, runtime_state: str = "stopped") -> dict[str, object]:
    return {
        "os_version": "26.5.1",
        "os_build": "25F80",
        "darwin": "25.5.0",
        "runtime_state": runtime_state,
        "runtime_observation": "not running",
        "runtime_receipt_sha256": "a" * 64,
        "storage_receipt_sha256": "b" * 64,
        "storage_free_bytes": 144 * 1024**3,
        "storage_campaign_bytes": 0,
        "evidence": {
            "john2-runtime-profile-receipt.json": True,
            "john2-cold-archive-root-receipt.json": True,
            "runtime-profile-receipt.json": True,
            "john1-reopen-receipt.json": True,
            "legacy-cleanup-receipt.json": True,
        },
    }


def test_red_status_is_truthful_closed_and_names_only_active_hosts() -> None:
    status = subject.build_red_status(facts(), updated_unix_ms=1_781_840_000_000)
    assert status["phase"] == "d0-blocked"
    assert status["legal_next_transitions"] == []
    assert status["hosts"]["john1"]["intent"] == "control"
    assert "runtime=stopped" in status["hosts"]["john1"]["detail"]
    assert "active_storage=john1-internal-apfs" in status["hosts"]["john1"]["detail"]
    assert status["hosts"]["john2"]["intent"] == "idle"
    assert status["hosts"]["john3"]["intent"] == "idle"
    assert "legacy_native_workspace=absent" in status["hosts"]["john3"]["detail"]
    assert "archive_commit=verified" in status["hosts"]["john3"]["detail"]
    assert "source_cleanup=verified" in status["hosts"]["john3"]["detail"]
    assert "role_qualification=pass" in status["hosts"]["john2"]["detail"]
    assert "role_qualification=pass" in status["hosts"]["john3"]["detail"]
    assert set(status["hosts"]) == {"john1", "john2", "john3"}
    assert status["training"]["active"] is False
    assert status["benchmark"]["active"] is False
    assert status["models"] == {"incumbent": None, "candidate": None, "opponent_pool": []}
    assert "john4" not in json.dumps(status).lower()


def test_red_status_refuses_unknown_runtime_state() -> None:
    with pytest.raises(subject.LiveDashboardError, match="runtime state"):
        subject.build_red_status(facts(runtime_state="unknown"), updated_unix_ms=1)


def test_local_observer_uses_the_isolated_colima_and_docker_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(argv: list[str], *, extra_env=None):
        calls.append((argv, dict(extra_env or {})))
        if argv[0].endswith("colima"):
            return 0, "running"
        return 0, "25F80" if "-buildVersion" in argv else "26.5.1"

    monkeypatch.setattr(subject, "_run", run)
    monkeypatch.setattr(subject, "_secure_file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(subject, "_john3_role_receipt_valid", lambda: True)
    monkeypatch.setattr(
        subject,
        "verify_canonical_storage",
        lambda **_kwargs: {
            "receipt_sha256": "b" * 64,
            "free_bytes": 100,
            "campaign_apparent_bytes": 10,
        },
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_EVIDENCE_SHA256",
        {Path("/missing"): "c" * 64},
    )

    observed = subject.collect_local_facts()

    assert observed["runtime_state"] == "running"
    colima = next(call for call in calls if call[0][0].endswith("colima"))
    assert colima[1] == {
        "COLIMA_HOME": "/Users/johnherrick/.local/share/cascadia-r2/colima",
        "DOCKER_CONFIG": "/Users/johnherrick/.config/cascadia-r2/docker",
    }


def test_john3_v4_role_receipt_requires_cleanup_lineage_and_pending_aggregate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime-profile-receipt.json"
    value = {
        "schema_id": "cascadia.r2-map.local-runtime-profile-receipt.v4",
        "schema_version": 4,
        "host": "john3",
        "role": "execution-only",
        "legacy_native_workspace": "absent",
        "certification": {
            "host_role_qualified": True,
            "d0_certified": False,
            "project_execution_authorized": False,
            "blocker": "Signed D0 topology aggregate is pending.",
        },
        "lineage": {
            "cleanup": {
                "status": "pass",
                "completion_receipt": {
                    "sha256": subject.EXPECTED_EVIDENCE_SHA256[
                        subject.JOHN3_CLEANUP_RECEIPT
                    ]
                },
            }
        },
    }
    path.write_bytes(canonical_json(value) + b"\n")
    path.chmod(0o600)
    assert subject._john3_role_receipt_valid(path)
    value["lineage"]["cleanup"]["completion_receipt"]["sha256"] = "0" * 64
    path.write_bytes(canonical_json(value) + b"\n")
    path.chmod(0o600)
    assert not subject._john3_role_receipt_valid(path)


def test_publish_is_atomic_owner_private_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    control = root / "control"
    control.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(subject, "CANONICAL_ROOT", root)
    monkeypatch.setattr(subject, "LOCK_PATH", control / ".dashboard-status.heartbeat.lock")
    path = control / "dashboard-status.json"
    status = subject.build_red_status(facts(), updated_unix_ms=1_781_840_000_000)

    result = subject.publish_status(
        status,
        path=path,
        storage_verifier=lambda **_kwargs: {"status": "pass"},
    )

    assert result["bytes"] == path.stat().st_size
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_bytes()) == status
    assert not list(control.glob(".dashboard-status.*.tmp"))


def test_publish_rejects_noncanonical_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "r2-map-v1"
    (root / "control").mkdir(parents=True)
    monkeypatch.setattr(subject, "CANONICAL_ROOT", root)
    status = subject.build_red_status(facts(), updated_unix_ms=1)
    with pytest.raises(subject.LiveDashboardError, match="storage boundary"):
        subject.publish_status(
            status,
            path=tmp_path / "elsewhere.json",
            storage_verifier=lambda **_kwargs: {"status": "pass"},
        )
