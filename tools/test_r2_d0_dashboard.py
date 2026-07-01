from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import r2_d0.dashboard as dashboard_module
from r2_d0.canonical import CAMPAIGN_ID, D0_RUN_ID, D0Error
from r2_d0.dashboard import (
    SPEC_SCHEMA,
    build_dashboard_diagnostic,
    publish_dashboard_diagnostic,
    render_diagnostic_spec,
    validate_diagnostic_spec,
)


def _spec(
    *,
    d0_gate: str = "red",
    w0_gate: str = "red",
    blockers: list[str] | None = None,
    expected: str = "absent",
    updated: int | None = None,
) -> dict[str, object]:
    value = {
        "schema_id": SPEC_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "run_id": D0_RUN_ID,
        "expected_current_sha256": expected,
        "updated_unix_ms": time.time_ns() // 1_000_000 if updated is None else updated,
        "stale_after_seconds": 3600,
        "gate_state": {
            "d0_gate": d0_gate,
            "w0_gate": w0_gate,
            "d0_state_sha256": "1" * 64,
            "d0_report_sha256": "2" * 64,
            "blocker_codes": blockers
            if blockers is not None
            else ([] if (d0_gate, w0_gate) == ("green", "green") else ["d0-not-green"]),
            "host_gates": {
                host: {
                    "status": d0_gate,
                    "state_sha256": str(index) * 64,
                    "evidence_sha256": str(index + 3) * 64,
                    "blocker_codes": [] if d0_gate == "green" else [f"{host}-not-green"],
                }
                for index, host in enumerate(("john1", "john2", "john3"), start=1)
            },
        },
    }
    return json.loads(render_diagnostic_spec(value))


def _storage() -> dict[str, object]:
    return {"status": "pass", "host_identity_sha256": "3" * 64}


def test_blocked_diagnostic_has_no_project_bootstrap_and_excludes_john4() -> None:
    specification = _spec()
    status = build_dashboard_diagnostic(specification)
    assert status["phase"] == "d0-blocked"
    assert status["legal_next_transitions"] == []
    assert set(status["hosts"]) == {"john1", "john2", "john3"}
    assert all(status["hosts"][host]["intent"] == "idle" for host in status["hosts"])
    assert all("d0-runtime:red" in status["hosts"][host]["detail"] for host in status["hosts"])


def test_contracts_ready_requires_both_green_gates_and_no_blockers() -> None:
    status = build_dashboard_diagnostic(_spec(d0_gate="green", w0_gate="green"))
    assert status["phase"] == "contracts-ready"
    assert status["legal_next_transitions"] == ["bootstrap-generating"]
    with pytest.raises(D0Error, match="gate state"):
        _spec(d0_gate="green", w0_gate="red", blockers=[])
    with pytest.raises(D0Error, match="gate state"):
        _spec(d0_gate="green", w0_gate="green", blockers=["false-blocker"])


def test_spec_hash_and_state_report_digests_are_tamper_evident() -> None:
    specification = _spec()
    changed = json.loads(json.dumps(specification))
    changed["gate_state"]["d0_report_sha256"] = "4" * 64
    with pytest.raises(D0Error, match="spec identity"):
        validate_diagnostic_spec(changed)


def test_dashboard_publish_is_atomic_cas_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "r2-map-v1"
    specification = _spec()
    commits: list[Path] = []
    monkeypatch.setattr(
        dashboard_module,
        "verify_canonical_commit_boundary",
        lambda path: commits.append(path) or {"status": "pass"},
    )
    control = root / "control"
    control.mkdir(parents=True, mode=0o700)
    partial = control / f".dashboard-status.{specification['spec_sha256']}.partial"
    partial.write_bytes(b"truncated-after-crash")
    partial.chmod(0o600)
    receipt = publish_dashboard_diagnostic(
        specification,
        campaign_root=root,
        storage_verifier=_storage,
        stability_seconds=0,
    )
    target = root / "control/dashboard-status.json"
    payload = target.read_bytes()
    assert receipt["receipt"]["disposition"] == "committed"
    assert receipt["observed_disposition"] == "installed"
    assert receipt["receipt"]["status_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["persistence"]["disposition"] == "installed"
    assert json.loads(payload)["phase"] == "d0-blocked"
    assert target.stat().st_mode & 0o777 == 0o600

    receipt_path = root / receipt["persistence"]["relative"]
    assert commits == [target, receipt_path]
    receipt_partial = receipt_path.with_name(f".{receipt_path.stem}.partial")
    receipt_payload = receipt_path.read_bytes()
    receipt_path.unlink()
    receipt_partial.write_bytes(b"short")
    receipt_partial.chmod(0o400)

    retried = publish_dashboard_diagnostic(
        specification,
        campaign_root=root,
        storage_verifier=_storage,
        stability_seconds=0,
    )
    assert retried["observed_disposition"] == "already-installed"
    assert retried["persistence"]["disposition"] == "installed"
    assert receipt_path.read_bytes() == receipt_payload
    assert commits == [target, receipt_path, receipt_path]

    changed = _spec(
        expected="f" * 64,
        updated=int(specification["updated_unix_ms"]) + 1000,
    )
    with pytest.raises(D0Error, match="compare-and-swap"):
        publish_dashboard_diagnostic(
            changed,
            campaign_root=root,
            storage_verifier=_storage,
            stability_seconds=0,
        )


def test_dashboard_single_writer_lock_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "r2-map-v1"
    lock_path = root / "control/.dashboard-status.json.lock"
    lock_path.parent.mkdir(parents=True, mode=0o700)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(D0Error, match="another dashboard writer"):
            publish_dashboard_diagnostic(
                _spec(),
                campaign_root=root,
                storage_verifier=_storage,
                stability_seconds=0,
            )
    finally:
        os.close(descriptor)
