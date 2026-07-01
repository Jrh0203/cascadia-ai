from __future__ import annotations

import json
import plistlib
import stat
from pathlib import Path

import blake3
import pytest
import r2_map_dashboard_fetch as subject


def payload() -> bytes:
    fixture = subject.REPOSITORY / "tests/fixtures/r2_map/dashboard-status-v1-contracts-ready.json"
    return fixture.read_bytes()


def test_projection_binds_john1_path_hash_timestamp_and_exact_payload(tmp_path: Path) -> None:
    source = payload()
    projection = subject.build_serving_projection(source, fetched_unix_ms=1_781_755_200_500)
    output = tmp_path / "projection.json"
    written = subject.write_projection(output, projection)
    decoded = json.loads(output.read_text())

    assert written == output.stat().st_size <= 64 * 1024
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert decoded["schema_id"] == subject.PROJECTION_SCHEMA_ID
    assert decoded["canonical_host"] == "john1"
    assert decoded["canonical_path"] == str(subject.CANONICAL_PATH)
    assert decoded["canonical_blake3"] == blake3.blake3(source).hexdigest()
    assert decoded["canonical_updated_unix_ms"] == 1_781_755_200_000
    assert decoded["fetched_unix_ms"] == 1_781_755_200_500
    assert decoded["canonical_payload"].encode() == source
    assert not list(tmp_path.glob(".projection.json.*.tmp"))
    assert not list(tmp_path.glob("*.lock"))


def test_invalid_source_never_replaces_last_good_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "projection.json"
    good = subject.build_serving_projection(payload(), fetched_unix_ms=10)
    subject.write_projection(output, good)
    before = output.read_bytes()

    broken = json.loads(payload())
    broken["hosts"]["john4"] = broken["hosts"].pop("john3")
    monkeypatch.setattr(
        subject,
        "fetch_canonical_payload",
        lambda: json.dumps(broken).encode(),
    )
    with pytest.raises(subject.DashboardFetchError, match="exactly john1"):
        subject.fetch_and_publish(output)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(schema_id="wrong"), "schema identity"),
        (lambda value: value.update(campaign_id="wrong"), "campaign identity"),
        (lambda value: value.update(updated_unix_ms=0), "timestamp"),
        (lambda value: value.update(extra=float("nan")), "invalid JSON"),
    ],
)
def test_source_envelope_fails_closed(mutation, message: str) -> None:
    value = json.loads(payload())
    mutation(value)
    encoded = json.dumps(value).encode()
    with pytest.raises(subject.DashboardFetchError, match=message):
        subject.validate_canonical_payload(encoded)


def _install_local_canonical_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: bytes | None = None,
) -> Path:
    root = tmp_path / "cascadia-bench" / "r2-map-v1"
    control = root / "control"
    control.mkdir(parents=True)
    for directory in (root.parent, root, control):
        directory.chmod(0o700)
    status_path = control / "dashboard-status.json"
    status_path.write_bytes(payload() if contents is None else contents)
    status_path.chmod(0o600)
    monkeypatch.setattr(subject, "CANONICAL_ROOT", root)
    monkeypatch.setattr(subject, "CANONICAL_PATH", status_path)
    return status_path


def test_fetch_uses_one_owner_private_local_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_local_canonical_status(tmp_path, monkeypatch)
    assert subject.fetch_canonical_payload() == payload()


def test_fetch_rejects_oversize_symlink_and_permissive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = _install_local_canonical_status(
        tmp_path,
        monkeypatch,
        b"x" * (subject.MAX_CANONICAL_BYTES + 1),
    )
    with pytest.raises(subject.DashboardFetchError, match="exceeds"):
        subject.fetch_canonical_payload()

    status_path.write_bytes(payload())
    status_path.chmod(0o644)
    with pytest.raises(subject.DashboardFetchError, match="ownership or mode"):
        subject.fetch_canonical_payload()

    status_path.unlink()
    target = tmp_path / "status-target.json"
    target.write_bytes(payload())
    status_path.symlink_to(target)
    with pytest.raises(subject.DashboardFetchError, match="unavailable"):
        subject.fetch_canonical_payload()


def test_fetch_rejects_permissive_or_linked_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = _install_local_canonical_status(tmp_path, monkeypatch)
    status_path.parent.chmod(0o755)
    with pytest.raises(subject.DashboardFetchError, match="ownership or mode"):
        subject.fetch_canonical_payload()

    status_path.parent.chmod(0o700)
    status_path.unlink()
    status_path.parent.rmdir()
    real_control = tmp_path / "real-control"
    real_control.mkdir(mode=0o700)
    (real_control / "dashboard-status.json").write_bytes(payload())
    (real_control / "dashboard-status.json").chmod(0o600)
    status_path.parent.symlink_to(real_control, target_is_directory=True)
    with pytest.raises(subject.DashboardFetchError, match="not a real directory"):
        subject.fetch_canonical_payload()


def test_launch_agents_wire_only_the_v2_projection() -> None:
    dashboard = plistlib.loads(
        (subject.REPOSITORY / "tools/com.johnherrick.cascadia.dashboard.plist").read_bytes()
    )
    arguments = dashboard["ProgramArguments"]
    status_index = arguments.index("--r2-map-status-path")
    assert arguments[status_index + 1] == (
        "artifacts/cluster/r2-map-dashboard-serving-projection-v2.json"
    )

    fetcher = plistlib.loads(
        (
            subject.REPOSITORY / "tools/com.johnherrick.cascadia.r2-map-dashboard-fetch.plist"
        ).read_bytes()
    )
    assert fetcher["ProgramArguments"] == [
        "/Users/johnherrick/cascadia/.venv/bin/python",
        "tools/r2_map_dashboard_fetch.py",
        "--watch",
        "--interval-seconds",
        "10",
    ]
    assert fetcher["StandardOutPath"] == "/dev/null"
    assert fetcher["StandardErrorPath"] == "/dev/null"

    heartbeat = plistlib.loads(
        (
            subject.REPOSITORY
            / "tools/com.johnherrick.cascadia.r2-map-d0-dashboard-watch.plist"
        ).read_bytes()
    )
    assert heartbeat["ProgramArguments"] == [
        "/Users/johnherrick/cascadia/.venv/bin/python",
        "tools/r2_map_d0_dashboard_watch.py",
        "--watch",
        "--interval-seconds",
        "5",
    ]
    assert heartbeat["StandardOutPath"] == "/dev/null"


def test_headless_supervisor_streams_artifacts_without_local_campaign_temp() -> None:
    script = (subject.REPOSITORY / "tools/r2_map_headless_resume.sh").read_text()
    assert 'CAMPAIGN_ROOT="/Users/john2/cascadia-bench/r2-map-v1"' in script
    assert "/Volumes/John_1" not in script
    assert "/tmp" not in script
    assert "mkfifo" not in script
    assert "--output-last-message" not in script
    assert "tools/r2_map_remote_storage.py" in script
    assert "/usr/bin/ssh" not in script
    assert "remote_exec" not in script
    assert '"${REMOTE_CLI[@]}" preflight' in script
    assert '"${REMOTE_CLI[@]}" put-stream' in script
    assert '"${REMOTE_CLI[@]}" lock acquire' in script
    assert '"${REMOTE_CLI[@]}" lock renew' in script
    assert '"${REMOTE_CLI[@]}" lock release' in script
    assert "r2_map_headless_turn.py" in script
    assert '--events-relative "${events_relative}"' in script
    assert '--stderr-relative "${stderr_relative}"' in script
    assert '--heartbeat-pid "${heartbeat_pid}"' in script
    assert 'if (( runner_status != 0 )); then' in script
    assert "sinks_verified" in script
    assert "> >(" not in script
    assert "2> >(" not in script

    for prompt_name in (
        "r2_map_headless_resume_prompt.txt",
        "r2_map_headless_continuation_prompt.txt",
    ):
        prompt = (subject.REPOSITORY / "tools" / prompt_name).read_text()
        assert "john2:/Users/john2/cascadia-bench/r2-map-v1" in prompt or (
            "host john2 at /Users/john2/cascadia-bench/r2-map-v1" in prompt
        )
        assert "/private/tmp" in prompt
        assert "64 MiB combined" in prompt
        assert "mandatory cleanup receipt" in prompt
        assert "Never touch" in prompt or "do not write" in prompt
