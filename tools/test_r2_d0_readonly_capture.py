from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest
import r2_d0_readonly_capture as capture


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def test_non_json_failure_persists_raw_output_and_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "probe.py"
    _write(
        probe,
        b'import sys; sys.stdout.write("not-json"); sys.stderr.write("root-cause"); '
        b"raise SystemExit(7)\n",
    )
    public_key = tmp_path / "public-key"
    _write(public_key, b"test-public-key\n")
    output_root = tmp_path / "capture"
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "schema_id": "cascadia.r2-map.d0-post-failure-network-authorization.v1",
        "schema_version": 1,
        "authorization_sha256": "pending",
        "capture_runner_sha256": capture.sha256_bytes(Path(capture.__file__).read_bytes()),
        "probe_sha256": capture.sha256_bytes(probe.read_bytes()),
        "public_key_sha256": capture.sha256_bytes(public_key.read_bytes()),
        "remote_output_root": str(output_root),
        "expires_unix_ms": time.time_ns() // 1_000_000 + 60_000,
        "status": "authorized-once",
        "command": [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            str(probe),
            "--authorization",
            str(authorization_path),
        ],
    }
    authorization["authorization_sha256"] = capture.document_sha256(
        authorization, "authorization_sha256"
    )
    _write(authorization_path, capture.canonical_json(authorization))
    monkeypatch.setattr(
        capture,
        "swap_sample",
        lambda: {"unix_ms": 1, "stdout_sha256": "0" * 64, "zero": True},
    )

    with pytest.raises(capture.CaptureError, match="raw output persisted"):
        capture.run(
            argparse.Namespace(
                authorization=authorization_path,
                probe=probe,
                public_key=public_key,
                output_root=output_root,
            )
        )

    assert (output_root / "stdout.json").read_bytes() == b"not-json"
    assert (output_root / "stderr.bin").read_bytes() == b"root-cause"
    failure = json.loads((output_root / "failure.json").read_bytes())
    state = json.loads((output_root / "runner-state.json").read_bytes())
    assert failure["returncode"] == 7
    assert failure["parse_error"] == "JSONDecodeError"
    assert state["status"] == "captured-fail"
    assert state["failure_sha256"] == failure["failure_sha256"]


def test_network_capture_environment_includes_homebrew_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "probe.py"
    _write(
        probe,
        b"import json,os,sys; a=json.load(open(sys.argv[2])); "
        b"r={'authorization_sha256':a['authorization_sha256'],"
        b"'path':os.environ['PATH'],'status':'pass-diagnostic'}; "
        b"r['result_sha256']='pending'; "
        b"import hashlib; "
        b"e=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')).encode(); "
        b"p={k:v for k,v in r.items() if k!='result_sha256'}; "
        b"r['result_sha256']=hashlib.sha256(e(p)).hexdigest(); "
        b"sys.stdout.buffer.write(e(r))\n",
    )
    public_key = tmp_path / "public-key"
    _write(public_key, b"test-public-key\n")
    output_root = tmp_path / "capture"
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "schema_id": "cascadia.r2-map.d0-post-failure-network-authorization.v1",
        "schema_version": 1,
        "authorization_sha256": "pending",
        "capture_runner_sha256": capture.sha256_bytes(Path(capture.__file__).read_bytes()),
        "probe_sha256": capture.sha256_bytes(probe.read_bytes()),
        "public_key_sha256": capture.sha256_bytes(public_key.read_bytes()),
        "remote_output_root": str(output_root),
        "expires_unix_ms": time.time_ns() // 1_000_000 + 60_000,
        "status": "authorized-once",
        "command": [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            str(probe),
            "--authorization",
            str(authorization_path),
        ],
    }
    authorization["authorization_sha256"] = capture.document_sha256(
        authorization, "authorization_sha256"
    )
    _write(authorization_path, capture.canonical_json(authorization))
    monkeypatch.setattr(
        capture,
        "swap_sample",
        lambda: {"unix_ms": 1, "stdout_sha256": "0" * 64, "zero": True},
    )

    state = capture.run(
        argparse.Namespace(
            authorization=authorization_path,
            probe=probe,
            public_key=public_key,
            output_root=output_root,
        )
    )
    result = json.loads((output_root / "stdout.json").read_bytes())
    assert state["status"] == "captured-pass"
    assert result["path"].startswith("/opt/homebrew/bin:")
