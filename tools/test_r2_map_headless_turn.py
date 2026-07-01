from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import signal
import sys
import threading
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("r2_map_headless_turn.py")
SPEC = importlib.util.spec_from_file_location("r2_map_headless_turn", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

RELATIVE = "logs/headless/session/turn-0001.jsonl"
PAYLOAD = b'{"type":"turn.completed"}\n'


def _receipt_relative(relative: str, maximum: int) -> str:
    semantic = hashlib.sha256(
        json.dumps(
            {
                "operation": "put-stream",
                "arguments": {
                    "relative": relative,
                    "max_bytes": maximum,
                    "expected_current": "absent",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"control/receipts/req-put-stream-{semantic[:32]}.json"


def _receipt() -> dict[str, object]:
    return {
        "relative": RELATIVE,
        "size": len(PAYLOAD),
        "mode": "0o400",
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "storage_receipt_relative": _receipt_relative(RELATIVE, 1024),
        "storage_receipt_sha256": "a" * 64,
    }


def _validate(value: object, *, maximum: int = 1024) -> dict[str, object]:
    return subject._validate_sink_receipt(
        json.dumps(value).encode(),
        relative=RELATIVE,
        maximum_bytes=maximum,
        bytes_written=len(PAYLOAD),
        stream_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
    )


def test_sink_receipt_binds_path_size_mode_payload_and_storage_receipt() -> None:
    assert _validate(_receipt()) == _receipt()


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("relative", "logs/headless/wrong"),
        ("size", True),
        ("size", len(PAYLOAD) + 1),
        ("mode", "0o600"),
        ("sha256", "b" * 64),
        ("storage_receipt_relative", "control/receipts/req-not-a-request.json"),
        ("storage_receipt_sha256", "not-a-sha"),
    ],
)
def test_sink_receipt_rejects_identity_drift(field: str, bad_value: object) -> None:
    value = _receipt()
    value[field] = bad_value
    with pytest.raises(subject.HeadlessTurnError, match="identity differs"):
        _validate(value)


def test_sink_receipt_rejects_bound_overflow_and_invalid_json() -> None:
    with pytest.raises(subject.HeadlessTurnError, match="identity differs"):
        _validate(_receipt(), maximum=len(PAYLOAD) - 1)
    with pytest.raises(subject.HeadlessTurnError, match="invalid JSON"):
        subject._validate_sink_receipt(
            b"not-json",
            relative=RELATIVE,
            maximum_bytes=1024,
            bytes_written=len(PAYLOAD),
            stream_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        )


class _RecordingBytesIO(io.BytesIO):
    snapshot = b""

    def close(self) -> None:
        self.snapshot = self.getvalue()
        super().close()


class _ShortWriteBytesIO(_RecordingBytesIO):
    def write(self, value: bytes) -> int:
        super().write(value)
        return len(value) - 1


def test_pump_copies_and_hashes_exact_bytes_then_closes_both_ends() -> None:
    source = io.BytesIO(PAYLOAD)
    destination = _RecordingBytesIO()
    result = subject.PumpResult()

    subject._pump(source, destination, result)

    assert source.closed is True
    assert destination.closed is True
    assert destination.snapshot == PAYLOAD
    assert result.bytes_written == len(PAYLOAD)
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert result.error is None


def test_pump_rejects_a_short_anonymous_pipe_write() -> None:
    result = subject.PumpResult()

    subject._pump(io.BytesIO(PAYLOAD), _ShortWriteBytesIO(), result)

    assert isinstance(result.error, subject.HeadlessTurnError)
    assert "no progress" in str(result.error)
    assert result.bytes_written == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


def test_capture_drains_but_bounds_retained_diagnostics() -> None:
    source = io.BytesIO(b"0123456789")
    result = subject.CaptureResult()

    subject._capture(source, result, maximum=4)

    assert source.closed is True
    assert result.payload == b"0123"
    assert result.overflow is True
    assert result.error is None


def test_terminate_kills_descendants_after_process_leader_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedLeader:
        pid = 424242

        @staticmethod
        def wait(timeout: int | None = None) -> int:
            del timeout
            return 0

        @staticmethod
        def poll() -> int:
            return 0

    group_exists = True
    signals: list[int] = []

    def killpg(_pid: int, sent_signal: int) -> None:
        nonlocal group_exists
        if not group_exists:
            raise ProcessLookupError
        if sent_signal != 0:
            signals.append(sent_signal)
        if sent_signal == signal.SIGKILL:
            group_exists = False

    monkeypatch.setattr(subject.os, "killpg", killpg)
    subject._terminate(ExitedLeader())  # type: ignore[arg-type]
    assert signals == [signal.SIGTERM, signal.SIGKILL]


class _BlockingReader:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._payload = b""
        self._offset = 0
        self._published = False

    def publish(self, payload: bytes) -> None:
        with self._condition:
            self._payload = payload
            self._published = True
            self._condition.notify_all()

    def read(self, size: int = -1) -> bytes:
        with self._condition:
            while not self._published:
                self._condition.wait()
            if self._offset == len(self._payload):
                return b""
            end = len(self._payload) if size < 0 else self._offset + size
            chunk = self._payload[self._offset : end]
            self._offset += len(chunk)
            return chunk

    def close(self) -> None:
        return None


class _SinkInput(io.BytesIO):
    def __init__(self, on_close) -> None:
        super().__init__()
        self._on_close = on_close

    def close(self) -> None:
        if not self.closed:
            self._on_close(self.getvalue())
        super().close()


class _FakeSink:
    def __init__(self, command: list[str], request_digit: str, *, exit_code: int = 0) -> None:
        self.relative = command[command.index("--relative") + 1]
        self.maximum = int(command[command.index("--max-bytes") + 1])
        self.returncode: int | None = None
        self.stdout = _BlockingReader()
        self.stderr = _BlockingReader()
        self.stdin = _SinkInput(self._finish)
        self._request_digit = request_digit
        self._exit_code = exit_code

    def _finish(self, payload: bytes) -> None:
        receipt = {
            "relative": self.relative,
            "size": len(payload),
            "mode": "0o400",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "storage_receipt_relative": _receipt_relative(self.relative, self.maximum),
            "storage_receipt_sha256": self._request_digit * 64,
        }
        self.stdout.publish(json.dumps(receipt).encode())
        self.stderr.publish(b"")
        self.returncode = self._exit_code

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakeCodex:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(PAYLOAD)
        self.stderr = io.BytesIO(b"diagnostic\n")

    def poll(self) -> int:
        return 0

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return 0

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _arguments() -> argparse.Namespace:
    executable = Path(sys.executable)
    return argparse.Namespace(
        codex_bin=executable,
        remote_python=executable,
        remote_tool=MODULE_PATH,
        python_root=MODULE_PATH.parent,
        repository=MODULE_PATH.parent,
        session_id="session",
        prompt_path=MODULE_PATH,
        events_relative=RELATIVE,
        stderr_relative="logs/headless/session/turn-0001.stderr.log",
        events_max_bytes=1024,
        stderr_max_bytes=1024,
        heartbeat_pid=os.getpid(),
    )


def test_run_turn_requires_two_verified_sink_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children: list[object] = []

    def popen(command: list[str], **_kwargs):
        if len(children) < 2:
            child: object = _FakeSink(command, str(len(children) + 1))
        else:
            child = _FakeCodex()
        children.append(child)
        return child

    monkeypatch.setattr(subject.subprocess, "Popen", popen)

    result = subject.run_turn(_arguments())

    assert result["codex_exit_code"] == 0
    assert result["sinks_verified"] is True
    assert [sink["bytes"] for sink in result["sinks"]] == [
        len(PAYLOAD),
        len(b"diagnostic\n"),
    ]
    assert len(children) == 3


def test_run_turn_propagates_nonzero_sink_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children: list[object] = []

    def popen(command: list[str], **_kwargs):
        if len(children) < 2:
            child: object = _FakeSink(
                command,
                str(len(children) + 1),
                exit_code=7 if not children else 0,
            )
        else:
            child = _FakeCodex()
        children.append(child)
        return child

    monkeypatch.setattr(subject.subprocess, "Popen", popen)

    with pytest.raises(subject.HeadlessTurnError, match="stream sinks failed"):
        subject.run_turn(_arguments())


def test_supervisor_uses_verified_multiplexer_and_no_process_substitution() -> None:
    script = MODULE_PATH.with_name("r2_map_headless_resume.sh").read_text()

    assert 'TURN_RUNNER="${REPOSITORY}/tools/r2_map_headless_turn.py"' in script
    assert '--events-relative "${events_relative}"' in script
    assert '--stderr-relative "${stderr_relative}"' in script
    assert '--heartbeat-pid "${heartbeat_pid}"' in script
    assert "if (( runner_status != 0 )); then" in script
    assert "sinks_verified" in script
    assert "bound_diagnostic" in script
    assert "trap cleanup EXIT" in script
    assert "trap 'exit 143' TERM" in script
    assert "> >(" not in script
    assert "2> >(" not in script
    assert "wait ||" not in script
