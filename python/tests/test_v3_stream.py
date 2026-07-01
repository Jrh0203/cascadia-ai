from __future__ import annotations

import io
import subprocess

from cascadia_v3_mlx.stream import BatchStreamError, _producer_eof_error


class _FinishedProcess:
    def __init__(self, status: int, detail: bytes):
        self.status = status
        self.stderr = io.BytesIO(detail)

    def wait(self, timeout: float | None = None) -> int:
        assert timeout == 5
        return self.status


class _LiveProcess:
    stderr = io.BytesIO(b"must not be consumed")

    def wait(self, timeout: float | None = None) -> int:
        assert timeout == 5
        raise subprocess.TimeoutExpired("v3-batch-stream", timeout)


def test_producer_eof_error_preserves_status_and_stderr() -> None:
    error = _producer_eof_error(
        _FinishedProcess(17, b"scientific corpus mismatch\n"),  # type: ignore[arg-type]
        BatchStreamError("native V3 batch stream ended unexpectedly"),
    )
    assert str(error) == (
        "native V3 producer ended before its terminal frame (status 17): "
        "scientific corpus mismatch"
    )


def test_producer_eof_error_does_not_block_on_live_process() -> None:
    original = BatchStreamError("native V3 batch stream ended unexpectedly")
    error = _producer_eof_error(_LiveProcess(), original)  # type: ignore[arg-type]
    assert str(error) == (
        "native V3 batch stream ended unexpectedly; "
        "native producer remained alive after closing stdout"
    )
