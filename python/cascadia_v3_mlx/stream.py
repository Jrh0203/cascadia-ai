"""Whole-batch client for the native Rust V3 CSR producer."""

from __future__ import annotations

import queue
import struct
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import mlx.core as mx
import numpy as np

from .contracts import BASE_FEATURE_ROWS, V3MlxConfig
from .model import CsrBatch, CsrRows

STREAM_MAGIC = b"CSV3BT1\0"
FRAME_MAGIC = b"BCH1"
STREAM_VERSION = 2


class BatchStreamError(ValueError):
    """Raised when the native batch process emits a malformed or failed stream."""


def _producer_eof_error(
    process: subprocess.Popen[bytes], original: BatchStreamError
) -> BatchStreamError:
    """Preserve the native producer's status and stderr when its pipe closes early."""
    try:
        status = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return BatchStreamError(
            f"{original}; native producer remained alive after closing stdout"
        )
    detail = ""
    if process.stderr is not None:
        detail = process.stderr.read().decode(errors="replace").strip()
    suffix = f": {detail}" if detail else ""
    return BatchStreamError(
        f"native V3 producer ended before its terminal frame (status {status}){suffix}"
    )


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise BatchStreamError("native V3 batch stream ended unexpectedly")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_csr(stream: BinaryIO, rows: int) -> CsrRows:
    offsets = np.frombuffer(_read_exact(stream, (rows + 1) * 4), dtype="<u4").astype(np.int64)
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise BatchStreamError("native V3 CSR offsets are noncanonical")
    count = int(offsets[-1])
    indices = np.frombuffer(_read_exact(stream, count * 4), dtype="<u4").astype(np.int32)
    counts = np.frombuffer(_read_exact(stream, count * 2), dtype="<u2").astype(np.float32)
    lengths = np.diff(offsets)
    row_indices = np.repeat(np.arange(rows, dtype=np.int32), lengths)
    gradient_positions = np.argsort(indices, kind="stable").astype(np.int32)
    sorted_features = indices[gradient_positions]
    gradient_features, starts = np.unique(sorted_features, return_index=True)
    gradient_offsets = np.concatenate(
        (starts.astype(np.int32), np.array([count], dtype=np.int32))
    )
    return CsrRows(
        offsets=mx.array(offsets.astype(np.int32)),
        indices=mx.array(indices),
        counts=mx.array(counts),
        row_indices=mx.array(row_indices),
        gradient_positions=mx.array(gradient_positions),
        gradient_features=mx.array(gradient_features.astype(np.int32)),
        gradient_offsets=mx.array(gradient_offsets),
    )


def _read_batch(stream: BinaryIO, config: V3MlxConfig) -> CsrBatch | None:
    if _read_exact(stream, 4) != FRAME_MAGIC:
        raise BatchStreamError("native V3 batch frame magic is invalid")
    rows = struct.unpack("<I", _read_exact(stream, 4))[0]
    if rows == 0:
        return None
    sparse = [_read_csr(stream, rows) for _ in range(6)]
    phases = np.frombuffer(_read_exact(stream, rows), dtype=np.uint8).astype(np.int32)
    targets = np.frombuffer(_read_exact(stream, rows * 4), dtype="<f4").astype(np.float32)
    confidence = np.frombuffer(_read_exact(stream, rows * 4), dtype="<f4").astype(np.float32)
    batch = CsrBatch(
        own_base=sparse[0],
        field_base=sparse[1],
        own_opportunities=sparse[2],
        field_opportunities=sparse[3],
        own_opportunity_factors=sparse[4],
        field_opportunity_factors=sparse[5],
        phase_buckets=mx.array(phases),
        targets=mx.array(targets),
        confidence_weights=mx.array(confidence),
    )
    batch.validate(config)
    return batch


class RustBatchStream(Iterator[CsrBatch]):
    """Double-buffered native producer; Python receives only complete batches."""

    def __init__(
        self,
        binary: Path,
        inputs: list[Path],
        config: V3MlxConfig,
        *,
        batch_size: int,
        epochs: int,
        allow_scientific_data: bool,
        d6_cycle: bool = False,
        campaign_state: Path | None = None,
        cycle: int | None = None,
        teacher_lambda: float | None = None,
        max_examples: int | None = None,
        uniform_phase: bool = False,
        d6_offset: int = 0,
        score_quantile_boundaries: tuple[float, ...] | None = None,
        expansion_threads: int = 4,
    ):
        if not inputs or batch_size <= 0 or epochs <= 0 or expansion_threads <= 0:
            raise ValueError("native V3 stream requires inputs, batch size, and epochs")
        command = [str(binary)]
        for path in inputs:
            command.extend(("--input", str(path)))
        command.extend(("--batch-size", str(batch_size), "--epochs", str(epochs)))
        command.extend(("--expansion-threads", str(expansion_threads)))
        if allow_scientific_data:
            command.append("--allow-scientific-data")
            if campaign_state is None:
                raise ValueError("scientific V3 stream requires a campaign state")
            command.extend(("--campaign-state", str(campaign_state)))
            if cycle is not None:
                command.extend(("--cycle", str(cycle)))
        elif campaign_state is not None or cycle is not None:
            raise ValueError("campaign state is only valid for scientific V3 streams")
        if teacher_lambda is not None:
            if not 0.0 <= teacher_lambda <= 1.0:
                raise ValueError("teacher lambda must be within [0, 1]")
            command.extend(("--teacher-lambda", repr(teacher_lambda)))
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max examples must be positive")
            if uniform_phase and max_examples % 8:
                raise ValueError("uniform-phase max examples must be divisible by eight")
            command.extend(("--max-examples", str(max_examples)))
        if uniform_phase:
            command.append("--uniform-phase")
        if score_quantile_boundaries is not None:
            if (
                len(score_quantile_boundaries) != 24
                or any(not np.isfinite(value) for value in score_quantile_boundaries)
                or any(
                    not (values[0] < values[1] < values[2])
                    for values in (
                        score_quantile_boundaries[offset : offset + 3]
                        for offset in range(0, 24, 3)
                    )
                )
            ):
                raise ValueError(
                    "score quantile boundaries must be eight finite increasing phase triples"
                )
            if batch_size % 32 or (max_examples is not None and max_examples % 32):
                raise ValueError("score-balanced stream sizes must be divisible by 32")
            command.append("--score-quantile-boundaries")
            command.extend(repr(value) for value in score_quantile_boundaries)
        if d6_offset < 0:
            raise ValueError("D6 offset cannot be negative")
        if d6_offset:
            command.extend(("--d6-offset", str(d6_offset)))
        if d6_cycle:
            command.append("--d6-cycle")
        self.config = config
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None
        header = _read_exact(self.process.stdout, 24)
        magic, version, flags, base_rows, opportunity_rows, factor_rows = struct.unpack(
            "<8sHHIII", header
        )
        if (
            magic != STREAM_MAGIC
            or version != STREAM_VERSION
            or flags != int(d6_cycle)
            or base_rows != BASE_FEATURE_ROWS
            or opportunity_rows != config.opportunity_feature_rows
            or factor_rows != config.opportunity_training_factor_rows
        ):
            self.process.kill()
            raise BatchStreamError("native V3 stream header is incompatible")
        self._queue: queue.Queue[CsrBatch | Exception | None] = queue.Queue(maxsize=2)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _publish(self, value: CsrBatch | Exception | None) -> None:
        while not self._stopping.is_set():
            try:
                self._queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        assert self.process.stdout is not None
        try:
            while not self._stopping.is_set() and (
                batch := _read_batch(self.process.stdout, self.config)
            ) is not None:
                self._publish(batch)
            if self._stopping.is_set():
                return
            status = self.process.wait()
            if status:
                assert self.process.stderr is not None
                detail = self.process.stderr.read().decode(errors="replace")
                raise BatchStreamError(f"native V3 producer failed ({status}): {detail}")
            self._publish(None)
        except Exception as error:  # propagated to the consumer thread
            if not self._stopping.is_set():
                if (
                    isinstance(error, BatchStreamError)
                    and str(error) == "native V3 batch stream ended unexpectedly"
                ):
                    error = _producer_eof_error(self.process, error)
                self._publish(error)

    def __iter__(self) -> RustBatchStream:
        return self

    def __next__(self) -> CsrBatch:
        value = self._queue.get()
        if value is None:
            raise StopIteration
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self._stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._thread.join(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    @property
    def producer_alive(self) -> bool:
        return self._thread.is_alive()
