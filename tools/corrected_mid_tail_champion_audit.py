#!/usr/bin/env python3
"""Independently audit the F5 corrected-mid-tail champion migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import blake3

AUDIT_SCHEMA_VERSION = 1
AUDIT_ID = "corrected-mid-tail-champion-audit-v1"
SCHEMA_ID = "legacy-mid-v4-fixed-v1"
HISTORICAL_SCHEMA_ID = "historical-legacy-mid-v4opp-11231"
HISTORICAL_MAGIC = b"NNUE"
CORRECTED_MAGIC = b"NNUC"
CORRECTED_CONTAINER_VERSION = 1
CORRECTED_SCHEMA_TAG = b"MIDTAIL-CORR-V1\0"
FEATURE_COUNT = 11_231
BASE_FEATURE_COUNT = 10_561
DEFECT_FEATURE_COUNT = 301
OPPONENT_FEATURE_COUNT = 369
CORRECTED_TAIL_FEATURE_COUNT = 301
HISTORICAL_DEFECT_BASE = 10_561
HISTORICAL_OPPONENT_BASE = 10_862
CORRECTED_OPPONENT_BASE = 10_561
CORRECTED_TAIL_BASE = 10_930
PRODUCTION_HIDDEN1 = 512
PRODUCTION_HIDDEN2 = 64
NUM_SPLIT_HEADS = 11
FLOAT_BYTES = 4
PRODUCTION_SOURCE_BYTES = 23_134_992
PRODUCTION_SOURCE_BLAKE3 = "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
PRODUCTION_SOURCE_SHA256 = "f40627623d3686d7d2d6a2f8f109445f54e449f0d7045552ebe831f955a58f48"


class AuditError(RuntimeError):
    """Raised when an input cannot be interpreted under the frozen contract."""


@dataclass(frozen=True)
class CorrectedHeader:
    container_version: int
    head_format_version: int
    schema_tag: bytes
    feature_count: int
    hidden1: int
    hidden2: int
    raw: bytes


@dataclass(frozen=True)
class TensorRegion:
    name: str
    float_count: int

    @property
    def byte_count(self) -> int:
        return self.float_count * FLOAT_BYTES


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "blake3":
        digest: Any = blake3.blake3()
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "bytes": path.stat().st_size,
        "blake3": _hash_file(path, "blake3"),
        "sha256": _hash_file(path, "sha256"),
    }


def _hash_region(path: Path, offset: int, byte_count: int, algorithm: str = "blake3") -> str:
    if algorithm == "blake3":
        digest: Any = blake3.blake3()
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = byte_count
        while remaining:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                failed_at = offset + byte_count - remaining
                raise AuditError(f"unexpected EOF in {path} at byte {failed_at}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _regions_equal(
    left: Path,
    left_offset: int,
    right: Path,
    right_offset: int,
    byte_count: int,
) -> bool:
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        left_handle.seek(left_offset)
        right_handle.seek(right_offset)
        remaining = byte_count
        while remaining:
            requested = min(1 << 20, remaining)
            left_block = left_handle.read(requested)
            right_block = right_handle.read(requested)
            if len(left_block) != requested or len(right_block) != requested:
                raise AuditError("unexpected EOF while comparing migration regions")
            if left_block != right_block:
                return False
            remaining -= requested
    return True


def _read_exact(handle: BinaryIO, byte_count: int, label: str) -> bytes:
    payload = handle.read(byte_count)
    if len(payload) != byte_count:
        raise AuditError(f"{label} is truncated")
    return payload


def _read_source_version(path: Path) -> tuple[int, bytes]:
    with path.open("rb") as handle:
        raw = _read_exact(handle, 8, "historical checkpoint header")
    if raw[:4] != HISTORICAL_MAGIC:
        raise AuditError("historical checkpoint does not use the NNUE magic")
    return struct.unpack_from("<I", raw, 4)[0], raw


def _read_corrected_header(path: Path) -> CorrectedHeader:
    with path.open("rb") as handle:
        raw = _read_exact(handle, 40, "corrected checkpoint header")
    return CorrectedHeader(
        container_version=struct.unpack_from("<I", raw, 4)[0],
        head_format_version=struct.unpack_from("<I", raw, 8)[0],
        schema_tag=raw[12:28],
        feature_count=struct.unpack_from("<I", raw, 28)[0],
        hidden1=struct.unpack_from("<I", raw, 32)[0],
        hidden2=struct.unpack_from("<I", raw, 36)[0],
        raw=raw,
    )


def downstream_tensors(head_format_version: int, hidden1: int, hidden2: int) -> list[TensorRegion]:
    if head_format_version not in range(1, 5):
        raise AuditError(f"unsupported NNUE head-format version {head_format_version}")
    tensors = [
        TensorRegion("b1", hidden1),
        TensorRegion("w2", hidden1 * hidden2),
        TensorRegion("b2", hidden2),
        TensorRegion("w3", hidden2),
        TensorRegion("b3", 1),
        TensorRegion("w3_policy", hidden2),
        TensorRegion("b3_policy", 1),
    ]
    if head_format_version >= 2:
        tensors.extend(
            [
                TensorRegion("w3_wildlife", hidden2),
                TensorRegion("b3_wildlife", 1),
                TensorRegion("w3_habitat", hidden2),
                TensorRegion("b3_habitat", 1),
            ]
        )
    if head_format_version >= 3:
        tensors.extend(
            [
                TensorRegion("w3_heads", NUM_SPLIT_HEADS * hidden2),
                TensorRegion("b3_heads", NUM_SPLIT_HEADS),
            ]
        )
    if head_format_version >= 4:
        tensors.extend(
            [
                TensorRegion("w3_var", hidden2),
                TensorRegion("b3_var", 1),
            ]
        )
    return tensors


def _signed_zero_counts(path: Path, offset: int, float_count: int) -> dict[str, int]:
    positive_zero = 0
    negative_zero = 0
    nonzero = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = float_count
        while remaining:
            count = min(32_768, remaining)
            payload = _read_exact(handle, count * FLOAT_BYTES, "floating-point region")
            for (bits,) in struct.iter_unpack("<I", payload):
                if bits == 0:
                    positive_zero += 1
                elif bits == 0x8000_0000:
                    negative_zero += 1
                else:
                    nonzero += 1
            remaining -= count
    return {
        "float_count": float_count,
        "positive_zero": positive_zero,
        "negative_zero": negative_zero,
        "nonzero": nonzero,
    }


def _checks_with_reasons(checks: dict[str, bool]) -> tuple[str, list[str]]:
    failures = [name for name, passed in checks.items() if not passed]
    return ("pass" if not failures else "fail", failures)


def audit_migration(
    source: Path,
    corrected: Path,
    *,
    source_label: str,
    corrected_label: str,
    expected_source_bytes: int = PRODUCTION_SOURCE_BYTES,
    expected_source_blake3: str = PRODUCTION_SOURCE_BLAKE3,
    expected_source_sha256: str = PRODUCTION_SOURCE_SHA256,
    expected_hidden1: int = PRODUCTION_HIDDEN1,
    expected_hidden2: int = PRODUCTION_HIDDEN2,
) -> dict[str, Any]:
    source = source.resolve()
    corrected = corrected.resolve()
    for path, label in ((source, "source"), (corrected, "corrected")):
        if path.is_symlink() or not path.is_file():
            raise AuditError(f"{label} checkpoint must be a regular non-symlink file: {path}")

    source_version, source_header = _read_source_version(source)
    corrected_header = _read_corrected_header(corrected)
    tensors = downstream_tensors(source_version, expected_hidden1, expected_hidden2)
    downstream_bytes = sum(tensor.byte_count for tensor in tensors)
    row_bytes = expected_hidden1 * FLOAT_BYTES
    w1_bytes = FEATURE_COUNT * row_bytes
    expected_source_size = 8 + w1_bytes + downstream_bytes
    expected_corrected_size = 40 + w1_bytes + downstream_bytes

    source_identity = file_identity(source, source_label)
    corrected_identity = file_identity(corrected, corrected_label)
    header_checks = {
        "source_magic_exact": source_header[:4] == HISTORICAL_MAGIC,
        "head_format_version_supported": source_version in range(1, 5),
        "corrected_magic_exact": corrected_header.raw[:4] == CORRECTED_MAGIC,
        "corrected_container_version_exact": (
            corrected_header.container_version == CORRECTED_CONTAINER_VERSION
        ),
        "corrected_head_format_matches_source": (
            corrected_header.head_format_version == source_version
        ),
        "corrected_schema_tag_exact": corrected_header.schema_tag == CORRECTED_SCHEMA_TAG,
        "corrected_feature_count_exact": corrected_header.feature_count == FEATURE_COUNT,
        "corrected_hidden1_exact": corrected_header.hidden1 == expected_hidden1,
        "corrected_hidden2_exact": corrected_header.hidden2 == expected_hidden2,
    }
    size_checks = {
        "source_production_size_exact": source_identity["bytes"] == expected_source_bytes,
        "source_layout_size_exact": source_identity["bytes"] == expected_source_size,
        "corrected_layout_size_exact": corrected_identity["bytes"] == expected_corrected_size,
        "source_has_no_trailing_bytes": source_identity["bytes"] == expected_source_size,
        "corrected_has_no_trailing_bytes": corrected_identity["bytes"] == expected_corrected_size,
    }
    identity_checks = {
        "source_production_blake3_exact": source_identity["blake3"] == expected_source_blake3,
        "source_production_sha256_exact": source_identity["sha256"] == expected_source_sha256,
    }

    source_w1_offset = 8
    corrected_w1_offset = 40
    base_bytes = BASE_FEATURE_COUNT * row_bytes
    defect_bytes = DEFECT_FEATURE_COUNT * row_bytes
    opponent_bytes = OPPONENT_FEATURE_COUNT * row_bytes
    corrected_tail_bytes = CORRECTED_TAIL_FEATURE_COUNT * row_bytes
    source_defect_offset = source_w1_offset + HISTORICAL_DEFECT_BASE * row_bytes
    source_opponent_offset = source_w1_offset + HISTORICAL_OPPONENT_BASE * row_bytes
    corrected_opponent_offset = corrected_w1_offset + CORRECTED_OPPONENT_BASE * row_bytes
    corrected_tail_offset = corrected_w1_offset + CORRECTED_TAIL_BASE * row_bytes

    base_identical = _regions_equal(
        source,
        source_w1_offset,
        corrected,
        corrected_w1_offset,
        base_bytes,
    )
    opponent_identical = _regions_equal(
        source,
        source_opponent_offset,
        corrected,
        corrected_opponent_offset,
        opponent_bytes,
    )
    tail_zero_counts = _signed_zero_counts(
        corrected,
        corrected_tail_offset,
        CORRECTED_TAIL_FEATURE_COUNT * expected_hidden1,
    )
    tail_signed_zero = tail_zero_counts["nonzero"] == 0
    first_layer_checks = {
        "base_rows_byte_identical": base_identical,
        "historical_defect_rows_discarded": (
            base_identical
            and opponent_identical
            and tail_signed_zero
            and BASE_FEATURE_COUNT + OPPONENT_FEATURE_COUNT + CORRECTED_TAIL_FEATURE_COUNT
            == FEATURE_COUNT
        ),
        "opponent_rows_byte_identical_after_remap": opponent_identical,
        "corrected_tail_all_ieee754_signed_zero": tail_signed_zero,
    }

    source_downstream_offset = source_w1_offset + w1_bytes
    corrected_downstream_offset = corrected_w1_offset + w1_bytes
    tensor_audits: list[dict[str, Any]] = []
    source_cursor = source_downstream_offset
    corrected_cursor = corrected_downstream_offset
    for tensor in tensors:
        identical = _regions_equal(
            source,
            source_cursor,
            corrected,
            corrected_cursor,
            tensor.byte_count,
        )
        tensor_audits.append(
            {
                "name": tensor.name,
                "float_count": tensor.float_count,
                "bytes": tensor.byte_count,
                "source_blake3": _hash_region(source, source_cursor, tensor.byte_count),
                "corrected_blake3": _hash_region(
                    corrected,
                    corrected_cursor,
                    tensor.byte_count,
                ),
                "byte_identical": identical,
            }
        )
        source_cursor += tensor.byte_count
        corrected_cursor += tensor.byte_count
    downstream_checks = {
        "all_downstream_tensors_byte_identical": all(
            tensor["byte_identical"] for tensor in tensor_audits
        ),
        "source_downstream_consumes_exact_file": source_cursor == source_identity["bytes"],
        "corrected_downstream_consumes_exact_file": (
            corrected_cursor == corrected_identity["bytes"]
        ),
    }

    checks = {
        **header_checks,
        **size_checks,
        **identity_checks,
        **first_layer_checks,
        **downstream_checks,
    }
    verdict, failure_reasons = _checks_with_reasons(checks)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "experiment_id": "corrected-mid-tail-v1",
        "gate": 12,
        "verdict": verdict,
        "failure_reasons": failure_reasons,
        "contract": {
            "historical_schema_id": HISTORICAL_SCHEMA_ID,
            "corrected_schema_id": SCHEMA_ID,
            "historical_magic": HISTORICAL_MAGIC.decode(),
            "corrected_magic": CORRECTED_MAGIC.decode(),
            "corrected_container_version": CORRECTED_CONTAINER_VERSION,
            "corrected_schema_tag_hex": CORRECTED_SCHEMA_TAG.hex(),
            "feature_count": FEATURE_COUNT,
            "hidden1": expected_hidden1,
            "hidden2": expected_hidden2,
            "row_bytes": row_bytes,
            "head_format_version": source_version,
        },
        "source": {
            **source_identity,
            "header_hex": source_header.hex(),
            "magic": source_header[:4].decode(errors="replace"),
            "head_format_version": source_version,
        },
        "corrected": {
            **corrected_identity,
            "header_hex": corrected_header.raw.hex(),
            "magic": corrected_header.raw[:4].decode(errors="replace"),
            "container_version": corrected_header.container_version,
            "head_format_version": corrected_header.head_format_version,
            "schema_tag_hex": corrected_header.schema_tag.hex(),
            "feature_count": corrected_header.feature_count,
            "hidden1": corrected_header.hidden1,
            "hidden2": corrected_header.hidden2,
        },
        "first_layer": {
            "source_base": {
                "range": [0, BASE_FEATURE_COUNT],
                "bytes": base_bytes,
                "blake3": _hash_region(source, source_w1_offset, base_bytes),
                "destination_range": [0, BASE_FEATURE_COUNT],
                "byte_identical": base_identical,
            },
            "source_historical_defect": {
                "range": [
                    HISTORICAL_DEFECT_BASE,
                    HISTORICAL_DEFECT_BASE + DEFECT_FEATURE_COUNT,
                ],
                "bytes": defect_bytes,
                "blake3": _hash_region(source, source_defect_offset, defect_bytes),
                "destination_range": None,
                "discarded": first_layer_checks["historical_defect_rows_discarded"],
                "value_classification": _signed_zero_counts(
                    source,
                    source_defect_offset,
                    DEFECT_FEATURE_COUNT * expected_hidden1,
                ),
            },
            "source_opponent_detail": {
                "range": [
                    HISTORICAL_OPPONENT_BASE,
                    HISTORICAL_OPPONENT_BASE + OPPONENT_FEATURE_COUNT,
                ],
                "bytes": opponent_bytes,
                "blake3": _hash_region(source, source_opponent_offset, opponent_bytes),
                "destination_range": [
                    CORRECTED_OPPONENT_BASE,
                    CORRECTED_OPPONENT_BASE + OPPONENT_FEATURE_COUNT,
                ],
                "byte_identical": opponent_identical,
            },
            "corrected_tail": {
                "range": [
                    CORRECTED_TAIL_BASE,
                    CORRECTED_TAIL_BASE + CORRECTED_TAIL_FEATURE_COUNT,
                ],
                "bytes": corrected_tail_bytes,
                "blake3": _hash_region(corrected, corrected_tail_offset, corrected_tail_bytes),
                "signed_zero_counts": tail_zero_counts,
                "all_ieee754_signed_zero": tail_signed_zero,
            },
        },
        "downstream": {
            "source_offset": source_downstream_offset,
            "corrected_offset": corrected_downstream_offset,
            "bytes": downstream_bytes,
            "source_blake3": _hash_region(
                source,
                source_downstream_offset,
                downstream_bytes,
            ),
            "corrected_blake3": _hash_region(
                corrected,
                corrected_downstream_offset,
                downstream_bytes,
            ),
            "all_byte_identical": downstream_checks["all_downstream_tensors_byte_identical"],
            "tensors": tensor_audits,
        },
        "checks": checks,
    }


def _production_expected_source() -> dict[str, Any]:
    return {
        "bytes": PRODUCTION_SOURCE_BYTES,
        "blake3": PRODUCTION_SOURCE_BLAKE3,
        "sha256": PRODUCTION_SOURCE_SHA256,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-label", default="nnue_weights_v4opp_modal_iter3.bin")
    parser.add_argument("--corrected-label", default="corrected-content-addressed-model")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_migration(
            args.source,
            args.corrected,
            source_label=args.source_label,
            corrected_label=args.corrected_label,
        )
    except (AuditError, OSError) as error:
        failure = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_id": AUDIT_ID,
            "experiment_id": "corrected-mid-tail-v1",
            "gate": 12,
            "verdict": "error",
            "failure_reasons": [str(error)],
            "expected_source": _production_expected_source(),
        }
        write_json_atomic(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
