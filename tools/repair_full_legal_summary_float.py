#!/usr/bin/env python3
"""Repair a JSON float spelling that fails the frozen Rust round-trip validator.

The full-legal audit summary is derived from immutable game records. A shortest
decimal emitted by serde_json can, for rare values, parse one ULP differently
when the frozen binary reads the file back. This tool changes only the decimal
spelling of one derived summary value, binds the repair to the exact f64 bits
reported by the frozen Rust recomputation, and requires the validator to accept
the repaired candidate before replacing the artifact.
"""

from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SUMMARY_SELECTORS = {
    "mean_champion_regret": lambda decision: decision["champion_regret"]["points"],
    "mean_champion_frontier_regret": lambda decision: decision["champion_frontier_regret"][
        "points"
    ],
    "mean_retained_screen_regret": lambda decision: decision["retained_screen_regret"]["points"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f64_bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def f64_from_bits(bits: str) -> float:
    if not re.fullmatch(r"[0-9a-fA-F]{16}", bits):
        raise ValueError("target f64 bits must be exactly 16 hexadecimal digits")
    value = struct.unpack(">d", int(bits, 16).to_bytes(8, "big"))[0]
    if not math.isfinite(value):
        raise ValueError("target f64 bits must encode a finite value")
    return value


def exact_decimal(value: float) -> str:
    return format(decimal.Decimal.from_float(value), "f")


def candidate_decimals(value: float) -> list[str]:
    target_bits = f64_bits(value)
    candidates: list[str] = []
    for precision in range(17, 35):
        candidate = format(value, f".{precision}g")
        if f64_bits(float(candidate)) == target_bits and candidate not in candidates:
            candidates.append(candidate)
    exact = exact_decimal(value)
    if exact not in candidates:
        candidates.append(exact)
    return candidates


def recompute_summary_value(report: dict[str, Any], field: str) -> float:
    decisions = [decision for game in report["games"] for decision in game["decisions"]]
    if not decisions:
        return 0.0
    selector = SUMMARY_SELECTORS[field]
    return sum(float(selector(decision)) for decision in decisions) / len(decisions)


def find_unique_token(path: Path, token: bytes) -> int:
    first = -1
    occurrences = 0
    offset = 0
    overlap = max(len(token) - 1, 0)
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            window = tail + chunk
            search_from = 0
            while True:
                found = window.find(token, search_from)
                if found < 0:
                    break
                absolute = offset - len(tail) + found
                if first < 0:
                    first = absolute
                occurrences += 1
                search_from = found + 1
            offset += len(chunk)
            tail = window[-overlap:] if overlap else b""
    if occurrences != 1:
        raise ValueError(f"expected exactly one raw token occurrence, found {occurrences}")
    return first


def copy_range(source: Any, target: Any, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("source ended while copying a bounded range")
        target.write(chunk)
        remaining -= len(chunk)


def write_repaired_candidate(
    source_path: Path,
    candidate_path: Path,
    *,
    position: int,
    old_token: bytes,
    new_token: bytes,
) -> None:
    with source_path.open("rb") as source, candidate_path.open("wb") as target:
        copy_range(source, target, position)
        observed = source.read(len(old_token))
        if observed != old_token:
            raise ValueError("source token changed after its position was located")
        target.write(new_token)
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    shutil.copymode(source_path, candidate_path)


def run_validator(validator: Path, input_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(validator), "validate", "--input", str(input_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def repair_summary_float(
    *,
    input_path: Path,
    field: str,
    stored_decimal: str,
    target_f64_bits: str,
    replacement_decimal: str | None,
    validator: Path,
    expected_original_error: str,
    evidence_path: Path,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    validator = validator.resolve()
    evidence_path = evidence_path.resolve()
    if field not in SUMMARY_SELECTORS:
        raise ValueError(f"unsupported derived summary field: {field}")
    target_value = f64_from_bits(target_f64_bits)
    target_f64_bits = target_f64_bits.lower()
    replacements = (
        [replacement_decimal]
        if replacement_decimal is not None
        else candidate_decimals(target_value)
    )
    replacements = [candidate for candidate in replacements if candidate != stored_decimal]
    if not replacements:
        raise ValueError("no replacement decimal spelling differs from the stored spelling")
    if not math.isfinite(float(stored_decimal)):
        raise ValueError("stored decimal must be finite")
    for candidate in replacements:
        parsed = float(candidate)
        if not math.isfinite(parsed):
            raise ValueError("replacement decimal must be finite")
        if f64_bits(parsed) != target_f64_bits:
            raise ValueError(
                f"replacement {parsed!r} encodes {f64_bits(parsed)}, "
                f"not target f64 bits {target_f64_bits}"
            )

    original_validation = run_validator(validator, input_path)
    original_output = original_validation.stdout + original_validation.stderr
    if original_validation.returncode == 0:
        raise ValueError("the original artifact already passes the frozen validator")
    if expected_original_error not in original_output:
        raise ValueError("the original validator failure did not match the expected invariant")
    mismatch_pattern = re.compile(
        rf"{re.escape(field)} stored=.*?\([0-9a-f]{{16}}\) "
        rf"recomputed=.*?\(({re.escape(target_f64_bits)})\)"
    )
    if mismatch_pattern.search(original_output.lower()) is None:
        raise ValueError(
            "the original validator failure did not report the requested field "
            f"with recomputed f64 bits {target_f64_bits}"
        )

    with input_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    summary_value = float(report["summary"][field])
    recomputed_value = recompute_summary_value(report, field)
    del report

    old_token = f'"{field}":{stored_decimal}'.encode()
    position = find_unique_token(input_path, old_token)
    original_sha256 = sha256_file(input_path)
    original_bytes = input_path.stat().st_size

    candidate_fd, candidate_name = tempfile.mkstemp(
        prefix=f".{input_path.name}.repair-",
        suffix=".json",
        dir=input_path.parent,
    )
    os.close(candidate_fd)
    candidate_path = Path(candidate_name)
    validation_attempts = []
    repaired_validation: subprocess.CompletedProcess[str] | None = None
    selected_replacement: str | None = None
    try:
        for candidate in replacements:
            new_token = f'"{field}":{candidate}'.encode()
            write_repaired_candidate(
                input_path,
                candidate_path,
                position=position,
                old_token=old_token,
                new_token=new_token,
            )
            attempted_validation = run_validator(validator, candidate_path)
            validation_attempts.append(
                {
                    "decimal": candidate,
                    "returncode": attempted_validation.returncode,
                    "stdout": attempted_validation.stdout,
                    "stderr": attempted_validation.stderr,
                }
            )
            if attempted_validation.returncode == 0:
                repaired_validation = attempted_validation
                selected_replacement = candidate
                break
        if repaired_validation is None or selected_replacement is None:
            rendered_attempts = "\n".join(
                f"{attempt['decimal']}: {attempt['stdout']}{attempt['stderr']}"
                for attempt in validation_attempts
            )
            raise ValueError(
                "all parser-stable replacement candidates failed the frozen validator:\n"
                f"{rendered_attempts}"
            )
        repaired_sha256 = sha256_file(candidate_path)
        repaired_bytes = candidate_path.stat().st_size
        os.replace(candidate_path, input_path)
        directory_fd = os.open(input_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        candidate_path.unlink(missing_ok=True)

    replacement_value = float(selected_replacement)
    evidence = {
        "schema_version": 2,
        "repair": "derived_summary_float_decimal_normalization",
        "input": str(input_path),
        "field": field,
        "stored_decimal": stored_decimal,
        "replacement_decimal": selected_replacement,
        "python_parsed_summary_f64_bits": f64_bits(summary_value),
        "python_recomputed_f64_bits": f64_bits(recomputed_value),
        "rust_recomputed_target_f64_bits": target_f64_bits,
        "replacement_f64_bits": f64_bits(replacement_value),
        "raw_token_byte_offset": position,
        "validation_attempts": validation_attempts,
        "original": {
            "sha256": original_sha256,
            "bytes": original_bytes,
            "validator_returncode": original_validation.returncode,
            "validator_stdout": original_validation.stdout,
            "validator_stderr": original_validation.stderr,
        },
        "repaired": {
            "sha256": repaired_sha256,
            "bytes": repaired_bytes,
            "validator_returncode": repaired_validation.returncode,
            "validator_stdout": repaired_validation.stdout,
            "validator_stderr": repaired_validation.stderr,
        },
        "semantic_invariant": (
            "Only one decimal spelling in the derived summary changed. The replacement "
            "encodes the exact f64 bits reported by the frozen Rust recomputation, and "
            "the complete artifact passes the frozen validator. Python values are "
            "diagnostic only because its parser and reduction order are not authoritative."
        ),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_evidence = evidence_path.with_suffix(evidence_path.suffix + ".tmp")
    temporary_evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_evidence, evidence_path)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--field", choices=sorted(SUMMARY_SELECTORS), required=True)
    parser.add_argument("--stored-decimal", required=True)
    parser.add_argument("--target-f64-bits", required=True)
    parser.add_argument(
        "--replacement-decimal",
        help="defaults to the exact decimal expansion of --target-f64-bits",
    )
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--expected-original-error", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = repair_summary_float(
        input_path=args.input,
        field=args.field,
        stored_decimal=args.stored_decimal,
        target_f64_bits=args.target_f64_bits,
        replacement_decimal=args.replacement_decimal,
        validator=args.validator,
        expected_original_error=args.expected_original_error,
        evidence_path=args.evidence,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
