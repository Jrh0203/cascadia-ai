from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import blake3
import pytest

MODULE_PATH = Path(__file__).with_name("corrected_mid_tail_champion_audit.py")
SPEC = importlib.util.spec_from_file_location("corrected_mid_tail_champion_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _u32_rows(rows: int, hidden1: int, *, seed: int) -> bytes:
    values = ((seed + index * 0x9E37_79B9) & 0xFFFF_FFFF for index in range(rows * hidden1))
    return b"".join(struct.pack("<I", value) for value in values)


def _tensor_payload(version: int, hidden1: int, hidden2: int) -> bytes:
    float_count = sum(
        tensor.float_count for tensor in audit.downstream_tensors(version, hidden1, hidden2)
    )
    return _u32_rows(float_count, 1, seed=0x1357_2468)


def _fixture(
    tmp_path: Path,
    *,
    hidden1: int = 2,
    hidden2: int = 1,
    version: int = 4,
    negative_tail_zero: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source.bin"
    corrected = tmp_path / "corrected.bin"
    base = _u32_rows(audit.BASE_FEATURE_COUNT, hidden1, seed=1)
    defect = _u32_rows(audit.DEFECT_FEATURE_COUNT, hidden1, seed=2)
    opponent = _u32_rows(audit.OPPONENT_FEATURE_COUNT, hidden1, seed=3)
    downstream = _tensor_payload(version, hidden1, hidden2)
    source.write_bytes(
        audit.HISTORICAL_MAGIC + struct.pack("<I", version) + base + defect + opponent + downstream
    )
    zero_word = struct.pack("<I", 0x8000_0000 if negative_tail_zero else 0)
    corrected.write_bytes(
        audit.CORRECTED_MAGIC
        + struct.pack("<II", audit.CORRECTED_CONTAINER_VERSION, version)
        + audit.CORRECTED_SCHEMA_TAG
        + struct.pack("<III", audit.FEATURE_COUNT, hidden1, hidden2)
        + base
        + opponent
        + zero_word * (audit.CORRECTED_TAIL_FEATURE_COUNT * hidden1)
        + downstream
    )
    source_bytes = source.read_bytes()
    expected = {
        "expected_source_bytes": len(source_bytes),
        "expected_source_blake3": blake3.blake3(source_bytes).hexdigest(),
        "expected_source_sha256": __import__("hashlib").sha256(source_bytes).hexdigest(),
        "expected_hidden1": hidden1,
        "expected_hidden2": hidden2,
    }
    return source, corrected, expected


def _run(source: Path, corrected: Path, expected: dict[str, object]) -> dict[str, object]:
    return audit.audit_migration(
        source,
        corrected,
        source_label="source.bin",
        corrected_label="corrected.bin",
        **expected,
    )


def test_exact_mapping_and_all_downstream_tensors_pass(tmp_path: Path) -> None:
    source, corrected, expected = _fixture(tmp_path)
    report = _run(source, corrected, expected)
    assert report["verdict"] == "pass"
    assert report["first_layer"]["source_historical_defect"]["discarded"] is True
    assert report["first_layer"]["source_opponent_detail"]["byte_identical"] is True
    assert report["first_layer"]["corrected_tail"]["signed_zero_counts"]["nonzero"] == 0
    assert report["downstream"]["all_byte_identical"] is True
    assert [tensor["name"] for tensor in report["downstream"]["tensors"]] == [
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "w3_policy",
        "b3_policy",
        "w3_wildlife",
        "b3_wildlife",
        "w3_habitat",
        "b3_habitat",
        "w3_heads",
        "b3_heads",
        "w3_var",
        "b3_var",
    ]


def test_negative_zero_tail_is_valid_signed_zero(tmp_path: Path) -> None:
    source, corrected, expected = _fixture(tmp_path, negative_tail_zero=True)
    report = _run(source, corrected, expected)
    counts = report["first_layer"]["corrected_tail"]["signed_zero_counts"]
    assert report["verdict"] == "pass"
    assert counts["negative_zero"] == audit.CORRECTED_TAIL_FEATURE_COUNT * 2
    assert counts["positive_zero"] == 0


@pytest.mark.parametrize("block", ["base", "opponent", "tail", "downstream"])
def test_any_migration_payload_difference_fails_closed(tmp_path: Path, block: str) -> None:
    source, corrected, expected = _fixture(tmp_path)
    payload = bytearray(corrected.read_bytes())
    row_bytes = 2 * audit.FLOAT_BYTES
    offsets = {
        "base": 40,
        "opponent": 40 + audit.CORRECTED_OPPONENT_BASE * row_bytes,
        "tail": 40 + audit.CORRECTED_TAIL_BASE * row_bytes,
        "downstream": 40 + audit.FEATURE_COUNT * row_bytes,
    }
    payload[offsets[block]] ^= 1
    corrected.write_bytes(payload)
    report = _run(source, corrected, expected)
    assert report["verdict"] == "fail"
    if block == "base":
        assert report["checks"]["base_rows_byte_identical"] is False
    elif block == "opponent":
        assert report["checks"]["opponent_rows_byte_identical_after_remap"] is False
    elif block == "tail":
        assert report["checks"]["corrected_tail_all_ieee754_signed_zero"] is False
    else:
        assert report["checks"]["all_downstream_tensors_byte_identical"] is False


def test_bad_header_and_trailing_byte_are_rejected(tmp_path: Path) -> None:
    source, corrected, expected = _fixture(tmp_path)
    bad_header = bytearray(corrected.read_bytes())
    bad_header[12] ^= 0xFF
    corrected.write_bytes(bad_header)
    report = _run(source, corrected, expected)
    assert report["verdict"] == "fail"
    assert report["checks"]["corrected_schema_tag_exact"] is False

    source, corrected, expected = _fixture(tmp_path)
    corrected.write_bytes(corrected.read_bytes() + b"\0")
    report = _run(source, corrected, expected)
    assert report["verdict"] == "fail"
    assert report["checks"]["corrected_has_no_trailing_bytes"] is False


def test_source_identity_drift_fails_even_when_layout_is_valid(tmp_path: Path) -> None:
    source, corrected, expected = _fixture(tmp_path)
    expected["expected_source_blake3"] = "0" * 64
    report = _run(source, corrected, expected)
    assert report["verdict"] == "fail"
    assert report["checks"]["source_production_blake3_exact"] is False


def test_non_nnue_source_is_rejected(tmp_path: Path) -> None:
    source, corrected, expected = _fixture(tmp_path)
    payload = bytearray(source.read_bytes())
    payload[:4] = b"NOPE"
    source.write_bytes(payload)
    with pytest.raises(audit.AuditError, match="NNUE magic"):
        _run(source, corrected, expected)
