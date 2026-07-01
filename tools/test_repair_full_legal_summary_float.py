from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/repair_full_legal_summary_float.py"
SPEC = importlib.util.spec_from_file_location("repair_full_legal_summary_float", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


def write_validator(
    path: Path,
    rejected_decimal: str,
    *,
    field: str = "mean_champion_frontier_regret",
    target_bits: str = "3fc3333333333334",
) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'input="$3"\n'
        f"if grep -q '{field}\":{rejected_decimal},' "
        '"$input"; then\n'
        "  echo 'Invariant(\"stored shard summary does not reproduce from game records: "
        f"{field} stored=0(0000000000000000) recomputed=0({target_bits})\")' >&2\n"
        "  exit 1\n"
        "fi\n"
        'echo "validated $input"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_repairs_one_derived_float_and_records_evidence(tmp_path: Path) -> None:
    stored = "0.15000000000000002"
    replacement = "0.1500000000000000222"
    artifact = tmp_path / "seed.json"
    artifact.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "decisions": [
                            {
                                "champion_regret": {"points": 0.1},
                                "champion_frontier_regret": {"points": 0.1},
                                "retained_screen_regret": {"points": 0.0},
                            },
                            {
                                "champion_regret": {"points": 0.2},
                                "champion_frontier_regret": {"points": 0.2},
                                "retained_screen_regret": {"points": 0.0},
                            },
                        ]
                    }
                ],
                "summary": {
                    "mean_champion_regret": 0.15,
                    "mean_champion_frontier_regret": 0.15000000000000002,
                    "mean_retained_screen_regret": 0.0,
                },
            },
            separators=(",", ":"),
        )
    )
    validator = tmp_path / "validator"
    write_validator(validator, stored)
    evidence_path = tmp_path / "repair.json"

    evidence = repair.repair_summary_float(
        input_path=artifact,
        field="mean_champion_frontier_regret",
        stored_decimal=stored,
        target_f64_bits=repair.f64_bits(float(replacement)),
        replacement_decimal=replacement,
        validator=validator,
        expected_original_error="stored shard summary does not reproduce",
        evidence_path=evidence_path,
    )

    raw = artifact.read_text()
    assert f'"mean_champion_frontier_regret":{stored},' not in raw
    assert f'"mean_champion_frontier_regret":{replacement},' in raw
    assert evidence["original"]["sha256"] != evidence["repaired"]["sha256"]
    assert evidence["repaired"]["bytes"] == evidence["original"]["bytes"] + (
        len(replacement) - len(stored)
    )
    assert evidence["replacement_f64_bits"] == evidence["rust_recomputed_target_f64_bits"]
    assert json.loads(evidence_path.read_text()) == evidence


def test_uses_rust_target_bits_when_python_recompute_differs(tmp_path: Path) -> None:
    stored = "0.15"
    target = 0.15000000000000002
    target_bits = repair.f64_bits(target)
    artifact = tmp_path / "seed.json"
    artifact.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "decisions": [
                            {
                                "champion_regret": {"points": 0.15},
                                "champion_frontier_regret": {"points": 0.15},
                                "retained_screen_regret": {"points": 0.0},
                            },
                        ]
                    }
                ],
                "summary": {
                    "mean_champion_regret": 0.15,
                    "mean_champion_frontier_regret": 0.15,
                    "mean_retained_screen_regret": 0.0,
                },
            },
            separators=(",", ":"),
        )
    )
    validator = tmp_path / "validator"
    write_validator(validator, stored, target_bits=target_bits)

    evidence = repair.repair_summary_float(
        input_path=artifact,
        field="mean_champion_frontier_regret",
        stored_decimal=stored,
        target_f64_bits=target_bits,
        replacement_decimal=None,
        validator=validator,
        expected_original_error="stored shard summary does not reproduce",
        evidence_path=tmp_path / "repair.json",
    )

    assert evidence["python_recomputed_f64_bits"] == repair.f64_bits(0.15)
    assert evidence["replacement_f64_bits"] == target_bits
    assert evidence["replacement_decimal"] != stored
    assert evidence["replacement_decimal"] in artifact.read_text()


def test_rejects_non_unique_raw_token_without_modifying_input(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    token = b'"mean_champion_frontier_regret":0.5'
    path.write_bytes(token + b"\n" + token)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="exactly one"):
        repair.find_unique_token(path, token)

    assert path.read_bytes() == before
