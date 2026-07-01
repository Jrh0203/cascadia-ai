from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from r2_map_paired_gate_power import (
    DEVELOPMENT_PAIRS,
    PowerAnalysisError,
    build_analysis,
    load_explicit_paired_deltas,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_ANALYSIS = REPOSITORY / "docs/v2/reports/r2-map-paired-gate-power-v1.json"


def test_provisional_analysis_freezes_nonadaptive_gate() -> None:
    report = build_analysis(
        effects=[1.0, 0.5, 0.25],
        sensitivity_standard_deviations=[6.0, 2.0, 4.0],
    )
    assert report["calibration_status"].startswith("provisional")
    assert report["observed_pilot"]["standard_deviation"] is None
    assert report["frozen_gate"] == {
        "strength_blinded_smoke_pairs": 20,
        "strength_outputs_blinded": True,
        "development_pairs": 250,
        "single_fixed_analysis": True,
        "outcome_driven_sample_extension_allowed": False,
        "promotion_rule": (
            "paired mean > 0 and two-sided 95% interval excludes 0, with all "
            "preregistered integrity, guardrail, and resource gates passing"
        ),
    }
    assert [item["standard_deviation"] for item in report["scenarios"]] == [2.0, 4.0, 6.0]
    assert DEVELOPMENT_PAIRS == 250


def test_repository_power_artifact_is_exact_regeneration() -> None:
    report = build_analysis(
        effects=[0.25, 0.50, 0.75, 1.00],
        sensitivity_standard_deviations=[2, 3, 4, 5, 6],
    )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    assert FROZEN_ANALYSIS.read_text(encoding="utf-8") == rendered


def test_explicit_pilot_reports_observed_sd_mde_and_required_n(tmp_path: Path) -> None:
    pilot = tmp_path / "named-pilot.json"
    pilot.write_text(json.dumps({"paired_deltas": [-1, 0, 1, 2]}), encoding="utf-8")
    deltas, digest = load_explicit_paired_deltas(pilot)
    report = build_analysis(
        effects=[0.5],
        paired_deltas=deltas,
        pilot_source_id="eligible-focal-pilot-v1",
        pilot_source_sha256=digest,
    )
    observed_sd = math.sqrt(5.0 / 3.0)
    assert report["observed_pilot"]["standard_deviation"] == pytest.approx(observed_sd)
    scenario = report["scenarios"][0]
    assert scenario["mde_at_250_pairs"] == pytest.approx(0.2287484751766896)
    assert scenario["required_pairs_by_effect"] == [{"effect": 0.5, "required_pairs": 53}]


def test_analysis_never_falls_back_to_repository_discovery(tmp_path: Path) -> None:
    (tmp_path / "tempting-accepted-pilot.json").write_text(
        json.dumps({"paired_deltas": [1, 2, 3]}), encoding="utf-8"
    )
    with pytest.raises(PowerAnalysisError, match="supply an explicit pilot"):
        build_analysis(effects=[0.5])


def test_zero_variance_or_mixed_calibration_is_rejected() -> None:
    with pytest.raises(PowerAnalysisError, match="zero variance"):
        build_analysis(
            effects=[0.5],
            paired_deltas=[1, 1],
            pilot_source_id="bad",
            pilot_source_sha256="a" * 64,
        )
    with pytest.raises(PowerAnalysisError, match="exclusive"):
        build_analysis(
            effects=[0.5],
            paired_deltas=[0, 1],
            pilot_source_id="mixed",
            pilot_source_sha256="b" * 64,
            sensitivity_standard_deviations=[2.0],
        )
