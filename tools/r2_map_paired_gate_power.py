#!/usr/bin/env python3
"""Deterministic power and MDE calculations for the frozen R2-MAP paired gate.

The tool never discovers pilot reports.  Calibration is either supplied as an
explicit JSON file of paired focal-score deltas or is reported as a provisional
sensitivity analysis over explicitly supplied standard deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

SCHEMA_ID = "cascadia.r2-map.paired-gate-power.v1"
PROTOCOL_ID = "r2-map-focal-paired-v1"
SMOKE_PAIRS = 20
DEVELOPMENT_PAIRS = 250
ALPHA_TWO_SIDED = 0.05
TARGET_POWER = 0.80
SMOKE_STRENGTH_BLINDED = True
OUTCOME_DRIVEN_EXTENSION_ALLOWED = False


class PowerAnalysisError(ValueError):
    """The requested analysis would violate the preregistered contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise PowerAnalysisError(f"{label} must be finite and positive")
    return value


def load_explicit_paired_deltas(path: Path) -> tuple[list[float], str]:
    """Load only the named pilot file; no directory or repository search occurs."""

    raw = path.read_bytes()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PowerAnalysisError(f"pilot JSON is malformed: {error}") from error
    values = decoded.get("paired_deltas") if isinstance(decoded, dict) else decoded
    if not isinstance(values, list) or len(values) < 2:
        raise PowerAnalysisError("pilot JSON must contain at least two paired_deltas")
    deltas: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PowerAnalysisError(f"paired_deltas[{index}] is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PowerAnalysisError(f"paired_deltas[{index}] is not finite")
        deltas.append(numeric)
    return deltas, _sha256_bytes(raw)


def normal_approximation(standard_deviation: float, effects: Sequence[float]) -> dict[str, Any]:
    """Return fixed-N MDE and fixed-effect N under a paired normal approximation."""

    sd = _finite_positive(standard_deviation, "standard deviation")
    z_alpha = NormalDist().inv_cdf(1.0 - ALPHA_TWO_SIDED / 2.0)
    z_power = NormalDist().inv_cdf(TARGET_POWER)
    multiplier = z_alpha + z_power
    mde = multiplier * sd / math.sqrt(DEVELOPMENT_PAIRS)
    required = []
    for effect in effects:
        effect = _finite_positive(effect, "configured effect")
        required.append(
            {
                "effect": effect,
                "required_pairs": math.ceil((multiplier * sd / effect) ** 2),
            }
        )
    return {
        "standard_deviation": sd,
        "mde_at_250_pairs": mde,
        "required_pairs_by_effect": required,
    }


def build_analysis(
    *,
    effects: Sequence[float],
    paired_deltas: Sequence[float] | None = None,
    pilot_source_id: str | None = None,
    pilot_source_sha256: str | None = None,
    sensitivity_standard_deviations: Sequence[float] = (),
) -> dict[str, Any]:
    configured_effects = sorted({_finite_positive(value, "configured effect") for value in effects})
    if not configured_effects:
        raise PowerAnalysisError("at least one configured effect is required")
    if paired_deltas is not None and sensitivity_standard_deviations:
        raise PowerAnalysisError("pilot calibration and sensitivity calibration are exclusive")
    if paired_deltas is None and not sensitivity_standard_deviations:
        raise PowerAnalysisError(
            "supply an explicit pilot JSON or at least one sensitivity standard deviation"
        )

    observed: dict[str, Any]
    scenarios: list[dict[str, Any]]
    if paired_deltas is not None:
        deltas = [float(value) for value in paired_deltas]
        if len(deltas) < 2 or not all(math.isfinite(value) for value in deltas):
            raise PowerAnalysisError("explicit pilot requires at least two finite paired deltas")
        if not pilot_source_id or not pilot_source_sha256:
            raise PowerAnalysisError("explicit pilot identity and SHA-256 are required")
        observed_sd = statistics.stdev(deltas)
        if observed_sd <= 0.0:
            raise PowerAnalysisError("explicit pilot paired deltas have zero variance")
        observed = {
            "available": True,
            "pair_count": len(deltas),
            "mean_paired_delta": statistics.fmean(deltas),
            "standard_deviation": observed_sd,
            "source_id": pilot_source_id,
            "source_sha256": pilot_source_sha256,
        }
        scenarios = [
            {
                "scenario_id": "explicit-pilot-observed-sd",
                **normal_approximation(observed_sd, configured_effects),
            }
        ]
        calibration_status = "calibrated-from-explicit-compatible-pilot"
    else:
        observed = {
            "available": False,
            "pair_count": 0,
            "mean_paired_delta": None,
            "standard_deviation": None,
            "reason": "no scientifically compatible focal-seat paired pilot was supplied",
        }
        scenarios = [
            {
                "scenario_id": f"sensitivity-sd-{float(sd):g}",
                **normal_approximation(float(sd), configured_effects),
            }
            for sd in sorted(
                {
                    _finite_positive(value, "sensitivity standard deviation")
                    for value in sensitivity_standard_deviations
                }
            )
        ]
        calibration_status = "provisional-sensitivity-until-first-eligible-pilot"

    result = {
        "schema_id": SCHEMA_ID,
        "protocol_id": PROTOCOL_ID,
        "calibration_status": calibration_status,
        "frozen_gate": {
            "strength_blinded_smoke_pairs": SMOKE_PAIRS,
            "strength_outputs_blinded": SMOKE_STRENGTH_BLINDED,
            "development_pairs": DEVELOPMENT_PAIRS,
            "single_fixed_analysis": True,
            "outcome_driven_sample_extension_allowed": OUTCOME_DRIVEN_EXTENSION_ALLOWED,
            "promotion_rule": (
                "paired mean > 0 and two-sided 95% interval excludes 0, with all "
                "preregistered integrity, guardrail, and resource gates passing"
            ),
        },
        "calculation": {
            "method": "paired-mean normal approximation",
            "alpha_two_sided": ALPHA_TWO_SIDED,
            "target_power": TARGET_POWER,
            "configured_effects": configured_effects,
            "assumptions": [
                "paired focal-score deltas are independent across registered pairs",
                "the future paired-delta standard deviation matches the calibration scenario",
                "the normal approximation is adequate at the planned sample size",
                "the configured effect is a fixed alternative chosen before candidate "
                "outcomes are opened",
            ],
            "limitations": [
                "this is a planning calculation, not evidence of model strength",
                "normal critical values do not add a finite-sample Student-t correction",
                "component, tail, integrity, memory, latency, and zero-swap gates can "
                "still prevent promotion",
                "D6 augmentation and multiple decisions within a game do not increase "
                "the benchmark pair count",
            ],
        },
        "observed_pilot": observed,
        "scenarios": scenarios,
    }
    result["analysis_sha256"] = _sha256_bytes(_canonical_json(result))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--effect", type=float, action="append", required=True)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--pilot-json", type=Path)
    source.add_argument("--sensitivity-sd", type=float, action="append")
    result.add_argument(
        "--pilot-source-id",
        help="immutable semantic identity for --pilot-json; required with a pilot",
    )
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.pilot_json is not None:
            if not arguments.pilot_source_id:
                raise PowerAnalysisError("--pilot-source-id is required with --pilot-json")
            deltas, source_hash = load_explicit_paired_deltas(arguments.pilot_json)
            analysis = build_analysis(
                effects=arguments.effect,
                paired_deltas=deltas,
                pilot_source_id=arguments.pilot_source_id,
                pilot_source_sha256=source_hash,
            )
        else:
            if arguments.pilot_source_id:
                raise PowerAnalysisError("--pilot-source-id is valid only with --pilot-json")
            analysis = build_analysis(
                effects=arguments.effect,
                sensitivity_standard_deviations=arguments.sensitivity_sd,
            )
        rendered = json.dumps(analysis, sort_keys=True, indent=2) + "\n"
        if arguments.output is None:
            sys.stdout.write(rendered)
        else:
            arguments.output.write_text(rendered, encoding="utf-8")
    except (OSError, PowerAnalysisError) as error:
        print(f"R2-MAP paired-gate power analysis refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
