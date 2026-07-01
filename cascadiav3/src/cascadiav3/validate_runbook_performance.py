"""Validate required v3 runbook and benchmark performance fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for candidate_key, value in payload.items():
            if candidate_key == key:
                values.append(value)
            values.extend(_find_values(value, key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_find_values(item, key))
    return values


def _first_number(payload: dict[str, Any], key: str) -> float | None:
    for value in _find_values(payload, key):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def validate_required_positive(payload: dict[str, Any], fields: list[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    missing: list[str] = []
    nonpositive: list[str] = []
    for field in fields:
        value = _first_number(payload, field)
        if value is None:
            missing.append(field)
        elif value <= 0.0:
            nonpositive.append(field)
        else:
            found[field] = value
    if missing or nonpositive:
        raise AssertionError(f"performance fields invalid missing={missing} nonpositive={nonpositive}")
    return found


def validate_time_ratio(payload: dict[str, Any], max_ratio: float) -> float:
    ratio = _first_number(payload, "treatment_control_time_ratio")
    if ratio is None:
        treatment = _first_number(payload, "treatment_mean_decision_seconds")
        control = _first_number(payload, "control_mean_decision_seconds")
        if treatment is not None and control is not None and control > 0:
            ratio = treatment / control
    if ratio is None:
        raise AssertionError("benchmark report missing treatment/control timing ratio")
    if ratio > max_ratio:
        raise AssertionError(f"treatment/control time ratio {ratio:.4f} > {max_ratio:.4f}")
    return ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runbook")
    parser.add_argument("--benchmark")
    parser.add_argument("--require-positive", default="")
    parser.add_argument("--max-treatment-control-time-ratio", type=float)
    args = parser.parse_args()
    if not args.runbook and not args.benchmark:
        parser.error("one of --runbook or --benchmark is required")
    path = Path(args.runbook or args.benchmark)
    payload = _load(path)
    report: dict[str, Any] = {"status": "pass", "path": str(path)}
    if args.require_positive:
        fields = [field.strip() for field in args.require_positive.split(",") if field.strip()]
        report["positive_fields"] = validate_required_positive(payload, fields)
    if args.max_treatment_control_time_ratio is not None:
        report["treatment_control_time_ratio"] = validate_time_ratio(
            payload,
            args.max_treatment_control_time_ratio,
        )
        report["max_treatment_control_time_ratio"] = args.max_treatment_control_time_ratio
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
