#!/usr/bin/env python3
"""Always-valid paired promotion gate for Cascadia V3 expert cycles."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

TIERS = ("direct", "k32-r64", "k32-r600", "equal-wall-time")
MIN_PAIRS = 100
MAX_PAIRS = 500
NULL_DELTA = -0.10
ALTERNATIVE_DELTA = 0.15
ALPHA = 0.05
BETA = 0.05
RAW_DELTA_LOWER = -200.0
RAW_DELTA_UPPER = 200.0
DECISION_DELTA_LOWER = -25.0
DECISION_DELTA_UPPER = 25.0
BET_FRACTIONS = tuple(index / 100 for index in range(1, 100))


def _log_mean_exp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values) / len(values))


def _e_value(values: list[float], boundary: float, direction: int) -> float:
    scale = max(
        abs(DECISION_DELTA_LOWER - boundary),
        abs(DECISION_DELTA_UPPER - boundary),
    )
    logs = []
    for fraction in BET_FRACTIONS:
        total = 0.0
        for value in values:
            normalized = direction * (value - boundary) / scale
            total += math.log1p(fraction * normalized)
        logs.append(total)
    return math.exp(min(_log_mean_exp(logs), 700.0))


def _decision_delta(value: float) -> float:
    return min(DECISION_DELTA_UPPER, max(DECISION_DELTA_LOWER, value))


def evaluate(records: list[dict[str, Any]]) -> dict[str, object]:
    raw_values: dict[str, list[float]] = defaultdict(list)
    decision_values: dict[str, list[float]] = defaultdict(list)
    integrity = True
    resources = True
    seen: dict[str, set[int]] = defaultdict(set)
    for record in records:
        tier = record.get("tier")
        pair_index = record.get("pair_index")
        delta = record.get("paired_delta")
        if tier not in TIERS or not isinstance(pair_index, int) or pair_index < 0:
            raise ValueError("promotion record has an invalid tier or pair index")
        if pair_index in seen[tier]:
            raise ValueError(f"duplicate pair index {pair_index} in {tier}")
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError("promotion paired delta must be finite")
        if not RAW_DELTA_LOWER <= float(delta) <= RAW_DELTA_UPPER:
            raise ValueError("promotion delta is outside the registered score bound")
        if len(raw_values[tier]) >= MAX_PAIRS:
            raise ValueError(f"{tier} exceeds the registered 500-pair maximum")
        seen[tier].add(pair_index)
        raw = float(delta)
        raw_values[tier].append(raw)
        decision_values[tier].append(_decision_delta(raw))
        integrity &= record.get("integrity_passed") is True
        resources &= record.get("resource_regression") is False

    tiers: dict[str, dict[str, object]] = {}
    for tier in TIERS:
        tier_values = decision_values[tier]
        tier_raw_values = raw_values[tier]
        null_e = _e_value(tier_values, NULL_DELTA, +1) if tier_values else 1.0
        alternative_e = _e_value(tier_values, ALTERNATIVE_DELTA, -1) if tier_values else 1.0
        enough = len(tier_values) >= MIN_PAIRS
        if enough and null_e >= 1 / ALPHA:
            boundary = "alternative"
        elif enough and alternative_e >= 1 / BETA:
            boundary = "null"
        elif len(tier_values) == MAX_PAIRS:
            boundary = "inconclusive-maximum"
        else:
            boundary = "continue"
        tiers[tier] = {
            "pairs": len(tier_values),
            "mean_delta": (
                sum(tier_raw_values) / len(tier_raw_values) if tier_raw_values else None
            ),
            "decision_mean_delta": (
                sum(tier_values) / len(tier_values) if tier_values else None
            ),
            "null_e_value": null_e,
            "alternative_e_value": alternative_e,
            "boundary": boundary,
        }

    boundaries = [tiers[tier]["boundary"] for tier in TIERS]
    if not integrity or not resources:
        verdict = "retain-incumbent-resource-or-integrity-regression"
    elif all(boundary == "alternative" for boundary in boundaries):
        verdict = "promote"
    elif any(boundary == "null" for boundary in boundaries):
        verdict = "retain-incumbent"
    elif all(boundary != "continue" for boundary in boundaries):
        verdict = "retain-incumbent-inconclusive"
    else:
        verdict = "continue"
    return {
        "schema_id": "cascadia-v3-always-valid-promotion-v2",
        "verdict": verdict,
        "integrity_passed": integrity,
        "resource_regression_absent": resources,
        "registered": {
            "minimum_pairs": MIN_PAIRS,
            "maximum_pairs": MAX_PAIRS,
            "null_delta": NULL_DELTA,
            "alternative_delta": ALTERNATIVE_DELTA,
            "alpha": ALPHA,
            "beta": BETA,
            "raw_delta_bounds": [RAW_DELTA_LOWER, RAW_DELTA_UPPER],
            "decision_delta_bounds": [DECISION_DELTA_LOWER, DECISION_DELTA_UPPER],
            "decision_estimand": "paired-score-delta-winsorized-at-plus-or-minus-25",
            "test": "bounded-mixture-betting-e-process",
        },
        "tiers": tiers,
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    result = evaluate(records)
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
