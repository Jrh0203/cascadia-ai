"""Validate expert-root score decomposition and score-to-go semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl
from .schema import validate_score_decomposition


def validate_roots(path: Path, *, tolerance: float = 1.0e-5) -> dict[str, Any]:
    records = read_replay_jsonl(path)
    checked_actions = 0
    failures: list[dict[str, Any]] = []
    for root_index, record in enumerate(records):
        try:
            validate_score_decomposition(record)
        except Exception as exc:  # pragma: no cover - surfaced in report.
            failures.append({"root_index": root_index, "field": "score_decomposition", "error": str(exc)})
        q_values = record["per_action_Q"]
        scores = record.get("exact_afterstate_score_active", [0.0] * len(q_values))
        to_go = record.get("per_action_score_to_go", q_values)
        valid = record.get("per_action_Q_valid", [True] * len(q_values))
        for action_index, is_valid in enumerate(valid):
            if not is_valid:
                continue
            checked_actions += 1
            lhs = float(scores[action_index]) + float(to_go[action_index])
            rhs = float(q_values[action_index])
            if abs(lhs - rhs) > tolerance:
                failures.append(
                    {
                        "root_index": root_index,
                        "action_index": action_index,
                        "afterstate_plus_score_to_go": lhs,
                        "q": rhs,
                    }
                )
    return {
        "status": "pass" if not failures else "fail",
        "roots": len(records),
        "valid_q_targets_checked": checked_actions,
        "tolerance": tolerance,
        "failures": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    report = validate_roots(Path(args.roots), tolerance=args.tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
