"""Merge wildlife catalog shards without provenance or receipt checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in row["counts"])


def _score(row: dict[str, Any]) -> float:
    optimum = row.get("optimum")
    if isinstance(optimum, dict):
        for key in ("score", "total", "optimum"):
            if key in optimum:
                return float(optimum[key])
    if isinstance(optimum, (int, float)):
        return float(optimum)
    return float("-inf")


def _prefer(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_proved = bool(candidate.get("proof_complete"))
    current_proved = bool(current.get("proof_complete"))
    if candidate_proved != current_proved:
        return candidate_proved
    return _score(candidate) > _score(current)


def _strip_audit(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_audit(item) for item in value]
    if not isinstance(value, dict):
        return value
    clean = {}
    for key, item in value.items():
        lowered = key.lower()
        if (
            "sha256" in lowered
            or lowered.endswith("_hash")
            or lowered in {"proof_provenance", "imported_ledgers"}
        ):
            continue
        clean[key] = _strip_audit(item)
    return clean


def merge_catalogs(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one catalog is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows: dict[tuple[int, ...], dict[str, Any]] = {}
    for payload in payloads:
        for raw_row in payload.get("results", []):
            row = _strip_audit(raw_row)
            key = _key(row)
            if key not in rows or _prefer(row, rows[key]):
                rows[key] = row
    result = _strip_audit(payloads[0])
    ordered = [rows[key] for key in sorted(rows)]
    expected_count = max(
        (
            int(payload.get("allocation_count", len(payload.get("results", []))))
            for payload in payloads
        ),
        default=len(ordered),
    )
    completed_count = sum(bool(row.get("proof_complete")) for row in ordered)
    result["results"] = ordered
    result["completed_count"] = completed_count
    result["allocation_count"] = expected_count
    result["proof_complete"] = (
        bool(ordered)
        and len(ordered) == expected_count
        and completed_count == expected_count
    )
    result["merged_inputs"] = [str(path) for path in paths]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge_catalogs(args.catalogs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": result["allocation_count"],
                "proved": result["completed_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
