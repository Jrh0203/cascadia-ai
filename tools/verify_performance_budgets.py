#!/usr/bin/env python3
"""Verify Cascadia v2 performance evidence against a versioned budget contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/performance-budgets-v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer}")
    value = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError(f"cannot traverse {pointer} through {type(value).__name__}")
    return value


def gate_passed(actual: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return actual <= threshold
    if operator == ">=":
        return actual >= threshold
    raise ValueError(f"unsupported performance operator: {operator}")


def qualify(config_path: Path, root: Path = ROOT) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported performance budget schema")

    reports: dict[str, dict[str, Any]] = {}
    report_evidence: dict[str, dict[str, str]] = {}
    for report_id, relative_path in config["reports"].items():
        path = (root / relative_path).resolve()
        reports[report_id] = json.loads(path.read_text())
        report_evidence[report_id] = {
            "path": str(path.relative_to(root.resolve())),
            "sha256": sha256(path),
        }

    results = []
    for gate in config["gates"]:
        actual_value = json_pointer(reports[gate["report"]], gate["pointer"])
        if isinstance(actual_value, bool) or not isinstance(actual_value, int | float):
            raise ValueError(f"{gate['id']} did not resolve to a numeric value")
        actual = float(actual_value)
        threshold = float(gate["threshold"])
        results.append(
            {
                "id": gate["id"],
                "report": gate["report"],
                "pointer": gate["pointer"],
                "operator": gate["operator"],
                "threshold": threshold,
                "actual": actual,
                "unit": gate["unit"],
                "passed": gate_passed(actual, gate["operator"], threshold),
            }
        )

    return {
        "schema_version": 1,
        "qualification_id": f"{config['profile_id']}-performance-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_id": config["profile_id"],
        "config": {
            "path": str(config_path.resolve().relative_to(root.resolve())),
            "sha256": sha256(config_path),
        },
        "reports": report_evidence,
        "gates": results,
        "passed": all(result["passed"] for result in results),
    }


def render_markdown(qualification: dict[str, Any]) -> str:
    verdict = "PASS" if qualification["passed"] else "FAIL"
    lines = [
        "# Cascadia V2 Performance Qualification",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Profile: `{qualification['profile_id']}`",
        "",
        "| Gate | Actual | Budget | Result |",
        "|---|---:|---:|---|",
    ]
    for gate in qualification["gates"]:
        lines.append(
            f"| `{gate['id']}` | {gate['actual']:.3f} {gate['unit']} | "
            f"{gate['operator']} {gate['threshold']:.3f} {gate['unit']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"Budget contract SHA-256: `{qualification['config']['sha256']}`",
            "",
        ]
    )
    for report_id, evidence in qualification["reports"].items():
        lines.append(f"- `{report_id}`: `{evidence['path']}` (`{evidence['sha256']}`)")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualification = qualify(args.config.resolve())
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(qualification, indent=2) + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(qualification))
    print(json.dumps(qualification, indent=2))
    if not qualification["passed"]:
        raise SystemExit("one or more performance budgets failed")


if __name__ == "__main__":
    main()
