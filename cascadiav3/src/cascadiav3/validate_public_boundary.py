"""Validate that expert roots keep model observations free of hidden state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl

FORBIDDEN_KEY_FRAGMENTS = (
    "hidden",
    "private",
    "stack_order",
    "tile_stack",
    "draw_order",
    "rng_state",
    "secret",
)


def _walk_public(value: Any, path: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                violations.append(f"{path}.{key}")
            _walk_public(child, f"{path}.{key}", violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_public(child, f"{path}[{index}]", violations)


def validate_roots(path: Path, *, deny_hidden_fields: bool) -> dict[str, Any]:
    records = read_replay_jsonl(path)
    violations: list[str] = []
    for root_index, record in enumerate(records):
        for field in ("public_tokens", "legal_actions"):
            if field in record:
                _walk_public(record[field], f"root[{root_index}].{field}", violations)
        observable = record.get("model_observation")
        if observable is not None:
            _walk_public(observable, f"root[{root_index}].model_observation", violations)

    status = "pass" if not violations or not deny_hidden_fields else "fail"
    return {
        "status": status,
        "roots": len(records),
        "deny_hidden_fields": deny_hidden_fields,
        "violations": violations,
        "checked_sections": ["public_tokens", "legal_actions", "model_observation"],
        "audit_private_allowed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--deny-hidden-fields", action="store_true")
    args = parser.parse_args()

    report = validate_roots(Path(args.roots), deny_hidden_fields=args.deny_hidden_fields)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
