from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_bound_probe_taskset import build_taskset, parse_case


def _catalog(path: Path) -> tuple[int, ...]:
    counts = (4, 4, 4, 4, 4)
    rows = [
        {
            "index": index,
            "ruleset": ruleset,
            "optimum": 0,
            "unresolved_counts": [list(counts)] if index == 0 else [],
        }
        for index, ruleset in enumerate(rules.rulesets())
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "all-wildlife-optimal-catalog-v1",
                "results": rows,
            }
        )
    )
    return counts


def test_taskset_freezes_explicit_unresolved_case(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    counts = _catalog(catalog)
    payload = build_taskset(catalog, [f"0:{','.join(map(str, counts))}"])
    assert payload["task_count"] == 1
    assert payload["tasks"][0]["ruleset"] == "AAAAA"
    assert payload["tasks"][0]["counts"] == list(counts)


def test_taskset_rejects_resolved_case(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    counts = _catalog(catalog)
    with pytest.raises(ValueError, match="not currently unresolved"):
        build_taskset(catalog, [f"1:{','.join(map(str, counts))}"])


def test_parse_case_rejects_noncanonical_counts() -> None:
    with pytest.raises(ValueError, match="invalid case"):
        parse_case("0:6,6,6,6,6")
