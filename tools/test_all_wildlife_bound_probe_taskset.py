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
    assert payload["tasks"][0]["current_upper"] == rules.count_upper(counts, "AAAAA")


def test_taskset_rejects_resolved_case(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    counts = _catalog(catalog)
    with pytest.raises(ValueError, match="not currently unresolved"):
        build_taskset(catalog, [f"1:{','.join(map(str, counts))}"])


def test_parse_case_rejects_noncanonical_counts() -> None:
    with pytest.raises(ValueError, match="invalid case"):
        parse_case("0:6,6,6,6,6")


def test_top_frontier_selection_uses_persisted_count_bounds(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    _catalog(catalog)
    payload = json.loads(catalog.read_text())
    row = payload["results"][0]
    ranked = sorted(
        rules.count_vectors(),
        key=lambda counts: rules.count_upper(counts, "AAAAA"),
        reverse=True,
    )
    highest = ranked[0]
    second = next(
        counts
        for counts in ranked
        if rules.count_upper(counts, "AAAAA") < rules.count_upper(highest, "AAAAA")
    )
    second_upper = rules.count_upper(second, "AAAAA")
    row["unresolved_counts"] = [list(highest), list(second)]
    row["unresolved_count_upper_bounds"] = [second_upper - 1, second_upper]
    row["sound_upper"] = second_upper
    catalog.write_text(json.dumps(payload))

    selected = build_taskset(
        catalog,
        top_frontier_above=second_upper - 1,
    )

    assert selected["selection"]["mode"] == "top_frontier"
    assert selected["task_count"] == 1
    assert selected["tasks"][0]["counts"] == list(second)
    assert selected["tasks"][0]["current_upper"] == second_upper


def test_frontier_layer_budget_keeps_ruleset_groups_whole(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    _catalog(catalog)
    payload = json.loads(catalog.read_text())
    ranked = sorted(
        rules.count_vectors(),
        key=lambda counts: rules.count_upper(counts, "AAAAA"),
        reverse=True,
    )
    levels = sorted(
        {rules.count_upper(counts, "AAAAA") for counts in ranked},
        reverse=True,
    )
    second_upper = levels[1]
    second_group = [
        counts
        for counts in ranked
        if rules.count_upper(counts, "AAAAA") == second_upper
    ]
    row = payload["results"][0]
    row["unresolved_counts"] = [list(counts) for counts in ranked]
    row["sound_upper"] = levels[0]
    catalog.write_text(json.dumps(payload))

    selected = build_taskset(
        catalog,
        top_frontier_above=second_upper - 1,
        frontier_layer=2,
        task_budget=len(second_group),
    )

    assert selected["selection"]["frontier_layer"] == 2
    assert selected["selection"]["selected_group_count"] == 1
    assert selected["task_count"] == len(second_group)
    assert {tuple(task["counts"]) for task in selected["tasks"]} == set(second_group)


def test_frontier_options_rejected_for_explicit_cases(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    counts = _catalog(catalog)
    with pytest.raises(ValueError, match="require top-frontier"):
        build_taskset(
            catalog,
            [f"0:{','.join(map(str, counts))}"],
            frontier_layer=2,
        )
