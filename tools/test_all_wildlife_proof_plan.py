from __future__ import annotations

import json

from tools import all_wildlife_rules as rules
from tools.all_wildlife_proof_plan import build_plan
from tools.test_all_wildlife_proof_catalog import _candidate_catalog


def test_plan_is_complete_disjoint_deterministic_and_balanced(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_candidate_catalog()))
    hosts = ["john1", "john2", "john3", "john4"]

    first = build_plan(candidates, hosts)
    second = build_plan(candidates, hosts)

    assert first == second
    assigned = [
        index for shard in first["shards"] for index in shard["indices"]
    ]
    assert sorted(assigned) == list(range(len(rules.rulesets())))
    assert len(assigned) == len(set(assigned))
    weights = [shard["estimated_weight"] for shard in first["shards"]]
    assert max(weights) - min(weights) <= max(
        row["unresolved_count_branches"] or 1 for row in first["rulesets"]
    )
