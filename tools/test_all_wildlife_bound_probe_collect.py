from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import all_wildlife_rules as rules
from tools.all_wildlife_bound_probe import SCHEMA
from tools.all_wildlife_bound_probe_collect import collect


def _tokens() -> list[dict[str, object]]:
    return [
        {
            "q": index,
            "r": 0,
            "wildlife": rules.SPECIES[index % len(rules.SPECIES)],
        }
        for index in range(rules.TOKEN_COUNT)
    ]


def _base(path: Path) -> tuple[dict[str, object], tuple[int, ...], str]:
    tokens = _tokens()
    counts = (4, 4, 4, 4, 4)
    rows = []
    for index, ruleset in enumerate(rules.rulesets()):
        breakdown = list(rules.score_tokens(tokens, ruleset))
        rows.append(
            {
                "index": index,
                "ruleset": ruleset,
                "proof_complete": index != 0,
                "optimum": sum(breakdown),
                "score_breakdown": breakdown,
                "counts": list(counts),
                "tokens": tokens,
                "unresolved_counts": [list(counts)] if index == 0 else [],
                "proof_paths": [],
            }
        )
    payload = {
        "schema": "all-wildlife-optimal-catalog-v1",
        "ruleset_count": len(rows),
        "results": rows,
    }
    encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    return rows[0], counts, hashlib.sha256(encoded).hexdigest()


def test_collector_merges_sound_infeasible_bound(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    row, counts, base_sha = _base(base_path)
    incumbent = int(row["optimum"])
    probe = {
        "schema": SCHEMA,
        "identity": {
            "ruleset_index": 0,
            "ruleset": "AAAAA",
            "base_catalog_sha256": base_sha,
            "probe_source_sha256": hashlib.sha256(
                Path("tools/all_wildlife_bound_probe.py").read_bytes()
            ).hexdigest(),
            "exact_source_sha256": hashlib.sha256(
                Path("tools/all_wildlife_exact.py").read_bytes()
            ).hexdigest(),
            "exact_support_source_sha256": hashlib.sha256(
                Path("tools/cbddb_wildlife_exact.py").read_bytes()
            ).hexdigest(),
            "rules_source_sha256": hashlib.sha256(
                Path("tools/all_wildlife_rules.py").read_bytes()
            ).hexdigest(),
            "connectivity_required": True,
        },
        "selected_counts": [list(counts)],
        "attempts": [
            {
                "counts": list(counts),
                "minimum_score": incumbent + 1,
                "analytical_upper": rules.count_upper(counts, "AAAAA"),
                "status": "INFEASIBLE",
                "model_objective": None,
                "best_bound": None,
                "refined_upper": incumbent,
                "witness_score": None,
                "score_breakdown": None,
                "tokens": None,
            }
        ],
    }
    (probe_dir / "task_0.json").write_text(json.dumps(probe))
    result = collect(base_path, [probe_dir], oracle=None)
    assert result["completed_rulesets"] == len(rules.rulesets())
    assert result["results"][0]["proof_complete"]
    assert result["results"][0]["sound_upper"] == incumbent
    assert result["holistic_sound_upper"] >= result["incumbent_holistic_maximum"]
