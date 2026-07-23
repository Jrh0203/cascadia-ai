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


def _probe(
    *,
    row: dict[str, object],
    counts: tuple[int, ...],
    base_sha: str,
    refined_upper: int,
) -> dict[str, object]:
    ruleset = str(row["ruleset"])
    return {
        "schema": SCHEMA,
        "identity": {
            "ruleset_index": int(row["index"]),
            "ruleset": ruleset,
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
                "minimum_score": int(row["optimum"]) + 1,
                "analytical_upper": rules.count_upper(counts, ruleset),
                "status": "UNKNOWN",
                "model_objective": None,
                "best_bound": refined_upper,
                "refined_upper": refined_upper,
                "witness_score": None,
                "score_breakdown": None,
                "tokens": None,
            }
        ],
    }


def test_collector_preserves_prior_count_bounds_and_provenance(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.json"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    row, counts, base_sha = _base(base_path)
    analytical = rules.count_upper(counts, "AAAAA")
    first_upper = analytical - 1
    assert first_upper > int(row["optimum"])
    first_path = first_dir / "task_0.json"
    first_path.write_text(
        json.dumps(
            _probe(
                row=row,
                counts=counts,
                base_sha=base_sha,
                refined_upper=first_upper,
            )
        )
    )
    first = collect(base_path, [first_dir], oracle=None)
    first_catalog = tmp_path / "first_catalog.json"
    first_catalog.write_text(json.dumps(first))

    second_row = first["results"][0]
    second_sha = hashlib.sha256(first_catalog.read_bytes()).hexdigest()
    second_path = second_dir / "task_1.json"
    second_path.write_text(
        json.dumps(
            _probe(
                row=second_row,
                counts=counts,
                base_sha=second_sha,
                refined_upper=first_upper,
            )
        )
    )
    second = collect(first_catalog, [second_dir], oracle=None)

    merged_row = second["results"][0]
    assert merged_row["unresolved_count_upper_bounds"] == [first_upper]
    assert merged_row["sound_upper"] == first_upper
    assert merged_row["bound_probe_paths"] == [str(first_path), str(second_path)]
    assert set(second["bound_probe_sha256"]) == {str(first_path), str(second_path)}
