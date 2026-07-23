from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_tail_taskset import build_taskset


def _tokens() -> list[dict[str, object]]:
    species = list(rules.SPECIES)
    return [
        {"q": index, "r": 0, "wildlife": species[index % len(species)]}
        for index in range(20)
    ]


def _candidate(index: int) -> dict[str, object]:
    ruleset = rules.rulesets()[index]
    tokens = _tokens()
    score_breakdown = list(rules.score_tokens(tokens, ruleset))
    return {
        "index": index,
        "ruleset": ruleset,
        "score": sum(score_breakdown),
        "score_breakdown": score_breakdown,
        "counts": [4, 4, 4, 4, 4],
        "tokens": tokens,
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidates = [_candidate(0), _candidate(1)]
    candidate_path = tmp_path / "candidates.json"
    comparison_path = tmp_path / "comparison.json"
    catalog_path = tmp_path / "catalog.json"
    candidate_payload = {
        "schema": "all-wildlife-merged-candidates-v1",
        "candidates": candidates,
    }
    _write(candidate_path, candidate_payload)
    _write(comparison_path, candidate_payload)
    _write(
        catalog_path,
        {
            "schema": "all-wildlife-optimal-catalog-v1",
            "results": [
                {
                    "index": 0,
                    "ruleset": rules.rulesets()[0],
                    "optimum": candidates[0]["score"],
                    "proof_complete": False,
                    "unresolved_counts": [
                        [4, 4, 4, 4, 4],
                        [3, 5, 4, 4, 4],
                    ],
                },
                {
                    "index": 1,
                    "ruleset": rules.rulesets()[1],
                    "optimum": candidates[1]["score"],
                    "proof_complete": False,
                    "unresolved_counts": [
                        [4, 4, 4, 4, 4],
                        [3, 5, 4, 4, 4],
                        [5, 3, 4, 4, 4],
                        [4, 3, 5, 4, 4],
                    ],
                },
            ],
        },
    )
    return catalog_path, candidate_path, comparison_path


def test_build_taskset_selects_complete_branch_slice(tmp_path: Path) -> None:
    catalog, candidate, comparison = _fixture(tmp_path)
    payload = build_taskset(
        catalog,
        candidate,
        minimum_branches=2,
        maximum_branches=3,
        comparison_candidate_path=comparison,
    )
    assert payload["task_count"] == 1
    assert payload["count_query_count"] == 2
    assert payload["tasks"][0]["index"] == 0
    assert payload["tasks"][0]["threshold"] == payload["tasks"][0]["incumbent"] + 1
    assert payload["comparison_candidate_sha256"]


def test_build_taskset_rejects_selected_board_mismatch(tmp_path: Path) -> None:
    catalog, candidate, comparison = _fixture(tmp_path)
    payload = json.loads(comparison.read_text())
    payload["candidates"][0]["tokens"][0]["q"] = 99
    _write(comparison, payload)
    with pytest.raises(ValueError, match=r"candidate score mismatch|board mismatch"):
        build_taskset(
            catalog,
            candidate,
            minimum_branches=2,
            maximum_branches=3,
            comparison_candidate_path=comparison,
        )


def test_build_taskset_rejects_invalid_branch_range(tmp_path: Path) -> None:
    catalog, candidate, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="branch limits"):
        build_taskset(
            catalog,
            candidate,
            minimum_branches=0,
            maximum_branches=3,
        )
