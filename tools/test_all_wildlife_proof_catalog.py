from __future__ import annotations

import json

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_proof_catalog import (
    _write_text_atomic,
    collect,
    render_markdown,
)
from tools.test_all_wildlife_rules import random_connected_board


def _candidate_catalog() -> dict[str, object]:
    tokens = random_connected_board(202)
    counts = [
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    ]
    candidates = []
    for index, ruleset in enumerate(rules.rulesets()):
        breakdown = list(rules.score_tokens(tokens, ruleset))
        candidates.append(
            {
                "index": index,
                "ruleset": ruleset,
                "score": sum(breakdown),
                "score_breakdown": breakdown,
                "counts": counts,
                "tokens": tokens,
            }
        )
    return {
        "schema": "all-wildlife-merged-candidates-v1",
        "candidates": candidates,
    }


def test_collect_without_proofs_emits_valid_incomplete_catalog(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_candidate_catalog()))
    proofs = tmp_path / "proofs"
    proofs.mkdir()

    catalog = collect(candidates, [proofs])

    assert not catalog["proof_complete"]
    assert catalog["completed_rulesets"] == 0
    assert catalog["ruleset_count"] == 1024
    assert catalog["holistic_optimum"] is None
    assert catalog["results"][0]["ruleset"] == "AAAAA"
    assert catalog["results"][-1]["ruleset"] == "DDDDD"
    assert "(unproven incumbent)" in render_markdown(catalog)


def test_collect_rejects_candidate_identity_mismatch(tmp_path) -> None:
    payload = _candidate_catalog()
    payload["candidates"][0]["ruleset"] = "AAAAB"  # type: ignore[index]
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(payload))
    proofs = tmp_path / "proofs"
    proofs.mkdir()

    with pytest.raises(ValueError, match="candidate identity mismatch"):
        collect(candidates, [proofs])


def test_atomic_markdown_writer_creates_parent_and_replaces(tmp_path) -> None:
    output = tmp_path / "nested" / "catalog.md"
    _write_text_atomic(output, "first\n")
    _write_text_atomic(output, "second\n")
    assert output.read_text() == "second\n"
