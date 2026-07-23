from __future__ import annotations

import hashlib
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


def test_collect_certifies_bound_matched_candidate_without_proof(
    tmp_path, monkeypatch
) -> None:
    payload = _candidate_catalog()
    candidate = payload["candidates"][0]  # type: ignore[index]
    score = candidate["score"]
    original_count_upper = rules.count_upper

    def count_upper(counts, ruleset):
        if ruleset == "AAAAA":
            return score
        return original_count_upper(counts, ruleset)

    monkeypatch.setattr(rules, "count_upper", count_upper)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(payload))
    proofs = tmp_path / "proofs"
    proofs.mkdir()

    catalog = collect(candidates, [proofs])

    assert catalog["completed_rulesets"] == 1
    assert catalog["results"][0]["proof_complete"]
    assert catalog["results"][0]["unresolved_counts"] == []


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


def test_collect_unions_connected_and_disconnected_exact_exclusions(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_candidate_catalog()))
    candidate_sha = hashlib.sha256(candidates.read_bytes()).hexdigest()
    candidate = json.loads(candidates.read_text())["candidates"][200]
    ruleset = candidate["ruleset"]
    unresolved = [
        counts
        for counts in rules.count_vectors()
        if rules.count_upper(counts, ruleset) > candidate["score"]
    ]
    assert len(unresolved) >= 2
    proof_directories = []
    for name, connected, excluded in (
        ("connected", True, unresolved[0]),
        ("disconnected", False, unresolved[1]),
    ):
        directory = tmp_path / name
        directory.mkdir()
        proof_directories.append(directory)
        local_unresolved = [
            list(counts) for counts in unresolved if counts != excluded
        ]
        proof = {
            "schema": "all-wildlife-global-proof-v1",
            "identity": {
                "ruleset_index": 200,
                "ruleset": ruleset,
                "candidate_sha256": candidate_sha,
                "proof_source_sha256": "proof",
                "exact_source_sha256": "exact",
                "exact_support_source_sha256": "support",
                "rules_source_sha256": "rules",
                "connectivity_required": connected,
            },
            "configuration": {"connectivity_required": connected},
            "proof_complete": False,
            "incumbent": candidate,
            "attempts": [
                {
                    "counts": list(excluded),
                    "threshold": candidate["score"] + 1,
                    "status": "INFEASIBLE",
                }
            ],
            "unresolved_counts": local_unresolved,
        }
        (directory / "ruleset_200.json").write_text(json.dumps(proof))

    catalog = collect(candidates, proof_directories)
    row = catalog["results"][200]

    assert len(row["proof_paths"]) == 2
    assert len(row["unresolved_counts"]) == len(unresolved) - 2
    assert catalog["connectivity_modes"] == [False, True]
