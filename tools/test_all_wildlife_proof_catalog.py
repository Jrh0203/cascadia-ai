from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import all_wildlife_rules as rules
from tools.all_wildlife_proof_catalog import (
    _legacy_identities,
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


def test_collect_requires_verified_adapter_for_legacy_identity(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(_candidate_catalog()))
    candidate_sha = hashlib.sha256(candidates.read_bytes()).hexdigest()
    candidate = json.loads(candidates.read_text())["candidates"][200]
    ruleset = candidate["ruleset"]
    directory = tmp_path / "proofs"
    directory.mkdir()
    proof = {
        "schema": "all-wildlife-global-proof-v1",
        "identity": {
            "ruleset_index": 200,
            "ruleset": ruleset,
            "candidate_sha256": candidate_sha,
            "proof_source_sha256": "legacy-proof",
            "exact_source_sha256": "legacy-exact",
        },
        "configuration": {"connectivity_required": True},
        "proof_complete": False,
        "incumbent": candidate,
        "attempts": [],
        "unresolved_counts": [
            list(counts)
            for counts in rules.count_vectors()
            if rules.count_upper(counts, ruleset) > candidate["score"]
        ],
    }
    (directory / "ruleset_200.json").write_text(json.dumps(proof))

    with pytest.raises(ValueError, match="unverified legacy proof identity"):
        collect(candidates, [directory])


def test_legacy_fleet_identity_is_reconstructed_from_pinned_revision() -> None:
    path = Path(
        "cascadiav3/fleet/all_cards_proof_calibration_20260723_fleet.json"
    )
    identities, hashes = _legacy_identities([path])

    assert hashes[str(path)]
    assert len(identities) == 1
    identity = next(iter(identities.values()))
    assert (
        identity["exact_support_source_sha256"]
        == "362b5d7f82a156579e33c4b2c630c06bff3f45fa08f72a4dc70fe378eadca329"
    )
    assert (
        identity["rules_source_sha256"]
        == "48cfe51e750cdbc755a1770d6b161d2551c066c14ca0fd0e70126db4f022d2d8"
    )
