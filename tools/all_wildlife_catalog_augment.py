#!/usr/bin/env python3
"""Rebase exact exclusions onto stronger boards and import global certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_exact as aaaaa
from tools import all_wildlife_rules as rules
from tools.all_wildlife_proof_catalog import (
    _write_atomic,
    _write_text_atomic,
    render_markdown,
)

COUNT_VECTORS = frozenset(rules.count_vectors())


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _validate_board(
    row: dict[str, Any],
    ruleset: str,
    score_key: str,
) -> list[dict[str, int | str]]:
    tokens = rules.normalized_tokens(row["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(row["counts"]) or any(count > rules.COUNT_CAP for count in counts):
        raise ValueError(f"{ruleset}: invalid counts")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"{ruleset}: disconnected board")
    breakdown = rules.score_tokens(tokens, ruleset)
    if (
        list(breakdown) != row["score_breakdown"]
        or sum(breakdown) != int(row[score_key])
    ):
        raise ValueError(f"{ruleset}: score mismatch")
    return tokens


def _validate_aaaaa_certificate(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    optimum = int(payload.get("optimal_score", -1))
    threshold = int(payload.get("minimum_score", -1))
    assumptions = payload.get("assumptions", {})
    if (
        not payload.get("proof_complete")
        or payload.get("counterexample_found") is not False
        or threshold != optimum + 1
        or assumptions.get("scoring_cards") != "AAAAA"
        or assumptions.get("occupied_connected_hexes") != rules.TOKEN_COUNT
        or assumptions.get("maximum_per_species") != rules.COUNT_CAP
        or assumptions.get("other_game_mechanics") != "ignored"
        or assumptions.get("global_hex_radius") != rules.TOKEN_COUNT - 1
    ):
        raise ValueError("AAAAA certificate contract mismatch")

    expected = {counts for counts, _ in aaaaa.count_vectors(threshold)}
    results = payload.get("results", [])
    observed = {tuple(row["counts"]) for row in results}
    if (
        len(results) != payload.get("allocation_count")
        or len(observed) != len(results)
        or observed != expected
    ):
        raise ValueError("AAAAA certificate allocation coverage mismatch")
    for result in results:
        counts = tuple(result["counts"])
        if (
            result.get("model_status") != "INFEASIBLE"
            or int(result.get("count_relaxation", -1)) != aaaaa.count_relaxation(counts)
            or int(result["count_relaxation"]) < threshold
        ):
            raise ValueError("AAAAA certificate contains an invalid exclusion")

    row = {
        "index": 0,
        "ruleset": "AAAAA",
        "proof_complete": True,
        "optimum": optimum,
        "score_breakdown": payload["optimal_score_breakdown"],
        "counts": payload["optimal_counts"],
        "tokens": payload["optimal_configuration"],
        "unresolved_counts": [],
        "proof_paths": [str(path)],
        "external_certificate_sha256": _sha256(encoded),
    }
    tokens = _validate_board(row, "AAAAA", "optimum")
    if (
        list(aaaaa.score_tokens(tokens)) != row["score_breakdown"]
        or sum(aaaaa.score_tokens(tokens)) != optimum
    ):
        raise ValueError("AAAAA certificate disagrees with its original scorer")
    identity = {
        "path": str(path),
        "sha256": _sha256(encoded),
        "model": payload.get("model"),
        "model_source_sha256": payload.get("model_source_sha256"),
        "production_verifier_sha256": payload.get("production_verifier_sha256"),
        "ortools_version": payload.get("ortools_version"),
        "threshold": threshold,
        "excluded_allocations": len(results),
    }
    return row, identity


def _validate_catalog_row(
    row: dict[str, Any],
    index: int,
    ruleset: str,
) -> None:
    if row.get("index") != index or row.get("ruleset") != ruleset:
        raise ValueError(f"{ruleset}: base catalog identity mismatch")
    _validate_board(row, ruleset, "optimum")
    unresolved = [tuple(counts) for counts in row.get("unresolved_counts", [])]
    if len(unresolved) != len(set(unresolved)) or any(
        counts not in COUNT_VECTORS for counts in unresolved
    ):
        raise ValueError(f"{ruleset}: invalid unresolved count set")
    if bool(row.get("proof_complete")) != (not unresolved):
        raise ValueError(f"{ruleset}: base completeness mismatch")


def _production_validate(rows: list[dict[str, Any]], oracle: Path) -> str:
    requests = [
        {"ruleset": row["ruleset"], "tokens": rules.normalized_tokens(row["tokens"])}
        for row in rows
    ]
    completed = subprocess.run(
        [str(oracle)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = json.loads(completed.stdout)
    if len(responses) != len(rows):
        raise ValueError("production oracle response count mismatch")
    for row, response in zip(rows, responses, strict=True):
        if (
            response["score"] != row["optimum"]
            or response["score_breakdown"] != row["score_breakdown"]
        ):
            raise ValueError(f"{row['ruleset']}: production score mismatch")
    canonical = json.dumps(responses, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical.encode())


def augment(
    base_catalog_path: Path,
    candidates_path: Path,
    aaaaa_certificate_path: Path,
    oracle: Path | None = None,
) -> dict[str, Any]:
    base_encoded = base_catalog_path.read_bytes()
    candidate_encoded = candidates_path.read_bytes()
    base = json.loads(base_encoded)
    candidates = json.loads(candidate_encoded)
    if (
        base.get("schema") != "all-wildlife-optimal-catalog-v1"
        or candidates.get("schema") != "all-wildlife-merged-candidates-v1"
        or len(base.get("results", [])) != len(rules.rulesets())
        or len(candidates.get("candidates", [])) != len(rules.rulesets())
    ):
        raise ValueError("unexpected catalog schema or row count")

    result = deepcopy(base)
    rows = []
    rebased = []
    for index, ruleset in enumerate(rules.rulesets()):
        base_row = deepcopy(base["results"][index])
        candidate = candidates["candidates"][index]
        _validate_catalog_row(base_row, index, ruleset)
        if candidate.get("index") != index or candidate.get("ruleset") != ruleset:
            raise ValueError(f"{ruleset}: candidate identity mismatch")
        _validate_board(candidate, ruleset, "score")
        if base_row["proof_complete"] and candidate["score"] > base_row["optimum"]:
            raise ValueError(f"{ruleset}: candidate contradicts certified optimum")
        if candidate["score"] > base_row["optimum"]:
            base_row.update(
                {
                    "optimum": candidate["score"],
                    "score_breakdown": candidate["score_breakdown"],
                    "counts": candidate["counts"],
                    "tokens": candidate["tokens"],
                }
            )
            rebased.append(ruleset)
            base_row["unresolved_counts"] = [
                counts
                for counts in base_row["unresolved_counts"]
                if rules.count_upper(tuple(counts), ruleset) > base_row["optimum"]
            ]
        base_row["proof_complete"] = not base_row["unresolved_counts"]
        rows.append(base_row)

    certificate_row, certificate_identity = _validate_aaaaa_certificate(
        aaaaa_certificate_path
    )
    if rows[0]["optimum"] > certificate_row["optimum"]:
        raise ValueError("AAAAA candidate contradicts global certificate")
    rows[0] = certificate_row

    for index, (ruleset, row) in enumerate(zip(rules.rulesets(), rows, strict=True)):
        _validate_catalog_row(row, index, ruleset)
    production_sha = _production_validate(rows, oracle) if oracle else None
    complete = all(row["proof_complete"] for row in rows)
    holistic = max(row["optimum"] for row in rows) if complete else None
    incumbent_maximum = max(row["optimum"] for row in rows)
    result.update(
        {
            "proof_complete": complete,
            "completed_rulesets": sum(row["proof_complete"] for row in rows),
            "candidate_sha256": _sha256(candidate_encoded),
            "base_catalog_sha256": _sha256(base_encoded),
            "base_candidate_sha256": base.get("candidate_sha256"),
            "external_ruleset_certificates": [certificate_identity],
            "rebased_improved_rulesets": rebased,
            "production_response_sha256": production_sha,
            "holistic_optimum": holistic,
            "holistic_rulesets": (
                [row["ruleset"] for row in rows if row["optimum"] == holistic]
                if holistic is not None
                else []
            ),
            "incumbent_holistic_maximum": incumbent_maximum,
            "incumbent_holistic_rulesets": [
                row["ruleset"] for row in rows if row["optimum"] == incumbent_maximum
            ],
            "results": rows,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--aaaaa-certificate", type=Path, required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    payload = augment(
        args.base_catalog,
        args.candidates,
        args.aaaaa_certificate,
        args.oracle,
    )
    _write_atomic(args.output, payload)
    if args.markdown:
        _write_text_atomic(args.markdown, render_markdown(payload) + "\n")
    print(
        json.dumps(
            {
                "proof_complete": payload["proof_complete"],
                "completed_rulesets": payload["completed_rulesets"],
                "rulesets": payload["ruleset_count"],
                "rebased_improved_rulesets": len(
                    payload["rebased_improved_rulesets"]
                ),
                "incumbent_holistic_maximum": payload[
                    "incumbent_holistic_maximum"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
