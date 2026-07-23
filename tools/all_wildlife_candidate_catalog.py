#!/usr/bin/env python3
"""Validate, cross-score, and merge all-card candidate fleet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules


def _canonical_tokens(tokens: list[dict[str, Any]]) -> str:
    return json.dumps(rules.normalized_tokens(tokens), sort_keys=True, separators=(",", ":"))


def _card_matrix(tokens: list[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    return tuple(rules.score_tokens(tokens, variant * 5) for variant in rules.VARIANTS)


def _score_from_matrix(
    matrix: tuple[tuple[int, ...], ...],
    ruleset: str,
) -> tuple[int, int, int, int, int]:
    cards = rules.parse_ruleset(ruleset)
    return tuple(
        matrix[rules.VARIANTS.index(cards[species])][species]
        for species in range(len(rules.SPECIES))
    )  # type: ignore[return-value]


def _validated_library_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    rows = payload.get("candidates", payload.get("results"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: no candidate/result rows")
    default_ruleset = str(
        payload.get(
            "scoring_cards",
            payload.get("assumptions", {}).get("scoring_cards", "LIBRARY"),
        )
    )
    result = []
    for index, row in enumerate(rows):
        if not row.get("tokens"):
            continue
        tokens = rules.normalized_tokens(row["tokens"])
        counts = tuple(
            sum(token["wildlife"] == species for token in tokens)
            for species in rules.SPECIES
        )
        if any(count > rules.COUNT_CAP for count in counts):
            raise ValueError(f"{path} row {index}: count cap exceeded")
        occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
        if len(rules.components(occupied)) != 1:
            raise ValueError(f"{path} row {index}: disconnected board")
        result.append(
            {
                "index": index,
                "ruleset": str(row.get("ruleset", default_ruleset)),
                "counts": list(counts),
                "tokens": tokens,
                "_source_path": str(path),
                "_fleet_direct": False,
            }
        )
    return result, hashlib.sha256(encoded).hexdigest()


def _validate_shards(
    paths: list[Path],
    libraries: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    indices: set[int] = set()
    shared: dict[str, Any] | None = None
    shard_hashes = {}
    for path in paths:
        encoded = path.read_bytes()
        shard_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()
        payload = json.loads(encoded)
        if payload.get("schema") != "all-wildlife-candidates-v1":
            raise ValueError(f"{path}: unexpected schema")
        configuration = {
            key: payload[key]
            for key in (
                "token_count",
                "count_cap",
                "seed",
                "restarts_per_ruleset",
                "iterations_per_restart",
            )
        }
        if shared is None:
            shared = configuration
        elif configuration != shared:
            raise ValueError(f"{path}: configuration differs across shards")
        for candidate in payload["candidates"]:
            index = int(candidate["index"])
            if index in indices:
                raise ValueError(f"duplicate candidate index {index}")
            expected_ruleset = rules.rulesets()[index]
            if candidate["ruleset"] != expected_ruleset:
                raise ValueError(f"index {index} has ruleset {candidate['ruleset']}")
            tokens = rules.normalized_tokens(candidate["tokens"])
            counts = tuple(
                sum(row["wildlife"] == species for row in tokens)
                for species in rules.SPECIES
            )
            if counts != tuple(candidate["counts"]):
                raise ValueError(f"{expected_ruleset}: count mismatch")
            if any(count > rules.COUNT_CAP for count in counts):
                raise ValueError(f"{expected_ruleset}: count cap exceeded")
            occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
            if len(rules.components(occupied)) != 1:
                raise ValueError(f"{expected_ruleset}: disconnected board")
            breakdown = rules.score_tokens(tokens, expected_ruleset)
            if list(breakdown) != candidate["score_breakdown"]:
                raise ValueError(f"{expected_ruleset}: independent score mismatch")
            if sum(breakdown) != candidate["score"]:
                raise ValueError(f"{expected_ruleset}: total score mismatch")
            candidate = dict(candidate)
            candidate["tokens"] = tokens
            candidate["_source_path"] = str(path)
            candidate["_fleet_direct"] = True
            candidates.append(candidate)
            indices.add(index)
    if indices != set(range(len(rules.rulesets()))):
        missing = sorted(set(range(len(rules.rulesets()))) - indices)
        raise ValueError(f"candidate coverage is incomplete; missing {missing[:20]}")
    assert shared is not None
    library_hashes = {}
    for path in libraries:
        rows, digest = _validated_library_rows(path)
        candidates.extend(rows)
        library_hashes[str(path)] = digest
    return candidates, {
        "configuration": shared,
        "shard_sha256": shard_hashes,
        "library_sha256": library_hashes,
    }


def merge(paths: list[Path], libraries: list[Path] | None = None) -> dict[str, Any]:
    candidates, provenance = _validate_shards(paths, libraries or [])
    best: list[dict[str, Any] | None] = [None] * len(rules.rulesets())
    direct_scores: dict[str, int] = {}
    cross_improvements = 0
    for candidate in candidates:
        source_ruleset = candidate["ruleset"]
        if candidate["_fleet_direct"]:
            direct = candidate.get("score")
            if direct is not None and source_ruleset in rules.rulesets():
                direct_scores[source_ruleset] = int(direct)
        matrix = _card_matrix(candidate["tokens"])
        canonical = _canonical_tokens(candidate["tokens"])
        for index, target_ruleset in enumerate(rules.rulesets()):
            breakdown = _score_from_matrix(matrix, target_ruleset)
            score = sum(breakdown)
            prior = best[index]
            prior_key = (
                int(prior["score"]) if prior else -1,
                str(prior["_canonical"]) if prior else "",
            )
            candidate_key = (score, canonical)
            if candidate_key[0] > prior_key[0] or (
                candidate_key[0] == prior_key[0] and candidate_key[1] < prior_key[1]
            ):
                best[index] = {
                    "index": index,
                    "ruleset": target_ruleset,
                    "score": score,
                    "score_breakdown": list(breakdown),
                    "counts": candidate["counts"],
                    "tokens": candidate["tokens"],
                    "source_ruleset": source_ruleset,
                    "source_index": candidate["index"],
                    "source_path": candidate["_source_path"],
                    "_canonical": canonical,
                }
    merged = []
    for row in best:
        assert row is not None
        row.pop("_canonical")
        upper, _ = rules.global_count_upper(str(row["ruleset"]))
        row["count_upper"] = upper
        row["upper_bound_matched"] = row["score"] == upper
        direct = direct_scores[str(row["ruleset"])]
        row["cross_score_gain"] = int(row["score"]) - direct
        cross_improvements += int(row["cross_score_gain"] > 0)
        merged.append(row)
    return {
        "schema": "all-wildlife-merged-candidates-v1",
        "token_count": rules.TOKEN_COUNT,
        "count_cap": rules.COUNT_CAP,
        "ruleset_count": len(rules.rulesets()),
        "source_candidate_count": len(candidates),
        "cross_improved_rulesets": cross_improvements,
        **provenance,
        "candidates": merged,
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--libraries", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = merge(args.shards, args.libraries)
    write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "rulesets": payload["ruleset_count"],
                "source_candidates": payload["source_candidate_count"],
                "cross_improved_rulesets": payload["cross_improved_rulesets"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
