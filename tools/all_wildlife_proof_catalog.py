#!/usr/bin/env python3
"""Collect and validate per-ruleset exact proofs into the final catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.all_wildlife_global_proof import _proof_complete
from tools.cbddb_wildlife_exact import render_tokens


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def _proof_paths(directories: list[Path]) -> dict[int, Path]:
    found = {}
    for directory in directories:
        for path in directory.glob("ruleset_*.json"):
            index = int(path.stem.removeprefix("ruleset_"))
            if index in found:
                raise ValueError(f"duplicate proof for index {index}")
            found[index] = path
    return found


def _validate_board(row: dict[str, Any], ruleset: str) -> None:
    tokens = rules.normalized_tokens(row["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(row["counts"]) or any(count > rules.COUNT_CAP for count in counts):
        raise ValueError(f"{ruleset}: invalid counts")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"{ruleset}: disconnected incumbent")
    breakdown = rules.score_tokens(tokens, ruleset)
    if list(breakdown) != row["score_breakdown"] or sum(breakdown) != row["score"]:
        raise ValueError(f"{ruleset}: incumbent score mismatch")


def collect(candidates_path: Path, directories: list[Path]) -> dict[str, Any]:
    candidate_encoded = candidates_path.read_bytes()
    candidate_sha = hashlib.sha256(candidate_encoded).hexdigest()
    candidates = json.loads(candidate_encoded)
    if candidates.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")
    paths = _proof_paths(directories)
    rows = []
    proof_hashes = {}
    proof_source_hashes: set[str] = set()
    exact_source_hashes: set[str] = set()
    rules_source_hashes: set[str] = set()
    for index, ruleset in enumerate(rules.rulesets()):
        path = paths.get(index)
        if path is None:
            candidate = candidates["candidates"][index]
            if candidate["index"] != index or candidate["ruleset"] != ruleset:
                raise ValueError(f"{ruleset}: candidate identity mismatch")
            _validate_board(candidate, ruleset)
            rows.append(
                {
                    "index": index,
                    "ruleset": ruleset,
                    "proof_complete": False,
                    "optimum": candidate["score"],
                    "score_breakdown": candidate["score_breakdown"],
                    "counts": candidate["counts"],
                    "tokens": candidate["tokens"],
                    "unresolved_counts": [
                        list(counts)
                        for counts in rules.count_vectors()
                        if rules.count_upper(counts, ruleset) > candidate["score"]
                    ],
                    "proof_path": None,
                }
            )
            continue
        encoded = path.read_bytes()
        proof_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()
        proof = json.loads(encoded)
        identity = proof["identity"]
        if (
            proof.get("schema") != "all-wildlife-global-proof-v1"
            or identity["ruleset_index"] != index
            or identity["ruleset"] != ruleset
            or identity["candidate_sha256"] != candidate_sha
        ):
            raise ValueError(f"{path}: proof identity mismatch")
        if not proof["configuration"]["connectivity_required"]:
            raise ValueError(f"{path}: proof omitted the connectivity constraint")
        proof_source_hashes.add(identity["proof_source_sha256"])
        exact_source_hashes.add(identity["exact_source_sha256"])
        rules_source_hashes.add(identity["rules_source_sha256"])
        incumbent = proof["incumbent"]
        _validate_board(incumbent, ruleset)
        exclusions: dict[tuple[int, ...], int] = {}
        for attempt in proof["attempts"]:
            if attempt["status"] == "INFEASIBLE":
                counts = tuple(attempt["counts"])
                threshold = int(attempt["threshold"])
                exclusions[counts] = min(exclusions.get(counts, threshold), threshold)
        complete = _proof_complete(ruleset, int(incumbent["score"]), exclusions)
        if complete != bool(proof["proof_complete"]):
            raise ValueError(f"{path}: proof completeness mismatch")
        expected_unresolved = [
            list(counts)
            for counts in rules.count_vectors()
            if rules.count_upper(counts, ruleset) > incumbent["score"]
            and exclusions.get(counts, int(incumbent["score"]) + 2)
            > int(incumbent["score"]) + 1
        ]
        if expected_unresolved != proof["unresolved_counts"]:
            raise ValueError(f"{path}: unresolved count set mismatch")
        rows.append(
            {
                "index": index,
                "ruleset": ruleset,
                "proof_complete": complete,
                "optimum": incumbent["score"],
                "score_breakdown": incumbent["score_breakdown"],
                "counts": incumbent["counts"],
                "tokens": incumbent["tokens"],
                "unresolved_counts": expected_unresolved,
                "proof_path": str(path),
                "proof_sha256": proof_hashes[str(path)],
            }
        )
    complete = all(row["proof_complete"] for row in rows)
    holistic = max(row["optimum"] for row in rows) if complete else None
    return {
        "schema": "all-wildlife-optimal-catalog-v1",
        "proof_complete": complete,
        "completed_rulesets": sum(row["proof_complete"] for row in rows),
        "ruleset_count": len(rows),
        "token_count": rules.TOKEN_COUNT,
        "count_cap": rules.COUNT_CAP,
        "candidate_sha256": candidate_sha,
        "proof_sha256": proof_hashes,
        "proof_source_sha256": sorted(proof_source_hashes),
        "exact_source_sha256": sorted(exact_source_hashes),
        "rules_source_sha256": sorted(rules_source_hashes),
        "holistic_optimum": holistic,
        "holistic_rulesets": (
            [row["ruleset"] for row in rows if row["optimum"] == holistic]
            if holistic is not None
            else []
        ),
        "results": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    status = "COMPLETE" if payload["proof_complete"] else "INCOMPLETE"
    lines = [
        "# Exact cap-six wildlife optimum for every card set",
        "",
        f"Proof status: **{status}** "
        f"({payload['completed_rulesets']}/{payload['ruleset_count']}).",
        "",
        "Each ruleset ID is ordered Bear/Elk/Salmon/Hawk/Fox. Every board has",
        "exactly 20 connected wildlife tokens and at most six of one species.",
        "All non-wildlife mechanics are ignored.",
        "",
    ]
    if payload["proof_complete"]:
        lines.extend(
            [
                f"Holistic optimum: **{payload['holistic_optimum']}**.",
                f"Rulesets attaining it: `{', '.join(payload['holistic_rulesets'])}`.",
                "",
            ]
        )
    for row in payload["results"]:
        marker = "" if row["proof_complete"] else " (unproven incumbent)"
        lines.extend(
            [
                f"## {row['ruleset']} — {row['optimum']}{marker}",
                "",
                f"Counts B/E/S/H/F: `{'/'.join(map(str, row['counts']))}`  ",
                f"Breakdown B/E/S/H/F: `{'/'.join(map(str, row['score_breakdown']))}`",
                "",
                "```text",
                render_tokens(row["tokens"]),
                "```",
                "",
                "Coordinates:",
                "",
                "```json",
                json.dumps(row["tokens"], separators=(",", ":")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--proof-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    payload = collect(args.candidates, args.proof_directories)
    _write_atomic(args.output, payload)
    if args.markdown:
        _write_text_atomic(args.markdown, render_markdown(payload) + "\n")
    print(
        json.dumps(
            {
                "proof_complete": payload["proof_complete"],
                "completed_rulesets": payload["completed_rulesets"],
                "rulesets": payload["ruleset_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["proof_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
