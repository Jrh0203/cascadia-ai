#!/usr/bin/env python3
"""Combine available per-ruleset exact proofs into a wildlife catalog."""

from __future__ import annotations

import argparse
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


def _proof_paths(directories: list[Path]) -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = {}
    for directory in directories:
        for path in directory.glob("ruleset_*.json"):
            index = int(path.stem.removeprefix("ruleset_"))
            found.setdefault(index, []).append(path)
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
    score = row.get("score", row.get("optimum"))
    if list(breakdown) != row["score_breakdown"] or sum(breakdown) != score:
        raise ValueError(f"{ruleset}: incumbent score mismatch")


def _proof_exclusions(
    proof: dict[str, Any],
    *,
    index: int,
    ruleset: str,
) -> tuple[dict[str, Any], dict[tuple[int, ...], int]]:
    if proof.get("schema") != "all-wildlife-global-proof-v1":
        raise ValueError(f"{ruleset}: unexpected proof schema")
    identity = proof.get("identity", {})
    if identity.get("ruleset_index") != index or identity.get("ruleset") != ruleset:
        raise ValueError(f"{ruleset}: proof ruleset mismatch")
    incumbent = proof["incumbent"]
    _validate_board(incumbent, ruleset)
    exclusions: dict[tuple[int, ...], int] = {}
    for attempt in proof.get("attempts", []):
        if attempt.get("status") == "INFEASIBLE":
            counts = tuple(int(value) for value in attempt["counts"])
            threshold = int(attempt["threshold"])
            exclusions[counts] = min(exclusions.get(counts, threshold), threshold)
    return incumbent, exclusions


def collect(
    candidates_path: Path,
    directories: list[Path],
    legacy_fleet_ledgers: list[Path] | None = None,
) -> dict[str, Any]:
    del legacy_fleet_ledgers  # accepted by the CLI for old command compatibility
    candidates = json.loads(candidates_path.read_text())
    if candidates.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")
    candidate_rows = candidates.get("candidates", [])
    if len(candidate_rows) != len(rules.rulesets()):
        raise ValueError("candidate catalog does not cover every ruleset")

    paths = _proof_paths(directories)
    rows = []
    connectivity_modes: set[bool] = set()
    for index, ruleset in enumerate(rules.rulesets()):
        candidate = candidate_rows[index]
        if candidate.get("index") != index or candidate.get("ruleset") != ruleset:
            raise ValueError(f"{ruleset}: candidate identity mismatch")
        _validate_board(candidate, ruleset)

        incumbents = [candidate]
        aggregate_exclusions: dict[tuple[int, ...], int] = {}
        used_paths = []
        for path in paths.get(index, []):
            proof = json.loads(path.read_text())
            incumbent, exclusions = _proof_exclusions(
                proof,
                index=index,
                ruleset=ruleset,
            )
            incumbents.append(incumbent)
            used_paths.append(str(path))
            connectivity_modes.add(
                bool(proof.get("configuration", {}).get("connectivity_required", True))
            )
            for counts, threshold in exclusions.items():
                aggregate_exclusions[counts] = min(
                    aggregate_exclusions.get(counts, threshold),
                    threshold,
                )

        incumbent = min(
            incumbents,
            key=lambda row: (
                -int(row.get("score", row.get("optimum"))),
                json.dumps(row["tokens"], sort_keys=True),
            ),
        )
        score = int(incumbent.get("score", incumbent.get("optimum")))
        complete = _proof_complete(ruleset, score, aggregate_exclusions)
        unresolved = [
            list(counts)
            for counts in rules.count_vectors()
            if rules.count_upper(counts, ruleset) > score
            and aggregate_exclusions.get(counts, score + 2) > score + 1
        ]
        rows.append(
            {
                "index": index,
                "ruleset": ruleset,
                "proof_complete": complete,
                "optimum": score,
                "score_breakdown": incumbent["score_breakdown"],
                "counts": incumbent["counts"],
                "tokens": incumbent["tokens"],
                "unresolved_counts": unresolved,
                "proof_paths": used_paths,
            }
        )

    complete = all(row["proof_complete"] for row in rows)
    best_score = max(int(row["optimum"]) for row in rows)
    return {
        "schema": "all-wildlife-optimal-catalog-v2",
        "proof_complete": complete,
        "completed_rulesets": sum(row["proof_complete"] for row in rows),
        "ruleset_count": len(rows),
        "token_count": rules.TOKEN_COUNT,
        "count_cap": rules.COUNT_CAP,
        "candidate_catalog": str(candidates_path),
        "proof_directories": [str(path) for path in directories],
        "connectivity_modes": sorted(connectivity_modes),
        "holistic_best_score": best_score,
        "holistic_best_rulesets": [
            row["ruleset"] for row in rows if row["optimum"] == best_score
        ],
        "holistic_optimum": best_score if complete else None,
        "results": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    status = "COMPLETE" if payload["proof_complete"] else "INCOMPLETE"
    best_score = payload.get(
        "holistic_best_score",
        payload.get("incumbent_holistic_maximum"),
    )
    best_rulesets = payload.get(
        "holistic_best_rulesets",
        payload.get("incumbent_holistic_rulesets", []),
    )
    if best_score is None:
        raise ValueError("catalog is missing its best incumbent score")
    lines = [
        "# Cap-six wildlife optimum for every card set",
        "",
        f"Proof status: **{status}** "
        f"({payload['completed_rulesets']}/{payload['ruleset_count']}).",
        "",
        f"Best score found: **{best_score}**.",
        f"Rulesets attaining it: `{', '.join(best_rulesets)}`.",
        "",
        "Each ruleset ID is ordered Bear/Elk/Salmon/Hawk/Fox. Every board has",
        "exactly 20 connected wildlife tokens and at most six of one species.",
        "All non-wildlife mechanics are ignored.",
        "",
    ]
    for row in payload["results"]:
        marker = "" if row["proof_complete"] else " (best known; proof incomplete)"
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
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--proof-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--legacy-fleet-ledgers", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    payload = collect(
        args.candidates,
        args.proof_directories,
        args.legacy_fleet_ledgers,
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
                "best_score": payload["holistic_best_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
