#!/usr/bin/env python3
"""Collect and validate per-ruleset exact proofs into the final catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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


def _git_blob_sha256(revision: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        capture_output=True,
        check=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _legacy_identities(
    ledgers: list[Path],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str]]:
    identities = {}
    ledger_hashes = {}
    for path in ledgers:
        encoded = path.read_bytes()
        ledger_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()
        ledger = json.loads(encoded)
        if (
            ledger.get("schema") != "all-wildlife-proof-fleet-v1"
            or ledger.get("state") != "complete"
        ):
            raise ValueError(f"{path}: invalid legacy fleet ledger")
        revision = ledger["source_revision"]
        proof_sha = _git_blob_sha256(revision, "tools/all_wildlife_global_proof.py")
        exact_sha = _git_blob_sha256(revision, "tools/all_wildlife_exact.py")
        support_sha = _git_blob_sha256(revision, "tools/cbddb_wildlife_exact.py")
        rules_sha = _git_blob_sha256(revision, "tools/all_wildlife_rules.py")
        if (
            proof_sha != ledger["proof_source_sha256"]
            or exact_sha != ledger["exact_source_sha256"]
            or (
                ledger.get("exact_support_source_sha256") is not None
                and support_sha != ledger["exact_support_source_sha256"]
            )
            or (
                ledger.get("rules_source_sha256") is not None
                and rules_sha != ledger["rules_source_sha256"]
            )
        ):
            raise ValueError(f"{path}: legacy source revision mismatch")
        key = (ledger["candidate_sha256"], proof_sha, exact_sha)
        if key in identities:
            raise ValueError(f"{path}: duplicate legacy identity")
        identities[key] = {
            "exact_support_source_sha256": support_sha,
            "rules_source_sha256": rules_sha,
            "result_sha256": {
                int(result["index"]): result["sha256"] for result in ledger["results"]
            },
            "result_summary": {
                int(result["index"]): result for result in ledger["results"]
            },
            "ledger_path": str(path),
        }
    return identities, ledger_hashes


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


def collect(
    candidates_path: Path,
    directories: list[Path],
    legacy_fleet_ledgers: list[Path] | None = None,
) -> dict[str, Any]:
    candidate_encoded = candidates_path.read_bytes()
    candidate_sha = hashlib.sha256(candidate_encoded).hexdigest()
    candidates = json.loads(candidate_encoded)
    if candidates.get("schema") != "all-wildlife-merged-candidates-v1":
        raise ValueError("unexpected candidate schema")
    paths = _proof_paths(directories)
    legacy_identities, legacy_ledger_hashes = _legacy_identities(
        legacy_fleet_ledgers or []
    )
    used_legacy_ledgers: set[str] = set()
    rows = []
    proof_hashes = {}
    proof_source_hashes: set[str] = set()
    exact_source_hashes: set[str] = set()
    exact_support_hashes: set[str] = set()
    rules_source_hashes: set[str] = set()
    connectivity_modes: set[bool] = set()
    for index, ruleset in enumerate(rules.rulesets()):
        proof_paths = paths.get(index, [])
        if not proof_paths:
            candidate = candidates["candidates"][index]
            if candidate["index"] != index or candidate["ruleset"] != ruleset:
                raise ValueError(f"{ruleset}: candidate identity mismatch")
            _validate_board(candidate, ruleset)
            unresolved = [
                list(counts)
                for counts in rules.count_vectors()
                if rules.count_upper(counts, ruleset) > candidate["score"]
            ]
            rows.append(
                {
                    "index": index,
                    "ruleset": ruleset,
                    "proof_complete": not unresolved,
                    "optimum": candidate["score"],
                    "score_breakdown": candidate["score_breakdown"],
                    "counts": candidate["counts"],
                    "tokens": candidate["tokens"],
                    "unresolved_counts": unresolved,
                    "proof_paths": [],
                }
            )
            continue
        aggregate_exclusions: dict[tuple[int, ...], int] = {}
        incumbents = []
        row_proof_hashes = {}
        for path in proof_paths:
            encoded = path.read_bytes()
            digest = hashlib.sha256(encoded).hexdigest()
            proof_hashes[str(path)] = digest
            row_proof_hashes[str(path)] = digest
            proof = json.loads(encoded)
            identity = proof["identity"]
            legacy_summary = None
            if (
                proof.get("schema") != "all-wildlife-global-proof-v1"
                or identity["ruleset_index"] != index
                or identity["ruleset"] != ruleset
                or identity["candidate_sha256"] != candidate_sha
            ):
                raise ValueError(f"{path}: proof identity mismatch")
            mode = bool(proof["configuration"]["connectivity_required"])
            if (
                "connectivity_required" in identity
                and bool(identity["connectivity_required"]) != mode
            ):
                raise ValueError(f"{path}: connectivity identity mismatch")
            connectivity_modes.add(mode)
            proof_source_hashes.add(identity["proof_source_sha256"])
            exact_source_hashes.add(identity["exact_source_sha256"])
            if (
                "exact_support_source_sha256" in identity
                and "rules_source_sha256" in identity
            ):
                exact_support_hash = identity["exact_support_source_sha256"]
                rules_source_hash = identity["rules_source_sha256"]
            else:
                legacy_key = (
                    identity["candidate_sha256"],
                    identity["proof_source_sha256"],
                    identity["exact_source_sha256"],
                )
                legacy = legacy_identities.get(legacy_key)
                if (
                    legacy is None
                    or legacy["result_sha256"].get(index) != digest
                ):
                    raise ValueError(f"{path}: unverified legacy proof identity")
                exact_support_hash = legacy["exact_support_source_sha256"]
                rules_source_hash = legacy["rules_source_sha256"]
                legacy_summary = legacy["result_summary"][index]
                used_legacy_ledgers.add(legacy["ledger_path"])
            exact_support_hashes.add(exact_support_hash)
            rules_source_hashes.add(rules_source_hash)
            incumbent = proof["incumbent"]
            _validate_board(incumbent, ruleset)
            incumbents.append(incumbent)
            local_exclusions: dict[tuple[int, ...], int] = {}
            for attempt in proof["attempts"]:
                if attempt["status"] == "INFEASIBLE":
                    counts = tuple(attempt["counts"])
                    threshold = int(attempt["threshold"])
                    local_exclusions[counts] = min(
                        local_exclusions.get(counts, threshold),
                        threshold,
                    )
                    aggregate_exclusions[counts] = min(
                        aggregate_exclusions.get(counts, threshold),
                        threshold,
                    )
            if legacy_summary is not None:
                if (
                    legacy_summary["ruleset"] != ruleset
                    or int(legacy_summary["score"]) != int(incumbent["score"])
                    or bool(legacy_summary["proof_complete"])
                    != bool(proof["proof_complete"])
                    or int(legacy_summary["attempts"]) != len(proof["attempts"])
                    or int(legacy_summary["unresolved_counts"])
                    != len(proof["unresolved_counts"])
                ):
                    raise ValueError(f"{path}: legacy result summary mismatch")
            else:
                local_complete = _proof_complete(
                    ruleset,
                    int(incumbent["score"]),
                    local_exclusions,
                )
                if local_complete != bool(proof["proof_complete"]):
                    raise ValueError(f"{path}: proof completeness mismatch")
                local_unresolved = [
                    list(counts)
                    for counts in rules.count_vectors()
                    if rules.count_upper(counts, ruleset) > incumbent["score"]
                    and local_exclusions.get(counts, int(incumbent["score"]) + 2)
                    > int(incumbent["score"]) + 1
                ]
                if local_unresolved != proof["unresolved_counts"]:
                    raise ValueError(f"{path}: unresolved count set mismatch")
        incumbent = min(
            incumbents,
            key=lambda row: (
                -int(row["score"]),
                json.dumps(row["tokens"], sort_keys=True),
            ),
        )
        complete = _proof_complete(
            ruleset,
            int(incumbent["score"]),
            aggregate_exclusions,
        )
        expected_unresolved = [
            list(counts)
            for counts in rules.count_vectors()
            if rules.count_upper(counts, ruleset) > incumbent["score"]
            and aggregate_exclusions.get(counts, int(incumbent["score"]) + 2)
            > int(incumbent["score"]) + 1
        ]
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
                "proof_paths": [str(path) for path in proof_paths],
                "proof_sha256": row_proof_hashes,
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
        "exact_support_source_sha256": sorted(exact_support_hashes),
        "rules_source_sha256": sorted(rules_source_hashes),
        "connectivity_modes": sorted(connectivity_modes),
        "legacy_fleet_ledger_sha256": {
            path: legacy_ledger_hashes[path] for path in sorted(used_legacy_ledgers)
        },
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
            },
            sort_keys=True,
        )
    )
    return 0 if payload["proof_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
