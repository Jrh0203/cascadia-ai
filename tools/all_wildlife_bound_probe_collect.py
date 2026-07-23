#!/usr/bin/env python3
"""Validate bounded-maximization probes and merge their bounds/witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.all_wildlife_bound_probe import SCHEMA as PROBE_SCHEMA
from tools.all_wildlife_catalog_augment import _production_validate, _validate_board
from tools.all_wildlife_proof_catalog import (
    _write_atomic,
    _write_text_atomic,
    render_markdown,
)

COUNT_VECTORS = frozenset(rules.count_vectors())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(directories: list[Path]) -> list[Path]:
    return sorted(
        (path for directory in directories for path in directory.glob("task_*.json")),
        key=str,
    )


def _expected_sources() -> dict[str, str]:
    return {
        "probe_source_sha256": _sha256(Path("tools/all_wildlife_bound_probe.py")),
        "exact_source_sha256": _sha256(Path("tools/all_wildlife_exact.py")),
        "exact_support_source_sha256": _sha256(
            Path("tools/cbddb_wildlife_exact.py")
        ),
        "rules_source_sha256": _sha256(Path("tools/all_wildlife_rules.py")),
    }


def _validate_probe(
    path: Path,
    payload: dict[str, Any],
    *,
    base_sha: str,
    base_row: dict[str, Any],
) -> list[dict[str, Any]]:
    identity = payload.get("identity", {})
    expected_sources = _expected_sources()
    index = int(base_row["index"])
    ruleset = base_row["ruleset"]
    if (
        payload.get("schema") != PROBE_SCHEMA
        or identity.get("ruleset_index") != index
        or identity.get("ruleset") != ruleset
        or identity.get("base_catalog_sha256") != base_sha
        or not identity.get("connectivity_required")
        or any(identity.get(key) != value for key, value in expected_sources.items())
    ):
        raise ValueError(f"{path}: probe identity mismatch")
    selected = [tuple(counts) for counts in payload.get("selected_counts", [])]
    base_unresolved = {tuple(counts) for counts in base_row["unresolved_counts"]}
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(counts not in base_unresolved for counts in selected)
    ):
        raise ValueError(f"{path}: invalid selected counts")
    attempts = payload.get("attempts", [])
    if len(attempts) > len(selected):
        raise ValueError(f"{path}: too many attempts")
    observed = set()
    for attempt in attempts:
        counts = tuple(attempt["counts"])
        if counts not in selected or counts in observed:
            raise ValueError(f"{path}: invalid attempted count")
        observed.add(counts)
        analytical = rules.count_upper(counts, ruleset)
        if int(attempt["analytical_upper"]) != analytical:
            raise ValueError(f"{path}: analytical upper mismatch")
        status = attempt["status"]
        refined = int(attempt["refined_upper"])
        witness_score = attempt.get("witness_score")
        if status == "DOMINATED":
            if refined != analytical or witness_score is not None:
                raise ValueError(f"{path}: invalid dominated result")
        elif status == "INFEASIBLE":
            if refined != int(attempt["minimum_score"]) - 1 or witness_score is not None:
                raise ValueError(f"{path}: invalid infeasible result")
        elif status in {"OPTIMAL", "FEASIBLE", "UNKNOWN"}:
            best_bound = attempt.get("best_bound")
            if best_bound is None or refined != min(analytical, int(best_bound)):
                raise ValueError(f"{path}: invalid objective bound")
            if status in {"OPTIMAL", "FEASIBLE"} and witness_score is None:
                raise ValueError(f"{path}: feasible result omitted witness")
        else:
            raise ValueError(f"{path}: unexpected status {status}")
        if refined > analytical:
            raise ValueError(f"{path}: refined upper exceeds analytical upper")
        if witness_score is not None:
            witness = {
                "tokens": attempt["tokens"],
                "counts": attempt["counts"],
                "score_breakdown": attempt["score_breakdown"],
                "score": witness_score,
            }
            _validate_board(witness, ruleset, "score")
            if int(witness_score) > refined:
                raise ValueError(f"{path}: witness exceeds refined upper")
    return attempts


def collect(
    base_catalog_path: Path,
    directories: list[Path],
    oracle: Path | None = None,
) -> dict[str, Any]:
    base_encoded = base_catalog_path.read_bytes()
    base_sha = hashlib.sha256(base_encoded).hexdigest()
    base = json.loads(base_encoded)
    if (
        base.get("schema") != "all-wildlife-optimal-catalog-v1"
        or len(base.get("results", [])) != len(rules.rulesets())
    ):
        raise ValueError("unexpected base catalog schema or row count")

    paths = _paths(directories)
    if not paths:
        raise ValueError("no bound-probe outputs found")
    by_index: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    probe_hashes = {}
    for path in paths:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        index = int(payload.get("identity", {}).get("ruleset_index", -1))
        if index < 0 or index >= len(rules.rulesets()):
            raise ValueError(f"{path}: invalid ruleset index")
        by_index.setdefault(index, []).append((path, payload))
        probe_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()

    rows = []
    improved_rulesets = []
    probe_source_hashes = set()
    exact_source_hashes = set()
    for index, ruleset in enumerate(rules.rulesets()):
        row = deepcopy(base["results"][index])
        if row.get("index") != index or row.get("ruleset") != ruleset:
            raise ValueError(f"{ruleset}: base identity mismatch")
        _validate_board(row, ruleset, "optimum")
        base_unresolved = [tuple(counts) for counts in row["unresolved_counts"]]
        if len(base_unresolved) != len(set(base_unresolved)) or any(
            counts not in COUNT_VECTORS for counts in base_unresolved
        ):
            raise ValueError(f"{ruleset}: invalid base unresolved set")
        bounds = {
            counts: rules.count_upper(counts, ruleset) for counts in base_unresolved
        }
        witnesses = []
        row_paths = []
        row_hashes = {}
        for path, probe in by_index.get(index, []):
            attempts = _validate_probe(
                path,
                probe,
                base_sha=base_sha,
                base_row=row,
            )
            probe_source_hashes.add(probe["identity"]["probe_source_sha256"])
            exact_source_hashes.add(probe["identity"]["exact_source_sha256"])
            row_paths.append(str(path))
            row_hashes[str(path)] = probe_hashes[str(path)]
            for attempt in attempts:
                counts = tuple(attempt["counts"])
                bounds[counts] = min(bounds[counts], int(attempt["refined_upper"]))
                if attempt.get("witness_score") is not None:
                    witnesses.append(
                        {
                            "optimum": int(attempt["witness_score"]),
                            "score_breakdown": attempt["score_breakdown"],
                            "counts": attempt["counts"],
                            "tokens": attempt["tokens"],
                        }
                    )
        candidates = [row, *witnesses]
        best = min(
            candidates,
            key=lambda candidate: (
                -int(candidate["optimum"]),
                json.dumps(candidate["tokens"], sort_keys=True),
            ),
        )
        if int(best["optimum"]) > int(row["optimum"]):
            improved_rulesets.append(ruleset)
            row.update(
                {
                    "optimum": best["optimum"],
                    "score_breakdown": best["score_breakdown"],
                    "counts": best["counts"],
                    "tokens": best["tokens"],
                }
            )
        incumbent = int(row["optimum"])
        unresolved = [
            list(counts) for counts in base_unresolved if bounds[counts] > incumbent
        ]
        row["unresolved_counts"] = unresolved
        row["proof_complete"] = not unresolved
        row["sound_upper"] = max([incumbent, *bounds.values()])
        if row_paths:
            row["bound_probe_paths"] = row_paths
            row["bound_probe_sha256"] = row_hashes
        rows.append(row)

    for index, (ruleset, row) in enumerate(zip(rules.rulesets(), rows, strict=True)):
        if row["index"] != index or row["ruleset"] != ruleset:
            raise ValueError("merged identity mismatch")
        _validate_board(row, ruleset, "optimum")
        if row["sound_upper"] < row["optimum"]:
            raise ValueError(f"{ruleset}: sound upper below incumbent")
        if bool(row["proof_complete"]) != (not row["unresolved_counts"]):
            raise ValueError(f"{ruleset}: merged completeness mismatch")

    production_sha = _production_validate(rows, oracle) if oracle else None
    complete = all(row["proof_complete"] for row in rows)
    incumbent_maximum = max(int(row["optimum"]) for row in rows)
    holistic_upper = max(int(row["sound_upper"]) for row in rows)
    result = deepcopy(base)
    result.update(
        {
            "proof_complete": complete,
            "completed_rulesets": sum(row["proof_complete"] for row in rows),
            "base_catalog_sha256": base_sha,
            "bound_probe_sha256": probe_hashes,
            "bound_probe_source_sha256": sorted(probe_source_hashes),
            "bound_probe_exact_source_sha256": sorted(exact_source_hashes),
            "bound_probe_improved_rulesets": improved_rulesets,
            "production_response_sha256": production_sha,
            "holistic_optimum": incumbent_maximum if complete else None,
            "holistic_rulesets": (
                [row["ruleset"] for row in rows if row["optimum"] == incumbent_maximum]
                if complete
                else []
            ),
            "incumbent_holistic_maximum": incumbent_maximum,
            "incumbent_holistic_rulesets": [
                row["ruleset"] for row in rows if row["optimum"] == incumbent_maximum
            ],
            "holistic_sound_upper": holistic_upper,
            "holistic_gap": holistic_upper - incumbent_maximum,
            "results": rows,
        }
    )
    return result


def render_bound_markdown(payload: dict[str, Any]) -> str:
    body = render_markdown(payload)
    header = "\n".join(
        [
            "# Bounded-maximization wildlife catalog",
            "",
            f"Holistic interval: **[{payload['incumbent_holistic_maximum']}, "
            f"{payload['holistic_sound_upper']}]**.",
            f"Certified rows: **{payload['completed_rulesets']}/{payload['ruleset_count']}**.",
            "",
            "The JSON artifact records `sound_upper` for every ruleset and all",
            "validated probe paths. Boards marked unproven remain incumbents.",
            "",
        ]
    )
    return header + "\n" + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--probe-directories", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    payload = collect(args.base_catalog, args.probe_directories, args.oracle)
    _write_atomic(args.output, payload)
    if args.markdown:
        _write_text_atomic(args.markdown, render_bound_markdown(payload) + "\n")
    print(
        json.dumps(
            {
                "completed_rulesets": payload["completed_rulesets"],
                "incumbent_holistic_maximum": payload[
                    "incumbent_holistic_maximum"
                ],
                "holistic_sound_upper": payload["holistic_sound_upper"],
                "improved_rulesets": len(payload["bound_probe_improved_rulesets"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
