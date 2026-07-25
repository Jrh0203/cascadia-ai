#!/usr/bin/env python3
"""Merge available bounded-maximization probes into a wildlife catalog."""

from __future__ import annotations

import argparse
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


def _paths(directories: list[Path]) -> list[Path]:
    return sorted(
        (path for directory in directories for path in directory.glob("task_*.json")),
        key=str,
    )


def _validate_attempt(
    attempt: dict[str, Any],
    *,
    ruleset: str,
    unresolved: set[tuple[int, ...]],
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in attempt["counts"])
    if counts not in unresolved:
        raise ValueError(f"{ruleset}: probe count is not unresolved")
    analytical = rules.count_upper(counts, ruleset)
    refined = int(attempt["refined_upper"])
    if refined > analytical:
        raise ValueError(f"{ruleset}: refined upper exceeds analytical upper")
    witness_score = attempt.get("witness_score")
    if witness_score is not None:
        witness = {
            "tokens": attempt["tokens"],
            "counts": attempt["counts"],
            "score_breakdown": attempt["score_breakdown"],
            "score": witness_score,
        }
        _validate_board(witness, ruleset, "score")
        if int(witness_score) > refined:
            raise ValueError(f"{ruleset}: witness exceeds refined upper")
    return counts


def collect(
    base_catalog_path: Path,
    directories: list[Path],
    oracle: Path | None = None,
) -> dict[str, Any]:
    base = json.loads(base_catalog_path.read_text())
    if base.get("schema") not in {
        "all-wildlife-optimal-catalog-v1",
        "all-wildlife-optimal-catalog-v2",
    } or len(base.get("results", [])) != len(rules.rulesets()):
        raise ValueError("unexpected base catalog schema or row count")

    by_index: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for path in _paths(directories):
        probe = json.loads(path.read_text())
        identity = probe.get("identity", {})
        index = int(identity.get("ruleset_index", -1))
        if (
            probe.get("schema") != PROBE_SCHEMA
            or index < 0
            or index >= len(rules.rulesets())
            or identity.get("ruleset") != rules.rulesets()[index]
        ):
            raise ValueError(f"{path}: probe ruleset mismatch")
        by_index.setdefault(index, []).append((path, probe))

    rows = []
    improved_rulesets = []
    used_probe_paths = []
    for index, ruleset in enumerate(rules.rulesets()):
        row = deepcopy(base["results"][index])
        if row.get("index") != index or row.get("ruleset") != ruleset:
            raise ValueError(f"{ruleset}: base identity mismatch")
        _validate_board(row, ruleset, "optimum")
        base_unresolved = [tuple(counts) for counts in row.get("unresolved_counts", [])]
        unresolved_set = set(base_unresolved)
        if len(base_unresolved) != len(unresolved_set) or any(
            counts not in COUNT_VECTORS for counts in base_unresolved
        ):
            raise ValueError(f"{ruleset}: invalid unresolved counts")

        bounds = {
            counts: rules.count_upper(counts, ruleset) for counts in base_unresolved
        }
        stored_bounds = row.get("unresolved_count_upper_bounds")
        if stored_bounds and len(stored_bounds) == len(base_unresolved):
            for counts, value in zip(base_unresolved, stored_bounds, strict=True):
                bounds[counts] = min(bounds[counts], int(value))

        witnesses = []
        row_paths = list(row.get("bound_probe_paths", []))
        for path, probe in by_index.get(index, []):
            for attempt in probe.get("attempts", []):
                counts = _validate_attempt(
                    attempt,
                    ruleset=ruleset,
                    unresolved=unresolved_set,
                )
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
            path_text = str(path)
            if path_text not in row_paths:
                row_paths.append(path_text)
                used_probe_paths.append(path_text)

        best = min(
            [row, *witnesses],
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
        remaining = [counts for counts in base_unresolved if bounds[counts] > incumbent]
        row["unresolved_counts"] = [list(counts) for counts in remaining]
        row["unresolved_count_upper_bounds"] = [bounds[counts] for counts in remaining]
        row["proof_complete"] = not remaining
        row["sound_upper"] = max([incumbent, *(bounds[counts] for counts in remaining)])
        if row_paths:
            row["bound_probe_paths"] = row_paths
        row.pop("bound_probe_sha256", None)
        row.pop("proof_sha256", None)
        rows.append(row)

    if oracle:
        _production_validate(rows, oracle)
    complete = all(row["proof_complete"] for row in rows)
    incumbent_maximum = max(int(row["optimum"]) for row in rows)
    holistic_upper = max(int(row["sound_upper"]) for row in rows)
    result = {
        key: value
        for key, value in base.items()
        if "sha256" not in key.lower() and not key.lower().endswith("_hash")
    }
    result.update(
        {
            "schema": "all-wildlife-optimal-catalog-v2",
            "proof_complete": complete,
            "completed_rulesets": sum(row["proof_complete"] for row in rows),
            "bound_probe_paths": used_probe_paths,
            "bound_probe_improved_rulesets": improved_rulesets,
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
            f"Proven rows: **{payload['completed_rulesets']}/{payload['ruleset_count']}**.",
            "",
        ]
    )
    return header + "\n" + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--probe-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--oracle", type=Path)
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
                "incumbent_holistic_maximum": payload["incumbent_holistic_maximum"],
                "holistic_sound_upper": payload["holistic_sound_upper"],
                "improved_rulesets": len(payload["bound_probe_improved_rulesets"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
