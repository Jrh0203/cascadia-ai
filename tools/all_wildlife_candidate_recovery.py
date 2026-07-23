#!/usr/bin/env python3
"""Validate a recovered targeted-candidate run and merge strict improvements."""

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
from tools.all_wildlife_candidate_catalog import _card_matrix, _score_from_matrix

EXPECTED_INDICES = (0, 233, 562, 572, 637, 1023)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _validate_board(
    row: dict[str, Any],
    ruleset: str,
) -> list[dict[str, Any]]:
    tokens = rules.normalized_tokens(row["tokens"])
    counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if counts != tuple(row["counts"]) or any(count > rules.COUNT_CAP for count in counts):
        raise ValueError(f"{ruleset}: count mismatch")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"{ruleset}: disconnected board")
    breakdown = rules.score_tokens(tokens, ruleset)
    if list(breakdown) != row["score_breakdown"] or sum(breakdown) != row["score"]:
        raise ValueError(f"{ruleset}: independent score mismatch")
    if rules.count_upper(counts, ruleset) < row["score"]:
        raise ValueError(f"{ruleset}: score exceeds sound count bound")
    return tokens


def _validate_targeted_files(
    paths: list[Path],
    parent: dict[str, Any],
    recovery: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if (
        parent.get("schema") != "all-wildlife-targeted-candidate-fleet-v1"
        or parent.get("state") != "failed_sealed"
        or recovery.get("schema") != "all-wildlife-targeted-candidate-recovery-v1"
        or recovery.get("state") not in {"terminal_collecting", "completed_validated"}
        or recovery.get("terminal_exit_code") != 0
        or parent.get("authorized_recovery_tag") != recovery.get("tag")
    ):
        raise ValueError("parent/recovery ledger state mismatch")
    expected_configuration = parent["configuration"]
    rows = []
    hashes = {}
    seen = set()
    for path in paths:
        payload = _read(path)
        if payload.get("schema") != "all-wildlife-candidates-v1":
            raise ValueError(f"{path}: unexpected candidate schema")
        if (
            payload.get("token_count") != rules.TOKEN_COUNT
            or payload.get("count_cap") != rules.COUNT_CAP
            or payload.get("seed") != expected_configuration["base_seed"]
            or payload.get("restarts_per_ruleset")
            != expected_configuration["restarts_per_ruleset"]
            or payload.get("iterations_per_restart")
            != expected_configuration["iterations_per_restart"]
            # The release runner clamps worker threads to the one-index range.
            or payload.get("threads") != 1
            or len(payload.get("candidates", [])) != 1
        ):
            raise ValueError(f"{path}: targeted configuration mismatch")
        row = dict(payload["candidates"][0])
        index = row.get("index")
        if index in seen or index not in EXPECTED_INDICES:
            raise ValueError(f"{path}: duplicate or unexpected ruleset index")
        seen.add(index)
        ruleset = rules.rulesets()[index]
        if row.get("ruleset") != ruleset:
            raise ValueError(f"{path}: ruleset identity mismatch")
        if not 0 <= row.get("states_evaluated", -1) <= expected_configuration[
            "maximum_states_per_ruleset"
        ]:
            raise ValueError(f"{path}: evaluated-state count exceeds budget")
        row["tokens"] = _validate_board(row, ruleset)
        row["_source_path"] = str(path)
        rows.append(row)
        hashes[str(path)] = _sha256(path)
    if seen != set(EXPECTED_INDICES):
        raise ValueError(f"targeted result coverage mismatch: {sorted(seen)}")
    return rows, hashes


def merge_recovery(
    baseline_path: Path,
    parent_ledger_path: Path,
    recovery_ledger_path: Path,
    targeted_paths: list[Path],
    *,
    oracle: Path,
) -> dict[str, Any]:
    baseline = _read(baseline_path)
    parent = _read(parent_ledger_path)
    recovery = _read(recovery_ledger_path)
    if (
        baseline.get("schema") != "all-wildlife-merged-candidates-v1"
        or len(baseline.get("candidates", [])) != len(rules.rulesets())
    ):
        raise ValueError("unexpected baseline catalog")
    targeted, targeted_hashes = _validate_targeted_files(
        targeted_paths, parent, recovery
    )

    candidates = []
    original_direct_scores = []
    for index, (ruleset, source) in enumerate(
        zip(rules.rulesets(), baseline["candidates"], strict=True)
    ):
        if source["index"] != index or source["ruleset"] != ruleset:
            raise ValueError(f"baseline row {index} is not canonical")
        source = dict(source)
        source["tokens"] = _validate_board(source, ruleset)
        candidates.append(source)
        original_direct_scores.append(source["score"] - source["cross_score_gain"])

    improved_indices = set()
    direct_changes = []
    for targeted_row in targeted:
        source_ruleset = targeted_row["ruleset"]
        baseline_direct = original_direct_scores[targeted_row["index"]]
        direct_changes.append(
            {
                "index": targeted_row["index"],
                "ruleset": source_ruleset,
                "baseline_direct_score": baseline_direct,
                "recovery_direct_score": targeted_row["score"],
                "delta": targeted_row["score"] - baseline_direct,
            }
        )
        matrix = _card_matrix(targeted_row["tokens"])
        for index, target_ruleset in enumerate(rules.rulesets()):
            breakdown = _score_from_matrix(matrix, target_ruleset)
            score = sum(breakdown)
            if score <= candidates[index]["score"]:
                continue
            improved_indices.add(index)
            candidates[index] = {
                "index": index,
                "ruleset": target_ruleset,
                "score": score,
                "score_breakdown": list(breakdown),
                "counts": targeted_row["counts"],
                "tokens": targeted_row["tokens"],
                "source_ruleset": source_ruleset,
                "source_index": targeted_row["index"],
                "source_path": targeted_row["_source_path"],
                "count_upper": rules.global_count_upper(target_ruleset)[0],
                "upper_bound_matched": score
                == rules.global_count_upper(target_ruleset)[0],
                "cross_score_gain": score - original_direct_scores[index],
            }

    requests = []
    for index, (ruleset, row) in enumerate(
        zip(rules.rulesets(), candidates, strict=True)
    ):
        if row["index"] != index or row["ruleset"] != ruleset:
            raise ValueError(f"merged row {index} is not canonical")
        row["tokens"] = _validate_board(row, ruleset)
        requests.append({"ruleset": ruleset, "tokens": row["tokens"]})
    completed = subprocess.run(
        [str(oracle)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = json.loads(completed.stdout)
    if len(responses) != len(candidates):
        raise ValueError("production oracle response count mismatch")
    for row, response in zip(candidates, responses, strict=True):
        if (
            response["score"] != row["score"]
            or response["score_breakdown"] != row["score_breakdown"]
        ):
            raise ValueError(f"{row['ruleset']}: production score mismatch")

    result = dict(baseline)
    result["source_candidate_count"] = baseline["source_candidate_count"] + len(
        targeted
    )
    result["cross_improved_rulesets"] = sum(
        row["cross_score_gain"] > 0 for row in candidates
    )
    result["deep_recovery"] = {
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "parent_ledger_path": str(parent_ledger_path),
        "parent_ledger_sha256": _sha256(parent_ledger_path),
        "recovery_ledger_path": str(recovery_ledger_path),
        "recovery_ledger_sha256": _sha256(recovery_ledger_path),
        "targeted_sha256": targeted_hashes,
        "strictly_improved_rulesets": len(improved_indices),
        "strictly_improved_indices": sorted(improved_indices),
        "direct_changes": direct_changes,
        "oracle_path": str(oracle),
        "oracle_sha256": _sha256(oracle),
        "production_response_sha256": hashlib.sha256(
            json.dumps(responses, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    result["candidates"] = candidates
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--recovery-ledger", type=Path, required=True)
    parser.add_argument("--targeted", type=Path, action="append", required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = merge_recovery(
        args.baseline,
        args.parent_ledger,
        args.recovery_ledger,
        args.targeted,
        oracle=args.oracle,
    )
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "rulesets": len(payload["candidates"]),
                "strictly_improved_rulesets": payload["deep_recovery"][
                    "strictly_improved_rulesets"
                ],
                "direct_changes": payload["deep_recovery"]["direct_changes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
