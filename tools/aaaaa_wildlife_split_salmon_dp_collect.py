#!/usr/bin/env python3
"""Collect split-Salmon bitset shards into exact AAAAA count certificates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tools import all_wildlife_rules as rules
from tools.aaaaa_wildlife_split_salmon_dp_screen import CASES


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


def _validate_candidate(
    row: dict[str, Any],
    counts: tuple[int, ...],
    optimum: int,
) -> list[dict[str, Any]]:
    tokens = rules.normalized_tokens(row["tokens"])
    observed_counts = tuple(
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    )
    if observed_counts != counts or any(value > rules.COUNT_CAP for value in counts):
        raise ValueError(f"{counts}: candidate count mismatch")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(rules.components(occupied)) != 1:
        raise ValueError(f"{counts}: disconnected candidate")
    breakdown = rules.score_tokens(tokens, "AAAAA")
    if list(breakdown) != row["score_breakdown"] or sum(breakdown) != row["score"]:
        raise ValueError(f"{counts}: independent score mismatch")
    if row["score"] != optimum:
        raise ValueError(f"{counts}: incumbent does not meet proved upper")
    return tokens


def collect(
    fleet_path: Path,
    shard_paths: list[Path],
    *,
    oracle: Path,
) -> dict[str, Any]:
    fleet = _read(fleet_path)
    if fleet.get("schema") != "aaaaa-split-salmon-bitset-fleet-v1":
        raise ValueError("unexpected fleet schema")
    if fleet.get("state") not in {"running", "completed"}:
        raise ValueError("fleet is neither running nor completed")
    by_case = {row["case_index"]: row for row in fleet["cases"]}
    if sorted(by_case) != list(range(len(CASES))) or len(by_case) != len(fleet["cases"]):
        raise ValueError("fleet case coverage mismatch")
    collected: dict[int, dict[str, Any]] = {}
    for path in shard_paths:
        shard = _read(path)
        if shard.get("schema") != "aaaaa-split-salmon-bitset-shard-v1":
            raise ValueError(f"{path}: unexpected shard schema")
        index = shard.get("case_index")
        if index in collected or index not in by_case:
            raise ValueError(f"{path}: duplicate or unknown case")
        case = by_case[index]
        counts, target = CASES[index]
        if (
            shard["counts"] != list(counts)
            or shard["target"] != target
            or case["counts"] != list(counts)
            or case["target"] != target
        ):
            raise ValueError(f"{path}: case identity mismatch")
        if shard["runtime"] != fleet["runtime"]:
            raise ValueError(f"{path}: runtime mismatch")
        result = shard["result"]
        if (
            result.get("status") != "INFEASIBLE"
            or result.get("upper_bound") != target - 1
            or result.get("target") != target
            or result.get("case_count") != case["expected_submodels"]
            or len(result.get("cases", [])) != case["expected_submodels"]
        ):
            raise ValueError(f"{path}: split branch is not complete")
        for submodel in result["cases"]:
            subresult = submodel.get("result", {})
            if (
                subresult.get("status") != "INFEASIBLE"
                or subresult.get("witness") is not None
            ):
                raise ValueError(f"{path}: non-infeasible submodel")
        collected[index] = shard
    if sorted(collected) != list(range(len(CASES))):
        raise ValueError("missing split-Salmon shard")

    maximum_path = Path(fleet["maximum_salmon_evidence"]["path"])
    candidate_path = Path(fleet["candidate_evidence"]["path"])
    maximum = _read(maximum_path)
    if (
        maximum.get("schema") != "aaaaa-two-missing-fox-screen-v1"
        or not maximum.get("proof_complete")
        or len(maximum.get("results", [])) != len(CASES)
    ):
        raise ValueError("maximum-Salmon evidence is incomplete")
    maximum_by_counts = {
        tuple(row["counts"]): row for row in maximum["results"]
    }
    candidates = _read(candidate_path)
    if candidates.get("schema") != "aaaaa-wildlife-candidates-v1":
        raise ValueError("unexpected candidate schema")
    candidate_by_counts = {
        tuple(row["counts"]): row for row in candidates["candidates"]
    }

    certificates = []
    requests = []
    for index, (counts, target) in enumerate(CASES):
        maximum_row = maximum_by_counts.get(counts)
        if (
            maximum_row is None
            or maximum_row["target"] != target
            or maximum_row["bound"]["status"] != "INFEASIBLE"
            or maximum_row["bound"]["upper_bound"] >= target
        ):
            raise ValueError(f"{counts}: maximum-Salmon branch is incomplete")
        candidate = candidate_by_counts.get(counts)
        if candidate is None:
            raise ValueError(f"{counts}: missing incumbent")
        tokens = _validate_candidate(candidate, counts, target - 1)
        requests.append({"ruleset": "AAAAA", "tokens": tokens})
        certificates.append(
            {
                "case_index": index,
                "counts": list(counts),
                "optimum": target - 1,
                "score_breakdown": candidate["score_breakdown"],
                "tokens": tokens,
                "maximum_salmon_upper": maximum_row["bound"]["upper_bound"],
                "split_salmon_upper": collected[index]["result"]["upper_bound"],
                "split_submodels": collected[index]["result"]["case_count"],
            }
        )

    completed = subprocess.run(
        [str(oracle)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = json.loads(completed.stdout)
    if len(responses) != len(certificates):
        raise ValueError("production oracle response count mismatch")
    for certificate, response in zip(certificates, responses, strict=True):
        if (
            response["score"] != certificate["optimum"]
            or response["score_breakdown"] != certificate["score_breakdown"]
        ):
            raise ValueError(f"{certificate['counts']}: production score mismatch")

    return {
        "schema": "aaaaa-split-salmon-bitset-certificate-v1",
        "proof_complete": True,
        "fleet": {
            "path": str(fleet_path),
            "tag": fleet.get("tag"),
        },
        "shards": [str(path) for path in shard_paths],
        "maximum_salmon_evidence": {"path": str(maximum_path)},
        "candidate_evidence": {"path": str(candidate_path)},
        "production_oracle": {
            "path": str(oracle),
        },
        "certificates": certificates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet-ledger", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("target/release/all_wildlife_score_oracle"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect(args.fleet_ledger, args.shard, oracle=args.oracle)
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "proof_complete": payload["proof_complete"],
                "certificates": len(payload["certificates"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
