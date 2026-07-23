#!/usr/bin/env python3
"""Resumable exact catalog of CBDDB wildlife optima for every count vector."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tools.cbddb_wildlife_exact import (
    COUNT_CAP,
    SPECIES,
    TOKEN_COUNT,
    components,
    count_relaxation,
    count_vectors,
    normalized_tokens,
    render_tokens,
    score_tokens,
    solve_counts,
)
from tools.wildlife_catalog_sharding import load_taskset, select_shard

SCHEMA = "cbddb-wildlife-optimal-catalog-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_witness(
    counts: tuple[int, int, int, int, int], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, int | str]], list[int]]:
    tokens = normalized_tokens(rows)
    if len(tokens) != TOKEN_COUNT:
        raise ValueError(f"{counts}: witness has {len(tokens)} tokens")
    observed = Counter(str(token["wildlife"]) for token in tokens)
    expected = dict(zip(SPECIES, counts, strict=True))
    if observed != Counter(expected):
        raise ValueError(f"{counts}: witness counts {observed}, expected {expected}")
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(occupied) != TOKEN_COUNT:
        raise ValueError(f"{counts}: witness tokens overlap")
    if len(components(occupied)) != 1:
        raise ValueError(f"{counts}: witness is disconnected")
    return tokens, list(score_tokens(tokens))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cbddb-wildlife-candidates-v1":
        raise ValueError(f"unsupported candidate schema in {path}")
    if payload.get("scoring_cards") != "CBDDB":
        raise ValueError(f"candidate file is not CBDDB: {path}")
    candidates = list(payload.get("candidates", []))
    expected = [counts for counts, _ in count_vectors()]
    observed = [tuple(int(value) for value in row["counts"]) for row in candidates]
    if observed != expected:
        raise ValueError("candidate count vectors do not match the canonical 826-vector order")
    for counts, candidate in zip(expected, candidates, strict=True):
        tokens, breakdown = validate_witness(counts, candidate["tokens"])
        if sum(breakdown) != int(candidate["score"]):
            raise ValueError(f"{counts}: candidate score does not match independent scorer")
        candidate["tokens"] = tokens
        candidate["score_breakdown"] = breakdown
    return candidates


def attempt_summary(result: dict[str, Any], connectivity: bool, threshold: int) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "connectivity_required": connectivity,
        "model_status": result["model_status"],
        "model_score": result["model_score"],
        "independent_score": result["objective"],
        "best_bound": result["best_bound"],
        "wall_seconds": result["wall_seconds"],
        "branches": result["branches"],
        "conflicts": result["conflicts"],
    }


def solve_one(task: dict[str, Any]) -> dict[str, Any]:
    counts = tuple(int(value) for value in task["counts"])
    upper = count_relaxation(counts)  # type: ignore[arg-type]
    tokens, breakdown = validate_witness(counts, task["tokens"])  # type: ignore[arg-type]
    incumbent = sum(breakdown)
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()

    if incumbent == upper:
        return {
            "counts": list(counts),
            "optimum": incumbent,
            "count_relaxation": upper,
            "score_breakdown": breakdown,
            "tokens": tokens,
            "proof_method": "witness_matches_count_relaxation",
            "proof_complete": True,
            "attempts": attempts,
            "wall_seconds": time.monotonic() - started,
        }

    while incumbent < upper:
        threshold = incumbent + 1
        relaxed = solve_counts(
            counts,  # type: ignore[arg-type]
            threshold,
            float(task["relaxation_time_limit"]),
            int(task["solver_workers"]),
            int(task["seed"]) + threshold * 2,
            maximize=False,
            maximum_score=upper,
            enforce_connectivity=False,
            initial_tokens=tokens,
        )
        attempts.append(attempt_summary(relaxed, False, threshold))
        if relaxed["model_status"] == "INFEASIBLE":
            return {
                "counts": list(counts),
                "optimum": incumbent,
                "count_relaxation": upper,
                "score_breakdown": breakdown,
                "tokens": tokens,
                "proof_method": "disconnected_relaxation_infeasible",
                "proof_complete": True,
                "attempts": attempts,
                "wall_seconds": time.monotonic() - started,
            }
        if relaxed["objective"] is not None:
            relaxed_tokens = normalized_tokens(relaxed["tokens"])
            occupied = {(int(row["q"]), int(row["r"])) for row in relaxed_tokens}
            if len(components(occupied)) == 1 and int(relaxed["objective"]) > incumbent:
                tokens, breakdown = validate_witness(counts, relaxed_tokens)  # type: ignore[arg-type]
                incumbent = sum(breakdown)
                continue

        connected = solve_counts(
            counts,  # type: ignore[arg-type]
            threshold,
            float(task["connected_time_limit"]),
            int(task["solver_workers"]),
            int(task["seed"]) + threshold * 2 + 1,
            maximize=False,
            maximum_score=upper,
            enforce_connectivity=True,
            initial_tokens=tokens,
        )
        attempts.append(attempt_summary(connected, True, threshold))
        if connected["model_status"] == "INFEASIBLE":
            return {
                "counts": list(counts),
                "optimum": incumbent,
                "count_relaxation": upper,
                "score_breakdown": breakdown,
                "tokens": tokens,
                "proof_method": "connected_model_infeasible",
                "proof_complete": True,
                "attempts": attempts,
                "wall_seconds": time.monotonic() - started,
            }
        if connected["objective"] is None:
            return {
                "counts": list(counts),
                "optimum": incumbent,
                "count_relaxation": upper,
                "score_breakdown": breakdown,
                "tokens": tokens,
                "proof_method": "incomplete_timeout",
                "proof_complete": False,
                "attempts": attempts,
                "wall_seconds": time.monotonic() - started,
            }
        tokens, breakdown = validate_witness(counts, connected["tokens"])  # type: ignore[arg-type]
        improved = sum(breakdown)
        if improved <= incumbent:
            raise RuntimeError(f"{counts}: threshold {threshold} did not improve {incumbent}")
        incumbent = improved

    return {
        "counts": list(counts),
        "optimum": incumbent,
        "count_relaxation": upper,
        "score_breakdown": breakdown,
        "tokens": tokens,
        "proof_method": "witness_matches_count_relaxation",
        "proof_complete": True,
        "attempts": attempts,
        "wall_seconds": time.monotonic() - started,
    }


def payload_for(
    args: argparse.Namespace,
    candidates_sha256: str,
    results: dict[tuple[int, ...], dict[str, Any]],
) -> dict[str, Any]:
    ordered_counts = [counts for counts, _ in count_vectors()]
    ordered_results = [results[counts] for counts in ordered_counts if counts in results]
    complete = len(ordered_results) == len(ordered_counts) and all(
        result["proof_complete"] for result in ordered_results
    )
    source = Path(__file__).resolve()
    exact_source = source.with_name("cbddb_wildlife_exact.py")
    candidate_source = (
        source.parents[1] / "crates" / "cascadia-game" / "src" / "bin" / "cbddb_wildlife_solver.rs"
    )
    candidate_support = candidate_source.with_name("wildlife_solver_support") / "mod.rs"
    return {
        "schema": SCHEMA,
        "proof_complete": complete,
        "completed_count": sum(result["proof_complete"] for result in ordered_results),
        "allocation_count": len(ordered_counts),
        "assumptions": {
            "occupied_connected_hexes": TOKEN_COUNT,
            "maximum_per_species": COUNT_CAP,
            "scoring_cards": "CBDDB",
            "other_game_mechanics": "ignored",
        },
        "configuration": {
            "jobs": args.jobs,
            "solver_workers_per_job": args.solver_workers,
            "relaxation_time_limit_seconds": args.relaxation_time_limit,
            "connected_time_limit_seconds": args.connected_time_limit,
            "base_seed": args.seed,
            "taskset": getattr(args, "taskset_record", None),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
        "imported_ledgers": getattr(args, "imported_ledger_records", []),
        "candidates_sha256": candidates_sha256,
        "catalog_source_sha256": file_sha256(source),
        "exact_model_source_sha256": file_sha256(exact_source),
        "candidate_generator_source_sha256": file_sha256(candidate_source),
        "candidate_support_source_sha256": file_sha256(candidate_support),
        "results": ordered_results,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    complete_results = [result for result in results if result["proof_complete"]]
    best = max(complete_results, key=lambda result: result["optimum"], default=None)
    lines = [
        "# CBDDB Wildlife Optimum by Animal Counts",
        "",
        "One exact optimal connected 20-animal configuration for every legal",
        "`(bear, elk, salmon, hawk, fox)` count vector with at most six of a species.",
        "Habitats, tile restrictions, drafting, Nature tokens, and all non-wildlife mechanics",
        "are intentionally ignored.",
        "",
        f"Proof status: **{'complete' if payload['proof_complete'] else 'INCOMPLETE'}** "
        f"({payload['completed_count']}/{payload['allocation_count']}).",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Holistic maximum",
                "",
                f"Highest certified score so far: **{best['optimum']}**, at B/E/S/H/F = "
                f"`{'/'.join(str(value) for value in best['counts'])}`, with breakdown "
                f"`{'/'.join(str(value) for value in best['score_breakdown'])}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Method",
            "",
            "A parallel Rust annealer constructs connected incumbents using an independent custom",
            "CBDDB scorer, and every final incumbent is checked by the production Rust scorer.",
            "The exact labeled-token CP-SAT model then asks whether a better layout exists. It",
            "models Bear C components, Elk B disjoint shapes, Salmon D runs and adjacent animals,",
            "Hawk D line-of-sight maximum-weight matching, and Fox B doubled-neighbor species.",
            "Every CP witness is rescored independently in Python; timeout/UNKNOWN is never proof.",
            "",
            "## Summary",
            "",
            "| Bears | Elk | Salmon | Hawks | Foxes | Optimum | B/E/S/H/F score | Certificate |",
            "|---:|---:|---:|---:|---:|---:|:---|:---|",
        ]
    )
    for result in results:
        breakdown = "/".join(str(value) for value in result["score_breakdown"])
        lines.append(
            f"| {' | '.join(str(value) for value in result['counts'])} | {result['optimum']} | "
            f"{breakdown} | `{result['proof_method']}` |"
        )
    lines.extend(["", "## Optimal boards", ""])
    for index, result in enumerate(results, 1):
        lines.extend(
            [
                f"### {index:03d}. B/E/S/H/F = "
                f"{'/'.join(str(value) for value in result['counts'])}",
                "",
                f"Optimum **{result['optimum']}**; breakdown "
                f"`{'/'.join(str(value) for value in result['score_breakdown'])}`.",
                "",
                "```text",
                render_tokens(result["tokens"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    candidate_path = Path(args.candidates)
    output = Path(args.output)
    candidates_sha256 = file_sha256(candidate_path)
    candidates = load_candidates(candidate_path)
    canonical_counts = [counts for counts, _ in count_vectors()]
    selected_counts = set(canonical_counts)
    args.taskset_record = None
    if args.counts_file:
        selected_counts, args.taskset_record = load_taskset(
            Path(args.counts_file),
            scoring_cards="CBDDB",
            canonical_counts=canonical_counts,
        )
    candidate_by_counts = {
        counts: candidate for counts, candidate in zip(canonical_counts, candidates, strict=True)
    }
    results: dict[tuple[int, ...], dict[str, Any]] = {}
    args.imported_ledger_records = []

    for imported_path_string in args.import_ledger:
        imported_path = Path(imported_path_string)
        imported_sha256 = file_sha256(imported_path)
        prior = json.loads(imported_path.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA:
            raise SystemExit(f"unsupported imported ledger schema: {imported_path}")
        if int(prior.get("allocation_count", -1)) != len(canonical_counts):
            raise SystemExit(f"imported ledger count mismatch: {imported_path}")
        ledger_record = {
            "path": str(imported_path),
            "sha256": imported_sha256,
            "schema": prior["schema"],
            "catalog_source_sha256": prior.get("catalog_source_sha256"),
            "exact_model_source_sha256": prior.get("exact_model_source_sha256"),
            "candidate_generator_source_sha256": prior.get(
                "candidate_generator_source_sha256"
            ),
            "candidate_support_source_sha256": prior.get(
                "candidate_support_source_sha256"
            ),
            "candidates_sha256": prior.get("candidates_sha256"),
            "configuration": prior.get("configuration"),
        }
        args.imported_ledger_records.append(ledger_record)
        for result in prior.get("results", []):
            counts = tuple(int(value) for value in result["counts"])
            if counts not in candidate_by_counts:
                raise SystemExit(f"imported ledger has unexpected counts {counts}")
            tokens, breakdown = validate_witness(counts, result["tokens"])
            if sum(breakdown) != int(result["optimum"]):
                raise SystemExit(f"imported ledger witness mismatch for {counts}")
            if result.get("proof_complete"):
                imported_result = dict(result)
                imported_result["tokens"] = tokens
                imported_result["score_breakdown"] = breakdown
                imported_result.setdefault("proof_provenance", ledger_record)
                results[counts] = imported_result
            elif sum(breakdown) > int(candidate_by_counts[counts]["score"]):
                candidate_by_counts[counts]["tokens"] = tokens
                candidate_by_counts[counts]["score"] = sum(breakdown)
                candidate_by_counts[counts]["score_breakdown"] = breakdown

    if args.resume and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("schema") != SCHEMA or prior.get("candidates_sha256") != candidates_sha256:
            raise SystemExit("resume ledger schema or candidate hash mismatch")
        current_provenance = payload_for(args, candidates_sha256, {})
        for field in (
            "catalog_source_sha256",
            "exact_model_source_sha256",
            "candidate_generator_source_sha256",
            "candidate_support_source_sha256",
        ):
            if prior.get(field) != current_provenance[field]:
                raise SystemExit(f"resume ledger {field} mismatch")
        if prior.get("configuration") != current_provenance["configuration"]:
            raise SystemExit("resume ledger configuration mismatch")
        if not args.imported_ledger_records:
            args.imported_ledger_records = list(prior.get("imported_ledgers", []))
        for result in prior.get("results", []):
            if result.get("proof_complete"):
                results[tuple(int(value) for value in result["counts"])] = result

    pending = []
    for index, (counts, candidate) in enumerate(zip(canonical_counts, candidates, strict=True)):
        if counts in results:
            continue
        if counts not in selected_counts:
            continue
        candidate = candidate_by_counts[counts]
        task = {
            "counts": list(counts),
            "tokens": candidate["tokens"],
            "solver_workers": args.solver_workers,
            "relaxation_time_limit": args.relaxation_time_limit,
            "connected_time_limit": args.connected_time_limit,
            "seed": args.seed + index * 1000,
            "candidate_gap": count_relaxation(counts) - int(candidate["score"]),
        }
        pending.append(task)
    pending.sort(key=lambda task: (task["candidate_gap"], task["counts"]))
    pending = select_shard(
        pending,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if args.limit is not None:
        pending = pending[: args.limit]
    tasks = []
    for task in pending:
        counts = tuple(int(value) for value in task["counts"])
        if task["candidate_gap"] == 0:
            results[counts] = solve_one(task)
        else:
            tasks.append(task)

    payload = payload_for(args, candidates_sha256, results)
    atomic_json(output, payload)
    print(
        f"precertified={payload['completed_count']}/{payload['allocation_count']} "
        f"selected={len(selected_counts)} shard={args.shard_index}/{args.shard_count} "
        f"assigned={len(pending)} queued={len(tasks)}",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_to_counts = {
            executor.submit(solve_one, task): tuple(task["counts"]) for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_counts):
            counts = future_to_counts[future]
            result = future.result()
            results[counts] = result
            if not result["proof_complete"]:
                print(f"INCOMPLETE counts={counts} incumbent={result['optimum']}", flush=True)
            payload = payload_for(args, candidates_sha256, results)
            atomic_json(output, payload)
            print(
                f"completed={payload['completed_count']}/{payload['allocation_count']} "
                f"counts={counts} optimum={result['optimum']} "
                f"method={result['proof_method']} wall={result['wall_seconds']:.3f}s",
                flush=True,
            )

    payload = payload_for(args, candidates_sha256, results)
    atomic_json(output, payload)
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(payload), encoding="utf-8")
    return 0 if payload["proof_complete"] or args.limit is not None else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--solver-workers", type=int, default=8)
    parser.add_argument("--relaxation-time-limit", type=float, default=60.0)
    parser.add_argument("--connected-time-limit", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--counts-file",
        help="JSON taskset restricting newly attempted count vectors",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--import-ledger",
        action="append",
        default=[],
        help="import completed proofs and stronger incomplete incumbents from a prior ledger",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs < 1 or args.solver_workers < 1:
        parser.error("--jobs and --solver-workers must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return args


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(run(parse_args()))
