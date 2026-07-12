"""Merge candidate-only Gumbel benchmark chunk reports into one report.

Group-sequential gate batteries run their seed budget in chunks (separate
`torch_cascadiaformer_gumbel_benchmark` invocations over disjoint seed
ranges). This module merges the per-chunk JSON reports plus their decision
JSONLs (`--decisions-out`) into a single report with the exact shape
`build_report` emits for a candidate-only (control=none) arm, so downstream
consumers such as `compare_search_shape` read it unchanged.

Every aggregate is recomputed exactly from the underlying rows — never
weighted-averaged from chunk summaries: decision rows are concatenated and
per-seed results are reconstructed, then the real `summarize_game_results`,
`summarize_market_decisions`, and `summarize_score_categories` rebuild the
summaries, and the decision-seconds percentiles reuse the benchmark's own
`_percentile` helper. Paired fields (`paired_score_deltas`,
`paired_delta_stats`, `gate`) are empty/None because a candidate-only arm
has no control to pair against.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats
from .torch_cascadiaformer_gumbel_benchmark import (
    summarize_market_decisions,
    summarize_score_categories,
)
from .torch_cascadiaformer_search_benchmark import _percentile, summarize_game_results

MERGED_RUNNER = "merge_benchmark_reports"


def _load_chunk_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"chunk report is not a JSON object: {path}")
    if report.get("status") != "pass":
        raise ValueError(f"chunk report is not passing (status={report.get('status')!r}): {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(
            f"merge requires candidate-only chunks (control.kind == 'none'), got "
            f"{report.get('control', {}).get('kind')!r}: {path}"
        )
    if not report.get("source_revision"):
        raise ValueError(f"chunk report has an empty source_revision: {path}")
    for key in ("ruleset_id", "manifest", "binary", "search", "seeds", "candidate_per_seed"):
        if report.get(key) is None:
            raise ValueError(f"chunk report is missing {key!r}: {path}")
    return report


def _load_decision_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "gumbel_decision":
                raise ValueError(
                    f"decisions file {path} line {line_number} has type "
                    f"{row.get('type')!r}; expected 'gumbel_decision'"
                )
            rows.append(row)
    return rows


def _chunk_seed_set(report: dict[str, Any], path: Path) -> set[int]:
    seeds = [int(seed) for seed in report["seeds"]]
    seed_set = set(seeds)
    if len(seed_set) != len(seeds):
        raise ValueError(f"chunk report has duplicate seeds: {path}")
    per_seed = report["candidate_per_seed"]
    per_seed_seeds = {int(row["seed"]) for row in per_seed}
    if per_seed_seeds != seed_set or len(per_seed) != len(seeds):
        raise ValueError(
            f"candidate_per_seed seeds do not match the seeds list in {path}: "
            f"seeds={sorted(seed_set)}, candidate_per_seed={sorted(per_seed_seeds)}"
        )
    for row in per_seed:
        if "seat_score_breakdowns" not in row:
            raise ValueError(
                f"candidate_per_seed row for seed {row.get('seed')} lacks "
                f"seat_score_breakdowns (needed for exact recomputation): {path}"
            )
    return seed_set


def _validate_cross_chunk(
    chunks: list[dict[str, Any]], chunk_paths: list[Path]
) -> None:
    first = chunks[0]
    first_path = chunk_paths[0]
    for report, path in zip(chunks[1:], chunk_paths[1:], strict=True):
        for key in ("ruleset_id", "source_revision", "manifest", "binary"):
            if report.get(key) != first.get(key):
                raise ValueError(
                    f"{key} mismatch between chunks: {first_path} has "
                    f"{first.get(key)!r}, {path} has {report.get(key)!r}"
                )
        if report.get("search") != first.get("search"):
            raise ValueError(
                f"search settings mismatch between chunks {first_path} and {path}: "
                f"{json.dumps(first.get('search'), sort_keys=True)} != "
                f"{json.dumps(report.get('search'), sort_keys=True)}"
            )
    jobs_values = {
        path: report["execution"].get("requested_jobs")
        for report, path in zip(chunks, chunk_paths, strict=True)
        if isinstance(report.get("execution"), dict)
        and "requested_jobs" in report["execution"]
    }
    if len(set(jobs_values.values())) > 1:
        raise ValueError(
            "execution requested_jobs mismatch between chunks: "
            + ", ".join(f"{path}={jobs}" for path, jobs in jobs_values.items())
        )


def build_merged_report(
    chunk_paths: list[Path],
    decisions_paths: list[Path],
    experiment_id: str,
) -> dict[str, Any]:
    """Merges chunk reports + decision JSONLs into one candidate-only report."""
    if not chunk_paths:
        raise ValueError("at least one --chunk report is required")
    if len(decisions_paths) != len(chunk_paths):
        raise ValueError(
            f"--decisions count ({len(decisions_paths)}) must match --chunk count "
            f"({len(chunk_paths)}), in the same order"
        )

    chunks = [_load_chunk_report(path) for path in chunk_paths]
    _validate_cross_chunk(chunks, chunk_paths)

    seen_seeds: set[int] = set()
    chunk_seed_sets: list[set[int]] = []
    for report, path in zip(chunks, chunk_paths, strict=True):
        seed_set = _chunk_seed_set(report, path)
        overlap = seen_seeds & seed_set
        if overlap:
            raise ValueError(
                f"chunk seed sets overlap: {path} repeats seeds {sorted(overlap)}"
            )
        seen_seeds |= seed_set
        chunk_seed_sets.append(seed_set)

    all_decision_rows: list[dict[str, Any]] = []
    for seed_set, chunk_path, decisions_path in zip(
        chunk_seed_sets, chunk_paths, decisions_paths, strict=True
    ):
        rows = _load_decision_rows(decisions_path)
        row_seeds = {int(row["seed"]) for row in rows}
        if row_seeds != seed_set:
            raise ValueError(
                f"decisions file {decisions_path} covers seeds {sorted(row_seeds)} "
                f"but chunk {chunk_path} ran seeds {sorted(seed_set)}"
            )
        all_decision_rows.extend(rows)
    all_decision_rows.sort(key=lambda row: (int(row["seed"]), int(row["ply"])))

    decisions_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in all_decision_rows:
        decisions_by_seed.setdefault(int(row["seed"]), []).append(row)

    candidate_per_seed = sorted(
        (row for report in chunks for row in report["candidate_per_seed"]),
        key=lambda row: int(row["seed"]),
    )
    # Reconstruct the search-benchmark-shaped result rows the summaries expect:
    # summarize_game_results reads done.scores totals plus per-decision timing
    # fields, and summarize_score_categories reads done.scores category arrays.
    results = [
        {
            "seed": int(row["seed"]),
            "done": {"scores": row["seat_score_breakdowns"]},
            "decisions": decisions_by_seed[int(row["seed"])],
        }
        for row in candidate_per_seed
    ]

    decision_seconds = [
        float(row.get("decision_seconds", 0.0)) for row in all_decision_rows
    ]
    first = chunks[0]
    execution = dict(first.get("execution") or {})
    execution["merged_chunks"] = len(chunks)
    execution["merged_runner"] = MERGED_RUNNER
    merged_from = [
        {
            "report": str(path),
            "seeds": [min(seed_set), max(seed_set)],
            "candidate_wall_seconds": float(report["candidate_wall_seconds"]),
        }
        for report, path, seed_set in zip(chunks, chunk_paths, chunk_seed_sets, strict=True)
    ]

    return {
        "status": first["status"],
        "ruleset_id": first["ruleset_id"],
        "source_revision": first["source_revision"],
        "candidate_per_seed": candidate_per_seed,
        "scientific_eligibility": "candidate_only_search_arm",
        "experiment_id": experiment_id,
        "execution": execution,
        "artifacts": copy.deepcopy(first.get("artifacts")),
        "raw_games_dir": None,
        "binary": first["binary"],
        "manifest": first["manifest"],
        "model_service": first.get("model_service"),
        "seeds": sorted(seen_seeds),
        "search": copy.deepcopy(first["search"]),
        "market_decisions": summarize_market_decisions(all_decision_rows),
        "candidate_score_breakdown": summarize_score_categories(results),
        "control_score_breakdown": None,
        "control": copy.deepcopy(first["control"]),
        "strategies": {
            "gumbel-search": summarize_game_results(results),
            "control": None,
        },
        "candidate_decision_seconds_p50": _percentile(decision_seconds, 0.50),
        "candidate_decision_seconds_p95": _percentile(decision_seconds, 0.95),
        "candidate_wall_seconds": sum(
            float(report["candidate_wall_seconds"]) for report in chunks
        ),
        "paired_score_deltas": [],
        "paired_delta_stats": paired_delta_stats([]),
        "gate": None,
        "merged_from": merged_from,
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    candidate = report["strategies"]["gumbel-search"]
    lines = [
        "# Merged Gumbel Benchmark (group-sequential chunks)",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"- Chunks merged: `{report['execution']['merged_chunks']}`",
        f"- Total seeds: `{len(report['seeds'])}`",
        f"- Mean seat score: `{candidate['mean_seat_score']:.4f}`",
        f"- Mean total decision seconds: `{candidate['mean_total_decision_seconds']:.4f}`",
        f"- Candidate wall seconds (sum over chunks): `{report['candidate_wall_seconds']:.1f}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk",
        action="append",
        required=True,
        dest="chunks",
        metavar="REPORT_JSON",
        help="Chunk report JSON (repeatable, one per chunk)",
    )
    parser.add_argument(
        "--decisions",
        action="append",
        required=True,
        dest="decisions",
        metavar="DECISIONS_JSONL",
        help="Chunk decisions JSONL (repeatable; same count and order as --chunk)",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args(argv)

    report = build_merged_report(
        [Path(path) for path in args.chunks],
        [Path(path) for path in args.decisions],
        args.experiment_id,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out:
        write_markdown_summary(report, Path(args.summary_out))
    candidate = report["strategies"]["gumbel-search"]
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "merged_chunks": report["execution"]["merged_chunks"],
                "seeds": len(report["seeds"]),
                "mean_seat_score": candidate["mean_seat_score"],
                "candidate_wall_seconds": report["candidate_wall_seconds"],
                "out": str(out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
