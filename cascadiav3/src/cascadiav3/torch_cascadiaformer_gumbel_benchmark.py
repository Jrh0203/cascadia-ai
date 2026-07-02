"""Paired-seed benchmark: Gumbel search (Rust-driven, batched model leaf
values) versus the legacy full rollout search control.

The candidate side spawns `--gumbel-policy-game` processes; each Rust process
owns its own model bridge session, so parallelism is process-level via
--jobs. The control side reuses the interactive full-search path from the
search benchmark. Both sides play the same seeds; the control runs with
--rollout-determinize by default so the comparison is against the honest
(no hidden-order peek) baseline.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from .torch_benchmark_stats import paired_delta_stats
from .torch_cascadiaformer_search_benchmark import (
    _percentile,
    parse_seeds,
    run_interactive_game,
    summarize_game_results,
)


def default_model_service_command(manifest: Path, device: str) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m cascadiav3.torch_inference_bridge "
        f"--manifest {shlex.quote(str(manifest))} --device {shlex.quote(device)}"
    )


def _contiguous_runs(seeds: list[int]) -> list[tuple[int, int]]:
    """Returns (first_seed, count) runs covering the seed list in order."""
    runs: list[tuple[int, int]] = []
    for seed in seeds:
        if runs and runs[-1][0] + runs[-1][1] == seed:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((seed, 1))
    return runs


def run_gumbel_games(
    binary: Path,
    *,
    first_seed: int,
    seed_count: int,
    model_service: str,
    model_manifest: Path,
    out_path: Path,
    n_simulations: int,
    top_m: int,
    depth_rounds: int,
    determinizations: int,
    blend_weight: float,
    k_interior: int,
    max_root_actions: int | None,
    rollout_max_actions: int,
    rollout_top_k: int,
    model_timeout_ms: int,
    exploration: bool,
) -> list[dict[str, Any]]:
    command = [
        str(binary),
        "--gumbel-policy-game",
        "--first-seed",
        str(first_seed),
        "--seed-count",
        str(seed_count),
        "--model-service",
        model_service,
        "--model-manifest",
        str(model_manifest),
        # No --allow-model-fallback: a bridge/checkpoint failure must kill the
        # benchmark loudly, never silently degrade to uniform-prior search.
        "--model-timeout-ms",
        str(model_timeout_ms),
        "--gumbel-n-simulations",
        str(n_simulations),
        "--gumbel-top-m",
        str(top_m),
        "--gumbel-depth-rounds",
        str(depth_rounds),
        "--gumbel-determinizations",
        str(determinizations),
        "--gumbel-blend-weight",
        str(blend_weight),
        "--gumbel-exploration",
        "on" if exploration else "off",
        "--k-interior",
        str(k_interior),
        "--max-actions",
        str(rollout_max_actions),
        "--rollout-top-k",
        str(rollout_top_k),
        "--out",
        str(out_path),
    ]
    if max_root_actions is not None:
        command.extend(["--gumbel-max-root-actions", str(max_root_actions)])
    # Preserve the Rust/bridge stderr (per-seed progress, any bridge
    # warnings) next to the decision JSONL for postmortems.
    stderr_path = out_path.with_name(out_path.name + ".stderr.log")
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True)
    if completed.returncode != 0:
        stderr_tail = stderr_path.read_text(encoding="utf-8")[-4000:]
        raise RuntimeError(
            f"gumbel policy game failed ({completed.returncode}): {stderr_tail}"
        )
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    return lines


def collect_gumbel_results(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups gumbel_decision/gumbel_game_done rows into search-benchmark-shaped results."""
    decisions_by_seed: dict[int, list[dict[str, Any]]] = {}
    results = []
    for line in lines:
        if line.get("type") == "gumbel_decision":
            decisions_by_seed.setdefault(int(line["seed"]), []).append(line)
        elif line.get("type") == "gumbel_game_done":
            seed = int(line["seed"])
            results.append(
                {
                    "seed": seed,
                    "strategy": "gumbel-search",
                    "selection_head": "gumbel",
                    "done": {
                        "scores": line["scores"],
                        "turns": line.get("decision_count"),
                        "elapsed_seconds": line.get("elapsed_seconds"),
                        "final_state_hash": None,
                    },
                    "decisions": decisions_by_seed.get(seed, []),
                }
            )
    return results


def run_gumbel_benchmark(
    *,
    binary: Path,
    manifest: Path,
    seeds: list[int],
    model_service: str | None,
    device_name: str,
    jobs: int,
    n_simulations: int,
    top_m: int,
    depth_rounds: int,
    determinizations: int,
    blend_weight: float,
    k_interior: int,
    max_root_actions: int | None,
    control: str,
    control_max_actions: int,
    control_rollouts_per_action: int,
    control_rollout_top_k: int,
    control_workers: int,
    rollout_determinize: bool,
    model_timeout_ms: int,
    experiment_id: str,
) -> dict[str, Any]:
    service = model_service or default_model_service_command(manifest, device_name)

    # Chunk seeds across jobs first, then split each chunk into contiguous
    # runs (one --gumbel-policy-game invocation per run).
    job_count = max(1, jobs)
    chunk_size = (len(seeds) + job_count - 1) // job_count
    runs: list[tuple[int, int]] = []
    for start in range(0, len(seeds), max(1, chunk_size)):
        runs.extend(_contiguous_runs(seeds[start : start + chunk_size]))
    candidate_lines: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def run_slice(index: int, first_seed: int, count: int) -> list[dict[str, Any]]:
            return run_gumbel_games(
                binary,
                first_seed=first_seed,
                seed_count=count,
                model_service=service,
                model_manifest=manifest,
                out_path=tmp_path / f"gumbel_{index}.jsonl",
                n_simulations=n_simulations,
                top_m=top_m,
                depth_rounds=depth_rounds,
                determinizations=determinizations,
                blend_weight=blend_weight,
                k_interior=k_interior,
                max_root_actions=max_root_actions,
                rollout_max_actions=control_max_actions,
                rollout_top_k=control_rollout_top_k,
                model_timeout_ms=model_timeout_ms,
                exploration=False,
            )

        if jobs <= 1 or len(runs) == 1:
            for index, (first_seed, count) in enumerate(runs):
                candidate_lines.extend(run_slice(index, first_seed, count))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(run_slice, index, first_seed, count)
                    for index, (first_seed, count) in enumerate(runs)
                ]
                for future in as_completed(futures):
                    candidate_lines.extend(future.result())
    candidate_elapsed = time.perf_counter() - started
    candidate_results = sorted(collect_gumbel_results(candidate_lines), key=lambda r: r["seed"])

    control_results: list[dict[str, Any]] = []
    if control == "full-search":

        def run_control(seed: int) -> dict[str, Any]:
            return run_interactive_game(
                binary,
                seed=seed,
                strategy="full-search",
                checkpoint=None,
                selection_head="full-search",
                retain_k=control_max_actions,
                max_actions=control_max_actions,
                rollouts_per_action=control_rollouts_per_action,
                rollout_top_k=control_rollout_top_k,
                shadow_full_search=False,
                rollout_determinize=rollout_determinize,
            )

        if control_workers <= 1:
            control_results = [run_control(seed) for seed in seeds]
        else:
            with ThreadPoolExecutor(max_workers=control_workers) as executor:
                futures = {executor.submit(run_control, seed): seed for seed in seeds}
                for future in as_completed(futures):
                    control_results.append(future.result())
            control_results.sort(key=lambda result: int(result["seed"]))

    candidate_summary = summarize_game_results(candidate_results)
    control_summary = summarize_game_results(control_results) if control_results else None

    control_by_seed = {int(result["seed"]): result for result in control_results}
    paired_deltas = []
    for result in candidate_results:
        control_result = control_by_seed.get(int(result["seed"]))
        if control_result is None:
            continue
        candidate_mean = mean(float(score["total"]) for score in result["done"]["scores"])
        control_mean = mean(float(score["total"]) for score in control_result["done"]["scores"])
        paired_deltas.append(
            {
                "seed": int(result["seed"]),
                "gumbel_mean_score_per_seat": candidate_mean,
                "control_mean_score_per_seat": control_mean,
                "delta_gumbel_minus_control": candidate_mean - control_mean,
            }
        )
    delta_values = [row["delta_gumbel_minus_control"] for row in paired_deltas]
    stats = paired_delta_stats(delta_values)

    decision_seconds = [
        float(decision.get("decision_seconds", 0.0))
        for result in candidate_results
        for decision in result["decisions"]
    ]
    gate = None
    if stats["n"] and stats["mean"] is not None:
        gate = {
            "candidate_beats_control": bool(stats["mean"] > 0.0),
            "ci_excludes_zero": stats["ci_excludes_zero"],
            "promotable": bool(stats["mean"] > 0.0 and stats["ci_excludes_zero"]),
        }
    return {
        "status": "pass",
        "scientific_eligibility": "gumbel_search_vs_rollout_search_paired_benchmark",
        "experiment_id": experiment_id,
        "binary": str(binary),
        "manifest": str(manifest),
        "model_service": service,
        "seeds": seeds,
        "search": {
            "n_simulations": n_simulations,
            "top_m": top_m,
            "depth_rounds": depth_rounds,
            "determinizations": determinizations,
            "blend_weight": blend_weight,
            "k_interior": k_interior,
            "max_root_actions": max_root_actions,
        },
        "control": {
            "kind": control,
            "max_actions": control_max_actions,
            "rollouts_per_action": control_rollouts_per_action,
            "rollout_top_k": control_rollout_top_k,
            "rollout_determinize": rollout_determinize,
        },
        "strategies": {
            "gumbel-search": candidate_summary,
            "control": control_summary,
        },
        "candidate_decision_seconds_p50": _percentile(decision_seconds, 0.50),
        "candidate_decision_seconds_p95": _percentile(decision_seconds, 0.95),
        "candidate_wall_seconds": candidate_elapsed,
        "paired_score_deltas": paired_deltas,
        "paired_delta_stats": stats,
        "gate": gate,
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    candidate = report["strategies"]["gumbel-search"]
    control = report["strategies"].get("control")
    stats = report["paired_delta_stats"]
    lines = [
        "# Gumbel Search Benchmark",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Manifest: `{report['manifest']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Search: `{json.dumps(report['search'], sort_keys=True)}`",
        f"Control: `{json.dumps(report['control'], sort_keys=True)}`",
        "",
        "## Gumbel Search",
        "",
        f"- Mean seat score: `{candidate['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{candidate['p90_seat_score']:.4f}`",
        f"- Decisions: `{candidate['decisions']}`",
        f"- Decision seconds p50/p95: `{report['candidate_decision_seconds_p50']:.4f}` / `{report['candidate_decision_seconds_p95']:.4f}`",
    ]
    if control is not None:
        lines.extend(
            [
                "",
                "## Control",
                "",
                f"- Mean seat score: `{control['mean_seat_score']:.4f}`",
                f"- P90 seat score: `{control['p90_seat_score']:.4f}`",
                f"- Mean total decision seconds: `{control['mean_total_decision_seconds']:.4f}`",
                "",
                "## Paired Delta (gumbel - control)",
                "",
                f"- n: `{stats['n']}`",
                f"- Mean: `{stats['mean']}` (se `{stats['se']}`)",
                f"- 95% t-CI: `[{stats['t_ci_low']}, {stats['t_ci_high']}]`",
                f"- 95% bootstrap CI: `[{stats['bootstrap_ci_low']}, {stats['bootstrap_ci_high']}]`",
                f"- CI excludes zero: `{stats['ci_excludes_zero']}`",
                f"- Gate: `{json.dumps(report['gate'], sort_keys=True)}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-service", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--first-seed", type=int, default=2026995000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--gumbel-n-simulations", type=int, default=64)
    parser.add_argument("--gumbel-top-m", type=int, default=16)
    parser.add_argument("--gumbel-depth-rounds", type=int, default=1)
    parser.add_argument("--gumbel-determinizations", type=int, default=4)
    parser.add_argument("--gumbel-blend-weight", type=float, default=0.5)
    parser.add_argument("--k-interior", type=int, default=16)
    parser.add_argument("--gumbel-max-root-actions", type=int, default=0, help="0 keeps the full legal set")
    parser.add_argument("--control", choices=["full-search", "none"], default="full-search")
    parser.add_argument("--control-max-actions", type=int, default=64)
    parser.add_argument("--control-rollouts-per-action", type=int, default=16)
    parser.add_argument("--control-rollout-top-k", type=int, default=4)
    parser.add_argument("--control-workers", type=int, default=4)
    parser.add_argument(
        "--no-rollout-determinize",
        action="store_true",
        help="Run the control with the legacy hidden-order-peeking rollouts",
    )
    parser.add_argument("--model-timeout-ms", type=int, default=120_000)
    parser.add_argument("--experiment-id", default="gumbel-search-vs-rollout-search-v1")
    parser.add_argument("--out", default="cascadiav3/reports/gumbel_benchmark.json")
    parser.add_argument("--summary-out", default="cascadiav3/reports/gumbel_benchmark_summary.md")
    args = parser.parse_args()

    seeds = parse_seeds(seeds=args.seeds, first_seed=args.first_seed, games=args.games)
    report = run_gumbel_benchmark(
        binary=Path(args.binary),
        manifest=Path(args.manifest),
        seeds=seeds,
        model_service=args.model_service or None,
        device_name=args.device,
        jobs=args.jobs,
        n_simulations=args.gumbel_n_simulations,
        top_m=args.gumbel_top_m,
        depth_rounds=args.gumbel_depth_rounds,
        determinizations=args.gumbel_determinizations,
        blend_weight=args.gumbel_blend_weight,
        k_interior=args.k_interior,
        max_root_actions=args.gumbel_max_root_actions or None,
        control=args.control,
        control_max_actions=args.control_max_actions,
        control_rollouts_per_action=args.control_rollouts_per_action,
        control_rollout_top_k=args.control_rollout_top_k,
        control_workers=args.control_workers,
        rollout_determinize=not args.no_rollout_determinize,
        model_timeout_ms=args.model_timeout_ms,
        experiment_id=args.experiment_id,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps({key: report[key] for key in ("strategies", "paired_delta_stats", "gate")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
