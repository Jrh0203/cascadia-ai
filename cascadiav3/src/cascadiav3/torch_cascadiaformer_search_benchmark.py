"""Search-integrated complete-game benchmark for CascadiaFormer checkpoints.

Rust owns the exact simulator and sampled search. This controller scores each
public root with a CascadiaFormer checkpoint, retains the model top-K action ids,
and asks Rust to run rollout search inside that retained set. A matched
full-search control can be run on the same seeds for promotion-style timing and
score comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from .torch_benchmark_stats import paired_delta_stats
from .torch_cascadiaformer_game_benchmark import completed_game_result_row, load_cascadiaformer_manifest, rank_root_with_model


def parse_seeds(*, seeds: str, first_seed: int, games: int) -> list[int]:
    if seeds.strip():
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--seeds did not contain any integers")
        return parsed
    if games <= 0:
        raise ValueError("--games must be positive")
    return [first_seed + offset for offset in range(games)]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _binary_command(
    binary: Path,
    *,
    seed: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
    rollout_determinize: bool = False,
    scoring_cards: str = "aaaaa",
) -> list[str]:
    command = [
        str(binary),
        "--interactive-policy-game",
        "--first-seed",
        str(seed),
        "--max-actions",
        str(max_actions),
        "--rollouts-per-action",
        str(rollouts_per_action),
        "--rollout-top-k",
        str(rollout_top_k),
    ]
    if rollout_determinize:
        command.append("--rollout-determinize")
    # Emitted only when non-default so default invocations stay replayable
    # against older pinned binaries.
    if scoring_cards != "aaaaa":
        command.extend(["--scoring-cards", scoring_cards])
    return command


def run_interactive_game(
    binary: Path,
    *,
    seed: int,
    strategy: str,
    checkpoint: dict[str, Any] | None,
    selection_head: str,
    retain_k: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
    shadow_full_search: bool,
    model_lock: threading.Lock | None = None,
    rollout_determinize: bool = False,
    scoring_cards: str = "aaaaa",
) -> dict[str, Any]:
    process = subprocess.Popen(
        _binary_command(
            binary,
            seed=seed,
            max_actions=max_actions,
            rollouts_per_action=rollouts_per_action,
            rollout_top_k=rollout_top_k,
            rollout_determinize=rollout_determinize,
            scoring_cards=scoring_cards,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open interactive simulator pipes")

    decisions: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    pending_model: dict[int, dict[str, Any]] = {}
    while True:
        line = process.stdout.readline()
        if not line:
            break
        message = json.loads(line)
        message_type = message.get("type")
        if message_type == "root":
            root = message["root"]
            ply_index = int(message["ply_index"])
            action_ids = [action["action_id"] for action in root["legal_actions"]]
            if strategy == "full-search":
                retained = action_ids
                pending_model[ply_index] = {
                    "model_score_seconds": 0.0,
                    "model_top_action_id": action_ids[0] if action_ids else None,
                    "model_ranked_action_ids": action_ids,
                    "selection_head": "full-search",
                }
            elif strategy == "cascadiaformer-search":
                if checkpoint is None:
                    raise ValueError("cascadiaformer-search requires a loaded checkpoint")
                started = time.perf_counter()
                if model_lock is None:
                    ranking = rank_root_with_model(root, checkpoint, selection_head=selection_head)
                else:
                    with model_lock:
                        ranking = rank_root_with_model(root, checkpoint, selection_head=selection_head)
                model_score_seconds = time.perf_counter() - started
                retained = ranking["ranked_action_ids"][: min(retain_k, len(ranking["ranked_action_ids"]))]
                pending_model[ply_index] = {
                    "model_score_seconds": model_score_seconds,
                    "model_top_action_id": ranking["ranked_action_ids"][0] if ranking["ranked_action_ids"] else None,
                    "model_top_logit": ranking["ranked_logits"][0] if ranking["ranked_logits"] else None,
                    "model_top_q": ranking["ranked_q"][0] if ranking["ranked_q"] else None,
                    "model_top_score_to_go": ranking["ranked_score_to_go"][0] if ranking["ranked_score_to_go"] else None,
                    "model_ranked_action_ids": ranking["ranked_action_ids"],
                    "selection_head": selection_head,
                }
            else:
                raise ValueError(f"unsupported strategy: {strategy}")
            process.stdin.write(
                json.dumps(
                    {
                        "retain_action_ids": retained,
                        "shadow_full_search": bool(shadow_full_search),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            process.stdin.flush()
        elif message_type == "decision":
            ply_index = int(message["ply_index"])
            decisions.append({**message, **pending_model.pop(ply_index, {}), "strategy": strategy})
        elif message_type == "done":
            done = {**message, "strategy": strategy, "selection_head": selection_head}
            break
        else:
            raise RuntimeError(f"unknown simulator message: {message}")

    return_code = process.wait()
    stderr = process.stderr.read() if process.stderr is not None else ""
    if return_code != 0:
        raise RuntimeError(f"interactive simulator exited {return_code}: {stderr}")
    if done is None:
        raise RuntimeError(f"interactive simulator ended without done message: {stderr}")
    return {
        "seed": seed,
        "strategy": strategy,
        "selection_head": selection_head,
        "done": done,
        "decisions": decisions,
        "stderr": stderr,
    }


def summarize_game_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    seat_scores = [
        float(score["total"])
        for result in results
        for score in result["done"]["scores"]
    ]
    per_game_mean_scores = [
        mean(float(score["total"]) for score in result["done"]["scores"])
        for result in results
    ]
    decisions = [decision for result in results for decision in result["decisions"]]
    model_seconds = [float(decision.get("model_score_seconds", 0.0)) for decision in decisions]
    search_seconds = [float(decision.get("decision_seconds", 0.0)) for decision in decisions]
    retained_counts = [float(decision.get("retained_count", 0.0)) for decision in decisions]
    candidate_counts = [float(decision.get("candidate_count", 0.0)) for decision in decisions]
    regrets = [
        float(decision["search_regret"])
        for decision in decisions
        if decision.get("search_regret") is not None
    ]
    retained_hits = [
        bool(decision["full_best_retained"])
        for decision in decisions
        if decision.get("full_best_retained") is not None
    ]
    total_seconds = [
        model + search
        for model, search in zip(model_seconds, search_seconds, strict=True)
    ]
    rollout_fraction = (
        mean(retained_counts) / mean(candidate_counts)
        if retained_counts and candidate_counts and mean(candidate_counts) > 0.0
        else 0.0
    )
    return {
        "games": len(results),
        "decisions": len(decisions),
        "mean_seat_score": mean(seat_scores) if seat_scores else 0.0,
        "p50_seat_score": _percentile(seat_scores, 0.50),
        "p90_seat_score": _percentile(seat_scores, 0.90),
        "mean_game_score_per_seat": mean(per_game_mean_scores) if per_game_mean_scores else 0.0,
        "mean_model_score_seconds": mean(model_seconds) if model_seconds else 0.0,
        "mean_search_seconds": mean(search_seconds) if search_seconds else 0.0,
        "mean_total_decision_seconds": mean(total_seconds) if total_seconds else 0.0,
        "mean_candidate_count": mean(candidate_counts) if candidate_counts else 0.0,
        "mean_retained_count": mean(retained_counts) if retained_counts else 0.0,
        "estimated_non_shadow_rollout_fraction": rollout_fraction,
        "estimated_non_shadow_rollout_savings": 1.0 - rollout_fraction,
        "shadow_full_best_retained_rate": (
            sum(1 for value in retained_hits if value) / len(retained_hits)
            if retained_hits
            else None
        ),
        "shadow_mean_search_regret": mean(regrets) if regrets else None,
        "shadow_p95_search_regret": _percentile(regrets, 0.95) if regrets else None,
        "shadow_zero_regret_rate": (
            sum(1 for value in regrets if abs(value) <= 1.0e-9) / len(regrets)
            if regrets
            else None
        ),
    }


def paired_score_deltas(candidate_results: list[dict[str, Any]], full_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_seed = {int(result["seed"]): result for result in full_results}
    deltas = []
    for result in candidate_results:
        seed = int(result["seed"])
        full = full_by_seed.get(seed)
        if full is None:
            continue
        candidate_mean = mean(float(score["total"]) for score in result["done"]["scores"])
        full_mean = mean(float(score["total"]) for score in full["done"]["scores"])
        deltas.append(
            {
                "seed": seed,
                "selection_head": result["selection_head"],
                "candidate_mean_score_per_seat": candidate_mean,
                "full_search_mean_score_per_seat": full_mean,
                "delta_candidate_minus_full_search": candidate_mean - full_mean,
            }
        )
    return deltas


def run_search_benchmark(
    *,
    binary: Path,
    manifest: Path,
    seeds: list[int],
    selection_head: str,
    retain_k: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
    shadow_full_search: bool,
    include_full_search_baseline: bool,
    candidate_workers: int,
    full_baseline_workers: int,
    device_name: str,
    experiment_id: str,
    decision_rows_path: Path | None,
    game_results_path: Path | None = None,
    rollout_determinize: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if selection_head not in {"policy", "q"}:
        raise ValueError("selection_head must be policy or q")
    if candidate_workers <= 0:
        raise ValueError("--candidate-workers must be positive")
    if full_baseline_workers <= 0:
        raise ValueError("--full-baseline-workers must be positive")
    if retain_k <= 0 or retain_k > max_actions:
        raise ValueError("--retain-k must be in [1, max-actions]")
    checkpoint = load_cascadiaformer_manifest(manifest, device_name=device_name)
    if decision_rows_path is not None:
        decision_rows_path.parent.mkdir(parents=True, exist_ok=True)
        decision_rows_path.write_text("", encoding="utf-8")
    if game_results_path is not None:
        game_results_path.parent.mkdir(parents=True, exist_ok=True)
        game_results_path.write_text("", encoding="utf-8")

    def remember(result: dict[str, Any]) -> dict[str, Any]:
        if decision_rows_path is not None:
            with decision_rows_path.open("a", encoding="utf-8") as handle:
                for decision in result["decisions"]:
                    handle.write(json.dumps(decision, sort_keys=True) + "\n")
        if game_results_path is not None:
            with game_results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(completed_game_result_row(result), sort_keys=True) + "\n")
        return result

    model_lock = threading.Lock()

    def run_candidate(seed: int) -> dict[str, Any]:
        return run_interactive_game(
            binary,
            seed=seed,
            strategy="cascadiaformer-search",
            checkpoint=checkpoint,
            selection_head=selection_head,
            retain_k=retain_k,
            max_actions=max_actions,
            rollouts_per_action=rollouts_per_action,
            rollout_top_k=rollout_top_k,
            shadow_full_search=shadow_full_search,
            model_lock=model_lock,
            rollout_determinize=rollout_determinize,
        )

    candidate_results: list[dict[str, Any]] = []
    if candidate_workers == 1:
        for seed in seeds:
            candidate_results.append(remember(run_candidate(seed)))
    else:
        with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
            futures = {executor.submit(run_candidate, seed): seed for seed in seeds}
            for future in as_completed(futures):
                candidate_results.append(remember(future.result()))
        candidate_results.sort(key=lambda result: int(result["seed"]))

    def run_full_baseline(seed: int) -> dict[str, Any]:
        return run_interactive_game(
            binary,
            seed=seed,
            strategy="full-search",
            checkpoint=None,
            selection_head="full-search",
            retain_k=max_actions,
            max_actions=max_actions,
            rollouts_per_action=rollouts_per_action,
            rollout_top_k=rollout_top_k,
            shadow_full_search=False,
            rollout_determinize=rollout_determinize,
        )

    full_results: list[dict[str, Any]] = []
    if include_full_search_baseline:
        if full_baseline_workers == 1:
            for seed in seeds:
                full_results.append(remember(run_full_baseline(seed)))
        else:
            with ThreadPoolExecutor(max_workers=full_baseline_workers) as executor:
                futures = {executor.submit(run_full_baseline, seed): seed for seed in seeds}
                for future in as_completed(futures):
                    full_results.append(remember(future.result()))
            full_results.sort(key=lambda result: int(result["seed"]))

    candidate_summary = summarize_game_results(candidate_results)
    full_summary = summarize_game_results(full_results) if full_results else None
    paired_deltas = paired_score_deltas(candidate_results, full_results)
    mean_delta = (
        mean(row["delta_candidate_minus_full_search"] for row in paired_deltas)
        if paired_deltas
        else None
    )
    timing_ratio = None
    if full_summary and full_summary["mean_total_decision_seconds"] > 0:
        timing_ratio = candidate_summary["mean_total_decision_seconds"] / full_summary["mean_total_decision_seconds"]
    return {
        "status": "pass",
        "scientific_eligibility": "cascadiaformer_search_integrated_complete_game_benchmark",
        "experiment_id": experiment_id,
        "binary": str(binary),
        "manifest": checkpoint["manifest_path"],
        "checkpoint": {
            "checkpoint_tag": checkpoint["manifest"].get("checkpoint_tag"),
            "step": checkpoint["manifest"].get("step"),
            "weights": checkpoint["manifest"].get("weights"),
            "training_limited_metrics": checkpoint["manifest"].get("metrics"),
        },
        "seeds": seeds,
        "selection_head": selection_head,
        "retain_k": retain_k,
        "max_actions": max_actions,
        "rollouts_per_action": rollouts_per_action,
        "rollout_top_k": rollout_top_k,
        "shadow_full_search": shadow_full_search,
        "include_full_search_baseline": include_full_search_baseline,
        "candidate_workers": candidate_workers,
        "full_baseline_workers": full_baseline_workers,
        "runtime": {
            "device": str(checkpoint["device"]),
            "device_name": checkpoint["device_name"],
            "torch_version": checkpoint["torch_version"],
            "torch_cuda": checkpoint["torch_cuda"],
            "cuda_available": checkpoint["cuda_available"],
        },
        "strategies": {
            "cascadiaformer-search": candidate_summary,
            "full-search": full_summary,
        },
        "rollout_determinize": rollout_determinize,
        "paired_score_deltas": paired_deltas,
        "mean_paired_delta_candidate_minus_full_search": mean_delta,
        "paired_delta_stats": paired_delta_stats(
            [row["delta_candidate_minus_full_search"] for row in paired_deltas]
        ),
        "treatment_mean_decision_seconds": candidate_summary["mean_total_decision_seconds"],
        "control_mean_decision_seconds": full_summary["mean_total_decision_seconds"] if full_summary else None,
        "treatment_control_time_ratio": timing_ratio,
        "games": [
            completed_game_result_row(result)
            for result in candidate_results + full_results
        ],
    }, candidate_results + full_results


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    candidate = report["strategies"]["cascadiaformer-search"]
    full = report["strategies"].get("full-search")
    lines = [
        "# CascadiaFormer Search Benchmark",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Manifest: `{report['manifest']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Selection head: `{report['selection_head']}`",
        f"Retain K: `{report['retain_k']}` of max `{report['max_actions']}`",
        f"Rollouts/action: `{report['rollouts_per_action']}`",
        f"Rollout top-k: `{report['rollout_top_k']}`",
        f"Shadow full search: `{report['shadow_full_search']}`",
        f"Candidate workers: `{report['candidate_workers']}`",
        f"Full baseline workers: `{report['full_baseline_workers']}`",
        f"Device: `{report['runtime']['device_name']}`",
        "",
        "## CascadiaFormer Search",
        "",
        f"- Mean seat score: `{candidate['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{candidate['p90_seat_score']:.4f}`",
        f"- Decisions: `{candidate['decisions']}`",
        f"- Mean total decision seconds: `{candidate['mean_total_decision_seconds']:.4f}`",
        f"- Full-search winner retained rate: `{candidate['shadow_full_best_retained_rate']}`",
        f"- Mean shadow search regret: `{candidate['shadow_mean_search_regret']}`",
        f"- Estimated non-shadow rollout savings: `{candidate['estimated_non_shadow_rollout_savings']:.4f}`",
    ]
    if full is not None:
        lines.extend(
            [
                "",
                "## Full Search Control",
                "",
                f"- Mean seat score: `{full['mean_seat_score']:.4f}`",
                f"- P90 seat score: `{full['p90_seat_score']:.4f}`",
                f"- Decisions: `{full['decisions']}`",
                f"- Mean total decision seconds: `{full['mean_total_decision_seconds']:.4f}`",
                f"- Mean paired delta candidate-control: `{report['mean_paired_delta_candidate_minus_full_search']}`",
                f"- Paired delta 95% t-CI: `[{report['paired_delta_stats'].get('t_ci_low')}, {report['paired_delta_stats'].get('t_ci_high')}]`"
                f" (n={report['paired_delta_stats'].get('n')}, se={report['paired_delta_stats'].get('se')})",
                f"- CI excludes zero: `{report['paired_delta_stats'].get('ci_excludes_zero')}`",
                f"- Treatment/control time ratio: `{report['treatment_control_time_ratio']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This is the search-integrated v3 gate: model-ranked retained sets feed sampled search, and promotion still requires paired gameplay evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-head", choices=["policy", "q"], default="q")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--first-seed", type=int, default=2026995000)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--retain-k", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--rollouts-per-action", type=int, default=16)
    parser.add_argument("--rollout-top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rollout-determinize",
        action="store_true",
        help="Public-information-legal rollouts: resample hidden order per rollout",
    )
    parser.add_argument("--shadow-full-search", action="store_true")
    parser.add_argument("--include-full-search-baseline", action="store_true")
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--full-baseline-workers", type=int, default=4)
    parser.add_argument("--experiment-id", default="cascadiaformer-search-integrated-benchmark-v1")
    parser.add_argument("--out", default="cascadiav3/reports/cascadiaformer_search_benchmark.json")
    parser.add_argument("--decisions-out", default="cascadiav3/reports/cascadiaformer_search_benchmark_decisions.jsonl")
    parser.add_argument("--game-results-out", default="cascadiav3/reports/cascadiaformer_search_benchmark_games.jsonl")
    parser.add_argument("--summary-out", default="cascadiav3/reports/cascadiaformer_search_benchmark_summary.md")
    args = parser.parse_args()

    seeds = parse_seeds(seeds=args.seeds, first_seed=args.first_seed, games=args.games)
    report, _raw_results = run_search_benchmark(
        binary=Path(args.binary),
        manifest=Path(args.manifest),
        seeds=seeds,
        selection_head=args.selection_head,
        retain_k=args.retain_k,
        max_actions=args.max_actions,
        rollouts_per_action=args.rollouts_per_action,
        rollout_top_k=args.rollout_top_k,
        shadow_full_search=args.shadow_full_search,
        include_full_search_baseline=args.include_full_search_baseline,
        candidate_workers=args.candidate_workers,
        full_baseline_workers=args.full_baseline_workers,
        device_name=args.device,
        experiment_id=args.experiment_id,
        decision_rows_path=Path(args.decisions_out),
        game_results_path=Path(args.game_results_out),
        rollout_determinize=args.rollout_determinize,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
