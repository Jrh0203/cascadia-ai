"""Interactive game pilot for CRT prefilter search.

This is the first v3 bridge from dry-run root replay into complete games. A
Rust process owns the exact Cascadia simulator and streams one legal root at a
time. This controller scores that root with a fixed semantic vanilla
public-token seed ensemble, returns the retained top-K action ids, and lets the
Rust side run sampled rollout search inside the retained set.

When shadow full search is enabled, Rust evaluates all retained-32 candidates
with the same rollout samples before applying the filtered choice. That makes
the run a decision-level safety pilot: it reports how often K16 retained the
full-search winner along the actual model-filtered trajectory. Shadow runs are
not speed evidence because they intentionally pay the full-search CPU cost.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from .torch_prefilter_eval import _config_from_report
from .torch_public_token_merit import build_public_token_transformer
from .torch_relation_bias_merit import _public_scores, _to_device
from .torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots


def parse_csv_paths(raw: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one checkpoint path is required")
    return paths


def parse_seeds(*, seeds: str, first_seed: int, games: int) -> list[int]:
    if seeds.strip():
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--seeds did not contain any integers")
        return parsed
    if games <= 0:
        raise ValueError("--games must be positive")
    return [first_seed + offset for offset in range(games)]


def _normalize_scores(values):  # type: ignore[no-untyped-def]
    std = values.std(unbiased=False).clamp_min(1.0e-6)
    return (values - values.mean()) / std


def load_vanilla_ensemble(checkpoint_paths: list[Path], *, device_name: str):
    import torch

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    models = []
    checkpoint_reports = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = _config_from_report(checkpoint["report"])
        model = build_public_token_transformer(config)
        model.load_state_dict(checkpoint["vanilla_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)
        checkpoint_reports.append(
            {
                "path": str(checkpoint_path),
                "experiment_id": checkpoint["report"].get("experiment_id"),
                "seed": checkpoint["report"].get("seed"),
                "config": checkpoint["report"].get("config"),
            }
        )
    return {
        "device": device,
        "models": models,
        "checkpoint_reports": checkpoint_reports,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def rank_root_with_ensemble(root: dict[str, Any], ensemble: dict[str, Any]) -> dict[str, Any]:
    import torch

    batch = collate_semantic_relation_bias_roots([root])
    eval_batch = _to_device(batch, ensemble["device"])
    valid_count = int(batch["action_mask"][0].sum().item())
    action_ids = batch["action_ids"][0][:valid_count]
    combined = None
    member_scores: list[list[float]] = []
    with torch.no_grad():
        for model in ensemble["models"]:
            raw = _public_scores(model, eval_batch)[0, :valid_count].detach().cpu()
            member_scores.append([float(value) for value in raw.tolist()])
            normalized = _normalize_scores(raw)
            combined = normalized if combined is None else combined + normalized
    assert combined is not None
    combined = combined / max(1, len(ensemble["models"]))
    ranked = torch.argsort(combined, descending=True).tolist()
    return {
        "action_ids": action_ids,
        "ranked_indices": [int(index) for index in ranked],
        "ranked_action_ids": [action_ids[int(index)] for index in ranked],
        "ranked_ensemble_scores": [float(combined[int(index)].item()) for index in ranked],
        "member_scores": member_scores,
    }


def _binary_command(
    binary: Path,
    *,
    seed: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
) -> list[str]:
    return [
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


def run_interactive_game(
    binary: Path,
    *,
    seed: int,
    strategy: str,
    ensemble: dict[str, Any] | None,
    retain_k: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
    shadow_full_search: bool,
) -> dict[str, Any]:
    process = subprocess.Popen(
        _binary_command(
            binary,
            seed=seed,
            max_actions=max_actions,
            rollouts_per_action=rollouts_per_action,
            rollout_top_k=rollout_top_k,
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
            if strategy == "full-search":
                action_ids = [action["action_id"] for action in root["legal_actions"]]
                retained = action_ids
                pending_model[ply_index] = {
                    "model_score_seconds": 0.0,
                    "model_top_action_id": action_ids[0],
                    "model_ranked_action_ids": action_ids,
                }
            elif strategy == "prefilter-search":
                if ensemble is None:
                    raise ValueError("prefilter-search requires an ensemble")
                started = time.perf_counter()
                ranking = rank_root_with_ensemble(root, ensemble)
                model_score_seconds = time.perf_counter() - started
                retained = ranking["ranked_action_ids"][: min(retain_k, len(ranking["ranked_action_ids"]))]
                pending_model[ply_index] = {
                    "model_score_seconds": model_score_seconds,
                    "model_top_action_id": ranking["ranked_action_ids"][0],
                    "model_top_score": ranking["ranked_ensemble_scores"][0],
                    "model_ranked_action_ids": ranking["ranked_action_ids"],
                    "model_ranked_ensemble_scores": ranking["ranked_ensemble_scores"],
                }
            else:
                raise ValueError(f"unsupported strategy: {strategy}")
            request = {
                "retain_action_ids": retained,
                "shadow_full_search": bool(shadow_full_search),
            }
            process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
            process.stdin.flush()
        elif message_type == "decision":
            ply_index = int(message["ply_index"])
            model_data = pending_model.pop(ply_index, {})
            decisions.append({**message, **model_data, "strategy": strategy})
        elif message_type == "done":
            done = {**message, "strategy": strategy}
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
        "done": done,
        "decisions": decisions,
        "stderr": stderr,
    }


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
    retained_counts = [float(decision["retained_count"]) for decision in decisions]
    candidate_counts = [float(decision["candidate_count"]) for decision in decisions]
    model_seconds = [float(decision.get("model_score_seconds", 0.0)) for decision in decisions]
    search_seconds = [float(decision.get("decision_seconds", 0.0)) for decision in decisions]
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
        "mean_total_decision_seconds": (
            mean([model + search for model, search in zip(model_seconds, search_seconds, strict=True)])
            if model_seconds and search_seconds
            else 0.0
        ),
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


def paired_score_deltas(prefilter_results: list[dict[str, Any]], full_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_seed = {int(result["seed"]): result for result in full_results}
    deltas = []
    for result in prefilter_results:
        seed = int(result["seed"])
        full = full_by_seed.get(seed)
        if full is None:
            continue
        prefilter_mean = mean(float(score["total"]) for score in result["done"]["scores"])
        full_mean = mean(float(score["total"]) for score in full["done"]["scores"])
        deltas.append(
            {
                "seed": seed,
                "prefilter_mean_score_per_seat": prefilter_mean,
                "full_search_mean_score_per_seat": full_mean,
                "delta_prefilter_minus_full": prefilter_mean - full_mean,
            }
        )
    return deltas


def run_game_pilot(
    *,
    binary: Path,
    checkpoint_paths: list[Path],
    seeds: list[int],
    retain_k: int,
    max_actions: int,
    rollouts_per_action: int,
    rollout_top_k: int,
    shadow_full_search: bool,
    include_full_search_baseline: bool,
    full_baseline_workers: int,
    device_name: str,
    experiment_id: str,
    decision_rows_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if full_baseline_workers <= 0:
        raise ValueError("full_baseline_workers must be positive")
    ensemble = load_vanilla_ensemble(checkpoint_paths, device_name=device_name)
    if decision_rows_path is not None:
        decision_rows_path.parent.mkdir(parents=True, exist_ok=True)
        decision_rows_path.write_text("", encoding="utf-8")

    def remember(result: dict[str, Any]) -> dict[str, Any]:
        if decision_rows_path is not None:
            with decision_rows_path.open("a", encoding="utf-8") as handle:
                for decision in result["decisions"]:
                    handle.write(json.dumps(decision, sort_keys=True) + "\n")
        return result

    prefilter_results = []
    for seed in seeds:
        prefilter_results.append(remember(run_interactive_game(
            binary,
            seed=seed,
            strategy="prefilter-search",
            ensemble=ensemble,
            retain_k=retain_k,
            max_actions=max_actions,
            rollouts_per_action=rollouts_per_action,
            rollout_top_k=rollout_top_k,
            shadow_full_search=shadow_full_search,
        )))
    full_results: list[dict[str, Any]] = []
    if include_full_search_baseline:
        def run_full_baseline(full_seed: int) -> dict[str, Any]:
            return run_interactive_game(
                binary,
                seed=full_seed,
                strategy="full-search",
                ensemble=None,
                retain_k=max_actions,
                max_actions=max_actions,
                rollouts_per_action=rollouts_per_action,
                rollout_top_k=rollout_top_k,
                shadow_full_search=False,
            )

        if full_baseline_workers == 1:
            for seed in seeds:
                full_results.append(remember(run_full_baseline(seed)))
        else:
            with ThreadPoolExecutor(max_workers=full_baseline_workers) as executor:
                futures = {
                    executor.submit(run_full_baseline, seed): seed
                    for seed in seeds
                }
                for future in as_completed(futures):
                    full_results.append(remember(future.result()))
            full_results.sort(key=lambda result: int(result["seed"]))
    paired_deltas = paired_score_deltas(prefilter_results, full_results)
    return {
        "status": "pass",
        "scientific_eligibility": "interactive_prefilter_game_pilot",
        "experiment_id": experiment_id,
        "binary": str(binary),
        "seeds": seeds,
        "retain_k": retain_k,
        "max_actions": max_actions,
        "rollouts_per_action": rollouts_per_action,
        "rollout_top_k": rollout_top_k,
        "shadow_full_search": shadow_full_search,
        "include_full_search_baseline": include_full_search_baseline,
        "full_baseline_workers": full_baseline_workers,
        "runtime": {
            "device": str(ensemble["device"]),
            "device_name": ensemble["device_name"],
            "torch_version": ensemble["torch_version"],
            "torch_cuda": ensemble["torch_cuda"],
            "cuda_available": ensemble["cuda_available"],
        },
        "checkpoints": ensemble["checkpoint_reports"],
        "strategies": {
            "prefilter-search": summarize_game_results(prefilter_results),
            "full-search": summarize_game_results(full_results) if full_results else None,
        },
        "paired_score_deltas": paired_deltas,
        "mean_paired_delta_prefilter_minus_full": (
            mean(row["delta_prefilter_minus_full"] for row in paired_deltas)
            if paired_deltas
            else None
        ),
        "games": [
            {
                "seed": result["seed"],
                "strategy": result["strategy"],
                "scores": result["done"]["scores"],
                "turns": result["done"]["turns"],
                "elapsed_seconds": result["done"]["elapsed_seconds"],
                "decision_count": len(result["decisions"]),
                "final_state_hash": result["done"]["final_state_hash"],
            }
            for result in prefilter_results + full_results
        ],
    }, prefilter_results + full_results


def write_decision_rows(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            for decision in result["decisions"]:
                handle.write(json.dumps(decision, sort_keys=True) + "\n")


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    prefilter = report["strategies"]["prefilter-search"]
    full = report["strategies"].get("full-search")
    lines = [
        "# CRT Vanilla Prefilter Game Pilot",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",
        f"Retain K: `{report['retain_k']}` of max `{report['max_actions']}`",
        f"Rollouts/action: `{report['rollouts_per_action']}`",
        f"Shadow full search: `{report['shadow_full_search']}`",
        f"Full baseline workers: `{report['full_baseline_workers']}`",
        "",
        "## Prefilter Search",
        "",
        f"- Mean seat score: `{prefilter['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{prefilter['p90_seat_score']:.4f}`",
        f"- Decisions: `{prefilter['decisions']}`",
        f"- Mean total decision seconds: `{prefilter['mean_total_decision_seconds']:.4f}`",
        f"- Full-search winner retained rate: `{prefilter['shadow_full_best_retained_rate']}`",
        f"- Mean shadow search regret: `{prefilter['shadow_mean_search_regret']}`",
        f"- Estimated non-shadow rollout savings: `{prefilter['estimated_non_shadow_rollout_savings']:.4f}`",
        "",
    ]
    if full is not None:
        lines.extend(
            [
                "## Full Search Baseline",
                "",
                f"- Mean seat score: `{full['mean_seat_score']:.4f}`",
                f"- P90 seat score: `{full['p90_seat_score']:.4f}`",
                f"- Decisions: `{full['decisions']}`",
                f"- Mean total decision seconds: `{full['mean_total_decision_seconds']:.4f}`",
                f"- Mean paired delta prefilter-full: `{report['mean_paired_delta_prefilter_minus_full']}`",
                "",
            ]
        )
    lines.extend(["## Interpretation", ""])
    if report["shadow_full_search"]:
        lines.append(
            "This is an interactive v3 search-prefilter pilot. Shadow full search measures retained-set safety, not speed, because it deliberately evaluates all candidates for comparison."
        )
    else:
        lines.append(
            "This is a non-shadow interactive v3 search-prefilter pilot. The search timing is real for the retained candidate set; missed full-search winners are not measured in this mode."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated vanilla checkpoint paths")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--first-seed", type=int, default=2026160000)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--retain-k", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--rollouts-per-action", type=int, default=16)
    parser.add_argument("--rollout-top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shadow-full-search", action="store_true")
    parser.add_argument("--include-full-search-baseline", action="store_true")
    parser.add_argument("--full-baseline-workers", type=int, default=1)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p80x2-vanilla-prefilter-game-pilot-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.json")
    parser.add_argument("--decisions-out", default="cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_decisions.jsonl")
    parser.add_argument("--summary-out", default="cascadiav3/reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_summary.md")
    args = parser.parse_args()

    checkpoint_paths = parse_csv_paths(args.checkpoints)
    seeds = parse_seeds(seeds=args.seeds, first_seed=args.first_seed, games=args.games)
    report, raw_results = run_game_pilot(
        binary=Path(args.binary),
        checkpoint_paths=checkpoint_paths,
        seeds=seeds,
        retain_k=args.retain_k,
        max_actions=args.max_actions,
        rollouts_per_action=args.rollouts_per_action,
        rollout_top_k=args.rollout_top_k,
        shadow_full_search=args.shadow_full_search,
        include_full_search_baseline=args.include_full_search_baseline,
        full_baseline_workers=args.full_baseline_workers,
        device_name=args.device,
        experiment_id=args.experiment_id,
        decision_rows_path=Path(args.decisions_out),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not Path(args.decisions_out).exists():
        write_decision_rows(raw_results, Path(args.decisions_out))
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
