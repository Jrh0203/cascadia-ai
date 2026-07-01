"""Benchmark a greedy-policy pretrain checkpoint in complete simulator games.

The learned policy and the greedy baseline are both evaluated through the exact
Rust simulator. The learned policy receives each interactive root, scores all
legal actions with the checkpoint's policy head, and returns a single action id.
The greedy baseline returns the first simulator-ranked greedy action id.
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

from .torch_greedy_policy_pretrain import GreedyPolicyPretrainConfig
from .torch_public_token_merit import build_public_token_transformer
from .torch_relation_bias_merit import _to_device
from .torch_semantic_relation_bias_merit import collate_semantic_relation_bias_roots


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


def _config_from_checkpoint_report(report: dict[str, Any]) -> GreedyPolicyPretrainConfig:
    raw_config = dict(report.get("config", {}))
    allowed = GreedyPolicyPretrainConfig.__dataclass_fields__.keys()
    filtered = {key: value for key, value in raw_config.items() if key in allowed}
    return GreedyPolicyPretrainConfig(**filtered)


def load_policy_checkpoint(checkpoint_path: Path, *, device_name: str) -> dict[str, Any]:
    import torch

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = checkpoint.get("report", {})
    config = _config_from_checkpoint_report(report)
    model = build_public_token_transformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_report": {
            "experiment_id": report.get("experiment_id"),
            "seed": report.get("seed"),
            "steps": report.get("steps"),
            "batch_size": report.get("batch_size"),
            "train_record_count": report.get("train_record_count"),
            "val_record_count": report.get("val_record_count"),
            "config": report.get("config"),
        },
        "config": config,
        "device": device,
        "model": model,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def rank_root_with_policy(root: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    import torch

    batch = collate_semantic_relation_bias_roots([root])
    valid_count = int(batch["action_mask"][0].sum().item())
    action_ids = batch["action_ids"][0][:valid_count]
    eval_batch = _to_device(batch, policy["device"])
    with torch.no_grad():
        outputs = policy["model"](
            eval_batch["tokens"],
            eval_batch["token_mask"],
            eval_batch["actions"],
            eval_batch["action_mask"],
        )
        logits = outputs["logits"][0, :valid_count].detach().cpu()
        ranked = torch.argsort(logits, descending=True).tolist()
    return {
        "action_ids": action_ids,
        "ranked_indices": [int(index) for index in ranked],
        "ranked_action_ids": [action_ids[int(index)] for index in ranked],
        "ranked_logits": [float(logits[int(index)].item()) for index in ranked],
    }


def _binary_command(binary: Path, *, seed: int, max_actions: int) -> list[str]:
    return [
        str(binary),
        "--interactive-policy-game",
        "--first-seed",
        str(seed),
        "--max-actions",
        str(max_actions),
        "--rollouts-per-action",
        "0",
    ]


def run_interactive_game(
    binary: Path,
    *,
    seed: int,
    strategy: str,
    policy: dict[str, Any] | None,
    max_actions: int,
) -> dict[str, Any]:
    process = subprocess.Popen(
        _binary_command(binary, seed=seed, max_actions=max_actions),
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
            legal_action_ids = [action["action_id"] for action in root["legal_actions"]]
            if strategy == "greedy":
                selected_action_id = legal_action_ids[0]
                pending_model[ply_index] = {
                    "model_score_seconds": 0.0,
                    "model_top_action_id": selected_action_id,
                    "greedy_top_action_id": legal_action_ids[0],
                    "model_matches_greedy_top": True,
                }
            elif strategy == "learned-policy":
                if policy is None:
                    raise ValueError("learned-policy requires a loaded policy")
                started = time.perf_counter()
                ranking = rank_root_with_policy(root, policy)
                model_score_seconds = time.perf_counter() - started
                selected_action_id = ranking["ranked_action_ids"][0]
                pending_model[ply_index] = {
                    "model_score_seconds": model_score_seconds,
                    "model_top_action_id": selected_action_id,
                    "model_top_logit": ranking["ranked_logits"][0],
                    "greedy_top_action_id": legal_action_ids[0],
                    "model_matches_greedy_top": selected_action_id == legal_action_ids[0],
                    "greedy_rank_in_model": ranking["ranked_action_ids"].index(legal_action_ids[0]) + 1,
                }
            else:
                raise ValueError(f"unsupported strategy: {strategy}")
            process.stdin.write(json.dumps({"action_id": selected_action_id}, sort_keys=True) + "\n")
            process.stdin.flush()
        elif message_type == "decision":
            ply_index = int(message["ply_index"])
            decisions.append({**message, **pending_model.pop(ply_index, {}), "strategy": strategy})
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


def summarize_game_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    seat_scores = [float(score["total"]) for result in results for score in result["done"]["scores"]]
    per_game_mean_scores = [
        mean(float(score["total"]) for score in result["done"]["scores"])
        for result in results
    ]
    decisions = [decision for result in results for decision in result["decisions"]]
    model_seconds = [float(decision.get("model_score_seconds", 0.0)) for decision in decisions]
    matches = [bool(decision.get("model_matches_greedy_top")) for decision in decisions]
    greedy_ranks = [
        int(decision["greedy_rank_in_model"])
        for decision in decisions
        if "greedy_rank_in_model" in decision
    ]
    return {
        "games": len(results),
        "decisions": len(decisions),
        "mean_seat_score": mean(seat_scores) if seat_scores else 0.0,
        "p50_seat_score": _percentile(seat_scores, 0.50),
        "p90_seat_score": _percentile(seat_scores, 0.90),
        "mean_game_score_per_seat": mean(per_game_mean_scores) if per_game_mean_scores else 0.0,
        "mean_model_score_seconds": mean(model_seconds) if model_seconds else 0.0,
        "action_match_rate_vs_greedy_top": (
            sum(1 for value in matches if value) / len(matches)
            if matches
            else None
        ),
        "mean_greedy_rank_in_model": mean(greedy_ranks) if greedy_ranks else None,
    }


def paired_score_deltas(model_results: list[dict[str, Any]], greedy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    greedy_by_seed = {int(result["seed"]): result for result in greedy_results}
    rows = []
    for result in model_results:
        seed = int(result["seed"])
        greedy = greedy_by_seed.get(seed)
        if greedy is None:
            continue
        model_mean = mean(float(score["total"]) for score in result["done"]["scores"])
        greedy_mean = mean(float(score["total"]) for score in greedy["done"]["scores"])
        rows.append(
            {
                "seed": seed,
                "learned_policy_mean_score_per_seat": model_mean,
                "greedy_mean_score_per_seat": greedy_mean,
                "delta_learned_minus_greedy": model_mean - greedy_mean,
            }
        )
    return rows


def run_benchmark(
    *,
    binary: Path,
    checkpoint_path: Path,
    seeds: list[int],
    max_actions: int,
    baseline_workers: int,
    device_name: str,
    experiment_id: str,
    decision_rows_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if baseline_workers <= 0:
        raise ValueError("--baseline-workers must be positive")
    policy = load_policy_checkpoint(checkpoint_path, device_name=device_name)
    if decision_rows_path is not None:
        decision_rows_path.parent.mkdir(parents=True, exist_ok=True)
        decision_rows_path.write_text("", encoding="utf-8")

    def remember(result: dict[str, Any]) -> dict[str, Any]:
        if decision_rows_path is not None:
            with decision_rows_path.open("a", encoding="utf-8") as handle:
                for decision in result["decisions"]:
                    handle.write(json.dumps(decision, sort_keys=True) + "\n")
        return result

    learned_results = [
        remember(run_interactive_game(
            binary,
            seed=seed,
            strategy="learned-policy",
            policy=policy,
            max_actions=max_actions,
        ))
        for seed in seeds
    ]

    def run_greedy(seed: int) -> dict[str, Any]:
        return run_interactive_game(
            binary,
            seed=seed,
            strategy="greedy",
            policy=None,
            max_actions=max_actions,
        )

    greedy_results: list[dict[str, Any]] = []
    if baseline_workers == 1:
        for seed in seeds:
            greedy_results.append(remember(run_greedy(seed)))
    else:
        with ThreadPoolExecutor(max_workers=baseline_workers) as executor:
            futures = {executor.submit(run_greedy, seed): seed for seed in seeds}
            for future in as_completed(futures):
                greedy_results.append(remember(future.result()))
        greedy_results.sort(key=lambda result: int(result["seed"]))

    paired_deltas = paired_score_deltas(learned_results, greedy_results)
    return {
        "status": "pass",
        "scientific_eligibility": "greedy_policy_complete_game_benchmark",
        "experiment_id": experiment_id,
        "binary": str(binary),
        "checkpoint": policy["checkpoint_path"],
        "checkpoint_report": policy["checkpoint_report"],
        "seeds": seeds,
        "max_actions": max_actions,
        "baseline_workers": baseline_workers,
        "runtime": {
            "device": str(policy["device"]),
            "device_name": policy["device_name"],
            "torch_version": policy["torch_version"],
            "torch_cuda": policy["torch_cuda"],
            "cuda_available": policy["cuda_available"],
        },
        "strategies": {
            "learned-policy": summarize_game_results(learned_results),
            "greedy": summarize_game_results(greedy_results),
        },
        "paired_score_deltas": paired_deltas,
        "mean_paired_delta_learned_minus_greedy": (
            mean(row["delta_learned_minus_greedy"] for row in paired_deltas)
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
            for result in learned_results + greedy_results
        ],
    }, learned_results + greedy_results


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    learned = report["strategies"]["learned-policy"]
    greedy = report["strategies"]["greedy"]
    lines = [
        "# Greedy Policy Complete-Game Benchmark",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Device: `{report['runtime']['device_name']}`",
        "",
        "## Learned Policy",
        "",
        f"- Mean seat score: `{learned['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{learned['p90_seat_score']:.4f}`",
        f"- Action match rate vs greedy top: `{learned['action_match_rate_vs_greedy_top']}`",
        f"- Mean greedy rank in model: `{learned['mean_greedy_rank_in_model']}`",
        "",
        "## Greedy Baseline",
        "",
        f"- Mean seat score: `{greedy['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{greedy['p90_seat_score']:.4f}`",
        "",
        "## Paired Delta",
        "",
        f"- Mean learned minus greedy: `{report['mean_paired_delta_learned_minus_greedy']}`",
        "",
        "This is a no-search complete-game benchmark. It measures whether the behavior-cloned transformer can execute the greedy policy strongly enough to match or exceed greedy play, not whether it is superhuman.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--first-seed", type=int, default=2026990000)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--baseline-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-id", default="greedy-policy-complete-game-benchmark-v1")
    parser.add_argument("--out", default="cascadiav3/reports/greedy_policy_game_benchmark.json")
    parser.add_argument("--decisions-out", default="cascadiav3/reports/greedy_policy_game_benchmark_decisions.jsonl")
    parser.add_argument("--summary-out", default="cascadiav3/reports/greedy_policy_game_benchmark_summary.md")
    args = parser.parse_args()

    seeds = parse_seeds(seeds=args.seeds, first_seed=args.first_seed, games=args.games)
    report, _raw_results = run_benchmark(
        binary=Path(args.binary),
        checkpoint_path=Path(args.checkpoint),
        seeds=seeds,
        max_actions=args.max_actions,
        baseline_workers=args.baseline_workers,
        device_name=args.device,
        experiment_id=args.experiment_id,
        decision_rows_path=Path(args.decisions_out),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
