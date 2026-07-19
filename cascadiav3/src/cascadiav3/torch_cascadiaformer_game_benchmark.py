"""Benchmark CascadiaFormer checkpoints in complete simulator games.

This is a no-search first-merit check. The model receives each public
interactive root from the Rust simulator, chooses one action by either policy
logit or Q value, and is paired against the simulator's greedy top action on
the same seeds.
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
from .torch_inference_bridge import _load_model, collate_inference_roots

# Ruleset identity resolved from --scoring-cards; must stay in lockstep with
# the exporter's RULESET_ID_AAAAA / RULESET_ID_CBDDB constants.
RULESET_IDS_BY_SCORING_CARDS = {
    "aaaaa": "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16",
    "cbddb": "cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19",
}
EXPECTED_RULESET_ID = RULESET_IDS_BY_SCORING_CARDS["aaaaa"]


def expected_ruleset_id_for(scoring_cards: str) -> str:
    if scoring_cards not in RULESET_IDS_BY_SCORING_CARDS:
        raise ValueError(
            f"unknown scoring cards {scoring_cards!r}; "
            f"expected one of {sorted(RULESET_IDS_BY_SCORING_CARDS)}"
        )
    return RULESET_IDS_BY_SCORING_CARDS[scoring_cards]


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


def _to_device(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def load_cascadiaformer_manifest(manifest_path: Path, *, device_name: str) -> dict[str, Any]:
    import torch

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = _load_model(manifest_path, manifest_path=manifest_path, manifest_payload=payload).to(device)
    model.eval()
    return {
        "manifest_path": str(manifest_path),
        "manifest": payload,
        "device": device,
        "model": model,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def rank_root_with_model(
    root: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    selection_head: str,
) -> dict[str, Any]:
    import torch

    if selection_head not in {"policy", "q"}:
        raise ValueError("selection_head must be policy or q")
    batch = collate_inference_roots([root])
    valid_count = int(batch["action_mask"][0].sum().item())
    action_ids = batch["action_ids"][0][:valid_count]
    eval_batch = _to_device(batch, checkpoint["device"])
    with torch.no_grad():
        outputs = checkpoint["model"](
            eval_batch["tokens"],
            eval_batch["token_mask"],
            eval_batch["actions"],
            eval_batch["action_mask"],
            relation_ids=eval_batch.get("relation_ids"),
        )
        logits = outputs["logits"][0, :valid_count].detach().cpu()
        score_to_go = outputs["q"][0, :valid_count].detach().cpu()
        exact_afterstate = batch["exact_afterstate_score_active"][0, :valid_count].detach().cpu()
        q_values = exact_afterstate + score_to_go
        scores = logits if selection_head == "policy" else q_values
        ranked = torch.argsort(scores, descending=True).tolist()
    return {
        "action_ids": action_ids,
        "ranked_indices": [int(index) for index in ranked],
        "ranked_action_ids": [action_ids[int(index)] for index in ranked],
        "ranked_logits": [float(logits[int(index)].item()) for index in ranked],
        "ranked_q": [float(q_values[int(index)].item()) for index in ranked],
        "ranked_score_to_go": [float(score_to_go[int(index)].item()) for index in ranked],
        "ranked_exact_afterstate_score_active": [float(exact_afterstate[int(index)].item()) for index in ranked],
        "selection_head": selection_head,
    }


def _binary_command(
    binary: Path, *, seed: int, max_actions: int, scoring_cards: str = "aaaaa"
) -> list[str]:
    command = [
        str(binary),
        "--interactive-policy-game",
        "--first-seed",
        str(seed),
        "--max-actions",
        str(max_actions),
        "--rollouts-per-action",
        "0",
    ]
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
    max_actions: int,
    model_lock: threading.Lock | None = None,
    scoring_cards: str = "aaaaa",
) -> dict[str, Any]:
    process = subprocess.Popen(
        _binary_command(
            binary, seed=seed, max_actions=max_actions, scoring_cards=scoring_cards
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
            legal_action_ids = [action["action_id"] for action in root["legal_actions"]]
            if strategy == "greedy":
                selected_action_id = legal_action_ids[0]
                pending_model[ply_index] = {
                    "model_score_seconds": 0.0,
                    "model_top_action_id": selected_action_id,
                    "greedy_top_action_id": legal_action_ids[0],
                    "model_matches_greedy_top": True,
                    "selection_head": "greedy",
                }
            elif strategy == "cascadiaformer":
                if checkpoint is None:
                    raise ValueError("cascadiaformer strategy requires a loaded checkpoint")
                started = time.perf_counter()
                if model_lock is None:
                    ranking = rank_root_with_model(root, checkpoint, selection_head=selection_head)
                else:
                    with model_lock:
                        ranking = rank_root_with_model(root, checkpoint, selection_head=selection_head)
                model_score_seconds = time.perf_counter() - started
                selected_action_id = ranking["ranked_action_ids"][0]
                pending_model[ply_index] = {
                    "model_score_seconds": model_score_seconds,
                    "model_top_action_id": selected_action_id,
                    "model_top_logit": ranking["ranked_logits"][0],
                    "model_top_q": ranking["ranked_q"][0],
                    "greedy_top_action_id": legal_action_ids[0],
                    "model_matches_greedy_top": selected_action_id == legal_action_ids[0],
                    "greedy_rank_in_model": ranking["ranked_action_ids"].index(legal_action_ids[0]) + 1,
                    "selection_head": selection_head,
                }
            else:
                raise ValueError(f"unsupported strategy: {strategy}")
            process.stdin.write(json.dumps({"action_id": selected_action_id}, sort_keys=True) + "\n")
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
                "selection_head": result["selection_head"],
                "cascadiaformer_mean_score_per_seat": model_mean,
                "greedy_mean_score_per_seat": greedy_mean,
                "delta_cascadiaformer_minus_greedy": model_mean - greedy_mean,
            }
        )
    return rows


def completed_game_result_row(result: dict[str, Any]) -> dict[str, Any]:
    seat_scores = [float(score["total"]) for score in result["done"]["scores"]]
    return {
        "seed": result["seed"],
        "strategy": result["strategy"],
        "selection_head": result["selection_head"],
        "scores": result["done"]["scores"],
        "seat_scores": seat_scores,
        "mean_score_per_seat": mean(seat_scores) if seat_scores else 0.0,
        "turns": result["done"]["turns"],
        "elapsed_seconds": result["done"]["elapsed_seconds"],
        "decision_count": len(result["decisions"]),
        "final_state_hash": result["done"]["final_state_hash"],
    }


def summarize_market_decisions(results: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [decision for result in results for decision in result["decisions"]]
    counts = {choice: 0 for choice in ("accept", "decline", "not_available")}
    for decision in decisions:
        choice = str(decision.get("free_three_of_a_kind_choice", "not_available"))
        if choice not in counts:
            raise RuntimeError(f"unknown free-three-of-a-kind choice: {choice}")
        counts[choice] += 1
    available = counts["accept"] + counts["decline"]
    return {
        "total_decisions": len(decisions),
        "available_decisions": available,
        "accepted": counts["accept"],
        "declined": counts["decline"],
        "not_available": counts["not_available"],
        "acceptance_rate_when_available": counts["accept"] / available if available else None,
    }


def run_benchmark(
    *,
    binary: Path,
    manifest: Path,
    seeds: list[int],
    selection_heads: list[str],
    max_actions: int,
    baseline_workers: int,
    treatment_workers: int,
    device_name: str,
    experiment_id: str,
    decision_rows_path: Path | None,
    game_results_path: Path | None = None,
    source_revision: str | None = None,
    scoring_cards: str = "aaaaa",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_ruleset_id = expected_ruleset_id_for(scoring_cards)
    if baseline_workers <= 0:
        raise ValueError("--baseline-workers must be positive")
    if treatment_workers <= 0:
        raise ValueError("--treatment-workers must be positive")
    checkpoint = load_cascadiaformer_manifest(manifest, device_name=device_name)
    model_lock = threading.Lock()
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

    model_results_by_head: dict[str, list[dict[str, Any]]] = {}
    def run_model(seed: int, selection_head: str) -> dict[str, Any]:
        return run_interactive_game(
            binary,
            seed=seed,
            strategy="cascadiaformer",
            checkpoint=checkpoint,
            selection_head=selection_head,
            max_actions=max_actions,
            model_lock=model_lock,
            scoring_cards=scoring_cards,
        )

    for selection_head in selection_heads:
        if treatment_workers == 1:
            model_results_by_head[selection_head] = [remember(run_model(seed, selection_head)) for seed in seeds]
        else:
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=treatment_workers) as executor:
                futures = {executor.submit(run_model, seed, selection_head): seed for seed in seeds}
                for future in as_completed(futures):
                    results.append(remember(future.result()))
            results.sort(key=lambda result: int(result["seed"]))
            model_results_by_head[selection_head] = results

    def run_greedy(seed: int) -> dict[str, Any]:
        return run_interactive_game(
            binary,
            seed=seed,
            strategy="greedy",
            checkpoint=None,
            selection_head="greedy",
            max_actions=max_actions,
            scoring_cards=scoring_cards,
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

    strategies: dict[str, Any] = {"greedy": summarize_game_results(greedy_results)}
    market_decisions: dict[str, Any] = {"greedy": summarize_market_decisions(greedy_results)}
    paired_by_head: dict[str, list[dict[str, Any]]] = {}
    paired_stats_by_head: dict[str, dict[str, Any]] = {}
    mean_delta_by_head: dict[str, float | None] = {}
    all_model_results: list[dict[str, Any]] = []
    for selection_head, model_results in model_results_by_head.items():
        key = f"cascadiaformer-{selection_head}"
        all_model_results.extend(model_results)
        strategies[key] = summarize_game_results(model_results)
        market_decisions[key] = summarize_market_decisions(model_results)
        paired = paired_score_deltas(model_results, greedy_results)
        paired_by_head[selection_head] = paired
        paired_stats_by_head[selection_head] = paired_delta_stats(
            [row["delta_cascadiaformer_minus_greedy"] for row in paired]
        )
        mean_delta_by_head[selection_head] = (
            mean(row["delta_cascadiaformer_minus_greedy"] for row in paired)
            if paired
            else None
        )

    all_results = all_model_results + greedy_results
    ruleset_ids = sorted({str(result["done"].get("ruleset_id")) for result in all_results})
    if ruleset_ids != [expected_ruleset_id]:
        raise RuntimeError(
            f"game output ruleset mismatch: expected {expected_ruleset_id}, got {ruleset_ids}"
        )

    return {
        "status": "pass",
        "ruleset_id": expected_ruleset_id,
        "source_revision": source_revision,
        "scientific_eligibility": "cascadiaformer_no_search_complete_game_benchmark",
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
        "selection_heads": selection_heads,
        "max_actions": max_actions,
        "baseline_workers": baseline_workers,
        "treatment_workers": treatment_workers,
        "runtime": {
            "device": str(checkpoint["device"]),
            "device_name": checkpoint["device_name"],
            "torch_version": checkpoint["torch_version"],
            "torch_cuda": checkpoint["torch_cuda"],
            "cuda_available": checkpoint["cuda_available"],
        },
        "strategies": strategies,
        "market_decisions": market_decisions,
        "paired_score_deltas": paired_by_head,
        "paired_delta_stats": paired_stats_by_head,
        "mean_paired_delta_cascadiaformer_minus_greedy": mean_delta_by_head,
        "games": [
            completed_game_result_row(result)
            for result in all_results
        ],
    }, all_results


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    greedy = report["strategies"]["greedy"]
    lines = [
        "# CascadiaFormer Complete-Game Benchmark",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Manifest: `{report['manifest']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Max actions/root: `{report['max_actions']}`",
        f"Device: `{report['runtime']['device_name']}`",
        f"Market decisions: `{json.dumps(report['market_decisions'], sort_keys=True)}`",
        "",
        "## Greedy Baseline",
        "",
        f"- Mean seat score: `{greedy['mean_seat_score']:.4f}`",
        f"- P90 seat score: `{greedy['p90_seat_score']:.4f}`",
    ]
    for selection_head in report["selection_heads"]:
        key = f"cascadiaformer-{selection_head}"
        row = report["strategies"][key]
        lines.extend(
            [
                "",
                f"## CascadiaFormer {selection_head}",
                "",
                f"- Mean seat score: `{row['mean_seat_score']:.4f}`",
                f"- P90 seat score: `{row['p90_seat_score']:.4f}`",
                f"- Mean paired delta vs greedy: `{report['mean_paired_delta_cascadiaformer_minus_greedy'][selection_head]}`",
                f"- Paired delta statistics: `{json.dumps(report['paired_delta_stats'][selection_head], sort_keys=True)}`",
                f"- Action match rate vs greedy top: `{row['action_match_rate_vs_greedy_top']}`",
                f"- Mean greedy rank in model: `{row['mean_greedy_rank_in_model']}`",
                f"- Mean model score seconds/root: `{row['mean_model_score_seconds']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "This is a no-search first-merit benchmark. It tests whether the trained checkpoint can choose useful complete-game actions directly from public roots, not whether it is superhuman with search.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-heads", default="policy,q")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--first-seed", type=int, default=2026990000)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument(
        "--scoring-cards",
        choices=sorted(RULESET_IDS_BY_SCORING_CARDS),
        default="aaaaa",
        help="Scoring-card selection passed to the exporter; also resolves "
        "the ruleset id expected in every emitted record",
    )
    parser.add_argument("--baseline-workers", type=int, default=8)
    parser.add_argument("--treatment-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-id", default="cascadiaformer-complete-game-benchmark-v1")
    parser.add_argument(
        "--source-revision",
        default="",
        help="Exact deployed Git revision used to build the Rust exporter",
    )
    parser.add_argument("--out", default="cascadiav3/reports/cascadiaformer_game_benchmark.json")
    parser.add_argument("--decisions-out", default="cascadiav3/reports/cascadiaformer_game_benchmark_decisions.jsonl")
    parser.add_argument("--game-results-out", default="cascadiav3/reports/cascadiaformer_game_benchmark_games.jsonl")
    parser.add_argument("--summary-out", default="cascadiav3/reports/cascadiaformer_game_benchmark_summary.md")
    args = parser.parse_args()

    selection_heads = [head.strip() for head in args.selection_heads.split(",") if head.strip()]
    unknown = sorted(set(selection_heads) - {"policy", "q"})
    if unknown:
        raise ValueError(f"unknown selection head(s): {unknown}")
    seeds = parse_seeds(seeds=args.seeds, first_seed=args.first_seed, games=args.games)
    report, _raw_results = run_benchmark(
        binary=Path(args.binary),
        manifest=Path(args.manifest),
        seeds=seeds,
        selection_heads=selection_heads,
        max_actions=args.max_actions,
        baseline_workers=args.baseline_workers,
        treatment_workers=args.treatment_workers,
        device_name=args.device,
        experiment_id=args.experiment_id,
        decision_rows_path=Path(args.decisions_out),
        game_results_path=Path(args.game_results_out),
        source_revision=args.source_revision or None,
        scoring_cards=args.scoring_cards,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
