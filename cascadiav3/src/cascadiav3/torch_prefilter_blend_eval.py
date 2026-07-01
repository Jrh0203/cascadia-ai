"""Train-selected score blending for CRT prefilter serving checks.

This is a calibration/evaluation tool, not a training loop. It loads a
relation-bias checkpoint containing relation, vanilla, and MLP models, computes
per-root source scores once, selects a normalized linear blend on a training
shard, and evaluates the selected blend on a held-out validation shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_prefilter_eval import _config_from_report, _serving_decision
from .torch_public_token_merit import build_public_token_mlp, build_public_token_transformer
from .torch_relation_bias_merit import _all_batches, _public_scores, _relation_dataset_summary, _relation_scores, _to_device, build_relation_bias_transformer

RELATION_SOURCE_NAMES = ("relation", "vanilla", "mlp", "immediate")
CROSS_SOURCE_NAMES = ("cross", "vanilla", "mlp", "immediate")
RESIDUAL_SOURCE_NAMES = ("residual", "vanilla", "mlp", "immediate")
ACTION_SET_SOURCE_NAMES = ("action_set", "vanilla", "mlp", "immediate")


def simplex_weight_grid(source_count: int, step: float) -> list[tuple[float, ...]]:
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1.0e-9:
        raise ValueError("step must evenly divide 1.0")

    rows: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining_sources: int, remaining_units: int) -> None:
        if remaining_sources == 1:
            rows.append(tuple((prefix + [remaining_units])[i] * step for i in range(source_count)))
            return
        for value in range(remaining_units + 1):
            rec(prefix + [value], remaining_sources - 1, remaining_units - value)

    rec([], source_count, units)
    return rows


def _normalize_scores(values):  # type: ignore[no-untyped-def]
    import torch

    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1.0e-6)
    return (values - mean) / std


def _collect_rows(path: Path, checkpoint: dict[str, Any], *, batch_size: int, device):  # type: ignore[no-untyped-def]
    import torch

    report = checkpoint["report"]
    config = _config_from_report(report)
    model_name = str(config.model_name)
    if model_name.startswith("CRT-semantic-action-set"):
        from .torch_semantic_action_set_merit import (
            _action_set_scores,
            build_semantic_action_set_transformer,
        )
        from .torch_semantic_relation_bias_merit import semantic_all_batches

        primary_name = "action_set"
        source_names = ACTION_SET_SOURCE_NAMES
        batches = semantic_all_batches(path, batch_size=batch_size)
        primary_model = build_semantic_action_set_transformer(config)
        primary_model.load_state_dict(checkpoint["action_set_state_dict"])
        primary_model = primary_model.to(device)
        primary_model.eval()
        primary_score_fn = _action_set_scores
    elif model_name.startswith("CRT-semantic-residual-attention"):
        from .torch_semantic_relation_bias_merit import semantic_all_batches
        from .torch_semantic_residual_attention_merit import (
            _residual_attention_scores,
            build_semantic_residual_attention_transformer,
        )

        primary_name = "residual"
        source_names = RESIDUAL_SOURCE_NAMES
        batches = semantic_all_batches(path, batch_size=batch_size)
        primary_model = build_semantic_residual_attention_transformer(config)
        primary_model.load_state_dict(checkpoint["residual_attention_state_dict"])
        primary_model = primary_model.to(device)
        primary_model.eval()
        primary_score_fn = _residual_attention_scores
    elif model_name.startswith("CRT-semantic-cross-attention"):
        from .torch_semantic_cross_attention_merit import (
            _cross_attention_scores,
            build_semantic_cross_attention_transformer,
        )
        from .torch_semantic_relation_bias_merit import semantic_all_batches

        primary_name = "cross"
        source_names = CROSS_SOURCE_NAMES
        batches = semantic_all_batches(path, batch_size=batch_size)
        primary_model = build_semantic_cross_attention_transformer(config)
        primary_model.load_state_dict(checkpoint["cross_attention_state_dict"])
        primary_model = primary_model.to(device)
        primary_model.eval()
        primary_score_fn = _cross_attention_scores
    else:
        primary_name = "relation"
        source_names = RELATION_SOURCE_NAMES
        batches = _all_batches(path, batch_size=batch_size)
        primary_model = build_relation_bias_transformer(config)
        primary_model.load_state_dict(checkpoint["relation_bias_state_dict"])
        primary_model = primary_model.to(device)
        primary_model.eval()
        primary_score_fn = _relation_scores

    vanilla_model = build_public_token_transformer(config)
    vanilla_model.load_state_dict(checkpoint["vanilla_state_dict"])
    vanilla_model = vanilla_model.to(device)
    vanilla_model.eval()

    mlp_model = build_public_token_mlp(config)
    mlp_model.load_state_dict(checkpoint["mlp_state_dict"])
    mlp_model = mlp_model.to(device)
    mlp_model.eval()

    rows = []
    for batch in batches:
        eval_batch = _to_device(batch, device)
        with torch.no_grad():
            source_scores = {
                primary_name: primary_score_fn(primary_model, eval_batch).detach().cpu(),
                "vanilla": _public_scores(vanilla_model, eval_batch).detach().cpu(),
                "mlp": _public_scores(mlp_model, eval_batch).detach().cpu(),
                "immediate": batch["immediate"].detach().cpu(),
            }
        target_q = batch["target_q"]
        mask = batch["action_mask"]
        for row_index in range(mask.shape[0]):
            valid_count = int(mask[row_index].sum().item())
            rows.append(
                {
                    "state_hash": batch["state_hashes"][row_index],
                    "action_count": valid_count,
                    "target_q": target_q[row_index, :valid_count].clone(),
                    "sources": {
                        name: source_scores[name][row_index, :valid_count].clone()
                        for name in source_names
                    },
                    "source_names": source_names,
                }
            )
    return rows


def _blend_scores(row: dict[str, Any], weights: tuple[float, ...]):  # type: ignore[no-untyped-def]
    score = None
    for source_name, weight in zip(row["source_names"], weights, strict=True):
        source = _normalize_scores(row["sources"][source_name])
        weighted = source * weight
        score = weighted if score is None else score + weighted
    return score


def _evaluate_rows(rows: list[dict[str, Any]], weights: tuple[float, ...], *, k_values: list[int]) -> dict[str, Any]:
    import torch

    total_roots = 0
    top1 = 0
    top4 = 0
    regret_sum = 0.0
    best_q_sum = 0.0
    selected_q_sum = 0.0
    prefilter_hits = {k: 0 for k in k_values}
    prefilter_regret_sum = {k: 0.0 for k in k_values}
    prefilter_q_sum = {k: 0.0 for k in k_values}

    for row in rows:
        q = row["target_q"]
        scores = _blend_scores(row, weights)
        valid_count = int(row["action_count"])
        teacher_best = int(torch.argmax(q).item())
        selected = int(torch.argmax(scores).item())
        ranked = torch.argsort(scores, descending=True)
        best_q = float(q[teacher_best].item())
        selected_q = float(q[selected].item())
        total_roots += 1
        top1 += int(selected == teacher_best)
        top4 += int(teacher_best in ranked[: min(4, valid_count)].tolist())
        regret_sum += best_q - selected_q
        best_q_sum += best_q
        selected_q_sum += selected_q
        for k in k_values:
            retained = ranked[: min(k, valid_count)]
            retained_indices = retained.tolist()
            retained_best_q = float(q[retained].max().item())
            prefilter_hits[k] += int(teacher_best in retained_indices)
            prefilter_regret_sum[k] += best_q - retained_best_q
            prefilter_q_sum[k] += retained_best_q

    return {
        "roots": total_roots,
        "top1_agreement": top1 / total_roots if total_roots else 0.0,
        "top4_recall": top4 / total_roots if total_roots else 0.0,
        "mean_regret": regret_sum / total_roots if total_roots else 0.0,
        "mean_best_q": best_q_sum / total_roots if total_roots else 0.0,
        "mean_selected_q": selected_q_sum / total_roots if total_roots else 0.0,
        "prefilter": {
            str(k): {
                "recall": prefilter_hits[k] / total_roots if total_roots else 0.0,
                "mean_oracle_regret": prefilter_regret_sum[k] / total_roots if total_roots else 0.0,
                "mean_oracle_q": prefilter_q_sum[k] / total_roots if total_roots else 0.0,
            }
            for k in k_values
        },
    }


def _selection_key(metrics: dict[str, Any], *, target_k: int, min_recall: float, max_oracle_regret: float):
    row = metrics["prefilter"][str(target_k)]
    recall = float(row["recall"])
    regret = float(row["mean_oracle_regret"])
    passes = recall >= min_recall and regret <= max_oracle_regret
    return (
        int(passes),
        recall,
        -regret,
        -float(metrics["mean_regret"]),
        float(metrics["top1_agreement"]),
    )


def run_blend_eval(
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device_name: str,
    grid_step: float,
    k_values: list[int],
    target_k: int,
    min_recall: float,
    max_oracle_regret: float,
    experiment_id: str,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    train_rows = _collect_rows(train_path, checkpoint, batch_size=batch_size, device=device)
    val_rows = _collect_rows(val_path, checkpoint, batch_size=batch_size, device=device)
    source_names = tuple(train_rows[0]["source_names"]) if train_rows else RELATION_SOURCE_NAMES
    candidates = []
    for weights in simplex_weight_grid(len(source_names), grid_step):
        train_metrics = _evaluate_rows(train_rows, weights, k_values=k_values)
        candidates.append(
            {
                "weights": dict(zip(source_names, weights, strict=True)),
                "train_metrics": train_metrics,
                "selection_key": _selection_key(
                    train_metrics,
                    target_k=target_k,
                    min_recall=min_recall,
                    max_oracle_regret=max_oracle_regret,
                ),
            }
        )
    candidates.sort(key=lambda row: row["selection_key"], reverse=True)
    selected = candidates[0]
    selected_weights = tuple(selected["weights"][name] for name in source_names)
    val_metrics = _evaluate_rows(val_rows, selected_weights, k_values=k_values)
    selected["val_metrics"] = val_metrics
    selected["serving_decision"] = _serving_decision(
        val_metrics,
        k_values=k_values,
        min_recall=min_recall,
        max_oracle_regret=max_oracle_regret,
    )
    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_blend_eval",
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_experiment_id": checkpoint["report"].get("experiment_id"),
        "train": str(train_path),
        "val": str(val_path),
        "train_dataset": _relation_dataset_summary(train_path),
        "val_dataset": _relation_dataset_summary(val_path),
        "batch_size": batch_size,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "sources": list(source_names),
        "grid_step": grid_step,
        "k_values": k_values,
        "target_k": target_k,
        "min_recall": min_recall,
        "max_oracle_regret": max_oracle_regret,
        "selected": selected,
        "top_train_candidates": [
            {
                "weights": row["weights"],
                "train_metrics": row["train_metrics"],
            }
            for row in candidates[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val.jsonl")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--k-values", default="4,8,16,24,32")
    parser.add_argument("--target-k", type=int, default=16)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-oracle-regret", type=float, default=0.25)
    parser.add_argument("--experiment-id", default="crt-wide32-r16-prefilter-blend-eval-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16_prefilter_blend_eval.json")
    args = parser.parse_args()

    k_values = sorted({int(part.strip()) for part in args.k_values.split(",") if part.strip()})
    result = run_blend_eval(
        Path(args.train),
        Path(args.val),
        Path(args.checkpoint),
        batch_size=args.batch_size,
        device_name=args.device,
        grid_step=args.grid_step,
        k_values=k_values,
        target_k=args.target_k,
        min_recall=args.min_recall,
        max_oracle_regret=args.max_oracle_regret,
        experiment_id=args.experiment_id,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
