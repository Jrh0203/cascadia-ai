"""Serving-shaped prefilter evaluation for trained CRT checkpoints.

This module does not train. It replays a validation JSONL through an existing
relation-bias checkpoint and asks whether the model can safely reduce a wider
candidate set before a downstream search/value stage sees it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .torch_public_token_merit import build_public_token_mlp, build_public_token_transformer
from .torch_relation_bias_merit import (
    RelationBiasConfig,
    _all_batches,
    _public_scores,
    _relation_dataset_summary,
    _relation_scores,
    _to_device,
    build_relation_bias_transformer,
)

DEFAULT_K_VALUES = (4, 8, 16, 24, 32)


def parse_k_values(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("at least one K value is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"K values must be positive: {values}")
    return values


def _config_from_report(report: dict[str, Any]) -> Any:
    config = dict(report.get("config") or {})
    model_name = str(config.get("model_name", ""))
    if model_name.startswith("CRT-semantic-vanilla-public-token"):
        from .torch_semantic_vanilla_public_token_merit import SemanticVanillaPublicTokenConfig

        allowed = SemanticVanillaPublicTokenConfig.__dataclass_fields__.keys()
        return SemanticVanillaPublicTokenConfig(**{key: config[key] for key in allowed if key in config})
    if model_name.startswith("CRT-semantic-action-set"):
        from .torch_semantic_action_set_merit import SemanticActionSetConfig

        allowed = SemanticActionSetConfig.__dataclass_fields__.keys()
        return SemanticActionSetConfig(**{key: config[key] for key in allowed if key in config})  # type: ignore[return-value]
    if model_name.startswith("CRT-semantic-residual-attention"):
        from .torch_semantic_residual_attention_merit import SemanticResidualAttentionConfig

        allowed = SemanticResidualAttentionConfig.__dataclass_fields__.keys()
        return SemanticResidualAttentionConfig(**{key: config[key] for key in allowed if key in config})  # type: ignore[return-value]
    if model_name.startswith("CRT-semantic-cross-attention"):
        from .torch_semantic_cross_attention_merit import SemanticCrossAttentionConfig

        allowed = SemanticCrossAttentionConfig.__dataclass_fields__.keys()
        return SemanticCrossAttentionConfig(**{key: config[key] for key in allowed if key in config})  # type: ignore[return-value]
    allowed = RelationBiasConfig.__dataclass_fields__.keys()
    return RelationBiasConfig(**{key: config[key] for key in allowed if key in config})


def _uses_semantic_action_features(config: Any) -> bool:
    return str(config.model_name).startswith("CRT-semantic-")


def _uses_cross_attention(config: Any) -> bool:
    return str(config.model_name).startswith("CRT-semantic-cross-attention")


def _uses_residual_attention(config: Any) -> bool:
    return str(config.model_name).startswith("CRT-semantic-residual-attention")


def _uses_action_set(config: Any) -> bool:
    return str(config.model_name).startswith("CRT-semantic-action-set")


def _uses_vanilla_public_token(config: Any) -> bool:
    return str(config.model_name).startswith("CRT-semantic-vanilla-public-token")


def _round(value: float) -> float:
    return round(float(value), 10)


def _serving_decision(
    metrics: dict[str, Any],
    *,
    k_values: list[int],
    min_recall: float,
    max_oracle_regret: float,
) -> dict[str, Any]:
    per_k = metrics["prefilter"]
    gates: dict[str, dict[str, Any]] = {}
    recommended: int | None = None
    for k in k_values:
        row = per_k[str(k)]
        passes_recall = float(row["recall"]) >= min_recall
        passes_regret = float(row["mean_oracle_regret"]) <= max_oracle_regret
        passes = passes_recall and passes_regret
        gates[str(k)] = {
            "passes": passes,
            "passes_recall": passes_recall,
            "passes_oracle_regret": passes_regret,
            "recall": row["recall"],
            "mean_oracle_regret": row["mean_oracle_regret"],
        }
        if passes and recommended is None:
            recommended = k
    best_available = min(
        k_values,
        key=lambda k: (
            float(per_k[str(k)]["mean_oracle_regret"]),
            -float(per_k[str(k)]["recall"]),
            k,
        ),
    )
    return {
        "criteria": (
            f"serving candidate requires teacher-best recall >= {min_recall:.3f} "
            f"and mean oracle regret <= {max_oracle_regret:.3f} sampled-teacher points"
        ),
        "passes": recommended is not None,
        "recommended_k": recommended,
        "best_available_k": best_available,
        "gates": gates,
    }


def _empty_prefilter_stats(k_values: list[int]) -> dict[str, Any]:
    return {
        "hits": {k: 0 for k in k_values},
        "regret_sum": {k: 0.0 for k in k_values},
        "oracle_q_sum": {k: 0.0 for k in k_values},
        "retained_count_sum": {k: 0 for k in k_values},
    }


def _evaluate_scores(
    batches: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], Any],
    *,
    k_values: list[int],
    device: Any,
    per_root_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    total_roots = 0
    top1 = 0
    top4 = 0
    regret_sum = 0.0
    best_q_sum = 0.0
    selected_q_sum = 0.0
    selected_score_sum = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    action_counts: list[int] = []
    prefilter = _empty_prefilter_stats(k_values)

    writer = per_root_path.open("w", encoding="utf-8") if per_root_path else None
    try:
        for batch in batches:
            eval_batch = _to_device(batch, device)
            with torch.no_grad():
                scores = score_fn(eval_batch).detach().cpu()
            target_q = batch["target_q"]
            mask = batch["action_mask"]
            for row in range(mask.shape[0]):
                valid_count = int(mask[row].sum().item())
                q = target_q[row, :valid_count]
                pred = scores[row, :valid_count]
                ranked = torch.argsort(pred, descending=True)
                teacher_best = int(torch.argmax(q).item())
                selected = int(torch.argmax(pred).item())
                top4 = top4 + int(teacher_best in ranked[: min(4, valid_count)].tolist())
                best_q = float(q[teacher_best].item())
                selected_q = float(q[selected].item())
                selected_score = float(pred[selected].item())

                total_roots += 1
                action_counts.append(valid_count)
                top1 += int(selected == teacher_best)
                regret_sum += best_q - selected_q
                best_q_sum += best_q
                selected_q_sum += selected_q
                selected_score_sum += selected_score

                per_root_prefilter: dict[str, Any] = {}
                for k in k_values:
                    retained = ranked[: min(k, valid_count)]
                    retained_indices = [int(index) for index in retained.tolist()]
                    retained_q = q[retained]
                    retained_best_offset = int(torch.argmax(retained_q).item())
                    retained_best = retained_indices[retained_best_offset]
                    retained_best_q = float(q[retained_best].item())
                    oracle_regret = best_q - retained_best_q
                    prefilter["hits"][k] += int(teacher_best in retained_indices)
                    prefilter["regret_sum"][k] += oracle_regret
                    prefilter["oracle_q_sum"][k] += retained_best_q
                    prefilter["retained_count_sum"][k] += len(retained_indices)
                    if writer is not None:
                        action_ids = batch["action_ids"][row]
                        per_root_prefilter[str(k)] = {
                            "retained_count": len(retained_indices),
                            "teacher_best_retained": teacher_best in retained_indices,
                            "oracle_best_index": retained_best,
                            "oracle_best_action_id": action_ids[retained_best],
                            "oracle_best_q": _round(retained_best_q),
                            "oracle_regret": _round(oracle_regret),
                            "retained_action_ids": [action_ids[index] for index in retained_indices],
                        }

                for left in range(valid_count):
                    for right in range(left + 1, valid_count):
                        q_diff = float(q[left] - q[right])
                        pred_diff = float(pred[left] - pred[right])
                        if abs(q_diff) < 1.0e-9 or abs(pred_diff) < 1.0e-9:
                            continue
                        pairwise_total += 1
                        pairwise_correct += int((q_diff > 0) == (pred_diff > 0))

                if writer is not None:
                    action_ids = batch["action_ids"][row]
                    ranked_indices = [int(index) for index in ranked.tolist()]
                    row_record = {
                        "state_hash": batch["state_hashes"][row],
                        "action_count": valid_count,
                        "teacher_best": {
                            "index": teacher_best,
                            "action_id": action_ids[teacher_best],
                            "q": _round(best_q),
                        },
                        "model_selected": {
                            "index": selected,
                            "action_id": action_ids[selected],
                            "predicted_q": _round(selected_score),
                            "teacher_q": _round(selected_q),
                            "regret": _round(best_q - selected_q),
                        },
                        "ranked_action_ids": [action_ids[index] for index in ranked_indices],
                        "ranked_predicted_q": [_round(float(pred[index].item())) for index in ranked_indices],
                        "ranked_teacher_q": [_round(float(q[index].item())) for index in ranked_indices],
                        "prefilter": per_root_prefilter,
                    }
                    writer.write(json.dumps(row_record, sort_keys=True) + "\n")
    finally:
        if writer is not None:
            writer.close()

    per_k = {
        str(k): {
            "recall": prefilter["hits"][k] / total_roots if total_roots else 0.0,
            "mean_oracle_regret": prefilter["regret_sum"][k] / total_roots if total_roots else 0.0,
            "mean_oracle_q": prefilter["oracle_q_sum"][k] / total_roots if total_roots else 0.0,
            "mean_retained_count": prefilter["retained_count_sum"][k] / total_roots if total_roots else 0.0,
        }
        for k in k_values
    }
    return {
        "roots": total_roots,
        "action_count_min": min(action_counts) if action_counts else 0,
        "action_count_max": max(action_counts) if action_counts else 0,
        "action_count_mean": sum(action_counts) / len(action_counts) if action_counts else 0.0,
        "top1_agreement": top1 / total_roots if total_roots else 0.0,
        "top4_recall": top4 / total_roots if total_roots else 0.0,
        "mean_regret": regret_sum / total_roots if total_roots else 0.0,
        "mean_best_q": best_q_sum / total_roots if total_roots else 0.0,
        "mean_selected_q": selected_q_sum / total_roots if total_roots else 0.0,
        "mean_selected_predicted_q": selected_score_sum / total_roots if total_roots else 0.0,
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else 0.0,
        "pairwise_total": pairwise_total,
        "prefilter": per_k,
    }


def run_prefilter_eval(
    val_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device_name: str,
    k_values: list[int],
    per_root_path: Path | None,
    min_recall: float,
    max_oracle_regret: float,
    include_baselines: bool,
    checkpoint_member: str,
    experiment_id: str,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_report = checkpoint["report"]
    config = _config_from_report(checkpoint_report)
    if _uses_semantic_action_features(config):
        from .torch_semantic_relation_bias_merit import semantic_all_batches, semantic_dataset_summary

        dataset_summary = semantic_dataset_summary(val_path)
        batches = semantic_all_batches(val_path, batch_size=batch_size)
    else:
        dataset_summary = _relation_dataset_summary(val_path)
        batches = _all_batches(val_path, batch_size=batch_size)
    max_actions = int(dataset_summary["action_counts"]["max"])
    if max(k_values) > max_actions:
        raise ValueError(f"requested K={max(k_values)} exceeds replay max action count {max_actions}")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    models: dict[str, Any] = {}
    per_root_written_by: list[str] = []

    def per_root_for(member: str) -> Path | None:
        if checkpoint_member == member:
            per_root_written_by.append(member)
            return per_root_path
        return None

    if _uses_vanilla_public_token(config):
        vanilla_model = build_public_token_transformer(config)
        vanilla_model.load_state_dict(checkpoint["vanilla_state_dict"])
        vanilla_model = vanilla_model.to(device)
        vanilla_model.eval()
        vanilla_metrics = _evaluate_scores(
            batches,
            lambda batch: _public_scores(vanilla_model, batch),
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("primary"),
        )
        models["vanilla_public_token_transformer"] = {
            "parameter_count": sum(parameter.numel() for parameter in vanilla_model.parameters()),
            "metrics": vanilla_metrics,
            "serving_decision": _serving_decision(
                vanilla_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
    elif _uses_action_set(config):
        from .torch_semantic_action_set_merit import _action_set_scores, build_semantic_action_set_transformer

        action_set_model = build_semantic_action_set_transformer(config)
        action_set_model.load_state_dict(checkpoint["action_set_state_dict"])
        action_set_model = action_set_model.to(device)
        action_set_model.eval()
        action_set_metrics = _evaluate_scores(
            batches,
            lambda batch: _action_set_scores(action_set_model, batch),
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("primary"),
        )
        models["action_set_transformer"] = {
            "parameter_count": sum(parameter.numel() for parameter in action_set_model.parameters()),
            "metrics": action_set_metrics,
            "serving_decision": _serving_decision(
                action_set_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
    elif _uses_residual_attention(config):
        from .torch_semantic_residual_attention_merit import (
            _residual_attention_scores,
            build_semantic_residual_attention_transformer,
        )

        residual_model = build_semantic_residual_attention_transformer(config)
        residual_model.load_state_dict(checkpoint["residual_attention_state_dict"])
        residual_model = residual_model.to(device)
        residual_model.eval()
        residual_metrics = _evaluate_scores(
            batches,
            lambda batch: _residual_attention_scores(residual_model, batch),
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("primary"),
        )
        models["residual_attention_transformer"] = {
            "parameter_count": sum(parameter.numel() for parameter in residual_model.parameters()),
            "metrics": residual_metrics,
            "serving_decision": _serving_decision(
                residual_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
    elif _uses_cross_attention(config):
        from .torch_semantic_cross_attention_merit import (
            _cross_attention_scores,
            build_semantic_cross_attention_transformer,
        )

        cross_model = build_semantic_cross_attention_transformer(config)
        cross_model.load_state_dict(checkpoint["cross_attention_state_dict"])
        cross_model = cross_model.to(device)
        cross_model.eval()
        cross_metrics = _evaluate_scores(
            batches,
            lambda batch: _cross_attention_scores(cross_model, batch),
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("primary"),
        )
        models["cross_attention_transformer"] = {
            "parameter_count": sum(parameter.numel() for parameter in cross_model.parameters()),
            "metrics": cross_metrics,
            "serving_decision": _serving_decision(
                cross_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
    else:
        relation_model = build_relation_bias_transformer(config)
        relation_model.load_state_dict(checkpoint["relation_bias_state_dict"])
        relation_model = relation_model.to(device)
        relation_model.eval()
        relation_metrics = _evaluate_scores(
            batches,
            lambda batch: _relation_scores(relation_model, batch),
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("primary"),
        )

        models["relation_bias_transformer"] = {
            "parameter_count": sum(parameter.numel() for parameter in relation_model.parameters()),
            "metrics": relation_metrics,
            "serving_decision": _serving_decision(
                relation_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
    if include_baselines or checkpoint_member in {"immediate", "vanilla", "mlp"}:
        immediate_metrics = _evaluate_scores(
            batches,
            lambda batch: batch["immediate"],
            k_values=k_values,
            device=device,
            per_root_path=per_root_for("immediate"),
        )
        models["immediate_score"] = {
            "metrics": immediate_metrics,
            "serving_decision": _serving_decision(
                immediate_metrics,
                k_values=k_values,
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        }
        if "vanilla_state_dict" in checkpoint:
            vanilla_model = build_public_token_transformer(config)
            vanilla_model.load_state_dict(checkpoint["vanilla_state_dict"])
            vanilla_model = vanilla_model.to(device)
            vanilla_model.eval()
            vanilla_metrics = _evaluate_scores(
                batches,
                lambda batch: _public_scores(vanilla_model, batch),
                k_values=k_values,
                device=device,
                per_root_path=per_root_for("vanilla"),
            )
            models["vanilla_public_token_transformer"] = {
                "parameter_count": sum(parameter.numel() for parameter in vanilla_model.parameters()),
                "metrics": vanilla_metrics,
                "serving_decision": _serving_decision(
                    vanilla_metrics,
                    k_values=k_values,
                    min_recall=min_recall,
                    max_oracle_regret=max_oracle_regret,
                ),
            }
        if "mlp_state_dict" in checkpoint:
            mlp_model = build_public_token_mlp(config)
            mlp_model.load_state_dict(checkpoint["mlp_state_dict"])
            mlp_model = mlp_model.to(device)
            mlp_model.eval()
            mlp_metrics = _evaluate_scores(
                batches,
                lambda batch: _public_scores(mlp_model, batch),
                k_values=k_values,
                device=device,
                per_root_path=per_root_for("mlp"),
            )
            models["token_pooled_mlp"] = {
                "parameter_count": sum(parameter.numel() for parameter in mlp_model.parameters()),
                "metrics": mlp_metrics,
                "serving_decision": _serving_decision(
                    mlp_metrics,
                    k_values=k_values,
                    min_recall=min_recall,
                    max_oracle_regret=max_oracle_regret,
                ),
            }

    if checkpoint_member != "primary":
        expected_model = {
            "immediate": "immediate_score",
            "vanilla": "vanilla_public_token_transformer",
            "mlp": "token_pooled_mlp",
        }[checkpoint_member]
        if expected_model not in models:
            raise ValueError(f"checkpoint member {checkpoint_member!r} is not available in {checkpoint_path}")
    if per_root_path is not None and checkpoint_member not in per_root_written_by:
        raise RuntimeError(f"per-root output was not written for checkpoint member {checkpoint_member!r}")

    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_eval",
        "experiment_id": experiment_id,
        "checkpoint_member": checkpoint_member,
        "checkpoint": str(checkpoint_path),
        "checkpoint_experiment_id": checkpoint_report.get("experiment_id"),
        "checkpoint_decision": checkpoint_report.get("decision"),
        "val": str(val_path),
        "val_dataset": dataset_summary,
        "batch_size": batch_size,
        "k_values": k_values,
        "min_recall": min_recall,
        "max_oracle_regret": max_oracle_regret,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "models": models,
        "per_root_out": str(per_root_path) if per_root_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_sampled_teacher_val.jsonl")
    parser.add_argument(
        "--checkpoint",
        default="cascadiav3/checkpoints/crt_wide32_sampled_teacher_relation_bias_pilot.pt",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-oracle-regret", type=float, default=0.25)
    parser.add_argument("--experiment-id", default="crt-wide32-relation-bias-prefilter-eval-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_prefilter_eval.json")
    parser.add_argument("--per-root-out", default="cascadiav3/reports/crt_wide32_prefilter_eval_roots.jsonl")
    parser.add_argument(
        "--checkpoint-member",
        choices=["primary", "mlp", "vanilla", "immediate"],
        default="primary",
        help="Checkpoint member to export in per-root rankings; default is the trained primary model.",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_root_path = Path(args.per_root_out) if args.per_root_out else None
    if per_root_path is not None:
        per_root_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_prefilter_eval(
        Path(args.val),
        Path(args.checkpoint),
        batch_size=args.batch_size,
        device_name=args.device,
        k_values=parse_k_values(args.k_values),
        per_root_path=per_root_path,
        min_recall=args.min_recall,
        max_oracle_regret=args.max_oracle_regret,
        include_baselines=not args.skip_baselines,
        checkpoint_member=args.checkpoint_member,
        experiment_id=args.experiment_id,
    )
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
