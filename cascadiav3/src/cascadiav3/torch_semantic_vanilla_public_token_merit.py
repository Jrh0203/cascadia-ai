"""First-class semantic vanilla public-token CRT merit pilot.

The p80x2 residual-attention hardening run produced an unexpected positive
signal: the same-run vanilla public-token Transformer member held the strict
K=16 prefilter gate better than the residual or MLP members. This module makes
that recipe a primary checkpoint family instead of a side member, while keeping
the same semantic 61-feature action tensor and top-k retention objective.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .torch_action_query_merit import parameter_count
from .torch_public_token_merit import (
    PUBLIC_TOKEN_FEATURE_DIM,
    _baseline_metrics_from_batches,
    build_public_token_mlp,
    build_public_token_transformer,
)
from .torch_relation_bias_merit import _public_scores
from .torch_semantic_cross_attention_merit import (
    _model_metrics,
    _public_loss_with_mode,
    _train_semantic_model,
)
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    semantic_all_batches,
    semantic_dataset_summary,
)


@dataclass(frozen=True)
class SemanticVanillaPublicTokenConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0
    model_name: str = "CRT-semantic-vanilla-public-token-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_improvement(numerator: float, denominator: float) -> float:
    return numerator / max(abs(denominator), 1.0e-9)


def _decision_for_vanilla(
    vanilla_metrics: dict[str, Any],
    mlp_metrics: dict[str, Any],
    immediate_metrics: dict[str, Any],
) -> dict[str, Any]:
    vanilla_regret = float(vanilla_metrics["mean_regret"])
    mlp_regret = float(mlp_metrics["mean_regret"])
    immediate_regret = float(immediate_metrics["mean_regret"])
    top1_gain_vs_immediate = float(vanilla_metrics["top1_agreement"]) - float(immediate_metrics["top1_agreement"])
    regret_improvement_vs_immediate = _safe_improvement(immediate_regret - vanilla_regret, immediate_regret)
    regret_improvement_vs_mlp = _safe_improvement(mlp_regret - vanilla_regret, mlp_regret)
    nonregresses_mlp = vanilla_regret <= mlp_regret + 1.0e-9
    beats_immediate = regret_improvement_vs_immediate >= 0.10 or top1_gain_vs_immediate >= 0.05
    return {
        "criteria": (
            "first-class vanilla-public-token merit requires >=10% lower regret or >=5pp top-1 gain "
            "versus immediate score, plus nonregression versus the semantic token-pooled MLP"
        ),
        "has_merit": bool(beats_immediate and nonregresses_mlp),
        "beats_immediate": bool(beats_immediate),
        "nonregresses_mlp": bool(nonregresses_mlp),
        "regret_improvement_vs_immediate": regret_improvement_vs_immediate,
        "regret_improvement_vs_mlp": regret_improvement_vs_mlp,
        "top1_gain_vs_immediate": top1_gain_vs_immediate,
        "note": "serving prefilter promotion is decided by torch_prefilter_eval, not this offline ranking gate",
    }


def run_semantic_vanilla_public_token_pilot(
    train_path: Path,
    val_path: Path,
    *,
    steps: int = 7600,
    batch_size: int = 12,
    lr: float = 3.2e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260630,
    device_name: str = "cuda",
    hidden_dim: int = 256,
    layers: int = 4,
    heads: int = 8,
    mlp_dim: int = 512,
    grad_clip: float = 1.0,
    loss_mode: str = "topk-retention",
    q_loss_weight: float = 0.15,
    policy_loss_weight: float = 0.25,
    best_margin_loss_weight: float = 1.0,
    retention_loss_weight: float = 1.5,
    retention_k: int = 16,
    pairwise_margin: float = 0.15,
    policy_temperature: float = 0.75,
    train_mlp_baseline: bool = True,
    experiment_id: str = "crt-semantic-vanilla-public-token-query-merit-v1",
) -> dict[str, Any]:
    import torch

    args = argparse.Namespace(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        loss_mode=loss_mode,
        q_loss_weight=q_loss_weight,
        policy_loss_weight=policy_loss_weight,
        best_margin_loss_weight=best_margin_loss_weight,
        retention_loss_weight=retention_loss_weight,
        retention_k=retention_k,
        pairwise_margin=pairwise_margin,
        policy_temperature=policy_temperature,
    )
    config = SemanticVanillaPublicTokenConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
    if config.hidden_dim % config.heads != 0:
        raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by heads {config.heads}")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_summary = semantic_dataset_summary(train_path)
    val_summary = semantic_dataset_summary(val_path)
    val_batches = semantic_all_batches(val_path, batch_size=batch_size)

    vanilla_model = build_public_token_transformer(config)
    vanilla_model, vanilla_optimizer, vanilla_losses = _train_semantic_model(
        vanilla_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
    )

    mlp_model = None
    mlp_optimizer = None
    mlp_losses: list[float] = []
    if train_mlp_baseline:
        mlp_model = build_public_token_mlp(config)
        mlp_model, mlp_optimizer, mlp_losses = _train_semantic_model(
            mlp_model,
            train_path,
            args=args,
            device=device,
            loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    vanilla_metrics = _model_metrics(vanilla_model, val_batches, device, _public_scores)
    immediate_metrics = _baseline_metrics_from_batches(val_batches, "immediate")
    models: dict[str, Any] = {
        "vanilla_public_token_transformer": {
            "parameter_count": parameter_count(vanilla_model),
            "loss_head": vanilla_losses[:5],
            "loss_tail": vanilla_losses[-5:],
            "metrics": vanilla_metrics,
        }
    }
    if mlp_model is not None:
        mlp_metrics = _model_metrics(mlp_model, val_batches, device, _public_scores)
        models["token_pooled_mlp"] = {
            "parameter_count": parameter_count(mlp_model),
            "loss_head": mlp_losses[:5],
            "loss_tail": mlp_losses[-5:],
            "metrics": mlp_metrics,
        }
    else:
        mlp_metrics = {
            "mean_regret": float("inf"),
            "top1_agreement": 0.0,
            "top4_recall": 0.0,
        }
    decision = _decision_for_vanilla(vanilla_metrics, mlp_metrics, immediate_metrics)

    result = {
        "status": "pass",
        "scientific_eligibility": "dry_run",
        "experiment_id": experiment_id,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss": {
            "mode": loss_mode,
            "q_loss_weight": q_loss_weight,
            "policy_loss_weight": policy_loss_weight,
            "best_margin_loss_weight": best_margin_loss_weight,
            "retention_loss_weight": retention_loss_weight,
            "retention_k": retention_k,
            "pairwise_margin": pairwise_margin,
            "policy_temperature": policy_temperature,
            "label_reliability": "sqrt(count/max_count) * inverse root-normalized variance, clamped to [0.05, 1.0]",
        },
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "train_dataset": train_summary,
        "val_dataset": val_summary,
        "models": models,
        "baselines": {
            "immediate_base": immediate_metrics,
        },
        "decision": decision,
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_max_memory_allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "vanilla_model": vanilla_model,
        "vanilla_optimizer": vanilla_optimizer,
    }
    if mlp_model is not None and mlp_optimizer is not None:
        result["mlp_model"] = mlp_model
        result["mlp_optimizer"] = mlp_optimizer
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
    parser.add_argument("--steps", type=int, default=7600)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3.2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=["standard", "top16-prefilter", "topk-retention"], default="topk-retention")
    parser.add_argument("--q-loss-weight", type=float, default=0.15)
    parser.add_argument("--policy-loss-weight", type=float, default=0.25)
    parser.add_argument("--best-margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-loss-weight", type=float, default=1.5)
    parser.add_argument("--retention-k", type=int, default=16)
    parser.add_argument("--pairwise-margin", type=float, default=0.15)
    parser.add_argument("--policy-temperature", type=float, default=0.75)
    parser.add_argument("--skip-mlp-baseline", action="store_true")
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-vanilla-public-token-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_vanilla_public_token_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_vanilla_public_token_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_semantic_vanilla_public_token_pilot(
        Path(args.train),
        Path(args.val),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        grad_clip=args.grad_clip,
        loss_mode=args.loss_mode,
        q_loss_weight=args.q_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        best_margin_loss_weight=args.best_margin_loss_weight,
        retention_loss_weight=args.retention_loss_weight,
        retention_k=args.retention_k,
        pairwise_margin=args.pairwise_margin,
        policy_temperature=args.policy_temperature,
        train_mlp_baseline=not args.skip_mlp_baseline,
        experiment_id=args.experiment_id,
    )
    vanilla_model = result.pop("vanilla_model")
    vanilla_optimizer = result.pop("vanilla_optimizer")
    mlp_model = result.pop("mlp_model", None)
    mlp_optimizer = result.pop("mlp_optimizer", None)

    checkpoint: dict[str, Any] = {
        "vanilla_state_dict": vanilla_model.state_dict(),
        "vanilla_optimizer_state_dict": vanilla_optimizer.state_dict(),
        "report": result,
    }
    if mlp_model is not None and mlp_optimizer is not None:
        checkpoint["mlp_state_dict"] = mlp_model.state_dict()
        checkpoint["mlp_optimizer_state_dict"] = mlp_optimizer.state_dict()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded["report"]["decision"] != result["decision"]:
        raise RuntimeError("checkpoint round-trip decision mismatch")
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_roundtrip"] = "pass"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
