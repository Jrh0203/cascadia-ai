"""Semantic action-set CRT merit pilot.

The phase-diverse shard says semantic action features are the strongest current
signal, while full public-token attention variants tend to damage that signal.
This model keeps public tokens only as a pooled root context, then applies a
Set-Transformer-style encoder over the legal action set. The hypothesis is that
action-action comparison can improve top-K retention without letting noisy
state-token attention override the MLP-like action semantics.
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
from .torch_relation_bias_merit import (
    _decision_with_vanilla,
    _loss_with_mode,
    _public_scores,
)
from .torch_semantic_cross_attention_merit import _model_metrics, _public_loss_with_mode, _train_semantic_model
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    SemanticRelationBiasConfig,
    semantic_all_batches,
    semantic_dataset_summary,
)


@dataclass(frozen=True)
class SemanticActionSetConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0
    model_name: str = "CRT-semantic-action-set-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_semantic_action_set_transformer(config: Any):
    import torch
    from torch import nn

    class SemanticActionSetTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_proj = nn.Linear(config.token_feature_dim, config.hidden_dim)
            self.action_proj = nn.Linear(config.action_feature_dim, config.hidden_dim)
            self.context_proj = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.type_embedding = nn.Embedding(2, config.hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.mlp_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.q_head = nn.Linear(config.hidden_dim, 1)
            self.policy_head = nn.Linear(config.hidden_dim, 1)

        def _state_context(self, tokens, token_mask):  # type: ignore[no-untyped-def]
            token_hidden = self.token_proj(tokens)
            token_mask_f = token_mask.unsqueeze(-1).to(token_hidden.dtype)
            token_sum = (token_hidden * token_mask_f).sum(dim=1)
            token_count = token_mask_f.sum(dim=1).clamp_min(1.0)
            token_mean = token_sum / token_count
            token_min = torch.finfo(token_hidden.dtype).min
            token_max = token_hidden.masked_fill(~token_mask.unsqueeze(-1), token_min).amax(dim=1)
            token_max = torch.where(torch.isfinite(token_max), token_max, torch.zeros_like(token_max))
            return self.context_proj(torch.cat([token_mean, token_max], dim=-1))

        def forward(self, tokens, token_mask, actions, action_mask):  # type: ignore[no-untyped-def]
            batch_size = actions.shape[0]
            context = self._state_context(tokens, token_mask)
            action_hidden = self.action_proj(actions) + context[:, None, :]
            root_type = self.type_embedding.weight[0].view(1, 1, -1)
            action_type = self.type_embedding.weight[1].view(1, 1, -1)
            root = context[:, None, :] + root_type
            action_hidden = action_hidden + action_type
            hidden = torch.cat([root, action_hidden], dim=1)
            root_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=action_mask.device)
            padding_mask = torch.cat([root_padding, ~action_mask], dim=1)
            encoded = self.norm(self.encoder(hidden, src_key_padding_mask=padding_mask))
            action_encoded = encoded[:, 1:]
            q = self.q_head(action_encoded).squeeze(-1).masked_fill(~action_mask, 0.0)
            logits = self.policy_head(action_encoded).squeeze(-1).masked_fill(~action_mask, 0.0)
            return {"q": q, "logits": logits}

    return SemanticActionSetTransformer()


def _action_set_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])["q"]


def _action_set_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
        args,
    )


def run_semantic_action_set_pilot(
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
    experiment_id: str = "crt-semantic-action-set-query-merit-v1",
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
    config = SemanticActionSetConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
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

    action_set_model = build_semantic_action_set_transformer(config)
    action_set_model, action_set_optimizer, action_set_losses = _train_semantic_model(
        action_set_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _action_set_loss_with_mode(model, batch, args),
    )
    baseline_config = SemanticRelationBiasConfig(
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        mlp_dim=mlp_dim,
        model_name=config.model_name,
    )
    vanilla_model = build_public_token_transformer(baseline_config)
    vanilla_model, vanilla_optimizer, vanilla_losses = _train_semantic_model(
        vanilla_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
    )
    mlp_model = build_public_token_mlp(baseline_config)
    mlp_model, mlp_optimizer, mlp_losses = _train_semantic_model(
        mlp_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    action_set_metrics = _model_metrics(action_set_model, val_batches, device, _action_set_scores)
    vanilla_metrics = _model_metrics(vanilla_model, val_batches, device, _public_scores)
    mlp_metrics = _model_metrics(mlp_model, val_batches, device, _public_scores)
    immediate_metrics = _baseline_metrics_from_batches(val_batches, "immediate")
    decision = _decision_with_vanilla(action_set_metrics, vanilla_metrics, mlp_metrics, immediate_metrics)
    decision["criteria"] = decision["criteria"].replace("relation-bias", "action-set")

    return {
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
        "models": {
            "action_set_transformer": {
                "parameter_count": parameter_count(action_set_model),
                "loss_head": action_set_losses[:5],
                "loss_tail": action_set_losses[-5:],
                "metrics": action_set_metrics,
            },
            "vanilla_public_token_transformer": {
                "parameter_count": parameter_count(vanilla_model),
                "loss_head": vanilla_losses[:5],
                "loss_tail": vanilla_losses[-5:],
                "metrics": vanilla_metrics,
            },
            "token_pooled_mlp": {
                "parameter_count": parameter_count(mlp_model),
                "loss_head": mlp_losses[:5],
                "loss_tail": mlp_losses[-5:],
                "metrics": mlp_metrics,
            },
        },
        "baselines": {
            "immediate_base": immediate_metrics,
        },
        "decision": decision,
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_max_memory_allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "action_set_model": action_set_model,
        "action_set_optimizer": action_set_optimizer,
        "vanilla_model": vanilla_model,
        "vanilla_optimizer": vanilla_optimizer,
        "mlp_model": mlp_model,
        "mlp_optimizer": mlp_optimizer,
    }


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
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-action-set-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_action_set_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_semantic_action_set_pilot(
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
        experiment_id=args.experiment_id,
    )
    action_set_model = result.pop("action_set_model")
    action_set_optimizer = result.pop("action_set_optimizer")
    vanilla_model = result.pop("vanilla_model")
    vanilla_optimizer = result.pop("vanilla_optimizer")
    mlp_model = result.pop("mlp_model")
    mlp_optimizer = result.pop("mlp_optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_set_state_dict": action_set_model.state_dict(),
            "action_set_optimizer_state_dict": action_set_optimizer.state_dict(),
            "vanilla_state_dict": vanilla_model.state_dict(),
            "vanilla_optimizer_state_dict": vanilla_optimizer.state_dict(),
            "mlp_state_dict": mlp_model.state_dict(),
            "mlp_optimizer_state_dict": mlp_optimizer.state_dict(),
            "report": result,
        },
        checkpoint_path,
    )
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
