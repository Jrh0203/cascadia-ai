"""Semantic residual-attention CRT merit pilot.

The phase-diverse shard says the semantic token-pooled MLP is almost good enough
for K=16, while full attention variants damage that action-feature signal. This
model keeps a trainable MLP anchor and adds a bounded transformer residual from
public state tokens. The intended test is whether attention can make small
corrections without overriding the strong semantic-action baseline.
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
    _evaluate_relation_scores,
    _loss_with_mode,
    _public_scores,
    _to_device,
)
from .torch_semantic_cross_attention_merit import (
    _model_metrics,
    _public_loss_with_mode,
    _train_semantic_model,
)
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    SemanticRelationBiasConfig,
    semantic_all_batches,
    semantic_dataset_summary,
)


@dataclass(frozen=True)
class SemanticResidualAttentionConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0
    residual_scale: float = 0.25
    model_name: str = "CRT-semantic-residual-attention-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_semantic_residual_attention_transformer(config: Any):
    import torch
    from torch import nn

    class ResidualDecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_norm = nn.LayerNorm(config.hidden_dim)
            self.state_norm = nn.LayerNorm(config.hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_dim,
                config.heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.dropout1 = nn.Dropout(config.dropout)
            self.ffn_norm = nn.LayerNorm(config.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.mlp_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mlp_dim, config.hidden_dim),
            )
            self.dropout2 = nn.Dropout(config.dropout)

        def forward(self, actions, state, token_mask):  # type: ignore[no-untyped-def]
            state_norm = self.state_norm(state)
            attn_out, _ = self.cross_attn(
                self.action_norm(actions),
                state_norm,
                state_norm,
                key_padding_mask=~token_mask,
                need_weights=False,
            )
            actions = actions + self.dropout1(attn_out)
            actions = actions + self.dropout2(self.ffn(self.ffn_norm(actions)))
            return actions

    class SemanticResidualAttentionTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.anchor = build_public_token_mlp(config)
            self.token_proj = nn.Linear(config.token_feature_dim, config.hidden_dim)
            self.action_proj = nn.Linear(config.action_feature_dim, config.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.mlp_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
            self.decoder_layers = nn.ModuleList([ResidualDecoderLayer() for _ in range(config.layers)])
            self.action_norm = nn.LayerNorm(config.hidden_dim)
            self.q_residual_head = nn.Linear(config.hidden_dim, 1)
            self.policy_residual_head = nn.Linear(config.hidden_dim, 1)
            nn.init.normal_(self.q_residual_head.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.q_residual_head.bias)
            nn.init.normal_(self.policy_residual_head.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.policy_residual_head.bias)

        def forward(self, tokens, token_mask, actions, action_mask):  # type: ignore[no-untyped-def]
            anchor = self.anchor(tokens, token_mask, actions, action_mask)
            state = self.token_proj(tokens)
            state = self.state_encoder(state, src_key_padding_mask=~token_mask)
            action_hidden = self.action_proj(actions)
            for layer in self.decoder_layers:
                action_hidden = layer(action_hidden, state, token_mask)
            action_hidden = self.action_norm(action_hidden)
            q_residual = self.q_residual_head(action_hidden).squeeze(-1)
            policy_residual = self.policy_residual_head(action_hidden).squeeze(-1)
            scale = float(getattr(config, "residual_scale", 0.25))
            q = (anchor["q"] + scale * q_residual).masked_fill(~action_mask, 0.0)
            logits = (anchor["logits"] + scale * policy_residual).masked_fill(~action_mask, 0.0)
            return {"q": q, "logits": logits}

    return SemanticResidualAttentionTransformer()


def _residual_attention_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])["q"]


def _residual_attention_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
        args,
    )


def run_semantic_residual_attention_pilot(
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
    residual_scale: float = 0.25,
    grad_clip: float = 1.0,
    loss_mode: str = "topk-retention",
    q_loss_weight: float = 0.15,
    policy_loss_weight: float = 0.25,
    best_margin_loss_weight: float = 1.0,
    retention_loss_weight: float = 1.5,
    retention_k: int = 16,
    pairwise_margin: float = 0.15,
    policy_temperature: float = 0.75,
    experiment_id: str = "crt-semantic-residual-attention-query-merit-v1",
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
    config = SemanticResidualAttentionConfig(
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        mlp_dim=mlp_dim,
        residual_scale=residual_scale,
    )
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

    residual_model = build_semantic_residual_attention_transformer(config)
    residual_model, residual_optimizer, residual_losses = _train_semantic_model(
        residual_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _residual_attention_loss_with_mode(model, batch, args),
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

    residual_metrics = _model_metrics(residual_model, val_batches, device, _residual_attention_scores)
    vanilla_metrics = _model_metrics(vanilla_model, val_batches, device, _public_scores)
    mlp_metrics = _model_metrics(mlp_model, val_batches, device, _public_scores)
    immediate_metrics = _baseline_metrics_from_batches(val_batches, "immediate")
    decision = _decision_with_vanilla(residual_metrics, vanilla_metrics, mlp_metrics, immediate_metrics)
    decision["criteria"] = decision["criteria"].replace("relation-bias", "residual-attention")

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
            "residual_attention_transformer": {
                "parameter_count": parameter_count(residual_model),
                "loss_head": residual_losses[:5],
                "loss_tail": residual_losses[-5:],
                "metrics": residual_metrics,
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
        "residual_attention_model": residual_model,
        "residual_attention_optimizer": residual_optimizer,
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
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=["standard", "top16-prefilter", "topk-retention"], default="topk-retention")
    parser.add_argument("--q-loss-weight", type=float, default=0.15)
    parser.add_argument("--policy-loss-weight", type=float, default=0.25)
    parser.add_argument("--best-margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-loss-weight", type=float, default=1.5)
    parser.add_argument("--retention-k", type=int, default=16)
    parser.add_argument("--pairwise-margin", type=float, default=0.15)
    parser.add_argument("--policy-temperature", type=float, default=0.75)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-residual-attention-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_residual_attention_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_residual_attention_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_semantic_residual_attention_pilot(
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
        residual_scale=args.residual_scale,
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
    residual_model = result.pop("residual_attention_model")
    residual_optimizer = result.pop("residual_attention_optimizer")
    vanilla_model = result.pop("vanilla_model")
    vanilla_optimizer = result.pop("vanilla_optimizer")
    mlp_model = result.pop("mlp_model")
    mlp_optimizer = result.pop("mlp_optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "residual_attention_state_dict": residual_model.state_dict(),
            "residual_attention_optimizer_state_dict": residual_optimizer.state_dict(),
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
