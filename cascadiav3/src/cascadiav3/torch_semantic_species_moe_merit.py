"""Species-aware semantic relation-bias CRT merit pilot.

The phase-diverse K=16 misses are concentrated around wildlife-specific
allocation choices, especially elk, hawk, and salmon. This module keeps the
semantic public-token/relation-bias encoder, then routes action scoring through
small per-species residual heads. The legal-action query contract is unchanged:
the model still scores every exact legal action.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from .torch_action_query_merit import _safe_float, parameter_count
from .torch_public_token_merit import (
    PUBLIC_TOKEN_FEATURE_DIM,
    PublicTokenJsonlDataset,
    _baseline_metrics_from_batches,
    build_public_token_mlp,
    build_public_token_transformer,
)
from .torch_relation_bias_merit import (
    RELATION_VOCAB_SIZE,
    _decision_with_vanilla,
    _evaluate_relation_scores,
    _loss_with_mode,
    _public_scores,
    _to_device,
)
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    collate_semantic_relation_bias_roots,
    semantic_all_batches,
    semantic_dataset_summary,
    make_semantic_relation_bias_loader,
)

SPECIES_EXPERT_COUNT = 6
SPECIES_EXPERT_NAMES = ("none", "bear", "elk", "salmon", "hawk", "fox")


@dataclass(frozen=True)
class SemanticSpeciesMoEConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    relation_vocab_size: int = RELATION_VOCAB_SIZE
    species_expert_count: int = SPECIES_EXPERT_COUNT
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0
    expert_scale: float = 0.5
    model_name: str = "CRT-semantic-species-moe-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _species_id(action: dict[str, Any]) -> int:
    species = int(_safe_float(action.get("wildlife_species"), -1.0))
    if 0 <= species < 5 and bool(action.get("wildlife_placement_present", True)):
        return species + 1
    return 0


def collate_semantic_species_moe_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_semantic_relation_bias_roots(records)
    max_actions = batch["actions"].shape[1]
    action_species = torch.zeros((len(records), max_actions), dtype=torch.long)
    for batch_index, record in enumerate(records):
        for action_index, action in enumerate(record["legal_actions"]):
            action_species[batch_index, action_index] = _species_id(action)
    batch["action_species"] = action_species
    return batch


def make_semantic_species_moe_loader(path: str | Path, *, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = PublicTokenJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_semantic_species_moe_roots,
    )


def species_all_batches(path: Path, *, batch_size: int) -> list[dict[str, Any]]:
    return list(make_semantic_species_moe_loader(path, batch_size=batch_size, shuffle=False))


def build_semantic_species_moe_transformer(config: SemanticSpeciesMoEConfig):
    import torch
    from torch import nn

    class RelationBiasedEncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.attn = nn.MultiheadAttention(
                config.hidden_dim,
                config.heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.dropout1 = nn.Dropout(config.dropout)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.mlp_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mlp_dim, config.hidden_dim),
            )
            self.dropout2 = nn.Dropout(config.dropout)
            self.relation_bias = nn.Embedding(config.relation_vocab_size, config.heads)
            nn.init.zeros_(self.relation_bias.weight)

        def forward(self, x, padding_mask, relation_ids):  # type: ignore[no-untyped-def]
            batch_size, seq_len, _ = x.shape
            bias = self.relation_bias(relation_ids)
            bias = bias.permute(0, 3, 1, 2).reshape(batch_size * config.heads, seq_len, seq_len)
            key_mask = padding_mask[:, None, None, :].expand(batch_size, config.heads, seq_len, seq_len)
            bias = bias.masked_fill(key_mask.reshape(batch_size * config.heads, seq_len, seq_len), -1.0e4)
            attn_in = self.norm1(x)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=bias, need_weights=False)
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x

    class SemanticSpeciesMoETransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_proj = nn.Linear(config.token_feature_dim, config.hidden_dim)
            self.action_proj = nn.Linear(config.action_feature_dim, config.hidden_dim)
            self.type_embedding = nn.Embedding(2, config.hidden_dim)
            self.species_embedding = nn.Embedding(config.species_expert_count, config.hidden_dim)
            self.layers = nn.ModuleList([RelationBiasedEncoderLayer() for _ in range(config.layers)])
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.shared_q_head = nn.Linear(config.hidden_dim, 1)
            self.shared_policy_head = nn.Linear(config.hidden_dim, 1)
            self.species_q_head = nn.Linear(config.hidden_dim, config.species_expert_count)
            self.species_policy_head = nn.Linear(config.hidden_dim, config.species_expert_count)

        def forward(self, tokens, token_mask, actions, action_mask, relation_ids, action_species):  # type: ignore[no-untyped-def]
            batch_size, token_count, _ = tokens.shape
            action_count = actions.shape[1]
            token_h = self.token_proj(tokens)
            species_ids = action_species.clamp(0, config.species_expert_count - 1)
            action_h = self.action_proj(actions) + self.species_embedding(species_ids)
            type_ids = torch.zeros((batch_size, token_count + action_count), dtype=torch.long, device=tokens.device)
            type_ids[:, token_count:] = 1
            hidden = torch.cat([token_h, action_h], dim=1) + self.type_embedding(type_ids)
            padding_mask = torch.cat([~token_mask, ~action_mask], dim=1)
            for layer in self.layers:
                hidden = layer(hidden, padding_mask, relation_ids)
            hidden = self.norm(hidden)
            action_hidden = hidden[:, token_count:]
            expert_index = species_ids.unsqueeze(-1)
            species_q = self.species_q_head(action_hidden).gather(2, expert_index).squeeze(-1)
            species_policy = self.species_policy_head(action_hidden).gather(2, expert_index).squeeze(-1)
            return {
                "q": self.shared_q_head(action_hidden).squeeze(-1) + config.expert_scale * species_q,
                "logits": self.shared_policy_head(action_hidden).squeeze(-1)
                + config.expert_scale * species_policy,
            }

    return SemanticSpeciesMoETransformer()


def _species_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(
        batch["tokens"],
        batch["token_mask"],
        batch["actions"],
        batch["action_mask"],
        batch["relation_ids"],
        batch["action_species"],
    )["q"]


def _species_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(
            batch["tokens"],
            batch["token_mask"],
            batch["actions"],
            batch["action_mask"],
            batch["relation_ids"],
            batch["action_species"],
        ),
        batch,
        args,
    )


def _semantic_public_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
        args,
    )


def _train_model(model, train_path: Path, *, args: argparse.Namespace, device, loss_fn, species_loader: bool):  # type: ignore[no-untyped-def]
    import torch

    loader = (
        make_semantic_species_moe_loader(train_path, batch_size=args.batch_size, shuffle=True)
        if species_loader
        else make_semantic_relation_bias_loader(train_path, batch_size=args.batch_size, shuffle=True)
    )
    loader_cycle = cycle(loader)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    model.train()
    for _ in range(args.steps):
        batch = _to_device(next(loader_cycle), device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, optimizer, losses


def _metrics(model, batches: list[dict[str, Any]], device, score_fn) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    model.eval()
    return _evaluate_relation_scores(batches, lambda batch: score_fn(model, batch), device=device)


def _species_distribution(path: Path) -> dict[str, int]:
    counts = {name: 0 for name in SPECIES_EXPERT_NAMES}
    for record in PublicTokenJsonlDataset(path).records:
        for action in record["legal_actions"]:
            counts[SPECIES_EXPERT_NAMES[_species_id(action)]] += 1
    return counts


def run_semantic_species_moe_pilot(
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
    loss_mode: str = "standard",
    q_loss_weight: float = 0.25,
    policy_loss_weight: float = 0.5,
    best_margin_loss_weight: float = 1.0,
    retention_loss_weight: float = 1.0,
    retention_k: int = 16,
    pairwise_margin: float = 0.25,
    policy_temperature: float = 0.5,
    expert_scale: float = 0.5,
    experiment_id: str = "crt-wide32-r16p20-semantic-species-moe-v1",
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
    config = SemanticSpeciesMoEConfig(
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        mlp_dim=mlp_dim,
        expert_scale=expert_scale,
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
    train_summary["species_expert_action_counts"] = _species_distribution(train_path)
    val_summary["species_expert_action_counts"] = _species_distribution(val_path)
    val_batches = species_all_batches(val_path, batch_size=batch_size)

    species_model = build_semantic_species_moe_transformer(config)
    species_model, species_optimizer, species_losses = _train_model(
        species_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _species_loss_with_mode(model, batch, args),
        species_loader=True,
    )

    vanilla_model = build_public_token_transformer(config)
    vanilla_model, vanilla_optimizer, vanilla_losses = _train_model(
        vanilla_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _semantic_public_loss_with_mode(model, batch, args),
        species_loader=False,
    )
    mlp_model = build_public_token_mlp(config)
    mlp_model, mlp_optimizer, mlp_losses = _train_model(
        mlp_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _semantic_public_loss_with_mode(model, batch, args),
        species_loader=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    species_metrics = _metrics(species_model, val_batches, device, _species_scores)
    semantic_val_batches = semantic_all_batches(val_path, batch_size=batch_size)
    vanilla_metrics = _metrics(vanilla_model, semantic_val_batches, device, _public_scores)
    mlp_metrics = _metrics(mlp_model, semantic_val_batches, device, _public_scores)
    immediate_metrics = _baseline_metrics_from_batches(semantic_val_batches, "immediate")
    decision = _decision_with_vanilla(species_metrics, vanilla_metrics, mlp_metrics, immediate_metrics)

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
            "semantic_species_moe_transformer": {
                "parameter_count": parameter_count(species_model),
                "loss_head": species_losses[:5],
                "loss_tail": species_losses[-5:],
                "metrics": species_metrics,
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
        "baselines": {"immediate_base": immediate_metrics},
        "decision": decision,
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "species_model": species_model,
        "species_optimizer": species_optimizer,
        "vanilla_model": vanilla_model,
        "vanilla_optimizer": vanilla_optimizer,
        "mlp_model": mlp_model,
        "mlp_optimizer": mlp_optimizer,
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    species = report["models"]["semantic_species_moe_transformer"]["metrics"]
    vanilla = report["models"]["vanilla_public_token_transformer"]["metrics"]
    mlp = report["models"]["token_pooled_mlp"]["metrics"]
    immediate = report["baselines"]["immediate_base"]
    rows = [
        ("Species-MoE Transformer", species),
        ("Vanilla Transformer", vanilla),
        ("Token-pooled MLP", mlp),
        ("Immediate", immediate),
    ]
    lines = [
        "# CRT Wide-32 R16p20 Semantic Species-MoE Pilot",
        "",
        f"Experiment: `{report['experiment_id']}`",
        "",
        "| Model | Top-1 | Mean regret | K=16 recall | K=16 oracle regret | K=24 recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            "| "
            f"{name} | {metrics['top1_agreement']:.4f} | {metrics['mean_regret']:.4f} | "
            f"{metrics['prefilter']['16']['recall']:.4f} | "
            f"{metrics['prefilter']['16']['mean_oracle_regret']:.4f} | "
            f"{metrics['prefilter']['24']['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `has_merit`: `{report['decision']['has_merit']}`",
            f"- `beats_vanilla_transformer`: `{report['decision']['beats_vanilla_transformer']}`",
            f"- Species expert scale: `{report['config']['expert_scale']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--loss-mode", choices=["standard", "top16-prefilter", "topk-retention"], default="standard")
    parser.add_argument("--q-loss-weight", type=float, default=0.25)
    parser.add_argument("--policy-loss-weight", type=float, default=0.5)
    parser.add_argument("--best-margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-k", type=int, default=16)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--expert-scale", type=float, default=0.5)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-species-moe-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_species_moe_pilot.json")
    parser.add_argument("--summary-out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_species_moe_pilot_summary.md")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_species_moe_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_semantic_species_moe_pilot(
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
        expert_scale=args.expert_scale,
        experiment_id=args.experiment_id,
    )
    species_model = result.pop("species_model")
    species_optimizer = result.pop("species_optimizer")
    vanilla_model = result.pop("vanilla_model")
    vanilla_optimizer = result.pop("vanilla_optimizer")
    mlp_model = result.pop("mlp_model")
    mlp_optimizer = result.pop("mlp_optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "species_moe_state_dict": species_model.state_dict(),
            "species_moe_optimizer_state_dict": species_optimizer.state_dict(),
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
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(result, summary_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
