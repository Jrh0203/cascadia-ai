"""Public-token CRT merit pilot.

This is the next step after the scalar `torch_action_query_merit` pilot. It
uses simulator-exported public tokens and C-GAB-style relation summaries instead
of a tiny scalar state vector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl
from .torch_action_query_merit import (
    _action_immediate_score,
    _baseline_metrics,
    _coord_features,
    _decision,
    _loss,
    _masked_fill_invalid,
    _normalizer,
    _safe_float,
    _species_one_hot,
    _tile_id,
    merit_action_features,
    parameter_count,
)

TOKEN_KINDS = (
    "player",
    "placed_tile",
    "frontier",
    "market_tile",
    "market_wildlife",
    "public_supply",
)
PUBLIC_TOKEN_FEATURE_DIM = 41
PUBLIC_TOKEN_ACTION_FEATURE_DIM = 33


@dataclass(frozen=True)
class PublicTokenConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = PUBLIC_TOKEN_ACTION_FEATURE_DIM
    hidden_dim: int = 160
    layers: int = 3
    heads: int = 5
    mlp_dim: int = 320
    dropout: float = 0.0
    model_name: str = "CRT-public-token-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relation_degrees(root: dict[str, Any]) -> dict[int, dict[str, float]]:
    degrees: dict[int, dict[str, float]] = {}
    for relation in root["public_tokens"].get("relations", []):
        source = int(relation["source"])
        target = int(relation["target"])
        source_degrees = degrees.setdefault(source, {})
        target_degrees = degrees.setdefault(target, {})
        kind = relation.get("relation_kind")
        if kind == "adjacent_hex":
            source_degrees["adjacent_out"] = source_degrees.get("adjacent_out", 0.0) + 1.0
            target_degrees["adjacent_in"] = target_degrees.get("adjacent_in", 0.0) + 1.0
            if relation.get("terrain_matches"):
                source_degrees["terrain_match_out"] = source_degrees.get("terrain_match_out", 0.0) + 1.0
        elif kind == "same_market_slot":
            source_degrees["market_pair"] = source_degrees.get("market_pair", 0.0) + 1.0
            target_degrees["market_pair"] = target_degrees.get("market_pair", 0.0) + 1.0
    return degrees


def public_token_features(root: dict[str, Any]) -> list[list[float]]:
    public_tokens = root.get("public_tokens")
    if not public_tokens:
        raise KeyError("root is missing public_tokens; regenerate with the enriched exporter")
    degrees = _relation_degrees(root)
    rows = []
    for token in public_tokens["tokens"]:
        kind = str(token.get("token_kind"))
        row = [1.0 if kind == expected else 0.0 for expected in TOKEN_KINDS]
        coord = token.get("coord_ref")
        degree = degrees.get(int(token["token_index"]), {})
        row.extend(
            [
                _normalizer(_safe_float(token.get("owner_seat"), -1.0), 3.0),
                _normalizer(_safe_float(token.get("relative_seat"), -1.0), 3.0),
                _normalizer(_safe_float(token.get("market_slot"), -1.0), 3.0),
            ]
        )
        row.extend(_coord_features(coord) if coord is not None else [0.0] * 6)
        row.extend(
            [
                _normalizer(_safe_float(token.get("nature_tokens")), 10.0),
                _normalizer(_safe_float(token.get("tile_count")), 23.0),
                _normalizer(_safe_float(token.get("current_base_score")), 100.0),
                _normalizer(_safe_float(token.get("current_wildlife_total")), 80.0),
                _normalizer(_safe_float(token.get("current_habitat_total")), 50.0),
                _normalizer(_safe_float(token.get("tile_id")), 84.0),
                _normalizer(_safe_float(token.get("terrain_a"), -1.0), 4.0),
                _normalizer(_safe_float(token.get("terrain_b"), -1.0), 4.0),
                _normalizer(_safe_float(token.get("wildlife_mask")), 31.0),
                1.0 if token.get("keystone") else 0.0,
                _normalizer(_safe_float(token.get("rotation")), 5.0),
                _normalizer(_safe_float(token.get("placed_wildlife"), -1.0), 4.0),
                _normalizer(_safe_float(token.get("species"), -1.0), 4.0),
                _normalizer(_safe_float(token.get("neighbor_count")), 6.0),
                1.0 if token.get("active_frontier") else 0.0,
                _normalizer(degree.get("adjacent_out", 0.0), 6.0),
                _normalizer(degree.get("adjacent_in", 0.0), 6.0),
                _normalizer(degree.get("terrain_match_out", 0.0), 6.0),
                _normalizer(degree.get("market_pair", 0.0), 2.0),
            ]
        )
        wildlife_bag = token.get("wildlife_bag") or [0, 0, 0, 0, 0]
        row.extend(_normalizer(_safe_float(value), 100.0) for value in wildlife_bag[:5])
        terrain_capacity = token.get("unseen_tile_terrain_capacity") or [0, 0, 0, 0, 0]
        wildlife_capacity = token.get("unseen_tile_wildlife_capacity") or [0, 0, 0, 0, 0]
        row.extend(
            [
                _normalizer(sum(_safe_float(value) for value in terrain_capacity), 100.0),
                _normalizer(sum(_safe_float(value) for value in wildlife_capacity), 100.0),
            ]
        )
        if len(row) != PUBLIC_TOKEN_FEATURE_DIM:
            raise ValueError(f"public token feature dimension mismatch: {len(row)}")
        rows.append(row)
    return rows


def public_token_action_features(root: dict[str, Any]) -> list[list[float]]:
    base_rows = merit_action_features(root)
    rows = []
    for base, action in zip(base_rows, root["legal_actions"], strict=True):
        row = list(base)
        row.extend(
            [
                _normalizer(_safe_float(action.get("tile_slot", action.get("draft_slot"))), 3.0),
                _normalizer(_safe_float(action.get("wildlife_slot", action.get("draft_slot"))), 3.0),
                _normalizer(_safe_float(action.get("tile_id"), _tile_id(action)), 84.0),
                _normalizer(_safe_float(action.get("tile_terrain_a"), -1.0), 4.0),
                _normalizer(_safe_float(action.get("tile_terrain_b"), -1.0), 4.0),
                _normalizer(_safe_float(action.get("tile_wildlife_mask")), 31.0),
                1.0 if action.get("tile_keystone") else 0.0,
                _normalizer(_safe_float(action.get("wildlife_species"), -1.0), 4.0),
            ]
        )
        if len(row) != PUBLIC_TOKEN_ACTION_FEATURE_DIM:
            raise ValueError(f"public token action feature dimension mismatch: {len(row)}")
        rows.append(row)
    return rows


class PublicTokenJsonlDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records = read_replay_jsonl(self.path)
        missing = [record["state_hash"] for record in self.records if "public_tokens" not in record]
        if missing:
            raise ValueError(f"{self.path}: {len(missing)} records are missing public_tokens")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    @property
    def action_counts(self) -> list[int]:
        return [len(record["legal_actions"]) for record in self.records]

    @property
    def token_counts(self) -> list[int]:
        return [record["public_tokens"]["token_count"] for record in self.records]

    @property
    def relation_counts(self) -> list[int]:
        return [record["public_tokens"]["relation_count"] for record in self.records]


def collate_public_token_roots(
    records: list[dict[str, Any]],
    *,
    action_feature_fn=public_token_action_features,  # type: ignore[no-untyped-def]
    action_feature_dim: int = PUBLIC_TOKEN_ACTION_FEATURE_DIM,
) -> dict[str, Any]:
    import torch

    if not records:
        raise ValueError("collate_public_token_roots requires at least one record")

    batch_size = len(records)
    action_counts = [len(record["legal_actions"]) for record in records]
    token_counts = [record["public_tokens"]["token_count"] for record in records]
    max_actions = max(action_counts)
    max_tokens = max(token_counts)
    tokens = torch.zeros((batch_size, max_tokens, PUBLIC_TOKEN_FEATURE_DIM), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    actions = torch.zeros((batch_size, max_actions, action_feature_dim), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    target_q = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_z = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_policy = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_q_count = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_q_variance = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    immediate = torch.zeros((batch_size, max_actions), dtype=torch.float32)

    for batch_index, record in enumerate(records):
        action_count = action_counts[batch_index]
        token_count = token_counts[batch_index]
        q_values = torch.tensor(record["per_action_Q"], dtype=torch.float32)
        q_counts = torch.tensor(record.get("per_action_Q_count", [1.0] * action_count), dtype=torch.float32)
        q_variances = torch.tensor(record.get("per_action_Q_variance", [0.0] * action_count), dtype=torch.float32)
        mean = q_values.mean()
        std = q_values.std(unbiased=False).clamp_min(1.0)
        z_values = (q_values - mean) / std
        tokens[batch_index, :token_count] = torch.tensor(public_token_features(record), dtype=torch.float32)
        token_mask[batch_index, :token_count] = True
        actions[batch_index, :action_count] = torch.tensor(
            action_feature_fn(record),
            dtype=torch.float32,
        )
        action_mask[batch_index, :action_count] = True
        target_q[batch_index, :action_count] = q_values
        target_z[batch_index, :action_count] = z_values
        target_policy[batch_index, :action_count] = torch.softmax(z_values, dim=0)
        target_q_count[batch_index, :action_count] = q_counts
        target_q_variance[batch_index, :action_count] = q_variances
        immediate[batch_index, :action_count] = torch.tensor(
            [_action_immediate_score(action) for action in record["legal_actions"]],
            dtype=torch.float32,
        )

    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "action_mask": action_mask,
        "target_q": target_q,
        "target_z": target_z,
        "target_policy": target_policy,
        "target_q_count": target_q_count,
        "target_q_variance": target_q_variance,
        "immediate": immediate,
        "action_counts": action_counts,
        "token_counts": token_counts,
        "relation_counts": [record["public_tokens"]["relation_count"] for record in records],
        "state_hashes": [record["state_hash"] for record in records],
        "action_ids": [[action["action_id"] for action in record["legal_actions"]] for record in records],
    }


def make_public_token_loader(path: str | Path, *, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = PublicTokenJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_public_token_roots,
    )


def _to_device(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    tensor_keys = {
        "tokens",
        "token_mask",
        "actions",
        "action_mask",
        "target_q",
        "target_z",
        "target_policy",
        "target_q_count",
        "target_q_variance",
        "immediate",
    }
    return {key: value.to(device) if key in tensor_keys else value for key, value in batch.items()}


def build_public_token_transformer(config: PublicTokenConfig):
    import torch
    from torch import nn

    class PublicTokenTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_proj = nn.Linear(config.token_feature_dim, config.hidden_dim)
            self.action_proj = nn.Linear(config.action_feature_dim, config.hidden_dim)
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

        def forward(self, tokens, token_mask, actions, action_mask):  # type: ignore[no-untyped-def]
            batch_size, token_count, _ = tokens.shape
            action_count = actions.shape[1]
            token_h = self.token_proj(tokens)
            action_h = self.action_proj(actions)
            type_ids = torch.zeros((batch_size, token_count + action_count), dtype=torch.long, device=tokens.device)
            type_ids[:, token_count:] = 1
            combined = torch.cat([token_h, action_h], dim=1) + self.type_embedding(type_ids)
            padding_mask = torch.cat([~token_mask, ~action_mask], dim=1)
            encoded = self.norm(self.encoder(combined, src_key_padding_mask=padding_mask))
            action_encoded = encoded[:, token_count:]
            return {
                "q": self.q_head(action_encoded).squeeze(-1),
                "logits": self.policy_head(action_encoded).squeeze(-1),
            }

    return PublicTokenTransformer()


def build_public_token_mlp(config: PublicTokenConfig):
    import torch
    from torch import nn

    class PublicTokenMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_proj = nn.Sequential(
                nn.Linear(config.token_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.net = nn.Sequential(
                nn.Linear(config.hidden_dim + config.action_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.q_head = nn.Linear(config.hidden_dim, 1)
            self.policy_head = nn.Linear(config.hidden_dim, 1)

        def forward(self, tokens, token_mask, actions, action_mask):  # type: ignore[no-untyped-def]
            token_h = self.token_proj(tokens)
            mask_f = token_mask.to(token_h.dtype).unsqueeze(-1)
            pooled = (token_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            pooled = pooled.unsqueeze(1).expand(-1, actions.shape[1], -1)
            hidden = self.net(torch.cat([pooled, actions], dim=-1))
            return {
                "q": self.q_head(hidden).squeeze(-1),
                "logits": self.policy_head(hidden).squeeze(-1),
            }

    return PublicTokenMLP()


def _model_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])["q"]


def _model_loss(model, batch):  # type: ignore[no-untyped-def]
    return _loss(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
    )


def _all_batches(path: Path, *, batch_size: int) -> list[dict[str, Any]]:
    return list(make_public_token_loader(path, batch_size=batch_size, shuffle=False))


def _baseline_metrics_from_batches(batches: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _baseline_metrics(batches, field)


def _evaluate_public_scores(batches: list[dict[str, Any]], score_fn, device=None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch

    total_roots = 0
    top1 = 0
    top4 = 0
    regret_sum = 0.0
    best_q_sum = 0.0
    selected_q_sum = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    prefilter_hits = {2: 0, 4: 0, 8: 0, 16: 0, 24: 0, 32: 0}
    prefilter_regret_sum = {2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0, 24: 0.0, 32: 0.0}
    prefilter_q_sum = {2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0, 24: 0.0, 32: 0.0}

    for batch in batches:
        eval_batch = _to_device(batch, device) if device is not None else batch
        with torch.no_grad():
            scores = score_fn(eval_batch).detach().cpu()
        target_q = batch["target_q"]
        mask = batch["action_mask"]
        for row in range(mask.shape[0]):
            valid_count = int(mask[row].sum().item())
            q = target_q[row, :valid_count]
            pred = scores[row, :valid_count]
            teacher_best = int(torch.argmax(q).item())
            selected = int(torch.argmax(pred).item())
            ranked = torch.argsort(pred, descending=True)
            topk = ranked[: min(4, valid_count)].tolist()
            best_q = float(q[teacher_best].item())
            selected_q = float(q[selected].item())
            total_roots += 1
            top1 += int(selected == teacher_best)
            top4 += int(teacher_best in topk)
            regret_sum += best_q - selected_q
            best_q_sum += best_q
            selected_q_sum += selected_q
            for k in prefilter_hits:
                retained = ranked[: min(k, valid_count)]
                retained_indices = retained.tolist()
                retained_best_q = float(q[retained].max().item())
                prefilter_hits[k] += int(teacher_best in retained_indices)
                prefilter_regret_sum[k] += best_q - retained_best_q
                prefilter_q_sum[k] += retained_best_q
            for left in range(valid_count):
                for right in range(left + 1, valid_count):
                    q_diff = float(q[left] - q[right])
                    if abs(q_diff) < 1.0e-9:
                        continue
                    pred_diff = float(pred[left] - pred[right])
                    if abs(pred_diff) < 1.0e-9:
                        continue
                    pairwise_total += 1
                    pairwise_correct += int((q_diff > 0) == (pred_diff > 0))

    return {
        "roots": total_roots,
        "top1_agreement": top1 / total_roots if total_roots else 0.0,
        "top4_recall": top4 / total_roots if total_roots else 0.0,
        "mean_regret": regret_sum / total_roots if total_roots else 0.0,
        "mean_best_q": best_q_sum / total_roots if total_roots else 0.0,
        "mean_selected_q": selected_q_sum / total_roots if total_roots else 0.0,
        "top2_recall": prefilter_hits[2] / total_roots if total_roots else 0.0,
        "top8_recall": prefilter_hits[8] / total_roots if total_roots else 0.0,
        "top16_recall": prefilter_hits[16] / total_roots if total_roots else 0.0,
        "top24_recall": prefilter_hits[24] / total_roots if total_roots else 0.0,
        "top32_recall": prefilter_hits[32] / total_roots if total_roots else 0.0,
        "mean_top4_oracle_regret": prefilter_regret_sum[4] / total_roots if total_roots else 0.0,
        "mean_top8_oracle_regret": prefilter_regret_sum[8] / total_roots if total_roots else 0.0,
        "mean_top16_oracle_regret": prefilter_regret_sum[16] / total_roots if total_roots else 0.0,
        "mean_top24_oracle_regret": prefilter_regret_sum[24] / total_roots if total_roots else 0.0,
        "mean_top32_oracle_regret": prefilter_regret_sum[32] / total_roots if total_roots else 0.0,
        "prefilter": {
            str(k): {
                "recall": prefilter_hits[k] / total_roots if total_roots else 0.0,
                "mean_oracle_regret": prefilter_regret_sum[k] / total_roots if total_roots else 0.0,
                "mean_oracle_q": prefilter_q_sum[k] / total_roots if total_roots else 0.0,
            }
            for k in prefilter_hits
        },
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else 0.0,
        "pairwise_total": pairwise_total,
    }


def _model_metrics(model, batches: list[dict[str, Any]], device) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    model.eval()
    return _evaluate_public_scores(
        batches,
        lambda batch: _model_scores(model, batch),
        device=device,
    )


def _train_model(model, train_path: Path, *, args: argparse.Namespace, device):  # type: ignore[no-untyped-def]
    import torch

    loader = make_public_token_loader(train_path, batch_size=args.batch_size, shuffle=True)
    loader_cycle = cycle(loader)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    model.train()
    for _ in range(args.steps):
        batch = _to_device(next(loader_cycle), device)
        optimizer.zero_grad(set_to_none=True)
        loss = _model_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, optimizer, losses


def _dataset_summary(path: Path) -> dict[str, Any]:
    dataset = PublicTokenJsonlDataset(path)
    action_counts = dataset.action_counts
    token_counts = dataset.token_counts
    relation_counts = dataset.relation_counts
    return {
        "path": str(path),
        "record_count": len(dataset),
        "action_counts": {
            "min": min(action_counts),
            "max": max(action_counts),
            "mean": sum(action_counts) / len(action_counts),
        },
        "token_counts": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
        },
        "relation_counts": {
            "min": min(relation_counts),
            "max": max(relation_counts),
            "mean": sum(relation_counts) / len(relation_counts),
        },
    }


def run_public_token_pilot(
    train_path: Path,
    val_path: Path,
    *,
    steps: int = 1600,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260630,
    device_name: str = "cuda",
    hidden_dim: int = 160,
    layers: int = 3,
    heads: int = 5,
    mlp_dim: int = 320,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    import torch

    args = argparse.Namespace(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    config = PublicTokenConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
    if config.hidden_dim % config.heads != 0:
        raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by heads {config.heads}")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_summary = _dataset_summary(train_path)
    val_summary = _dataset_summary(val_path)
    val_batches = _all_batches(val_path, batch_size=batch_size)

    transformer = build_public_token_transformer(config)
    transformer, transformer_optimizer, transformer_losses = _train_model(
        transformer,
        train_path,
        args=args,
        device=device,
    )
    mlp = build_public_token_mlp(config)
    mlp, mlp_optimizer, mlp_losses = _train_model(
        mlp,
        train_path,
        args=args,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    transformer_metrics = _model_metrics(transformer, val_batches, device)
    mlp_metrics = _model_metrics(mlp, val_batches, device)
    immediate_metrics = _baseline_metrics_from_batches(val_batches, "immediate")
    decision = _decision(transformer_metrics, mlp_metrics, immediate_metrics)

    return {
        "status": "pass",
        "scientific_eligibility": "dry_run",
        "experiment_id": "crt-public-token-query-merit-v1",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "train_dataset": train_summary,
        "val_dataset": val_summary,
        "models": {
            "transformer": {
                "parameter_count": parameter_count(transformer),
                "loss_head": transformer_losses[:5],
                "loss_tail": transformer_losses[-5:],
                "metrics": transformer_metrics,
            },
            "mlp": {
                "parameter_count": parameter_count(mlp),
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
        "transformer_model": transformer,
        "transformer_optimizer": transformer_optimizer,
        "mlp_model": mlp,
        "mlp_optimizer": mlp_optimizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_token_merit_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_token_merit_val.jsonl")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--mlp-dim", type=int, default=320)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="cascadiav3/reports/crt_public_token_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_public_token_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_public_token_pilot(
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
    )
    transformer = result.pop("transformer_model")
    transformer_optimizer = result.pop("transformer_optimizer")
    mlp = result.pop("mlp_model")
    mlp_optimizer = result.pop("mlp_optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "transformer_state_dict": transformer.state_dict(),
            "transformer_optimizer_state_dict": transformer_optimizer.state_dict(),
            "mlp_state_dict": mlp.state_dict(),
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
