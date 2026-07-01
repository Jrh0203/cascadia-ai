"""CRT-mini action-query merit pilot.

This is the first real offline transformer-vs-baseline check for the v3 plan.
It is deliberately not a strength claim: labels come from the current
real-root dry-run exporter unless the caller supplies stronger teacher shards.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl

SPECIES = ("Bear", "Elk", "Salmon", "Hawk", "Fox")
MERIT_STATE_FEATURE_DIM = 12
MERIT_ACTION_FEATURE_DIM = 25


@dataclass(frozen=True)
class MeritConfig:
    state_feature_dim: int = MERIT_STATE_FEATURE_DIM
    action_feature_dim: int = MERIT_ACTION_FEATURE_DIM
    hidden_dim: int = 128
    layers: int = 3
    heads: int = 4
    mlp_dim: int = 256
    dropout: float = 0.0
    policy_temperature: float = 1.0
    model_name: str = "CRT-mini-action-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalizer(value: float, scale: float) -> float:
    return value / scale if scale else value


def _action_immediate_score(action: dict[str, Any]) -> float:
    return _safe_float(action.get("immediate_pre_rollout_base_score"), 0.0)


def _tile_id(action: dict[str, Any]) -> float:
    tile_ref = str(action.get("tile_ref", ""))
    if tile_ref.startswith("tile:"):
        try:
            return float(tile_ref.split(":", 1)[1].split("@", 1)[0])
        except ValueError:
            return 0.0
    return 0.0


def _species_one_hot(action: dict[str, Any]) -> list[float]:
    wildlife_ref = str(action.get("wildlife_ref", ""))
    return [1.0 if wildlife_ref.startswith(species) else 0.0 for species in SPECIES]


def _coord_features(coord: dict[str, Any], *, radius_scale: float = 6.0) -> list[float]:
    return [
        _normalizer(_safe_float(coord.get("q")), radius_scale),
        _normalizer(_safe_float(coord.get("r")), radius_scale),
        _normalizer(_safe_float(coord.get("s")), radius_scale),
        1.0 if coord.get("kind") == "canonical" else 0.0,
        1.0 if coord.get("kind") == "overflow" else 0.0,
        _normalizer(
            _safe_float(coord.get("cell_index"), -1.0)
            if coord.get("cell_index") is not None
            else -1.0,
            126.0,
        ),
    ]


def merit_state_features(root: dict[str, Any]) -> list[float]:
    actions = root["legal_actions"]
    immediate = [_action_immediate_score(action) for action in actions]
    mean_immediate = sum(immediate) / len(immediate)
    variance = sum((value - mean_immediate) ** 2 for value in immediate) / len(immediate)
    metadata = root.get("metadata", {})
    return [
        _normalizer(_safe_float(root.get("active_seat")), 3.0),
        _normalizer(float(len(actions)), 64.0),
        _normalizer(_safe_float(metadata.get("completed_turns")), 80.0),
        _normalizer(_safe_float(metadata.get("turns_remaining")), 80.0),
        _normalizer(_safe_float(metadata.get("max_actions"), len(actions)), 64.0),
        1.0 if metadata.get("prelude_replace_three_of_a_kind") else 0.0,
        _normalizer(_safe_float(metadata.get("prelude_wildlife_wipe_count")), 4.0),
        _normalizer(mean_immediate, 100.0),
        _normalizer(math.sqrt(variance), 50.0),
        _normalizer(max(immediate), 100.0),
        _normalizer(min(immediate), 100.0),
        sum(_safe_float(action.get("nature_spend")) for action in actions) / len(actions),
    ]


def merit_action_features(root: dict[str, Any]) -> list[list[float]]:
    rows = []
    for action in root["legal_actions"]:
        row = [
            _normalizer(_safe_float(action.get("active_seat")), 3.0),
            _safe_float(action.get("nature_spend")),
            _normalizer(_safe_float(action.get("draft_slot")), 3.0),
            _normalizer(_safe_float(action.get("rotation")), 5.0),
        ]
        row.extend(_coord_features(action["target_coord_ref"]))
        row.extend(_coord_features(action["wildlife_coord_ref"]))
        row.extend(
            [
                _normalizer(_action_immediate_score(action), 100.0),
                1.0 if action.get("wildlife_placement_present") else 0.0,
                _normalizer(_tile_id(action), 84.0),
                0.0 if action.get("cleanup_choice") == "none" else 1.0,
            ]
        )
        row.extend(_species_one_hot(action))
        if len(row) != MERIT_ACTION_FEATURE_DIM:
            raise ValueError(f"action feature dimension mismatch: {len(row)}")
        rows.append(row)
    return rows


class MeritJsonlDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records = read_replay_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    @property
    def action_counts(self) -> list[int]:
        return [len(record["legal_actions"]) for record in self.records]


def collate_merit_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not records:
        raise ValueError("collate_merit_roots requires at least one record")

    batch_size = len(records)
    action_counts = [len(record["legal_actions"]) for record in records]
    max_actions = max(action_counts)
    state = torch.zeros((batch_size, MERIT_STATE_FEATURE_DIM), dtype=torch.float32)
    actions = torch.zeros((batch_size, max_actions, MERIT_ACTION_FEATURE_DIM), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    target_q = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_z = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_policy = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    immediate = torch.zeros((batch_size, max_actions), dtype=torch.float32)

    for batch_index, record in enumerate(records):
        count = action_counts[batch_index]
        q_values = torch.tensor(record["per_action_Q"], dtype=torch.float32)
        valid_actions = record["legal_actions"]
        immediate_values = torch.tensor(
            [_action_immediate_score(action) for action in valid_actions],
            dtype=torch.float32,
        )
        mean = q_values.mean()
        std = q_values.std(unbiased=False).clamp_min(1.0)
        z_values = (q_values - mean) / std
        state[batch_index] = torch.tensor(merit_state_features(record), dtype=torch.float32)
        actions[batch_index, :count] = torch.tensor(merit_action_features(record), dtype=torch.float32)
        action_mask[batch_index, :count] = True
        target_q[batch_index, :count] = q_values
        target_z[batch_index, :count] = z_values
        target_policy[batch_index, :count] = torch.softmax(z_values, dim=0)
        immediate[batch_index, :count] = immediate_values

    return {
        "state": state,
        "actions": actions,
        "action_mask": action_mask,
        "target_q": target_q,
        "target_z": target_z,
        "target_policy": target_policy,
        "immediate": immediate,
        "action_counts": action_counts,
        "state_hashes": [record["state_hash"] for record in records],
    }


def make_merit_loader(path: str | Path, *, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = MeritJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_merit_roots,
    )


def _masked_fill_invalid(values, mask, fill_value=-1.0e9):  # type: ignore[no-untyped-def]
    return values.masked_fill(~mask, fill_value)


def _loss(output: dict[str, Any], batch: dict[str, Any]):
    import torch

    mask = batch["action_mask"]
    mask_f = mask.to(output["q"].dtype)
    q_loss = (((output["q"] - batch["target_z"]) ** 2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    log_probs = torch.log_softmax(_masked_fill_invalid(output["logits"], mask), dim=1)
    policy_loss = -(batch["target_policy"] * log_probs).sum(dim=1).mean()
    return q_loss + 0.5 * policy_loss


def _to_device(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    tensor_keys = {
        "state",
        "actions",
        "action_mask",
        "target_q",
        "target_z",
        "target_policy",
        "immediate",
    }
    return {key: value.to(device) if key in tensor_keys else value for key, value in batch.items()}


def build_action_query_transformer(config: MeritConfig):
    import torch
    from torch import nn

    class ActionQueryTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.state_proj = nn.Linear(config.state_feature_dim, config.hidden_dim)
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

        def forward(self, state, actions, action_mask):  # type: ignore[no-untyped-def]
            batch_size, action_count, _ = actions.shape
            state_token = self.state_proj(state).unsqueeze(1)
            action_tokens = self.action_proj(actions)
            type_ids = torch.zeros((batch_size, action_count + 1), dtype=torch.long, device=actions.device)
            type_ids[:, 1:] = 1
            tokens = torch.cat([state_token, action_tokens], dim=1) + self.type_embedding(type_ids)
            padding_mask = torch.cat(
                [
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=actions.device),
                    ~action_mask,
                ],
                dim=1,
            )
            encoded = self.norm(self.encoder(tokens, src_key_padding_mask=padding_mask))
            action_encoded = encoded[:, 1:]
            return {
                "q": self.q_head(action_encoded).squeeze(-1),
                "logits": self.policy_head(action_encoded).squeeze(-1),
            }

    return ActionQueryTransformer()


def build_action_mlp(config: MeritConfig):
    from torch import nn

    class ActionMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.net = nn.Sequential(
                nn.Linear(config.state_feature_dim + config.action_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.q_head = nn.Linear(config.hidden_dim, 1)
            self.policy_head = nn.Linear(config.hidden_dim, 1)

        def forward(self, state, actions, action_mask):  # type: ignore[no-untyped-def]
            import torch

            state_expanded = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
            hidden = self.net(torch.cat([state_expanded, actions], dim=-1))
            return {
                "q": self.q_head(hidden).squeeze(-1),
                "logits": self.policy_head(hidden).squeeze(-1),
            }

    return ActionMLP()


def parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return sum(param.numel() for param in model.parameters())


def _all_batches(path: Path, *, batch_size: int) -> list[dict[str, Any]]:
    return list(make_merit_loader(path, batch_size=batch_size, shuffle=False))


def _evaluate_scores(batches: list[dict[str, Any]], score_fn, device=None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
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
        "pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "pairwise_total": pairwise_total,
    }


def _model_metrics(model, batches: list[dict[str, Any]], device) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    model.eval()
    return _evaluate_scores(
        batches,
        lambda batch: model(batch["state"], batch["actions"], batch["action_mask"])["q"],
        device=device,
    )


def _baseline_metrics(batches: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _evaluate_scores(batches, lambda batch: batch[field])


def baseline_metrics_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure-Python-testable baseline wrapper for validated root records."""
    batches = [collate_merit_roots(records)]
    return {
        "immediate_base": _baseline_metrics(batches, "immediate"),
    }


def _train_model(model, train_path: Path, *, config: MeritConfig, args: argparse.Namespace, device):  # type: ignore[no-untyped-def]
    import torch

    loader = make_merit_loader(train_path, batch_size=args.batch_size, shuffle=True)
    loader_cycle = cycle(loader)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    model.train()
    for _ in range(args.steps):
        batch = _to_device(next(loader_cycle), device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["state"], batch["actions"], batch["action_mask"])
        loss = _loss(output, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, optimizer, losses


def _dataset_summary(path: Path) -> dict[str, Any]:
    dataset = MeritJsonlDataset(path)
    action_counts = dataset.action_counts
    return {
        "path": str(path),
        "record_count": len(dataset),
        "action_counts": {
            "min": min(action_counts),
            "max": max(action_counts),
            "mean": sum(action_counts) / len(action_counts),
        },
    }


def _decision(transformer: dict[str, Any], mlp: dict[str, Any], immediate: dict[str, Any]) -> dict[str, Any]:
    immediate_regret = immediate["mean_regret"]
    transformer_regret = transformer["mean_regret"]
    mlp_regret = mlp["mean_regret"]
    regret_improvement = (
        (immediate_regret - transformer_regret) / immediate_regret
        if immediate_regret > 1.0e-9
        else 0.0
    )
    top1_gain = transformer["top1_agreement"] - immediate["top1_agreement"]
    beats_immediate = regret_improvement >= 0.10 or top1_gain >= 0.05
    beats_mlp = (
        transformer_regret <= mlp_regret + 1.0e-9
        and transformer["top1_agreement"] >= mlp["top1_agreement"] - 1.0e-9
    )
    return {
        "has_merit": bool(beats_immediate and beats_mlp),
        "beats_immediate": bool(beats_immediate),
        "beats_mlp": bool(beats_mlp),
        "regret_improvement_vs_immediate": regret_improvement,
        "top1_gain_vs_immediate": top1_gain,
        "criteria": (
            "has_merit requires >=10% lower regret or >=5pp top1 gain versus "
            "immediate-base, and nonregression versus the trained MLP baseline"
        ),
    }


def run_merit_pilot(
    train_path: Path,
    val_path: Path,
    *,
    steps: int = 1200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 20260630,
    device_name: str = "cuda",
    hidden_dim: int = 128,
    layers: int = 3,
    heads: int = 4,
    mlp_dim: int = 256,
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
    config = MeritConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    train_summary = _dataset_summary(train_path)
    val_summary = _dataset_summary(val_path)
    val_batches = _all_batches(val_path, batch_size=batch_size)

    transformer = build_action_query_transformer(config)
    transformer, transformer_optimizer, transformer_losses = _train_model(
        transformer,
        train_path,
        config=config,
        args=args,
        device=device,
    )
    mlp = build_action_mlp(config)
    mlp, mlp_optimizer, mlp_losses = _train_model(
        mlp,
        train_path,
        config=config,
        args=args,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    transformer_metrics = _model_metrics(transformer, val_batches, device)
    mlp_metrics = _model_metrics(mlp, val_batches, device)
    immediate_metrics = _baseline_metrics(val_batches, "immediate")
    decision = _decision(transformer_metrics, mlp_metrics, immediate_metrics)

    return {
        "status": "pass",
        "scientific_eligibility": "dry_run",
        "experiment_id": "crt-mini-action-query-merit-v1",
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
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_merit_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_merit_val.jsonl")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="cascadiav3/reports/crt_merit_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_merit_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_merit_pilot(
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
