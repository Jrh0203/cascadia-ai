"""Relation-biased public-token CRT merit pilot.

This iteration keeps the public-token/action-query setup but replaces scalarized
relation-degree summaries as the only structural signal. It builds C-GAB-style
relation id matrices and learns additive per-head attention biases for public
state relations and action-to-state pointer relations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from .torch_action_query_merit import _decision, _loss, _masked_fill_invalid, parameter_count
from .torch_public_token_merit import (
    PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    PUBLIC_TOKEN_FEATURE_DIM,
    PublicTokenJsonlDataset,
    _baseline_metrics_from_batches,
    _dataset_summary,
    build_public_token_mlp,
    build_public_token_transformer,
    collate_public_token_roots,
)

RELATION_KINDS = (
    "none",
    "adjacent_hex",
    "terrain_match_adjacent",
    "same_market_slot",
    "same_owner_board",
    "action_uses_tile_slot",
    "action_uses_wildlife_slot",
    "action_targets_tile_frontier",
    "action_targets_wildlife_cell",
)
RELATION_TO_ID = {name: index for index, name in enumerate(RELATION_KINDS)}
RELATION_VOCAB_SIZE = len(RELATION_KINDS)


@dataclass(frozen=True)
class RelationBiasConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = PUBLIC_TOKEN_ACTION_FEATURE_DIM
    relation_vocab_size: int = RELATION_VOCAB_SIZE
    hidden_dim: int = 160
    layers: int = 3
    heads: int = 5
    mlp_dim: int = 320
    dropout: float = 0.0
    model_name: str = "CRT-relation-bias-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coord_key(coord: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not coord:
        return None
    kind = coord.get("kind")
    if kind == "canonical":
        return ("canonical", int(coord["cell_index"]))
    return (
        "overflow",
        int(coord.get("owner_seat", -1)),
        int(coord.get("placement_id", -1)),
        int(coord["q"]),
        int(coord["r"]),
    )


def _set_relation(matrix: list[list[int]], source: int, target: int, relation_id: int, *, overwrite: bool = False) -> None:
    if source < 0 or target < 0 or source >= len(matrix) or target >= len(matrix):
        return
    if source == target:
        return
    if overwrite or matrix[source][target] == 0:
        matrix[source][target] = relation_id


def _token_indexes(root: dict[str, Any]) -> dict[str, dict[Any, int]]:
    by_slot_tile: dict[int, int] = {}
    by_slot_wildlife: dict[int, int] = {}
    active_frontier_by_coord: dict[tuple[Any, ...], int] = {}
    active_tile_by_coord: dict[tuple[Any, ...], int] = {}
    active_seat = int(root["active_seat"])
    for token in root["public_tokens"]["tokens"]:
        index = int(token["token_index"])
        kind = token.get("token_kind")
        if kind == "market_tile":
            by_slot_tile[int(token["market_slot"])] = index
        elif kind == "market_wildlife":
            by_slot_wildlife[int(token["market_slot"])] = index
        elif kind == "frontier" and int(token.get("owner_seat", -1)) == active_seat:
            key = _coord_key(token.get("coord_ref"))
            if key is not None:
                active_frontier_by_coord[key] = index
        elif kind == "placed_tile" and int(token.get("owner_seat", -1)) == active_seat:
            key = _coord_key(token.get("coord_ref"))
            if key is not None:
                active_tile_by_coord[key] = index
    return {
        "market_tile": by_slot_tile,
        "market_wildlife": by_slot_wildlife,
        "active_frontier": active_frontier_by_coord,
        "active_tile": active_tile_by_coord,
    }


def combined_relation_ids_array(
    root: dict[str, Any],
    *,
    action_offset: int | None = None,
    seq_len: int | None = None,
):
    """Vectorized relation-id matrix (numpy int64), semantics-identical to the
    legacy list-of-lists builder. The dense O(seq_len^2) allocation and the
    same-owner block fill dominate matrix cost; both are numpy here. The sparse
    relation/action-pointer passes stay sequential scalar assignments so that
    duplicate-edge overwrite order matches the legacy builder exactly."""
    import numpy as np

    token_count = int(root["public_tokens"]["token_count"])
    action_count = len(root["legal_actions"])
    action_offset = token_count if action_offset is None else action_offset
    seq_len = action_offset + action_count if seq_len is None else seq_len
    matrix = np.zeros((seq_len, seq_len), dtype=np.int64)

    same_board_id = RELATION_TO_ID["same_owner_board"]
    tokens_by_owner: dict[int, list[int]] = {}
    for token in root["public_tokens"]["tokens"]:
        kind = token.get("token_kind")
        owner = token.get("owner_seat")
        if owner is None or kind not in {"player", "placed_tile", "frontier"}:
            continue
        tokens_by_owner.setdefault(int(owner), []).append(int(token["token_index"]))
    for indexes in tokens_by_owner.values():
        in_range = [index for index in indexes if 0 <= index < seq_len]
        if not in_range:
            continue
        block = np.asarray(in_range, dtype=np.intp)
        matrix[np.ix_(block, block)] = same_board_id
        matrix[block, block] = 0

    def _assign(source: int, target: int, relation_id: int, *, overwrite: bool) -> None:
        if source < 0 or target < 0 or source >= seq_len or target >= seq_len:
            return
        if source == target:
            return
        if overwrite or matrix[source, target] == 0:
            matrix[source, target] = relation_id

    for relation in root["public_tokens"].get("relations", []):
        source = int(relation["source"])
        target = int(relation["target"])
        kind = relation.get("relation_kind")
        if kind == "adjacent_hex":
            relation_id = (
                RELATION_TO_ID["terrain_match_adjacent"]
                if relation.get("terrain_matches")
                else RELATION_TO_ID["adjacent_hex"]
            )
            _assign(source, target, relation_id, overwrite=True)
        elif kind == "same_market_slot":
            _assign(source, target, RELATION_TO_ID["same_market_slot"], overwrite=True)

    indexes = _token_indexes(root)
    for action_index, action in enumerate(root["legal_actions"]):
        action_pos = action_offset + action_index
        tile_slot = int(action.get("tile_slot", action.get("draft_slot", -1)))
        wildlife_slot = int(action.get("wildlife_slot", action.get("draft_slot", -1)))
        tile_token = indexes["market_tile"].get(tile_slot)
        wildlife_token = indexes["market_wildlife"].get(wildlife_slot)
        target_frontier = indexes["active_frontier"].get(_coord_key(action.get("target_coord_ref")))
        wildlife_key = _coord_key(action.get("wildlife_coord_ref"))
        wildlife_target = indexes["active_tile"].get(wildlife_key)
        if wildlife_target is None:
            wildlife_target = indexes["active_frontier"].get(wildlife_key)

        for target, relation_name in (
            (tile_token, "action_uses_tile_slot"),
            (wildlife_token, "action_uses_wildlife_slot"),
            (target_frontier, "action_targets_tile_frontier"),
            (wildlife_target, "action_targets_wildlife_cell"),
        ):
            if target is None:
                continue
            relation_id = RELATION_TO_ID[relation_name]
            _assign(action_pos, target, relation_id, overwrite=True)
            _assign(target, action_pos, relation_id, overwrite=True)
    return matrix


def combined_relation_ids(
    root: dict[str, Any],
    *,
    action_offset: int | None = None,
    seq_len: int | None = None,
) -> list[list[int]]:
    return combined_relation_ids_array(
        root, action_offset=action_offset, seq_len=seq_len
    ).tolist()


def relation_counts(matrix) -> dict[str, int]:
    import numpy as np

    array = np.asarray(matrix, dtype=np.int64)
    totals = np.bincount(array.ravel(), minlength=len(RELATION_KINDS))
    return {
        name: int(count)
        for name, count in zip(RELATION_KINDS, totals.tolist())
        if count
    }


def collate_relation_bias_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_public_token_roots(records)
    max_tokens = batch["tokens"].shape[1]
    max_actions = batch["actions"].shape[1]
    seq_len = max_tokens + max_actions
    relation_ids = torch.zeros((len(records), seq_len, seq_len), dtype=torch.long)
    relation_summaries = []
    for batch_index, record in enumerate(records):
        matrix = combined_relation_ids(record, action_offset=max_tokens, seq_len=seq_len)
        relation_ids[batch_index] = torch.tensor(matrix, dtype=torch.long)
        relation_summaries.append(relation_counts(matrix))
    batch["relation_ids"] = relation_ids
    batch["combined_seq_len"] = seq_len
    batch["relation_id_counts"] = relation_summaries
    return batch


def make_relation_bias_loader(path: str | Path, *, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = PublicTokenJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_relation_bias_roots,
    )


def _to_device(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    tensor_keys = {
        "tokens",
        "token_mask",
        "actions",
        "action_mask",
        "relation_ids",
        "target_q",
        "target_z",
        "target_policy",
        "target_q_count",
        "target_q_variance",
        "immediate",
        "action_species",
    }
    return {key: value.to(device) if key in tensor_keys else value for key, value in batch.items()}


def build_relation_bias_transformer(config: RelationBiasConfig):
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
            attn_out, _ = self.attn(
                attn_in,
                attn_in,
                attn_in,
                attn_mask=bias,
                need_weights=False,
            )
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x

    class RelationBiasTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_proj = nn.Linear(config.token_feature_dim, config.hidden_dim)
            self.action_proj = nn.Linear(config.action_feature_dim, config.hidden_dim)
            self.type_embedding = nn.Embedding(2, config.hidden_dim)
            self.layers = nn.ModuleList([RelationBiasedEncoderLayer() for _ in range(config.layers)])
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.q_head = nn.Linear(config.hidden_dim, 1)
            self.policy_head = nn.Linear(config.hidden_dim, 1)

        def forward(self, tokens, token_mask, actions, action_mask, relation_ids):  # type: ignore[no-untyped-def]
            batch_size, token_count, _ = tokens.shape
            action_count = actions.shape[1]
            token_h = self.token_proj(tokens)
            action_h = self.action_proj(actions)
            type_ids = torch.zeros((batch_size, token_count + action_count), dtype=torch.long, device=tokens.device)
            type_ids[:, token_count:] = 1
            hidden = torch.cat([token_h, action_h], dim=1) + self.type_embedding(type_ids)
            padding_mask = torch.cat([~token_mask, ~action_mask], dim=1)
            for layer in self.layers:
                hidden = layer(hidden, padding_mask, relation_ids)
            hidden = self.norm(hidden)
            action_hidden = hidden[:, token_count:]
            return {
                "q": self.q_head(action_hidden).squeeze(-1),
                "logits": self.policy_head(action_hidden).squeeze(-1),
            }

    return RelationBiasTransformer()


def _relation_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(
        batch["tokens"],
        batch["token_mask"],
        batch["actions"],
        batch["action_mask"],
        batch["relation_ids"],
    )["q"]


def _relation_loss(model, batch):  # type: ignore[no-untyped-def]
    return _loss(
        model(
            batch["tokens"],
            batch["token_mask"],
            batch["actions"],
            batch["action_mask"],
            batch["relation_ids"],
        ),
        batch,
    )


def _public_scores(model, batch):  # type: ignore[no-untyped-def]
    return model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])["q"]


def _public_loss(model, batch):  # type: ignore[no-untyped-def]
    return _loss(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
    )


def _label_reliability(batch):  # type: ignore[no-untyped-def]
    import torch

    mask_f = batch["action_mask"].to(batch["target_q"].dtype)
    counts = batch.get("target_q_count")
    variances = batch.get("target_q_variance")
    if counts is None:
        counts = torch.ones_like(batch["target_q"])
    if variances is None:
        variances = torch.zeros_like(batch["target_q"])
    max_count = (counts * mask_f).amax(dim=1, keepdim=True).clamp_min(1.0)
    count_weight = (counts.clamp_min(1.0) / max_count).sqrt()
    root_variance_scale = (
        (variances.clamp_min(0.0) * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    ).clamp_min(1.0)
    variance_weight = 1.0 / (1.0 + variances.clamp_min(0.0) / root_variance_scale)
    return (count_weight * variance_weight).clamp(0.05, 1.0) * mask_f


def _top16_prefilter_loss(
    output: dict[str, Any],
    batch: dict[str, Any],
    *,
    q_weight: float,
    policy_weight: float,
    best_margin_weight: float,
    pairwise_margin: float,
    policy_temperature: float,
):
    import torch

    mask = batch["action_mask"]
    mask_f = mask.to(output["q"].dtype)
    reliability = _label_reliability(batch).to(output["q"].dtype)
    q_loss = (((output["q"] - batch["target_z"]) ** 2) * reliability).sum() / reliability.sum().clamp_min(1.0)

    target_logits = _masked_fill_invalid(batch["target_z"] / max(policy_temperature, 1.0e-6), mask)
    target_policy = torch.softmax(target_logits, dim=1)
    log_probs = torch.log_softmax(_masked_fill_invalid(output["logits"], mask), dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

    target_q = batch["target_q"]
    teacher_best = torch.argmax(_masked_fill_invalid(target_q, mask), dim=1)
    best_index = teacher_best[:, None]
    best_pred = output["q"].gather(1, best_index)
    best_q = target_q.gather(1, best_index)
    best_reliability = reliability.gather(1, best_index)
    pred_gap = best_pred - output["q"]
    q_gap = (best_q - target_q).clamp_min(0.0)
    action_positions = torch.arange(output["q"].shape[1], device=output["q"].device)[None, :]
    pair_mask = mask & (action_positions != best_index)
    pair_mask_f = pair_mask.to(output["q"].dtype)
    q_gap_scale = (q_gap * pair_mask_f).sum(dim=1, keepdim=True) / pair_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    q_gap_scale = q_gap_scale.clamp_min(1.0)
    gap_weight = (q_gap / q_gap_scale).clamp(0.0, 3.0) + 0.25
    pair_reliability = (reliability * best_reliability).sqrt()
    pair_weight = gap_weight * pair_reliability * pair_mask_f
    pairwise_loss = (torch.nn.functional.softplus(pairwise_margin - pred_gap) * pair_weight).sum()
    pairwise_loss = pairwise_loss / pair_weight.sum().clamp_min(1.0)

    return q_weight * q_loss + policy_weight * policy_loss + best_margin_weight * pairwise_loss


def _topk_retention_loss(
    output: dict[str, Any],
    batch: dict[str, Any],
    *,
    retention_k: int,
    q_weight: float,
    policy_weight: float,
    retention_loss_weight: float,
    pairwise_margin: float,
    policy_temperature: float,
):
    import torch

    mask = batch["action_mask"]
    mask_f = mask.to(output["q"].dtype)
    reliability = _label_reliability(batch).to(output["q"].dtype)

    q_loss = (((output["q"] - batch["target_z"]) ** 2) * reliability).sum() / reliability.sum().clamp_min(1.0)

    target_q = batch["target_q"]
    masked_target_q = _masked_fill_invalid(target_q, mask)
    ranked = torch.argsort(masked_target_q, dim=1, descending=True)
    ranks = torch.empty_like(ranked)
    action_positions = torch.arange(ranked.shape[1], device=ranked.device)[None, :].expand_as(ranked)
    ranks.scatter_(1, ranked, action_positions)
    valid_counts = mask.sum(dim=1, keepdim=True)
    effective_k = torch.minimum(
        valid_counts,
        torch.full_like(valid_counts, max(1, retention_k)),
    )
    positive_mask = (ranks < effective_k) & mask
    negative_mask = (ranks >= effective_k) & mask

    positive_f = positive_mask.to(output["q"].dtype)
    positive_count = positive_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_policy = positive_f / positive_count
    log_probs = torch.log_softmax(_masked_fill_invalid(output["logits"], mask), dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

    pos_pred = output["q"][:, :, None]
    neg_pred = output["q"][:, None, :]
    pair_mask = positive_mask[:, :, None] & negative_mask[:, None, :]
    pair_mask_f = pair_mask.to(output["q"].dtype)
    q_gap = (target_q[:, :, None] - target_q[:, None, :]).clamp_min(0.0)
    q_gap_scale = (q_gap * pair_mask_f).sum(dim=(1, 2), keepdim=True)
    q_gap_scale = q_gap_scale / pair_mask_f.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    q_gap_scale = q_gap_scale.clamp_min(1.0)
    gap_weight = (q_gap / q_gap_scale).clamp(0.0, 3.0) + 0.25
    pair_reliability = (reliability[:, :, None] * reliability[:, None, :]).sqrt()
    pair_weight = gap_weight * pair_reliability * pair_mask_f
    retention_loss = torch.nn.functional.softplus(pairwise_margin - (pos_pred - neg_pred)) * pair_weight
    retention_loss = retention_loss.sum() / pair_weight.sum().clamp_min(1.0)

    return q_weight * q_loss + policy_weight * policy_loss + retention_loss_weight * retention_loss


def _loss_with_mode(output: dict[str, Any], batch: dict[str, Any], args: argparse.Namespace):
    if args.loss_mode == "standard":
        return _loss(output, batch)
    if args.loss_mode == "top16-prefilter":
        return _top16_prefilter_loss(
            output,
            batch,
            q_weight=args.q_loss_weight,
            policy_weight=args.policy_loss_weight,
            best_margin_weight=args.best_margin_loss_weight,
            pairwise_margin=args.pairwise_margin,
            policy_temperature=args.policy_temperature,
        )
    if args.loss_mode == "topk-retention":
        return _topk_retention_loss(
            output,
            batch,
            retention_k=args.retention_k,
            q_weight=args.q_loss_weight,
            policy_weight=args.policy_loss_weight,
            retention_loss_weight=args.retention_loss_weight,
            pairwise_margin=args.pairwise_margin,
            policy_temperature=args.policy_temperature,
        )
    raise ValueError(f"unsupported loss mode: {args.loss_mode}")


def _relation_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(
            batch["tokens"],
            batch["token_mask"],
            batch["actions"],
            batch["action_mask"],
            batch["relation_ids"],
        ),
        batch,
        args,
    )


def _public_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
        args,
    )


def _all_batches(path: Path, *, batch_size: int) -> list[dict[str, Any]]:
    return list(make_relation_bias_loader(path, batch_size=batch_size, shuffle=False))


def _evaluate_relation_scores(batches: list[dict[str, Any]], score_fn, device=None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
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


def _model_metrics(model, batches: list[dict[str, Any]], device, score_fn) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    model.eval()
    return _evaluate_relation_scores(
        batches,
        lambda batch: score_fn(model, batch),
        device=device,
    )


def _train_model(model, train_path: Path, *, args: argparse.Namespace, device, loss_fn):  # type: ignore[no-untyped-def]
    import torch

    loader = make_relation_bias_loader(train_path, batch_size=args.batch_size, shuffle=True)
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


def _relation_dataset_summary(path: Path) -> dict[str, Any]:
    summary = _dataset_summary(path)
    records = PublicTokenJsonlDataset(path).records
    counts = {name: 0 for name in RELATION_KINDS}
    label_counts = []
    label_variances = []
    truncated_counts = []
    for record in records:
        matrix = combined_relation_ids(record)
        for name, count in relation_counts(matrix).items():
            counts[name] += count
        label_counts.extend(record.get("per_action_Q_count", []))
        label_variances.extend(record.get("per_action_Q_variance", []))
        truncated_counts.extend(record.get("per_action_truncated_count", []))
    summary["relation_id_totals"] = {name: count for name, count in counts.items() if count}
    if label_counts:
        summary["label_counts"] = {
            "min": min(label_counts),
            "max": max(label_counts),
            "mean": sum(label_counts) / len(label_counts),
            "total": sum(label_counts),
        }
    if label_variances:
        summary["label_variance"] = {
            "min": min(label_variances),
            "max": max(label_variances),
            "mean": sum(label_variances) / len(label_variances),
        }
    if truncated_counts:
        truncated_total = sum(truncated_counts)
        summary["truncated_rollouts"] = {
            "total": truncated_total,
            "action_labels_with_truncation": sum(1 for count in truncated_counts if count),
            "mean_per_action": truncated_total / len(truncated_counts),
            "sample_rate": truncated_total / sum(label_counts) if label_counts else 0.0,
        }
    return summary


def _decision_with_vanilla(
    relation_bias: dict[str, Any],
    vanilla: dict[str, Any],
    mlp: dict[str, Any],
    immediate: dict[str, Any],
) -> dict[str, Any]:
    decision = _decision(relation_bias, mlp, immediate)
    relation_regret = float(relation_bias["mean_regret"])
    vanilla_regret = float(vanilla["mean_regret"])
    relation_top1 = float(relation_bias["top1_agreement"])
    vanilla_top1 = float(vanilla["top1_agreement"])
    decision.update(
        {
            "beats_vanilla_transformer": relation_regret <= vanilla_regret and relation_top1 >= vanilla_top1 - 0.02,
            "regret_improvement_vs_vanilla": (
                (vanilla_regret - relation_regret) / vanilla_regret if vanilla_regret else 0.0
            ),
            "top1_gain_vs_vanilla": relation_top1 - vanilla_top1,
        }
    )
    decision["has_merit"] = bool(decision["has_merit"] and decision["beats_vanilla_transformer"])
    decision["criteria"] = (
        decision["criteria"]
        + "; relation-bias also must nonregress versus the same-run vanilla public-token transformer"
    )
    return decision


def run_relation_bias_pilot(
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
    loss_mode: str = "standard",
    q_loss_weight: float = 0.25,
    policy_loss_weight: float = 0.5,
    best_margin_loss_weight: float = 1.0,
    retention_loss_weight: float = 1.0,
    retention_k: int = 16,
    pairwise_margin: float = 0.25,
    policy_temperature: float = 0.5,
    experiment_id: str = "crt-relation-bias-query-merit-v1",
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
    config = RelationBiasConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
    if config.hidden_dim % config.heads != 0:
        raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by heads {config.heads}")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    train_summary = _relation_dataset_summary(train_path)
    val_summary = _relation_dataset_summary(val_path)
    val_batches = _all_batches(val_path, batch_size=batch_size)

    relation_model = build_relation_bias_transformer(config)
    relation_model, relation_optimizer, relation_losses = _train_model(
        relation_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _relation_loss_with_mode(model, batch, args),
    )
    vanilla_model = build_public_token_transformer(config)
    vanilla_model, vanilla_optimizer, vanilla_losses = _train_model(
        vanilla_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
    )
    mlp_model = build_public_token_mlp(config)
    mlp_model, mlp_optimizer, mlp_losses = _train_model(
        mlp_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _public_loss_with_mode(model, batch, args),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    relation_metrics = _model_metrics(relation_model, val_batches, device, _relation_scores)
    vanilla_metrics = _model_metrics(vanilla_model, val_batches, device, _public_scores)
    mlp_metrics = _model_metrics(mlp_model, val_batches, device, _public_scores)
    immediate_metrics = _baseline_metrics_from_batches(val_batches, "immediate")
    decision = _decision_with_vanilla(relation_metrics, vanilla_metrics, mlp_metrics, immediate_metrics)

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
            "relation_bias_transformer": {
                "parameter_count": parameter_count(relation_model),
                "loss_head": relation_losses[:5],
                "loss_tail": relation_losses[-5:],
                "metrics": relation_metrics,
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
        "relation_bias_model": relation_model,
        "relation_bias_optimizer": relation_optimizer,
        "vanilla_model": vanilla_model,
        "vanilla_optimizer": vanilla_optimizer,
        "mlp_model": mlp_model,
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
    parser.add_argument("--loss-mode", choices=["standard", "top16-prefilter", "topk-retention"], default="standard")
    parser.add_argument("--q-loss-weight", type=float, default=0.25)
    parser.add_argument("--policy-loss-weight", type=float, default=0.5)
    parser.add_argument("--best-margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-loss-weight", type=float, default=1.0)
    parser.add_argument("--retention-k", type=int, default=16)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--experiment-id", default="crt-relation-bias-query-merit-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_relation_bias_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_relation_bias_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_relation_bias_pilot(
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
    relation_model = result.pop("relation_bias_model")
    relation_optimizer = result.pop("relation_bias_optimizer")
    vanilla_model = result.pop("vanilla_model")
    vanilla_optimizer = result.pop("vanilla_optimizer")
    mlp_model = result.pop("mlp_model")
    mlp_optimizer = result.pop("mlp_optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "relation_bias_state_dict": relation_model.state_dict(),
            "relation_bias_optimizer_state_dict": relation_optimizer.state_dict(),
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
