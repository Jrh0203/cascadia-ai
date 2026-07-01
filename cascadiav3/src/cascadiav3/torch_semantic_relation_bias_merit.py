"""Semantic action-conditioned relation-bias CRT merit pilot.

The first relation-bias model learned attention structure, but its action token
only exposed raw draft/tile/species fields. This module keeps the same public
state tokens and relation-bias encoder, then appends action-conditioned pattern
features computed from the exported public tokens. The features are still
derived entirely from public state and legal action JSON, so old replay shards
can be reused without changing the exporter schema.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

from .torch_action_query_merit import _normalizer, _safe_float, parameter_count
from .torch_public_token_merit import (
    PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    PUBLIC_TOKEN_FEATURE_DIM,
    PublicTokenJsonlDataset,
    _baseline_metrics_from_batches,
    build_public_token_mlp,
    build_public_token_transformer,
    collate_public_token_roots,
    public_token_action_features,
)
from .torch_relation_bias_merit import (
    RELATION_VOCAB_SIZE,
    _decision_with_vanilla,
    _evaluate_relation_scores,
    _loss_with_mode,
    _public_scores,
    _relation_dataset_summary,
    _relation_scores,
    _to_device,
    build_relation_bias_transformer,
    combined_relation_ids,
    relation_counts,
)

DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
AXIS_DIRECTIONS = DIRECTIONS[:3]
WILDLIFE_COUNT = 5

SEMANTIC_ACTION_FEATURE_NAMES = (
    "target_neighbor_count",
    "target_habitat_match_count",
    "target_habitat_mismatch_count",
    "target_open_edge_count",
    "tile_supports_drafted_species",
    "tile_wildlife_option_count",
    "active_species_count",
    "active_empty_species_slot_count",
    "public_market_species_count",
    "opponent_max_species_count",
    "opponent_species_count_gap",
    "wildlife_adjacent_same_species_count",
    "wildlife_adjacent_any_wildlife_count",
    "wildlife_adjacent_unique_other_species_count",
    "bear_pair_signal",
    "bear_overcluster_signal",
    "elk_best_line_length",
    "elk_adjacent_elk_count",
    "salmon_component_size",
    "salmon_local_degree",
    "salmon_branch_risk",
    "hawk_isolated_signal",
    "hawk_line_of_sight_count",
    "hawk_adjacent_penalty",
    "fox_unique_adjacent_species_count",
    "fox_nonfox_neighbor_count",
    "wildlife_bag_species_count",
    "unseen_tile_species_capacity",
)
SEMANTIC_ACTION_FEATURE_DIM = len(SEMANTIC_ACTION_FEATURE_NAMES)
SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM = PUBLIC_TOKEN_ACTION_FEATURE_DIM + SEMANTIC_ACTION_FEATURE_DIM


@dataclass(frozen=True)
class SemanticRelationBiasConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    relation_vocab_size: int = RELATION_VOCAB_SIZE
    hidden_dim: int = 160
    layers: int = 3
    heads: int = 5
    mlp_dim: int = 320
    dropout: float = 0.0
    model_name: str = "CRT-semantic-relation-bias-query-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coord_key(coord: dict[str, Any] | None) -> tuple[int, int] | None:
    if not coord:
        return None
    try:
        return (int(coord["q"]), int(coord["r"]))
    except (KeyError, TypeError, ValueError):
        return None


def _neighbors(coord: tuple[int, int]) -> list[tuple[int, int]]:
    q, r = coord
    return [(q + dq, r + dr) for dq, dr in DIRECTIONS]


def _species_from_action(action: dict[str, Any]) -> int:
    return int(_safe_float(action.get("wildlife_species"), -1.0))


def _wildlife_mask_contains(mask: Any, species: int) -> bool:
    if species < 0 or species >= WILDLIFE_COUNT:
        return False
    return (int(_safe_float(mask)) & (1 << species)) != 0


def _tile_terrain_on_edge(tile: dict[str, Any], edge: int) -> int:
    terrain_a = int(_safe_float(tile.get("terrain_a", tile.get("tile_terrain_a")), -1.0))
    terrain_b = int(_safe_float(tile.get("terrain_b", tile.get("tile_terrain_b")), -1.0))
    if terrain_b < 0:
        return terrain_a
    rotation = int(_safe_float(tile.get("rotation"), 0.0)) % 6
    offset = (edge + 6 - rotation) % 6
    return terrain_a if offset < 3 else terrain_b


def _state_view(root: dict[str, Any]) -> dict[str, Any]:
    active_seat = int(root["active_seat"])
    placed_by_owner: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    wildlife_by_owner: dict[int, dict[int, set[tuple[int, int]]]] = {}
    empty_species_slots: dict[int, int] = {species: 0 for species in range(WILDLIFE_COUNT)}
    market_species_counts: dict[int, int] = {species: 0 for species in range(WILDLIFE_COUNT)}
    supply_bag = [0.0] * WILDLIFE_COUNT
    supply_capacity = [0.0] * WILDLIFE_COUNT

    for token in root["public_tokens"]["tokens"]:
        kind = token.get("token_kind")
        if kind == "placed_tile":
            owner = int(token.get("owner_seat", -1))
            coord = _coord_key(token.get("coord_ref"))
            if coord is None:
                continue
            placed_by_owner.setdefault(owner, {})[coord] = token
            wildlife = int(_safe_float(token.get("placed_wildlife"), -1.0))
            if 0 <= wildlife < WILDLIFE_COUNT:
                wildlife_by_owner.setdefault(owner, {}).setdefault(wildlife, set()).add(coord)
            elif owner == active_seat:
                mask = int(_safe_float(token.get("wildlife_mask"), 0.0))
                for species in range(WILDLIFE_COUNT):
                    if _wildlife_mask_contains(mask, species):
                        empty_species_slots[species] += 1
        elif kind == "market_wildlife":
            species = int(_safe_float(token.get("species"), -1.0))
            if 0 <= species < WILDLIFE_COUNT:
                market_species_counts[species] += 1
        elif kind == "public_supply":
            bag = token.get("wildlife_bag") or []
            capacity = token.get("unseen_tile_wildlife_capacity") or []
            for species in range(min(WILDLIFE_COUNT, len(bag))):
                supply_bag[species] = _safe_float(bag[species])
            for species in range(min(WILDLIFE_COUNT, len(capacity))):
                supply_capacity[species] = _safe_float(capacity[species])

    return {
        "active_seat": active_seat,
        "placed_by_owner": placed_by_owner,
        "active_tiles": placed_by_owner.get(active_seat, {}),
        "wildlife_by_owner": wildlife_by_owner,
        "active_wildlife": wildlife_by_owner.get(active_seat, {}),
        "empty_species_slots": empty_species_slots,
        "market_species_counts": market_species_counts,
        "supply_bag": supply_bag,
        "supply_capacity": supply_capacity,
    }


def _line_length_through(coord: tuple[int, int], positions: set[tuple[int, int]], direction: tuple[int, int]) -> int:
    dq, dr = direction
    length = 1
    current = (coord[0] + dq, coord[1] + dr)
    while current in positions:
        length += 1
        current = (current[0] + dq, current[1] + dr)
    current = (coord[0] - dq, coord[1] - dr)
    while current in positions:
        length += 1
        current = (current[0] - dq, current[1] - dr)
    return length


def _component_size(coord: tuple[int, int], positions: set[tuple[int, int]]) -> int:
    if coord not in positions:
        return 0
    seen = {coord}
    stack = [coord]
    while stack:
        current = stack.pop()
        for neighbor in _neighbors(current):
            if neighbor in positions and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen)


def _hawk_line_of_sight_count(coord: tuple[int, int], hawks: set[tuple[int, int]]) -> int:
    count = 0
    for dq, dr in DIRECTIONS:
        current = (coord[0] + dq, coord[1] + dr)
        distance = 1
        while distance <= 16:
            if current in hawks:
                count += int(distance > 1)
                break
            current = (current[0] + dq, current[1] + dr)
            distance += 1
    return count


def _habitat_edge_counts(action: dict[str, Any], state: dict[str, Any]) -> tuple[int, int, int]:
    target = _coord_key(action.get("target_coord_ref"))
    if target is None:
        return (0, 0, 6)
    active_tiles = state["active_tiles"]
    matches = 0
    mismatches = 0
    for edge, neighbor in enumerate(_neighbors(target)):
        neighbor_tile = active_tiles.get(neighbor)
        if neighbor_tile is None:
            continue
        action_terrain = _tile_terrain_on_edge(
            {
                "terrain_a": action.get("tile_terrain_a"),
                "terrain_b": action.get("tile_terrain_b"),
                "rotation": action.get("rotation"),
            },
            edge,
        )
        neighbor_terrain = _tile_terrain_on_edge(neighbor_tile, (edge + 3) % 6)
        if action_terrain == neighbor_terrain:
            matches += 1
        else:
            mismatches += 1
    open_edges = 6 - matches - mismatches
    return (matches, mismatches, open_edges)


def semantic_action_features(root: dict[str, Any]) -> list[list[float]]:
    state = _state_view(root)
    active_seat = state["active_seat"]
    active_wildlife: dict[int, set[tuple[int, int]]] = state["active_wildlife"]
    all_active_wildlife = {
        coord: species
        for species, coords in active_wildlife.items()
        for coord in coords
    }
    rows = []
    for action in root["legal_actions"]:
        species = _species_from_action(action)
        target = _coord_key(action.get("target_coord_ref"))
        wildlife_coord = _coord_key(action.get("wildlife_coord_ref"))
        wildlife_present = bool(action.get("wildlife_placement_present")) and wildlife_coord is not None
        matches, mismatches, open_edges = _habitat_edge_counts(action, state)
        target_neighbor_count = matches + mismatches

        active_species_before = len(active_wildlife.get(species, set())) if 0 <= species < WILDLIFE_COUNT else 0
        opponent_max = 0
        for owner, by_species in state["wildlife_by_owner"].items():
            if owner == active_seat:
                continue
            opponent_max = max(opponent_max, len(by_species.get(species, set())))

        after_species_positions = {key: set(value) for key, value in active_wildlife.items()}
        if 0 <= species < WILDLIFE_COUNT and wildlife_present:
            after_species_positions.setdefault(species, set()).add(wildlife_coord)

        after_all_wildlife = dict(all_active_wildlife)
        if 0 <= species < WILDLIFE_COUNT and wildlife_present:
            after_all_wildlife[wildlife_coord] = species

        same_neighbors = 0
        any_neighbors = 0
        other_species = set()
        if wildlife_coord is not None:
            for neighbor in _neighbors(wildlife_coord):
                neighbor_species = after_all_wildlife.get(neighbor)
                if neighbor_species is None:
                    continue
                any_neighbors += 1
                if neighbor_species == species:
                    same_neighbors += 1
                else:
                    other_species.add(neighbor_species)

        bear_pair_signal = float(species == 0 and same_neighbors == 1)
        bear_overcluster_signal = float(species == 0 and same_neighbors > 1)

        elk_positions = after_species_positions.get(1, set())
        elk_line_length = 0
        if species == 1 and wildlife_coord is not None and wildlife_coord in elk_positions:
            elk_line_length = max(_line_length_through(wildlife_coord, elk_positions, direction) for direction in AXIS_DIRECTIONS)

        salmon_positions = after_species_positions.get(2, set())
        salmon_component_size = 0
        salmon_degree = 0
        salmon_branch_risk = 0
        if species == 2 and wildlife_coord is not None and wildlife_coord in salmon_positions:
            salmon_component_size = _component_size(wildlife_coord, salmon_positions)
            salmon_degree = sum(1 for neighbor in _neighbors(wildlife_coord) if neighbor in salmon_positions)
            neighbor_degrees = [
                sum(1 for second in _neighbors(neighbor) if second in salmon_positions)
                for neighbor in _neighbors(wildlife_coord)
                if neighbor in salmon_positions
            ]
            salmon_branch_risk = int(salmon_degree > 2 or any(degree > 2 for degree in neighbor_degrees))

        hawks = after_species_positions.get(3, set())
        hawk_isolated = float(species == 3 and same_neighbors == 0)
        hawk_los = _hawk_line_of_sight_count(wildlife_coord, hawks) if species == 3 and wildlife_coord is not None else 0
        hawk_adjacent_penalty = float(species == 3 and same_neighbors > 0)

        fox_unique = 0
        fox_nonfox = 0
        if species == 4 and wildlife_coord is not None:
            adjacent_species = [
                after_all_wildlife[neighbor]
                for neighbor in _neighbors(wildlife_coord)
                if neighbor in after_all_wildlife
            ]
            fox_unique = len(set(adjacent_species))
            fox_nonfox = sum(1 for neighbor_species in adjacent_species if neighbor_species != 4)

        semantic = [
            _normalizer(float(target_neighbor_count), 6.0),
            _normalizer(float(matches), 6.0),
            _normalizer(float(mismatches), 6.0),
            _normalizer(float(open_edges), 6.0),
            float(_wildlife_mask_contains(action.get("tile_wildlife_mask"), species)),
            _normalizer(float(int(_safe_float(action.get("tile_wildlife_mask"), 0.0)).bit_count()), 5.0),
            _normalizer(float(active_species_before), 20.0),
            _normalizer(float(state["empty_species_slots"].get(species, 0)), 20.0),
            _normalizer(float(state["market_species_counts"].get(species, 0)), 4.0),
            _normalizer(float(opponent_max), 20.0),
            _normalizer(float(opponent_max - active_species_before), 20.0),
            _normalizer(float(same_neighbors), 6.0),
            _normalizer(float(any_neighbors), 6.0),
            _normalizer(float(len(other_species)), 4.0),
            bear_pair_signal,
            bear_overcluster_signal,
            _normalizer(float(min(elk_line_length, 4)), 4.0),
            _normalizer(float(same_neighbors if species == 1 else 0), 6.0),
            _normalizer(float(min(salmon_component_size, 7)), 7.0),
            _normalizer(float(min(salmon_degree, 3)), 3.0),
            float(salmon_branch_risk),
            hawk_isolated,
            _normalizer(float(min(hawk_los, 6)), 6.0),
            hawk_adjacent_penalty,
            _normalizer(float(fox_unique), 5.0),
            _normalizer(float(fox_nonfox), 6.0),
            _normalizer(float(state["supply_bag"][species] if 0 <= species < WILDLIFE_COUNT else 0.0), 100.0),
            _normalizer(float(state["supply_capacity"][species] if 0 <= species < WILDLIFE_COUNT else 0.0), 100.0),
        ]
        if len(semantic) != SEMANTIC_ACTION_FEATURE_DIM:
            raise ValueError(f"semantic feature dimension mismatch: {len(semantic)}")
        rows.append(semantic)
    return rows


def semantic_public_token_action_features(root: dict[str, Any]) -> list[list[float]]:
    base_rows = public_token_action_features(root)
    semantic_rows = semantic_action_features(root)
    rows = []
    for base, semantic in zip(base_rows, semantic_rows, strict=True):
        row = list(base) + semantic
        if len(row) != SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM:
            raise ValueError(f"semantic public-token action feature dimension mismatch: {len(row)}")
        rows.append(row)
    return rows


def collate_semantic_relation_bias_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_public_token_roots(
        records,
        action_feature_fn=semantic_public_token_action_features,
        action_feature_dim=SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    )
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


def make_semantic_relation_bias_loader(path: str | Path, *, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = PublicTokenJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_semantic_relation_bias_roots,
    )


def semantic_all_batches(path: Path, *, batch_size: int) -> list[dict[str, Any]]:
    return list(make_semantic_relation_bias_loader(path, batch_size=batch_size, shuffle=False))


def semantic_dataset_summary(path: Path) -> dict[str, Any]:
    summary = _relation_dataset_summary(path)
    records = PublicTokenJsonlDataset(path).records
    sums = [0.0] * SEMANTIC_ACTION_FEATURE_DIM
    abs_sums = [0.0] * SEMANTIC_ACTION_FEATURE_DIM
    count = 0
    for record in records:
        for row in semantic_action_features(record):
            for index, value in enumerate(row):
                sums[index] += value
                abs_sums[index] += abs(value)
            count += 1
    summary["semantic_action_features"] = {
        "base_action_feature_dim": PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "semantic_feature_dim": SEMANTIC_ACTION_FEATURE_DIM,
        "combined_action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "feature_names": list(SEMANTIC_ACTION_FEATURE_NAMES),
        "means": {
            name: sums[index] / count if count else 0.0
            for index, name in enumerate(SEMANTIC_ACTION_FEATURE_NAMES)
        },
        "mean_abs": {
            name: abs_sums[index] / count if count else 0.0
            for index, name in enumerate(SEMANTIC_ACTION_FEATURE_NAMES)
        },
    }
    return summary


def _semantic_relation_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
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


def _semantic_public_loss_with_mode(model, batch, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return _loss_with_mode(
        model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]),
        batch,
        args,
    )


def _train_semantic_model(model, train_path: Path, *, args: argparse.Namespace, device, loss_fn):  # type: ignore[no-untyped-def]
    import torch

    loader = make_semantic_relation_bias_loader(train_path, batch_size=args.batch_size, shuffle=True)
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


def _model_metrics(model, batches: list[dict[str, Any]], device, score_fn) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    model.eval()
    return _evaluate_relation_scores(
        batches,
        lambda batch: score_fn(model, batch),
        device=device,
    )


def run_semantic_relation_bias_pilot(
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
    experiment_id: str = "crt-semantic-relation-bias-query-merit-v1",
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
    config = SemanticRelationBiasConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
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

    relation_model = build_relation_bias_transformer(config)
    relation_model, relation_optimizer, relation_losses = _train_semantic_model(
        relation_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _semantic_relation_loss_with_mode(model, batch, args),
    )
    vanilla_model = build_public_token_transformer(config)
    vanilla_model, vanilla_optimizer, vanilla_losses = _train_semantic_model(
        vanilla_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _semantic_public_loss_with_mode(model, batch, args),
    )
    mlp_model = build_public_token_mlp(config)
    mlp_model, mlp_optimizer, mlp_losses = _train_semantic_model(
        mlp_model,
        train_path,
        args=args,
        device=device,
        loss_fn=lambda model, batch: _semantic_public_loss_with_mode(model, batch, args),
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
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16x2_sampled_teacher_val.jsonl")
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
    parser.add_argument("--experiment-id", default="crt-wide32-r16x2-semantic-relation-bias-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16x2_semantic_relation_bias_pilot.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16x2_semantic_relation_bias_pilot.pt")
    args = parser.parse_args()

    import torch

    result = run_semantic_relation_bias_pilot(
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
