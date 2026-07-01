"""Torch Dataset/DataLoader support for Cascadia v3 JSONL replay shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl
from .torch_features import (
    ACTION_FEATURE_DIM,
    SCORE_CATEGORIES,
    STATE_FEATURE_DIM,
    action_features,
    state_features,
    target_score_decomposition,
)


class SearchRootJsonlDataset:
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


def collate_search_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not records:
        raise ValueError("collate_search_roots requires at least one record")

    batch_size = len(records)
    action_counts = [len(record["legal_actions"]) for record in records]
    max_actions = max(action_counts)
    state = torch.zeros((batch_size, STATE_FEATURE_DIM), dtype=torch.float32)
    actions = torch.zeros((batch_size, max_actions, ACTION_FEATURE_DIM), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    target_q = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_q_valid = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    target_score_to_go = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_value = torch.zeros((batch_size, 4), dtype=torch.float32)
    target_rank = torch.zeros((batch_size, 4), dtype=torch.float32)
    target_score = torch.zeros((batch_size, len(SCORE_CATEGORIES), 4), dtype=torch.float32)

    for batch_index, record in enumerate(records):
        count = action_counts[batch_index]
        state[batch_index] = torch.tensor(state_features(record), dtype=torch.float32)
        actions[batch_index, :count] = torch.tensor(action_features(record), dtype=torch.float32)
        action_mask[batch_index, :count] = True
        target_q[batch_index, :count] = torch.tensor(record["per_action_Q"], dtype=torch.float32)
        q_valid = record.get("per_action_Q_valid", [True] * count)
        target_q_valid[batch_index, :count] = torch.tensor(q_valid, dtype=torch.bool)
        target_score_to_go[batch_index, :count] = torch.tensor(
            record.get("per_action_score_to_go", record["per_action_Q"]),
            dtype=torch.float32,
        )
        target_value[batch_index] = torch.tensor(record["final_score_vector"], dtype=torch.float32)
        target_rank[batch_index] = torch.tensor(record["rank_vector"], dtype=torch.float32)
        target_score[batch_index] = torch.tensor(
            target_score_decomposition(record),
            dtype=torch.float32,
        )

    return {
        "state": state,
        "actions": actions,
        "action_mask": action_mask,
        "target_q": target_q,
        "target_q_valid": target_q_valid,
        "target_score_to_go": target_score_to_go,
        "target_value": target_value,
        "target_rank": target_rank,
        "target_score": target_score,
        "action_counts": action_counts,
        "state_hashes": [record["state_hash"] for record in records],
        "schema_ids": [record["schema_id"] for record in records],
    }


def make_replay_loader(path: str | Path, *, batch_size: int = 2, shuffle: bool = False):
    from torch.utils.data import DataLoader

    dataset = SearchRootJsonlDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_search_roots,
    )
