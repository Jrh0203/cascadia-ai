"""Greedy no-search behavior-cloning pretraining for v3 Transformers.

This is not expert iteration. It is a large, cheap representation pretraining
stage: generate complete games from the canonical greedy ranker, train a
semantic public-token Transformer to imitate the selected greedy action, then
use that checkpoint as an initialization or diagnostic baseline before
search-improved targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .greedy_tensor_shards import DEFAULT_SOURCE, load_tensor_shard_arrays, tensor_shard_record_count
from .torch_action_query_merit import parameter_count
from .torch_public_token_merit import (
    PUBLIC_TOKEN_FEATURE_DIM,
    build_public_token_transformer,
)
from .torch_relation_bias_merit import _to_device
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    collate_semantic_relation_bias_roots,
)


@dataclass(frozen=True)
class GreedyPolicyPretrainConfig:
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0
    model_name: str = "CRT-greedy-policy-pretrain-vanilla-public-token-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_paths(raw: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one corpus path is required")
    return paths


def iter_jsonl_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


class GreedyPolicyIterableDataset:
    def __init__(
        self,
        paths: list[Path],
        *,
        shuffle_buffer: int,
        seed: int,
        require_source: str = DEFAULT_SOURCE,
    ) -> None:
        self.paths = paths
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.seed = seed
        self.require_source = require_source

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed)
        buffer: list[dict[str, Any]] = []
        for record in iter_jsonl_records(self.paths):
            if record.get("metadata", {}).get("source") != self.require_source:
                continue
            if self.shuffle_buffer <= 1:
                yield record
                continue
            if len(buffer) < self.shuffle_buffer:
                buffer.append(record)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
        rng.shuffle(buffer)
        yield from buffer


class GreedyTensorShardIterableDataset:
    def __init__(
        self,
        paths: list[Path],
        *,
        shuffle_buffer: int,
        seed: int,
    ) -> None:
        self.paths = paths
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.seed = seed

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = random.Random(self.seed)
        paths = list(self.paths)
        if self.shuffle_buffer > 1:
            rng.shuffle(paths)
        for path in paths:
            shard = load_tensor_shard_arrays(path)
            tokens = shard["tokens"]
            actions = shard["actions"]
            token_offsets = shard["token_offsets"]
            action_offsets = shard["action_offsets"]
            selected = shard["selected_action_index"]
            indices = list(range(int(selected.shape[0])))
            if self.shuffle_buffer > 1:
                rng.shuffle(indices)
            for index in indices:
                token_start = int(token_offsets[index])
                token_end = int(token_offsets[index + 1])
                action_start = int(action_offsets[index])
                action_end = int(action_offsets[index + 1])
                yield {
                    "tokens": tokens[token_start:token_end],
                    "actions": actions[action_start:action_end],
                    "selected_action_index": int(selected[index]),
                }


def selected_action_indices(records: list[dict[str, Any]]) -> list[int]:
    indices = []
    for record in records:
        selected = record["selected_action"]
        action_ids = [action["action_id"] for action in record["legal_actions"]]
        try:
            indices.append(action_ids.index(selected))
        except ValueError as exc:
            raise ValueError(f"selected action missing from legal actions for {record['state_hash']}") from exc
    return indices


def collate_greedy_policy_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = collate_semantic_relation_bias_roots(records)
    batch["selected_action_index"] = torch.tensor(selected_action_indices(records), dtype=torch.long)
    batch["selected_action_ids"] = [record["selected_action"] for record in records]
    return batch


def collate_greedy_tensor_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("collate_greedy_tensor_examples requires at least one example")
    batch_size = len(examples)
    token_counts = [int(example["tokens"].shape[0]) for example in examples]
    action_counts = [int(example["actions"].shape[0]) for example in examples]
    token_dim = int(examples[0]["tokens"].shape[1])
    action_dim = int(examples[0]["actions"].shape[1])
    max_tokens = max(token_counts)
    max_actions = max(action_counts)
    tokens = torch.zeros((batch_size, max_tokens, token_dim), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    actions = torch.zeros((batch_size, max_actions, action_dim), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    selected = torch.zeros((batch_size,), dtype=torch.long)
    for batch_index, example in enumerate(examples):
        token_count = token_counts[batch_index]
        action_count = action_counts[batch_index]
        tokens[batch_index, :token_count] = torch.as_tensor(example["tokens"], dtype=torch.float32)
        token_mask[batch_index, :token_count] = True
        actions[batch_index, :action_count] = torch.as_tensor(example["actions"], dtype=torch.float32)
        action_mask[batch_index, :action_count] = True
        selected[batch_index] = int(example["selected_action_index"])
    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "action_mask": action_mask,
        "selected_action_index": selected,
        "token_counts": token_counts,
        "action_counts": action_counts,
    }


def make_greedy_policy_loader(
    paths: list[Path],
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    corpus_format: str = "jsonl",
):
    from torch.utils.data import DataLoader, IterableDataset

    class TorchGreedyPolicyIterableDataset(IterableDataset):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield from GreedyPolicyIterableDataset(
                paths,
                shuffle_buffer=shuffle_buffer,
                seed=seed,
            )

    class TorchGreedyTensorShardIterableDataset(IterableDataset):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield from GreedyTensorShardIterableDataset(
                paths,
                shuffle_buffer=shuffle_buffer,
                seed=seed,
            )

    if corpus_format == "jsonl":
        dataset = TorchGreedyPolicyIterableDataset()
        collate_fn = collate_greedy_policy_roots
    elif corpus_format == "npz":
        dataset = TorchGreedyTensorShardIterableDataset()
        collate_fn = collate_greedy_tensor_examples
    else:
        raise ValueError(f"unsupported corpus format {corpus_format!r}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


def _policy_loss(outputs: dict[str, Any], batch: dict[str, Any]):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    logits = outputs["logits"].masked_fill(~batch["action_mask"], -1.0e9)
    return F.cross_entropy(logits, batch["selected_action_index"])


def _move_greedy_batch(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    moved = _to_device(batch, device)
    moved["selected_action_index"] = batch["selected_action_index"].to(device)
    return moved


def _slice_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    sliced = {}
    for key, value in batch.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] >= count:
            sliced[key] = value[:count]
        elif isinstance(value, list) and len(value) >= count:
            sliced[key] = value[:count]
        else:
            sliced[key] = value
    return sliced


def _policy_metrics_from_loader(model, loader, *, max_records: int, device):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    total = 0
    top1 = 0
    top4 = 0
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            remaining = max_records - total
            if remaining <= 0:
                break
            if batch["selected_action_index"].shape[0] > remaining:
                batch = _slice_batch(batch, remaining)
            batch = _move_greedy_batch(batch, device)
            outputs = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
            logits = outputs["logits"].masked_fill(~batch["action_mask"], -1.0e9)
            losses.append(float(F.cross_entropy(logits, batch["selected_action_index"]).detach().cpu()))
            selected = batch["selected_action_index"]
            k = min(4, logits.shape[1])
            ranked = torch.topk(logits, k=k, dim=1).indices
            top1 += int((ranked[:, 0] == selected).sum().item())
            top4 += int((ranked == selected.unsqueeze(1)).any(dim=1).sum().item())
            total += int(selected.shape[0])
    if total == 0:
        raise ValueError("no greedy policy records available for evaluation")
    return {
        "records": total,
        "cross_entropy": sum(losses) / len(losses),
        "top1_accuracy": top1 / total,
        "top4_accuracy": top4 / total,
    }


def count_records(paths: list[Path], *, corpus_format: str) -> int:
    if corpus_format == "npz":
        return sum(tensor_shard_record_count(path) for path in paths)
    if corpus_format != "jsonl":
        raise ValueError(f"unsupported corpus format {corpus_format!r}")
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    return count


def run_greedy_policy_pretrain(
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    train_format: str,
    val_format: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    hidden_dim: int,
    layers: int,
    heads: int,
    mlp_dim: int,
    grad_clip: float,
    shuffle_buffer: int,
    max_val_records: int,
    experiment_id: str,
) -> dict[str, Any]:
    import torch

    config = GreedyPolicyPretrainConfig(hidden_dim=hidden_dim, layers=layers, heads=heads, mlp_dim=mlp_dim)
    if config.hidden_dim % config.heads != 0:
        raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by heads {config.heads}")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = build_public_token_transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_greedy_policy_loader(
        train_paths,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        corpus_format=train_format,
    )
    losses = []
    model.train()
    loader_iter = iter(loader)
    while len(losses) < steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = _move_greedy_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"])
        loss = _policy_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    train_eval_loader = make_greedy_policy_loader(
        train_paths,
        batch_size=batch_size,
        shuffle_buffer=1,
        seed=seed,
        corpus_format=train_format,
    )
    val_eval_loader = make_greedy_policy_loader(
        val_paths,
        batch_size=batch_size,
        shuffle_buffer=1,
        seed=seed,
        corpus_format=val_format,
    )
    train_eval = _policy_metrics_from_loader(
        model,
        train_eval_loader,
        max_records=max_val_records,
        device=device,
    )
    val_eval = _policy_metrics_from_loader(
        model,
        val_eval_loader,
        max_records=max_val_records,
        device=device,
    )
    return {
        "status": "pass",
        "scientific_eligibility": "behavior_clone_pretraining",
        "experiment_id": experiment_id,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "shuffle_buffer": shuffle_buffer,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "train_paths": [str(path) for path in train_paths],
        "val_paths": [str(path) for path in val_paths],
        "train_format": train_format,
        "val_format": val_format,
        "train_record_count": count_records(train_paths, corpus_format=train_format),
        "val_record_count": count_records(val_paths, corpus_format=val_format),
        "model": {
            "parameter_count": parameter_count(model),
            "loss_head": losses[:5],
            "loss_tail": losses[-5:],
            "train_eval": train_eval,
            "val_eval": val_eval,
        },
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Comma-separated greedy corpus paths")
    parser.add_argument("--val", required=True, help="Comma-separated greedy corpus paths")
    parser.add_argument("--train-format", choices=["jsonl", "npz"], default="jsonl")
    parser.add_argument("--val-format", choices=["jsonl", "npz"], default="jsonl")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260660)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--max-val-records", type=int, default=20000)
    parser.add_argument("--experiment-id", default="crt-greedy-policy-pretrain-v1")
    parser.add_argument("--out", default="cascadiav3/reports/greedy_policy_pretrain.json")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/greedy_policy_pretrain.pt")
    args = parser.parse_args()

    import torch

    result = run_greedy_policy_pretrain(
        parse_paths(args.train),
        parse_paths(args.val),
        train_format=args.train_format,
        val_format=args.val_format,
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
        shuffle_buffer=args.shuffle_buffer,
        max_val_records=args.max_val_records,
        experiment_id=args.experiment_id,
    )
    model_state_dict = result.pop("model_state_dict")
    optimizer_state_dict = result.pop("optimizer_state_dict")
    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "report": result,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded["report"]["experiment_id"] != result["experiment_id"]:
        raise RuntimeError("checkpoint round-trip experiment_id mismatch")
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_roundtrip"] = "pass"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
