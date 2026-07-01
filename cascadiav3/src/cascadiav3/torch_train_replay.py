"""Tiny JSONL replay-batch training smoke for CascadiaFormer-Zero-S-tiny."""

from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

from .torch_model import CascadiaFormerZeroSConfig, build_tiny_model, parameter_count
from .torch_replay import SearchRootJsonlDataset, make_replay_loader


def _to_device(batch: dict[str, Any], device):
    tensor_keys = [
        "state",
        "actions",
        "action_mask",
        "target_q",
        "target_value",
        "target_rank",
        "target_score",
    ]
    return {key: value.to(device) if key in tensor_keys else value for key, value in batch.items()}


def _masked_mse(pred, target, mask):  # type: ignore[no-untyped-def]
    import torch

    mask_f = mask.to(pred.dtype)
    return (((pred - target) ** 2) * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _loss(out: dict[str, Any], batch: dict[str, Any]):
    import torch

    return (
        _masked_mse(out["legal_action_logits"], batch["target_q"], batch["action_mask"])
        + 0.05 * torch.nn.functional.mse_loss(out["value_vector"], batch["target_value"])
        + 0.01
        * torch.nn.functional.mse_loss(out["rank_logits"].mean(dim=2), batch["target_rank"])
        + 0.01 * torch.nn.functional.mse_loss(out["score_decomposition"], batch["target_score"])
    )


def run_replay_train(
    replay_path: Path,
    *,
    steps: int = 400,
    batch_size: int = 2,
    lr: float = 3e-3,
    seed: int = 20260629,
    device_name: str = "cuda",
) -> dict[str, Any]:
    import torch

    dataset = SearchRootJsonlDataset(replay_path)
    loader = make_replay_loader(replay_path, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    config = CascadiaFormerZeroSConfig()
    model = build_tiny_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    first_batch_cpu = next(iter(loader))
    first_batch = _to_device(first_batch_cpu, device)
    with torch.no_grad():
        initial_out = model(
            first_batch["state"],
            first_batch["actions"],
            first_batch["action_mask"],
        )
        initial_loss = float(_loss(initial_out, first_batch).detach().cpu())

    losses: list[float] = []
    loader_cycle = cycle(loader)
    for _ in range(steps):
        batch = _to_device(next(loader_cycle), device)
        opt.zero_grad(set_to_none=True)
        out = model(batch["state"], batch["actions"], batch["action_mask"])
        loss = _loss(out, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        final_out = model(
            first_batch["state"],
            first_batch["actions"],
            first_batch["action_mask"],
        )
        final_loss = float(_loss(final_out, first_batch).detach().cpu())

    if final_loss >= initial_loss:
        raise RuntimeError(f"replay train did not reduce loss: {initial_loss} -> {final_loss}")

    return {
        "status": "pass",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "replay_path": str(replay_path),
        "record_count": len(dataset),
        "action_counts": dataset.action_counts,
        "max_legal_actions": max(dataset.action_counts),
        "mask_shape": list(first_batch["action_mask"].shape),
        "valid_action_count": int(first_batch["action_mask"].sum().item()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "loss_head": losses[:5],
        "loss_tail": losses[-5:],
        "legal_action_logits_shape": list(final_out["legal_action_logits"].shape),
        "value_vector_shape": list(final_out["value_vector"].shape),
        "rank_logits_shape": list(final_out["rank_logits"].shape),
        "score_decomposition_shape": list(final_out["score_decomposition"].shape),
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
        "cuda_max_memory_allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "model": model,
        "optimizer": opt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="cascadiav3")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replay",
        default=None,
        help="JSONL replay shard path; defaults to <root>/fixtures/tiny_replay.jsonl",
    )
    parser.add_argument("--out", default="cascadiav3/reports/tiny_replay_train.json")
    parser.add_argument(
        "--checkpoint",
        default="cascadiav3/checkpoints/tiny_replay_train.pt",
        help="Checkpoint path for model, optimizer, and report metadata",
    )
    args = parser.parse_args()

    import torch

    root = Path(args.root)
    replay_path = Path(args.replay) if args.replay is not None else root / "fixtures" / "tiny_replay.jsonl"
    result = run_replay_train(
        replay_path,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_name=args.device,
    )
    model = result.pop("model")
    optimizer = result.pop("optimizer")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "report": result,
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if loaded["report"]["final_loss"] != result["final_loss"]:
        raise RuntimeError("checkpoint round-trip report mismatch")
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_roundtrip"] = "pass"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
