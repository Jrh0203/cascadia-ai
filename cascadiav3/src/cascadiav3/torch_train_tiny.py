"""One-root GPU overfit smoke for CascadiaFormer-Zero-S-tiny."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_features import action_features, state_features, target_score_decomposition
from .torch_model import CascadiaFormerZeroSConfig, build_tiny_model, parameter_count


def _tensorize(root: dict[str, Any], device_name: str):
    import torch

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    batch = {
        "state": torch.tensor(state_features(root), device=device, dtype=dtype).unsqueeze(0),
        "actions": torch.tensor(action_features(root), device=device, dtype=dtype),
        "target_q": torch.tensor(root["per_action_Q"], device=device, dtype=dtype),
        "target_value": torch.tensor(root["final_score_vector"], device=device, dtype=dtype),
        "target_rank": torch.tensor(root["rank_vector"], device=device, dtype=dtype),
        "target_score": torch.tensor(target_score_decomposition(root), device=device, dtype=dtype),
    }
    return batch, device


def _loss(out: dict[str, Any], batch: dict[str, Any]):
    import torch

    return (
        torch.nn.functional.mse_loss(out["legal_action_logits"], batch["target_q"])
        + 0.05 * torch.nn.functional.mse_loss(out["value_vector"], batch["target_value"])
        + 0.01 * torch.nn.functional.mse_loss(out["rank_logits"].mean(dim=1), batch["target_rank"])
        + 0.01 * torch.nn.functional.mse_loss(
            out["score_decomposition"],
            batch["target_score"],
        )
    )


def run_overfit(
    root_path: Path,
    *,
    steps: int = 300,
    lr: float = 3e-3,
    seed: int = 20260629,
    device_name: str = "cuda",
) -> dict[str, Any]:
    import torch

    root = json.loads(root_path.read_text(encoding="utf-8"))
    batch, device = _tensorize(root, device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    config = CascadiaFormerZeroSConfig()
    model = build_tiny_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses: list[float] = []
    with torch.no_grad():
        initial_out = model(batch["state"], batch["actions"])
        initial_loss = float(_loss(initial_out, batch).detach().cpu())

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch["state"], batch["actions"])
        loss = _loss(out, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        final_out = model(batch["state"], batch["actions"])
        final_loss = float(_loss(final_out, batch).detach().cpu())

    if final_loss >= initial_loss:
        raise RuntimeError(f"tiny overfit did not reduce loss: {initial_loss} -> {final_loss}")

    return {
        "status": "pass",
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "action_count": len(root["legal_actions"]),
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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="cascadiav3/reports/tiny_overfit.json")
    parser.add_argument(
        "--checkpoint",
        default="cascadiav3/checkpoints/tiny_overfit.pt",
        help="Checkpoint path for model, optimizer, and report metadata",
    )
    args = parser.parse_args()

    import torch

    root = Path(args.root)
    result = run_overfit(
        root / "fixtures" / "tiny_search_root.json",
        steps=args.steps,
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
