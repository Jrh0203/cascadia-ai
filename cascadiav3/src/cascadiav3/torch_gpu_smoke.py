"""Tiny GPU smoke for CascadiaFormer-Zero-S tensor contracts.

This module intentionally stays small. It is not the real transformer and it is
not training evidence. It verifies that the tiny search-root fixture can become
GPU tensors, flow through a model-shaped module, run backward, and report a
machine-readable environment summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .torch_features import action_features, state_features
from .torch_model import CascadiaFormerZeroSConfig, build_tiny_model, parameter_count


def run_smoke(root_path: Path, device_name: str = "cuda") -> dict[str, Any]:
    import torch

    root = json.loads(root_path.read_text(encoding="utf-8"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    action_x = torch.tensor(action_features(root), device=device, dtype=dtype)
    state_x = torch.tensor(state_features(root), device=device, dtype=dtype).unsqueeze(0)

    torch.manual_seed(20260629)
    config = CascadiaFormerZeroSConfig()
    model = build_tiny_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target_q = torch.tensor(root["per_action_Q"], device=device, dtype=dtype)
    target_value = torch.tensor(root["final_score_vector"], device=device, dtype=dtype)

    before_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    out = model(state_x, action_x)
    loss = (
        torch.nn.functional.mse_loss(out["legal_action_logits"], target_q)
        + 0.01 * torch.nn.functional.mse_loss(out["value_vector"], target_value)
    )
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    after_memory = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    return {
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "action_count": len(root["legal_actions"]),
        "legal_action_logits_shape": list(out["legal_action_logits"].shape),
        "value_vector_shape": list(out["value_vector"].shape),
        "rank_logits_shape": list(out["rank_logits"].shape),
        "score_decomposition_shape": list(out["score_decomposition"].shape),
        "loss": float(loss.detach().cpu()),
        "cuda_memory_allocated_before": int(before_memory),
        "cuda_memory_allocated_after": int(after_memory),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="cascadiav3",
        help="Path to the cascadiav3 root containing fixtures/",
    )
    parser.add_argument(
        "--out",
        default="cascadiav3/reports/gpu_smoke.json",
        help="Path for the JSON smoke report",
    )
    args = parser.parse_args()

    root = Path(args.root)
    result = run_smoke(root / "fixtures" / "tiny_search_root.json")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
