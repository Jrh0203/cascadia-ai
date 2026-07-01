"""Evaluate CascadiaFormer checkpoints on expert tensor validation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_inference_bridge import _load_model
from .torch_train_cascadiaformer import (
    LossWeights,
    _evaluate_records,
    _load_corpus,
)


def _parse_paths(raw: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one validation path is required")
    return paths


def evaluate_manifests(
    manifests: list[Path],
    *,
    val_paths: list[Path],
    val_format: str,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    import torch

    if not manifests:
        raise ValueError("at least one checkpoint manifest is required")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    records = _load_corpus(val_paths, corpus_format=val_format)
    rows: list[dict[str, Any]] = []
    try:
        for manifest in manifests:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            model = _load_model(manifest, manifest_path=manifest, manifest_payload=payload).to(device)
            model.eval()
            weights = LossWeights(**payload.get("loss_weights", {}))
            metrics = _evaluate_records(
                model=model,
                records=records,
                corpus_format=val_format,
                weights=weights,
                device=device,
                batch_size=batch_size,
                max_batches=None,
            )
            rows.append(
                {
                    "manifest": str(manifest),
                    "checkpoint_tag": payload.get("checkpoint_tag"),
                    "step": payload.get("step"),
                    "weights": payload.get("weights"),
                    "training_selection_metric": payload.get("selection_metric"),
                    "training_limited_metrics": payload.get("metrics", {}),
                    "full_validation_metrics": metrics,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        close = getattr(records, "close", None)
        if close is not None:
            close()

    ranked = sorted(rows, key=lambda row: float(row["full_validation_metrics"]["locked_val_total"]))
    return {
        "status": "pass",
        "device": str(device),
        "torch_version": torch.__version__,
        "val_paths": [str(path) for path in val_paths],
        "val_format": val_format,
        "batch_size": batch_size,
        "rows": rows,
        "best_by_full_locked_val_total": ranked[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="Comma-separated checkpoint manifest paths")
    parser.add_argument("--val", required=True, help="Comma-separated validation shard paths")
    parser.add_argument("--val-format", choices=["jsonl", "npz"], default="npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = evaluate_manifests(
        _parse_paths(args.manifests),
        val_paths=_parse_paths(args.val),
        val_format=args.val_format,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
