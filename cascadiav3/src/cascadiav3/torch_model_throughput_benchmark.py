"""Benchmark CascadiaFormer bridge throughput on fixed, identical roots.

This is an engineering probe, never gameplay or promotion evidence. It times
the complete in-process bridge path (collate, device transfer, model forward,
host copy, and packed response encoding) because a smaller model only buys
more search if the whole serving path becomes faster.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_positive_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("benchmark sizes must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one benchmark size is required")
    return values


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("manifest label must be non-empty")
        return label, Path(path)
    path = Path(raw)
    return path.stem, path


def load_roots(path: Path) -> list[dict[str, Any]]:
    roots = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not roots:
        raise ValueError("throughput benchmark requires at least one root")
    for index, root in enumerate(roots):
        if _root_action_count(root) <= 0:
            raise ValueError(f"root {index} has no action_ids")
    return roots


def _root_action_count(root: dict[str, Any]) -> int:
    action_ids = root.get("action_ids")
    if isinstance(action_ids, list):
        return len(action_ids)
    legal_actions = root.get("legal_actions")
    if isinstance(legal_actions, list):
        return len(legal_actions)
    return 0


def _sync_device(device_name: str) -> None:
    import torch

    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_name == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _clear_device(device_name: str) -> None:
    import torch

    gc.collect()
    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_name == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _response_digest(responses: list[dict[str, Any]]) -> str:
    payload = json.dumps(responses, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_model(
    *,
    label: str,
    model: Any,
    model_kind: str,
    model_size: str,
    parameters: int,
    roots: list[dict[str, Any]],
    batch_sizes: list[int],
    warmup_iterations: int,
    measured_iterations: int,
    device_name: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from .torch_inference_bridge import _model_eval_batch

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if measured_iterations < 2:
        raise ValueError("measured_iterations must be at least two")
    batches: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch = list(islice(cycle(roots), batch_size))
        action_count = sum(_root_action_count(root) for root in batch)
        for _ in range(warmup_iterations):
            _model_eval_batch(
                model,
                batch,
                device_name=device_name,
                packed_response=True,
            )
        _sync_device(device_name)

        timings: list[float] = []
        expected_digest: str | None = None
        for _ in range(measured_iterations):
            _sync_device(device_name)
            started = time.perf_counter()
            responses = _model_eval_batch(
                model,
                batch,
                device_name=device_name,
                packed_response=True,
            )
            _sync_device(device_name)
            timings.append(time.perf_counter() - started)
            digest = _response_digest(responses)
            if expected_digest is None:
                expected_digest = digest
            elif digest != expected_digest:
                raise RuntimeError(
                    f"{label} batch {batch_size} produced non-deterministic repeated outputs"
                )

        median_seconds = statistics.median(timings)
        batches.append(
            {
                "batch_size": batch_size,
                "actions": action_count,
                "iterations": measured_iterations,
                "min_seconds": min(timings),
                "median_seconds": median_seconds,
                "p95_seconds": _p95(timings),
                "rows_per_second": batch_size / median_seconds,
                "actions_per_second": action_count / median_seconds,
                "response_sha256": expected_digest,
            }
        )
    return {
        "label": label,
        "model_kind": model_kind,
        "model_size": model_size,
        "parameters": parameters,
        "provenance": provenance,
        "batches": batches,
    }


def _manifest_model(label: str, path: Path, device_name: str) -> tuple[Any, dict[str, Any]]:
    from .torch_inference_bridge import _load_model, resolve_checkpoint_path

    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = resolve_checkpoint_path(
        str(payload.get("weights", "")),
        manifest_path=path,
        checkpoint_path=path,
    )
    if not weights.is_file():
        raise FileNotFoundError(f"weights for {label} are missing: {weights}")
    model = _load_model(
        path,
        manifest_path=path,
        manifest_payload=payload,
        device_name=device_name,
    )
    return model, {
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256(weights),
        "checkpoint_tag": payload.get("checkpoint_tag"),
    }


def run_benchmark(
    *,
    roots_path: Path,
    manifests: list[tuple[str, Path]],
    synthetic_model_sizes: list[str],
    batch_sizes: list[int],
    warmup_iterations: int,
    measured_iterations: int,
    device_name: str,
    baseline_label: str | None,
    source_revision: str | None,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    from .torch_cascadiaformer import (
        build_cascadiaformer,
        config_for_size,
        parameter_count,
    )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA throughput probe requested but CUDA is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS throughput probe requested but MPS is unavailable")

    roots = load_roots(roots_path)
    model_reports: list[dict[str, Any]] = []
    labels: set[str] = set()

    for label, manifest_path in manifests:
        if label in labels:
            raise ValueError(f"duplicate model label: {label}")
        labels.add(label)
        model, provenance = _manifest_model(label, manifest_path, device_name)
        cfg = model.config
        model_reports.append(
            benchmark_model(
                label=label,
                model=model,
                model_kind="checkpoint",
                model_size=str(cfg.model_size),
                parameters=parameter_count(model),
                roots=roots,
                batch_sizes=batch_sizes,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                device_name=device_name,
                provenance=provenance,
            )
        )
        del model
        _clear_device(device_name)

    for raw_size in synthetic_model_sizes:
        cfg = config_for_size(raw_size)
        label = f"synthetic_{cfg.model_size}"
        if label in labels:
            raise ValueError(f"duplicate model label: {label}")
        labels.add(label)
        torch.manual_seed(seed)
        model = build_cascadiaformer(cfg).to(device_name).eval()
        model_reports.append(
            benchmark_model(
                label=label,
                model=model,
                model_kind="synthetic_shape_probe",
                model_size=cfg.model_size,
                parameters=parameter_count(model),
                roots=roots,
                batch_sizes=batch_sizes,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                device_name=device_name,
                provenance={"seed": seed, "config": cfg.to_dict()},
            )
        )
        del model
        _clear_device(device_name)

    if not model_reports:
        raise ValueError("at least one checkpoint or synthetic model is required")
    baseline_label = baseline_label or model_reports[0]["label"]
    by_label = {report["label"]: report for report in model_reports}
    if baseline_label not in by_label:
        raise ValueError(f"baseline label is absent: {baseline_label}")
    baseline_batches = {
        row["batch_size"]: row for row in by_label[baseline_label]["batches"]
    }
    comparisons: list[dict[str, Any]] = []
    for report in model_reports:
        comparisons.append(
            {
                "label": report["label"],
                "parameter_ratio_vs_baseline": (
                    by_label[baseline_label]["parameters"] / report["parameters"]
                ),
                "batches": [
                    {
                        "batch_size": row["batch_size"],
                        "throughput_speedup_vs_baseline": (
                            row["rows_per_second"]
                            / baseline_batches[row["batch_size"]]["rows_per_second"]
                        ),
                    }
                    for row in report["batches"]
                ],
            }
        )
    return {
        "status": "pass",
        "scientific_eligibility": "engineering_throughput_only",
        "source_revision": source_revision,
        "device": device_name,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "environment": {
            "cgab_fused": os.environ.get("CASCADIA_CGAB_FUSED") == "1",
            "bridge_bucket": os.environ.get("CASCADIA_BRIDGE_BUCKET") == "1",
            "bridge_compile": os.environ.get("CASCADIA_BRIDGE_COMPILE") == "1",
            "bridge_tf32": os.environ.get("CASCADIA_BRIDGE_TF32") == "1",
            "bridge_autocast": os.environ.get("CASCADIA_BRIDGE_AUTOCAST", ""),
            "eval_cell_budget": os.environ.get("CASCADIA_EVAL_CELL_BUDGET", ""),
        },
        "roots": {
            "path": str(roots_path),
            "sha256": _sha256(roots_path),
            "unique_roots": len(roots),
            "action_counts": [_root_action_count(root) for root in roots],
            "schema_ids": sorted(
                {str(root.get("schema_id")) for root in roots if root.get("schema_id")}
            ),
            "ruleset_ids": sorted(
                {str(root.get("ruleset_id")) for root in roots if root.get("ruleset_id")}
            ),
        },
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "baseline_label": baseline_label,
        "models": model_reports,
        "comparisons": comparisons,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline_label"]
    comparisons = {row["label"]: row for row in report["comparisons"]}
    lines = [
        "# CascadiaFormer Throughput Probe",
        "",
        f"Device: `{report['device']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Baseline: `{baseline}`",
        f"Eligibility: `{report['scientific_eligibility']}`",
        "",
        "| Model | Parameters | Batch | Median seconds | Rows/s | Speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        speedups = {
            row["batch_size"]: row["throughput_speedup_vs_baseline"]
            for row in comparisons[model["label"]]["batches"]
        }
        for row in model["batches"]:
            lines.append(
                f"| {model['label']} | {model['parameters']:,} | {row['batch_size']} | "
                f"{row['median_seconds']:.6f} | {row['rows_per_second']:.2f} | "
                f"{speedups[row['batch_size']]:.3f}x |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Checkpoint manifest as LABEL=PATH; repeat for multiple models.",
    )
    parser.add_argument(
        "--synthetic-model-sizes",
        default="",
        help="Comma-separated untrained shape probes (tiny,XS,S,M,L).",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--baseline-label", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    manifests = [_parse_labeled_path(raw) for raw in args.manifest]
    synthetic_sizes = [
        part.strip() for part in args.synthetic_model_sizes.split(",") if part.strip()
    ]
    report = run_benchmark(
        roots_path=Path(args.roots),
        manifests=manifests,
        synthetic_model_sizes=synthetic_sizes,
        batch_sizes=parse_positive_ints(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        device_name=args.device,
        baseline_label=args.baseline_label or None,
        source_revision=args.source_revision or None,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
