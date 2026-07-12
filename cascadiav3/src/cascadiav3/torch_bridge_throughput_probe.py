"""Bridge serving-path throughput probe: eager vs opt-in fast paths.

Engineering-only; never gameplay or promotion evidence. Times the complete
in-process bridge path (chunk, collate, H2D, forward, D2H, response encode)
via ``_model_eval_batch`` for a matrix of arms:

- ``eager``          — production default path (baseline)
- ``bucket``         — CASCADIA_BRIDGE_BUCKET=1 shape bucketing
- ``compile``        — CASCADIA_BRIDGE_COMPILE=1 (mode from
                       CASCADIA_BRIDGE_COMPILE_MODE, default reduce-overhead)
- ``compile_bucket`` — both (the intended shipping pairing: bucketing keeps
                       the torch.compile recompile set finite)

Each arm loads the model through the production ``_load_model`` gate so the
measured path is exactly what serving would run. TF32 is forced OFF (battery
parity). For every non-eager arm the probe replays identical inputs and
records the max abs output difference vs eager per response key — this is the
evidence that decides whether adopting the arm needs a paired score gate.

See torch_model_throughput_benchmark.py for the multi-checkpoint variant;
this probe intentionally reuses its root loading/packing helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from contextlib import contextmanager
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Iterator

from .torch_model_throughput_benchmark import (
    _p95,
    _root_action_count,
    _sha256,
    _sync_device,
    load_roots,
    parse_positive_ints,
    prepare_roots,
)

ARMS = ("eager", "bucket", "compile", "compile_bucket")
RESPONSE_KEYS = ("priors", "q", "score_to_go", "uncertainty", "value")


def _arm_env(arm: str) -> dict[str, str | None]:
    """Env deltas an arm applies on top of the ambient environment. ``None``
    means the variable must be unset for the arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown probe arm: {arm}")
    env: dict[str, str | None] = {
        "CASCADIA_BRIDGE_COMPILE": None,
        "CASCADIA_BRIDGE_BUCKET": None,
    }
    if arm in ("compile", "compile_bucket"):
        env["CASCADIA_BRIDGE_COMPILE"] = "1"
    if arm in ("bucket", "compile_bucket"):
        env["CASCADIA_BRIDGE_BUCKET"] = "1"
    return env


@contextmanager
def _patched_env(mapping: dict[str, str | None]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in mapping}
    try:
        for key, value in mapping.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_tf32_off() -> None:
    if os.environ.get("CASCADIA_BRIDGE_TF32") == "1":
        raise RuntimeError(
            "bridge throughput probe runs with TF32 OFF (battery parity); "
            "unset CASCADIA_BRIDGE_TF32"
        )
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_arm_model(manifest_path: Path, device_name: str) -> Any:
    from .torch_inference_bridge import _load_model

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _load_model(
        manifest_path,
        manifest_path=manifest_path,
        manifest_payload=payload,
        device_name=device_name,
    )


def _max_abs_diff(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, float]:
    import numpy as np

    if len(baseline) != len(candidate):
        raise ValueError("response count mismatch between arms")
    diffs = {key: 0.0 for key in RESPONSE_KEYS}
    for base_row, cand_row in zip(baseline, candidate):
        if base_row["action_ids"] != cand_row["action_ids"]:
            raise ValueError("action_ids diverged between arms")
        for key in RESPONSE_KEYS:
            base = np.asarray(base_row[key], dtype=np.float64)
            cand = np.asarray(cand_row[key], dtype=np.float64)
            if base.shape != cand.shape:
                raise ValueError(f"{key} shape diverged between arms")
            if base.size:
                diffs[key] = max(diffs[key], float(np.max(np.abs(base - cand))))
    return diffs


def _time_batches(
    model: Any,
    roots: list[dict[str, Any]],
    *,
    batch_sizes: list[int],
    warmup_iterations: int,
    measured_iterations: int,
    device_name: str,
) -> list[dict[str, Any]]:
    from .torch_inference_bridge import _model_eval_batch

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch = list(islice(cycle(roots), batch_size))
        action_count = sum(_root_action_count(root) for root in batch)
        for _ in range(warmup_iterations):
            _model_eval_batch(model, batch, device_name=device_name, packed_response=True)
        _sync_device(device_name)
        timings: list[float] = []
        for _ in range(measured_iterations):
            _sync_device(device_name)
            started = time.perf_counter()
            _model_eval_batch(model, batch, device_name=device_name, packed_response=True)
            _sync_device(device_name)
            timings.append(time.perf_counter() - started)
        median_seconds = statistics.median(timings)
        rows.append(
            {
                "batch_size": batch_size,
                "actions": action_count,
                "iterations": measured_iterations,
                "min_seconds": min(timings),
                "median_seconds": median_seconds,
                "p95_seconds": _p95(timings),
                "rows_per_second": batch_size / median_seconds,
                "actions_per_second": action_count / median_seconds,
            }
        )
    return rows


def run_probe(
    *,
    manifest_path: Path,
    roots_path: Path,
    batch_sizes: list[int],
    warmup_iterations: int,
    measured_iterations: int,
    device_name: str,
    arms: list[str],
    source_revision: str | None,
) -> dict[str, Any]:
    import torch

    from .torch_inference_bridge import _model_eval_batch, bridge_env_provenance

    if "eager" not in arms:
        raise ValueError("the eager baseline arm is required")
    # Baseline first (numerics and speedups are relative to it), duplicates dropped.
    ordered = ["eager"]
    for arm in arms:
        if arm not in ordered:
            ordered.append(arm)
    arms = ordered
    if warmup_iterations < 1:
        raise ValueError("compile arms need at least one warmup iteration")
    if measured_iterations < 2:
        raise ValueError("measured_iterations must be at least two")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested but CUDA is unavailable")
    _require_tf32_off()

    roots = prepare_roots(load_roots(roots_path), "production-packed")
    numerics_batch = list(islice(cycle(roots), max(batch_sizes)))

    arm_reports: list[dict[str, Any]] = []
    baseline_responses: list[dict[str, Any]] | None = None
    for arm in arms:
        with _patched_env(_arm_env(arm)):
            model = _load_arm_model(manifest_path, device_name)
            env = bridge_env_provenance()
            batches = _time_batches(
                model,
                roots,
                batch_sizes=batch_sizes,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                device_name=device_name,
            )
            responses = _model_eval_batch(
                model, numerics_batch, device_name=device_name, packed_response=False
            )
            del model
        if arm == "eager":
            baseline_responses = responses
            numerics: dict[str, Any] = {key: 0.0 for key in RESPONSE_KEYS}
        else:
            assert baseline_responses is not None
            numerics = _max_abs_diff(baseline_responses, responses)
        arm_reports.append(
            {
                "arm": arm,
                "bridge_env": env,
                "batches": batches,
                "max_abs_diff_vs_eager": numerics,
                "bit_identical_to_eager": all(value == 0.0 for value in numerics.values()),
            }
        )

    baseline_rows = {
        row["batch_size"]: row for row in arm_reports[0]["batches"]
    }
    for report in arm_reports:
        for row in report["batches"]:
            row["speedup_vs_eager"] = (
                row["rows_per_second"]
                / baseline_rows[row["batch_size"]]["rows_per_second"]
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
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "roots": {
            "path": str(roots_path),
            "sha256": _sha256(roots_path),
            "unique_roots": len(roots),
            "action_counts": [_root_action_count(root) for root in roots],
        },
        "batch_sizes": batch_sizes,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "numerics_batch_rows": len(numerics_batch),
        "arms": arm_reports,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Bridge Throughput Probe (eager vs opt-in fast paths)",
        "",
        f"Device: `{report['device']}` · Host: `{report['host']}` · "
        f"Torch: `{report['torch_version']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Manifest: `{report['manifest']['path']}`",
        f"Eligibility: `{report['scientific_eligibility']}` (TF32 forced off)",
        "",
        "| Arm | Batch | Median s | Rows/s | Actions/s | Speedup vs eager |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        for row in arm["batches"]:
            lines.append(
                f"| {arm['arm']} | {row['batch_size']} | {row['median_seconds']:.6f} | "
                f"{row['rows_per_second']:.2f} | {row['actions_per_second']:.1f} | "
                f"{row['speedup_vs_eager']:.3f}x |"
            )
    lines += [
        "",
        "## Numerics vs eager (max abs diff, identical inputs, "
        f"{report['numerics_batch_rows']} rows)",
        "",
        "| Arm | " + " | ".join(RESPONSE_KEYS) + " | Bit-identical |",
        "|---|" + "---:|" * len(RESPONSE_KEYS) + "---|",
    ]
    for arm in report["arms"]:
        diffs = arm["max_abs_diff_vs_eager"]
        lines.append(
            f"| {arm['arm']} | "
            + " | ".join(f"{diffs[key]:.3e}" for key in RESPONSE_KEYS)
            + f" | {'yes' if arm['bit_identical_to_eager'] else 'NO'} |"
        )
    lines += [
        "",
        "Adoption rule: a non-bit-identical arm can only ship behind a paired",
        "score gate (same class as the TF32/bucketing precedent); a",
        "bit-identical arm can ship on throughput evidence alone.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json",
    )
    parser.add_argument("--roots", required=True)
    parser.add_argument("--batch-sizes", default="8,32,96,192")
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    arms = [part.strip() for part in args.arms.split(",") if part.strip()]
    report = run_probe(
        manifest_path=Path(args.manifest),
        roots_path=Path(args.roots),
        batch_sizes=parse_positive_ints(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        device_name=args.device,
        arms=arms,
        source_revision=args.source_revision or None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
