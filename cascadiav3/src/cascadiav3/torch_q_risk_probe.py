"""Fixed-root diagnostic for distributional-Q serving policies.

This is an engineering screen, not gameplay or promotion evidence. It runs one
checkpoint once over a provenance-locked root corpus, measures raw quantile
crossing, and reports how monotone-rearranged q25/q50/q75 serving changes the
direct derived-Q argmax relative to the established quantile mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path
from statistics import mean, median
from typing import Any

from .torch_inference_bridge import (
    Q_RISK_MODES,
    _eval_batch_chunks,
    _load_model,
    _model_inputs_to_device,
    collate_inference_roots,
    resolve_checkpoint_path,
)
from .torch_model_throughput_benchmark import production_packed_root


REPORT_SCHEMA = "cascadiav3.q_risk_probe.v1"
RISK_TARGETS = {"q25": 0.25, "q50": 0.50, "q75": 0.75}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _risk_quantile(sorted_quantiles: Any, mode: str) -> Any:
    if mode == "mean":
        return sorted_quantiles.mean(axis=-1)
    count = int(sorted_quantiles.shape[-1])
    position = RISK_TARGETS[mode] * count - 0.5
    if position <= 0:
        return sorted_quantiles[..., 0]
    if position >= count - 1:
        return sorted_quantiles[..., count - 1]
    lower = int(position)
    weight = position - lower
    return (
        sorted_quantiles[..., lower] * (1.0 - weight)
        + sorted_quantiles[..., lower + 1] * weight
    )


def summarize_q_risk_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize already-evaluated rows; kept pure for deterministic tests."""
    import numpy as np

    if not rows:
        raise ValueError("q-risk probe requires at least one evaluated root")

    quantile_count: int | None = None
    action_count = 0
    adjacent_pairs = 0
    crossing_pairs = 0
    actions_with_crossing = 0
    crossing_magnitudes: list[float] = []
    raw_level_values: list[list[float]] = []
    rearranged_level_values: list[list[float]] = []
    mean_consistency_max_abs = 0.0
    mode_state = {
        mode: {
            "flips": 0,
            "mean_q_regrets": [],
            "score_to_go_offsets": [],
            "examples": [],
        }
        for mode in Q_RISK_MODES
        if mode != "mean"
    }

    for row in rows:
        raw = np.asarray(row["quantiles"], dtype=np.float64)
        exact = np.asarray(row["exact_afterstate_score_active"], dtype=np.float64)
        served_mean = np.asarray(row["served_mean"], dtype=np.float64)
        action_ids = list(row["action_ids"])
        if raw.ndim != 2 or raw.shape[0] != len(action_ids):
            raise ValueError("quantile rows must be [actions, quantiles]")
        if exact.shape != (raw.shape[0],) or served_mean.shape != (raw.shape[0],):
            raise ValueError("q-risk row fields must align with action count")
        if quantile_count is None:
            quantile_count = int(raw.shape[1])
            if quantile_count <= 1:
                raise ValueError("q-risk probe requires a distributional-Q checkpoint")
            raw_level_values = [[] for _ in range(quantile_count)]
            rearranged_level_values = [[] for _ in range(quantile_count)]
        elif raw.shape[1] != quantile_count:
            raise ValueError("all q-risk rows must have the same quantile count")

        sorted_quantiles = np.sort(raw, axis=-1)
        raw_mean = raw.mean(axis=-1)
        mean_consistency_max_abs = max(
            mean_consistency_max_abs,
            float(np.max(np.abs(raw_mean - served_mean))),
        )
        for level in range(quantile_count):
            raw_level_values[level].extend(raw[:, level].tolist())
            rearranged_level_values[level].extend(sorted_quantiles[:, level].tolist())

        differences = raw[:, :-1] - raw[:, 1:]
        crossing_mask = differences > 0.0
        adjacent_pairs += int(crossing_mask.size)
        crossing_pairs += int(crossing_mask.sum())
        actions_with_crossing += int(crossing_mask.any(axis=1).sum())
        crossing_magnitudes.extend(differences[crossing_mask].tolist())
        action_count += int(raw.shape[0])

        mean_final_q = exact + served_mean
        mean_index = int(np.argmax(mean_final_q))
        for mode, state in mode_state.items():
            statistic = _risk_quantile(sorted_quantiles, mode)
            final_q = exact + statistic
            selected_index = int(np.argmax(final_q))
            flipped = selected_index != mean_index
            state["flips"] += int(flipped)
            state["mean_q_regrets"].append(
                max(0.0, float(mean_final_q[mean_index] - mean_final_q[selected_index]))
            )
            state["score_to_go_offsets"].extend(
                np.abs(statistic - served_mean).tolist()
            )
            if flipped and len(state["examples"]) < 20:
                state["examples"].append(
                    {
                        "state_hash": row.get("state_hash"),
                        "mean_action_id": action_ids[mean_index],
                        "risk_action_id": action_ids[selected_index],
                        "mean_q_regret": float(
                            mean_final_q[mean_index] - mean_final_q[selected_index]
                        ),
                    }
                )

    root_count = len(rows)
    return {
        "root_count": root_count,
        "action_count": action_count,
        "quantile_count": quantile_count,
        "mean_output_consistency_max_abs": mean_consistency_max_abs,
        "raw_quantile_crossing": {
            "adjacent_pair_count": adjacent_pairs,
            "crossing_pair_count": crossing_pairs,
            "crossing_pair_rate": crossing_pairs / adjacent_pairs,
            "actions_with_crossing": actions_with_crossing,
            "action_crossing_rate": actions_with_crossing / action_count,
            "crossing_magnitude": _distribution(crossing_magnitudes),
        },
        "raw_level_means": [float(mean(values)) for values in raw_level_values],
        "rearranged_level_means": [
            float(mean(values)) for values in rearranged_level_values
        ],
        "serving_modes": {
            mode: {
                "direct_argmax_flip_count_vs_mean": state["flips"],
                "direct_argmax_flip_rate_vs_mean": state["flips"] / root_count,
                "mean_policy_q_regret": _distribution(state["mean_q_regrets"]),
                "absolute_score_to_go_offset_from_mean": _distribution(
                    state["score_to_go_offsets"]
                ),
                "flip_examples": state["examples"],
            }
            for mode, state in mode_state.items()
        },
    }


def _load_roots(path: Path, max_roots: int) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                roots.append(json.loads(line))
                if max_roots > 0 and len(roots) >= max_roots:
                    break
    if not roots:
        raise ValueError("q-risk probe root corpus is empty")
    return roots


def run_q_risk_probe(
    *,
    manifest: Path,
    roots_path: Path,
    device_name: str,
    max_roots: int,
    chunk_size: int,
    source_revision: str,
) -> dict[str, Any]:
    import torch

    if not source_revision.strip():
        raise ValueError("source_revision is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(manifest_payload.get("config", {}).get("q_quantiles", 1)) <= 1:
        raise ValueError("q-risk probe requires a distributional-Q manifest")
    weights = resolve_checkpoint_path(
        manifest_payload["weights"],
        manifest_path=manifest,
        checkpoint_path=manifest,
    )
    roots = _load_roots(roots_path, max_roots)
    prepared_roots = [production_packed_root(root) for root in roots]
    model = _load_model(
        manifest,
        manifest_path=manifest,
        manifest_payload=manifest_payload,
        device_name=device_name,
    )
    device = torch.device(
        device_name
        if device_name != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    evaluated: list[dict[str, Any]] = []
    for chunk in _eval_batch_chunks(prepared_roots, chunk_size=chunk_size):
        batch = collate_inference_roots(chunk)
        inputs = _model_inputs_to_device(batch, device)
        with torch.inference_mode():
            outputs = model(
                inputs["tokens"],
                inputs["token_mask"],
                inputs["actions"],
                inputs["action_mask"],
                relation_ids=inputs.get("relation_ids"),
                relation_tail=inputs.get("relation_tail"),
            )
        quantiles = outputs.get("q_quantile_values")
        if quantiles is None:
            raise ValueError("loaded model did not emit q_quantile_values")
        quantiles = quantiles.float().cpu().numpy()
        served_mean = outputs["q"].float().cpu().numpy()
        exact = batch["exact_afterstate_score_active"].numpy()
        for index, root in enumerate(chunk):
            count = batch["action_counts"][index]
            evaluated.append(
                {
                    "state_hash": root.get("state_hash"),
                    "action_ids": batch["action_ids"][index],
                    "exact_afterstate_score_active": exact[index, :count].tolist(),
                    "served_mean": served_mean[index, :count].tolist(),
                    "quantiles": quantiles[index, :count, :].tolist(),
                }
            )

    return {
        "schema_id": REPORT_SCHEMA,
        "scientific_eligibility": "engineering_fixed_root_screen_only",
        "source_revision": source_revision,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device_requested": device_name,
        "device_resolved": str(device),
        "checkpoint": {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "checkpoint_tag": manifest_payload.get("checkpoint_tag"),
            "step": manifest_payload.get("step"),
        },
        "roots": {
            "path": str(roots_path),
            "source_file_sha256": _sha256(roots_path),
            "selected_record_sha256": _records_sha256(roots),
            "production_packed_record_sha256": _records_sha256(prepared_roots),
            "max_roots": max_roots,
        },
        "serving_definition": {
            "mean": "checkpoint q output (arithmetic mean of raw heads)",
            "non_mean": "linear interpolation after per-action monotone rearrangement",
            "modes": list(Q_RISK_MODES),
        },
        "results": summarize_q_risk_rows(evaluated),
    }


def render_markdown(report: dict[str, Any]) -> str:
    results = report["results"]
    crossing = results["raw_quantile_crossing"]
    lines = [
        "# Distributional-Q Risk-Serving Fixed-Root Probe",
        "",
        f"- Eligibility: `{report['scientific_eligibility']}`",
        f"- Source revision: `{report['source_revision']}`",
        f"- Host/device: `{report['host']}` / `{report['device_resolved']}`",
        f"- Roots/actions: {results['root_count']} / {results['action_count']}",
        f"- Raw adjacent-head crossing rate: {crossing['crossing_pair_rate']:.3%}",
        f"- Actions with at least one crossing: {crossing['action_crossing_rate']:.3%}",
        "",
        "| Mode | Direct argmax flip vs mean | Mean-Q regret | P95 regret |",
        "|---|---:|---:|---:|",
    ]
    for mode, values in results["serving_modes"].items():
        regret = values["mean_policy_q_regret"]
        lines.append(
            f"| {mode} | {values['direct_argmax_flip_rate_vs_mean']:.3%} "
            f"({values['direct_argmax_flip_count_vs_mean']}) | "
            f"{regret['mean']:.4f} | {regret['p95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This diagnoses whether the existing head contains a distinct serving signal. "
            "It is not gameplay evidence and cannot promote a policy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-roots", type=int, default=160)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args()
    report = run_q_risk_probe(
        manifest=Path(args.manifest),
        roots_path=Path(args.roots),
        device_name=args.device,
        max_roots=args.max_roots,
        chunk_size=args.chunk_size,
        source_revision=args.source_revision,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "pass", "out": str(out_path), **report["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
