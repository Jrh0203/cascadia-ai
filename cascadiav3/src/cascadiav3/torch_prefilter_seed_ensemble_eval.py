"""Fixed seed-ensemble evaluation for CRT prefilter replay rows.

This evaluator reads per-root ranked prediction JSONL emitted by
``torch_prefilter_eval`` for multiple independent checkpoints. It normalizes
each checkpoint's action scores within a root, averages them with fixed weights,
and evaluates the resulting action ranking against the saved teacher Q labels.

It is deliberately not train-selected: weights default to a uniform average and
are supplied explicitly when changed. That makes it a cheap robustness check for
near-threshold K=16 prefilter results without using validation labels to tune a
combiner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_K_VALUES = [4, 8, 16, 24, 32]


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("at least one K value is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"K values must be positive: {values}")
    return values


def parse_weights(raw: str | None, source_count: int) -> list[float]:
    if raw is None or not raw.strip():
        return [1.0 / source_count for _ in range(source_count)]
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(weights) != source_count:
        raise ValueError(f"expected {source_count} weights, got {len(weights)}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"weights must be nonnegative: {weights}")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weight sum must be positive")
    return [weight / total for weight in weights]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(variance, 1.0e-12))
    return {action_id: (score - mean) / std for action_id, score in scores.items()}


def _score_map(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(action_id): float(score)
        for action_id, score in zip(row["ranked_action_ids"], row["ranked_predicted_q"], strict=True)
    }


def _teacher_q_map(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(action_id): float(score)
        for action_id, score in zip(row["ranked_action_ids"], row["ranked_teacher_q"], strict=True)
    }


def read_per_root_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "state_hash",
                "ranked_action_ids",
                "ranked_predicted_q",
                "ranked_teacher_q",
                "teacher_best",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number} missing keys {missing}")
            if len(row["ranked_action_ids"]) != len(row["ranked_predicted_q"]):
                raise ValueError(f"{path}:{line_number} action/predicted length mismatch")
            if len(row["ranked_action_ids"]) != len(row["ranked_teacher_q"]):
                raise ValueError(f"{path}:{line_number} action/teacher length mismatch")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any rows")
    return rows


def _align_sources(sources: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    first = sources[0]
    aligned: list[list[dict[str, Any]]] = []
    for index, row in enumerate(first):
        state_hash = row["state_hash"]
        action_set = set(row["ranked_action_ids"])
        group = [row]
        for source_index, source_rows in enumerate(sources[1:], start=1):
            try:
                other = source_rows[index]
            except IndexError as exc:
                raise ValueError(f"source {source_index} has fewer rows than source 0") from exc
            if other["state_hash"] != state_hash:
                raise ValueError(
                    f"state mismatch at row {index}: {state_hash} != {other['state_hash']} in source {source_index}"
                )
            if set(other["ranked_action_ids"]) != action_set:
                raise ValueError(f"action set mismatch at row {index} in source {source_index}")
            group.append(other)
        aligned.append(group)
    for source_index, source_rows in enumerate(sources[1:], start=1):
        if len(source_rows) != len(first):
            raise ValueError(f"source {source_index} row count {len(source_rows)} != source 0 row count {len(first)}")
    return aligned


def _serving_decision(
    metrics: dict[str, Any],
    *,
    k_values: list[int],
    min_recall: float,
    max_oracle_regret: float,
) -> dict[str, Any]:
    gates = {}
    recommended_k = None
    for k in k_values:
        row = metrics["prefilter"][str(k)]
        recall = float(row["recall"])
        regret = float(row["mean_oracle_regret"])
        passes_recall = recall >= min_recall
        passes_oracle_regret = regret <= max_oracle_regret
        passes = passes_recall and passes_oracle_regret
        gates[str(k)] = {
            "recall": recall,
            "mean_oracle_regret": regret,
            "passes_recall": passes_recall,
            "passes_oracle_regret": passes_oracle_regret,
            "passes": passes,
        }
        if passes and recommended_k is None:
            recommended_k = k
    return {
        "criteria": (
            f"serving candidate requires teacher-best recall >= {min_recall:.3f} "
            f"and mean oracle regret <= {max_oracle_regret:.3f} sampled-teacher points"
        ),
        "gates": gates,
        "passes": recommended_k is not None,
        "recommended_k": recommended_k,
        "best_available_k": max(k_values),
    }


def evaluate_aligned_groups(
    groups: list[list[dict[str, Any]]],
    *,
    weights: list[float],
    k_values: list[int],
    per_root_path: Path | None = None,
) -> dict[str, Any]:
    if len(weights) != len(groups[0]):
        raise ValueError("weight count must match source count")

    total = 0
    top1 = 0
    top4 = 0
    regret_sum = 0.0
    best_q_sum = 0.0
    selected_q_sum = 0.0
    prefilter_hits = {k: 0 for k in k_values}
    prefilter_regret_sum = {k: 0.0 for k in k_values}
    prefilter_q_sum = {k: 0.0 for k in k_values}
    per_root_rows: list[dict[str, Any]] = []

    for group in groups:
        reference = group[0]
        teacher_q = _teacher_q_map(reference)
        teacher_best_action_id = str(reference["teacher_best"]["action_id"])
        best_q = float(reference["teacher_best"]["q"])
        action_ids = list(reference["ranked_action_ids"])
        ensemble_scores = {action_id: 0.0 for action_id in action_ids}
        for weight, row in zip(weights, group, strict=True):
            normalized = _normalize(_score_map(row))
            for action_id in action_ids:
                ensemble_scores[action_id] += weight * normalized[action_id]

        ranked_action_ids = sorted(action_ids, key=lambda action_id: ensemble_scores[action_id], reverse=True)
        selected_action_id = ranked_action_ids[0]
        selected_q = teacher_q[selected_action_id]
        total += 1
        top1 += int(selected_action_id == teacher_best_action_id)
        top4 += int(teacher_best_action_id in ranked_action_ids[: min(4, len(ranked_action_ids))])
        regret_sum += best_q - selected_q
        best_q_sum += best_q
        selected_q_sum += selected_q

        per_root_prefilter = {}
        for k in k_values:
            retained = ranked_action_ids[: min(k, len(ranked_action_ids))]
            retained_best_action_id = max(retained, key=lambda action_id: teacher_q[action_id])
            retained_best_q = teacher_q[retained_best_action_id]
            hit = teacher_best_action_id in retained
            prefilter_hits[k] += int(hit)
            prefilter_regret_sum[k] += best_q - retained_best_q
            prefilter_q_sum[k] += retained_best_q
            per_root_prefilter[str(k)] = {
                "teacher_best_retained": hit,
                "oracle_regret": best_q - retained_best_q,
                "oracle_best_action_id": retained_best_action_id,
                "oracle_best_q": retained_best_q,
                "retained_count": len(retained),
                "retained_action_ids": retained,
            }

        per_root_rows.append(
            {
                "state_hash": reference["state_hash"],
                "action_count": len(action_ids),
                "teacher_best": reference["teacher_best"],
                "model_selected": {
                    "action_id": selected_action_id,
                    "predicted_score": ensemble_scores[selected_action_id],
                    "teacher_q": selected_q,
                    "regret": best_q - selected_q,
                },
                "ranked_action_ids": ranked_action_ids,
                "ranked_ensemble_scores": [ensemble_scores[action_id] for action_id in ranked_action_ids],
                "ranked_teacher_q": [teacher_q[action_id] for action_id in ranked_action_ids],
                "prefilter": per_root_prefilter,
            }
        )

    if per_root_path is not None:
        per_root_path.parent.mkdir(parents=True, exist_ok=True)
        with per_root_path.open("w", encoding="utf-8") as handle:
            for row in per_root_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "roots": total,
        "top1_agreement": top1 / total if total else 0.0,
        "top4_recall": top4 / total if total else 0.0,
        "mean_regret": regret_sum / total if total else 0.0,
        "mean_best_q": best_q_sum / total if total else 0.0,
        "mean_selected_q": selected_q_sum / total if total else 0.0,
        "prefilter": {
            str(k): {
                "recall": prefilter_hits[k] / total if total else 0.0,
                "mean_oracle_regret": prefilter_regret_sum[k] / total if total else 0.0,
                "mean_oracle_q": prefilter_q_sum[k] / total if total else 0.0,
                "mean_retained_count": min(k, max((len(group[0]["ranked_action_ids"]) for group in groups), default=0)),
            }
            for k in k_values
        },
    }


def run_seed_ensemble_eval(
    inputs: list[Path],
    *,
    weights: list[float],
    k_values: list[int],
    min_recall: float,
    max_oracle_regret: float,
    experiment_id: str,
    per_root_path: Path | None,
) -> dict[str, Any]:
    sources = [read_per_root_jsonl(path) for path in inputs]
    groups = _align_sources(sources)
    metrics = evaluate_aligned_groups(groups, weights=weights, k_values=k_values, per_root_path=per_root_path)
    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_seed_ensemble_eval",
        "experiment_id": experiment_id,
        "inputs": [str(path) for path in inputs],
        "weights": weights,
        "k_values": k_values,
        "min_recall": min_recall,
        "max_oracle_regret": max_oracle_regret,
        "metrics": metrics,
        "serving_decision": _serving_decision(
            metrics,
            k_values=k_values,
            min_recall=min_recall,
            max_oracle_regret=max_oracle_regret,
        ),
        "per_root_out": str(per_root_path) if per_root_path else None,
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    prefilter = report["metrics"]["prefilter"]
    decision = report["serving_decision"]
    lines = [
        "# CRT Seed-Ensemble Prefilter Eval",
        "",
        f"Experiment: `{report['experiment_id']}`",
        "",
        "## Inputs",
        "",
    ]
    for input_path, weight in zip(report["inputs"], report["weights"], strict=True):
        lines.append(f"- `{input_path}` weight `{weight:.4f}`")
    lines.extend(
        [
            "",
            "## Result",
            "",
            "| K | Recall | Oracle regret | Pass |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k in report["k_values"]:
        row = prefilter[str(k)]
        gate = decision["gates"][str(k)]
        lines.append(
            f"| {k} | {row['recall']:.4f} | {row['mean_oracle_regret']:.4f} | {gate['passes']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Passes: `{decision['passes']}`",
            f"- Recommended K: `{decision['recommended_k']}`",
            f"- Mean regret: `{report['metrics']['mean_regret']:.4f}`",
            f"- Top-1 agreement: `{report['metrics']['top1_agreement']:.4f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Comma-separated per-root JSONL files")
    parser.add_argument("--weights", default=None, help="Comma-separated nonnegative source weights; default uniform")
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-oracle-regret", type=float, default=0.25)
    parser.add_argument("--experiment-id", default="crt-prefilter-seed-ensemble-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_prefilter_seed_ensemble_eval.json")
    parser.add_argument("--per-root-out", default="")
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args()

    inputs = [Path(part.strip()) for part in args.inputs.split(",") if part.strip()]
    if len(inputs) < 2:
        raise ValueError("seed ensemble needs at least two input files")
    k_values = parse_k_values(args.k_values)
    weights = parse_weights(args.weights, len(inputs))
    per_root_path = Path(args.per_root_out) if args.per_root_out else None
    result = run_seed_ensemble_eval(
        inputs,
        weights=weights,
        k_values=k_values,
        min_recall=args.min_recall,
        max_oracle_regret=args.max_oracle_regret,
        experiment_id=args.experiment_id,
        per_root_path=per_root_path,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_summary(result, summary_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
