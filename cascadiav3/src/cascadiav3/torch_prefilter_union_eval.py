"""Train-selected source-union prefilter evaluation.

Forensics showed that MLP is the best single K=16 prefilter, but other sources
recover a meaningful number of its misses. This evaluator searches simple
source-quota union rules on train, then evaluates the selected rule on the
held-out validation shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_prefilter_blend_eval import _collect_rows
from .torch_prefilter_eval import _serving_decision


def quota_grid(source_count: int, k: int) -> list[tuple[int, ...]]:
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    rows: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining_sources: int, remaining: int) -> None:
        if remaining_sources == 1:
            rows.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining_sources - 1, remaining - value)

    rec([], source_count, k)
    return rows


def _source_rankings(row: dict[str, Any]):  # type: ignore[no-untyped-def]
    import torch

    return {
        source: [int(index) for index in torch.argsort(scores, descending=True).tolist()]
        for source, scores in row["sources"].items()
    }


def _union_retained(row: dict[str, Any], quotas: tuple[int, ...], fill_source: str, *, k: int) -> list[int]:
    source_names = list(row["source_names"])
    rankings = _source_rankings(row)
    retained: list[int] = []
    seen: set[int] = set()

    def add_from(source: str, limit: int | None = None) -> None:
        ranking = rankings[source]
        count = 0
        for index in ranking:
            if index in seen:
                continue
            retained.append(index)
            seen.add(index)
            count += 1
            if len(retained) >= k:
                return
            if limit is not None and count >= limit:
                return

    for source, quota in zip(source_names, quotas, strict=True):
        if quota > 0:
            add_from(source, quota)
        if len(retained) >= k:
            return retained[:k]
    add_from(fill_source, None)
    for source in source_names:
        if len(retained) >= k:
            break
        add_from(source, None)
    return retained[:k]


def _evaluate_rows(rows: list[dict[str, Any]], quotas: tuple[int, ...], fill_source: str, *, k: int) -> dict[str, Any]:
    import torch

    total = 0
    hits = 0
    regret_sum = 0.0
    oracle_q_sum = 0.0
    selected_regret_sum = 0.0
    retained_count_sum = 0
    for row in rows:
        q = row["target_q"]
        teacher_best = int(torch.argmax(q).item())
        best_q = float(q[teacher_best].item())
        retained = _union_retained(row, quotas, fill_source, k=k)
        retained_q = q[retained]
        retained_best_q = float(retained_q.max().item())
        total += 1
        hits += int(teacher_best in retained)
        regret_sum += best_q - retained_best_q
        oracle_q_sum += retained_best_q
        selected_regret_sum += best_q - float(q[retained[0]].item())
        retained_count_sum += len(retained)
    return {
        "roots": total,
        "prefilter": {
            str(k): {
                "recall": hits / total if total else 0.0,
                "mean_oracle_regret": regret_sum / total if total else 0.0,
                "mean_oracle_q": oracle_q_sum / total if total else 0.0,
                "mean_retained_count": retained_count_sum / total if total else 0.0,
            }
        },
        "mean_selected_regret": selected_regret_sum / total if total else 0.0,
    }


def _selection_key(metrics: dict[str, Any], *, k: int, min_recall: float, max_oracle_regret: float):
    row = metrics["prefilter"][str(k)]
    recall = float(row["recall"])
    regret = float(row["mean_oracle_regret"])
    passes = recall >= min_recall and regret <= max_oracle_regret
    return (int(passes), recall, -regret, -float(metrics["mean_selected_regret"]))


def run_union_eval(
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device_name: str,
    k: int,
    min_recall: float,
    max_oracle_regret: float,
    experiment_id: str,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    train_rows = _collect_rows(train_path, checkpoint, batch_size=batch_size, device=device)
    val_rows = _collect_rows(val_path, checkpoint, batch_size=batch_size, device=device)
    source_names = tuple(train_rows[0]["source_names"]) if train_rows else ()
    candidates = []
    for quotas in quota_grid(len(source_names), k):
        for fill_source in source_names:
            train_metrics = _evaluate_rows(train_rows, quotas, fill_source, k=k)
            candidates.append(
                {
                    "quotas": dict(zip(source_names, quotas, strict=True)),
                    "fill_source": fill_source,
                    "train_metrics": train_metrics,
                    "selection_key": _selection_key(
                        train_metrics,
                        k=k,
                        min_recall=min_recall,
                        max_oracle_regret=max_oracle_regret,
                    ),
                }
            )
    candidates.sort(key=lambda row: row["selection_key"], reverse=True)
    selected = candidates[0]
    quota_tuple = tuple(int(selected["quotas"][source]) for source in source_names)
    val_metrics = _evaluate_rows(val_rows, quota_tuple, selected["fill_source"], k=k)
    selected["val_metrics"] = val_metrics
    selected["serving_decision"] = _serving_decision(
        val_metrics,
        k_values=[k],
        min_recall=min_recall,
        max_oracle_regret=max_oracle_regret,
    )
    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_union_eval",
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_experiment_id": checkpoint["report"].get("experiment_id"),
        "train": str(train_path),
        "val": str(val_path),
        "batch_size": batch_size,
        "k": k,
        "min_recall": min_recall,
        "max_oracle_regret": max_oracle_regret,
        "source_names": list(source_names),
        "selected": selected,
        "top_train_candidates": candidates[:10],
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    selected = report["selected"]
    train = selected["train_metrics"]["prefilter"][str(report["k"])]
    val = selected["val_metrics"]["prefilter"][str(report["k"])]
    lines = [
        "# CRT Wide-32 R16p20 Source-Union Prefilter Eval",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## Selected Rule",
        "",
        f"- Quotas: `{selected['quotas']}`",
        f"- Fill source: `{selected['fill_source']}`",
        "",
        "## Result",
        "",
        "| Split | K=16 recall | K=16 oracle regret |",
        "|---|---:|---:|",
        f"| Train | {train['recall']:.4f} | {train['mean_oracle_regret']:.4f} |",
        f"| Validation | {val['recall']:.4f} | {val['mean_oracle_regret']:.4f} |",
        "",
        "## Decision",
        "",
        f"- Validation passes: `{selected['serving_decision']['passes']}`",
        f"- Recommended K: `{selected['serving_decision']['recommended_k']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-oracle-regret", type=float, default=0.25)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-source-union-prefilter-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_source_union_prefilter.json")
    parser.add_argument("--summary-out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_source_union_prefilter_summary.md")
    args = parser.parse_args()

    result = run_union_eval(
        Path(args.train),
        Path(args.val),
        Path(args.checkpoint),
        batch_size=args.batch_size,
        device_name=args.device,
        k=args.k,
        min_recall=args.min_recall,
        max_oracle_regret=args.max_oracle_regret,
        experiment_id=args.experiment_id,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(result, summary_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
