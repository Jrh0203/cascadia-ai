"""Analyze search benchmark decision traces for retained-set follow-ups.

The EI-0 search gate records one JSONL row per decision. Candidate rows include
the model ranking over the legal action menu and the full-search winning action
id, so they can answer the most important next question: how wide does the
model-retained set need to be to keep the full-search winner?

This analyzer deliberately avoids pretending it can replay alternative K values.
Without per-action rollout scores, a trace can exactly measure winner retention
for any K, but it cannot exactly know which action a smaller or larger retained
set would have selected when the full-search winner is missed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_K_VALUES = (1, 4, 8, 16, 24, 32, 40, 48, 56, 64)
PHASES = (
    ("opening", 0, 19),
    ("early_mid", 20, 39),
    ("late_mid", 40, 59),
    ("endgame", 60, 79),
)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _phase_for_ply(ply_index: int) -> str:
    for name, start, end in PHASES:
        if start <= ply_index <= end:
            return name
    return "unknown"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("strategy") == "cascadiaformer-search"
        and row.get("full_best_action_id")
        and isinstance(row.get("model_ranked_action_ids"), list)
    ]


def _enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        ranked = list(row.get("model_ranked_action_ids", []))
        full_best = str(row.get("full_best_action_id"))
        try:
            full_best_rank = ranked.index(full_best) + 1
        except ValueError:
            full_best_rank = None
        enriched.append(
            {
                **row,
                "full_best_model_rank": full_best_rank,
                "phase": _phase_for_ply(int(row.get("ply_index", -1))),
            }
        )
    return enriched


def _retention_for_rows(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total = len(rows)
    for k in k_values:
        ranks = [row["full_best_model_rank"] for row in rows if row["full_best_model_rank"] is not None]
        hits = [rank for rank in ranks if rank <= k]
        missed_rows = [row for row in rows if row["full_best_model_rank"] is not None and row["full_best_model_rank"] > k]
        missing_rank_rows = [row for row in rows if row["full_best_model_rank"] is None]
        missed_regrets = [
            float(row["search_regret"])
            for row in missed_rows
            if row.get("search_regret") is not None
        ]
        result[str(k)] = {
            "k": k,
            "decisions": total,
            "ranked_decisions": len(ranks),
            "missing_rank_decisions": len(missing_rank_rows),
            "full_best_retained_count": len(hits),
            "full_best_missed_count": len(missed_rows),
            "full_best_retained_rate": _round(len(hits) / len(ranks) if ranks else 0.0),
            "mean_observed_k32_regret_on_k_misses": _round(_mean(missed_regrets)),
            "p95_observed_k32_regret_on_k_misses": _round(_percentile(missed_regrets, 0.95)),
            "estimated_non_shadow_rollout_fraction": _round(k / max((int(row.get("candidate_count", k)) for row in rows), default=k)),
        }
    return result


def _phase_table(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(row)
    return {
        phase: {
            "decisions": len(phase_rows),
            "retention_by_k": _retention_for_rows(phase_rows, k_values),
        }
        for phase, phase_rows in sorted(by_phase.items())
    }


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [float(row["full_best_model_rank"]) for row in rows if row["full_best_model_rank"] is not None]
    return {
        "ranked_decisions": len(ranks),
        "missing_rank_decisions": len(rows) - len(ranks),
        "mean": _round(_mean(ranks)),
        "p50": _round(_percentile(ranks, 0.50)),
        "p75": _round(_percentile(ranks, 0.75)),
        "p90": _round(_percentile(ranks, 0.90)),
        "p95": _round(_percentile(ranks, 0.95)),
        "p99": _round(_percentile(ranks, 0.99)),
        "max": _round(max(ranks) if ranks else None),
    }


def _miss_examples(rows: list[dict[str, Any]], *, k: int, limit: int) -> list[dict[str, Any]]:
    misses = [
        row
        for row in rows
        if row["full_best_model_rank"] is not None and row["full_best_model_rank"] > k
    ]
    misses.sort(
        key=lambda row: (
            float(row.get("search_regret") or 0.0),
            int(row.get("full_best_model_rank") or 0),
        ),
        reverse=True,
    )
    return [
        {
            "seed_u64": int(row.get("seed_u64", -1)),
            "ply_index": int(row.get("ply_index", -1)),
            "phase": row["phase"],
            "active_seat": int(row.get("active_seat", -1)),
            "candidate_count": int(row.get("candidate_count", 0)),
            "full_best_model_rank": int(row["full_best_model_rank"]),
            "search_regret": _round(float(row.get("search_regret") or 0.0)),
            "selected_active_score": _round(float(row.get("selected_active_score") or 0.0)),
            "full_best_active_score": _round(float(row.get("full_best_active_score") or 0.0)),
            "model_top_q": _round(float(row.get("model_top_q") or 0.0)),
            "model_top_score_to_go": _round(float(row.get("model_top_score_to_go") or 0.0)),
        }
        for row in misses[:limit]
    ]


def _recommended_k(retention_by_k: dict[str, Any], *, target_recall: float) -> int:
    eligible = [
        int(k)
        for k, row in retention_by_k.items()
        if float(row["full_best_retained_rate"]) >= target_recall
    ]
    return min(eligible) if eligible else max(int(k) for k in retention_by_k)


def build_report(
    rows: list[dict[str, Any]],
    *,
    source_path: str,
    k_values: list[int],
    target_recall: float,
    miss_example_k: int,
    miss_example_limit: int,
) -> dict[str, Any]:
    candidate = _enrich_rows(_candidate_rows(rows))
    retention_by_k = _retention_for_rows(candidate, k_values)
    recommended = _recommended_k(retention_by_k, target_recall=target_recall)
    strategy_counts = Counter(str(row.get("strategy", "unknown")) for row in rows)
    selection_heads = Counter(str(row.get("selection_head", "unknown")) for row in rows)
    return {
        "status": "pass",
        "analysis": "search_decision_trace_retained_set_forensics",
        "source_path": source_path,
        "total_rows": len(rows),
        "candidate_rows": len(candidate),
        "strategy_counts": dict(strategy_counts),
        "selection_head_counts": dict(selection_heads),
        "k_values": k_values,
        "target_recall": target_recall,
        "recommended_min_k_for_target_recall": recommended,
        "caveat": (
            "Winner retention by K is exact from model_ranked_action_ids and full_best_action_id. "
            "Alternative-K selected action and regret cannot be reconstructed exactly without per-action rollout scores."
        ),
        "rank_summary": _rank_summary(candidate),
        "retention_by_k": retention_by_k,
        "phase_summary": _phase_table(candidate, k_values),
        "active_seat_counts": dict(Counter(int(row.get("active_seat", -1)) for row in candidate)),
        "candidate_count_counts": dict(Counter(int(row.get("candidate_count", 0)) for row in candidate)),
        "largest_k_misses": _miss_examples(candidate, k=miss_example_k, limit=miss_example_limit),
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Search Decision Trace Forensics",
        "",
        f"Source: `{report['source_path']}`",
        f"Candidate decisions: `{report['candidate_rows']}`",
        f"Target recall: `{report['target_recall']}`",
        f"Recommended minimum K: `{report['recommended_min_k_for_target_recall']}`",
        "",
        "## Rank Summary",
        "",
    ]
    rank = report["rank_summary"]
    for key in ("mean", "p50", "p75", "p90", "p95", "p99", "max"):
        lines.append(f"- Full-search winner model-rank {key}: `{rank[key]}`")
    lines.extend(["", "## Retention By K", "", "| K | Recall | Misses | Rollout Fraction | K32 Regret On Misses |", "|---:|---:|---:|---:|---:|"])
    for key in sorted(report["retention_by_k"], key=lambda value: int(value)):
        row = report["retention_by_k"][key]
        lines.append(
            "| {k} | {recall:.4f} | {misses} | {fraction:.4f} | {regret} |".format(
                k=row["k"],
                recall=float(row["full_best_retained_rate"]),
                misses=row["full_best_missed_count"],
                fraction=float(row["estimated_non_shadow_rollout_fraction"]),
                regret=row["mean_observed_k32_regret_on_k_misses"],
            )
        )
    lines.extend(["", "## Phase Recall", ""])
    for phase, payload in report["phase_summary"].items():
        fragments = []
        for k in (16, 24, 32, 40, 48, 56, 64):
            if str(k) in payload["retention_by_k"]:
                fragments.append(f"K{k}={payload['retention_by_k'][str(k)]['full_best_retained_rate']}")
        lines.append(f"- {phase} (`{payload['decisions']}` decisions): " + ", ".join(fragments))
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            report["caveat"],
            "",
            "Use this report to choose the next retained width. It is not a substitute for a non-shadow gameplay benchmark at that width.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("--k-values must include at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--k-values must be positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--miss-example-k", type=int, default=32)
    parser.add_argument("--miss-example-limit", type=int, default=20)
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    rows = _load_rows(decisions_path)
    report = build_report(
        rows,
        source_path=str(decisions_path),
        k_values=parse_k_values(args.k_values),
        target_recall=args.target_recall,
        miss_example_k=args.miss_example_k,
        miss_example_limit=args.miss_example_limit,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, Path(args.summary_out))
    print(json.dumps({"status": "pass", "out": str(out_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
