"""Miss-set forensics for CRT serving prefilter experiments.

This module replays a trained checkpoint over a validation shard and explains
which roots miss the K=16 teacher-best retention gate. It is meant to guide the
next architecture/feature iteration, not to train a model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .torch_prefilter_blend_eval import _collect_rows
from .torch_semantic_relation_bias_merit import SEMANTIC_ACTION_FEATURE_NAMES, semantic_action_features
from .torch_public_token_merit import PublicTokenJsonlDataset, _action_immediate_score

WILDLIFE_NAMES = ("bear", "elk", "salmon", "hawk", "fox")


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _player_token(record: dict[str, Any]) -> dict[str, Any]:
    active_seat = int(record.get("active_seat", 0))
    for token in record["public_tokens"]["tokens"]:
        if token.get("token_kind") == "player" and int(token.get("owner_seat", -1)) == active_seat:
            return token
    return {}


def _action_category(action: dict[str, Any]) -> dict[str, Any]:
    species = int(_safe_float(action.get("wildlife_species"), -1.0))
    return {
        "wildlife_species": WILDLIFE_NAMES[species] if 0 <= species < len(WILDLIFE_NAMES) else "none",
        "tile_slot": str(int(_safe_float(action.get("tile_slot", action.get("draft_slot", -1)), -1.0))),
        "wildlife_slot": str(int(_safe_float(action.get("wildlife_slot", action.get("draft_slot", -1)), -1.0))),
        "nature_spend": str(int(_safe_float(action.get("nature_spend", 0.0)))),
        "cleanup_choice": str(action.get("cleanup_choice", "none")),
        "wildlife_present": str(bool(action.get("wildlife_placement_present"))),
    }


def _root_phase_features(record: dict[str, Any]) -> dict[str, float]:
    player = _player_token(record)
    active_tile_count = _safe_float(player.get("tile_count"))
    return {
        "active_tile_count": active_tile_count,
        "active_turns_remaining_est": max(0.0, 20.0 - active_tile_count),
        "active_current_base_score": _safe_float(player.get("current_base_score")),
        "active_current_wildlife_total": _safe_float(player.get("current_wildlife_total")),
        "active_current_habitat_total": _safe_float(player.get("current_habitat_total")),
        "active_nature_tokens": _safe_float(player.get("nature_tokens")),
        "public_token_count": _safe_float(record["public_tokens"].get("token_count")),
        "public_relation_count": _safe_float(record["public_tokens"].get("relation_count")),
    }


def _teacher_features(record: dict[str, Any]) -> dict[str, float]:
    import torch

    q = torch.tensor(record["per_action_Q"], dtype=torch.float32)
    ranked = torch.argsort(q, descending=True)
    best_index = int(ranked[0].item())
    threshold_index = int(ranked[min(15, len(ranked) - 1)].item())
    best_action = record["legal_actions"][best_index]
    semantic = semantic_action_features(record)[best_index]
    features: dict[str, float] = {
        "teacher_best_q": float(q[best_index].item()),
        "teacher_q_spread": float(q.max().item() - q.min().item()),
        "teacher_best_to_16th_margin": float(q[best_index].item() - q[threshold_index].item()),
        "teacher_best_variance": _safe_float(record.get("per_action_Q_variance", [0.0] * len(q))[best_index]),
        "teacher_best_count": _safe_float(record.get("per_action_Q_count", [1.0] * len(q))[best_index]),
        "teacher_best_immediate": _safe_float(best_action.get("immediate_pre_rollout_base_score")),
        "teacher_best_immediate_delta_vs_root": _action_immediate_score(best_action)
        - _root_phase_features(record)["active_current_base_score"],
    }
    for name, value in zip(SEMANTIC_ACTION_FEATURE_NAMES, semantic, strict=True):
        features[f"best_{name}"] = float(value)
    return features


def _record_features(record: dict[str, Any]) -> dict[str, float]:
    features = _root_phase_features(record)
    features.update(_teacher_features(record))
    return features


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _numeric_group_summary(records: list[dict[str, Any]], feature_names: list[str]) -> dict[str, float]:
    return {name: _round(_mean([float(row["features"].get(name, 0.0)) for row in records])) for name in feature_names}


def _feature_deltas(hits: list[dict[str, Any]], misses: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    hit_means = _numeric_group_summary(hits, feature_names)
    miss_means = _numeric_group_summary(misses, feature_names)
    deltas = {
        name: _round(miss_means[name] - hit_means[name])
        for name in feature_names
    }
    ranked = sorted(deltas, key=lambda name: abs(deltas[name]), reverse=True)
    return {
        "hit_means": hit_means,
        "miss_means": miss_means,
        "miss_minus_hit": {name: deltas[name] for name in ranked},
    }


def _categorical_summary(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row["categories"][key]) for row in rows)
    return dict(counter.most_common())


def _miss_category_summary(hits: list[dict[str, Any]], misses: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("wildlife_species", "tile_slot", "wildlife_slot", "nature_spend", "cleanup_choice", "wildlife_present")
    return {
        key: {
            "hit": _categorical_summary(hits, key),
            "miss": _categorical_summary(misses, key),
        }
        for key in keys
    }


def _ranked_misses(rows: list[dict[str, Any]], source: str, *, limit: int) -> list[dict[str, Any]]:
    misses = [row for row in rows if not row["sources"][source]["top16_hit"]]
    misses.sort(
        key=lambda row: (
            row["sources"][source]["top16_oracle_regret"],
            row["sources"][source]["teacher_best_pred_rank"],
            row["features"]["teacher_best_to_16th_margin"],
        ),
        reverse=True,
    )
    selected = []
    for row in misses[:limit]:
        best_action = row["teacher_best_action"]
        selected.append(
            {
                "state_hash": row["state_hash"],
                "teacher_best_action_id": best_action.get("action_id"),
                "species": row["categories"]["wildlife_species"],
                "tile_slot": row["categories"]["tile_slot"],
                "wildlife_slot": row["categories"]["wildlife_slot"],
                "teacher_best_q": _round(row["features"].get("teacher_best_q", 0.0)),
                "best_to_16th_margin": _round(row["features"].get("teacher_best_to_16th_margin", 0.0)),
                "source_rank": row["sources"][source]["teacher_best_pred_rank"],
                "oracle_regret": _round(row["sources"][source]["top16_oracle_regret"]),
                "selected_regret": _round(row["sources"][source]["selected_regret"]),
                "active_tile_count": _round(row["features"].get("active_tile_count", 0.0)),
            }
        )
    return selected


def _source_metrics(rows: list[dict[str, Any]], source: str, *, k: int) -> dict[str, Any]:
    hits = [row for row in rows if row["sources"][source]["top16_hit"]]
    misses = [row for row in rows if not row["sources"][source]["top16_hit"]]
    total = len(rows)
    hit_count = len(hits)
    needed = max(0, math.ceil(0.75 * total) - hit_count)
    feature_names = [
        "active_tile_count",
        "active_turns_remaining_est",
        "active_current_base_score",
        "active_current_wildlife_total",
        "active_current_habitat_total",
        "active_nature_tokens",
        "public_token_count",
        "public_relation_count",
        "teacher_q_spread",
        "teacher_best_to_16th_margin",
        "teacher_best_variance",
        "teacher_best_immediate_delta_vs_root",
        "best_bear_pair_signal",
        "best_elk_best_line_length",
        "best_salmon_component_size",
        "best_hawk_isolated_signal",
        "best_fox_unique_adjacent_species_count",
        "best_public_market_species_count",
        "best_opponent_species_count_gap",
        "best_wildlife_bag_species_count",
        "best_unseen_tile_species_capacity",
    ]
    return {
        "roots": total,
        "k": k,
        "hits": hit_count,
        "misses": len(misses),
        "recall": _round(hit_count / total if total else 0.0),
        "hits_needed_for_0_750": needed,
        "mean_top16_oracle_regret": _round(_mean([row["sources"][source]["top16_oracle_regret"] for row in rows])),
        "mean_miss_teacher_best_pred_rank": _round(_mean([row["sources"][source]["teacher_best_pred_rank"] for row in misses])),
        "mean_miss_oracle_regret": _round(_mean([row["sources"][source]["top16_oracle_regret"] for row in misses])),
        "feature_deltas": _feature_deltas(hits, misses, feature_names),
        "categories": _miss_category_summary(hits, misses),
        "largest_misses": _ranked_misses(rows, source, limit=20),
    }


def _overlap(rows: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    miss_sets = {
        source: {row["state_hash"] for row in rows if not row["sources"][source]["top16_hit"]}
        for source in sources
    }
    pairwise = {}
    for left in sources:
        for right in sources:
            if left >= right:
                continue
            intersection = len(miss_sets[left] & miss_sets[right])
            union = len(miss_sets[left] | miss_sets[right])
            pairwise[f"{left}|{right}"] = {
                "intersection": intersection,
                "union": union,
                "jaccard": _round(intersection / union if union else 0.0),
            }
    consensus_misses = sorted(set.intersection(*(miss_sets[source] for source in sources))) if sources else []
    any_model_hits = {
        source: sorted(miss_sets["mlp"] - miss_sets[source])
        for source in sources
        if source != "mlp" and "mlp" in miss_sets
    }
    return {
        "pairwise_miss_overlap": pairwise,
        "consensus_miss_count": len(consensus_misses),
        "consensus_miss_state_hashes": consensus_misses[:50],
        "mlp_misses_recovered_by_other_sources": {
            source: len(values) for source, values in any_model_hits.items()
        },
    }


def _build_forensic_rows(val_path: Path, checkpoint_path: Path, *, batch_size: int, device_name: str) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    score_rows = _collect_rows(val_path, checkpoint, batch_size=batch_size, device=device)
    records = PublicTokenJsonlDataset(val_path).records
    by_hash = {record["state_hash"]: record for record in records}
    source_names = list(score_rows[0]["source_names"]) if score_rows else []

    forensic_rows = []
    for score_row in score_rows:
        import torch

        record = by_hash[score_row["state_hash"]]
        q = score_row["target_q"]
        teacher_best = int(torch.argmax(q).item())
        best_q = float(q[teacher_best].item())
        features = _record_features(record)
        categories = _action_category(record["legal_actions"][teacher_best])
        source_data = {}
        for source in source_names:
            scores = score_row["sources"][source]
            ranked = torch.argsort(scores, descending=True)
            retained = ranked[: min(16, int(score_row["action_count"]))]
            retained_indices = [int(index) for index in retained.tolist()]
            retained_best_q = float(q[retained].max().item())
            selected = int(ranked[0].item())
            teacher_rank = retained_indices.index(teacher_best) + 1 if teacher_best in retained_indices else int((ranked == teacher_best).nonzero()[0].item()) + 1
            source_data[source] = {
                "top16_hit": teacher_best in retained_indices,
                "top16_oracle_regret": best_q - retained_best_q,
                "teacher_best_pred_rank": teacher_rank,
                "selected_regret": best_q - float(q[selected].item()),
                "selected_action_id": record["legal_actions"][selected]["action_id"],
            }
        forensic_rows.append(
            {
                "state_hash": score_row["state_hash"],
                "features": features,
                "categories": categories,
                "teacher_best_index": teacher_best,
                "teacher_best_action": record["legal_actions"][teacher_best],
                "sources": source_data,
            }
        )
    report = checkpoint["report"]
    return forensic_rows, source_names, {
        "checkpoint_experiment_id": report.get("experiment_id"),
        "checkpoint_decision": report.get("decision"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def run_forensics(
    val_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device_name: str,
    k: int,
    experiment_id: str,
) -> dict[str, Any]:
    rows, sources, runtime = _build_forensic_rows(
        val_path,
        checkpoint_path,
        batch_size=batch_size,
        device_name=device_name,
    )
    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_forensics",
        "experiment_id": experiment_id,
        "val": str(val_path),
        "checkpoint": str(checkpoint_path),
        "source_names": sources,
        "k": k,
        "roots": len(rows),
        "runtime": runtime,
        "sources": {source: _source_metrics(rows, source, k=k) for source in sources},
        "overlap": _overlap(rows, sources),
    }


def _top_delta_lines(metrics: dict[str, Any], *, limit: int = 8) -> list[str]:
    deltas = metrics["feature_deltas"]["miss_minus_hit"]
    return [
        f"- `{name}`: {value:+.4f}"
        for name, value in list(deltas.items())[:limit]
    ]


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    source_priority = [source for source in ("mlp", "action_set", "vanilla", "relation", "residual", "cross") if source in report["sources"]]
    lines = [
        "# CRT Wide-32 R16p20 Prefilter Forensics",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Validation roots: `{report['roots']}`",
        "",
        "## Source Metrics",
        "",
        "| Source | K=16 recall | Misses | Need For 0.750 | Oracle regret | Mean miss rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in report["source_names"]:
        metrics = report["sources"][source]
        lines.append(
            f"| {source} | {metrics['recall']:.4f} | {metrics['misses']} | "
            f"{metrics['hits_needed_for_0_750']} | {metrics['mean_top16_oracle_regret']:.4f} | "
            f"{metrics['mean_miss_teacher_best_pred_rank']:.2f} |"
        )
    lines.extend(["", "## Miss Feature Deltas", ""])
    for source in source_priority[:3]:
        lines.append(f"### {source}")
        lines.extend(_top_delta_lines(report["sources"][source]))
        lines.append("")
    lines.extend(["## Miss Overlap", ""])
    overlap = report["overlap"]
    lines.append(f"- Consensus miss count: `{overlap['consensus_miss_count']}`")
    for source, count in overlap.get("mlp_misses_recovered_by_other_sources", {}).items():
        lines.append(f"- MLP misses recovered by `{source}`: `{count}`")
    lines.extend(["", "## Largest MLP Misses", ""])
    if "mlp" in report["sources"]:
        for row in report["sources"]["mlp"]["largest_misses"][:10]:
            lines.append(
                f"- `{row['state_hash']}` species `{row['species']}`, rank `{row['source_rank']}`, "
                f"regret `{row['oracle_regret']:.4f}`, margin `{row['best_to_16th_margin']:.4f}`, "
                f"tiles `{row['active_tile_count']:.0f}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-prefilter-forensics-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_forensics.json")
    parser.add_argument("--summary-out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_forensics_summary.md")
    args = parser.parse_args()

    result = run_forensics(
        Path(args.val),
        Path(args.checkpoint),
        batch_size=args.batch_size,
        device_name=args.device,
        k=args.k,
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
