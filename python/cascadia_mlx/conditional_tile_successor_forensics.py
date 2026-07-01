"""Open-data successor forensics for the conditional tile retrieval program."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGE_WIDTHS,
    STAGES,
    HierarchicalFactorCache,
    _group_target,
    _host,
    _resource_usage,
    _scientific_blake3,
    _SelectionAccumulator,
    checksum,
    configure_mlx_memory,
    load_stage_model,
    score_stage_shard,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)

EXPERIMENT_ID = "conditional-tile-successor-forensics-v1"
WIDTH_BUCKETS = (
    "within_budget",
    "width_33_64",
    "width_65_96",
    "width_97_128",
    "width_129_plus",
)
SELECTOR_METHODS = (
    "rank_log_sum",
    "rank_percentile_sum",
    "rank_worst_percentile",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _width_bucket(width: int) -> str:
    if width <= STAGE_WIDTHS["tile"]:
        return "within_budget"
    if width <= 64:
        return "width_33_64"
    if width <= 96:
        return "width_65_96"
    if width <= 128:
        return "width_97_128"
    return "width_129_plus"


def _empty_sampling_bucket() -> dict[str, int]:
    return {
        "queries": 0,
        "target_factors": 0,
        "target_hits": 0,
        "target_misses": 0,
        "exact_queries": 0,
    }


def _sampling_split(
    *,
    cache: HierarchicalFactorCache,
    weights: Path,
) -> dict[str, Any]:
    model = load_stage_model("tile", weights)
    buckets = {name: _empty_sampling_bucket() for name in WIDTH_BUCKETS}
    phases = {f"phase_{phase}": _empty_sampling_bucket() for phase in range(3)}
    overall = _empty_sampling_bucket()
    finite = True
    items = 0
    for arrays in cache.iter_shards():
        scores = score_stage_shard(model, arrays, "tile")
        finite &= bool(np.all(np.isfinite(scores)))
        items += len(scores)
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        groups = arrays["tile_query_group"]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            left = int(left)
            right = int(right)
            width = right - left
            selected = sorted(
                range(left, right),
                key=lambda index: (-float(scores[index]), index),
            )[: min(STAGE_WIDTHS["tile"], width)]
            quota = int(np.sum(targets[left:right]))
            hits = int(np.sum(targets[selected]))
            values = {
                "queries": 1,
                "target_factors": quota,
                "target_hits": hits,
                "target_misses": quota - hits,
                "exact_queries": int(hits == quota),
            }
            for destination in (
                overall,
                buckets[_width_bucket(width)],
                phases[f"phase_{int(arrays['phase'][groups[query_index]])}"],
            ):
                for key, value in values.items():
                    destination[key] += value

    def finalize(
        values: Mapping[str, int],
        *,
        totals: Mapping[str, int],
    ) -> dict[str, float | int]:
        queries = int(values["queries"])
        targets = int(values["target_factors"])
        misses = int(values["target_misses"])
        query_share = queries / max(int(totals["queries"]), 1)
        target_share = targets / max(int(totals["target_factors"]), 1)
        miss_share = misses / max(int(totals["target_misses"]), 1)
        return {
            **values,
            "target_factor_recall": int(values["target_hits"]) / max(targets, 1),
            "exact_query_fraction": int(values["exact_queries"]) / max(queries, 1),
            "query_share": query_share,
            "target_share": target_share,
            "miss_share": miss_share,
            "target_mass_to_query_share": target_share / max(query_share, 1e-12),
            "miss_mass_to_query_share": miss_share / max(query_share, 1e-12),
        }

    return {
        "queries": overall["queries"],
        "items": items,
        "all_queries_scored_once": overall["queries"] == int(cache.manifest["queries"]["tile"]),
        "all_items_scored_once": items == int(cache.manifest["items"]["tile"]),
        "all_scores_finite": finite,
        "overall": finalize(overall, totals=overall),
        "width": {name: finalize(values, totals=overall) for name, values in buckets.items()},
        "phase": {name: finalize(values, totals=overall) for name, values in phases.items()},
    }


def classify_sampling_mass(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Select sampling mismatch only when the same width stratum replicates."""
    matching: list[str] = []
    for name in WIDTH_BUCKETS:
        train_bucket = train["width"][name]
        validation_bucket = validation["width"][name]
        if (
            float(train_bucket["miss_share"]) >= 0.10
            and float(validation_bucket["miss_share"]) >= 0.10
            and float(train_bucket["miss_mass_to_query_share"]) >= 1.50
            and float(validation_bucket["miss_mass_to_query_share"]) >= 1.50
            and float(train_bucket["target_mass_to_query_share"]) >= 1.25
        ):
            matching.append(name)
    classification = (
        "target_mass_sampling_mismatch" if matching else "uniform_query_sampling_not_explanatory"
    )
    return classification, matching


def run_sampling_audit(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    tile_weights: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    train = _sampling_split(cache=train_cache, weights=tile_weights)
    validation = _sampling_split(cache=validation_cache, weights=tile_weights)
    classification, matching = classify_sampling_mass(train, validation)
    scientific = {
        "arm": "sampling-mass",
        "tile_weights_blake3": checksum(tile_weights),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation_cache.manifest["payload_blake3"],
        "train": train,
        "validation": validation,
        "matching_width_strata": matching,
        "classification": classification,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _report(scientific, allocator=allocator, started=started)


def _score_scale_split(
    *,
    cache: HierarchicalFactorCache,
    weights: Mapping[str, Path],
) -> dict[str, Any]:
    models = {stage: load_stage_model(stage, weights[stage]) for stage in STAGES}
    stage_values: dict[str, dict[str, list[float] | int | bool]] = {
        stage: {
            "query_means": [],
            "query_standard_deviations": [],
            "query_top_scores": [],
            "query_ranges": [],
            "queries": 0,
            "items": 0,
            "finite": True,
        }
        for stage in STAGES
    }
    for arrays in cache.iter_shards():
        for stage in STAGES:
            scores = score_stage_shard(models[stage], arrays, stage)
            values = stage_values[stage]
            values["items"] = int(values["items"]) + len(scores)
            values["finite"] = bool(values["finite"]) and bool(np.all(np.isfinite(scores)))
            offsets = arrays[f"{stage}_query_offsets"]
            for left, right in pairwise(offsets):
                query_scores = scores[int(left) : int(right)].astype(
                    np.float64,
                    copy=False,
                )
                values["queries"] = int(values["queries"]) + 1
                values["query_means"].append(float(np.mean(query_scores)))
                values["query_standard_deviations"].append(float(np.std(query_scores)))
                values["query_top_scores"].append(float(np.max(query_scores)))
                values["query_ranges"].append(float(np.max(query_scores) - np.min(query_scores)))

    def summary(stage: str) -> dict[str, Any]:
        values = stage_values[stage]

        def distribution(name: str) -> dict[str, float]:
            array = np.asarray(values[name], dtype=np.float64)
            return {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "p10": float(np.quantile(array, 0.10)),
                "p90": float(np.quantile(array, 0.90)),
            }

        return {
            "queries": values["queries"],
            "items": values["items"],
            "all_queries_scored_once": int(values["queries"])
            == int(cache.manifest["queries"][stage]),
            "all_items_scored_once": int(values["items"]) == int(cache.manifest["items"][stage]),
            "all_scores_finite": values["finite"],
            "query_mean": distribution("query_means"),
            "query_standard_deviation": distribution("query_standard_deviations"),
            "query_top_score": distribution("query_top_scores"),
            "query_range": distribution("query_ranges"),
        }

    return {stage: summary(stage) for stage in STAGES}


def classify_score_scale(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    ratios: dict[str, float] = {}
    replicated = True
    for metric in ("query_standard_deviation", "query_range"):
        for split_name, split in (("train", train), ("validation", validation)):
            values = [max(float(split[stage][metric]["median"]), 1e-12) for stage in STAGES]
            ratio = max(values) / min(values)
            ratios[f"{split_name}_{metric}_median_ratio"] = ratio
        replicated &= (
            ratios[f"train_{metric}_median_ratio"] >= 4.0
            and ratios[f"validation_{metric}_median_ratio"] >= 4.0
        )
    return (
        (
            "cross_stage_score_scale_mismatch"
            if replicated
            else "cross_stage_score_scale_not_dominant"
        ),
        ratios,
    )


def run_score_scale_audit(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
    weights: Mapping[str, Path],
) -> dict[str, Any]:
    started = time.perf_counter()
    allocator = configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    train = _score_scale_split(cache=train_cache, weights=weights)
    validation = _score_scale_split(cache=validation_cache, weights=weights)
    classification, ratios = classify_score_scale(train, validation)
    scientific = {
        "arm": "score-scale",
        "weights_blake3": {stage: checksum(weights[stage]) for stage in STAGES},
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation_cache.manifest["payload_blake3"],
        "train": train,
        "validation": validation,
        "scale_ratios": ratios,
        "classification": classification,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _report(scientific, allocator=allocator, started=started)


def _rank_selector_scores(
    *,
    ranks: np.ndarray,
    rank_mask: np.ndarray,
    offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {method: np.full(len(ranks), -1e9, dtype=np.float32) for method in SELECTOR_METHODS}
    for left, right in pairwise(offsets):
        left = int(left)
        right = int(right)
        valid = np.flatnonzero(rank_mask[left:right]) + left
        if not len(valid):
            continue
        ordered = sorted(
            valid,
            key=lambda index: (float(ranks[index]), int(index)),
        )
        denominator = max(len(ordered) - 1, 1)
        for position, index in enumerate(ordered):
            rank = max(float(ranks[index]), 0.0)
            percentile = 1.0 - position / denominator
            result["rank_log_sum"][index] = -math.log1p(rank)
            result["rank_percentile_sum"][index] = percentile
            result["rank_worst_percentile"][index] = percentile
    return result


@dataclass
class _SelectorSlices:
    overall: _SelectionAccumulator = field(default_factory=_SelectionAccumulator)
    phases: dict[int, _SelectionAccumulator] = field(
        default_factory=lambda: {phase: _SelectionAccumulator() for phase in range(3)}
    )
    subsets: dict[str, _SelectionAccumulator] = field(
        default_factory=lambda: {
            "nature_token_available": _SelectionAccumulator(),
            "independent_draft_winner": _SelectionAccumulator(),
        }
    )

    def report(self) -> dict[str, Any]:
        return {
            "overall": self.overall.report(),
            "phase": {
                {0: "early", 1: "middle", 2: "late"}[phase]: values.report()
                for phase, values in self.phases.items()
            },
            "subsets": {name: values.report() for name, values in self.subsets.items()},
        }


def _factor_selector_split(cache: HierarchicalFactorCache) -> dict[str, Any]:
    accumulators = {method: _SelectorSlices() for method in SELECTOR_METHODS}
    groups = 0
    candidates = 0
    finite = True
    proposal_counts: list[int] = []
    for arrays in cache.iter_shards():
        stage_scores = {
            stage: _rank_selector_scores(
                ranks=arrays[f"{stage}_item_rank"],
                rank_mask=arrays[f"{stage}_item_rank_mask"],
                offsets=arrays[f"{stage}_query_offsets"],
            )
            for stage in STAGES
        }
        action_offsets = arrays["group_action_offsets"]
        for group_index, (left, right) in enumerate(pairwise(action_offsets)):
            left = int(left)
            right = int(right)
            count = right - left
            flags = arrays["action_source_flags"][left:right]
            frontier = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
            maps = {stage: arrays[f"{stage}_action_item"][left:right] for stage in STAGES}
            eligible = ~frontier
            passing = eligible.copy()
            for stage in STAGES:
                mapping = maps[stage]
                valid = mapping >= 0
                passing &= valid
                passing[valid] &= arrays[f"{stage}_item_target"][mapping[valid]]
            proposal_indices = np.flatnonzero(frontier | passing).astype(np.int32)
            proposal_counts.append(len(proposal_indices))
            ranks = arrays["action_expected_rank"][left:right]
            rank_mask = arrays["action_expected_rank_mask"][left:right]
            target = _group_target(
                expected_rank=ranks,
                expected_rank_mask=rank_mask,
                source_flags=flags,
                action_hash=arrays["action_hash"][left:right],
            )
            winner = int(arrays["selected_index"][group_index])
            kwargs = {
                "target": target,
                "source_flags": flags,
                "winner": winner,
                "r4800_mean": arrays["action_r4800_mean"][left:right],
                "r4800_mask": arrays["action_r4800_mask"][left:right],
            }
            for method in SELECTOR_METHODS:
                combined = np.zeros(count, dtype=np.float32)
                valid_components = np.zeros(count, dtype=np.int8)
                if method == "rank_worst_percentile":
                    combined.fill(np.inf)
                for stage in STAGES:
                    mapping = maps[stage]
                    valid = mapping >= 0
                    values = stage_scores[stage][method][mapping[valid]]
                    finite &= bool(np.all(np.isfinite(values)))
                    if method == "rank_worst_percentile":
                        combined[valid] = np.minimum(combined[valid], values)
                    else:
                        combined[valid] += values
                    valid_components[valid] += 1
                combined[valid_components != len(STAGES)] = -1e9
                local = frontier_anchored_retained_indices(
                    scores=combined[proposal_indices],
                    source_flags=flags[proposal_indices],
                    action_hashes=arrays["action_hash"][left:right][proposal_indices],
                )
                retained = proposal_indices[local]
                slices = accumulators[method]
                slices.overall.add(retained=retained, **kwargs)
                phase = int(arrays["phase"][group_index])
                slices.phases[phase].add(retained=retained, **kwargs)
                if int(arrays["nature_tokens"][group_index]) > 0:
                    slices.subsets["nature_token_available"].add(
                        retained=retained,
                        **kwargs,
                    )
                if int(arrays["action_draft_kind"][left + winner]) == 1:
                    slices.subsets["independent_draft_winner"].add(
                        retained=retained,
                        **kwargs,
                    )
            groups += 1
            candidates += count
    proposal_array = np.asarray(proposal_counts, dtype=np.float64)
    return {
        "groups": groups,
        "candidates": candidates,
        "all_groups_scored_once": groups == cache.group_count,
        "all_candidates_scored_once": candidates == cache.candidate_count,
        "all_scores_finite": finite,
        "mean_oracle_proposal_count": float(np.mean(proposal_array)),
        "maximum_oracle_proposal_count": int(np.max(proposal_array)),
        "methods": {method: values.report() for method, values in accumulators.items()},
    }


def _selector_key(report: Mapping[str, Any]) -> tuple[float, ...]:
    overall = report["overall"]
    return (
        float(overall["target_positive_recall"]),
        float(overall["r4800_winner_retention"]),
        -float(overall["mean_retained_r4800_regret"]),
        float(overall["target_set_exact_fraction"]),
    )


def selector_gates(
    report: Mapping[str, Any],
) -> dict[str, bool]:
    overall = report["overall"]
    gates = {
        "target_recall_above_0_98": float(overall["target_positive_recall"]) > 0.98,
        "winner_retention_above_0_98": float(overall["r4800_winner_retention"]) > 0.98,
        "mean_regret_below_0_15": float(overall["mean_retained_r4800_regret"]) < 0.15,
    }
    for name, values in report["phase"].items():
        gates[f"{name}_winner_retention_at_least_0_97"] = (
            float(values["r4800_winner_retention"]) >= 0.97
        )
        gates[f"{name}_regret_below_0_20"] = float(values["mean_retained_r4800_regret"]) < 0.20
    for name, values in report["subsets"].items():
        gates[f"{name}_winner_retention_at_least_0_95"] = (
            float(values["r4800_winner_retention"]) >= 0.95
        )
        gates[f"{name}_regret_below_0_25"] = float(values["mean_retained_r4800_regret"]) < 0.25
    return gates


def run_factor_selector_audit(
    *,
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    train = _factor_selector_split(train_cache)
    selected_method = max(
        SELECTOR_METHODS,
        key=lambda method: _selector_key(train["methods"][method]),
    )
    validation = _factor_selector_split(validation_cache)
    train_gates = selector_gates(train["methods"][selected_method])
    validation_gates = selector_gates(validation["methods"][selected_method])
    pipeline_passed = all(
        (
            train["all_groups_scored_once"],
            train["all_candidates_scored_once"],
            train["all_scores_finite"],
            validation["all_groups_scored_once"],
            validation["all_candidates_scored_once"],
            validation["all_scores_finite"],
        )
    )
    classification = (
        "factor_selector_audit_invalid"
        if not pipeline_passed
        else "fixed_factor_selector_sufficient"
        if all(train_gates.values()) and all(validation_gates.values())
        else "complete_action_selector_required"
    )
    scientific = {
        "arm": "factor-selector-ceiling",
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": validation_cache.manifest["payload_blake3"],
        "selection_uses_train_only": True,
        "selected_method": selected_method,
        "train": train,
        "validation": validation,
        "train_gates": train_gates,
        "validation_gates": validation_gates,
        "pipeline_passed": pipeline_passed,
        "classification": classification,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _report(scientific, allocator=None, started=started)


def _report(
    scientific: dict[str, Any],
    *,
    allocator: dict[str, Any] | None,
    started: float,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **_resource_usage(),
        },
    }
    if allocator is not None:
        report["execution"]["mlx_allocator"] = allocator
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cache_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--train-cache", type=Path, required=True)
        command.add_argument("--validation-cache", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)

    sampling = subparsers.add_parser("sampling-mass")
    add_cache_arguments(sampling)
    sampling.add_argument("--tile-weights", type=Path, required=True)

    scale = subparsers.add_parser("score-scale")
    add_cache_arguments(scale)
    for stage in STAGES:
        scale.add_argument(f"--{stage}-weights", type=Path, required=True)

    selector = subparsers.add_parser("factor-selector-ceiling")
    add_cache_arguments(selector)
    args = parser.parse_args()

    if args.command == "sampling-mass":
        report = run_sampling_audit(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            tile_weights=args.tile_weights,
        )
    elif args.command == "score-scale":
        report = run_score_scale_audit(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            weights={stage: getattr(args, f"{stage}_weights") for stage in STAGES},
        )
    else:
        report = run_factor_selector_audit(
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
        )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
