#!/usr/bin/env python3
"""Run ADR 0121 conditional-tile generalization forensics."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    HierarchicalFactorCache,
    build_stage_model,
    score_stage_shard,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

EXPERIMENT_ID = "conditional-tile-generalization-forensics-v1"
STAGE = "tile"
WIDTH_BINS = (
    ("le32", 0, 32),
    ("33_64", 33, 64),
    ("65_96", 65, 96),
    ("97_128", 97, 128),
    ("ge129", 129, None),
)


def _host() -> str:
    return socket.gethostname().split(".")[0]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _scientific_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _width_bin(width: int) -> str:
    for name, lower, upper in WIDTH_BINS:
        if width >= lower and (upper is None or width <= upper):
            return name
    raise AssertionError("tile width did not match a frozen bin")


def _query_fingerprints(
    arrays: dict[str, np.ndarray],
) -> tuple[list[bytes], np.ndarray]:
    """Hash exact deployed pointwise inputs and return their labels."""
    offsets = arrays["tile_query_offsets"]
    groups = arrays["tile_query_group"]
    contexts = arrays["tile_query_context"]
    items = arrays["tile_item_features"]
    labels = arrays["tile_item_target"]
    fingerprints: list[bytes] = []
    cursor = 0
    for query_index, (left, right) in enumerate(pairwise(offsets)):
        left = int(left)
        right = int(right)
        prefix = blake3.blake3(
            arrays["group_state"][int(groups[query_index])].tobytes()
            + contexts[query_index].tobytes()
        ).digest()
        for index in range(left, right):
            fingerprints.append(
                blake3.blake3(prefix + items[index].tobytes()).digest(length=16)
            )
            cursor += 1
    if cursor != len(items) or len(fingerprints) != len(labels):
        raise AssertionError("alias audit item coverage drifted")
    return fingerprints, labels


def _alias_split(
    cache: HierarchicalFactorCache,
) -> tuple[dict[bytes, list[int]], dict[str, Any]]:
    entries: dict[bytes, list[int]] = {}
    observations = 0
    positives = 0
    for arrays in cache.iter_shards():
        fingerprints, labels = _query_fingerprints(arrays)
        for fingerprint, target in zip(fingerprints, labels, strict=True):
            value = entries.setdefault(fingerprint, [0, 0])
            value[int(bool(target))] += 1
            observations += 1
            positives += int(bool(target))
    contradictory = {
        fingerprint: counts
        for fingerprint, counts in entries.items()
        if counts[0] and counts[1]
    }
    contradictory_observations = sum(sum(counts) for counts in contradictory.values())
    contradictory_positives = sum(counts[1] for counts in contradictory.values())
    return entries, {
        "observations": observations,
        "positive_observations": positives,
        "unique_fingerprints": len(entries),
        "repeated_fingerprints": sum(sum(counts) > 1 for counts in entries.values()),
        "contradictory_fingerprints": len(contradictory),
        "contradictory_observations": contradictory_observations,
        "contradictory_positive_observations": contradictory_positives,
        "contradictory_observation_fraction": contradictory_observations
        / max(observations, 1),
        "contradictory_positive_fraction": contradictory_positives
        / max(positives, 1),
        "all_items_covered_once": observations == int(cache.manifest["items"][STAGE]),
    }


def run_alias_audit(
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Measure exact pointwise-input label contradictions."""
    started = time.perf_counter()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    train, train_report = _alias_split(train_cache)
    validation, validation_report = _alias_split(validation_cache)
    overlap = set(train) & set(validation)
    overlapping_observations = 0
    contradictory_overlap_observations = 0
    contradictory_overlap_fingerprints = 0
    for fingerprint in overlap:
        train_counts = train[fingerprint]
        validation_counts = validation[fingerprint]
        observations = sum(validation_counts)
        overlapping_observations += observations
        train_labels = {index for index, count in enumerate(train_counts) if count}
        validation_labels = {
            index for index, count in enumerate(validation_counts) if count
        }
        if train_labels != validation_labels:
            contradictory_overlap_fingerprints += 1
            contradictory_overlap_observations += observations
    cross_fraction = contradictory_overlap_observations / max(
        overlapping_observations, 1
    )
    pipeline = {
        "train_items_covered_once": bool(train_report["all_items_covered_once"]),
        "validation_items_covered_once": bool(
            validation_report["all_items_covered_once"]
        ),
        "cache_payloads_distinct": (
            train_cache.manifest["payload_blake3"]
            != validation_cache.manifest["payload_blake3"]
        ),
    }
    material = (
        float(train_report["contradictory_positive_fraction"]) >= 0.01
        or cross_fraction >= 0.01
    )
    scientific = {
        "classification": (
            "observable_label_aliasing_material"
            if material
            else "observable_label_aliasing_not_material"
        ),
        "train": train_report,
        "validation": validation_report,
        "cross_split": {
            "overlap_fingerprints": len(overlap),
            "overlap_validation_observations": overlapping_observations,
            "contradictory_overlap_fingerprints": (
                contradictory_overlap_fingerprints
            ),
            "contradictory_overlap_observations": (
                contradictory_overlap_observations
            ),
            "contradictory_overlap_fraction": cross_fraction,
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": (
            validation_cache.manifest["payload_blake3"]
        ),
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": "exact-observable-aliasing",
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


class _BlockStats:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.square = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.total):
            raise ValueError("distribution block shape drifted")
        self.count += len(matrix)
        self.total += np.sum(matrix, axis=0)
        self.square += np.sum(matrix * matrix, axis=0)
        self.minimum = np.minimum(self.minimum, np.min(matrix, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(matrix, axis=0))

    def finish(self) -> dict[str, np.ndarray | int]:
        mean = self.total / max(self.count, 1)
        variance = np.maximum(self.square / max(self.count, 1) - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "variance": variance,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def _distribution_split(
    cache: HierarchicalFactorCache,
) -> tuple[dict[str, dict[str, np.ndarray | int]], Counter[int]]:
    stats: dict[str, _BlockStats] | None = None
    widths: Counter[int] = Counter()
    for arrays in cache.iter_shards():
        if stats is None:
            stats = {
                "group_state": _BlockStats(arrays["group_state"].shape[1]),
                "query_context": _BlockStats(
                    arrays["tile_query_context"].shape[1]
                ),
                "item_features": _BlockStats(
                    arrays["tile_item_features"].shape[1]
                ),
            }
        stats["group_state"].add(arrays["group_state"])
        stats["query_context"].add(arrays["tile_query_context"])
        stats["item_features"].add(arrays["tile_item_features"])
        offsets = arrays["tile_query_offsets"]
        widths.update(int(right - left) for left, right in pairwise(offsets))
    if stats is None:
        raise ValueError("distribution cache is empty")
    return {name: value.finish() for name, value in stats.items()}, widths


def _js_divergence(left: Counter[int], right: Counter[int]) -> float:
    support = sorted(set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    p = np.asarray([left[value] / max(left_total, 1) for value in support])
    q = np.asarray([right[value] / max(right_total, 1) for value in support])
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    return 0.5 * (kl(p) + kl(q))


def _block_shift(
    train: dict[str, np.ndarray | int],
    validation: dict[str, np.ndarray | int],
    validation_cache: HierarchicalFactorCache,
    block: str,
) -> dict[str, Any]:
    train_mean = np.asarray(train["mean"])
    validation_mean = np.asarray(validation["mean"])
    train_variance = np.asarray(train["variance"])
    validation_variance = np.asarray(validation["variance"])
    pooled = np.sqrt(0.5 * (train_variance + validation_variance))
    active = pooled > 1e-8
    smd = np.zeros_like(pooled)
    smd[active] = np.abs(validation_mean[active] - train_mean[active]) / pooled[
        active
    ]
    train_minimum = np.asarray(train["minimum"])
    train_maximum = np.asarray(train["maximum"])
    outside = 0
    cells = 0
    key = {
        "group_state": "group_state",
        "query_context": "tile_query_context",
        "item_features": "tile_item_features",
    }[block]
    for arrays in validation_cache.iter_shards():
        values = np.asarray(arrays[key])
        outside += int(
            np.sum((values < train_minimum) | (values > train_maximum))
        )
        cells += int(values.size)
    active_smd = smd[active]
    return {
        "train_count": int(train["count"]),
        "validation_count": int(validation["count"]),
        "dimensions": len(smd),
        "active_dimensions": int(np.sum(active)),
        "fraction_active_dimensions_smd_at_least_0_50": float(
            np.mean(active_smd >= 0.5) if len(active_smd) else 0.0
        ),
        "median_absolute_smd": float(
            np.median(active_smd) if len(active_smd) else 0.0
        ),
        "p95_absolute_smd": float(
            np.quantile(active_smd, 0.95) if len(active_smd) else 0.0
        ),
        "maximum_absolute_smd": float(np.max(active_smd, initial=0.0)),
        "validation_outside_train_support_cells": outside,
        "validation_cells": cells,
        "validation_outside_train_support_fraction": outside / max(cells, 1),
        "train_mean": train_mean.tolist(),
        "validation_mean": validation_mean.tolist(),
        "train_variance": train_variance.tolist(),
        "validation_variance": validation_variance.tolist(),
        "train_minimum": train_minimum.tolist(),
        "train_maximum": train_maximum.tolist(),
    }


def run_shift_audit(
    train_cache_root: Path,
    validation_cache_root: Path,
) -> dict[str, Any]:
    """Measure exact train/validation input-distribution shift."""
    started = time.perf_counter()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    train, train_widths = _distribution_split(train_cache)
    validation, validation_widths = _distribution_split(validation_cache)
    blocks = {
        name: _block_shift(
            train[name],
            validation[name],
            validation_cache,
            name,
        )
        for name in ("group_state", "query_context", "item_features")
    }
    width_js = _js_divergence(train_widths, validation_widths)
    material = width_js >= 0.10 or any(
        float(report["fraction_active_dimensions_smd_at_least_0_50"]) >= 0.10
        or float(report["validation_outside_train_support_fraction"]) >= 0.01
        for report in blocks.values()
    )
    pipeline = {
        "train_group_coverage": (
            int(train["group_state"]["count"]) == int(train_cache.manifest["groups"])
        ),
        "validation_group_coverage": (
            int(validation["group_state"]["count"])
            == int(validation_cache.manifest["groups"])
        ),
        "train_query_coverage": (
            int(train["query_context"]["count"])
            == int(train_cache.manifest["queries"][STAGE])
        ),
        "validation_query_coverage": (
            int(validation["query_context"]["count"])
            == int(validation_cache.manifest["queries"][STAGE])
        ),
        "train_item_coverage": (
            int(train["item_features"]["count"])
            == int(train_cache.manifest["items"][STAGE])
        ),
        "validation_item_coverage": (
            int(validation["item_features"]["count"])
            == int(validation_cache.manifest["items"][STAGE])
        ),
        "all_summaries_finite": all(
            math.isfinite(float(report[key]))
            for report in blocks.values()
            for key in (
                "median_absolute_smd",
                "p95_absolute_smd",
                "maximum_absolute_smd",
                "validation_outside_train_support_fraction",
            )
        )
        and math.isfinite(width_js),
    }
    scientific = {
        "classification": (
            "input_covariate_shift_material"
            if material
            else "input_covariate_shift_not_material"
        ),
        "blocks": blocks,
        "query_width": {
            "train_histogram": dict(sorted(train_widths.items())),
            "validation_histogram": dict(sorted(validation_widths.items())),
            "jensen_shannon_divergence": width_js,
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": (
            validation_cache.manifest["payload_blake3"]
        ),
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": "input-distribution-shift",
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _margin_summary(values: list[float], widths: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    by_width = {}
    for name, lower, upper in WIDTH_BINS:
        selected = np.asarray(
            [
                value
                for value, width in zip(values, widths, strict=True)
                if width >= lower and (upper is None or width <= upper)
            ],
            dtype=np.float64,
        )
        by_width[name] = _quantiles(selected)
    return {**_quantiles(array), "by_width": by_width}


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {
            "queries": 0,
            "p10": None,
            "median": None,
            "p90": None,
            "positive_fraction": None,
        }
    return {
        "queries": len(values),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def _score_margins(
    model: Any,
    cache: HierarchicalFactorCache,
) -> dict[str, Any]:
    margins: list[float] = []
    widths: list[int] = []
    queries = 0
    items = 0
    finite = True
    for arrays in cache.iter_shards():
        scores = score_stage_shard(model, arrays, STAGE)
        finite &= bool(np.all(np.isfinite(scores)))
        offsets = arrays["tile_query_offsets"]
        targets = arrays["tile_item_target"]
        for left, right in pairwise(offsets):
            left = int(left)
            right = int(right)
            local_scores = scores[left:right]
            local_targets = targets[left:right]
            queries += 1
            items += right - left
            if not np.any(local_targets) or np.all(local_targets):
                continue
            scale = float(np.std(local_scores))
            raw_margin = float(
                np.min(local_scores[local_targets])
                - np.max(local_scores[~local_targets])
            )
            margins.append(raw_margin / max(scale, 1e-8))
            widths.append(right - left)
    return {
        **_margin_summary(margins, widths),
        "all_scores_finite": finite,
        "all_queries_scored_once": queries == int(cache.manifest["queries"][STAGE]),
        "all_items_scored_once": items == int(cache.manifest["items"][STAGE]),
        "eligible_queries": len(margins),
    }


def run_margin_audit(
    train_cache_root: Path,
    validation_cache_root: Path,
    source_weights: Path,
    extended_weights: Path,
) -> dict[str, Any]:
    """Compare normalized target-boundary margins across frozen checkpoints."""
    started = time.perf_counter()
    configure_mlx_memory()
    train_cache = HierarchicalFactorCache(train_cache_root)
    validation_cache = HierarchicalFactorCache(validation_cache_root)
    reports = {}
    for name, path in (
        ("source_20_epoch", source_weights),
        ("extended_200_epoch", extended_weights),
    ):
        model = build_stage_model(STAGE)
        model.load_weights(str(path))
        mx.eval(model.parameters())
        reports[name] = {
            "weights_blake3": checksum(path),
            "train": _score_margins(model, train_cache),
            "validation": _score_margins(model, validation_cache),
        }
        del model
        mx.clear_cache()
    source = reports["source_20_epoch"]
    extended = reports["extended_200_epoch"]
    train_improvement = float(extended["train"]["median"]) - float(
        source["train"]["median"]
    )
    validation_improvement = float(extended["validation"]["median"]) - float(
        source["validation"]["median"]
    )
    source_gap = float(source["train"]["median"]) - float(
        source["validation"]["median"]
    )
    extended_gap = float(extended["train"]["median"]) - float(
        extended["validation"]["median"]
    )
    gap_expansion = extended_gap - source_gap
    specialized = (
        train_improvement >= 0.50
        and validation_improvement <= 0.10
        and gap_expansion >= 0.50
    )
    pipeline = {
        "source_weights_identity": (
            source["weights_blake3"]
            == "5c13fe87d7b4ac0a8ff9f647f57c69b8d9ab583b3ce2e85e41ee0f3d97e8f514"
        ),
        "extended_weights_identity": (
            extended["weights_blake3"]
            == "7acd245b20bf5a35bb3bcab848f3b4b3014d763058fa803b0b4ae3b17c80205d"
        ),
        "all_scores_finite": all(
            bool(report[split]["all_scores_finite"])
            for report in reports.values()
            for split in ("train", "validation")
        ),
        "all_queries_scored_once": all(
            bool(report[split]["all_queries_scored_once"])
            for report in reports.values()
            for split in ("train", "validation")
        ),
        "all_items_scored_once": all(
            bool(report[split]["all_items_scored_once"])
            for report in reports.values()
            for split in ("train", "validation")
        ),
    }
    scientific = {
        "classification": (
            "late_fit_margin_specialization"
            if specialized
            else "late_fit_margin_specialization_not_proven"
        ),
        "checkpoints": reports,
        "comparison": {
            "train_median_improvement": train_improvement,
            "validation_median_improvement": validation_improvement,
            "source_train_validation_gap": source_gap,
            "extended_train_validation_gap": extended_gap,
            "gap_expansion": gap_expansion,
        },
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "train_cache_payload_blake3": train_cache.manifest["payload_blake3"],
        "validation_cache_payload_blake3": (
            validation_cache.manifest["payload_blake3"]
        ),
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": "normalized-margin-specialization",
        "host": _host(),
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "elapsed_seconds": time.perf_counter() - started,
    }


def combine(alias: dict[str, Any], shift: dict[str, Any], margin: dict[str, Any]) -> dict[str, Any]:
    """Combine independent arms and select the frozen structural successor."""
    arms = {"alias": alias, "shift": shift, "margin": margin}
    pipeline = {
        name: (
            report.get("experiment_id") == EXPERIMENT_ID
            and bool(report["scientific"]["pipeline_passed"])
        )
        for name, report in arms.items()
    }
    classifications = {
        name: report["scientific"]["classification"]
        for name, report in arms.items()
    }
    if not all(pipeline.values()):
        successor = "generalization_forensics_pipeline_invalid"
    elif classifications["alias"] == "observable_label_aliasing_material":
        successor = "query_set_aware_tile_scorer"
    elif classifications["shift"] == "input_covariate_shift_material":
        successor = "distribution_robust_representation"
    elif classifications["margin"] == "late_fit_margin_specialization":
        successor = "structural_regularization"
    else:
        successor = "query_set_aware_complete_action_mechanism"
    scientific = {
        "classifications": classifications,
        "pipeline_gates": pipeline,
        "pipeline_passed": all(pipeline.values()),
        "mechanical_successor_if_adr0120_fails": successor,
        "arms": arms,
        "sealed_test_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
    }


def render_markdown(report: dict[str, Any]) -> str:
    scientific = report["scientific"]
    alias = scientific["arms"]["alias"]["scientific"]
    shift = scientific["arms"]["shift"]["scientific"]
    margin = scientific["arms"]["margin"]["scientific"]
    maximum_smd_fraction = max(
        float(value["fraction_active_dimensions_smd_at_least_0_50"])
        for value in shift["blocks"].values()
    )
    maximum_outside_fraction = max(
        float(value["validation_outside_train_support_fraction"])
        for value in shift["blocks"].values()
    )
    return f"""# Conditional Tile Generalization Forensics V1 Result

Date: 2026-06-16

Experiment ID: `{EXPERIMENT_ID}`

## Classifications

- Exact observable aliasing: `{alias["classification"]}`
- Input distribution shift: `{shift["classification"]}`
- Normalized margin specialization: `{margin["classification"]}`

## Key Measurements

- Train positive mass under contradictory exact fingerprints:
  `{alias["train"]["contradictory_positive_fraction"]:.4%}`
- Contradictory exact cross-split overlap:
  `{alias["cross_split"]["contradictory_overlap_fraction"]:.4%}`
- Tile-query width Jensen-Shannon divergence:
  `{shift["query_width"]["jensen_shannon_divergence"]:.6f}`
- Largest block fraction above absolute SMD 0.50:
  `{maximum_smd_fraction:.4%}`
- Largest validation outside-support cell fraction:
  `{maximum_outside_fraction:.4%}`
- Train median margin improvement:
  `{margin["comparison"]["train_median_improvement"]:+.4f}`
- Validation median margin improvement:
  `{margin["comparison"]["validation_median_improvement"]:+.4f}`
- Train-validation gap expansion:
  `{margin["comparison"]["gap_expansion"]:+.4f}`

## Decision

If ADR 0120 fails, the frozen successor is
`{scientific["mechanical_successor_if_adr0120_fails"]}`.

Combined scientific BLAKE3:
`{report["scientific_blake3"]}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("alias", "shift"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--train-cache", type=Path, required=True)
        subparser.add_argument("--validation-cache", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
    margin = subparsers.add_parser("margin")
    margin.add_argument("--train-cache", type=Path, required=True)
    margin.add_argument("--validation-cache", type=Path, required=True)
    margin.add_argument("--source-weights", type=Path, required=True)
    margin.add_argument("--extended-weights", type=Path, required=True)
    margin.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--alias", type=Path, required=True)
    combined.add_argument("--shift", type=Path, required=True)
    combined.add_argument("--margin", type=Path, required=True)
    combined.add_argument("--output", type=Path, required=True)
    combined.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "alias":
        report = run_alias_audit(args.train_cache, args.validation_cache)
    elif args.command == "shift":
        report = run_shift_audit(args.train_cache, args.validation_cache)
    elif args.command == "margin":
        report = run_margin_audit(
            args.train_cache,
            args.validation_cache,
            args.source_weights,
            args.extended_weights,
        )
    else:
        report = combine(
            json.loads(args.alias.read_text()),
            json.loads(args.shift.read_text()),
            json.loads(args.margin.read_text()),
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report))
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "arm": report.get("arm", "combined"),
                "scientific_blake3": report["scientific_blake3"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
