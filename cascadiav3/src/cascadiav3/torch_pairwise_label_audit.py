"""Audit whether search tensor shards support pairwise action supervision.

Only actions with real search targets (``q_valid``) enter pair labels.  The
report separates raw score margins from variance-aware confidence because a
large completed-Q difference based on one visit is not a reliable comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .expert_tensor_shards import ExpertTensorShard, SHARD_VERSION_V2, SHARD_VERSION_V3


MARGIN_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)
SNR_THRESHOLDS = (1.0, 1.96, 3.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


@dataclass
class _Reservoir:
    limit: int
    seed: int
    values: list[float] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        replacement = self._rng.randrange(self.seen)
        if replacement < self.limit:
            self.values[replacement] = value


def _comparison_snr(
    margin: float,
    left_variance: float,
    left_count: float,
    right_variance: float,
    right_count: float,
) -> float | None:
    # A sample variance of zero at count one is not evidence of certainty.
    if left_count < 2.0 or right_count < 2.0:
        return None
    standard_error_sq = left_variance / left_count + right_variance / right_count
    if standard_error_sq < 0.0 or not math.isfinite(standard_error_sq):
        return None
    if standard_error_sq == 0.0:
        return math.inf if margin > 0.0 else 0.0
    return margin / math.sqrt(standard_error_sq)


def summarize_pairwise_examples(
    examples: Iterable[dict[str, Any]],
    *,
    reservoir_size: int = 1_000_000,
) -> dict[str, Any]:
    import numpy as np

    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    margin_sample = _Reservoir(reservoir_size, seed=0)
    snr_sample = _Reservoir(reservoir_size, seed=1)
    valid_actions_per_root: list[float] = []
    pairs_per_root: list[float] = []
    top_two_margins: list[float] = []
    top_two_snrs: list[float] = []
    top_two_zero_se_non_ties = 0
    roots = 0
    actions = 0
    valid_actions = 0
    eligible_roots = 0
    exact_like_roots = 0
    exact_endgame_roots = 0
    selected_is_q_best = 0
    pair_count = 0
    tied_pairs = 0
    confidence_evaluable_pairs = 0
    zero_se_confident_pairs = 0
    margin_threshold_counts = {threshold: 0 for threshold in MARGIN_THRESHOLDS}
    snr_threshold_counts = {threshold: 0 for threshold in SNR_THRESHOLDS}
    roots_with_margin = {threshold: 0 for threshold in MARGIN_THRESHOLDS}

    for example in examples:
        q = np.asarray(example["target_q"], dtype=np.float64)
        q_valid = np.asarray(example["q_valid"], dtype=bool)
        variance = np.asarray(example["q_variance"], dtype=np.float64)
        count = np.asarray(example["q_count"], dtype=np.float64)
        visits = np.asarray(example.get("visits", np.zeros_like(q)), dtype=np.float64)
        if not (q.ndim == q_valid.ndim == variance.ndim == count.ndim == visits.ndim == 1):
            raise ValueError("pairwise audit arrays must be rank one")
        if len({q.size, q_valid.size, variance.size, count.size, visits.size}) != 1:
            raise ValueError("pairwise audit arrays must have equal lengths")
        if np.any(variance < 0.0) or np.any(count < 0.0):
            raise ValueError("q variance/count values must be nonnegative")
        valid = np.flatnonzero(q_valid & np.isfinite(q) & np.isfinite(variance) & (count > 0.0))
        roots += 1
        exact_endgame_roots += int(bool(example.get("exact_endgame", False)))
        actions += int(q.size)
        valid_actions += int(valid.size)
        valid_actions_per_root.append(float(valid.size))
        if valid.size >= 2:
            eligible_roots += 1
        improved = example.get("improved_policy")
        if visits.sum() == 0.0 and improved is not None:
            improved_array = np.asarray(improved, dtype=np.float64)
            if improved_array.size == q.size and np.count_nonzero(improved_array > 0.0) == 1:
                exact_like_roots += 1
        selected = int(example["selected_action_index"])
        if valid.size and selected in valid:
            best_q = float(q[valid].max())
            if math.isclose(float(q[selected]), best_q, rel_tol=0.0, abs_tol=1.0e-9):
                selected_is_q_best += 1

        root_pairs = 0
        root_margin_flags = {threshold: False for threshold in MARGIN_THRESHOLDS}
        for left_position, left in enumerate(valid):
            for right in valid[left_position + 1 :]:
                margin = abs(float(q[left]) - float(q[right]))
                pair_count += 1
                root_pairs += 1
                margin_sample.add(margin)
                if margin <= 1.0e-12:
                    tied_pairs += 1
                for threshold in MARGIN_THRESHOLDS:
                    if margin >= threshold:
                        margin_threshold_counts[threshold] += 1
                        root_margin_flags[threshold] = True
                snr = _comparison_snr(
                    margin,
                    float(variance[left]),
                    float(count[left]),
                    float(variance[right]),
                    float(count[right]),
                )
                if snr is None:
                    continue
                confidence_evaluable_pairs += 1
                if math.isinf(snr):
                    zero_se_confident_pairs += 1
                else:
                    snr_sample.add(snr)
                for threshold in SNR_THRESHOLDS:
                    if snr >= threshold:
                        snr_threshold_counts[threshold] += 1
        pairs_per_root.append(float(root_pairs))
        for threshold, present in root_margin_flags.items():
            roots_with_margin[threshold] += int(present)

        if valid.size >= 2:
            ranked = valid[np.argsort(-q[valid], kind="stable")]
            best, second = int(ranked[0]), int(ranked[1])
            margin = float(q[best] - q[second])
            top_two_margins.append(margin)
            snr = _comparison_snr(
                margin,
                float(variance[best]),
                float(count[best]),
                float(variance[second]),
                float(count[second]),
            )
            if snr is not None:
                if math.isinf(snr):
                    top_two_zero_se_non_ties += 1
                else:
                    top_two_snrs.append(snr)

    if roots == 0:
        raise ValueError("pairwise audit requires at least one record")

    def fraction(count_value: int, denominator: int) -> float | None:
        return count_value / denominator if denominator else None

    projected_pairs = pair_count / roots * 100_000.0
    return {
        "record_count": roots,
        "action_count": actions,
        "valid_action_count": valid_actions,
        "roots_with_at_least_two_valid_actions": eligible_roots,
        "eligible_root_fraction": eligible_roots / roots,
        "exact_like_root_count": exact_like_roots,
        "exact_endgame_root_count": exact_endgame_roots,
        "selected_is_q_best_count": selected_is_q_best,
        "selected_is_q_best_fraction": selected_is_q_best / roots,
        "valid_actions_per_root": _distribution(valid_actions_per_root),
        "pairs_per_root": _distribution(pairs_per_root),
        "pair_count": pair_count,
        "tied_pair_count": tied_pairs,
        "tied_pair_fraction": fraction(tied_pairs, pair_count),
        "absolute_margin": {
            "sample": _distribution(margin_sample.values),
            "sample_size": len(margin_sample.values),
            "population_seen": margin_sample.seen,
            "fraction_at_least": {
                str(threshold): fraction(margin_threshold_counts[threshold], pair_count)
                for threshold in MARGIN_THRESHOLDS
            },
            "root_fraction_with_at_least_one": {
                str(threshold): roots_with_margin[threshold] / roots
                for threshold in MARGIN_THRESHOLDS
            },
        },
        "variance_aware_confidence": {
            "evaluable_pair_count": confidence_evaluable_pairs,
            "evaluable_pair_fraction": fraction(confidence_evaluable_pairs, pair_count),
            "zero_standard_error_non_tie_count": zero_se_confident_pairs,
            "finite_snr_sample": _distribution(snr_sample.values),
            "finite_snr_sample_size": len(snr_sample.values),
            "finite_snr_population_seen": snr_sample.seen,
            "fraction_at_least_among_evaluable": {
                str(threshold): fraction(
                    snr_threshold_counts[threshold], confidence_evaluable_pairs
                )
                for threshold in SNR_THRESHOLDS
            },
        },
        "top_two_margin": _distribution(top_two_margins),
        "top_two_snr": _distribution(top_two_snrs),
        "top_two_zero_standard_error_non_tie_count": top_two_zero_se_non_ties,
        "projected_pair_count_per_100k_roots": projected_pairs,
        "label_contract": {
            "pairs_use_only_q_valid_actions": True,
            "confidence_requires_two_or_more_samples_per_action": True,
            "standard_error_assumption": "independent_action_estimates_conservative_without_covariance",
            "recommended_training": "antisymmetric pair loss weighted by margin confidence; exclude invalid and one-sample variance claims",
            "exact_endgame_is_explicit_in_v3": True,
        },
    }


def audit_shards(paths: list[Path], *, reservoir_size: int = 1_000_000) -> dict[str, Any]:
    shards: list[ExpertTensorShard] = []
    try:
        for path in paths:
            shard = ExpertTensorShard(path)
            if shard.version not in {SHARD_VERSION_V2, SHARD_VERSION_V3}:
                raise ValueError(f"pairwise audit requires v2+ Gumbel targets: {path}")
            shards.append(shard)
        summary = summarize_pairwise_examples(
            (shard.example(index) for shard in shards for index in range(len(shard))),
            reservoir_size=reservoir_size,
        )
        return {
            "status": "pass",
            "schema_id": "cascadiav3.pairwise_label_audit.v1",
            "scientific_eligibility": "training_label_feasibility_only",
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "version": shard.version,
                    "record_count": len(shard),
                    "metadata": shard.metadata,
                }
                for path, shard in zip(paths, shards)
            ],
            "summary": summary,
        }
    finally:
        for shard in shards:
            shard.close()


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    margin = summary["absolute_margin"]
    confidence = summary["variance_aware_confidence"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    def decimal(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        "# Pairwise Label Audit",
        "",
        f"Records: `{summary['record_count']}`",
        f"Valid pairs: `{summary['pair_count']}`",
        f"Eligible roots: `{percent(summary['eligible_root_fraction'])}`",
        f"Mean valid actions/root: `{decimal(summary['valid_actions_per_root']['mean'])}`",
        f"Pairs with margin >= 0.5: `{percent(margin['fraction_at_least']['0.5'])}`",
        f"Variance-evaluable pairs: `{percent(confidence['evaluable_pair_fraction'])}`",
        f"Evaluable pairs with SNR >= 1.96: `{percent(confidence['fraction_at_least_among_evaluable']['1.96'])}`",
        f"Projected pairs / 100k roots: `{summary['projected_pair_count_per_100k_roots']:.0f}`",
        "",
        "Engineering feasibility only; this is not gameplay or promotion evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reservoir-size", type=int, default=1_000_000)
    args = parser.parse_args()
    report = audit_shards(args.tensor, reservoir_size=args.reservoir_size)
    report["source_revision"] = args.source_revision
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out is not None:
        write_markdown(report, args.summary_out)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
