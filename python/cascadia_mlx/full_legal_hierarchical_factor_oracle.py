"""ADR 0114 hierarchical complete-action factor oracle."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    _system_swap_used_bytes,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_calibrated_adamw import (
    _report,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    build_expected_rank_target_mask,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    Scale16ExpectedRankDataset,
)

EXPERIMENT_ID = "full-legal-hierarchical-factor-oracle-v1"
FROZEN_ADR0113_BLAKE3 = (
    "c2d373d58cb36fc3e876854736183c6580638460a98381dbd17240001d1d74d1"
)
ARMS = (
    "conditional-compact",
    "conditional-balanced",
    "conditional-wide",
    "independent-wide",
)
ARM_WIDTHS = {
    "conditional-compact": (4, 8, 2),
    "conditional-balanced": (8, 16, 4),
    "conditional-wide": (16, 32, 8),
    "independent-wide": (16, 32, 8),
}
DEFAULT_EVIDENCE_PATH = Path(
    "artifacts/experiments/"
    "complete-action-frontier-local-geometry-balanced-target-control-v1/"
    "reports/combined.json"
)


@dataclass(frozen=True)
class FactorRows:
    """Exact action factor identities for one decision group."""

    draft: tuple[bytes, ...]
    tile: tuple[bytes, ...]
    wildlife: tuple[bytes, ...]


def factor_rows(action_features: np.ndarray) -> FactorRows:
    """Partition exact decoded action-choice bytes."""
    actions = np.asarray(action_features, dtype=np.float32)
    draft_values = np.concatenate(
        [actions[:, :34], actions[:, 45:128]],
        axis=-1,
    )
    return FactorRows(
        draft=tuple(
            np.ascontiguousarray(row).tobytes()
            for row in draft_values
        ),
        tile=tuple(
            np.ascontiguousarray(row).tobytes()
            for row in actions[:, 34:42]
        ),
        wildlife=tuple(
            np.ascontiguousarray(row).tobytes()
            for row in actions[:, 42:45]
        ),
    )


def _rank_keys(
    indices: list[int],
    keys: tuple[bytes, ...],
    ranks: np.ndarray,
    rank_mask: np.ndarray,
    width: int,
) -> set[bytes]:
    best: dict[bytes, float] = {}
    for index in indices:
        if not rank_mask[index]:
            continue
        key = keys[index]
        best[key] = min(best.get(key, np.inf), float(ranks[index]))
    return {
        key
        for key, _rank in sorted(
            best.items(),
            key=lambda item: (item[1], item[0]),
        )[:width]
    }


def oracle_candidate_indices(
    factors: FactorRows,
    ranks: np.ndarray,
    rank_mask: np.ndarray,
    source_flags: np.ndarray,
    *,
    arm: str,
) -> np.ndarray:
    """Apply one frozen hierarchical factor budget."""
    draft_width, tile_width, wildlife_width = ARM_WIDTHS[arm]
    count = len(ranks)
    indices = list(range(count))
    frontier = (
        source_flags.astype(np.int64)
        & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = [index for index in indices if not frontier[index]]
    selected_drafts = _rank_keys(
        eligible,
        factors.draft,
        ranks,
        rank_mask,
        draft_width,
    )
    passing_draft = [
        index
        for index in eligible
        if factors.draft[index] in selected_drafts
    ]

    selected_tiles: set[tuple[bytes, bytes]] = set()
    selected_wildlife: set[tuple[bytes, bytes, bytes]] = set()
    if arm == "independent-wide":
        tile_keys = _rank_keys(
            passing_draft,
            factors.tile,
            ranks,
            rank_mask,
            tile_width,
        )
        wildlife_keys = _rank_keys(
            passing_draft,
            factors.wildlife,
            ranks,
            rank_mask,
            wildlife_width,
        )
        passing = [
            index
            for index in passing_draft
            if factors.tile[index] in tile_keys
            and factors.wildlife[index] in wildlife_keys
        ]
    else:
        for draft in selected_drafts:
            draft_indices = [
                index
                for index in passing_draft
                if factors.draft[index] == draft
            ]
            for tile in _rank_keys(
                draft_indices,
                factors.tile,
                ranks,
                rank_mask,
                tile_width,
            ):
                selected_tiles.add((draft, tile))
        passing_tile = [
            index
            for index in passing_draft
            if (factors.draft[index], factors.tile[index])
            in selected_tiles
        ]
        for draft, tile in selected_tiles:
            prefix_indices = [
                index
                for index in passing_tile
                if factors.draft[index] == draft
                and factors.tile[index] == tile
            ]
            for wildlife in _rank_keys(
                prefix_indices,
                factors.wildlife,
                ranks,
                rank_mask,
                wildlife_width,
            ):
                selected_wildlife.add((draft, tile, wildlife))
        passing = [
            index
            for index in passing_tile
            if (
                factors.draft[index],
                factors.tile[index],
                factors.wildlife[index],
            )
            in selected_wildlife
        ]
    return np.asarray(
        sorted(set(np.flatnonzero(frontier)) | set(passing)),
        dtype=np.int32,
    )


def _group_result(batch: Any, row: int, arm: str) -> dict[str, Any]:
    candidate_mask = np.asarray(batch.candidate_mask)[row]
    count = int(np.sum(candidate_mask))
    actions = np.asarray(batch.action_features)[row, :count]
    factors = factor_rows(actions)
    complete_keys = list(
        zip(
            factors.draft,
            factors.tile,
            factors.wildlife,
            strict=True,
        )
    )
    bijective = len(set(complete_keys)) == count
    ranks = np.asarray(batch.expected_rank)[row, :count]
    rank_mask = np.asarray(batch.expected_rank_mask)[row, :count]
    flags = np.asarray(batch.source_flags)[row, :count]
    hashes = np.asarray(batch.action_hash)[row, :count]
    target = build_expected_rank_target_mask(
        expected_rank=np.asarray(batch.expected_rank)[row : row + 1],
        expected_rank_mask=np.asarray(batch.expected_rank_mask)[
            row : row + 1
        ],
        source_flags=np.asarray(batch.source_flags)[row : row + 1],
        candidate_mask=np.asarray(batch.candidate_mask)[row : row + 1],
        action_hashes=np.asarray(batch.action_hash)[row : row + 1],
    )[0, :count]
    candidates = oracle_candidate_indices(
        factors,
        ranks,
        rank_mask,
        flags,
        arm=arm,
    )
    subset_scores = np.where(
        rank_mask[candidates],
        -ranks[candidates],
        -1e9,
    )
    retained_local = frontier_anchored_retained_indices(
        scores=subset_scores,
        source_flags=flags[candidates],
        action_hashes=hashes[candidates],
    )
    retained = candidates[retained_local]
    retained_nonfrontier = retained[
        (
            flags[retained]
            & GRADED_SOURCE_CHAMPION_FRONTIER
        )
        == 0
    ]
    target_slots = int(np.sum(target))
    target_hits = int(np.sum(target[retained_nonfrontier]))
    winner = int(np.asarray(batch.selected_index)[row])
    frontier_indices = set(np.flatnonzero(
        (
            flags.astype(np.int64)
            & GRADED_SOURCE_CHAMPION_FRONTIER
        )
        != 0
    ))
    return {
        "group_id": int(np.asarray(batch.group_id)[row]) & ((1 << 64) - 1),
        "candidate_count": count,
        "proposal_count": len(candidates),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact": target_hits == target_slots,
        "r4800_winner_retained": winner in retained,
        "factor_bijection": bijective,
        "all_frontier_preserved": frontier_indices.issubset(
            set(candidates)
        ),
        "finite_ranks": bool(np.all(np.isfinite(ranks[rank_mask]))),
    }


def _aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    target_slots = sum(int(group["target_slots"]) for group in groups)
    target_hits = sum(int(group["target_hits"]) for group in groups)
    proposal_counts = np.asarray(
        [group["proposal_count"] for group in groups],
        dtype=np.float64,
    )
    return {
        "groups": len(groups),
        "candidates": sum(int(group["candidate_count"]) for group in groups),
        "target_slots": target_slots,
        "target_hits": target_hits,
        "target_positive_recall": target_hits / max(target_slots, 1),
        "target_set_exact_fraction": sum(
            int(group["target_set_exact"]) for group in groups
        )
        / len(groups),
        "r4800_winner_retention": sum(
            int(group["r4800_winner_retained"]) for group in groups
        )
        / len(groups),
        "mean_proposal_count": float(np.mean(proposal_counts)),
        "p90_proposal_count": float(
            np.quantile(proposal_counts, 0.90, method="higher")
        ),
        "p99_proposal_count": float(
            np.quantile(proposal_counts, 0.99, method="higher")
        ),
        "maximum_proposal_count": int(np.max(proposal_counts)),
        "all_factor_bijections": all(
            bool(group["factor_bijection"]) for group in groups
        ),
        "all_frontier_preserved": all(
            bool(group["all_frontier_preserved"]) for group in groups
        ),
        "all_ranks_finite": all(
            bool(group["finite_ranks"]) for group in groups
        ),
    }


def _audit_split(
    dataset_root: Path,
    cache_root: Path,
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Scale16ExpectedRankDataset(dataset_root, cache_root)
    groups: list[dict[str, Any]] = []
    for batch in dataset.batches(
        4,
        maximum_actions_per_batch=16384,
        shuffle=False,
    ):
        for row in range(batch.action_features.shape[0]):
            groups.append(_group_result(batch, row, arm))
    if len(groups) != dataset.group_count:
        raise ValueError("hierarchical oracle split coverage drifted")
    aggregate = _aggregate(groups)
    if aggregate["candidates"] != dataset.candidate_count:
        raise ValueError("hierarchical oracle candidate coverage drifted")
    aggregate["dataset_manifest_blake3"] = blake3.blake3(
        (dataset.root / "dataset.json").read_bytes()
    ).hexdigest()
    aggregate["cache_manifest_blake3"] = blake3.blake3(
        (cache_root / "manifest.json").read_bytes()
    ).hexdigest()
    return aggregate, groups


def run_arm(
    train_dataset: Path,
    train_cache: Path,
    validation_dataset: Path,
    validation_cache: Path,
    evidence_path: Path,
    arm_index: int,
) -> dict[str, Any]:
    """Run one independently scheduled full-open-data oracle arm."""
    if not 0 <= arm_index < len(ARMS):
        raise ValueError("hierarchical oracle arm index is outside 0-3")
    evidence = json.loads(evidence_path.read_text())
    if (
        blake3.blake3(evidence_path.read_bytes()).hexdigest()
        != FROZEN_ADR0113_BLAKE3
        or evidence["scientific"]["classification"]
        != "shared_adapter_capacity_insufficient"
    ):
        raise ValueError("ADR 0113 frozen evidence differs")
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    arm = ARMS[arm_index]
    train, train_groups = _audit_split(
        train_dataset,
        train_cache,
        arm,
    )
    validation, validation_groups = _audit_split(
        validation_dataset,
        validation_cache,
        arm,
    )
    scientific = {
        "arm": arm,
        "arm_index": arm_index,
        "widths": ARM_WIDTHS[arm],
        "conditional": arm != "independent-wide",
        "adr0113_evidence_blake3": FROZEN_ADR0113_BLAKE3,
        "train": train,
        "validation": validation,
        "group_identity_blake3": blake3.blake3(
            json.dumps(
                {
                    "train": train_groups,
                    "validation": validation_groups,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "training_used": False,
        "gradients_used": False,
        "optimizer_updates_used": False,
        "gates": {
            "all_800_groups_covered": (
                train["groups"] == 560
                and validation["groups"] == 240
            ),
            "all_factor_bijections": bool(
                train["all_factor_bijections"]
                and validation["all_factor_bijections"]
            ),
            "all_frontier_preserved": bool(
                train["all_frontier_preserved"]
                and validation["all_frontier_preserved"]
            ),
            "all_ranks_finite": bool(
                train["all_ranks_finite"]
                and validation["all_ranks_finite"]
            ),
            "training_gradients_optimizer_unused": True,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    return _report(
        scientific,
        started,
        swap_before,
        experiment_id=EXPERIMENT_ID,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("arm",))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
    )
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_arm(
        args.dataset,
        args.cache,
        args.selected_run,
        args.analytic,
        args.evidence,
        args.group_index,
    )
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "arm": report["scientific"]["arm"],
                "validation_recall": report["scientific"][
                    "validation"
                ]["target_positive_recall"],
                "resource_qualification_passed": bool(
                    all(report["scientific"]["gates"].values())
                    and report["telemetry"]["peak_process_rss_bytes"]
                    <= 4 * 1024**3
                    and report["telemetry"]["process_swaps"] == 0
                    and report["telemetry"][
                        "system_swap_delta_bytes"
                    ]
                    is not None
                    and report["telemetry"][
                        "system_swap_delta_bytes"
                    ]
                    <= 0
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
