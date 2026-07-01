"""Stable per-decision panels and paired statistics for ADR 0166."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_cache import LOW_SUPPLY_MAX_UNSEEN
from cascadia_mlx.relational_substrate_mlx_metrics import (
    CANDIDATE_CHUNK,
)

BOOTSTRAP_SEED = 2026061719
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_CHUNK = 1_000
STRATEGIC_OPPORTUNITIES = ("elk", "salmon", "hawk")
U64_MASK = (1 << 64) - 1
I64_MIN = -(1 << 63)


def collect_decision_panel(
    model: object,
    dataset: object,
    *,
    candidate_chunk: int = CANDIDATE_CHUNK,
) -> list[dict[str, Any]]:
    """Score each validation decision once and retain paired sufficient data."""
    if candidate_chunk <= 0:
        raise ValueError("paired-panel candidate chunk must be positive")
    opportunity_members = {
        name: set(
            int(row)
            for row in np.asarray(rows, dtype=np.int64)
        )
        for name, rows in dataset.opportunity_rows.items()
    }
    model.eval()
    records = []
    for row in range(dataset.group_count):
        batch = dataset.batch(
            [row],
            arm="c0-exact-r2",
            transform_ids=[0],
        )
        mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)[0]
        count = int(mask.sum())
        if count <= 0:
            raise ValueError("paired validation decision has no candidates")
        parent = model.encode_parent(batch)
        mx.eval(parent)
        score_chunks = []
        for start in range(0, count, candidate_chunk):
            prediction = model.predict(
                batch,
                candidate_slice=slice(
                    start,
                    min(start + candidate_chunk, count),
                ),
                parent_state=parent,
            )
            mx.eval(prediction.scores)
            score_chunks.append(np.asarray(prediction.scores)[0])
        scores = np.concatenate(score_chunks).astype(np.float64)
        if len(scores) != count or np.any(~np.isfinite(scores)):
            raise ValueError("paired validation scores are incomplete")

        hashes = np.asarray(batch.base.action_hash)[0, :count]
        winner = int(np.asarray(batch.base.selected_index)[0])
        teacher = np.asarray(batch.base.r4800_mean)[0, :count].astype(
            np.float64
        )
        teacher_mask = np.asarray(
            batch.base.r4800_mask,
            dtype=np.bool_,
        )[0, :count]
        if winner >= count or not teacher_mask[winner]:
            raise ValueError("paired validation winner lacks an R4800 label")
        ranking = _stable_ranking(scores, hashes)
        winner_rank = int(np.flatnonzero(ranking == winner)[0]) + 1
        retained = ranking[: min(64, count)]
        retained_labeled = retained[teacher_mask[retained]]
        labeled_teacher = teacher[teacher_mask]
        regret = (
            float(np.max(labeled_teacher) - np.max(teacher[retained_labeled]))
            if len(retained_labeled)
            else float(np.max(labeled_teacher) - np.min(labeled_teacher))
        )
        errors = scores[teacher_mask] - labeled_teacher
        turn = int(np.asarray(batch.base.turn)[0])
        draft_kind = np.asarray(batch.base.draft_kind)[0, :count]
        group_id = int(np.asarray(batch.base.group_id)[0])
        score_digest = blake3.blake3()
        score_digest.update(hashes.tobytes(order="C"))
        score_digest.update(scores.astype("<f4").tobytes())
        records.append(
            {
                "row": row,
                "group_id": group_id,
                "turn": turn,
                "candidates": count,
                "labeled_candidates": int(teacher_mask.sum()),
                "teacher_winner_index": winner,
                "teacher_winner_action_hash": bytes(hashes[winner]).hex(),
                "winner_rank": winner_rank,
                "top64_recalled": winner_rank <= 64,
                "top64_regret": regret,
                "absolute_error_sum": float(np.abs(errors).sum()),
                "squared_error_sum": float(np.square(errors).sum()),
                "bias_sum": float(errors.sum()),
                "low_supply": 81 - turn <= LOW_SUPPLY_MAX_UNSEEN,
                "independent_draft_winner": int(draft_kind[winner]) == 1,
                "phase": (
                    "early" if turn < 27 else "middle" if turn < 54 else "late"
                ),
                "opportunities": {
                    name: row in opportunity_members.get(name, set())
                    for name in ("elk", "salmon", "hawk", "bear")
                },
                "prediction_blake3": score_digest.hexdigest(),
            }
        )
    return records


def panel_identity(records: Sequence[dict[str, Any]]) -> str:
    """Hash one ordered panel after validating its one-row-per-group shape."""
    _panel_arrays(records)
    return blake3.blake3(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def compare_decision_panels(
    treatment: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute frozen paired deltas with complete decisions as the unit."""
    if bootstrap_replicates <= 0:
        raise ValueError("paired bootstrap replicate count must be positive")
    left = _panel_arrays(treatment)
    right = _panel_arrays(control)
    _require_aligned_panels(left, right)
    strategic = np.any(
        np.stack(
            [left["opportunities"][name] for name in STRATEGIC_OPPORTUNITIES],
            axis=1,
        ),
        axis=1,
    )
    rng = np.random.default_rng(bootstrap_seed)
    return {
        "schema_version": 1,
        "groups": len(left["row"]),
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "unit": "complete-decision",
        },
        "global_top64_recall": _paired_mean_statistic(
            left["top64_recalled"].astype(np.float64),
            right["top64_recalled"].astype(np.float64),
            rng=rng,
            replicates=bootstrap_replicates,
            favorable="greater",
        ),
        "strategic_top64_recall": _paired_mean_statistic(
            left["top64_recalled"][strategic].astype(np.float64),
            right["top64_recalled"][strategic].astype(np.float64),
            rng=rng,
            replicates=bootstrap_replicates,
            favorable="greater",
        ),
        "top64_regret": _paired_mean_statistic(
            left["top64_regret"],
            right["top64_regret"],
            rng=rng,
            replicates=bootstrap_replicates,
            favorable="less",
        ),
        "r4800_rmse": _paired_rmse_statistic(
            left,
            right,
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "protected": {
            "low_supply": _point_recall_delta(
                left,
                right,
                left["low_supply"],
            ),
            "independent_draft_winner": _point_recall_delta(
                left,
                right,
                left["independent_draft_winner"],
            ),
        },
        "phase": {
            name: _point_recall_delta(
                left,
                right,
                left["phase"] == name,
            )
            for name in ("early", "middle", "late")
        },
        "opportunity": {
            name: _point_recall_delta(
                left,
                right,
                left["opportunities"][name],
            )
            for name in ("elk", "salmon", "hawk", "bear")
        },
    }


def factorial_effects(
    panels: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Report raw two-by-two effects for the four registered query arms."""
    required = {
        "c0-parent-conditioned",
        "t1-supply-query",
        "t2-frontier-query",
        "t3-combined-query",
    }
    if set(panels) != required:
        raise ValueError("factorial panels must contain exactly four arms")
    arrays = {arm: _panel_arrays(panel) for arm, panel in panels.items()}
    control = arrays["c0-parent-conditioned"]
    for candidate in arrays.values():
        _require_aligned_panels(candidate, control)
    strategic = np.any(
        np.stack(
            [
                control["opportunities"][name]
                for name in STRATEGIC_OPPORTUNITIES
            ],
            axis=1,
        ),
        axis=1,
    )
    return {
        "global_top64_recall": _factorial_metric(
            {
                arm: values["top64_recalled"].astype(np.float64)
                for arm, values in arrays.items()
            }
        ),
        "strategic_top64_recall": _factorial_metric(
            {
                arm: values["top64_recalled"][strategic].astype(np.float64)
                for arm, values in arrays.items()
            }
        ),
        "top64_regret": _factorial_metric(
            {
                arm: values["top64_regret"]
                for arm, values in arrays.items()
            }
        ),
    }


def _panel_arrays(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("paired panel cannot be empty")
    rows = np.asarray([record["row"] for record in records], dtype=np.int64)
    group_ids = _group_id_bits(records)
    winner_hashes = [record["teacher_winner_action_hash"] for record in records]
    if (
        not np.array_equal(rows, np.arange(len(records), dtype=np.int64))
        or len(np.unique(group_ids)) != len(group_ids)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in winner_hashes
        )
    ):
        raise ValueError("paired panel rows or teacher identities are malformed")
    opportunities = {
        name: np.asarray(
            [bool(record["opportunities"][name]) for record in records],
            dtype=np.bool_,
        )
        for name in ("elk", "salmon", "hawk", "bear")
    }
    return {
        "row": rows,
        "group_id": group_ids,
        "winner_hash": np.asarray(winner_hashes),
        "top64_recalled": np.asarray(
            [bool(record["top64_recalled"]) for record in records],
            dtype=np.bool_,
        ),
        "top64_regret": np.asarray(
            [float(record["top64_regret"]) for record in records],
            dtype=np.float64,
        ),
        "squared_error_sum": np.asarray(
            [float(record["squared_error_sum"]) for record in records],
            dtype=np.float64,
        ),
        "labeled_candidates": np.asarray(
            [int(record["labeled_candidates"]) for record in records],
            dtype=np.int64,
        ),
        "low_supply": np.asarray(
            [bool(record["low_supply"]) for record in records],
            dtype=np.bool_,
        ),
        "independent_draft_winner": np.asarray(
            [
                bool(record["independent_draft_winner"])
                for record in records
            ],
            dtype=np.bool_,
        ),
        "phase": np.asarray([record["phase"] for record in records]),
        "opportunities": opportunities,
    }


def _group_id_bits(
    records: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Normalize signed or unsigned JSON integers to identical u64 bits."""
    values = []
    for record in records:
        value = record.get("group_id")
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("paired panel group ID must be an integer")
        integer = int(value)
        if integer < I64_MIN or integer > U64_MASK:
            raise ValueError("paired panel group ID is outside 64-bit range")
        values.append(integer & U64_MASK)
    return np.asarray(values, dtype=np.uint64)


def _require_aligned_panels(
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    if (
        not np.array_equal(left["row"], right["row"])
        or not np.array_equal(left["group_id"], right["group_id"])
        or not np.array_equal(left["winner_hash"], right["winner_hash"])
        or not np.array_equal(
            left["labeled_candidates"],
            right["labeled_candidates"],
        )
        or not np.array_equal(left["low_supply"], right["low_supply"])
        or not np.array_equal(
            left["independent_draft_winner"],
            right["independent_draft_winner"],
        )
        or not np.array_equal(left["phase"], right["phase"])
        or any(
            not np.array_equal(
                left["opportunities"][name],
                right["opportunities"][name],
            )
            for name in ("elk", "salmon", "hawk", "bear")
        )
    ):
        raise ValueError("paired panels do not describe identical decisions")


def _paired_mean_statistic(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
    favorable: str,
) -> dict[str, Any]:
    if treatment.shape != control.shape or not len(treatment):
        raise ValueError("paired mean statistic requires aligned observations")
    differences = treatment - control
    draws = _bootstrap_means(differences, rng=rng, replicates=replicates)
    return {
        "treatment": float(np.mean(treatment)),
        "control": float(np.mean(control)),
        "delta": float(np.mean(differences)),
        "confidence_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_favorable": float(
            np.mean(draws > 0.0)
            if favorable == "greater"
            else np.mean(draws < 0.0)
        ),
        "favorable_direction": favorable,
    }


def _paired_rmse_statistic(
    treatment: dict[str, Any],
    control: dict[str, Any],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    counts = treatment["labeled_candidates"]
    treatment_rmse = float(
        np.sqrt(treatment["squared_error_sum"].sum() / counts.sum())
    )
    control_rmse = float(
        np.sqrt(control["squared_error_sum"].sum() / counts.sum())
    )
    draws = np.empty(replicates, dtype=np.float64)
    groups = len(counts)
    written = 0
    while written < replicates:
        size = min(BOOTSTRAP_CHUNK, replicates - written)
        indices = rng.integers(0, groups, size=(size, groups))
        sampled_counts = counts[indices].sum(axis=1)
        left = np.sqrt(
            treatment["squared_error_sum"][indices].sum(axis=1)
            / sampled_counts
        )
        right = np.sqrt(
            control["squared_error_sum"][indices].sum(axis=1)
            / sampled_counts
        )
        draws[written : written + size] = left - right
        written += size
    return {
        "treatment": treatment_rmse,
        "control": control_rmse,
        "delta": treatment_rmse - control_rmse,
        "confidence_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_noninferior_at_plus_0_03": float(
            np.mean(draws <= 0.03)
        ),
        "favorable_direction": "less",
    }


def _bootstrap_means(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    draws = np.empty(replicates, dtype=np.float64)
    groups = len(differences)
    written = 0
    while written < replicates:
        size = min(BOOTSTRAP_CHUNK, replicates - written)
        indices = rng.integers(0, groups, size=(size, groups))
        draws[written : written + size] = differences[indices].mean(axis=1)
        written += size
    return draws


def _point_recall_delta(
    treatment: dict[str, Any],
    control: dict[str, Any],
    selected: np.ndarray,
) -> dict[str, Any]:
    if not np.any(selected):
        raise ValueError("paired recall slice cannot be empty")
    left = treatment["top64_recalled"][selected].astype(np.float64)
    right = control["top64_recalled"][selected].astype(np.float64)
    return {
        "groups": int(selected.sum()),
        "treatment": float(left.mean()),
        "control": float(right.mean()),
        "delta": float((left - right).mean()),
    }


def _factorial_metric(values: dict[str, np.ndarray]) -> dict[str, float]:
    means = {arm: float(np.mean(array)) for arm, array in values.items()}
    c0 = means["c0-parent-conditioned"]
    t1 = means["t1-supply-query"]
    t2 = means["t2-frontier-query"]
    t3 = means["t3-combined-query"]
    return {
        "c0_parent_conditioned": c0,
        "t1_supply_query": t1,
        "t2_frontier_query": t2,
        "t3_combined_query": t3,
        "supply_main_effect": ((t1 - c0) + (t3 - t2)) / 2.0,
        "frontier_main_effect": ((t2 - c0) + (t3 - t1)) / 2.0,
        "interaction": t3 - t1 - t2 + c0,
    }


def _stable_ranking(
    scores: np.ndarray,
    hashes: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), bytes(hashes[index])),
        ),
        dtype=np.int32,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "collect_decision_panel",
    "compare_decision_panels",
    "factorial_effects",
    "panel_identity",
]
