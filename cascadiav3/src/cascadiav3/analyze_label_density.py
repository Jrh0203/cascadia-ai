"""R1.4 Stage 0 label-noise audit over packed expert tensor shards.

CPU-only analyzer (no torch) for the R1.4 densification design
(`docs/v3/R1_4_DENSIFICATION_DESIGN.md`, section 5 Stage 0). It reads packed
v2+ expert tensor shards through :class:`cascadiav3.expert_tensor_shards.
ExpertTensorShard` and emits three sections plus a trajectory-adjacency
probe, in the same JSON+Markdown report shape as ``analyze_menu_coverage``:

1. **Density census** — per-root q-valid fraction (visited actions divided by
   retained actions), improved-policy probability mass resting on UNVISITED
   actions, and visit concentration (fraction of root visits on the top-1
   action), each with mean/median/P10/P90 over the analyzed roots.

2. **V1 falsifier** — for the ACTIVE seat of each record, compares the
   exported ``search_root_value`` against the realized terminal outcome
   ``final_score_vector[active_seat]``. Operationalization of the
   preregistered continuation bar ("search_root_value must cut value-target
   RMSE >= 20% vs the raw outcome at |bias| <= 0.5 points"):

   - ``rmse_search_root_value``    = RMSE(outcome_i, search_root_value_i);
   - ``rmse_baseline_within_phase`` = RMSE(outcome_i, mean(outcome | phase_i)),
     i.e. the best constant-per-phase predictor built from the outcomes
     themselves — the "raw outcome as its own predictor" baseline. Records in
     the ``unknown`` phase use the unknown-group mean;
   - ``rmse_baseline_global_mean`` = RMSE(outcome_i, mean(outcome)) — the
     population outcome standard deviation (ddof=0), reported as the 1-sample
     noise scale;
   - ``rmse_reduction_vs_within_phase_baseline_pct``
     = 100 * (1 - rmse_search_root_value / rmse_baseline_within_phase);
   - ``bias`` = mean(search_root_value_i - outcome_i), signed.

   All of the above are reported overall and stratified by game phase.

3. **Hard-root census (D1)** — fraction of roots whose top1-top2 completed-Q
   gap (over q-valid actions with positive simulation count) is below the
   pairwise SE proxy ``sqrt(var1/count1 + var2/count2)``. Roots need >= 2
   eligible actions; eligibility coverage is reported. Overall and by phase.

Phase stratification: packed v2-v4 arrays carry no game/ply fields (design
doc section 2.5), so the phase is recovered from the packed public-token
features: the ACTIVE player's token row (kind one-hot ``player`` and
``relative_seat == 0``; layout per ``real-root-exporter/src/
feature_tensors.rs::public_token_features``) stores ``tile_count / 23`` at
column 16. Boards start with 3 starter tiles (``Board::from_starter``), so
``turns_played = tile_count - 3`` in [0, 19] is the active player's own turn
counter — a turns-based proxy, not the true game ply. Turn bins mirror the
80-ply ``PHASES`` bins of ``analyze_search_decision_trace`` divided by 4
seats: opening 0-4, early_mid 5-9, late_mid 10-14, endgame 15-19. Records
whose player token cannot be recovered fall in phase ``unknown``; if more
than half the records are unknown the analyzer falls back to record-index
quartiles within each shard (documented in the report as
``record_index_within_shard`` — only meaningful if packed order is
generation order, see the adjacency probe).

Trajectory-adjacency probe (design doc section 8 unknown): packed shards
discard game/seed/ply metadata, but records from the same game share one
backfilled ``final_score_vector``. The probe run-length-encodes consecutive
records with bit-identical final-score rows and checks whether the active
seat increments by one (mod 4) inside runs. Long seat-cycling runs are
evidence (not proof — distinct games could collide on all four scores) that
packed record order preserves trajectory adjacency.

Usage::

    python -m cascadiav3.analyze_label_density \
        --shards <dir-or-glob-or-file> [--max-records N] \
        [--out report.json] [--summary-out report.md]

``--max-records N`` analyzes N evenly strided records across the corpus
(deterministic). The adjacency probe always runs over the full per-record
arrays, which are cheap, so subsampling never degrades its evidence.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
from pathlib import Path
from typing import Any

from .expert_tensor_shards import ExpertTensorShard

# Preregistered Stage 0 continuation bar for V1 (design doc section 5).
V1_RMSE_REDUCTION_PCT_REQUIRED = 20.0
V1_ABS_BIAS_MAX = 0.5

# Per-player-turn phase bins: analyze_search_decision_trace.PHASES covers
# game plies 0-79 in four 20-ply bins; with 4 seats that is 5 own turns each.
PHASE_BINS = (
    ("opening", 0, 4),
    ("early_mid", 5, 9),
    ("late_mid", 10, 14),
    ("endgame", 15, 19),
)
PHASE_NAMES = tuple(name for name, _, _ in PHASE_BINS) + ("unknown",)

# Packed public-token feature columns (real-root-exporter/src/
# feature_tensors.rs::public_token_features): 6 token-kind one-hots,
# owner_seat/3, relative_seat/3, market_slot/3, 6 coord features, then
# nature_tokens/10 at 15 and tile_count/23 at 16.
# Public on purpose: `torch_train_cascadiaformer._active_seat_and_tile_count`
# (R1.4 Stage 1 V1b phase gate) imports these so the trainer's batch-side
# extraction and this analyzer's per-record extraction cannot drift.
TOKEN_COL_KIND_PLAYER = 0
TOKEN_COL_OWNER_SEAT = 6
TOKEN_COL_RELATIVE_SEAT = 7
TOKEN_COL_TILE_COUNT = 16
OWNER_SEAT_SCALE = 3.0
TILE_COUNT_SCALE = 23.0
STARTER_TILE_COUNT = 3
_TURNS_PER_PLAYER = 20

# Fall back to record-index stratification when tile-count recovery fails
# for more than this fraction of analyzed records.
_MAX_UNKNOWN_PHASE_FRACTION = 0.5


def _phase_for_turn(turns_played: int) -> str:
    for name, start, end in PHASE_BINS:
        if start <= turns_played <= end:
            return name
    return "unknown"


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _distribution_stats(values: list[float]) -> dict[str, Any]:
    import numpy as np

    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.shape[0]),
        "mean": _round(float(array.mean())),
        "median": _round(float(np.percentile(array, 50.0))),
        "p10": _round(float(np.percentile(array, 10.0))),
        "p90": _round(float(np.percentile(array, 90.0))),
    }


def _resolve_shard_paths(shards: str) -> list[Path]:
    root = Path(shards)
    if root.is_dir():
        paths = sorted(root.glob("*.npz"))
    elif root.is_file():
        paths = [root]
    else:
        paths = sorted(Path(match) for match in glob_module.glob(shards))
    if not paths:
        raise ValueError(f"no shard files matched {shards!r}")
    return paths


def _analyzed_indices(record_count: int, max_records: int | None) -> list[int]:
    """Evenly strided deterministic subsample of the global record range."""
    if max_records is None or max_records >= record_count:
        return list(range(record_count))
    if max_records <= 0:
        raise ValueError("--max-records must be positive")
    return sorted({(index * record_count) // max_records for index in range(max_records)})


def _active_player_token_row(shard: ExpertTensorShard, index: int) -> Any | None:
    """Return the active player's packed token feature row, if recoverable."""
    import numpy as np

    token_start = int(shard.token_offsets[index])
    token_end = int(shard.token_offsets[index + 1])
    tokens = np.asarray(shard.tokens[token_start:token_end], dtype=np.float32)
    if tokens.size == 0:
        return None
    is_player = tokens[:, TOKEN_COL_KIND_PLAYER] > 0.5
    is_active = np.abs(tokens[:, TOKEN_COL_RELATIVE_SEAT]) < 0.5 / OWNER_SEAT_SCALE
    rows = np.flatnonzero(is_player & is_active)
    if rows.shape[0] != 1:
        return None
    return tokens[int(rows[0])]


def _record_phase_fields(shard: ExpertTensorShard, index: int) -> dict[str, Any]:
    """Recover active seat, tile count, and turn phase for one record."""
    row = _active_player_token_row(shard, index)
    token_seat = None if row is None else int(round(float(row[TOKEN_COL_OWNER_SEAT]) * OWNER_SEAT_SCALE))
    tile_count = None if row is None else int(round(float(row[TOKEN_COL_TILE_COUNT]) * TILE_COUNT_SCALE))
    if shard.active_seat is not None:
        active_seat = int(shard.active_seat[index])
    else:
        active_seat = token_seat
    turns_played = None
    phase = "unknown"
    if tile_count is not None:
        turns_played = max(0, min(_TURNS_PER_PLAYER - 1, tile_count - STARTER_TILE_COUNT))
        phase = _phase_for_turn(turns_played)
    return {
        "active_seat": active_seat,
        "token_seat": token_seat,
        "tile_count": tile_count,
        "turns_played": turns_played,
        "phase": phase,
    }


def _collect_rows(
    paths: list[Path],
    max_records: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load per-analyzed-record rows, per-shard summaries, and adjacency evidence."""
    import numpy as np

    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    adjacency_shards: list[dict[str, Any]] = []
    record_counts: list[int] = []
    shards: list[ExpertTensorShard] = []
    try:
        for path in paths:
            shard = ExpertTensorShard(path)
            shards.append(shard)
            if shard.search_root_value is None or shard.improved_policy is None:
                raise ValueError(
                    f"shard {path} is version {shard.version} and lacks "
                    "improved_policy/search_root_value; the Stage 0 label-noise "
                    "audit requires v2+ expert tensor shards"
                )
            record_counts.append(len(shard))
        total_records = sum(record_counts)
        global_indices = _analyzed_indices(total_records, max_records)
        cursor = 0
        base = 0
        for shard, record_count in zip(shards, record_counts, strict=True):
            local_indices: list[int] = []
            while cursor < len(global_indices) and global_indices[cursor] < base + record_count:
                local_indices.append(global_indices[cursor] - base)
                cursor += 1
            base += record_count
            for index in local_indices:
                action_start = int(shard.action_offsets[index])
                action_end = int(shard.action_offsets[index + 1])
                q_valid = np.asarray(shard.q_valid[action_start:action_end]).astype(bool)
                visits = np.asarray(shard.visits[action_start:action_end], dtype=np.float64)
                policy = np.asarray(shard.improved_policy[action_start:action_end], dtype=np.float64)
                target_q = np.asarray(shard.target_q[action_start:action_end], dtype=np.float64)
                q_variance = np.asarray(shard.q_variance[action_start:action_end], dtype=np.float64)
                q_count = np.asarray(shard.q_count[action_start:action_end], dtype=np.float64)
                fields = _record_phase_fields(shard, index)
                if fields["active_seat"] is None:
                    raise ValueError(
                        f"record {index} of {shard.path} has no packed active_seat and "
                        "no recoverable active player token; cannot attribute the outcome"
                    )
                total_visits = float(visits.sum())
                retained = int(q_valid.shape[0])
                row: dict[str, Any] = {
                    **fields,
                    "shard_ordinal": len(shard_summaries),
                    "record_index": index,
                    "record_position": index / max(1, record_count - 1),
                    "retained_actions": retained,
                    "q_valid_actions": int(q_valid.sum()),
                    "q_valid_fraction": float(q_valid.sum()) / max(1, retained),
                    "unvisited_policy_mass": float(policy[~q_valid].sum()),
                    "top1_visit_fraction": (
                        float(visits.max()) / total_visits if total_visits > 0.0 else None
                    ),
                    "outcome": float(shard.final_score_vector[index][fields["active_seat"]]),
                    "search_root_value": float(shard.search_root_value[index]),
                }
                eligible = q_valid & (q_count > 0.0)
                row["hard_root_eligible"] = bool(eligible.sum() >= 2)
                if row["hard_root_eligible"]:
                    order = np.argsort(-np.where(eligible, target_q, -np.inf), kind="stable")
                    top1, top2 = int(order[0]), int(order[1])
                    gap = float(target_q[top1] - target_q[top2])
                    pairwise_se = float(
                        np.sqrt(
                            q_variance[top1] / q_count[top1] + q_variance[top2] / q_count[top2]
                        )
                    )
                    row["top1_top2_gap"] = gap
                    row["pairwise_se"] = pairwise_se
                    row["hard_root"] = bool(gap < pairwise_se)
                rows.append(row)
            shard_summaries.append(
                {
                    "path": str(shard.path),
                    "version": shard.version,
                    "record_count": record_count,
                    "records_analyzed": len(local_indices),
                }
            )
            adjacency_shards.append(_shard_adjacency(shard))
    finally:
        for shard in shards:
            shard.close()
    adjacency = _merge_adjacency(adjacency_shards)
    return rows, shard_summaries, adjacency


def _shard_adjacency(shard: ExpertTensorShard) -> dict[str, Any]:
    """Run-length evidence over one full shard (never subsampled)."""
    import numpy as np

    scores = np.asarray(shard.final_score_vector, dtype=np.float64)
    record_count = int(scores.shape[0])
    if record_count == 0:
        return {
            "record_count": 0,
            "run_count": 0,
            "records_in_runs_ge2": 0,
            "max_run_length": 0,
            "seat_cycle_pairs": 0,
            "within_run_pairs": 0,
        }
    same_as_previous = np.zeros((record_count,), dtype=bool)
    if record_count > 1:
        same_as_previous[1:] = (scores[1:] == scores[:-1]).all(axis=1)
    run_starts = np.flatnonzero(~same_as_previous)
    run_lengths = np.diff(np.append(run_starts, record_count))
    seat_cycle_pairs = 0
    within_run_pairs = int(same_as_previous.sum())
    if shard.active_seat is not None and record_count > 1:
        seats = np.asarray(shard.active_seat, dtype=np.int64)
        increments = (seats[1:] - seats[:-1]) % 4 == 1
        seat_cycle_pairs = int((same_as_previous[1:] & increments).sum())
    return {
        "record_count": record_count,
        "run_count": int(run_starts.shape[0]),
        "records_in_runs_ge2": int(run_lengths[run_lengths >= 2].sum()),
        "max_run_length": int(run_lengths.max()),
        "seat_cycle_pairs": seat_cycle_pairs,
        "within_run_pairs": within_run_pairs,
    }


def _merge_adjacency(shard_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(evidence[key] for evidence in shard_evidence)
        for key in (
            "record_count",
            "run_count",
            "records_in_runs_ge2",
            "seat_cycle_pairs",
            "within_run_pairs",
        )
    }
    record_count = totals["record_count"]
    run_count = totals["run_count"]
    within_run_pairs = totals["within_run_pairs"]
    return {
        **totals,
        "max_run_length": max(
            (evidence["max_run_length"] for evidence in shard_evidence), default=0
        ),
        "mean_run_length": _round(record_count / run_count) if run_count else None,
        "fraction_records_in_runs_ge2": (
            _round(totals["records_in_runs_ge2"] / record_count) if record_count else None
        ),
        "seat_cycle_fraction_within_runs": (
            _round(totals["seat_cycle_pairs"] / within_run_pairs) if within_run_pairs else None
        ),
        "method": (
            "consecutive records sharing a bit-identical backfilled final_score_vector "
            "are treated as same-game candidates; active-seat +1 (mod 4) cycling inside "
            "runs corroborates ply order. Identical score vectors across distinct games "
            "would inflate the evidence, so this is suggestive, not proof."
        ),
    }


def _phase_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["phase"], []).append(row)
    return {name: grouped[name] for name in PHASE_NAMES if name in grouped}


def _apply_record_index_fallback(rows: list[dict[str, Any]]) -> None:
    """Stratify by record position within each shard when tokens fail us."""
    for row in rows:
        quartile = min(3, int(row["record_position"] * 4.0))
        row["phase"] = PHASE_BINS[quartile][0]


def _density_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visited = [row["top1_visit_fraction"] for row in rows if row["top1_visit_fraction"] is not None]
    return {
        "records": len(rows),
        "q_valid_fraction": _distribution_stats([row["q_valid_fraction"] for row in rows]),
        "unvisited_policy_mass": _distribution_stats([row["unvisited_policy_mass"] for row in rows]),
        "top1_visit_fraction": _distribution_stats(visited),
        "records_without_visits": len(rows) - len(visited),
        "retained_actions": _distribution_stats([float(row["retained_actions"]) for row in rows]),
        "q_valid_actions": _distribution_stats([float(row["q_valid_actions"]) for row in rows]),
    }


def _rmse(errors: list[float]) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


def _v1_group(rows: list[dict[str, Any]], baseline_errors: list[float]) -> dict[str, Any]:
    import numpy as np

    outcomes = np.asarray([row["outcome"] for row in rows], dtype=np.float64)
    signed = [row["search_root_value"] - row["outcome"] for row in rows]
    return {
        "records": len(rows),
        "outcome_mean": _round(float(outcomes.mean())),
        "outcome_std": _round(float(outcomes.std())),
        "rmse_search_root_value": _round(_rmse(signed)),
        "bias_search_root_value": _round(float(np.mean(signed))),
        "rmse_baseline_within_phase": _round(_rmse(baseline_errors)),
    }


def _v1_falsifier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    by_phase = _phase_rows(rows)
    phase_means = {
        phase: float(np.mean([row["outcome"] for row in phase_rows]))
        for phase, phase_rows in by_phase.items()
    }
    baseline_errors = {
        phase: [row["outcome"] - phase_means[phase] for row in phase_rows]
        for phase, phase_rows in by_phase.items()
    }
    all_baseline = [error for errors in baseline_errors.values() for error in errors]
    overall = _v1_group(rows, all_baseline)
    rmse_srv = overall["rmse_search_root_value"]
    rmse_within = overall["rmse_baseline_within_phase"]
    reduction_within = (
        _round(100.0 * (1.0 - rmse_srv / rmse_within)) if rmse_within else None
    )
    global_baseline = overall["outcome_std"]
    reduction_global = (
        _round(100.0 * (1.0 - rmse_srv / global_baseline)) if global_baseline else None
    )
    bias = overall["bias_search_root_value"]
    return {
        **overall,
        "rmse_baseline_global_mean": global_baseline,
        "rmse_reduction_vs_within_phase_baseline_pct": reduction_within,
        "rmse_reduction_vs_global_baseline_pct": reduction_global,
        "preregistered_bar": {
            "rmse_reduction_pct_required": V1_RMSE_REDUCTION_PCT_REQUIRED,
            "abs_bias_max": V1_ABS_BIAS_MAX,
            "rmse_reduction_pct_observed": reduction_within,
            "abs_bias_observed": _round(abs(bias)) if bias is not None else None,
            "passes": (
                reduction_within is not None
                and bias is not None
                and reduction_within >= V1_RMSE_REDUCTION_PCT_REQUIRED
                and abs(bias) <= V1_ABS_BIAS_MAX
            ),
        },
        "by_phase": {
            phase: _v1_group(phase_rows, baseline_errors[phase])
            for phase, phase_rows in by_phase.items()
        },
    }


def _hard_root_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["hard_root_eligible"]]
    hard = [row for row in eligible if row["hard_root"]]
    return {
        "records": len(rows),
        "eligible_records": len(eligible),
        "eligible_coverage": _round(len(eligible) / len(rows)) if rows else None,
        "hard_records": len(hard),
        "hard_fraction": _round(len(hard) / len(eligible)) if eligible else None,
        "top1_top2_gap": _distribution_stats([row["top1_top2_gap"] for row in eligible]),
        "pairwise_se": _distribution_stats([row["pairwise_se"] for row in eligible]),
    }


def _hard_root_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_hard_root_group(rows),
        "by_phase": {
            phase: _hard_root_group(phase_rows)
            for phase, phase_rows in _phase_rows(rows).items()
        },
    }


def analyze(shards: str, *, max_records: int | None = None) -> dict[str, Any]:
    paths = _resolve_shard_paths(shards)
    rows, shard_summaries, adjacency = _collect_rows(paths, max_records)
    if not rows:
        raise ValueError("no records analyzed")
    unknown = sum(1 for row in rows if row["phase"] == "unknown")
    seat_mismatches = sum(
        1
        for row in rows
        if row["token_seat"] is not None and row["token_seat"] != row["active_seat"]
    )
    tile_counts = [row["tile_count"] for row in rows if row["tile_count"] is not None]
    stratification = {
        "mode": "active_player_tile_count",
        "description": (
            "phase = active player's own-turn count recovered from the packed player "
            "token (tile_count column 16 x 23 minus 3 starter tiles); turn bins "
            "opening 0-4 / early_mid 5-9 / late_mid 10-14 / endgame 15-19 mirror the "
            "80-ply PHASES bins of analyze_search_decision_trace divided by 4 seats. "
            "This is a turns-based proxy: packed arrays carry no true game/ply fields."
        ),
        "unknown_phase_records": unknown,
        "token_seat_mismatch_records": seat_mismatches,
        "tile_count_min": min(tile_counts) if tile_counts else None,
        "tile_count_max": max(tile_counts) if tile_counts else None,
    }
    if unknown > _MAX_UNKNOWN_PHASE_FRACTION * len(rows):
        _apply_record_index_fallback(rows)
        stratification["mode"] = "record_index_within_shard"
        stratification["description"] = (
            "player-token tile counts were unrecoverable for most records; phases are "
            "record-position quartiles within each shard. Valid only if packed record "
            "order is generation order (see trajectory_adjacency)."
        )
    return {
        "status": "pass",
        "shards": shard_summaries,
        "records_total": sum(summary["record_count"] for summary in shard_summaries),
        "records_analyzed": len(rows),
        "max_records": max_records,
        "stratification": stratification,
        "density_census": _density_census(rows),
        "v1_falsifier": _v1_falsifier(rows),
        "hard_root_census": _hard_root_census(rows),
        "trajectory_adjacency": adjacency,
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    census = report["density_census"]
    falsifier = report["v1_falsifier"]
    bar = falsifier["preregistered_bar"]
    hard = report["hard_root_census"]
    adjacency = report["trajectory_adjacency"]
    lines = [
        "# Stage 0 Label-Noise Audit (R1.4)",
        "",
        f"Shards: `{len(report['shards'])}`, records analyzed: "
        f"`{report['records_analyzed']}` of `{report['records_total']}`",
        f"Phase stratification: `{report['stratification']['mode']}` "
        f"(unknown-phase records `{report['stratification']['unknown_phase_records']}`)",
        "",
        "## Density census",
        "",
        f"q-valid fraction per root: mean `{_fmt(census['q_valid_fraction']['mean'])}`, "
        f"median `{_fmt(census['q_valid_fraction']['median'])}`, "
        f"P10 `{_fmt(census['q_valid_fraction']['p10'])}`, "
        f"P90 `{_fmt(census['q_valid_fraction']['p90'])}`",
        f"Improved-policy mass on unvisited actions: "
        f"mean `{_fmt(census['unvisited_policy_mass']['mean'])}`, "
        f"median `{_fmt(census['unvisited_policy_mass']['median'])}`",
        f"Top-1 visit fraction: mean `{_fmt(census['top1_visit_fraction']['mean'])}`, "
        f"median `{_fmt(census['top1_visit_fraction']['median'])}`",
        "",
        "## V1 falsifier (search_root_value vs realized outcome, active seat)",
        "",
        f"RMSE(outcome, search_root_value): `{_fmt(falsifier['rmse_search_root_value'])}`",
        f"Baseline RMSE (within-phase outcome mean): "
        f"`{_fmt(falsifier['rmse_baseline_within_phase'])}`",
        f"Baseline RMSE (global outcome mean = outcome std): "
        f"`{_fmt(falsifier['rmse_baseline_global_mean'])}`",
        f"RMSE reduction vs within-phase baseline: "
        f"`{_fmt(falsifier['rmse_reduction_vs_within_phase_baseline_pct'], '.2f')}%` "
        f"(bar `>= {bar['rmse_reduction_pct_required']:.0f}%`)",
        f"Bias mean(search_root_value - outcome): "
        f"`{_fmt(falsifier['bias_search_root_value'], '+.4f')}` "
        f"(bar `|bias| <= {bar['abs_bias_max']}`)",
        f"Preregistered continuation bar: `{'PASS' if bar['passes'] else 'FAIL'}`",
        "",
        "| phase | records | RMSE(srv) | bias | baseline RMSE |",
        "|---|---|---|---|---|",
    ]
    for phase, group in falsifier["by_phase"].items():
        lines.append(
            f"| {phase} | {group['records']} | {_fmt(group['rmse_search_root_value'])} "
            f"| {_fmt(group['bias_search_root_value'], '+.4f')} "
            f"| {_fmt(group['rmse_baseline_within_phase'])} |"
        )
    lines.extend(
        [
            "",
            "## Hard-root census (D1)",
            "",
            f"Eligible roots (>=2 q-valid actions with counts): "
            f"`{hard['eligible_records']}/{hard['records']}` "
            f"(coverage `{_fmt(hard['eligible_coverage'])}`)",
            f"Hard fraction (top1-top2 gap < pairwise SE): `{_fmt(hard['hard_fraction'])}`",
            "",
            "| phase | eligible | hard fraction |",
            "|---|---|---|",
        ]
    )
    for phase, group in hard["by_phase"].items():
        lines.append(
            f"| {phase} | {group['eligible_records']}/{group['records']} "
            f"| {_fmt(group['hard_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory adjacency (packed record order)",
            "",
            f"Identical-final-score runs: `{adjacency['run_count']}` over "
            f"`{adjacency['record_count']}` records, mean length "
            f"`{_fmt(adjacency['mean_run_length'], '.2f')}`, max "
            f"`{adjacency['max_run_length']}`",
            f"Records inside runs >= 2: "
            f"`{_fmt(adjacency['fraction_records_in_runs_ge2'])}`",
            f"Seat +1 (mod 4) cycling inside runs: "
            f"`{_fmt(adjacency['seat_cycle_fraction_within_runs'])}`",
            "",
            "Runs are same-game *candidates* by shared backfilled final scores; "
            "score collisions across games can inflate this, so treat it as "
            "evidence, not proof. The audit is Stage 0 measurement only — it is "
            "never promotion evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        required=True,
        help="Shard .npz file, directory of shards, or glob pattern",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Analyze at most N evenly strided records across the corpus",
    )
    parser.add_argument("--out", help="Write the full JSON report here")
    parser.add_argument("--summary-out", help="Write the Markdown summary here")
    args = parser.parse_args()
    report = analyze(args.shards, max_records=args.max_records)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out:
        write_markdown(report, Path(args.summary_out))
    falsifier = report["v1_falsifier"]
    print(
        json.dumps(
            {
                "records_analyzed": report["records_analyzed"],
                "q_valid_fraction_mean": report["density_census"]["q_valid_fraction"]["mean"],
                "rmse_search_root_value": falsifier["rmse_search_root_value"],
                "rmse_reduction_vs_within_phase_baseline_pct": falsifier[
                    "rmse_reduction_vs_within_phase_baseline_pct"
                ],
                "bias_search_root_value": falsifier["bias_search_root_value"],
                "v1_bar_passes": falsifier["preregistered_bar"]["passes"],
                "hard_fraction": report["hard_root_census"]["hard_fraction"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
