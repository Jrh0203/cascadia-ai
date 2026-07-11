"""R1.1a contention-audit analyzer.

Consumes the `--table-contention-audit` exporter output and bounds the prize
of cooperative table optimization: how many table points per game the seats
leave on the table by choosing their own-Q action over the best model-Q
alternative, overall and conditioned on a small own-Q sacrifice.

Read the output as a *bound estimator*, not a gate: the alternative ranking
uses model derived Q (the search's completed-Q runner-up is not recoverable
from ledgers without re-searching), table values at non-terminal afterstates
come from the value head, and per-decision deltas are first-order (they do
not compound across a trajectory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_EPSILONS = (0.1, 0.25, 0.5, 1.0)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _subset_stats(
    rows: list[dict[str, Any]], games: int, epsilon: float | None
) -> dict[str, Any]:
    if epsilon is None:
        subset = rows
    else:
        subset = [
            row
            for row in rows
            if float(row["own_q_sacrifice_chosen_minus_runner"]) <= epsilon
        ]
    deltas = [float(row["table_delta_runner_minus_chosen"]) for row in subset]
    positive = [delta for delta in deltas if delta > 0.0]
    recoverable_table = sum(positive)
    return {
        "epsilon": epsilon,
        "decisions": len(subset),
        "fraction_of_decisions": len(subset) / len(rows) if rows else 0.0,
        "flip_rate": len(positive) / len(subset) if subset else 0.0,
        "mean_table_delta": sum(deltas) / len(subset) if subset else 0.0,
        "mean_positive_table_delta_per_decision": (
            recoverable_table / len(subset) if subset else 0.0
        ),
        "recoverable_table_points_per_game": recoverable_table / games if games else 0.0,
        "recoverable_gate_points_per_game": (
            recoverable_table / games / 4.0 if games else 0.0
        ),
    }


def analyze(path: Path, epsilons: tuple[float, ...] = DEFAULT_EPSILONS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "contention_decision":
            rows.append(record)
        elif record.get("type") == "contention_summary":
            summary = record
    if not rows:
        raise ValueError(f"no contention_decision rows in {path}")
    if summary is None:
        raise ValueError(f"no contention_summary row in {path} (incomplete audit)")

    games = len({int(row["seed"]) for row in rows})
    deltas = [float(row["table_delta_runner_minus_chosen"]) for row in rows]
    sacrifices = [float(row["own_q_sacrifice_chosen_minus_runner"]) for row in rows]
    model_estimated = [
        row
        for row in rows
        if not (row["chosen_table_exact"] and row["runner_table_exact"])
    ]
    chosen_is_model_best = sum(
        1 for row in rows if row["chosen"]["index"] == row["model_best"]["index"]
    )

    return {
        "status": "pass",
        "ruleset_id": rows[0].get("ruleset_id"),
        "games": games,
        "decisions": len(rows),
        "single_action_skipped": summary.get("single_action_skipped"),
        "chosen_is_model_best_rate": chosen_is_model_best / len(rows),
        "table_delta": {
            "mean": sum(deltas) / len(deltas),
            "p50": _percentile(deltas, 0.50),
            "p90": _percentile(deltas, 0.90),
            "max": max(deltas),
        },
        "own_q_sacrifice": {
            "p50": _percentile(sacrifices, 0.50),
            "p90": _percentile(sacrifices, 0.90),
        },
        "model_estimated_fraction": len(model_estimated) / len(rows),
        "unconditional": _subset_stats(rows, games, None),
        "by_epsilon": [_subset_stats(rows, games, epsilon) for epsilon in epsilons],
        "caveats": [
            "runner-up ranking uses model derived Q, not search completed-Q",
            "non-terminal table values are value-head estimates",
            "per-decision deltas are first-order and do not compound",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Table-Contention Audit (R1.1a)",
        "",
        f"Games: `{report['games']}` — decisions audited: `{report['decisions']}`",
        f"Chosen action is the model-Q best: `{report['chosen_is_model_best_rate']:.1%}`",
        f"Value-head-estimated table pairs: `{report['model_estimated_fraction']:.1%}`",
        "",
        "## The bound",
        "",
        "| own-Q sacrifice ≤ | decisions | flip rate | table pts/game | gate pts/game |",
        "|---|---:|---:|---:|---:|",
    ]
    unconditional = report["unconditional"]
    lines.append(
        f"| (any) | {unconditional['decisions']} | {unconditional['flip_rate']:.1%} "
        f"| {unconditional['recoverable_table_points_per_game']:+.3f} "
        f"| {unconditional['recoverable_gate_points_per_game']:+.3f} |"
    )
    for entry in report["by_epsilon"]:
        lines.append(
            f"| {entry['epsilon']} | {entry['decisions']} | {entry['flip_rate']:.1%} "
            f"| {entry['recoverable_table_points_per_game']:+.3f} "
            f"| {entry['recoverable_gate_points_per_game']:+.3f} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        *[f"- {caveat}" for caveat in report["caveats"]],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument(
        "--epsilons",
        default=",".join(str(value) for value in DEFAULT_EPSILONS),
        help="Comma-separated own-Q sacrifice ceilings",
    )
    args = parser.parse_args()
    epsilons = tuple(float(value) for value in args.epsilons.split(",") if value.strip())
    report = analyze(Path(args.input), epsilons)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps({key: report[key] for key in ("games", "decisions", "unconditional")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
