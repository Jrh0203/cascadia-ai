"""Paired wildlife/habitat/Nature attribution from complete game ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
CATEGORIES = ("wildlife", "habitat", "nature_tokens", "total")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path, source_revision: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("source_revision") != source_revision:
        raise ValueError(f"source revision mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"category attribution requires a candidate-only arm: {path}")
    seeds = report.get("seeds")
    per_seed = report.get("candidate_per_seed")
    if not isinstance(seeds, list) or len(seeds) < 2 or not isinstance(per_seed, list):
        raise ValueError(f"report lacks complete per-seed results: {path}")
    expected = [int(seed) for seed in seeds]
    actual = [int(row["seed"]) for row in per_seed]
    if expected != actual or len(set(expected)) != len(expected):
        raise ValueError(f"report seed ordering/coverage mismatch: {path}")
    return report


def _normalized_search(search: dict[str, Any], *, ledger: bool) -> dict[str, Any]:
    def value(report_key: str, ledger_key: str | None = None, default: Any = None) -> Any:
        key = ledger_key if ledger and ledger_key is not None else report_key
        return search.get(key, default)

    return {
        "n_simulations": int(value("n_simulations", default=-1)),
        "top_m": int(value("top_m", default=-1)),
        "depth_rounds": int(value("depth_rounds", default=-1)),
        "determinizations": int(value("determinizations", "determinization_samples", -1)),
        "market_decision_samples": int(value("market_decision_samples", default=-1)),
        "exact_endgame_turns": int(value("exact_endgame_turns", default=0)),
        "blend_weight": float(value("blend_weight", "rollout_blend_weight", -1.0)),
        "k_interior": int(value("k_interior", default=-1)),
    }


def _score_categories(score: dict[str, Any], *, seed: int, seat: int) -> dict[str, float]:
    wildlife = score.get("wildlife")
    habitat = score.get("habitat")
    if not isinstance(wildlife, list) or not isinstance(habitat, list):
        raise ValueError(f"seed {seed} seat {seat} lacks category arrays")
    row = {
        "wildlife": float(sum(float(value) for value in wildlife)),
        "habitat": float(sum(float(value) for value in habitat)),
        "nature_tokens": float(score["nature_tokens"]),
        "total": float(score["total"]),
    }
    if abs(row["wildlife"] + row["habitat"] + row["nature_tokens"] - row["total"]) > 1e-9:
        raise ValueError(f"seed {seed} seat {seat} category sum does not equal total")
    return row


def _load_games(path: Path, report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_seeds = [int(seed) for seed in report["seeds"]]
    if len(rows) != len(expected_seeds):
        raise ValueError(f"game ledger row count mismatch: {path}")
    report_search = _normalized_search(report["search"], ledger=False)
    report_means = {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in report["candidate_per_seed"]
    }
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if row.get("type") != "gumbel_game_done":
            raise ValueError(f"non-terminal row in game ledger {path}: seed {seed}")
        if row.get("ruleset_id") != RULESET_ID:
            raise ValueError(f"game-ledger ruleset mismatch: seed {seed}")
        if int(row.get("decision_count", -1)) != 80:
            raise ValueError(f"game-ledger decision count mismatch: seed {seed}")
        if seed in by_seed:
            raise ValueError(f"duplicate game-ledger seed {seed}")
        ledger_search = _normalized_search(row.get("search", {}), ledger=True)
        if ledger_search != report_search:
            raise ValueError(f"game-ledger search mismatch: seed {seed}")
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError(f"seed {seed} does not contain four seat scores")
        seat_rows = [
            _score_categories(score, seed=seed, seat=seat)
            for seat, score in enumerate(scores)
        ]
        game_mean = {
            category: mean(seat_row[category] for seat_row in seat_rows)
            for category in CATEGORIES
        }
        if abs(game_mean["total"] - report_means.get(seed, float("nan"))) > 1e-9:
            raise ValueError(f"game-ledger/report total mismatch: seed {seed}")
        by_seed[seed] = {"seat_scores": seat_rows, "game_mean": game_mean}
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError(f"game-ledger seed coverage mismatch: {path}")
    return by_seed


def _arm_summary(games: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(games)
    all_seats = [
        seat
        for seed in seeds
        for seat in games[seed]["seat_scores"]
    ]
    return {
        "games": len(seeds),
        "overall_mean": {
            category: mean(row[category] for row in all_seats) for category in CATEGORIES
        },
        "by_seat_mean": [
            {
                category: mean(games[seed]["seat_scores"][seat][category] for seed in seeds)
                for category in CATEGORIES
            }
            for seat in range(4)
        ],
        "games_mean_at_least_100": sum(
            games[seed]["game_mean"]["total"] >= 100.0 for seed in seeds
        ),
        "seat_scores_at_least_100": sum(row["total"] >= 100.0 for row in all_seats),
    }


def build_category_comparison(
    *,
    left_report_path: Path,
    left_games_path: Path,
    right_report_path: Path,
    right_games_path: Path,
    source_revision: str,
    label: str,
) -> dict[str, Any]:
    left_report = _load_report(left_report_path, source_revision)
    right_report = _load_report(right_report_path, source_revision)
    if left_report["seeds"] != right_report["seeds"]:
        raise ValueError("category comparison reports do not share one seed set")
    if _normalized_search(left_report["search"], ledger=False) != _normalized_search(
        right_report["search"], ledger=False
    ):
        raise ValueError("category comparison reports do not share one search contract")
    left_games = _load_games(left_games_path, left_report)
    right_games = _load_games(right_games_path, right_report)
    seeds = sorted(left_games)
    category_stats: dict[str, Any] = {}
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        deltas = {
            category: (
                left_games[seed]["game_mean"][category]
                - right_games[seed]["game_mean"][category]
            )
            for category in CATEGORIES
        }
        if abs(
            deltas["wildlife"] + deltas["habitat"] + deltas["nature_tokens"]
            - deltas["total"]
        ) > 1e-9:
            raise ValueError(f"paired category delta does not sum for seed {seed}")
        per_seed.append({"seed": seed, "left_minus_right": deltas})
    for category in CATEGORIES:
        category_stats[category] = paired_delta_stats(
            [row["left_minus_right"][category] for row in per_seed]
        )
    if abs(
        category_stats["wildlife"]["mean"]
        + category_stats["habitat"]["mean"]
        + category_stats["nature_tokens"]["mean"]
        - category_stats["total"]["mean"]
    ) > 1e-9:
        raise ValueError("mean paired category deltas do not sum to total")
    return {
        "status": "pass",
        "comparison": "paired_game_score_categories_v1",
        "label": label,
        "ruleset_id": RULESET_ID,
        "source_revision": source_revision,
        "search": _normalized_search(left_report["search"], ledger=False),
        "seeds": seeds,
        "left": {
            "experiment_id": left_report["experiment_id"],
            "report": str(left_report_path),
            "report_sha256": _sha256(left_report_path),
            "games": str(left_games_path),
            "games_sha256": _sha256(left_games_path),
            "summary": _arm_summary(left_games),
        },
        "right": {
            "experiment_id": right_report["experiment_id"],
            "report": str(right_report_path),
            "report_sha256": _sha256(right_report_path),
            "games": str(right_games_path),
            "games_sha256": _sha256(right_games_path),
            "summary": _arm_summary(right_games),
        },
        "paired_left_minus_right": category_stats,
        "per_seed": per_seed,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    left = report["left"]
    right = report["right"]
    lines = [
        "# Paired Game-Score Category Attribution",
        "",
        f"- Comparison: {report['label']}",
        f"- Seeds: {len(report['seeds'])}",
        "",
        "| Category | Left | Right | Delta | 95% t-CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        stats = report["paired_left_minus_right"][category]
        lines.append(
            f"| {category} | {left['summary']['overall_mean'][category]:.4f} | "
            f"{right['summary']['overall_mean'][category]:.4f} | {stats['mean']:+.4f} | "
            f"[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}] |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--left-games", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--right-games", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    report = build_category_comparison(
        left_report_path=args.left_report,
        left_games_path=args.left_games,
        right_report_path=args.right_report,
        right_games_path=args.right_games,
        source_revision=args.source_revision,
        label=args.label,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.summary_out)
    print(json.dumps({"status": "pass", "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
