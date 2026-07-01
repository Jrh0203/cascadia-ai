#!/usr/bin/env python3
"""Validate and analyze the frozen full-legal decision-regret audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "full-legal-decision-regret-audit-v1-20260615"
AUDIT_PROTOCOL_ID = "full-legal-decision-regret-audit-v1"
RULES_PROTOCOL_ID = "cascadia-aaaaa-4p-base-v1"
FROZEN_FIRST_SEED = 61_000
FROZEN_GAMES = 13
FROZEN_DECISIONS = 1_040
FROZEN_REALIZED_HIDDEN_PER_GAME = 3
FROZEN_CONFIG: dict[str, Any] = {
    "protocol_id": AUDIT_PROTOCOL_ID,
    "champion_rollouts": 600,
    "screen_limit": 64,
    "sentinel_count": 16,
    "substantial_rollouts": 1_200,
    "high_confidence_limit": 8,
    "high_confidence_rollouts": 4_800,
    "audited_completed_turns": None,
    "realized_hidden_completed_turns": [12, 39, 66],
    "paid_wipe_determinizations": 8,
    "paid_wipe_followup_determinizations": 2,
    "paid_wipe_followup_width": 3,
}
FROZEN_OWNERSHIP = {
    "john1": range(61_000, 61_005),
    "john2": range(61_005, 61_009),
    "john3": range(61_009, 61_013),
}
WILDLIFE = ("Bear", "Elk", "Salmon", "Hawk", "Fox")
SHARD_FILENAME_RE = re.compile(r"^seed-[0-9]+\.json$")
TIME_REAL_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?) real\s+", re.MULTILINE)
TIME_USER_RE = re.compile(
    r"^\s*[0-9]+(?:\.[0-9]+)? real\s+([0-9]+(?:\.[0-9]+)?) user",
    re.MULTILINE,
)
TIME_SYS_RE = re.compile(
    r"^\s*[0-9]+(?:\.[0-9]+)? real\s+[0-9]+(?:\.[0-9]+)? user\s+"
    r"([0-9]+(?:\.[0-9]+)?) sys",
    re.MULTILINE,
)
TIME_RSS_RE = re.compile(r"^\s*([0-9]+)\s+maximum resident set size", re.MULTILINE)
TIME_SWAPS_RE = re.compile(r"^\s*([0-9]+)\s+swaps", re.MULTILINE)
SYSTEM_SWAP_USED_RE = re.compile(r"\bused = ([0-9]+(?:\.[0-9]+)?)([KMG])\b")


@dataclass(frozen=True)
class AuditInput:
    path: Path
    host: str
    seed: int
    sha256: str
    screen_contract_sha256: str
    bytes: int
    config: dict[str, Any]
    provenance: dict[str, Any]
    summary: dict[str, Any]
    final_scores: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def screen_contract_sha256(
    report: dict[str, Any],
    *,
    completed_turns: set[int] | None = None,
) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        digest.update(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        )
        digest.update(b"\n")

    update(
        {
            "schema_version": 1,
            "fingerprint": "full-legal-screen-contract-v1",
            "completed_turns": (sorted(completed_turns) if completed_turns is not None else None),
        }
    )
    for game in report["games"]:
        update(
            {
                "raw_seed": game["raw_seed"],
                "final_scores": game["final_scores"],
                "final_state_blake3": game["final_state_blake3"],
            }
        )
        for decision in game["decisions"]:
            if (
                completed_turns is not None
                and int(decision["completed_turns"]) not in completed_turns
            ):
                continue
            update(
                {
                    key: decision[key]
                    for key in (
                        "raw_seed",
                        "completed_turns",
                        "current_player",
                        "personal_turn",
                        "phase",
                        "public_state_blake3",
                        "staged_public_state_blake3",
                        "prelude",
                        "current_score",
                        "public_supply",
                        "opponent_eligible_wildlife_slots",
                        "opponent_placed_wildlife",
                        "action_count",
                        "champion_frontier_count",
                        "champion_action_hash",
                    )
                }
            )
            for action in decision["actions"]:
                update(
                    {
                        key: action[key]
                        for key in (
                            "canonical_index",
                            "canonical_hash",
                            "same_slot_independent",
                            "exact_score_delta",
                            "exact_resulting_score",
                            "action",
                            "drafted_tile_id",
                            "drafted_wildlife",
                            "model_immediate_score",
                            "model_remaining_value",
                            "screen_value",
                            "screen_rank",
                            "visible_wildlife_count",
                            "public_bag_wildlife_count",
                            "uniform_market_survival_proxy",
                        )
                    }
                )
                update(
                    {
                        "canonical_hash": action["canonical_hash"],
                        "champion_selected": action["sources"]["champion_selected"],
                        "champion_frontier": action["sources"]["champion_frontier"],
                        "champion_frontier_r600": action["champion_frontier_r600"],
                    }
                )
    return digest.hexdigest()


def discover_reports(input_dirs: Iterable[Path]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for directory in input_dirs:
        host = directory.name
        for path in sorted(
            path for path in directory.glob("seed-*.json") if SHARD_FILENAME_RE.fullmatch(path.name)
        ):
            found.append((path, host))
    return found


def validate_with_binary(binary: Path, path: Path) -> None:
    result = subprocess.run(
        [str(binary), "validate", "--input", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Rust validation failed for {path}: {result.stderr.strip() or result.stdout.strip()}"
        )


def validate_score(score: dict[str, Any], context: str) -> None:
    habitat = score.get("habitat")
    wildlife = score.get("wildlife")
    if not isinstance(habitat, list) or len(habitat) != 5:
        raise ValueError(f"{context}: habitat score is not length five")
    if not isinstance(wildlife, list) or len(wildlife) != 5:
        raise ValueError(f"{context}: wildlife score is not length five")
    expected = sum(habitat) + sum(wildlife) + int(score["nature_tokens"])
    if int(score["base_total"]) != expected:
        raise ValueError(f"{context}: base score does not match its decomposition")
    if int(score.get("habitat_bonus", [0, 0, 0, 0, 0])[0]) != 0 or any(
        int(value) != 0 for value in score.get("habitat_bonus", [])
    ):
        raise ValueError(f"{context}: habitat bonuses must be disabled")


def expected_host_for_seed(seed: int) -> str:
    for host, seeds in FROZEN_OWNERSHIP.items():
        if seed in seeds:
            return host
    raise ValueError(f"seed {seed} is outside frozen ownership")


def validate_report_contract(
    report: dict[str, Any],
    *,
    path: Path,
    host: str,
    expected_decisions_per_game: int,
    require_frozen_config: bool,
) -> int:
    if report.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported shard schema")
    if report.get("games_requested") != 1 or len(report.get("games", [])) != 1:
        raise ValueError(f"{path}: each resumable artifact must contain exactly one game")
    seed = int(report["first_seed"])
    game = report["games"][0]
    if int(game.get("raw_seed", -1)) != seed:
        raise ValueError(f"{path}: game seed and shard seed disagree")
    if require_frozen_config and report.get("config") != FROZEN_CONFIG:
        raise ValueError(f"{path}: frozen audit configuration drifted")
    if seed in range(FROZEN_FIRST_SEED, FROZEN_FIRST_SEED + FROZEN_GAMES):
        owner = expected_host_for_seed(seed)
        if host != owner:
            raise ValueError(f"{path}: seed {seed} belongs to {owner}, not {host}")
    decisions = game.get("decisions", [])
    if len(decisions) != expected_decisions_per_game:
        raise ValueError(
            f"{path}: found {len(decisions)} decisions, expected {expected_decisions_per_game}"
        )
    if int(report["summary"]["decisions"]) != expected_decisions_per_game:
        raise ValueError(f"{path}: shard summary decision count is inconsistent")
    if int(report["summary"]["games"]) != 1:
        raise ValueError(f"{path}: shard summary must contain one game")
    if int(report["bridge_diagnostics"]["fallbacks"]) != 0:
        raise ValueError(f"{path}: legacy bridge fallback occurred")
    if int(report["batch_diagnostics"]["policy_fallbacks"]) != 0:
        raise ValueError(f"{path}: rollout policy fallback occurred")
    if int(report["batch_diagnostics"]["bootstrapped_samples"]) != 0:
        raise ValueError(f"{path}: full-terminal audit unexpectedly bootstrapped a sample")
    if not math.isfinite(float(report["summary"]["elapsed_seconds"])):
        raise ValueError(f"{path}: non-finite elapsed time")
    for seat, score in enumerate(game.get("final_scores", [])):
        validate_score(score, f"{path}: final seat {seat}")
    if len(game.get("final_scores", [])) != 4:
        raise ValueError(f"{path}: terminal game must have four seat scores")
    for decision in decisions:
        if len(decision.get("actions", [])) != int(decision["action_count"]):
            raise ValueError(f"{path}: action count mismatch at turn {decision['completed_turns']}")
        if not decision["actions"]:
            raise ValueError(f"{path}: no actions at turn {decision['completed_turns']}")
        if any(not math.isfinite(float(action["screen_value"])) for action in decision["actions"]):
            raise ValueError(
                f"{path}: non-finite screen value at turn {decision['completed_turns']}"
            )
        has_paid = decision.get("paid_wipe_diagnostic") is not None
        paid_enabled = int(report["config"]["paid_wipe_determinizations"]) > 0
        if has_paid != (paid_enabled and int(decision["current_score"]["nature_tokens"]) > 0):
            raise ValueError(
                f"{path}: paid-wipe coverage mismatch at turn {decision['completed_turns']}"
            )
    return seed


def load_inputs(
    input_dirs: list[Path],
    *,
    binary: Path | None,
    expected_decisions_per_game: int,
    require_frozen_config: bool,
) -> list[AuditInput]:
    inputs: list[AuditInput] = []
    for path, host in discover_reports(input_dirs):
        if binary is not None:
            validate_with_binary(binary, path)
        report = json.loads(path.read_text())
        seed = validate_report_contract(
            report,
            path=path,
            host=host,
            expected_decisions_per_game=expected_decisions_per_game,
            require_frozen_config=require_frozen_config,
        )
        game = report["games"][0]
        inputs.append(
            AuditInput(
                path=path,
                host=host,
                seed=seed,
                sha256=sha256_file(path),
                screen_contract_sha256=screen_contract_sha256(report),
                bytes=path.stat().st_size,
                config=report["config"],
                provenance=report["provenance"],
                summary=report["summary"],
                final_scores=game["final_scores"],
                decisions=[
                    {
                        **compact_decision(seed, decision),
                        "host": host,
                    }
                    for decision in game["decisions"]
                ],
            )
        )
    inputs.sort(key=lambda item: item.seed)
    if len({item.seed for item in inputs}) != len(inputs):
        raise ValueError("duplicate audit seed discovered")
    return inputs


def load_screen_contract_references(input_dirs: list[Path]) -> dict[int, str]:
    references: dict[int, str] = {}
    for path, _host in discover_reports(input_dirs):
        report = json.loads(path.read_text())
        if report.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported reference shard schema")
        if report.get("games_requested") != 1 or len(report.get("games", [])) != 1:
            raise ValueError(f"{path}: reference artifact must contain exactly one game")
        seed = int(report["first_seed"])
        if int(report["games"][0].get("raw_seed", -1)) != seed:
            raise ValueError(f"{path}: reference game seed and shard seed disagree")
        fingerprint = screen_contract_sha256(report)
        previous = references.setdefault(seed, fingerprint)
        if previous != fingerprint:
            raise ValueError(f"conflicting screen-contract references for seed {seed}")
    return references


def draft_kind(action: dict[str, Any]) -> str:
    draft = action["action"]["draft"]
    if len(draft) != 1:
        raise ValueError("turn action draft must contain one variant")
    return next(iter(draft))


def action_change_kind(champion: dict[str, Any], winner: dict[str, Any]) -> str:
    if champion["canonical_hash"] == winner["canonical_hash"]:
        return "same_action"
    if champion["action"]["draft"] != winner["action"]["draft"]:
        return "draft_choice"
    if champion["drafted_wildlife"] != winner["drafted_wildlife"]:
        return "wildlife_choice"
    if champion["drafted_tile_id"] != winner["drafted_tile_id"]:
        return "tile_choice"
    if champion["action"]["tile"] != winner["action"]["tile"]:
        return "tile_placement"
    if champion["action"]["wildlife"] != winner["action"]["wildlife"]:
        return "wildlife_placement"
    return "prelude_or_other"


def rank_bucket(rank: int) -> str:
    if rank <= 8:
        return "01-08"
    if rank <= 32:
        return "09-32"
    if rank <= 64:
        return "33-64"
    if rank <= 128:
        return "65-128"
    return "129+"


def token_bucket(tokens: int) -> str:
    return str(tokens) if tokens <= 2 else "3+"


def supply_bucket(count: int) -> str:
    if count <= 4:
        return "00-04"
    if count <= 9:
        return "05-09"
    return "10+"


def demand_bucket(count: int) -> str:
    if count <= 2:
        return "low_0_2"
    if count <= 5:
        return "medium_3_5"
    return "high_6_plus"


def score_delta_difference(winner: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    winner_delta = winner["exact_score_delta"]
    champion_delta = champion["exact_score_delta"]
    return {
        "habitat": [
            float(winner_delta["habitat"][index]) - float(champion_delta["habitat"][index])
            for index in range(5)
        ],
        "wildlife": [
            float(winner_delta["wildlife"][index]) - float(champion_delta["wildlife"][index])
            for index in range(5)
        ],
        "nature_tokens": float(winner_delta["nature_tokens"])
        - float(champion_delta["nature_tokens"]),
        "base_total": float(winner_delta["base_total"]) - float(champion_delta["base_total"]),
    }


def hidden_summary(
    diagnostic: dict[str, Any] | None,
    champion_hash: str,
) -> dict[str, Any] | None:
    if diagnostic is None:
        return None
    scores = {
        action["canonical_hash"]: float(action["final_score"]["base_total"])
        for action in diagnostic["actions"]
    }
    realized_winner = diagnostic["realized_winner_hash"]
    public_winner = diagnostic["public_winner_hash"]
    if champion_hash not in scores or public_winner not in scores or realized_winner not in scores:
        raise ValueError("realized-hidden diagnostic omitted a required finalist")
    best = scores[realized_winner]
    return {
        "public_matches_realized": public_winner == realized_winner,
        "champion_matches_realized": champion_hash == realized_winner,
        "public_hindsight_regret": best - scores[public_winner],
        "champion_hindsight_regret": best - scores[champion_hash],
        "best_realized_score": best,
    }


def compact_decision(seed: int, decision: dict[str, Any]) -> dict[str, Any]:
    actions = {action["canonical_hash"]: action for action in decision["actions"]}
    champion = actions[decision["champion_action_hash"]]
    winner = actions[decision["best_complete_screen_hash"]]
    substantial_frontier = actions[decision["best_champion_frontier_hash"]]
    species = winner["drafted_wildlife"]
    wildlife_index = WILDLIFE.index(species)
    champion_regret = float(decision["champion_regret"]["points"])
    recorded_frontier_regret = float(decision["champion_frontier_regret"]["points"])
    champion_high = champion["high_confidence_r4800"]
    winner_high = winner["high_confidence_r4800"]
    high_confidence_frontier = [
        action
        for action in actions.values()
        if action["sources"]["champion_frontier"] and action["high_confidence_r4800"] is not None
    ]
    if champion_high is None or winner_high is None or not high_confidence_frontier:
        raise ValueError("high-confidence set omitted a required frontier comparator")
    frontier = max(
        high_confidence_frontier,
        key=lambda action: (
            float(action["high_confidence_r4800"]["mean"]),
            action["canonical_hash"],
        ),
    )
    frontier_mean = float(frontier["high_confidence_r4800"]["mean"])
    frontier_regret = max(0.0, float(winner_high["mean"]) - frontier_mean)
    retained_regret = float(decision["retained_screen_regret"]["points"])
    paid = decision.get("paid_wipe_diagnostic")
    paid_summary = None
    if paid is not None:
        paid_summary = {
            "expected_gain_over_stop": float(paid["expected_gain_over_stop"]),
            "preferred_probability": float(paid["paid_wipe_preferred_probability"]),
            "best_option_mask": int(paid["best_option_mask"]),
            "best_slot_count": int(paid["best_option_mask"]).bit_count(),
            "initial_tokens": int(paid["initial_nature_tokens"]),
            "recursive_followup_exercised": bool(paid["recursive_followup_exercised"]),
        }
    return {
        "seed": seed,
        "completed_turns": int(decision["completed_turns"]),
        "current_player": int(decision["current_player"]),
        "personal_turn": int(decision["personal_turn"]),
        "phase": decision["phase"],
        "action_count": int(decision["action_count"]),
        "champion_frontier_count": int(decision["champion_frontier_count"]),
        "substantial_count": int(decision["substantial_count"]),
        "high_confidence_count": int(decision["high_confidence_count"]),
        "top_screen_recalled_winner": bool(decision["top_screen_recalled_winner"]),
        "champion_regret": champion_regret,
        "frontier_regret": frontier_regret,
        "recorded_substantial_frontier_regret": recorded_frontier_regret,
        "frontier_comparator_corrected": frontier["canonical_hash"]
        != substantial_frontier["canonical_hash"],
        "selection_regret": max(0.0, champion_regret - frontier_regret),
        "retained_screen_regret": retained_regret,
        "champion_changed": champion["canonical_hash"] != winner["canonical_hash"],
        "winner_outside_frontier": not bool(winner["sources"]["champion_frontier"]),
        "winner_in_champion_frontier": bool(winner["sources"]["champion_frontier"]),
        "winner_is_rank_stratified_sentinel": bool(winner["sources"]["rank_stratified_sentinel"]),
        "winner_is_champion_selected": bool(winner["sources"]["champion_selected"]),
        "top64_or_champion_frontier_recalled_winner": bool(
            winner["sources"]["top_complete_screen"] or winner["sources"]["champion_frontier"]
        ),
        "winner_screen_rank": int(winner["screen_rank"]),
        "champion_screen_rank": int(champion["screen_rank"]),
        "winner_rank_bucket": rank_bucket(int(winner["screen_rank"])),
        "champion_rank_bucket": rank_bucket(int(champion["screen_rank"])),
        "winner_draft_kind": draft_kind(winner),
        "champion_draft_kind": draft_kind(champion),
        "winner_wildlife": species,
        "champion_wildlife": champion["drafted_wildlife"],
        "change_kind": action_change_kind(champion, winner),
        "current_tokens": int(decision["current_score"]["nature_tokens"]),
        "token_bucket": token_bucket(int(decision["current_score"]["nature_tokens"])),
        "winner_visible_count": int(winner["visible_wildlife_count"]),
        "winner_public_bag_count": int(winner["public_bag_wildlife_count"]),
        "winner_supply_bucket": supply_bucket(int(winner["public_bag_wildlife_count"])),
        "winner_survival_proxy": float(winner["uniform_market_survival_proxy"]),
        "winner_opponent_slots": int(decision["opponent_eligible_wildlife_slots"][wildlife_index]),
        "winner_opponent_placed": int(decision["opponent_placed_wildlife"][wildlife_index]),
        "winner_demand_bucket": demand_bucket(
            int(decision["opponent_eligible_wildlife_slots"][wildlife_index])
        ),
        "winner_minus_champion_immediate": float(winner["exact_resulting_score"]["base_total"])
        - float(champion["exact_resulting_score"]["base_total"]),
        "winner_minus_champion_screen": float(winner["screen_value"])
        - float(champion["screen_value"]),
        "winner_minus_champion_score_delta": score_delta_difference(winner, champion),
        "paid": paid_summary,
        "hidden": hidden_summary(
            decision.get("realized_hidden_future"),
            decision["champion_action_hash"],
        ),
        "timings": {key: float(value) for key, value in decision["timings"].items()},
        "champion_hash": champion["canonical_hash"],
        "winner_hash": winner["canonical_hash"],
        "frontier_hash": frontier["canonical_hash"],
        "substantial_frontier_hash": substantial_frontier["canonical_hash"],
    }


def block_statistic(
    decisions: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = [decision for decision in decisions if predicate is None or predicate(decision)]
    if not selected:
        return {
            "count": 0,
            "games": 0,
            "mean": None,
            "game_block_standard_error": None,
            "normal_confidence_95": [None, None],
            "bootstrap_confidence_95": [None, None],
        }
    grouped: dict[int, list[float]] = defaultdict(list)
    for decision in selected:
        grouped[int(decision["seed"])].append(float(value(decision)))
    seeds = sorted({int(decision["seed"]) for decision in decisions})
    sums = np.array([sum(grouped.get(seed, [])) for seed in seeds], dtype=np.float64)
    counts = np.array([len(grouped.get(seed, [])) for seed in seeds], dtype=np.int64)
    mean = float(sums.sum() / counts.sum())
    per_game = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts != 0,
    )
    finite_game_means = per_game[np.isfinite(per_game)]
    if len(finite_game_means) > 1:
        standard_error = float(
            np.std(finite_game_means, ddof=1) / math.sqrt(len(finite_game_means))
        )
    else:
        standard_error = 0.0
    rng = np.random.default_rng(bootstrap_seed)
    samples: list[np.ndarray] = []
    remaining = bootstrap_samples
    while remaining:
        batch_size = min(remaining, 4_096)
        indices = rng.integers(0, len(seeds), size=(batch_size, len(seeds)))
        sampled_sums = sums[indices].sum(axis=1)
        sampled_counts = counts[indices].sum(axis=1)
        valid = sampled_counts > 0
        samples.append(sampled_sums[valid] / sampled_counts[valid])
        remaining -= batch_size
    bootstrap = np.concatenate(samples) if samples else np.array([mean])
    bootstrap_ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "count": len(selected),
        "games": len(grouped),
        "mean": mean,
        "game_block_standard_error": standard_error,
        "normal_confidence_95": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "bootstrap_confidence_95": [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
    }


def grouped_regret(
    decisions: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    categories = sorted({str(decision[field]) for decision in decisions})
    result: dict[str, Any] = {}
    for index, category in enumerate(categories):

        def predicate(
            decision: dict[str, Any],
            *,
            expected_category: str = category,
        ) -> bool:
            return str(decision[field]) == expected_category

        result[category] = {
            "champion_regret": block_statistic(
                decisions,
                lambda decision: decision["champion_regret"],
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 1,
            ),
            "frontier_regret": block_statistic(
                decisions,
                lambda decision: decision["frontier_regret"],
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 2,
            ),
            "selection_regret": block_statistic(
                decisions,
                lambda decision: decision["selection_regret"],
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 3,
            ),
            "winner_outside_frontier_rate": block_statistic(
                decisions,
                lambda decision: float(decision["winner_outside_frontier"]),
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 4,
            ),
            "action_change_rate": block_statistic(
                decisions,
                lambda decision: float(decision["champion_changed"]),
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 5,
            ),
            "retained_screen_regret": block_statistic(
                decisions,
                lambda decision: decision["retained_screen_regret"],
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 6,
            ),
            "screen_recall": block_statistic(
                decisions,
                lambda decision: float(decision["top_screen_recalled_winner"]),
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 7,
            ),
            "screen_or_frontier_union_recall": block_statistic(
                decisions,
                lambda decision: float(decision["top64_or_champion_frontier_recalled_winner"]),
                predicate=predicate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index * 11 + 8,
            ),
        }
    return result


def parse_time_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")

    def match_float(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    rss = TIME_RSS_RE.search(text)
    swaps = TIME_SWAPS_RE.search(text)
    return {
        "real_seconds": match_float(TIME_REAL_RE),
        "user_seconds": match_float(TIME_USER_RE),
        "system_seconds": match_float(TIME_SYS_RE),
        "maximum_resident_bytes": int(rss.group(1)) if rss else None,
        "swaps": int(swaps.group(1)) if swaps else None,
        "sha256": sha256_file(path),
    }


def parse_system_swap_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace").strip()
    match = SYSTEM_SWAP_USED_RE.search(text)
    if match is None:
        return {
            "used_bytes": None,
            "raw": text,
            "sha256": sha256_file(path),
        }
    scale = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }[match.group(2)]
    return {
        "used_bytes": int(float(match.group(1)) * scale),
        "raw": text,
        "sha256": sha256_file(path),
    }


def host_utilization(inputs: list[AuditInput]) -> dict[str, Any]:
    by_host: dict[str, list[AuditInput]] = defaultdict(list)
    for item in inputs:
        by_host[item.host].append(item)
    result: dict[str, Any] = {}
    for host, items in sorted(by_host.items()):
        timing_records = []
        for item in items:
            timing = parse_time_file(item.path.with_suffix(".time"))
            if timing is not None:
                timing_records.append({"seed": item.seed, **timing})
        timing_telemetry_complete = len(timing_records) == len(items) and all(
            all(
                record[field] is not None
                for field in (
                    "real_seconds",
                    "user_seconds",
                    "system_seconds",
                    "maximum_resident_bytes",
                    "swaps",
                )
            )
            for record in timing_records
        )
        productive = sum(float(item.summary["elapsed_seconds"]) for item in items)
        measured_wall = sum(
            float(record["real_seconds"])
            for record in timing_records
            if record["real_seconds"] is not None
        )
        swap_before = parse_system_swap_file(items[0].path.parent / "swap-before.txt")
        swap_after = parse_system_swap_file(items[0].path.parent / "swap-after.txt")
        swap_delta = None
        if (
            swap_before is not None
            and swap_before["used_bytes"] is not None
            and swap_after is not None
            and swap_after["used_bytes"] is not None
        ):
            swap_delta = swap_after["used_bytes"] - swap_before["used_bytes"]
        result[host] = {
            "assigned_games": len(FROZEN_OWNERSHIP.get(host, [])),
            "completed_games": len(items),
            "productive_wall_seconds": productive,
            "measured_process_wall_seconds": measured_wall,
            "runner_overhead_seconds": max(0.0, measured_wall - productive),
            "idle_with_work_queued_seconds": 0.0 if len(items) else None,
            "failures_or_retries_observed": 0,
            "maximum_resident_bytes": max(
                (
                    int(record["maximum_resident_bytes"])
                    for record in timing_records
                    if record["maximum_resident_bytes"] is not None
                ),
                default=None,
            ),
            "timing_telemetry_complete": timing_telemetry_complete,
            "process_swaps": (
                sum(int(record["swaps"]) for record in timing_records)
                if timing_telemetry_complete
                else None
            ),
            "system_swap_before": swap_before,
            "system_swap_after": swap_after,
            "system_swap_delta_bytes": swap_delta,
            "timings": timing_records,
        }
    return result


def action_score_delta_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "habitat": [
            statistics.fmean(
                decision["winner_minus_champion_score_delta"]["habitat"][index]
                for decision in decisions
            )
            for index in range(5)
        ],
        "wildlife": [
            statistics.fmean(
                decision["winner_minus_champion_score_delta"]["wildlife"][index]
                for decision in decisions
            )
            for index in range(5)
        ],
        "nature_tokens": statistics.fmean(
            decision["winner_minus_champion_score_delta"]["nature_tokens"] for decision in decisions
        ),
        "base_total": statistics.fmean(
            decision["winner_minus_champion_score_delta"]["base_total"] for decision in decisions
        ),
    }


def analyze(
    inputs: list[AuditInput],
    *,
    experiment_id: str = EXPERIMENT_ID,
    expected_first_seed: int,
    expected_games: int,
    expected_decisions: int,
    expected_hidden_per_game: int,
    bootstrap_samples: int,
    reference_screen_contract_by_seed: dict[int, str] | None = None,
    require_dominant_error_gate: bool = True,
) -> dict[str, Any]:
    expected_seeds = set(range(expected_first_seed, expected_first_seed + expected_games))
    found_seeds = {item.seed for item in inputs}
    if found_seeds != expected_seeds:
        raise ValueError(
            f"seed coverage mismatch; missing={sorted(expected_seeds - found_seeds)}, "
            f"extra={sorted(found_seeds - expected_seeds)}"
        )
    if (
        reference_screen_contract_by_seed is not None
        and set(reference_screen_contract_by_seed) != expected_seeds
    ):
        reference_seeds = set(reference_screen_contract_by_seed)
        raise ValueError(
            "reference seed coverage mismatch; "
            f"missing={sorted(expected_seeds - reference_seeds)}, "
            f"extra={sorted(reference_seeds - expected_seeds)}"
        )
    configurations = {
        json.dumps(item.config, sort_keys=True, separators=(",", ":")) for item in inputs
    }
    if len(configurations) != 1:
        raise ValueError("audit artifacts do not share one evaluation configuration")
    config = inputs[0].config
    screen_limit = int(config["screen_limit"])
    paid_wipe_enabled = int(config["paid_wipe_determinizations"]) > 0
    decisions: list[dict[str, Any]] = []
    final_scores: list[float] = []
    provenance_sets: dict[str, set[str]] = defaultdict(set)
    source_context_sets: dict[str, set[str]] = defaultdict(set)
    for item in inputs:
        provenance = item.provenance
        provenance_sets["executable_blake3"].add(provenance["executable_blake3"])
        provenance_sets["model_json_blake3"].add(provenance["model_json_blake3"])
        provenance_sets["model_safetensors_blake3"].add(provenance["model_safetensors_blake3"])
        provenance_sets["v2_source_blake3"].add(provenance["source"]["v2_source_blake3"])
        source_context_sets["git_revision"].add(provenance["source"]["git_revision"])
        source_context_sets["git_status_blake3"].add(provenance["source"]["git_status_blake3"])
        final_scores.extend(float(score["base_total"]) for score in item.final_scores)
        decisions.extend(item.decisions)
    if len(decisions) != expected_decisions:
        raise ValueError(f"found {len(decisions)} decisions, expected {expected_decisions}")
    if any(len(values) != 1 for values in provenance_sets.values()):
        raise ValueError("audit artifacts do not share one executable, model, and source identity")

    bootstrap_seed = 20_260_615

    def metric(key: str, offset: int) -> dict[str, Any]:
        return block_statistic(
            decisions,
            lambda decision: float(decision[key]),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )

    champion_regret = metric("champion_regret", 1)
    frontier_regret = metric("frontier_regret", 2)
    selection_regret = metric("selection_regret", 3)
    retained_regret = metric("retained_screen_regret", 4)
    top_recall = metric("top_screen_recalled_winner", 5)
    action_change = metric("champion_changed", 6)
    outside_frontier = metric("winner_outside_frontier", 7)
    union_recall = metric("top64_or_champion_frontier_recalled_winner", 8)
    sentinel_winner = metric("winner_is_rank_stratified_sentinel", 9)
    recall_widths = tuple(sorted({64, 128, 256, 512, 1_024, 2_048, screen_limit}))
    rank_recall_curve = {
        str(width): block_statistic(
            decisions,
            lambda decision, width=width: float(decision["winner_screen_rank"] <= width),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 30 + index,
        )
        for index, width in enumerate(recall_widths)
    }
    smallest_observed_width_at_98_percent = next(
        (width for width in recall_widths if float(rank_recall_curve[str(width)]["mean"]) >= 0.98),
        None,
    )

    error_components = {
        "proposal_frontier_regret": frontier_regret,
        "within_frontier_selection_regret": selection_regret,
        "top64_truncation_regret": retained_regret,
    }
    ranked_components = sorted(
        error_components.items(),
        key=lambda item: float(item[1]["mean"] or 0.0),
        reverse=True,
    )
    dominant_name, dominant_metric = ranked_components[0]
    runner_up_name, runner_up_metric = ranked_components[1]
    dominant_supported = (
        dominant_metric["bootstrap_confidence_95"][0]
        > runner_up_metric["bootstrap_confidence_95"][1]
    )

    paid_decisions = [decision for decision in decisions if decision["paid"] is not None]
    hidden_decisions = [decision for decision in decisions if decision["hidden"] is not None]
    expected_token_decisions = (
        sum(decision["current_tokens"] > 0 for decision in decisions) if paid_wipe_enabled else 0
    )

    def has_paid(decision: dict[str, Any]) -> bool:
        return decision["paid"] is not None

    def has_hidden(decision: dict[str, Any]) -> bool:
        return decision["hidden"] is not None

    paid_gain = block_statistic(
        decisions,
        lambda decision: decision["paid"]["expected_gain_over_stop"],
        predicate=has_paid,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 20,
    )
    paid_preferred = block_statistic(
        decisions,
        lambda decision: decision["paid"]["preferred_probability"],
        predicate=has_paid,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 21,
    )
    hidden_public_regret = block_statistic(
        decisions,
        lambda decision: decision["hidden"]["public_hindsight_regret"],
        predicate=has_hidden,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 22,
    )
    hidden_champion_regret = block_statistic(
        decisions,
        lambda decision: decision["hidden"]["champion_hindsight_regret"],
        predicate=has_hidden,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 23,
    )
    hidden_public_match = block_statistic(
        decisions,
        lambda decision: float(decision["hidden"]["public_matches_realized"]),
        predicate=has_hidden,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 24,
    )

    decompositions = {}
    for offset, field in enumerate(
        (
            "host",
            "phase",
            "change_kind",
            "winner_draft_kind",
            "winner_wildlife",
            "winner_rank_bucket",
            "token_bucket",
            "winner_visible_count",
            "winner_supply_bucket",
            "winner_demand_bucket",
        )
    ):
        decompositions[field] = grouped_regret(
            decisions,
            field,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100 + offset * 1_000,
        )

    worst = sorted(
        decisions,
        key=lambda decision: (
            decision["champion_regret"],
            decision["frontier_regret"],
            -decision["seed"],
            -decision["completed_turns"],
        ),
        reverse=True,
    )[:25]
    worst_decisions = [
        {
            key: decision[key]
            for key in (
                "seed",
                "completed_turns",
                "current_player",
                "personal_turn",
                "phase",
                "champion_regret",
                "frontier_regret",
                "selection_regret",
                "retained_screen_regret",
                "winner_screen_rank",
                "champion_screen_rank",
                "winner_draft_kind",
                "champion_draft_kind",
                "winner_wildlife",
                "champion_wildlife",
                "change_kind",
                "current_tokens",
                "winner_opponent_slots",
                "winner_visible_count",
                "winner_public_bag_count",
                "champion_hash",
                "winner_hash",
            )
        }
        for decision in worst
    ]

    recall_passed = float(top_recall["mean"]) >= 0.98
    retained_passed = float(retained_regret["mean"]) <= 0.15
    screen_contract_by_seed = {item.seed: item.screen_contract_sha256 for item in inputs}
    screen_contract_mismatches = (
        []
        if reference_screen_contract_by_seed is None
        else [
            seed
            for seed in sorted(expected_seeds)
            if screen_contract_by_seed[seed] != reference_screen_contract_by_seed[seed]
        ]
    )
    exact_screen_contract_pairing = not screen_contract_mismatches
    gates = {
        "all_games_complete": len(inputs) == expected_games,
        "all_decisions_complete": len(decisions) == expected_decisions,
        "every_action_screened": all(decision["action_count"] > 0 for decision in decisions),
        "single_frozen_executable_model_source_set": all(
            len(values) == 1 for values in provenance_sets.values()
        ),
        "exact_screen_contract_pairing": exact_screen_contract_pairing,
        "screen_recall_at_least_98_percent": recall_passed,
        "retained_screen_mean_regret_at_most_0_15": retained_passed,
        "paid_wipe_diagnostic_complete": len(paid_decisions) == expected_token_decisions,
        "realized_hidden_diagnostic_complete": len(hidden_decisions)
        == expected_games * expected_hidden_per_game,
        "dominant_error_source_supported_by_game_block_ci": (
            dominant_supported if require_dominant_error_gate else True
        ),
    }
    substantive_passed = all(gates.values())
    first_order_headroom = float(champion_regret["mean"]) * 20.0

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "complete" if substantive_passed else "gate_failed",
        "rules_protocol_id": RULES_PROTOCOL_ID,
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "generated_at": timestamp(),
        "configuration": config,
        "coverage": {
            "first_seed": expected_first_seed,
            "last_seed": expected_first_seed + expected_games - 1,
            "games": len(inputs),
            "decisions": len(decisions),
            "actions_screened": sum(decision["action_count"] for decision in decisions),
            "paid_wipe_decisions": len(paid_decisions),
            "realized_hidden_decisions": len(hidden_decisions),
            "hosts": dict(sorted(Counter(item.host for item in inputs).items())),
        },
        "provenance": {key: next(iter(values)) for key, values in sorted(provenance_sets.items())},
        "source_context": {
            key: sorted(values) for key, values in sorted(source_context_sets.items())
        },
        "screen_contract": {
            "algorithm": "sha256-canonical-json-v1",
            "pairing_required": reference_screen_contract_by_seed is not None,
            "pairing_passed": exact_screen_contract_pairing,
            "mismatched_seeds": screen_contract_mismatches,
            "by_seed": {
                str(seed): fingerprint
                for seed, fingerprint in sorted(screen_contract_by_seed.items())
            },
            "reference_by_seed": (
                None
                if reference_screen_contract_by_seed is None
                else {
                    str(seed): fingerprint
                    for seed, fingerprint in sorted(reference_screen_contract_by_seed.items())
                }
            ),
        },
        "champion_corpus": {
            "seat_scores": len(final_scores),
            "mean_base_score": statistics.fmean(final_scores),
            "minimum_base_score": min(final_scores),
            "maximum_base_score": max(final_scores),
        },
        "public_decision_regret": {
            "champion": champion_regret,
            "best_champion_frontier": frontier_regret,
            "within_frontier_selection": selection_regret,
            "screen_limit": screen_limit,
            "retained_screen": retained_regret,
            "screen_recall": top_recall,
            "screen_or_champion_frontier_union_recall": union_recall,
            "retained_top64": retained_regret,
            "top64_recall": top_recall,
            "champion_action_change_rate": action_change,
            "winner_outside_champion_frontier_rate": outside_frontier,
            "top64_or_champion_frontier_union_recall": union_recall,
            "rank_stratified_sentinel_winner_rate": sentinel_winner,
            "screen_rank_recall_curve": rank_recall_curve,
            "smallest_observed_width_at_98_percent": (smallest_observed_width_at_98_percent),
            "maximum_observed_winner_screen_rank": max(
                decision["winner_screen_rank"] for decision in decisions
            ),
            "first_order_20_turn_headroom_points": first_order_headroom,
            "first_order_headroom_is_not_online_oracle_evidence": True,
        },
        "dominant_error": {
            "name": dominant_name,
            "runner_up": runner_up_name,
            "supported_by_nonoverlapping_bootstrap_intervals": dominant_supported,
            "required_for_substantive_gate": require_dominant_error_gate,
            "components": error_components,
        },
        "frontier_bookkeeping": {
            "decisions_where_champion_beat_r1200_best_frontier_at_r4800": sum(
                decision["frontier_comparator_corrected"] for decision in decisions
            ),
            "analysis_uses_best_r4800_frontier_finalist": True,
            "raw_r1200_selected_frontier_regret_mean": statistics.fmean(
                decision["recorded_substantial_frontier_regret"] for decision in decisions
            ),
        },
        "screen_miss_breakdown": {
            "screen_misses": sum(
                not decision["top_screen_recalled_winner"] for decision in decisions
            ),
            "top64_misses": sum(
                not decision["top_screen_recalled_winner"] for decision in decisions
            ),
            "misses_recovered_by_champion_frontier": sum(
                not decision["top_screen_recalled_winner"]
                and decision["winner_in_champion_frontier"]
                for decision in decisions
            ),
            "misses_discovered_by_rank_stratified_sentinel": sum(
                not decision["top_screen_recalled_winner"]
                and decision["winner_is_rank_stratified_sentinel"]
                for decision in decisions
            ),
            "winner_screen_rank_over_128": sum(
                decision["winner_screen_rank"] > 128 for decision in decisions
            ),
            "winner_screen_rank_over_256": sum(
                decision["winner_screen_rank"] > 256 for decision in decisions
            ),
            "winner_screen_rank_over_1024": sum(
                decision["winner_screen_rank"] > 1_024 for decision in decisions
            ),
        },
        "paid_wipe": {
            "decisions": len(paid_decisions),
            "expected_gain_over_stop": paid_gain,
            "preferred_probability": paid_preferred,
            "positive_expected_gain_decisions": sum(
                decision["paid"]["expected_gain_over_stop"] > 0.0 for decision in paid_decisions
            ),
            "recursive_followup_decisions": sum(
                decision["paid"]["recursive_followup_exercised"] for decision in paid_decisions
            ),
        },
        "realized_hidden_future": {
            "decisions": len(hidden_decisions),
            "public_hindsight_regret": hidden_public_regret,
            "champion_hindsight_regret": hidden_champion_regret,
            "public_winner_match_rate": hidden_public_match,
            "never_used_for_public_selection": True,
        },
        "winner_minus_champion_immediate_score_delta": action_score_delta_summary(decisions),
        "decompositions": decompositions,
        "timings": {
            stage: block_statistic(
                decisions,
                lambda decision, stage=stage: decision["timings"][stage],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 10_000 + index,
            )
            for index, stage in enumerate(
                (
                    "champion_seconds",
                    "enumeration_seconds",
                    "screening_seconds",
                    "substantial_seconds",
                    "high_confidence_seconds",
                    "paid_wipe_seconds",
                    "realized_hidden_seconds",
                    "total_seconds",
                )
            )
        },
        "host_utilization": host_utilization(inputs),
        "worst_champion_regret_decisions": worst_decisions,
        "substantive_gates": gates,
        "substantive_gates_passed": substantive_passed,
    }


def format_interval(metric: dict[str, Any], *, percent: bool = False) -> str:
    mean = metric["mean"]
    low, high = metric["bootstrap_confidence_95"]
    if mean is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{mean * scale:.3f}{suffix} [{low * scale:.3f}, {high * scale:.3f}]"


def render_markdown(report: dict[str, Any]) -> str:
    regret = report["public_decision_regret"]
    screen_limit = int(regret["screen_limit"])
    dominant = report["dominant_error"]
    gates = report["substantive_gates"]
    title = (
        "Full-Legal Screen-Width Recovery"
        if "screen-width-recovery" in report["experiment_id"]
        else "Full-Legal Decision Regret Audit"
    )
    lines = [
        f"# {title}",
        "",
        f"- Status: **{report['status']}**",
        (
            f"- Coverage: {report['coverage']['games']} games, "
            f"{report['coverage']['decisions']} decisions, "
            f"{report['coverage']['actions_screened']:,} canonical actions"
        ),
        (f"- Champion corpus mean: {report['champion_corpus']['mean_base_score']:.3f} base points"),
        (f"- Mean champion decision regret: {format_interval(regret['champion'])} points"),
        (
            f"- Top-{screen_limit:,} recall: "
            f"{format_interval(regret['screen_recall'], percent=True)}"
        ),
        (
            f"- Top-{screen_limit:,} plus champion-frontier union recall: "
            f"{
                format_interval(
                    regret['screen_or_champion_frontier_union_recall'],
                    percent=True,
                )
            }"
        ),
        (
            "- Rank-stratified sentinel winner rate: "
            f"{
                format_interval(
                    regret['rank_stratified_sentinel_winner_rate'],
                    percent=True,
                )
            }"
        ),
        (
            f"- Retained top-{screen_limit:,} regret: "
            f"{format_interval(regret['retained_screen'])} points"
        ),
        (
            "- Git revision/status context variants: "
            f"{len(report['source_context']['git_revision'])}/"
            f"{len(report['source_context']['git_status_blake3'])}. "
            "These are recorded context; executable, model, and source-root "
            "digests define frozen identity."
        ),
        (
            "- Smallest observed screen width reaching 98% recall: "
            f"{regret['smallest_observed_width_at_98_percent']}"
        ),
        (
            "- Maximum observed high-confidence winner screen rank: "
            f"{regret['maximum_observed_winner_screen_rank']}"
        ),
        "",
        "## Screen Recall Curve",
        "",
        "| Width | Recall | Game-block bootstrap 95% CI |",
        "|---:|---:|---:|",
    ]
    for width, metric in regret["screen_rank_recall_curve"].items():
        low, high = metric["bootstrap_confidence_95"]
        lines.append(
            f"| {int(width):,} | {metric['mean'] * 100.0:.3f}% | "
            f"[{low * 100.0:.3f}%, {high * 100.0:.3f}%] |"
        )
    lines.extend(
        [
            "",
            "## Error Decomposition",
            "",
            "| Component | Mean | Game-block bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "proposal_frontier_regret": "Champion frontier / proposal",
        "within_frontier_selection_regret": "Within-frontier selection",
        "top64_truncation_regret": (f"Complete-screen top-{screen_limit:,} truncation"),
    }
    for name, metric in dominant["components"].items():
        low, high = metric["bootstrap_confidence_95"]
        lines.append(f"| {labels[name]} | {metric['mean']:.3f} | [{low:.3f}, {high:.3f}] |")
    lines.extend(
        [
            "",
            f"- Dominant source: `{dominant['name']}`",
            (
                "- Dominance supported by non-overlapping game-block intervals: "
                f"`{dominant['supported_by_nonoverlapping_bootstrap_intervals']}`"
            ),
            (
                "- First-order 20-turn local headroom: "
                f"{regret['first_order_20_turn_headroom_points']:.3f} points. "
                "This is diagnostic, not an online-oracle score claim."
            ),
            "",
            "## Nature Tokens",
            "",
            f"- Token-bearing decisions: {report['paid_wipe']['decisions']}",
            (
                "- Expected paid-wipe gain over stopping: "
                f"{format_interval(report['paid_wipe']['expected_gain_over_stop'])} points"
            ),
            (
                "- Paid wipe preferred probability: "
                f"{format_interval(report['paid_wipe']['preferred_probability'], percent=True)}"
            ),
            (
                "- Positive expected-gain states: "
                f"{report['paid_wipe']['positive_expected_gain_decisions']}"
            ),
            "",
            "## Hindsight Diagnostic",
            "",
            (
                "- Public winner hindsight regret: "
                f"{format_interval(report['realized_hidden_future']['public_hindsight_regret'])}"
            ),
            (
                "- Champion hindsight regret: "
                f"{format_interval(report['realized_hidden_future']['champion_hindsight_regret'])}"
            ),
            (
                "- Public winner matches realized winner: "
                f"{
                    format_interval(
                        report['realized_hidden_future']['public_winner_match_rate'],
                        percent=True,
                    )
                }"
            ),
            "",
            "## Substantive Gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | `{passed}` |" for name, passed in gates.items())
    lines.extend(
        [
            "",
            "## Host Reproduction",
            "",
            (
                f"| Host | Decisions | Champion regret | Top-{screen_limit:,} recall | "
                "Retained regret | Union recall |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for host, metrics in report["decompositions"]["host"].items():
        lines.append(
            f"| {host} | {metrics['champion_regret']['count']} | "
            f"{metrics['champion_regret']['mean']:.3f} | "
            f"{metrics['screen_recall']['mean'] * 100.0:.3f}% | "
            f"{metrics['retained_screen_regret']['mean']:.3f} | "
            f"{metrics['screen_or_frontier_union_recall']['mean'] * 100.0:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Host Utilization",
            "",
            (
                "| Host | Games | Productive wall | Process wall | Peak RSS | "
                "Process swaps | System swap delta | Telemetry complete |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for host, utilization in report["host_utilization"].items():
        peak = utilization["maximum_resident_bytes"]
        peak_text = f"{peak / (1024**2):.1f} MiB" if peak is not None else "n/a"
        process_swaps = utilization["process_swaps"]
        process_swaps_text = str(process_swaps) if process_swaps is not None else "n/a"
        swap_delta = utilization["system_swap_delta_bytes"]
        swap_delta_text = f"{swap_delta / (1024**2):+.1f} MiB" if swap_delta is not None else "n/a"
        lines.append(
            f"| {host} | {utilization['completed_games']} | "
            f"{utilization['productive_wall_seconds']:.1f}s | "
            f"{utilization['measured_process_wall_seconds']:.1f}s | {peak_text} | "
            f"{process_swaps_text} | "
            f"{swap_delta_text} | `{utilization['timing_telemetry_complete']}` |"
        )
    lines.extend(
        [
            "",
            "## Highest-Regret Decisions",
            "",
            "| Seed | Turn | Phase | Regret | Frontier | Rank | Change | Wildlife |",
            "|---:|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for decision in report["worst_champion_regret_decisions"][:15]:
        lines.append(
            f"| {decision['seed']} | {decision['completed_turns']} | "
            f"{decision['phase']} | {decision['champion_regret']:.3f} | "
            f"{decision['frontier_regret']:.3f} | {decision['winner_screen_rank']} | "
            f"{decision['change_kind']} | {decision['winner_wildlife']} |"
        )
    return "\n".join(lines) + "\n"


def write_index(path: Path, inputs: list[AuditInput], report: dict[str, Any]) -> None:
    index = {
        "schema_version": 1,
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "generated_at": report["generated_at"],
        "rules_protocol_id": RULES_PROTOCOL_ID,
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "coverage": report["coverage"],
        "provenance": report["provenance"],
        "source_context": report["source_context"],
        "files": [
            {
                "seed": item.seed,
                "host": item.host,
                "path": str(item.path),
                "bytes": item.bytes,
                "sha256": item.sha256,
                "screen_contract_sha256": item.screen_contract_sha256,
                "time": parse_time_file(item.path.with_suffix(".time")),
                "stdout_sha256": (
                    sha256_file(item.path.with_suffix(".stdout"))
                    if item.path.with_suffix(".stdout").is_file()
                    else None
                ),
                "validation_sha256": (
                    sha256_file(item.path.with_suffix(".validate"))
                    if item.path.with_suffix(".validate").is_file()
                    else None
                ),
            }
            for item in inputs
        ],
    }
    write_json_atomic(path, index)


def status(input_dirs: list[Path]) -> dict[str, Any]:
    rows = []
    for path, host in discover_reports(input_dirs):
        try:
            seed = int(path.stem.removeprefix("seed-"))
        except ValueError:
            continue
        rows.append(
            {
                "host": host,
                "seed": seed,
                "bytes": path.stat().st_size,
                "validated": path.with_suffix(".validate").is_file(),
                "time_complete": TIME_REAL_RE.search(
                    path.with_suffix(".time").read_text(errors="replace")
                )
                is not None
                if path.with_suffix(".time").is_file()
                else False,
            }
        )
    rows.sort(key=lambda row: row["seed"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "completed_games": len(rows),
        "expected_games": FROZEN_GAMES,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--input-dir", type=Path, action="append", required=True)

    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument("--input", type=Path, action="append", required=True)
    fingerprint_parser.add_argument("--completed-turn", type=int, action="append")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input-dir", type=Path, action="append", required=True)
    analyze_parser.add_argument("--reference-dir", type=Path, action="append")
    analyze_parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    analyze_parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/full-legal-audit"),
    )
    analyze_parser.add_argument("--skip-rust-validation", action="store_true")
    analyze_parser.add_argument("--expected-first-seed", type=int, default=FROZEN_FIRST_SEED)
    analyze_parser.add_argument("--expected-games", type=int, default=FROZEN_GAMES)
    analyze_parser.add_argument("--expected-decisions", type=int, default=FROZEN_DECISIONS)
    analyze_parser.add_argument("--expected-decisions-per-game", type=int, default=80)
    analyze_parser.add_argument(
        "--expected-hidden-per-game",
        type=int,
        default=FROZEN_REALIZED_HIDDEN_PER_GAME,
    )
    analyze_parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    analyze_parser.add_argument("--allow-nonfrozen-config", action="store_true")
    analyze_parser.add_argument("--skip-dominant-error-gate", action="store_true")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--markdown-output", type=Path, required=True)
    analyze_parser.add_argument("--index-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fingerprint":
        completed_turns = set(args.completed_turn) if args.completed_turn is not None else None
        rows = []
        for path in args.input:
            resolved = path.resolve()
            report = json.loads(resolved.read_text())
            rows.append(
                {
                    "path": str(resolved),
                    "first_seed": int(report["first_seed"]),
                    "completed_turns": (
                        sorted(completed_turns) if completed_turns is not None else None
                    ),
                    "screen_contract_sha256": screen_contract_sha256(
                        report,
                        completed_turns=completed_turns,
                    ),
                }
            )
        print(json.dumps(rows, indent=2))
        return

    input_dirs = [path.resolve() for path in args.input_dir]
    if args.command == "status":
        print(json.dumps(status(input_dirs), indent=2))
        return

    binary = None if args.skip_rust_validation else args.binary.resolve()
    inputs = load_inputs(
        input_dirs,
        binary=binary,
        expected_decisions_per_game=args.expected_decisions_per_game,
        require_frozen_config=not args.allow_nonfrozen_config,
    )
    reference_screen_contract_by_seed = (
        None
        if args.reference_dir is None
        else load_screen_contract_references([path.resolve() for path in args.reference_dir])
    )
    report = analyze(
        inputs,
        experiment_id=args.experiment_id,
        expected_first_seed=args.expected_first_seed,
        expected_games=args.expected_games,
        expected_decisions=args.expected_decisions,
        expected_hidden_per_game=args.expected_hidden_per_game,
        bootstrap_samples=args.bootstrap_samples,
        reference_screen_contract_by_seed=reference_screen_contract_by_seed,
        require_dominant_error_gate=not args.skip_dominant_error_gate,
    )
    write_json_atomic(args.output.resolve(), report)
    markdown = args.markdown_output.resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report))
    write_index(args.index_output.resolve(), inputs, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "games": report["coverage"]["games"],
                "decisions": report["coverage"]["decisions"],
                "screen_limit": report["public_decision_regret"]["screen_limit"],
                "screen_recall": report["public_decision_regret"]["screen_recall"]["mean"],
                "champion_regret": report["public_decision_regret"]["champion"]["mean"],
                "retained_screen_regret": report["public_decision_regret"]["retained_screen"][
                    "mean"
                ],
                "substantive_gates_passed": report["substantive_gates_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
