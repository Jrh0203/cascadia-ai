#!/usr/bin/env python3
"""Resumable all-seat benchmark adapter for the frozen v1 champion binary."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAYER = re.compile(
    r"SYMPLAYER p=(?P<seat>\d+) base=(?P<base>\d+) bonus=(?P<bonus>\d+) "
    r"hab=(?P<habitat>\d+) wl=(?P<wildlife>\d+) tok=(?P<tokens>\d+) "
    r"bear=(?P<bear>\d+) elk=(?P<elk>\d+) salmon=(?P<salmon>\d+) "
    r"hawk=(?P<hawk>\d+) fox=(?P<fox>\d+)"
)


@dataclass(frozen=True)
class Config:
    binary: Path
    weights: Path
    games: int
    first_seed: int
    jobs: int
    progress: Path
    output: Path


def parse_players(text: str) -> list[dict[str, int]]:
    """Parse exactly one complete four-seat v1 symmetric game."""
    players = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in PLAYER.finditer(text)
    ]
    players.sort(key=lambda player: player["seat"])
    if [player["seat"] for player in players] != [0, 1, 2, 3]:
        raise ValueError(f"expected seats 0-3, found {[player['seat'] for player in players]}")
    return players


def run_game(config: Config, seed: int) -> dict[str, Any]:
    """Execute one deterministic v1 game and retain enough evidence to audit it."""
    environment = os.environ.copy()
    environment["CASCADIA_SEAT_STRATEGIES"] = ":".join(["mce_wide_v1"] * 4)
    environment["CASCADIA_SEED_OFFSET"] = str(seed)
    command = [
        str(config.binary),
        "1",
        "--nnue",
        "--weights",
        str(config.weights),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"seed {seed} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    players = parse_players(completed.stdout + "\n" + completed.stderr)
    return {
        "seed_offset": seed,
        "elapsed_seconds": elapsed,
        "players": players,
    }


def summarize(config: Config, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build game-block and all-seat statistics without trusting v1 summary output."""
    results.sort(key=lambda result: result["seed_offset"])
    players = [player for result in results for player in result["players"]]
    scores = [player["base"] for player in players]
    game_means = [
        statistics.fmean(player["base"] for player in result["players"]) for result in results
    ]
    mean = statistics.fmean(game_means)
    sample_sd = statistics.stdev(game_means) if len(game_means) > 1 else 0.0
    standard_error = sample_sd / math.sqrt(len(game_means)) if game_means else 0.0
    sorted_scores = sorted(scores)

    def percentile(probability: float) -> float:
        if not sorted_scores:
            return 0.0
        index = probability * (len(sorted_scores) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        fraction = index - lower
        return sorted_scores[lower] * (1 - fraction) + sorted_scores[upper] * fraction

    categories = ["habitat", "wildlife", "tokens", "bear", "elk", "salmon", "hawk", "fox"]
    return {
        "schema_version": 1,
        "status": "complete",
        "comparability": "v1 reference only; rules and accounting are not canonical v2",
        "strategy": "v1 mce_wide_v1 + nnue_weights_v4opp_modal_iter3.bin",
        "games": len(results),
        "seat_games": len(players),
        "first_seed": config.first_seed,
        "last_seed": config.first_seed + len(results) - 1,
        "binary": {
            "path": str(config.binary),
            "sha256": checksum(config.binary),
        },
        "weights": {
            "path": str(config.weights),
            "sha256": checksum(config.weights),
        },
        "protocol": {
            "environment": {
                "CASCADIA_SEAT_STRATEGIES": ":".join(["mce_wide_v1"] * 4),
                "CASCADIA_SEED_OFFSET": "one deterministic value per game",
            },
            "command": "<binary> 1 --nnue --weights <weights>",
            "accounting": "adapter parses all SYMPLAYER rows; v1 printed summary is seat 0 only",
        },
        "mean_score": mean,
        "game_block_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "seat_standard_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "percentiles": {
            "p10": percentile(0.10),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
        },
        "category_means": {
            category: statistics.fmean(player[category] for player in players)
            for category in categories
        },
        "total_wall_seconds": sum(result["elapsed_seconds"] for result in results),
        "mean_wall_seconds_per_game": statistics.fmean(
            result["elapsed_seconds"] for result in results
        ),
        "results": results,
    }


def load_progress(path: Path) -> dict[int, dict[str, Any]]:
    """Recover every durable completed seed from append-only JSONL."""
    if not path.exists():
        return {}
    recovered: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                result = json.loads(line)
                recovered[int(result["seed_offset"])] = result
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid progress line {line_number}: {error}") from error
    return recovered


def append_progress(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target-mid-v4/release/cascadia-cli"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("nnue_weights_v4opp_modal_iter3.bin"),
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("artifacts/v1-champion-50/progress.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v1-champion-50/report.json"),
    )
    args = parser.parse_args()
    config = Config(
        binary=args.binary.resolve(),
        weights=args.weights.resolve(),
        games=args.games,
        first_seed=args.first_seed,
        jobs=args.jobs,
        progress=args.progress,
        output=args.output,
    )
    if config.games <= 0 or config.jobs <= 0:
        raise ValueError("games and jobs must be positive")
    if not config.binary.is_file() or not config.weights.is_file():
        raise FileNotFoundError("v1 binary and weights must exist")

    completed = load_progress(config.progress)
    requested = range(config.first_seed, config.first_seed + config.games)
    pending = [seed for seed in requested if seed not in completed]
    print(
        f"v1 champion: {len(completed)}/{config.games} recovered, "
        f"{len(pending)} pending, jobs={config.jobs}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.jobs) as executor:
        futures = {executor.submit(run_game, config, seed): seed for seed in pending}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            result = future.result()
            append_progress(config.progress, result)
            completed[seed] = result
            mean = statistics.fmean(
                player["base"]
                for completed_result in completed.values()
                for player in completed_result["players"]
            )
            print(
                f"completed seed={seed} ({len(completed)}/{config.games}), "
                f"all-seat mean={mean:.3f}, wall={result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results = [completed[seed] for seed in requested]
    report = summarize(config, results)
    write_atomic(config.output, report)
    print(json.dumps({key: report[key] for key in ("games", "mean_score", "ci95")}, indent=2))


if __name__ == "__main__":
    main()
