"""Fail-closed validator for one-seed raw-game category replays.

Two corrected-rules n1024/d16 arms each lost exactly one raw per-seed game
file to the pre-durable-first temp-dir race: scalar seed 2027070908 and
distq seed 2027070962. Their aggregate reports pin the seat totals and their
80-row decision ledgers are durable, so a replay under the identical d20
contract is admissible only if it reproduces both exactly. This module
enforces the recovery contract from handoff-2026-07-09:

1. all 80 replayed chosen-action IDs match the existing decision ledger;
2. all optional-refresh decisions match;
3. all four replayed seat totals equal the report's pinned totals;
4. every seat's category components sum exactly to its total;
5. installation requires the target raw-games directory to be missing
   exactly this seed, and leaves it complete.

No category value is ever synthesized from totals; the replayed game-done
row is admitted or the validation fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


class ReplayValidationError(RuntimeError):
    pass


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def _fail(message: str) -> None:
    raise ReplayValidationError(message)


def category_components_total(score: dict[str, Any]) -> int:
    return (
        sum(int(v) for v in score["wildlife"])
        + sum(int(v) for v in score["habitat"])
        + int(score["nature_tokens"])
    )


def validate_replay(
    *,
    replayed_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    pinned_seat_totals: list[float],
    seed: int,
) -> dict[str, Any]:
    """Validate a replayed 81-row seed file against the durable evidence.

    Returns the replayed game-done row on success; raises
    ReplayValidationError on the first contract violation."""
    decisions = [r for r in replayed_rows if r.get("type") == "gumbel_decision"]
    done_rows = [r for r in replayed_rows if r.get("type") == "gumbel_game_done"]
    if len(decisions) != 80:
        _fail(f"replay has {len(decisions)} decision rows, expected 80")
    if len(done_rows) != 1:
        _fail(f"replay has {len(done_rows)} game-done rows, expected 1")
    done = done_rows[0]
    if int(done["seed"]) != seed:
        _fail(f"replay game-done seed {done['seed']} != expected {seed}")

    ledger = [r for r in ledger_rows if int(r.get("seed", -1)) == seed]
    if len(ledger) != 80:
        _fail(f"decision ledger has {len(ledger)} rows for seed {seed}, expected 80")

    ledger_by_ply = {int(r["ply"]): r for r in ledger}
    replay_by_ply = {int(r["ply"]): r for r in decisions}
    if sorted(ledger_by_ply) != sorted(replay_by_ply):
        _fail("replay and ledger ply sets differ")

    for ply in sorted(ledger_by_ply):
        want = ledger_by_ply[ply]
        got = replay_by_ply[ply]
        if want.get("ruleset_id") != got.get("ruleset_id"):
            _fail(f"ply {ply}: ruleset mismatch {got.get('ruleset_id')}")
        if want["chosen_action_id"] != got["chosen_action_id"]:
            _fail(
                f"ply {ply}: chosen action {got['chosen_action_id']} != "
                f"ledger {want['chosen_action_id']}"
            )
        if want.get("free_three_of_a_kind_choice") != got.get(
            "free_three_of_a_kind_choice"
        ):
            _fail(
                f"ply {ply}: refresh decision "
                f"{got.get('free_three_of_a_kind_choice')} != "
                f"ledger {want.get('free_three_of_a_kind_choice')}"
            )

    scores = done["scores"]
    if len(scores) != 4:
        _fail(f"replay game-done has {len(scores)} seats, expected 4")
    got_totals = [float(s["total"]) for s in scores]
    if got_totals != [float(t) for t in pinned_seat_totals]:
        _fail(f"replayed seat totals {got_totals} != pinned {pinned_seat_totals}")
    for index, score in enumerate(scores):
        component_sum = category_components_total(score)
        if component_sum != int(score["total"]):
            _fail(
                f"seat {index}: category components sum {component_sum} != "
                f"total {score['total']}"
            )
    return done


def pinned_totals_from_report(report: dict[str, Any], seed: int) -> list[float]:
    for row in report["candidate_per_seed"]:
        if int(row["seed"]) == seed:
            return [float(v) for v in row["seat_scores"]]
    _fail(f"aggregate report has no candidate_per_seed row for seed {seed}")
    raise AssertionError  # unreachable


def install_replay(
    *, replayed_path: Path, raw_games_dir: Path, seed: int, expected_total_seeds: int
) -> Path:
    existing = sorted(raw_games_dir.glob("gumbel_game_seed_*.jsonl"))
    existing_seeds = {int(p.stem.rsplit("_", 1)[1]) for p in existing}
    if seed in existing_seeds:
        _fail(f"raw games dir already contains seed {seed}; refusing to overwrite")
    if len(existing) != expected_total_seeds - 1:
        _fail(
            f"raw games dir has {len(existing)} seed files; expected exactly "
            f"{expected_total_seeds - 1} before installing seed {seed}"
        )
    target = raw_games_dir / f"gumbel_game_seed_{seed}.jsonl"
    shutil.copyfile(replayed_path, target)
    final = len(sorted(raw_games_dir.glob("gumbel_game_seed_*.jsonl")))
    if final != expected_total_seeds:
        _fail(f"after install the dir has {final} files, expected {expected_total_seeds}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replayed-file", required=True)
    parser.add_argument("--decisions-ledger", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--raw-games-dir", required=True)
    parser.add_argument("--expected-total-seeds", type=int, default=100)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy the validated file into the raw games directory",
    )
    args = parser.parse_args()

    replayed_path = Path(args.replayed_file)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    done = validate_replay(
        replayed_rows=_load_jsonl(replayed_path),
        ledger_rows=_load_jsonl(Path(args.decisions_ledger)),
        pinned_seat_totals=pinned_totals_from_report(report, args.seed),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "seed": args.seed,
                "seat_totals": [s["total"] for s in done["scores"]],
                "installed": False,
            }
        )
    )
    if args.install:
        target = install_replay(
            replayed_path=replayed_path,
            raw_games_dir=Path(args.raw_games_dir),
            seed=args.seed,
            expected_total_seeds=args.expected_total_seeds,
        )
        print(json.dumps({"status": "installed", "target": str(target)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
