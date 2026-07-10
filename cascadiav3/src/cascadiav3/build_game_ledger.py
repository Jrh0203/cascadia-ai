"""Build a complete seed-ordered game ledger + category summary from raw files.

The n1024 raw-ledger watcher died before publishing either arm's
`*_games.jsonl` / `*_category_summary.json`; this tool rebuilds them from a
complete raw-games directory (one 81-row per-seed file per report seed, e.g.
after the one-seed d20 replays are installed). It is fail-closed: every seed
in the aggregate report must have exactly one `gumbel_game_done` row and 80
decision rows, and the assembled ledger must pass the *consumer's* own
validation (`compare_game_categories._load_games`) — ruleset, search
contract, category sums, and report-total agreement — before anything is
published. The category summary is derived only from the validated ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from cascadiav3.compare_game_categories import _arm_summary, _load_games


class LedgerBuildError(RuntimeError):
    pass


def collect_done_rows(
    raw_games_dir: Path, expected_seeds: list[int]
) -> list[dict[str, Any]]:
    present = {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in raw_games_dir.glob("gumbel_game_seed_*.jsonl")
    }
    missing = sorted(set(expected_seeds) - set(present))
    extra = sorted(set(present) - set(expected_seeds))
    if missing:
        raise LedgerBuildError(f"raw games dir missing seeds: {missing}")
    if extra:
        raise LedgerBuildError(f"raw games dir has unexpected seeds: {extra}")
    done_rows: list[dict[str, Any]] = []
    for seed in sorted(expected_seeds):
        rows = [
            json.loads(line)
            for line in present[seed].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decisions = [row for row in rows if row.get("type") == "gumbel_decision"]
        done = [row for row in rows if row.get("type") == "gumbel_game_done"]
        if len(rows) != 81 or len(decisions) != 80 or len(done) != 1:
            raise LedgerBuildError(
                f"seed {seed}: {len(rows)} rows / {len(decisions)} decisions / "
                f"{len(done)} done rows; expected 81/80/1"
            )
        if int(done[0].get("seed", -1)) != seed:
            raise LedgerBuildError(f"seed {seed}: done row claims seed {done[0].get('seed')}")
        done_rows.append(done[0])
    return done_rows


def build_ledger(
    *,
    raw_games_dir: Path,
    report: dict[str, Any],
    games_out: Path,
    category_summary_out: Path,
) -> dict[str, Any]:
    expected_seeds = [int(seed) for seed in report["seeds"]]
    done_rows = collect_done_rows(raw_games_dir, expected_seeds)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in done_rows)
    games_out.parent.mkdir(parents=True, exist_ok=True)
    games_out.write_text(payload, encoding="utf-8")
    try:
        games = _load_games(games_out, report)  # the consumer's own validation
    except ValueError as error:
        games_out.unlink()
        raise LedgerBuildError(f"assembled ledger failed consumer validation: {error}")
    summary = _arm_summary(games)
    summary_payload = {
        "status": "complete",
        "experiment_id": report.get("experiment_id"),
        "ruleset_id": report.get("ruleset_id"),
        "source_revision": report.get("source_revision"),
        "games_jsonl_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "summary": summary,
    }
    category_summary_out.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-games-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--games-out", required=True)
    parser.add_argument("--category-summary-out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = build_ledger(
        raw_games_dir=Path(args.raw_games_dir),
        report=report,
        games_out=Path(args.games_out),
        category_summary_out=Path(args.category_summary_out),
    )
    print(json.dumps({"status": summary["status"], "games": summary["summary"]["games"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
