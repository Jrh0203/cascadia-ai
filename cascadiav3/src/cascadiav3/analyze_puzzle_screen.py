"""R2.1 puzzle-screen scorer.

Joins a candidate screen run (the `--puzzle-bank` exporter mode at candidate
serving flags, repeats=1) against the frozen mega-budget bank on the same
roots and scores the candidate by *bank regret*: for each root, the bank's
mean completed-Q of the bank-best action minus the bank value of the action
the candidate search chose. Low mean regret at parity cost is the screen
signal; a candidate must still win a preregistered paired gate before any
adoption — the screen only decides which candidates deserve one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats


def _load_roots(directory: Path) -> dict[tuple[int, int], dict[str, Any]]:
    roots: dict[tuple[int, int], dict[str, Any]] = {}
    shards = sorted(directory.glob("puzzle_seed_*.jsonl"))
    if not shards:
        raise ValueError(f"no puzzle shards under {directory}")
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "puzzle_root":
                continue
            key = (int(record["seed"]), int(record["ply"]))
            if key in roots:
                raise ValueError(f"duplicate puzzle root {key} in {shard}")
            roots[key] = record
    return roots


def analyze(bank_dir: Path, screen_dir: Path) -> dict[str, Any]:
    bank = _load_roots(bank_dir)
    screen = _load_roots(screen_dir)
    joined = sorted(set(bank) & set(screen))
    if not joined:
        raise ValueError("bank and screen share no roots")
    per_root: list[dict[str, Any]] = []
    for key in joined:
        bank_root = bank[key]
        screen_root = screen[key]
        if bank_root["action_ids"] != screen_root["action_ids"]:
            raise ValueError(
                f"action menu mismatch at root {key} — bank and screen must "
                "replay the same ledger at the same menu cap"
            )
        bank_q = [float(q) for q in bank_root["mean_completed_q"]]
        bank_best = max(range(len(bank_q)), key=bank_q.__getitem__)
        screen_choice = int(screen_root["repeat_chosen_indexes"][0])
        regret = bank_q[bank_best] - bank_q[screen_choice]
        per_root.append(
            {
                "seed": key[0],
                "ply": key[1],
                "bank_best_index": bank_best,
                "screen_choice_index": screen_choice,
                "regret": regret,
                "chose_bank_best": screen_choice == bank_best,
                "bank_repeat_agreement": bank_root.get("repeat_agreement"),
            }
        )
    regrets = [row["regret"] for row in per_root]
    return {
        "status": "pass",
        "roots": len(per_root),
        "bank_only_roots": len(set(bank) - set(screen)),
        "screen_only_roots": len(set(screen) - set(bank)),
        "mean_regret": sum(regrets) / len(regrets),
        "p95_regret": sorted(regrets)[int(0.95 * (len(regrets) - 1))],
        "zero_regret_rate": sum(1 for row in per_root if row["chose_bank_best"])
        / len(per_root),
        "regret_stats": paired_delta_stats(regrets),
        "bank_search": next(iter(bank.values()))["search"],
        "screen_search": next(iter(screen.values()))["search"],
        "per_root": per_root,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Puzzle-Bank Screen",
        "",
        f"Roots joined: `{report['roots']}` (bank-only `{report['bank_only_roots']}`, "
        f"screen-only `{report['screen_only_roots']}`)",
        f"Mean bank regret: `{report['mean_regret']:+.4f}` "
        f"(95% t-CI `[{report['regret_stats']['t_ci_low']:+.4f}, "
        f"{report['regret_stats']['t_ci_high']:+.4f}]`)",
        f"P95 regret: `{report['p95_regret']:+.4f}`",
        f"Chose-bank-best rate: `{report['zero_regret_rate']:.1%}`",
        "",
        "Screens rank candidates for gates; they are never promotion evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = analyze(Path(args.bank_dir), Path(args.screen_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(
        json.dumps(
            {
                "roots": report["roots"],
                "mean_regret": report["mean_regret"],
                "zero_regret_rate": report["zero_regret_rate"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
