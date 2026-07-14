"""R1.4 D1 label-movement pilot scorer.

Reads a mega-budget relabel bank (the `--puzzle-bank` exporter mode with
repeats >= 2 on a serving decision ledger) and measures how often the
mega-budget label disagrees with the serving-time choice. The preregistered
kill test (EXPERIMENT_LOG 2026-07-14) is evaluated on the repeat-STABLE
stratum only — roots where every mega repeat picked the same argmax — so a
"moved" label is a stable relabel, not mega-search noise. Repeat-unstable
roots are reported separately as the noise-flippable census D1 would target.

Bar: stable-stratum movement rate >= 0.20 keeps D1 funded. Guard: if the bar
passes but the mean mega-regret of moved stable roots is < 0.05 points, the
movement is near-tie churn and the report flags it; both numbers are part of
the preregistered verdict, and neither is promotion evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MOVEMENT_BAR = 0.20
NEAR_TIE_GUARD = 0.05
REGRET_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
# tile_count proxy: seats start at 3 tiles and place one per own turn, so the
# active seat holds 3 + ply//4 tiles when deciding. Phase cuts mirror the
# Stage 0 census (late = tile_count >= 13, the V1b gate).
_LATE_TILES = 13
_MID_TILES = 8


def _phase(ply: int) -> str:
    tiles = 3 + ply // 4
    if tiles >= _LATE_TILES:
        return "late"
    if tiles >= _MID_TILES:
        return "mid"
    return "opening"


def _load_roots(directory: Path) -> list[dict[str, Any]]:
    shards = sorted(directory.glob("puzzle_seed_*.jsonl"))
    if not shards:
        raise ValueError(f"no puzzle shards under {directory}")
    roots: dict[tuple[int, int], dict[str, Any]] = {}
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
    return [roots[key] for key in sorted(roots)]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"roots": 0}
    moved = [row for row in rows if row["moved"]]
    regrets = sorted(row["mega_regret"] for row in moved)
    summary: dict[str, Any] = {
        "roots": len(rows),
        "moved": len(moved),
        "movement_rate": len(moved) / len(rows),
    }
    if moved:
        summary["mean_regret_moved"] = sum(regrets) / len(regrets)
        summary["median_regret_moved"] = regrets[len(regrets) // 2]
        summary["p95_regret_moved"] = regrets[int(0.95 * (len(regrets) - 1))]
        summary["moved_regret_at_least"] = {
            str(threshold): sum(1 for regret in regrets if regret >= threshold)
            for threshold in REGRET_THRESHOLDS
        }
    return summary


def analyze(bank_dir: Path) -> dict[str, Any]:
    records = _load_roots(bank_dir)
    per_root: list[dict[str, Any]] = []
    for record in records:
        repeats = int(record.get("repeats", 0))
        if repeats < 2:
            raise ValueError(
                "label-movement analysis needs repeats >= 2 to measure "
                f"stability; bank root ({record['seed']}, {record['ply']}) "
                f"has repeats={repeats}"
            )
        ledger_chosen = record.get("ledger_chosen_action_id")
        if not ledger_chosen:
            raise ValueError(
                "bank root missing ledger_chosen_action_id — the bank must "
                "be generated from a serving decision ledger"
            )
        action_ids = record["action_ids"]
        try:
            serving_index = action_ids.index(ledger_chosen)
        except ValueError as error:
            raise ValueError(
                "ledger choice not in the replayed menu at root "
                f"({record['seed']}, {record['ply']}) — menu drift between "
                "serving and replay breaks the comparison"
            ) from error
        mega_q = [float(q) for q in record["mean_completed_q"]]
        mega_best = max(range(len(mega_q)), key=mega_q.__getitem__)
        ordered = sorted(mega_q, reverse=True)
        per_root.append(
            {
                "seed": int(record["seed"]),
                "ply": int(record["ply"]),
                "phase": _phase(int(record["ply"])),
                "stable": float(record["repeat_agreement"]) == 1.0,
                "moved": mega_best != serving_index,
                "mega_best_index": mega_best,
                "serving_index": serving_index,
                "mega_regret": mega_q[mega_best] - mega_q[serving_index],
                "top2_gap": ordered[0] - ordered[1],
            }
        )
    stable = [row for row in per_root if row["stable"]]
    unstable = [row for row in per_root if not row["stable"]]
    stable_summary = _summarize(stable)
    bar_pass = (
        stable_summary.get("roots", 0) > 0
        and stable_summary["movement_rate"] >= MOVEMENT_BAR
    )
    near_tie_churn = bool(
        bar_pass
        and stable_summary.get("mean_regret_moved", 0.0) < NEAR_TIE_GUARD
    )
    return {
        "roots": len(per_root),
        "search": records[0]["search"],
        "repeats": int(records[0]["repeats"]),
        "overall": _summarize(per_root),
        "stable": stable_summary,
        "unstable": _summarize(unstable),
        "unstable_fraction": len(unstable) / len(per_root),
        "by_phase": {
            phase: _summarize([row for row in stable if row["phase"] == phase])
            for phase in ("opening", "mid", "late")
        },
        "preregistered_bar": {
            "description": (
                "stable-stratum movement rate >= 0.20 keeps D1 funded; "
                "guard flags near-tie churn when mean moved regret < 0.05"
            ),
            "movement_bar": MOVEMENT_BAR,
            "near_tie_guard": NEAR_TIE_GUARD,
            "stable_movement_rate": stable_summary.get("movement_rate"),
            "bar_pass": bar_pass,
            "near_tie_churn_flag": near_tie_churn,
        },
        "per_root": per_root,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    bar = report["preregistered_bar"]
    stable = report["stable"]
    lines = [
        "# D1 Label-Movement Pilot",
        "",
        f"Roots: `{report['roots']}` (repeat-stable `{stable.get('roots', 0)}`, "
        f"unstable fraction `{report['unstable_fraction']:.3f}`)",
        f"Stable movement rate: `{stable.get('movement_rate', 0.0):.3f}` "
        f"vs bar `{bar['movement_bar']:.2f}` — "
        f"**{'PASS' if bar['bar_pass'] else 'FAIL'}**"
        + (" (near-tie churn flagged)" if bar["near_tie_churn_flag"] else ""),
    ]
    if stable.get("moved"):
        lines.append(
            f"Moved stable roots: `{stable['moved']}`; mega-regret mean "
            f"`{stable['mean_regret_moved']:.4f}`, median "
            f"`{stable['median_regret_moved']:.4f}`, p95 "
            f"`{stable['p95_regret_moved']:.4f}`"
        )
    lines += [
        "",
        "| phase (stable) | roots | movement | mean regret moved |",
        "|---|---|---|---|",
    ]
    for phase in ("opening", "mid", "late"):
        block = report["by_phase"][phase]
        if not block.get("roots"):
            lines.append(f"| {phase} | 0 | — | — |")
            continue
        regret = block.get("mean_regret_moved")
        lines.append(
            f"| {phase} | {block['roots']} | {block['movement_rate']:.3f} | "
            f"{regret:.4f} |" if regret is not None
            else f"| {phase} | {block['roots']} | {block['movement_rate']:.3f} | — |"
        )
    lines += [
        "",
        "Pilot measurement only — never promotion evidence; the funding "
        "decision applies the preregistered bar above.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = analyze(Path(args.bank_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(
        json.dumps(
            {
                "roots": report["roots"],
                "unstable_fraction": report["unstable_fraction"],
                "stable_movement_rate": report["preregistered_bar"][
                    "stable_movement_rate"
                ],
                "bar_pass": report["preregistered_bar"]["bar_pass"],
                "near_tie_churn_flag": report["preregistered_bar"][
                    "near_tie_churn_flag"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
