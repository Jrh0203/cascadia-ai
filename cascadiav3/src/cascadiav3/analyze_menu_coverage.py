"""R1.3a menu-coverage audit scorer.

Joins the capped-menu arm (default greedy-ranked 256-action root menu) and the
full-menu arm (`--gumbel-root-menu 0`, the entire legal set) of the same
ledger replay by (seed, ply) and asks whether the cap drops the action the
full-menu search rates best. For every joined root the capped menu must be a
subset of the full menu (same ledger, same stride); the full-run
mean-completed-Q is the shared yardstick: when the full-run argmax action is
absent from the capped menu, the regret is the full-run Q gap to the best
action the cap did keep, otherwise zero. Roots where the capped arm is
missing or the subset invariant fails are skipped and counted; more than
``MAX_SKIP_FRACTION`` of them fails the audit closed. The audit measures menu
coverage only — it is not promotion evidence for any serving change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from .analyze_puzzle_screen import _load_roots
from .torch_benchmark_stats import paired_delta_stats

MAX_SKIP_FRACTION = 0.10


def analyze(capped_dir: Path, full_dir: Path) -> dict[str, Any]:
    capped = _load_roots(capped_dir)
    full = _load_roots(full_dir)
    joined = sorted(set(capped) & set(full))
    if not joined:
        raise ValueError("capped and full arms share no roots")
    capped_missing = sorted(set(full) - set(capped))
    subset_mismatch: list[tuple[int, int]] = []
    per_root: list[dict[str, Any]] = []
    for key in joined:
        capped_ids = {str(a) for a in capped[key]["action_ids"]}
        full_ids = [str(a) for a in full[key]["action_ids"]]
        if not capped_ids or not capped_ids <= set(full_ids):
            subset_mismatch.append(key)
            continue
        full_q = [float(q) for q in full[key]["mean_completed_q"]]
        full_best = max(range(len(full_q)), key=full_q.__getitem__)
        dropped = full_ids[full_best] not in capped_ids
        if dropped:
            best_kept = max(
                (i for i in range(len(full_q)) if full_ids[i] in capped_ids),
                key=full_q.__getitem__,
            )
            regret = full_q[full_best] - full_q[best_kept]
        else:
            regret = 0.0
        per_root.append(
            {
                "seed": key[0],
                "ply": key[1],
                "capped_menu_size": len(capped_ids),
                "full_menu_size": len(full_ids),
                "full_best_index": full_best,
                "full_best_action_id": full_ids[full_best],
                "dropped": dropped,
                "regret": regret,
            }
        )
    skipped = len(capped_missing) + len(subset_mismatch)
    if skipped > MAX_SKIP_FRACTION * len(full):
        raise ValueError(
            f"skipped {skipped}/{len(full)} roots "
            f"(capped-missing {len(capped_missing)}, "
            f"subset-mismatch {len(subset_mismatch)}) — exceeds the "
            f"{MAX_SKIP_FRACTION:.0%} tolerance; arms likely replayed "
            "different ledgers or strides"
        )
    if not per_root:
        raise ValueError("no analyzable roots after skipping mismatches")
    regrets = [row["regret"] for row in per_root]
    drops = [row for row in per_root if row["dropped"]]
    return {
        "status": "pass",
        "roots": len(per_root),
        "capped_missing_roots": len(capped_missing),
        "subset_mismatch_roots": len(subset_mismatch),
        "skipped_roots": skipped,
        "capped_only_roots": len(set(capped) - set(full)),
        "capped_menu_median_size": median(row["capped_menu_size"] for row in per_root),
        "full_menu_median_size": median(row["full_menu_size"] for row in per_root),
        "drop_count": len(drops),
        "drop_rate": len(drops) / len(per_root),
        "mean_regret_when_dropped": (
            sum(row["regret"] for row in drops) / len(drops) if drops else None
        ),
        "mean_regret_overall": sum(regrets) / len(regrets),
        "p95_regret_overall": sorted(regrets)[int(0.95 * (len(regrets) - 1))],
        "regret_stats": paired_delta_stats(regrets),
        "capped_search": next(iter(capped.values()))["search"],
        "full_search": next(iter(full.values()))["search"],
        "per_root": per_root,
    }


def _fmt(value: Any, spec: str = "+.4f") -> str:
    return "n/a" if value is None else format(value, spec)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["regret_stats"]
    lines = [
        "# Menu-Coverage Audit (R1.3a)",
        "",
        f"Roots joined: `{report['roots']}` (skipped `{report['skipped_roots']}`: "
        f"capped-missing `{report['capped_missing_roots']}`, "
        f"subset-mismatch `{report['subset_mismatch_roots']}`)",
        f"Median menu size: capped `{report['capped_menu_median_size']}` vs "
        f"full `{report['full_menu_median_size']}`",
        f"Full-best dropped by cap: `{report['drop_count']}/{report['roots']}` "
        f"(`{report['drop_rate']:.1%}`)",
        f"Mean regret when dropped: `{_fmt(report['mean_regret_when_dropped'])}`",
        f"Mean regret overall: `{_fmt(report['mean_regret_overall'])}` "
        f"(95% t-CI `[{_fmt(stats['t_ci_low'])}, {_fmt(stats['t_ci_high'])}]`)",
        f"P95 regret overall: `{_fmt(report['p95_regret_overall'])}`",
        "",
        "Regret is measured on the full-run Q scale. The audit measures menu "
        "coverage only; it is never promotion evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capped-dir", required=True)
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = analyze(Path(args.capped_dir), Path(args.full_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(
        json.dumps(
            {
                "roots": report["roots"],
                "drop_rate": report["drop_rate"],
                "mean_regret_overall": report["mean_regret_overall"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
