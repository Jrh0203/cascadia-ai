"""R0.2 offline kill-test analyzer for the search-stability probe.

Consumes `--search-stability-probe` exporter output (repeated root searches,
unpaired vs paired rollout streams, equal repeat = equal search seed) and
computes the preregistered decision quantity: the pooled variance of the
top1-top2 completed-Q gap among visited actions, per variant. The
preregistered rule (EXPERIMENT_LOG 2026-07-10): proceed to the n256 gate iff
paired pooled gap variance is at least 20% below unpaired. Chosen-action
flip rates across repeats are reported as the secondary stability read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

VARIANTS = ("unpaired", "paired")


def _variant_name(paired_rollouts: bool) -> str:
    return "paired" if paired_rollouts else "unpaired"


def _population_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _flip_rate(chosen: list[int]) -> float:
    modal_count = Counter(chosen).most_common(1)[0][1]
    return 1.0 - modal_count / len(chosen)


def analyze(path: Path, variance_reduction_floor: float = 0.20) -> dict[str, Any]:
    by_root: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {variant: [] for variant in VARIANTS}
    )
    summary: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "stability_search":
            key = (int(record["seed"]), int(record["ply"]))
            by_root[key][_variant_name(bool(record["paired_rollouts"]))].append(record)
        elif record.get("type") == "stability_summary":
            summary = record
    if not by_root:
        raise ValueError(f"no stability_search rows in {path}")
    if summary is None:
        raise ValueError(f"no stability_summary row in {path} (incomplete probe)")

    per_root: list[dict[str, Any]] = []
    skipped_thin_menus = 0
    for (seed, ply), variants in sorted(by_root.items()):
        root_row: dict[str, Any] = {"seed": seed, "ply": ply}
        usable = True
        for variant in VARIANTS:
            repeats = variants[variant]
            gaps: list[float] = []
            for record in repeats:
                top_visited = record["top_visited"]
                if len(top_visited) < 2:
                    usable = False
                    break
                gaps.append(
                    float(top_visited[0]["completed_q"])
                    - float(top_visited[1]["completed_q"])
                )
            if not usable or len(gaps) < 2:
                usable = False
                break
            root_row[f"{variant}_gap_variance"] = _population_variance(gaps)
            root_row[f"{variant}_mean_gap"] = sum(gaps) / len(gaps)
            root_row[f"{variant}_flip_rate"] = _flip_rate(
                [int(record["chosen_index"]) for record in repeats]
            )
        if not usable:
            skipped_thin_menus += 1
            continue
        per_root.append(root_row)
    if not per_root:
        raise ValueError("every sampled root was unusable (thin menus?)")
    # Distinct RNG streams always leave float-level traces in leaf values,
    # so exact equality of every per-root statistic across the two variants
    # means rollout randomness never reached the leaves (e.g. rollout
    # top-k 1 greedy rollouts) — an invalid run, not a null result. Small
    # synthetic fixtures are exempt.
    if len(per_root) >= 5 and all(
        row["unpaired_gap_variance"] == row["paired_gap_variance"]
        and row["unpaired_mean_gap"] == row["paired_mean_gap"]
        and row["unpaired_flip_rate"] == row["paired_flip_rate"]
        for row in per_root
    ):
        raise ValueError(
            "paired and unpaired variants are bit-identical across all roots — "
            "rollout randomness is not reaching leaf values (vacuous config, "
            "e.g. --rollout-top-k 1); this run is invalid, not a null"
        )

    pooled_unpaired = sum(row["unpaired_gap_variance"] for row in per_root) / len(per_root)
    pooled_paired = sum(row["paired_gap_variance"] for row in per_root) / len(per_root)
    variance_reduction = (
        1.0 - pooled_paired / pooled_unpaired if pooled_unpaired > 0.0 else 0.0
    )
    variance_deltas = [
        row["unpaired_gap_variance"] - row["paired_gap_variance"] for row in per_root
    ]
    flip_deltas = [row["unpaired_flip_rate"] - row["paired_flip_rate"] for row in per_root]

    return {
        "status": "pass",
        "roots": len(per_root),
        "skipped_thin_menus": skipped_thin_menus,
        "repeats_per_variant": summary.get("repeats_per_variant"),
        "search": summary.get("search"),
        "pooled_gap_variance": {
            "unpaired": pooled_unpaired,
            "paired": pooled_paired,
            "reduction": variance_reduction,
        },
        "per_root_variance_delta_stats": paired_delta_stats(variance_deltas),
        "flip_rate": {
            "unpaired_mean": sum(row["unpaired_flip_rate"] for row in per_root)
            / len(per_root),
            "paired_mean": sum(row["paired_flip_rate"] for row in per_root)
            / len(per_root),
            "per_root_delta_stats": paired_delta_stats(flip_deltas),
        },
        "variance_reduction_floor": variance_reduction_floor,
        "proceed_to_gate": variance_reduction >= variance_reduction_floor,
        "per_root": per_root,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    pooled = report["pooled_gap_variance"]
    flips = report["flip_rate"]
    lines = [
        "# Search-Stability Probe (R0.2 offline kill test)",
        "",
        f"Roots: `{report['roots']}` (skipped thin menus: "
        f"`{report['skipped_thin_menus']}`) — repeats/variant: "
        f"`{report['repeats_per_variant']}`",
        "",
        "## Primary: pooled top1-top2 completed-Q gap variance (visited)",
        "",
        f"- Unpaired: `{pooled['unpaired']:.6f}`",
        f"- Paired: `{pooled['paired']:.6f}`",
        f"- Reduction: `{pooled['reduction']:.1%}` "
        f"(preregistered floor `{report['variance_reduction_floor']:.0%}`)",
        f"- **Proceed to n256 gate: `{report['proceed_to_gate']}`**",
        "",
        "## Secondary: chosen-action flip rate across repeats",
        "",
        f"- Unpaired mean: `{flips['unpaired_mean']:.4f}`",
        f"- Paired mean: `{flips['paired_mean']:.4f}`",
        f"- Per-root delta (unpaired - paired) mean: "
        f"`{flips['per_root_delta_stats']['mean']:+.4f}`, 95% t-CI "
        f"`[{flips['per_root_delta_stats']['t_ci_low']:+.4f}, "
        f"{flips['per_root_delta_stats']['t_ci_high']:+.4f}]`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--variance-reduction-floor", type=float, default=0.20)
    args = parser.parse_args()
    report = analyze(Path(args.input), args.variance_reduction_floor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(
        json.dumps(
            {
                "roots": report["roots"],
                "pooled_gap_variance": report["pooled_gap_variance"],
                "proceed_to_gate": report["proceed_to_gate"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
