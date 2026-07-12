"""Interim verdict for group-sequential paired gates.

Wraps `compare_search_shape.build_comparison` (all arm-identity
validation is inherited) and adds the group-sequential layer: the
current look is inferred from the number of matched pairs, boundaries
come from `sequential_boundaries` (Lan-DeMets spending), and the
decision is expressed as a repeated confidence interval (RCI):

    RCI_k = mean +/- t_k * SE,   t_k from the look's boundary z_k

Stopping is allowed only when the verdict category is already decided
and more data cannot change the decision under the preregistered rule:

- rule `superiority`: stop when the RCI excludes zero (either side).
- rule `noninferiority`: stop when the RCI lies entirely above or
  entirely below the margin.

An RCI that straddles the threshold always continues to the next look;
at the final look it is reported as inconclusive. The naive fixed-N 95%
CI is included for reference only — it is NOT promotion evidence in a
sequential design; the RCI is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compare_search_shape import DEFAULT_VARIED_KEYS, build_comparison
from .sequential_boundaries import (
    NO_EXIT_BOUNDARY,
    boundary_nominal_alpha,
    sequential_boundaries,
)
from .torch_benchmark_stats import t_quantile

RULES = ("superiority", "noninferiority")


def parse_looks(raw: str) -> list[int]:
    try:
        looks = [int(part) for part in raw.replace(",", " ").split()]
    except ValueError as error:
        raise ValueError(f"unparseable looks specification: {raw!r}") from error
    if not looks:
        raise ValueError("at least one look is required")
    previous = 0
    for look in looks:
        if look <= previous:
            raise ValueError(f"looks must be strictly increasing positives: {looks}")
        previous = look
    return looks


def sequential_decision(
    rci_low: float,
    rci_high: float,
    *,
    is_final_look: bool,
    rule: str,
    margin: float,
) -> str:
    if rule == "superiority":
        threshold = 0.0
        positive, negative = "positive", "negative"
    elif rule == "noninferiority":
        threshold = margin
        positive, negative = "noninferior", "inferior"
    else:
        raise ValueError(f"unknown rule: {rule!r}")
    prefix = "final" if is_final_look else "stop"
    if rci_low > threshold:
        return f"{prefix}_{positive}"
    if rci_high < threshold:
        return f"{prefix}_{negative}"
    return "final_inconclusive" if is_final_look else "continue"


def build_sequential_verdict(
    baseline_path: Path,
    candidate_path: Path,
    looks: list[int],
    source_revision: str | None = None,
    varied_keys: tuple[str, ...] = DEFAULT_VARIED_KEYS,
    alpha: float = 0.05,
    spending: str = "obrien_fleming",
    rule: str = "superiority",
    margin: float = -0.25,
) -> dict[str, Any]:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    comparison = build_comparison(
        baseline_path, candidate_path, source_revision, varied_keys=varied_keys
    )
    pairs = len(comparison["seeds"])
    if pairs not in looks:
        raise ValueError(
            f"matched pair count {pairs} does not equal any planned look {looks}; "
            "sequential verdicts are only valid at preplanned looks"
        )
    look_index = looks.index(pairs)  # 0-based
    planned_final = looks[-1]
    fractions = [look / planned_final for look in looks]
    boundaries = sequential_boundaries(fractions, alpha=alpha, spending=spending)
    boundary_z = boundaries[look_index]

    stats = comparison["paired_delta_stats"]
    mean = stats["mean"]
    se = stats["se"]
    if mean is None or se is None or se == 0.0:
        raise ValueError("sequential verdict requires nonzero-variance paired deltas")
    df = pairs - 1
    if boundary_z >= NO_EXIT_BOUNDARY:
        t_critical = float("inf")
        rci_low, rci_high = float("-inf"), float("inf")
    else:
        nominal = boundary_nominal_alpha(boundary_z)
        t_critical = t_quantile(1.0 - nominal / 2.0, df)
        rci_low = mean - t_critical * se
        rci_high = mean + t_critical * se
    decision = sequential_decision(
        rci_low,
        rci_high,
        is_final_look=(look_index == len(looks) - 1),
        rule=rule,
        margin=margin,
    )

    promotion_scale = planned_final >= 100
    comparison["scientific_eligibility"] = (
        "promotion_scale_sequential_gate" if promotion_scale else "engineering_smoke_only"
    )
    comparison["proceed_to_high_budget"] = bool(
        promotion_scale and decision in ("stop_positive", "final_positive")
    )
    comparison["sequential"] = {
        "looks": looks,
        "fractions": fractions,
        "alpha": alpha,
        "spending": spending,
        "rule": rule,
        "margin": margin if rule == "noninferiority" else None,
        "boundaries_z": boundaries,
        "current_look": look_index + 1,
        "total_looks": len(looks),
        "pairs": pairs,
        "planned_final_pairs": planned_final,
        "boundary_z": boundary_z,
        "t_critical": t_critical,
        "rci_low": rci_low,
        "rci_high": rci_high,
        "naive_ci_low_non_inferential": stats["t_ci_low"],
        "naive_ci_high_non_inferential": stats["t_ci_high"],
        "decision": decision,
    }
    return comparison


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["paired_delta_stats"]
    seq = report["sequential"]
    timing = report["timing"]
    varied_keys = report.get("varied_keys", list(DEFAULT_VARIED_KEYS))
    varied_lines = [
        f"- {key}: `{report['search'].get(f'baseline_{key}')}` -> "
        f"`{report['search'].get(f'candidate_{key}')}`"
        for key in varied_keys
    ]
    boundary_rows = [
        f"| {i + 1} | {look} | {fraction:.2f} | {boundary:.4f} |"
        for i, (look, fraction, boundary) in enumerate(
            zip(seq["looks"], seq["fractions"], seq["boundaries_z"], strict=True)
        )
    ]
    margin_line = (
        [f"- Noninferiority margin: `{seq['margin']:+.4f}`"]
        if seq["rule"] == "noninferiority"
        else []
    )
    lines = [
        "# Sequential Gate Verdict",
        "",
        f"Look: `{seq['current_look']}/{seq['total_looks']}` "
        f"(`{seq['pairs']}` of `{seq['planned_final_pairs']}` planned pairs)",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Scientific eligibility: `{report['scientific_eligibility']}`",
        f"Design: two-sided alpha `{seq['alpha']}`, `{seq['spending']}` spending, "
        f"rule `{seq['rule']}`",
        "",
        "## Varied settings",
        "",
        *varied_lines,
        "",
        "## Decision",
        "",
        f"- **`{seq['decision'].upper()}`**",
        f"- Paired delta: `{stats['mean']:+.4f}`",
        f"- Repeated CI (boundary z `{seq['boundary_z']:.4f}`, t "
        f"`{seq['t_critical']:.4f}`): `[{seq['rci_low']:+.4f}, {seq['rci_high']:+.4f}]`",
        *margin_line,
        f"- Naive 95% CI (reference only, NOT evidence): "
        f"`[{seq['naive_ci_low_non_inferential']:+.4f}, "
        f"{seq['naive_ci_high_non_inferential']:+.4f}]`",
        f"- Baseline / candidate mean seat score: "
        f"`{report['baseline_mean_seat_score']:.4f}` / "
        f"`{report['candidate_mean_seat_score']:.4f}`",
        "",
        "## Boundary schedule",
        "",
        "| Look | Pairs | Fraction | z boundary |",
        "|---:|---:|---:|---:|",
        *boundary_rows,
        "",
        "## Cost (accumulated so far)",
        "",
        f"- Mean decision seconds: `{timing['baseline_mean_decision_seconds']:.4f}` -> "
        f"`{timing['candidate_mean_decision_seconds']:.4f}`",
        f"- Whole-arm wall seconds: `{timing['baseline_wall_seconds']:.1f}` -> "
        f"`{timing['candidate_wall_seconds']:.1f}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument(
        "--varied-key",
        action="append",
        default=None,
        help="Search key allowed to differ between arms; repeatable.",
    )
    parser.add_argument(
        "--looks",
        required=True,
        help="Planned cumulative pair counts, e.g. '40,60,80,100'.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--spending", default="obrien_fleming")
    parser.add_argument("--rule", choices=RULES, default="superiority")
    parser.add_argument(
        "--margin",
        type=float,
        default=-0.25,
        help="Noninferiority margin (used only with --rule noninferiority).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    varied_keys = tuple(args.varied_key) if args.varied_key else DEFAULT_VARIED_KEYS
    report = build_sequential_verdict(
        Path(args.baseline),
        Path(args.candidate),
        parse_looks(args.looks),
        args.source_revision or None,
        varied_keys=varied_keys,
        alpha=args.alpha,
        spending=args.spending,
        rule=args.rule,
        margin=args.margin,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(report["sequential"]["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
