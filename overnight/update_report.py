#!/usr/bin/env python3
"""Update MORNING_REPORT.md with current bench results from logs."""
import re
import os
from glob import glob

OUT_DIR = "overnight"
REPORT = os.path.join(OUT_DIR, "MORNING_REPORT.md")


def parse_summary(path):
    """Extract base, bonus, time from a single bench log."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            text = f.read()
    except Exception:
        return None
    m_base = re.search(r'Base Score \(no habitat bonus\):\s*\n\s*Mean:\s*(\S+)', text)
    m_bonus = re.search(r'With Habitat Bonus:\s*\n\s*Mean:\s*(\S+)', text)
    m_time = re.search(r'(\d+) games in (\S+)s', text)
    if not (m_base and m_bonus):
        return None
    return {
        'base': float(m_base.group(1)),
        'bonus': float(m_bonus.group(1)),
        'time_s': float(m_time.group(2)) if m_time else None,
    }


# Variant lists (keep in sync with bench scripts)
PRIMARY = [
    ("00_baseline_halving", "baseline halving"),
    ("01_prefilter_k4", "prefilter k=4"),
    ("02_prefilter_k6", "prefilter k=6"),
    ("03_prefilter_k8", "prefilter k=8"),
    ("04_prefilter_k12", "prefilter k=12"),
    ("05_exact_endgame_1", "exact endgame=1"),
    ("06_exact_endgame_2", "exact endgame=2"),
    ("07_exact_endgame_3", "exact endgame=3"),
    ("08_pf6_eg2", "prefilter k=6 + exact endgame=2"),
]

ADVANCED = [
    ("10_halving_ci", "halving-CI alone"),
    ("11_halving_ci_pf8", "halving-CI + prefilter k=8"),
    ("12_expanded_pf8", "expanded cands + prefilter k=8"),
    ("13_expanded_pf12", "expanded cands + prefilter k=12"),
    ("14_pf8_eg2_500r", "prefilter k=8 + exact endgame=2 @ 500 rollouts"),
    ("15_pf8_eg2_750r", "prefilter k=8 + exact endgame=2 @ 750 rollouts"),
]


def gen_table(variants, baseline_base=None):
    lines = ["| Variant | Base | Bonus | Δ Base | Time(s) | Notes |",
             "|---------|-----:|------:|-------:|--------:|-------|"]
    for name, note in variants:
        path = os.path.join(OUT_DIR, name + ".log")
        r = parse_summary(path)
        if r:
            delta = ""
            if baseline_base is not None:
                delta = f"{r['base'] - baseline_base:+.1f}"
            t = f"{int(r['time_s'])}" if r['time_s'] else "?"
            lines.append(f"| {name} | {r['base']:.1f} | {r['bonus']:.1f} | {delta} | {t} | {note} |")
        else:
            lines.append(f"| {name} | TBD | TBD | | | {note} |")
    return "\n".join(lines)


def main():
    # Find baseline base score (for delta column)
    baseline = parse_summary(os.path.join(OUT_DIR, "00_baseline_halving.log"))
    baseline_base = baseline['base'] if baseline else None

    primary_table = gen_table(PRIMARY, baseline_base)
    advanced_table = gen_table(ADVANCED, baseline_base)

    if not os.path.exists(REPORT):
        print(f"# {REPORT} not found")
        return

    with open(REPORT) as f:
        text = f.read()

    # Replace primary table
    text = re.sub(
        r"(## Bench Results \(Primary Batch\)\n\n).*?(\n\n)(\[TABLE.*?\]|## )",
        lambda m: f"{m.group(1)}{primary_table}\n\n{m.group(3)}",
        text, count=1, flags=re.DOTALL,
    )
    # Replace advanced table
    text = re.sub(
        r"(## Bench Results \(Advanced Batch — runs after primary\)\n\n).*?(\n\n)(\[TABLE.*?\]|## )",
        lambda m: f"{m.group(1)}{advanced_table}\n\n{m.group(3)}",
        text, count=1, flags=re.DOTALL,
    )

    with open(REPORT, 'w') as f:
        f.write(text)
    print(f"Updated {REPORT}")


if __name__ == '__main__':
    main()
