#!/usr/bin/env python3
"""Analyze bench output logs to compare per-animal scoring across variants."""
import re
import sys
import os
from glob import glob

OUT_DIR = "overnight"
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "*.log"

ANIMALS = ['Bear', 'Elk', 'Salmon', 'Hawk', 'Fox']
HABITATS = ['Forest', 'Prairie', 'Wetland', 'Mountain', 'River']


def parse_log(path):
    """Extract structured data from a bench log."""
    with open(path) as f:
        text = f.read()

    out = {'name': os.path.basename(path).replace('.log', '')}

    # Base score
    m = re.search(r'Base Score \(no habitat bonus\):\s*\n\s*Mean:\s*(\S+)', text)
    if m: out['base'] = float(m.group(1))
    m = re.search(r'With Habitat Bonus:\s*\n\s*Mean:\s*(\S+)', text)
    if m: out['bonus'] = float(m.group(1))

    # P10/P90/Median
    m = re.search(r'Base Score.*?\n\s*Mean:\s*\S+\s*\n\s*Median:\s*(\S+)\s*\n\s*P10:\s*(\S+)\s*\n\s*P90:\s*(\S+)', text, re.DOTALL)
    if m:
        out['median'] = float(m.group(1))
        out['p10'] = float(m.group(2))
        out['p90'] = float(m.group(3))

    # Habitat sub-scores
    out['habitat'] = {}
    for h in HABITATS:
        m = re.search(rf'\s+{h}\s+(\S+)\s+\|', text)
        if m: out['habitat'][h] = float(m.group(1))

    # Wildlife sub-scores
    out['wildlife'] = {}
    for a in ANIMALS:
        m = re.search(rf'\s+{a}\s+(\S+)\s+\|', text)
        if m: out['wildlife'][a] = float(m.group(1))

    # Tokens
    m = re.search(r'Tokens:\s*(\S+)', text)
    if m: out['tokens'] = float(m.group(1))

    # Habitat total + bonus
    m = re.search(r'Habitat:\s*(\S+).*?\(\+(\S+)\s*bonus\)', text, re.DOTALL)
    if m:
        out['habitat_total'] = float(m.group(1))
        out['habitat_bonus'] = float(m.group(2))

    return out


def main():
    files = sorted(glob(os.path.join(OUT_DIR, PATTERN)))
    files = [f for f in files if os.path.basename(f) != "MORNING_REPORT.md"]
    results = []
    for f in files:
        try:
            r = parse_log(f)
            if 'base' in r:
                results.append(r)
        except Exception as e:
            print(f"# Failed to parse {f}: {e}")

    if not results:
        print("# No parseable logs found")
        return

    print(f"# {len(results)} variants parsed\n")

    # Top-level table
    print(f"{'Variant':<28} {'Base':>6} {'Bonus':>6} {'P10':>5} {'P90':>5}  Top animals (Bear/Elk/Salmon/Hawk/Fox)")
    print("-" * 100)
    baseline_base = None
    for r in results:
        # Identify baseline if it's the first one or named so
        if baseline_base is None and 'baseline' in r['name'].lower():
            baseline_base = r['base']

    for r in results:
        delta = ""
        if baseline_base is not None and 'base' in r:
            d = r['base'] - baseline_base
            delta = f" ({d:+.1f})"
        wl = r.get('wildlife', {})
        wl_str = " / ".join(f"{wl.get(a, 0):.1f}" for a in ANIMALS)
        print(f"{r['name']:<28} {r['base']:>6.1f} {r.get('bonus', 0):>6.1f} "
              f"{r.get('p10', 0):>5.1f} {r.get('p90', 0):>5.1f}  {wl_str}{delta}")

    # Habitat detail
    print()
    print(f"\n{'Variant':<28} {'Habitat':>7} {'+Bonus':>7} {'Forest':>7} {'Prairie':>8} {'Wetland':>8} {'Mtn':>5} {'River':>6}")
    print("-" * 100)
    for r in results:
        h = r.get('habitat', {})
        ht = r.get('habitat_total', 0)
        hb = r.get('habitat_bonus', 0)
        print(f"{r['name']:<28} {ht:>7.1f} {hb:>7.1f} {h.get('Forest', 0):>7.1f} "
              f"{h.get('Prairie', 0):>8.1f} {h.get('Wetland', 0):>8.1f} "
              f"{h.get('Mountain', 0):>5.1f} {h.get('River', 0):>6.1f}")


if __name__ == '__main__':
    main()
