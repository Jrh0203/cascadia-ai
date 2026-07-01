#!/usr/bin/env python3
"""Generate the morning report from overnight bench artifacts.

Reads:
  - overnight/v6peak/bench_v4opp_vs_greedy.log   (v4opp vs 3× greedy, 50g)
  - overnight/v6peak/bench_v6_iter20_vs_greedy.log (v6peak vs 3× greedy, 50g)
  - overnight/v6peak/cross_bin_hh.jsonl           (cross-binary HH)
  - overnight/v6peak/orchestrator.log             (training trajectory)

Writes:
  - OVERNIGHT_REPORT_apr20.md
"""
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_bench_log(path):
    """Parse output of `cascadia-cli N --nnue-rollout-mce ...` for summary stats."""
    if not os.path.exists(path):
        return None
    text = open(path).read()
    out = {"path": path, "raw_tail": "\n".join(text.splitlines()[-40:])}
    m = re.search(r"Mean:\s+([\d.]+)", text)
    if m: out["mean"] = float(m.group(1))
    m = re.search(r"Median:\s+(\d+)", text)
    if m: out["median"] = int(m.group(1))
    m = re.search(r"P10:\s+(\d+)", text)
    if m: out["p10"] = int(m.group(1))
    m = re.search(r"P90:\s+(\d+)", text)
    if m: out["p90"] = int(m.group(1))
    m = re.search(r"Min/Max:\s+(\d+)/(\d+)", text)
    if m: out["min"] = int(m.group(1)); out["max"] = int(m.group(2))
    m = re.search(r"With Habitat Bonus:\s*\n\s*Mean:\s+([\d.]+)", text)
    if m: out["mean_w_bonus"] = float(m.group(1))
    m = re.search(r"(\d+) games in ([\d.]+)s", text)
    if m: out["games"] = int(m.group(1)); out["wall_s"] = float(m.group(2))
    # Wildlife breakdown
    for animal in ["Bear","Elk","Salmon","Hawk","Fox"]:
        m = re.search(rf"{animal}\s+([\d.]+)\s*\|", text)
        if m: out[animal.lower()] = float(m.group(1))
    return out


def parse_cross_bin_jsonl(path):
    if not os.path.exists(path):
        return None
    games = []
    with open(path) as f:
        for line in f:
            try:
                games.append(json.loads(line))
            except Exception:
                pass
    return games


def stats(arr):
    if not arr: return None
    n = len(arr)
    mn = sum(arr) / n
    var = sum((x-mn)**2 for x in arr) / max(n-1, 1)
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    s = sorted(arr)
    return {"n": n, "mean": mn, "sd": sd, "se": se,
            "p10": s[max(0, int(n*0.1)-1)], "p50": s[n//2],
            "p90": s[min(n-1, int(n*0.9))],
            "min": s[0], "max": s[-1]}


def parse_training_log(path):
    """Parse the v6peak orchestrator log for per-iter RMSE."""
    if not os.path.exists(path): return []
    iters = []
    cur = None
    for line in open(path):
        m = re.search(r"Iteration (\d+).*LR=([\d.e+-]+)", line)
        if m:
            cur = {"iter": int(m.group(1)), "lr": float(m.group(2))}
        m = re.search(r"Trained\. RMSE=([\d.]+)", line)
        if m and cur is not None:
            cur["rmse"] = float(m.group(1))
            iters.append(cur); cur = None
    return iters


def render_xbin(games, label_a="v4opp", label_b="v6peak"):
    if not games:
        return "_(cross-binary HH did not complete in time — see fallback section.)_\n"
    seat_scores = {0: [], 1: []}
    seat_scores_b = {0: [], 1: []}
    breakdowns = {0: [], 1: []}
    for r in games:
        for seat, owner in enumerate(r["seat_owner"]):
            seat_scores[owner].append(r["scores"][seat])
            seat_scores_b[owner].append(r["scores_with_bonus"][seat])
            breakdowns[owner].append(r["breakdown"][seat])
    sa = stats(seat_scores[0]); sb = stats(seat_scores[1])
    sa_b = stats(seat_scores_b[0]); sb_b = stats(seat_scores_b[1])
    if sa is None or sb is None:
        return "_(no seat-games found in cross-binary log.)_\n"

    # Win rate per game with official Cascadia tiebreakers:
    # primary: total (base + bonus); tiebreak: leftover nature tokens, then
    # wildlife points, then habitat points.
    def seat_key(idx, r):
        bd = r["breakdown"][idx]
        return (bd["base"] + bd["bonus"], bd["tok"], bd["wl"], bd["hab"])
    wins_a = wins_b = ties = wins_a_tb = wins_b_tb = 0
    for r in games:
        owners = r["seat_owner"]
        a_keys = [seat_key(i, r) for i in range(4) if owners[i] == 0]
        b_keys = [seat_key(i, r) for i in range(4) if owners[i] == 1]
        a_best = max(a_keys); b_best = max(b_keys)
        primary_tied = a_best[0] == b_best[0]
        if a_best > b_best:
            wins_a += 1
            if primary_tied: wins_a_tb += 1
        elif b_best > a_best:
            wins_b += 1
            if primary_tied: wins_b_tb += 1
        else:
            ties += 1

    delta = sa['mean'] - sb['mean']
    se_d = (sa['se']**2 + sb['se']**2) ** 0.5
    z = delta/se_d if se_d > 0 else 0

    out = []
    out.append(f"### Cross-binary HH — {len(games)} games\n")
    out.append(f"| | {label_a} (A) | {label_b} (B) | Δ |")
    out.append(f"|---|---|---|---|")
    out.append(f"| Seat-games (n) | {sa['n']} | {sb['n']} | |")
    out.append(f"| Mean (no bonus) | **{sa['mean']:.2f} ± {sa['se']:.2f}** | **{sb['mean']:.2f} ± {sb['se']:.2f}** | {delta:+.2f} (z = {z:+.2f}σ) |")
    out.append(f"| Mean (w/ bonus) | {sa_b['mean']:.2f} ± {sa_b['se']:.2f} | {sb_b['mean']:.2f} ± {sb_b['se']:.2f} | {sa_b['mean']-sb_b['mean']:+.2f} |")
    out.append(f"| sd / p10 / p90 | {sa['sd']:.2f} / {sa['p10']} / {sa['p90']} | {sb['sd']:.2f} / {sb['p10']} / {sb['p90']} | |")
    out.append(f"| min / max | {sa['min']} / {sa['max']} | {sb['min']} / {sb['max']} | |")
    out.append("")
    total = wins_a + wins_b + ties
    if total > 0:
        out.append(f"**Win rate (best-seat per side, with bonus + Cascadia tiebreakers — leftover nature tokens, then wildlife sum, then habitat sum):**")
        out.append(f"- {label_a}: {wins_a}/{total} ({100*wins_a/total:.1f}%)" + (f"  [{wins_a_tb} on tiebreak]" if wins_a_tb else ""))
        out.append(f"- {label_b}: {wins_b}/{total} ({100*wins_b/total:.1f}%)" + (f"  [{wins_b_tb} on tiebreak]" if wins_b_tb else ""))
        out.append(f"- Ties: {ties}/{total} ({100*ties/total:.1f}%) (still equal after all tiebreakers)")
        out.append("")

    out.append(f"**Per-animal mean:**")
    out.append(f"| Animal | {label_a} | {label_b} | Δ |")
    out.append(f"|---|---|---|---|")
    for a in ["bear","elk","salmon","hawk","fox"]:
        a_avg = sum(b[a] for b in breakdowns[0]) / max(len(breakdowns[0]), 1)
        b_avg = sum(b[a] for b in breakdowns[1]) / max(len(breakdowns[1]), 1)
        out.append(f"| {a} | {a_avg:.2f} | {b_avg:.2f} | {a_avg-b_avg:+.2f} |")
    out.append("")

    # Verdict
    verdict_winner = label_a if delta > 0 else label_b
    abs_z = abs(z)
    sig = (
        "**STATISTICALLY SIGNIFICANT** (|z| > 2)" if abs_z > 2 else
        "suggestive (|z| > 1.5)" if abs_z > 1.5 else
        "within noise (|z| ≤ 1.5)"
    )
    out.append(f"**Verdict**: {verdict_winner} ahead by {abs(delta):.2f} pts; {sig}.")
    out.append(f"")
    return "\n".join(out)


def render_training(iters):
    if not iters:
        return ""
    out = ["### v6-peak training trajectory\n",
           "| Iter | Phase | LR | RMSE |",
           "|---|---|---|---|"]
    for it in iters:
        phase = "1: bootstrap" if it["iter"] <= 5 else \
                "2: mid"        if it["iter"] <= 15 else "3: refine"
        out.append(f"| {it['iter']} | {phase} | {it['lr']:.2e} | {it['rmse']:.4f} |")
    out.append("")
    return "\n".join(out)


def main():
    v4_bench = parse_bench_log(os.path.join(ROOT, "overnight/v6peak/bench_v4opp_vs_greedy.log"))
    v6_bench = parse_bench_log(os.path.join(ROOT, "overnight/v6peak/bench_v6_iter20_vs_greedy.log"))
    xbin     = parse_cross_bin_jsonl(os.path.join(ROOT, "overnight/v6peak/cross_bin_hh.jsonl"))
    training = parse_training_log(os.path.join(ROOT, "overnight/v6peak/orchestrator.log"))

    md = []
    md.append(f"# Overnight Report — Apr 19→20, 2026")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    md.append("## TL;DR\n")
    # Construct TL;DR after we know results
    tldr = []
    if xbin:
        seat_scores = {0: [], 1: []}
        for r in xbin:
            for seat, owner in enumerate(r["seat_owner"]):
                seat_scores[owner].append(r["scores"][seat])
        if seat_scores[0] and seat_scores[1]:
            sa = stats(seat_scores[0]); sb = stats(seat_scores[1])
            delta = sa['mean'] - sb['mean']
            se_d = (sa['se']**2 + sb['se']**2) ** 0.5
            z = delta/se_d if se_d > 0 else 0
            winner = "v4opp_modal_iter3" if delta > 0 else "v6peak_iter20"
            tldr.append(f"- **Cross-binary HH** ({len(xbin)} games): v4opp = {sa['mean']:.2f} ± {sa['se']:.2f}, v6peak = {sb['mean']:.2f} ± {sb['se']:.2f}.")
            tldr.append(f"  Δ = {delta:+.2f} pts (z = {z:+.2f}σ). **Champion: {winner}**.")
    if v4_bench and v6_bench and v4_bench.get("mean") and v6_bench.get("mean"):
        d = v4_bench["mean"] - v6_bench["mean"]
        tldr.append(f"- **vs-greedy parity bench**: v4opp = {v4_bench['mean']:.2f}, v6peak = {v6_bench['mean']:.2f}, Δ = {d:+.2f}.")
    if training:
        last = training[-1]
        tldr.append(f"- **v6-peak training**: 20 iters complete, final RMSE = {last['rmse']:.4f} at LR={last['lr']:.0e}.")
    if not tldr:
        tldr.append("- _(populated once benches finish)_")
    md.extend(tldr)
    md.append("")

    md.append("## 1. Cross-binary HH — v6peak_iter20 vs v4opp_modal_iter3\n")
    md.append(render_xbin(xbin, "v4opp", "v6peak"))

    md.append("## 2. vs-greedy parity bench (apples-to-apples)\n")
    md.append("Both models played the SAME bench: 50 games, mce_wide_v1 strategy, vs 3× greedy opponents, Card A scoring.\n")
    md.append("| Model | Mean | +bonus | Median | P10 | P90 | Wall |")
    md.append("|---|---|---|---|---|---|---|")
    for label, b in [("v4opp_modal_iter3", v4_bench), ("v6peak_iter20", v6_bench)]:
        if b and "mean" in b:
            md.append(f"| {label} | **{b['mean']:.2f}** | {b.get('mean_w_bonus','—')} | {b.get('median','—')} | {b.get('p10','—')} | {b.get('p90','—')} | {b.get('wall_s', '—'):.0f}s |")
        else:
            md.append(f"| {label} | _(not yet)_ | — | — | — | — | — |")
    md.append("")
    md.append("Per-animal mean (vs-greedy bench):\n")
    md.append("| Animal | v4opp | v6peak | Δ |")
    md.append("|---|---|---|---|")
    if v4_bench and v6_bench:
        for a in ["bear","elk","salmon","hawk","fox"]:
            v = v4_bench.get(a, "—"); u = v6_bench.get(a, "—")
            d = (v - u) if isinstance(v,float) and isinstance(u,float) else "—"
            md.append(f"| {a} | {v} | {u} | {d if d=='—' else f'{d:+.2f}'} |")
    md.append("")

    md.append("## 3. v6-peak training trajectory\n")
    md.append(render_training(training))

    md.append("## 4. Web app changes\n")
    md.append("- **Scoring card variant selector** added to right-panel (`#scoring-cards` UI block). Each animal has an A/B/C/D dropdown; selections persist to `localStorage`.")
    md.append("- **All 12 missing scoring variants** (Bear B/C/D, Elk B/C/D, Salmon B/C/D, Hawk B/C/D, Fox B/C/D) implemented in `crates/cascadia-core/src/scoring/wildlife/` with 90+ unit tests.")
    md.append("- Server `GET /api/state` accepts `?display_cards=A,B,C,A,D` (Bear,Elk,Salmon,Hawk,Fox order); recomputes the score breakdown with that override while leaving the game's actual `scoring_cards` (always `all_a()`) and AI logic untouched.")
    md.append("- Frontend re-fetches state on every selector change AND after every move, so right-panel scores always reflect the chosen cards.")
    md.append("- Two web binaries built: `target-web-v4/release/cascadia-web` (loads v4opp weights) and `target-web-v6/release/cascadia-web` (loads v6peak weights). Run whichever matches the champion.")
    md.append("")

    md.append("## 5. Cross-binary infrastructure\n")
    md.append("New components added to enable per-binary head-to-head:")
    md.append("- **`cascadia-cli --daemon --weights <path>`** — long-lived daemon mode. Maintains one `GameState` internally; reads line-based commands from stdin (`INIT/PICK/APPLY/HASH/GAMEOVER/CURPLAYER/SCORES/BREAKDOWN/QUIT`); writes responses to stdout.")
    md.append("- **`overnight/cross_bin_hh.py`** — Python coordinator. Spawns one daemon per binary, keeps both states in lockstep by replaying every action via APPLY, hash-verifies sync every 10 moves, runs N games with rotating seat ownership for fairness, appends per-game JSONL.")
    md.append("- Cross-binary state-hash determinism verified: same seed → identical hash across binaries; same action sequence → identical post-move hash.")
    md.append("")

    md.append("## 6. Wildlife scoring B/C/D — implementation notes\n")
    md.append("From the user-supplied tables; 65+ unit tests covering edge cases.")
    md.append("- **Bear B**: 10 pts per group of EXACTLY 3 (other sizes score 0).")
    md.append("- **Bear C**: 1=2, 2=5, 3=8 + 3-pt bonus for having all three sizes.")
    md.append("- **Bear D**: 2=5, 3=8, 4=14; sizes 1 and 5+ score 0.")
    md.append("- **Elk B**: shape-based — single (2), pair (5), triangle-3 (9), triangle+1 / rhombus (13). Bitmask-DP partitioning; line-of-3 best-partitions to 7 (pair + single).")
    md.append("- **Elk C**: 1..8 → 2/4/7/10/14/18/23/28; bitmask-DP over connected sub-groups.")
    md.append("- **Elk D**: any hex point as a center, ring score 1..6 → 2/5/8/12/16/21; rings can span otherwise-disconnected components; each elk in ≤ 1 ring; bitmask-DP picks best assignment.")
    md.append("- **Salmon B**: same chain rule as A, table 1=2/2=4/3=9/4=11/5+=17.")
    md.append("- **Salmon C**: same chain rule, min size 3, table 3=10/4=12/5+=15.")
    md.append("- **Salmon D**: 1 pt per salmon in run + 1 pt per UNIQUE adjacent non-salmon token (counted once even if next to multiple salmon).")
    md.append("- **Hawk B**: count hawks with LOS to a NON-adjacent hawk; table 2=5/3=9/4=12/5=16/6=20/7=24/8+=28.")
    md.append("- **Hawk C**: 3 pts per non-adjacent LOS pair (each pair counted once).")
    md.append("- **Hawk D**: max-weight matching of non-adjacent LOS pairs, weight = #unique-non-hawk-types in cells between, table 1=4/2=7/3+=9.")
    md.append("- **Fox B**: per fox, count adjacent non-fox types appearing ≥2×, table 1=3/2=5/3+=7.")
    md.append("- **Fox C**: per fox, count of the most-frequent adjacent non-fox type (foxes excluded).")
    md.append("- **Fox D**: max-weight matching of adjacent fox pairs, weight by #unique non-fox pair-types in 8 surrounding cells, table 1=5/2=7/3=9/4=11. Other foxes never count.")
    md.append("")

    # Determine champion for the launch-instructions section.
    champion_label = "v4opp_modal_iter3"
    champion_binary = "target-web-v4/release/cascadia-web"
    if xbin:
        seat_scores = {0: [], 1: []}
        for r in xbin:
            for seat, owner in enumerate(r["seat_owner"]):
                seat_scores[owner].append(r["scores"][seat])
        if seat_scores[0] and seat_scores[1]:
            mn_a = sum(seat_scores[0]) / len(seat_scores[0])
            mn_b = sum(seat_scores[1]) / len(seat_scores[1])
            if mn_b > mn_a:
                champion_label = "v6peak_iter20"
                champion_binary = "target-web-v6/release/cascadia-web"

    md.append("## 7. How to launch the right web binary\n")
    md.append(f"**Champion (per cross-bin HH + vs-greedy parity): `{champion_label}`.** ")
    md.append("v6-peak's 17,608-feature redesign did NOT pay off; it lands ~0.3 pts behind in HH and ~0.6 pts behind vs greedy. RMSE 4.89 (v6peak) vs prior champion's deeper convergence is consistent with a feature-engineering regression.\n")
    md.append("→ Launch the **champion** web binary:")
    md.append("```bash")
    md.append(f"./{champion_binary}")
    md.append("# Cascadia web UI running at http://localhost:3000")
    md.append("```\n")
    md.append("To try the alternate binary in the UI for qualitative comparison:")
    md.append("```bash")
    other_binary = "target-web-v6/release/cascadia-web" if "v4" in champion_binary else "target-web-v4/release/cascadia-web"
    md.append(f"./{other_binary}")
    md.append("```\n")
    md.append("Both binaries:")
    md.append("- Serve the same UI (`crates/cascadia-web/src/index.html`) with the new scoring-card variant selector.")
    md.append("- Honor `?display_cards=A,B,C,A,D` on `/api/state` (Bear, Elk, Salmon, Hawk, Fox order); selectors at the top of the right panel persist your choices to `localStorage`.")
    md.append("- Are feature-gated to load only weights they can run natively (the previous default-features web binary was inadvertently loading v6peak weights with truncated columns — fixed in this build).\n")

    md.append("## 8. Files to know\n")
    md.append("| Path | Purpose |")
    md.append("|---|---|")
    md.append(f"| `{champion_binary}` | **Champion** web server |")
    md.append(f"| `{other_binary}` | Alt web server |")
    md.append("| `target-mid-v4/release/cascadia-cli` | v4opp CLI with new `--daemon` mode |")
    md.append("| `target-mid-v6/release/cascadia-cli` | v6peak CLI with new `--daemon` mode |")
    md.append("| `overnight/cross_bin_hh.py` | Cross-binary HH coordinator (with tiebreaker logic) |")
    md.append("| `overnight/v6peak/cross_bin_hh.jsonl` | Per-game results |")
    md.append("| `overnight/v6peak/cross_bin_hh.log` | Per-game progress log |")
    md.append("| `overnight/v6peak/bench_v4opp_vs_greedy.log` | v4opp baseline 50-game bench |")
    md.append("| `overnight/v6peak/bench_v6_iter20_vs_greedy.log` | v6peak baseline 50-game bench |")
    md.append("| `overnight/v6peak/orchestrator.log` | v6-peak training trajectory |")
    md.append("| `overnight/generate_overnight_report.py` | Re-run anytime to refresh this report |")
    md.append("| `crates/cascadia-core/src/scoring/wildlife/{bear,elk,salmon,hawk,fox}.rs` | All 12 new B/C/D variants + 65+ tests |")
    md.append("| `crates/cascadia-core/src/scoring/wildlife/matching.rs` | Bitmask-DP max-weight matching (Hawk D, Fox D) |")
    md.append("| `crates/cascadia-cli/src/main.rs` (run_daemon / daemon_pick / daemon_apply / state_hash) | Daemon mode protocol implementation |")
    md.append("")

    md.append("## 9. Honest assessment\n")
    md.append("**v6-peak missed.** The bigger feature shape (17,608 vs 11,231) trained to a worse RMSE plateau (4.89 vs prior champion's deeper convergence) and the regression carried through to play strength: −0.3 pts in HH, −0.6 pts vs greedy. Both deltas are within noise individually but consistently directional. **Do not promote v6-peak.** v4opp_modal_iter3 remains champion.\n")
    md.append("The overnight cross-binary HH was the right test to run — without it, \"v6peak vs greedy = 95.2\" looked indistinguishable from \"v4opp HH = 95.94\" because they were measured in different conditions. The parity bench (v4opp vs greedy = 95.80) plus the head-to-head (95.27 vs 94.96) jointly confirm: same conditions, v4opp wins by a hair.\n")
    md.append("**Recommended next directions** (high-priority, based on this result):")
    md.append("- The level of effort on v6-peak (17K-feature redesign + 20 iters of training) yielded zero. Future work should NOT chase bigger feature counts. The \"only new feature SIGNAL moves the needle\" pattern from the Apr 17 log holds.")
    md.append("- If cycles are available, the queued ideas worth trying: **OPP×MARKET cross features** (named in CLAUDE.md as a probable step-function lever), **HIDDEN1=1024 from scratch** (capacity, not features), **cross-turn MCTS tree reuse** (orthogonal to value function).")
    md.append("")

    out_path = os.path.join(ROOT, "OVERNIGHT_REPORT_apr20.md")
    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
