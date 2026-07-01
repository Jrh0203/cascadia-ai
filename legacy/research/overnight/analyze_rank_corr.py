#!/usr/bin/env python3
"""Analyze RANKCORR raw data and produce a PDF report.

Usage:
    python3 overnight/analyze_rank_corr.py overnight/rank_corr_1game.txt
    python3 overnight/analyze_rank_corr.py overnight/rank_corr_raw_local.txt

Reads RANKCORR + RANKCORR_SCORE lines from the input file.
Produces a multi-page PDF at overnight/rank_corr_report.pdf.
"""

import re
import sys
import numpy as np
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def parse_file(path):
    """Parse JSONL or legacy RANKCORR text format."""
    import json
    lines = []
    scores = []
    game_idx = -1
    prev_turn = 999
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            # Try JSONL first
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                    if obj.get("type") == "score":
                        obj.setdefault("game", game_idx)
                        scores.append(obj)
                    elif "turn" in obj and "nnue_rank" in obj:
                        turn = obj["turn"]
                        if turn <= prev_turn and turn == 1:
                            game_idx += 1
                        prev_turn = turn
                        obj.setdefault("game", game_idx)
                        obj.setdefault("mce_std", obj.get("mce_std", 0))
                        # Normalize field names (JSONL uses mce_mean, old used mce_score)
                        if "mce_mean" in obj and "mce_score" not in obj:
                            obj["mce_score"] = obj["mce_mean"]
                        lines.append(obj)
                    continue
                except json.JSONDecodeError:
                    pass
            # Legacy text format fallback
            rc_re = re.compile(
                r"RANKCORR turn=(\d+) n_cands=(\d+) ci=(\d+) market=(\d+) "
                r"nnue_rank=(\d+) mce_rank=(\d+) nnue_score=([\d.\-]+) mce_score=([\d.\-]+)"
            )
            m = rc_re.match(raw)
            if m:
                turn = int(m.group(1))
                if turn <= prev_turn and turn == 1:
                    game_idx += 1
                prev_turn = turn
                lines.append({
                    "game": game_idx, "turn": turn,
                    "n_cands": int(m.group(2)),
                    "market": int(m.group(4)),
                    "nnue_rank": int(m.group(5)), "mce_rank": int(m.group(6)),
                    "nnue_score": float(m.group(7)), "mce_score": float(m.group(8)),
                    "mce_std": 0,
                })
    return lines, scores


def compute_kendall_tau(a, b):
    n = len(a)
    if n < 2: return 0.0
    c = d = ta = tb = 0
    for i in range(n):
        for j in range(i+1, n):
            da, db = a[i]-a[j], b[i]-b[j]
            if da == 0 and db == 0: ta += 1; tb += 1
            elif da == 0: ta += 1
            elif db == 0: tb += 1
            elif (da > 0) == (db > 0): c += 1
            else: d += 1
    p = n*(n-1)/2
    da, db = p-ta, p-tb
    return (c-d)/max((da*db)**0.5, 1e-9)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 analyze_rank_corr.py <raw_data.txt> [output.pdf]")
    path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else "overnight/rank_corr_report.pdf"

    lines, scores = parse_file(path)
    print(f"Parsed {len(lines)} candidate lines, {len(scores)} score lines from {path}")

    # Group by position
    positions = defaultdict(list)
    for l in lines:
        positions[(l["game"], l["turn"])].append(l)

    n_pos = len(positions)
    n_games = max(l["game"] for l in lines) + 1 if lines else 0
    TOP_X = [1, 2, 4, 8, 12, 16, 24, 32, 64, 100]

    # Per-position metrics
    per_turn = defaultdict(lambda: {
        "tau": [], "rho": [], "miss": {x: [] for x in TOP_X},
        "gap": [], "std_all": [], "std_mce1": [], "std_nnue1": [],
        "top1_agree": [],
    })
    all_tau, all_rho = [], []
    all_miss = {x: [] for x in TOP_X}
    all_gap = []
    disagreements = []

    for (gi, turn), cands in positions.items():
        if len(cands) < 2: continue
        nr = [c["nnue_rank"] for c in cands]
        mr = [c["mce_rank"] for c in cands]
        tau = compute_kendall_tau(nr, mr)
        rho_val = 1 - 6*sum((a-b)**2 for a,b in zip(nr,mr))/(len(nr)*(len(nr)**2-1)) if len(nr)>1 else 0
        all_tau.append(tau); all_rho.append(rho_val)
        per_turn[turn]["tau"].append(tau)
        per_turn[turn]["rho"].append(rho_val)

        stds = [c["mce_std"] for c in cands]
        per_turn[turn]["std_all"].extend(stds)

        mb = [c for c in cands if c["mce_rank"] == 0]
        nb = [c for c in cands if c["nnue_rank"] == 0]
        if mb: per_turn[turn]["std_mce1"].append(mb[0]["mce_std"])
        if nb: per_turn[turn]["std_nnue1"].append(nb[0]["mce_std"])

        agree = 1 if (mb and nb and mb[0]["nnue_rank"] == 0) else 0
        per_turn[turn]["top1_agree"].append(agree)

        if mb:
            r = mb[0]["nnue_rank"]
            for x in TOP_X:
                m = 1 if r >= x else 0
                all_miss[x].append(m)
                per_turn[turn]["miss"][x].append(m)
            if r >= 8 and nb:
                gap = mb[0]["mce_score"] - nb[0]["mce_score"]
                all_gap.append(gap)
                per_turn[turn]["gap"].append(gap)

        if nb and mb and (nb[0]["mce_rank"] != 0 or mb[0]["nnue_rank"] != 0):
            disagreements.append({
                "game": gi, "turn": turn,
                "mce_gap": mb[0]["mce_score"] - nb[0]["mce_score"],
                "nnue_gap": nb[0]["nnue_score"] - mb[0]["nnue_score"],
                "nnue_market": nb[0]["market"], "mce_market": mb[0]["market"],
                "nnue_mce_rank": nb[0]["mce_rank"], "mce_nnue_rank": mb[0]["nnue_rank"],
                "n_cands": cands[0]["n_cands"],
            })

    mean = lambda xs: sum(xs)/len(xs) if xs else 0
    std = lambda xs: (sum((x-mean(xs))**2 for x in xs)/max(len(xs)-1,1))**0.5 if len(xs)>1 else 0

    # ====== PDF ======
    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Summary ---
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.axis("off")
        ai_scores = [s for s in scores if s["player"] == 0]
        text = f"NNUE vs MCE Rank Correlation Report\n"
        text += f"{'='*50}\n\n"
        text += f"Data: {path}\n"
        text += f"Games: {n_games}   Positions: {n_pos}   Candidates/turn: ~{mean([c['n_cands'] for c in lines[:200]]):.0f}\n"
        text += f"Rollouts per candidate: 100\n\n"

        if ai_scores:
            bases = [s["base"] for s in ai_scores]
            text += f"--- Game Scores (player 0) ---\n"
            text += f"  Base: mean={mean(bases):.1f}  std={std(bases):.1f}  min={min(bases)}  max={max(bases)}\n"
            for wl in ["bear","elk","salmon","hawk","fox"]:
                v = [s[wl] for s in ai_scores]
                text += f"  {wl:>6}: mean={mean(v):5.1f}  std={std(v):4.1f}  min={min(v):3}  max={max(v):3}\n"
            text += f"  hab: {mean([s['hab'] for s in ai_scores]):.1f}  "
            text += f"wl_total: {mean([s['wl'] for s in ai_scores]):.1f}  "
            text += f"tokens: {mean([s['tok'] for s in ai_scores]):.1f}\n\n"

        text += f"--- Rank Correlation ---\n"
        text += f"  Kendall tau:  {mean(all_tau):.3f} (std={std(all_tau):.3f})\n"
        text += f"  Spearman rho: {mean(all_rho):.3f} (std={std(all_rho):.3f})\n"
        all_agree = [v for dd in per_turn.values() for v in dd["top1_agree"]]
        text += f"  Top-1 agree:  {100*mean(all_agree):.1f}% ({sum(all_agree)}/{len(all_agree)})\n\n"

        text += f"--- MCE Top-1 in NNUE Top-X ---\n"
        for x in TOP_X:
            if all_miss[x]:
                hit = 100*(1-mean(all_miss[x]))
                text += f"  K={x:>3}: {hit:5.1f}% hit  {100-hit:5.1f}% miss\n"
        if all_gap:
            text += f"\n  Avg gap when K=8 misses: {mean(all_gap):.2f} pts\n\n"

        text += f"--- MCE Rollout Std Dev ---\n"
        all_stds = [c["mce_std"] for c in lines]
        mce1_stds = [c["mce_std"] for c in lines if c["mce_rank"]==0]
        nnue1_stds = [c["mce_std"] for c in lines if c["nnue_rank"]==0]
        text += f"  All cands:  mean={mean(all_stds):.2f}\n"
        text += f"  MCE top-1:  mean={mean(mce1_stds):.2f}\n"
        text += f"  NNUE top-1: mean={mean(nnue1_stds):.2f}\n"

        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig); plt.close()

        # --- Page 2: Hit rate curve ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        # Overall
        ax = axes[0]
        xs = [x for x in TOP_X if all_miss.get(x)]
        ys = [100*(1-mean(all_miss[x])) for x in xs]
        ax.bar(range(len(xs)), ys, tick_label=[f"K={x}" for x in xs], color="steelblue")
        ax.set_ylabel("Hit Rate (%)")
        ax.set_title("MCE Top-1 Found in NNUE Top-K (Overall)")
        ax.set_ylim(0, 105)
        for i, v in enumerate(ys):
            ax.text(i, v+1, f"{v:.0f}%", ha="center", fontsize=8)

        # By turn bucket
        ax = axes[1]
        buckets = [(1,5,"T1-5"),(6,10,"T6-10"),(11,15,"T11-15"),(16,20,"T16-20")]
        show_x = [1, 4, 8, 16, 32]
        width = 0.15
        for bi, (lo,hi,label) in enumerate(buckets):
            vals = []
            for x in show_x:
                bm = []
                for t in range(lo,hi+1):
                    bm.extend(per_turn[t]["miss"].get(x,[]))
                vals.append(100*(1-mean(bm)) if bm else 0)
            positions_x = [i + bi*width for i in range(len(show_x))]
            ax.bar(positions_x, vals, width, label=label)
        ax.set_xticks([i + 1.5*width for i in range(len(show_x))])
        ax.set_xticklabels([f"K={x}" for x in show_x])
        ax.set_ylabel("Hit Rate (%)")
        ax.set_title("MCE Top-1 in NNUE Top-K (by Turn Phase)")
        ax.legend()
        ax.set_ylim(0, 105)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # --- Page 3: Per-turn breakdown ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        turns = sorted(per_turn.keys())

        # Miss rate by turn
        ax = axes[0]
        miss8 = [100*mean(per_turn[t]["miss"].get(8,[])) if per_turn[t]["miss"].get(8) else 0 for t in turns]
        miss16 = [100*mean(per_turn[t]["miss"].get(16,[])) if per_turn[t]["miss"].get(16) else 0 for t in turns]
        ax.plot(turns, miss8, "o-", label="K=8 miss%", color="red")
        ax.plot(turns, miss16, "s--", label="K=16 miss%", color="orange")
        ax.set_xlabel("Turn"); ax.set_ylabel("Miss Rate (%)")
        ax.set_title("NNUE Prefilter Miss Rate by Turn")
        ax.legend(); ax.set_ylim(0, 70); ax.grid(True, alpha=0.3)

        # Tau + Rho by turn
        ax = axes[1]
        taus = [mean(per_turn[t]["tau"]) for t in turns]
        rhos = [mean(per_turn[t]["rho"]) for t in turns]
        ax.plot(turns, taus, "o-", label="Kendall tau", color="blue")
        ax.plot(turns, rhos, "s-", label="Spearman rho", color="green")
        ax.set_xlabel("Turn"); ax.set_ylabel("Correlation")
        ax.set_title("NNUE-MCE Rank Correlation by Turn")
        ax.legend(); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

        # MCE std by turn
        ax = axes[2]
        std_all = [mean(per_turn[t]["std_all"]) if per_turn[t]["std_all"] else 0 for t in turns]
        std_m1 = [mean(per_turn[t]["std_mce1"]) if per_turn[t]["std_mce1"] else 0 for t in turns]
        std_n1 = [mean(per_turn[t]["std_nnue1"]) if per_turn[t]["std_nnue1"] else 0 for t in turns]
        ax.plot(turns, std_all, "o-", label="All cands", color="gray")
        ax.plot(turns, std_m1, "s-", label="MCE top-1", color="blue")
        ax.plot(turns, std_n1, "^-", label="NNUE top-1", color="red")
        ax.set_xlabel("Turn"); ax.set_ylabel("MCE Rollout Std Dev")
        ax.set_title("MCE Evaluation Noise by Turn")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # --- Page 4: Scatter plot NNUE rank vs MCE rank ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for bi, (lo, hi, label) in enumerate(buckets):
            ax = axes[bi//2][bi%2]
            subset = [c for c in lines if lo <= c["turn"] <= hi]
            if subset:
                ax.scatter([c["nnue_rank"] for c in subset],
                          [c["mce_rank"] for c in subset],
                          alpha=0.1, s=3, color="steelblue")
                ax.plot([0, 100], [0, 100], "r--", alpha=0.5)
            ax.set_xlabel("NNUE Rank"); ax.set_ylabel("MCE Rank")
            ax.set_title(f"Turn {lo}-{hi}")
            ax.set_xlim(-1, 100); ax.set_ylim(-1, 100)
        plt.suptitle("NNUE Rank vs MCE Rank (each dot = one candidate)")
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # --- Page 5: Top disagreements + per-turn table ---
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.axis("off")
        text = "Per-Turn Detail Table\n"
        text += f"{'Turn':>4} {'tau':>6} {'rho':>6} {'miss8':>6} {'miss16':>7} {'gap':>5} {'std':>5} {'agree':>6} {'n':>4}\n"
        text += "-" * 60 + "\n"
        for t in turns:
            d = per_turn[t]
            m8 = 100*mean(d["miss"].get(8,[])) if d["miss"].get(8) else 0
            m16 = 100*mean(d["miss"].get(16,[])) if d["miss"].get(16) else 0
            g = mean(d["gap"]) if d["gap"] else 0
            s = mean(d["std_all"]) if d["std_all"] else 0
            a = 100*mean(d["top1_agree"]) if d["top1_agree"] else 0
            n = len(d["tau"])
            text += f"{t:>4} {mean(d['tau']):>6.3f} {mean(d['rho']):>6.3f} {m8:>5.0f}% {m16:>6.0f}% {g:>5.1f} {s:>5.2f} {a:>5.0f}% {n:>4}\n"

        text += f"\n\nTop 20 Disagreements (NNUE top-1 != MCE top-1, by MCE gap)\n"
        text += "-" * 70 + "\n"
        disagreements.sort(key=lambda d: -d["mce_gap"])
        for d in disagreements[:20]:
            text += (f"  G{d['game']:2} T{d['turn']:2}: "
                     f"NNUE→mkt{d['nnue_market']}(MCE rk {d['nnue_mce_rank']:2}) "
                     f"MCE→mkt{d['mce_market']}(NNUE rk {d['mce_nnue_rank']:2}) "
                     f"gap={d['mce_gap']:.1f}\n")

        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig); plt.close()

        # --- Page 6: Wildlife + Score distributions ---
        if ai_scores:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            ax = axes[0]
            bases = [s["base"] for s in ai_scores]
            ax.hist(bases, bins=range(min(bases)-1, max(bases)+2), color="steelblue", edgecolor="white")
            ax.set_xlabel("Base Score"); ax.set_ylabel("Count")
            ax.set_title(f"Game Score Distribution (n={len(ai_scores)}, mean={mean(bases):.1f})")

            ax = axes[1]
            wl_names = ["bear", "elk", "salmon", "hawk", "fox"]
            wl_means = [mean([s[w] for s in ai_scores]) for w in wl_names]
            wl_stds = [std([s[w] for s in ai_scores]) for w in wl_names]
            bars = ax.bar(wl_names, wl_means, yerr=wl_stds, color=["#8B4513","#DAA520","#FF6347","#4682B4","#FF8C00"],
                         capsize=5, edgecolor="white")
            ax.set_ylabel("Score"); ax.set_title("Wildlife Score Distribution")
            for bar, v in zip(bars, wl_means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{v:.1f}", ha="center", fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print(f"\nPDF saved to {pdf_path}")


if __name__ == "__main__":
    main()
