"""Modal-dispatched NNUE-vs-MCE rank correlation diagnostic.

Measures how well the NNUE prefilter ranking predicts the MCE "ground truth"
ranking of candidate moves, across many game positions (seeds x turns).

Each Modal worker runs one game with --rank-correlation, producing per-turn
per-candidate RANKCORR lines. The local entrypoint aggregates all lines and
computes:
  - Kendall's tau and Spearman's rho (overall and per-turn-bucket)
  - NNUE top-K miss rate (how often the MCE-best move was outside NNUE top-K)
  - Average score gap when the MCE-best was missed
  - Top disagreements (positions where NNUE and MCE most disagree)

Usage:
    modal run overnight/rank_correlation_modal.py \
        --num-games 40 --rollouts-per-cand 100 \
        --weights nnue_weights_mid_fsp_iter10.bin

Cost estimate: ~40 games x ~5min each x 8 vCPUs x $0.000014/s = ~$1.35
"""

import hashlib
import modal
import os
import re
import sys
import time
from collections import defaultdict

app = modal.App("cascadia-rankcorr")

# Persistent weights cache (shared with head_to_head_modal.py)
weights_volume = modal.Volume.from_name("cascadia-weights-cache", create_if_missing=True)

# Build image with mid-features (required for iter10 weights)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir("crates", remote_path="/app/crates", copy=True)
    .add_local_file("Cargo.toml", remote_path="/app/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", remote_path="/app/Cargo.lock", copy=True)
    .run_commands(
        "cd /app && cargo build --release --features mid-features --bin cascadia-cli",
    )
)


def _content_addressed_name(local_path: str) -> str:
    """basename + first 8 chars of sha256 for dedup on the weights volume."""
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()[:8]
    base = os.path.basename(local_path)
    stem, ext = os.path.splitext(base)
    return f"{stem}-{digest}{ext}"


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=1800,
    volumes={"/weights": weights_volume},
)
def rank_correlation_single_game(
    seed_offset: int,
    weight_name: str,
    rollouts_per_cand: int,
    mce_opponents: bool = False,
) -> dict:
    """Run one game's rank-correlation diagnostic. Returns parsed JSONL lines."""
    import subprocess as sp

    weights_path = f"/weights/{weight_name}"
    cmd = [
        "/app/target/release/cascadia-cli", "0",
        "--rank-correlation",
        "--weights", weights_path,
        "--rollouts-per-cand", str(rollouts_per_cand),
        "--max-candidates", "100",
    ]
    if mce_opponents:
        cmd.append("--mce-opponents")
    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    env["MCE_MAX_EXTRA_CANDS"] = "150"

    result = sp.run(cmd, capture_output=True, text=True, cwd="/app", env=env)

    # Parse JSONL lines from stdout
    import json as _json
    lines = []
    scores = []
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        if obj.get("type") == "score":
            scores.append(obj)
        elif "turn" in obj and "nnue_rank" in obj:
            lines.append(obj)

    return {
        "seed_offset": seed_offset,
        "lines": lines,
        "scores": scores,
        "n_turns": max((l["turn"] for l in lines), default=0),
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def compute_kendall_tau(ranks_a, ranks_b):
    """Kendall's tau-b rank correlation. Pure Python, no scipy needed."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    tied_a = 0
    tied_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = ranks_a[i] - ranks_a[j]
            db = ranks_b[i] - ranks_b[j]
            if da == 0 and db == 0:
                tied_a += 1
                tied_b += 1
            elif da == 0:
                tied_a += 1
            elif db == 0:
                tied_b += 1
            elif (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom_a = n_pairs - tied_a
    denom_b = n_pairs - tied_b
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return (concordant - discordant) / ((denom_a * denom_b) ** 0.5)


def compute_spearman_rho(ranks_a, ranks_b):
    """Spearman's rho rank correlation."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


@app.local_entrypoint()
def run(
    num_games: int = 40,
    rollouts_per_cand: int = 100,
    weights: str = "nnue_weights_mid_fsp_iter10.bin",
    top_k: int = 8,
    mce_opponents: bool = False,
):
    """Dispatch rank-correlation diagnostic across Modal workers."""

    # Upload weights
    if not os.path.exists(weights):
        sys.exit(f"Weights file not found: {weights}")
    remote_name = _content_addressed_name(weights)

    print(f"Syncing weights to Modal volume...")
    t_up = time.time()
    try:
        existing = {e.path.lstrip("/") for e in weights_volume.iterdir("/")}
    except Exception:
        existing = set()
    if remote_name not in existing:
        with weights_volume.batch_upload() as batch:
            batch.put_file(weights, f"/{remote_name}")
        print(f"  Uploaded {weights} as {remote_name} ({time.time() - t_up:.1f}s)")
    else:
        print(f"  Weights already on volume ({remote_name}), skipped upload")

    print(f"\nRank correlation diagnostic:")
    print(f"  Games: {num_games}")
    print(f"  Rollouts per candidate: {rollouts_per_cand}")
    print(f"  Weights: {weights}")
    print(f"  Top-K for miss rate: {top_k}")
    print()

    # Dispatch all games in parallel
    t0 = time.time()
    futures = []
    for game_i in range(num_games):
        seed_offset = game_i * 1000
        f = rank_correlation_single_game.spawn(
            seed_offset=seed_offset,
            weight_name=remote_name,
            rollouts_per_cand=rollouts_per_cand,
            mce_opponents=mce_opponents,
        )
        futures.append((game_i, f))

    print(f"Dispatched {len(futures)} games in {time.time() - t0:.1f}s, awaiting results...")

    # Gather all results
    all_lines = []
    all_scores = []
    for game_i, f in futures:
        result = f.get()
        n_lines = len(result["lines"])
        n_turns = result["n_turns"]
        elapsed = time.time() - t0
        print(f"  Game {game_i}: {n_turns} turns, {n_lines} candidates ({elapsed:.0f}s elapsed)")
        for line in result["lines"]:
            line["game"] = game_i
        all_lines.extend(result["lines"])
        for s in result.get("scores", []):
            s["game"] = game_i
        all_scores.extend(result.get("scores", []))

    total_elapsed = time.time() - t0
    print(f"\nAll games complete in {total_elapsed:.0f}s, {len(all_lines)} total candidate lines")

    # Save raw data as JSONL for offline re-analysis
    import json
    opp_tag = "mce_opp" if mce_opponents else "nnue_opp"
    raw_data_path = f"overnight/rank_corr_raw_{num_games}g_{rollouts_per_cand}r_{opp_tag}.jsonl"
    with open(raw_data_path, "w") as f:
        for line in all_lines:
            f.write(json.dumps(line) + "\n")
        for score in all_scores:
            f.write(json.dumps(score) + "\n")
    print(f"Raw data saved to {raw_data_path} ({len(all_lines)} candidate lines, {len(all_scores)} score lines)")
    print(f"  Re-analyze locally: python3 overnight/analyze_rank_corr.py {raw_data_path}")

    # ========== ANALYSIS ==========
    positions = defaultdict(list)
    for line in all_lines:
        positions[(line["game"], line["turn"])].append(line)

    print(f"Analyzing {len(positions)} positions across {num_games} games...\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Per-position metrics
    all_tau = []
    all_rho = []
    TOP_X_VALUES = [1, 2, 4, 8, 12, 16, 24, 32, 64]
    all_miss = {}  # keyed by top_x: list of 0/1
    for x in TOP_X_VALUES:
        all_miss[x] = []
    all_gap_when_missed = []
    per_turn = defaultdict(lambda: {
        "tau": [], "rho": [],
        "miss": {x: [] for x in TOP_X_VALUES},
        "gap": [],
        "mce_std_all": [],       # std of ALL candidates this turn
        "mce_std_mce_top1": [],  # std of MCE's top-1
        "mce_std_nnue_top1": [], # std of NNUE's top-1
    })
    disagreements = []

    for (game_i, turn), cands in positions.items():
        if len(cands) < 2:
            continue

        nnue_ranks = [c["nnue_rank"] for c in cands]
        mce_ranks = [c["mce_rank"] for c in cands]

        tau = compute_kendall_tau(nnue_ranks, mce_ranks)
        rho = compute_spearman_rho(nnue_ranks, mce_ranks)
        all_tau.append(tau)
        all_rho.append(rho)
        per_turn[turn]["tau"].append(tau)
        per_turn[turn]["rho"].append(rho)

        # MCE std stats
        all_stds = [c.get("mce_std", 0) for c in cands]
        per_turn[turn]["mce_std_all"].extend(all_stds)

        mce_best = [c for c in cands if c["mce_rank"] == 0]
        nnue_best = [c for c in cands if c["nnue_rank"] == 0]
        if mce_best:
            per_turn[turn]["mce_std_mce_top1"].append(mce_best[0].get("mce_std", 0))
        if nnue_best:
            per_turn[turn]["mce_std_nnue_top1"].append(nnue_best[0].get("mce_std", 0))

        # Miss rate at various K thresholds
        if mce_best:
            mce_best_nnue_rank = mce_best[0]["nnue_rank"]
            for x in TOP_X_VALUES:
                missed = 1 if mce_best_nnue_rank >= x else 0
                all_miss[x].append(missed)
                per_turn[turn]["miss"][x].append(missed)

            if mce_best_nnue_rank >= top_k and nnue_best:
                gap = mce_best[0]["mce_score"] - nnue_best[0]["mce_score"]
                all_gap_when_missed.append(gap)
                per_turn[turn]["gap"].append(gap)

        # Disagreements
        if nnue_best and mce_best:
            if nnue_best[0]["mce_rank"] != 0 or mce_best[0]["nnue_rank"] != 0:
                nnue_gap = nnue_best[0]["nnue_score"] - mce_best[0]["nnue_score"]
                mce_gap = mce_best[0]["mce_score"] - nnue_best[0]["mce_score"]
                disagreements.append({
                    "game": game_i, "turn": turn,
                    "nnue_top1_market": nnue_best[0]["market"],
                    "mce_top1_market": mce_best[0]["market"],
                    "nnue_gap": nnue_gap, "mce_gap": mce_gap,
                    "nnue_top1_mce_rank": nnue_best[0]["mce_rank"],
                    "mce_top1_nnue_rank": mce_best[0]["nnue_rank"],
                    "n_cands": cands[0]["n_cands"],
                })

    # ========== REPORT ==========
    print("=" * 70)
    print("  NNUE vs MCE Rank Correlation Report")
    print("=" * 70)

    # --- 1. Game score distribution ---
    ai_scores = [s for s in all_scores if s["player"] == 0]
    if ai_scores:
        bases = [s["base"] for s in ai_scores]
        bonuses = [s["bonus"] for s in ai_scores]
        print(f"\n--- Game Score Distribution (player 0, {len(ai_scores)} games) ---")
        print(f"  Base:  mean={mean(bases):.1f}  min={min(bases)}  max={max(bases)}  std={std(bases):.1f}")
        print(f"  Bonus: mean={mean(bonuses):.1f}  min={min(bonuses)}  max={max(bonuses)}")
        print(f"\n--- Wildlife Score Distribution ---")
        for wl in ["bear", "elk", "salmon", "hawk", "fox"]:
            vals = [s[wl] for s in ai_scores]
            print(f"  {wl:>6}: mean={mean(vals):.1f}  min={min(vals)}  max={max(vals)}  std={std(vals):.1f}")
        hab_vals = [s["hab"] for s in ai_scores]
        tok_vals = [s["tok"] for s in ai_scores]
        wl_vals = [s["wl"] for s in ai_scores]
        print(f"  {'hab':>6}: mean={mean(hab_vals):.1f}  {'wl_total':>6}: mean={mean(wl_vals):.1f}  {'tokens':>6}: mean={mean(tok_vals):.1f}")
    else:
        print("\n(No game scores captured)")

    # --- 2. Rank correlation ---
    print(f"\n--- Rank Correlation ({len(positions)} positions) ---")
    print(f"  Kendall's tau:  {mean(all_tau):.3f}  (std={std(all_tau):.3f})")
    print(f"  Spearman's rho: {mean(all_rho):.3f}  (std={std(all_rho):.3f})")

    # --- 3. Top-1 agreement ---
    n_top1_agree = len(positions) - len(disagreements)
    print(f"\n--- Top-1 Agreement ---")
    print(f"  NNUE top-1 == MCE top-1: {100.0 * n_top1_agree / max(len(positions), 1):.1f}% ({n_top1_agree}/{len(positions)})")

    # --- 4. MCE top-1 in NNUE top-X ---
    print(f"\n--- MCE Top-1 Found in NNUE Top-X (avg ~{mean([c['n_cands'] for c in all_lines[:100]]):.0f} candidates/turn) ---")
    for x in TOP_X_VALUES:
        if all_miss[x]:
            hit_rate = 100.0 * (1.0 - mean(all_miss[x]))
            n_effective = sum(1 for c in all_lines if c.get("n_cands", 999) >= x) // max(1, len(set((c.get("game",0), c["turn"]) for c in all_lines)))
            print(f"  Top-{x:>3}: {hit_rate:5.1f}% hit   {100-hit_rate:5.1f}% miss   (prefilter at K={x} would catch {hit_rate:.0f}% of MCE-best)")

    if all_gap_when_missed:
        print(f"\n  Avg MCE score gap when missed (top-{top_k}): {mean(all_gap_when_missed):.2f} pts")

    # --- 5. MCE std deviation ---
    all_stds_flat = [c.get("mce_std", 0) for c in all_lines if "mce_std" in c]
    all_mce_top1_stds = []
    all_nnue_top1_stds = []
    for (_, _), cands in positions.items():
        mb = [c for c in cands if c["mce_rank"] == 0]
        nb = [c for c in cands if c["nnue_rank"] == 0]
        if mb: all_mce_top1_stds.append(mb[0].get("mce_std", 0))
        if nb: all_nnue_top1_stds.append(nb[0].get("mce_std", 0))

    print(f"\n--- MCE Rollout Std Deviation ---")
    print(f"  All candidates:  mean={mean(all_stds_flat):.2f}  std={std(all_stds_flat):.2f}")
    print(f"  MCE top-1 only:  mean={mean(all_mce_top1_stds):.2f}  std={std(all_mce_top1_stds):.2f}")
    print(f"  NNUE top-1 only: mean={mean(all_nnue_top1_stds):.2f}  std={std(all_nnue_top1_stds):.2f}")

    # --- 6. Per-turn breakdown ---
    print(f"\n--- Per-Turn Breakdown ---")
    print(f"{'Turn':>5} {'tau':>6} {'rho':>6} {'miss8':>6} {'gap':>5} {'std_all':>8} {'std_mce1':>9} {'std_nnue1':>10} {'top1_agree':>11} {'n':>4}")
    print("-" * 85)

    for t in sorted(per_turn.keys()):
        d = per_turn[t]
        if not d["tau"]: continue
        miss8 = 100.0 * mean(d["miss"].get(8, [])) if d["miss"].get(8) else 0
        gap_v = mean(d["gap"]) if d["gap"] else 0
        std_all = mean(d["mce_std_all"]) if d["mce_std_all"] else 0
        std_m1 = mean(d["mce_std_mce_top1"]) if d["mce_std_mce_top1"] else 0
        std_n1 = mean(d["mce_std_nnue_top1"]) if d["mce_std_nnue_top1"] else 0
        # top-1 agreement for this turn
        n_pos = len(d["tau"])
        n_agree = n_pos - sum(1 for m8 in d["miss"].get(1, []) if m8 == 1)
        agree_pct = 100.0 * n_agree / max(n_pos, 1)
        print(f"{t:>5} {mean(d['tau']):>6.3f} {mean(d['rho']):>6.3f} {miss8:>5.0f}% {gap_v:>5.1f} {std_all:>8.2f} {std_m1:>9.2f} {std_n1:>10.2f} {agree_pct:>10.0f}% {n_pos:>4}")

    # --- 7. Per-turn-bucket summary ---
    print(f"\n--- Per-Turn-Bucket Summary ---")
    for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20)]:
        bt, br, bm, bg = [], [], [], []
        bs_all, bs_m1, bs_n1 = [], [], []
        for t in range(lo, hi + 1):
            bt.extend(per_turn[t]["tau"])
            br.extend(per_turn[t]["rho"])
            bm.extend(per_turn[t]["miss"].get(8, []))
            bg.extend(per_turn[t]["gap"])
            bs_all.extend(per_turn[t]["mce_std_all"])
            bs_m1.extend(per_turn[t]["mce_std_mce_top1"])
            bs_n1.extend(per_turn[t]["mce_std_nnue_top1"])
        if not bt: continue
        print(f"  Turn {lo:2}-{hi:2}: tau={mean(bt):.3f} rho={mean(br):.3f} "
              f"miss8={100*mean(bm):.0f}% gap={mean(bg):.1f} "
              f"std_all={mean(bs_all):.2f} std_mce1={mean(bs_m1):.2f} std_nnue1={mean(bs_n1):.2f} "
              f"(n={len(bt)})")

    # --- 8. MCE top-1 in NNUE top-X by turn bucket ---
    show_x = [1, 2, 4, 8, 12, 16, 24, 32]
    print(f"\n--- MCE Top-1 in NNUE Top-X (by turn bucket) ---")
    print(f"  {'Turns':>6}", end="")
    for x in show_x:
        print(f"  {'K='+str(x):>6}", end="")
    print()
    print(f"  {'------':>6}", end="")
    for _ in show_x:
        print(f"  {'------':>6}", end="")
    print()
    for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20)]:
        print(f"  {lo:2}-{hi:2}  ", end="")
        for x in show_x:
            bm = []
            for t in range(lo, hi + 1):
                bm.extend(per_turn[t]["miss"].get(x, []))
            hit = 100.0 * (1.0 - mean(bm)) if bm else 0
            print(f"  {hit:5.1f}%", end="")
        print()
    # All turns
    print(f"  {'All':>5}  ", end="")
    for x in show_x:
        if all_miss.get(x):
            hit = 100.0 * (1.0 - mean(all_miss[x]))
            print(f"  {hit:5.1f}%", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

    # --- 9. Top disagreements ---
    print(f"\n--- Top 15 Disagreements (sorted by MCE score gap) ---")
    disagreements.sort(key=lambda d: -d["mce_gap"])
    for d in disagreements[:15]:
        print(f"  Game {d['game']:2} Turn {d['turn']:2}: "
              f"NNUE→market={d['nnue_top1_market']} (MCE rank {d['nnue_top1_mce_rank']}), "
              f"MCE→market={d['mce_top1_market']} (NNUE rank {d['mce_top1_nnue_rank']}), "
              f"MCE gap={d['mce_gap']:.1f}, n={d['n_cands']}")

    estimated_cost = num_games * 300 * 8 * 0.000014
    print(f"\nWall clock: {total_elapsed:.0f}s  Estimated cost: ~${estimated_cost:.2f}")


def std(xs):
    """Standard deviation."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    variance = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return variance ** 0.5
