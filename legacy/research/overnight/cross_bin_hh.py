#!/usr/bin/env python3
"""Cross-binary head-to-head runner for Cascadia.

Spawns two long-lived `cascadia-cli --daemon` processes — one per binary —
and coordinates them so each plays the seats it owns. Both daemons hold
their own copy of the GameState; we keep them in lockstep by replaying every
move via APPLY on the daemon that didn't pick it.

Per-game protocol (see Rust daemon for full grammar):
  INIT <seed>            both daemons
  loop while not GAMEOVER:
    daemon_picker = daemons[seat_owner[current_player]]
    actions       = daemon_picker.PICK     (mutates picker's state)
    daemon_other.APPLY each action          (keeps states in sync)
    verify state hashes match
  SCORES + BREAKDOWN     both daemons (any one is canonical)

Usage:
  python3 overnight/cross_bin_hh.py \
      --binary-a target-mid-v4/release/cascadia-cli \
      --weights-a nnue_weights_v4opp_modal_iter3.bin \
      --label-a v4opp \
      --binary-b target-mid-v6/release/cascadia-cli \
      --weights-b nnue_weights_v6peak_iter20.bin \
      --label-b v6peak \
      --num-games 50 \
      --jsonl-out overnight/v6peak/cross_bin_hh.jsonl
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

_STOP = {"flag": False}


def _on_sigint(signum, frame):
    _STOP["flag"] = True
    print("\n[cross_bin_hh] Ctrl-C — finishing current game then stopping.", flush=True)


class Daemon:
    """One long-lived `cascadia-cli --daemon` subprocess. Talks via stdin/stdout."""

    def __init__(self, binary, weights, label):
        self.binary = binary
        self.weights = weights
        self.label = label
        self.proc = subprocess.Popen(
            [binary, "--daemon", "--weights", weights],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain READY line
        line = self._readline_blocking()
        if line.strip() != "READY":
            raise RuntimeError(f"{label}: expected READY, got: {line!r}")

    def _readline_blocking(self):
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError(f"{self.label}: daemon EOF; stderr: {err!r}")
        return line

    def cmd(self, *parts):
        msg = " ".join(str(p) for p in parts) + "\n"
        try:
            self.proc.stdin.write(msg)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            err = self.proc.stderr.read()
            raise RuntimeError(f"{self.label}: write failed ({e}); stderr: {err!r}")
        return self._readline_blocking().rstrip("\n")

    def init_game(self, seed):
        resp = self.cmd("INIT", seed)
        if not resp.startswith("OK "):
            raise RuntimeError(f"{self.label}: INIT failed: {resp!r}")
        return int(resp.split()[1])

    def pick(self):
        resp = self.cmd("PICK")
        if resp == "ACTIONS":
            return []
        if not resp.startswith("ACTIONS "):
            raise RuntimeError(f"{self.label}: bad PICK resp: {resp!r}")
        return resp[len("ACTIONS "):].split(";")

    def apply(self, action):
        resp = self.cmd("APPLY", action)
        if not resp.startswith("OK "):
            raise RuntimeError(f"{self.label}: APPLY '{action}' failed: {resp!r}")
        return int(resp.split()[1])

    def hash_state(self):
        resp = self.cmd("HASH")
        return int(resp.split()[1])

    def gameover(self):
        return self.cmd("GAMEOVER") == "YES"

    def curplayer(self):
        return int(self.cmd("CURPLAYER").split()[1])

    def scores(self):
        resp = self.cmd("SCORES")
        return [int(x) for x in resp.split()[1:]]

    def breakdown(self):
        resp = self.cmd("BREAKDOWN")
        # BREAKDOWN p=0 ...| p=1 ...| ...
        line = resp[len("BREAKDOWN "):]
        out = []
        kv_re = re.compile(r"(\w+)=(-?\d+)")
        for chunk in line.split("|"):
            d = {}
            for k, v in kv_re.findall(chunk):
                d[k] = int(v)
            out.append(d)
        return out

    def quit(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(daemon_a, daemon_b, seat_owner, seed, verify_each_move):
    """Play one full game. seat_owner[i] in {0,1}: 0 = daemon_a owns seat i."""
    daemons = (daemon_a, daemon_b)

    h_a = daemon_a.init_game(seed)
    h_b = daemon_b.init_game(seed)
    if h_a != h_b:
        raise RuntimeError(f"INIT hash mismatch: a={h_a} b={h_b}")

    move_count = 0
    while not daemon_a.gameover():
        if daemon_b.gameover():
            raise RuntimeError("daemon disagree on game-over")
        cur = daemon_a.curplayer()
        cur_b = daemon_b.curplayer()
        if cur != cur_b:
            raise RuntimeError(f"curplayer disagree: a={cur} b={cur_b}")

        owner = seat_owner[cur]
        picker = daemons[owner]
        peer = daemons[1 - owner]

        actions = picker.pick()
        for action in actions:
            peer.apply(action)
        move_count += 1

        if verify_each_move or (move_count % 10 == 0):
            ha = daemon_a.hash_state()
            hb = daemon_b.hash_state()
            if ha != hb:
                raise RuntimeError(
                    f"State drift after move {move_count} (seat {cur} owned by "
                    f"daemon-{'AB'[owner]}): a={ha} b={hb}"
                )

    # Both daemons should agree on scores; capture from daemon_a.
    breakdown = daemon_a.breakdown()
    scores = [p["base"] for p in breakdown]
    bonus = [p["bonus"] for p in breakdown]
    return {
        "scores": scores,
        "bonus": bonus,
        "scores_with_bonus": [s + b for s, b in zip(scores, bonus)],
        "breakdown": breakdown,
        "move_count": move_count,
    }


def seat_assignments(num_games, num_seats=4):
    """Yield seat ownership patterns for fairness — alternate which binary owns
    which seats. Uses a simple round-robin: 4 base patterns × N/4 repeats.
    Each pattern has 2 seats per binary.
    """
    base = [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    for i in range(num_games):
        yield i, base[i % len(base)]


def already_done(jsonl_path):
    seen = set()
    if not os.path.exists(jsonl_path):
        return seen
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(rec.get("game_idx"))
            except Exception:
                pass
    return seen


def summarize(jsonl_path, label_a, label_b):
    """Print pooled stats for daemon A vs daemon B, using seat ownership info."""
    seat_scores = {0: [], 1: []}  # by binary owner
    seat_scores_bonus = {0: [], 1: []}
    games = []
    breakdowns = {0: [], 1: []}

    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            owners = r["seat_owner"]
            scores = r["scores"]
            scores_b = r["scores_with_bonus"]
            for seat, owner in enumerate(owners):
                seat_scores[owner].append(scores[seat])
                seat_scores_bonus[owner].append(scores_b[seat])
                breakdowns[owner].append(r["breakdown"][seat])
            games.append(r)

    def stats(arr):
        if not arr:
            return None
        n = len(arr)
        mn = sum(arr) / n
        var = sum((x - mn) ** 2 for x in arr) / max(n - 1, 1)
        sd = var ** 0.5
        se = sd / (n ** 0.5)
        srt = sorted(arr)
        p10 = srt[max(0, int(n * 0.1) - 1)]
        p90 = srt[min(n - 1, int(n * 0.9))]
        med = srt[n // 2]
        return {"n": n, "mean": mn, "sd": sd, "se": se, "p10": p10,
                "median": med, "p90": p90, "min": srt[0], "max": srt[-1]}

    sa = stats(seat_scores[0])
    sb = stats(seat_scores[1])
    sa_b = stats(seat_scores_bonus[0])
    sb_b = stats(seat_scores_bonus[1])

    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  CROSS-BINARY HH SUMMARY — {len(games)} games")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"\n{label_a} (daemon A): {sa['n']} seat-games")
    print(f"  Base    mean={sa['mean']:.2f} ± {sa['se']:.2f} (sd {sa['sd']:.2f})")
    print(f"          p10={sa['p10']}  med={sa['median']}  p90={sa['p90']}  min/max {sa['min']}/{sa['max']}")
    print(f"  +bonus  mean={sa_b['mean']:.2f} ± {sa_b['se']:.2f}")
    print(f"\n{label_b} (daemon B): {sb['n']} seat-games")
    print(f"  Base    mean={sb['mean']:.2f} ± {sb['se']:.2f} (sd {sb['sd']:.2f})")
    print(f"          p10={sb['p10']}  med={sb['median']}  p90={sb['p90']}  min/max {sb['min']}/{sb['max']}")
    print(f"  +bonus  mean={sb_b['mean']:.2f} ± {sb_b['se']:.2f}")

    # Delta with z-score
    delta = sa['mean'] - sb['mean']
    se_delta = (sa['se']**2 + sb['se']**2) ** 0.5
    z = delta / se_delta if se_delta > 0 else 0
    print(f"\n  Δ (A − B) = {delta:+.2f}  z = {z:+.2f}σ  ({'A' if delta > 0 else 'B'} ahead)")

    # Win rate per game (true HH winner) using official Cascadia tiebreakers:
    # primary: total (base + habitat bonus, includes nature tokens as 1pt each)
    # tiebreak 1: leftover nature tokens (more wins)
    # tiebreak 2: wildlife points (more wins)
    # tiebreak 3: habitat points (more wins)
    def seat_key(seat_idx, r):
        bd = r['breakdown'][seat_idx]
        total = bd['base'] + bd['bonus']
        return (total, bd['tok'], bd['wl'], bd['hab'])

    wins_a = wins_b = ties = wins_a_tiebreak = wins_b_tiebreak = 0
    for r in games:
        owners = r['seat_owner']
        a_keys = [seat_key(i, r) for i in range(4) if owners[i] == 0]
        b_keys = [seat_key(i, r) for i in range(4) if owners[i] == 1]
        a_best = max(a_keys); b_best = max(b_keys)
        primary_tied = (a_best[0] == b_best[0])
        if a_best > b_best:
            wins_a += 1
            if primary_tied: wins_a_tiebreak += 1
        elif b_best > a_best:
            wins_b += 1
            if primary_tied: wins_b_tiebreak += 1
        else:
            ties += 1
    total = wins_a + wins_b + ties
    if total > 0:
        print(f"\n  Win rate (best-seat per side, w/ official Cascadia tiebreakers):")
        print(f"    {label_a}: {wins_a}/{total} ({100*wins_a/total:.1f}%)  [{wins_a_tiebreak} on tiebreak]")
        print(f"    {label_b}: {wins_b}/{total} ({100*wins_b/total:.1f}%)  [{wins_b_tiebreak} on tiebreak]")
        print(f"    Ties:   {ties}/{total} ({100*ties/total:.1f}%) (still equal after tiebreakers)")

    # Per-animal stats per side
    print(f"\n  Per-animal mean (A | B):")
    for animal in ["bear", "elk", "salmon", "hawk", "fox"]:
        a_avg = sum(b[animal] for b in breakdowns[0]) / max(len(breakdowns[0]), 1)
        b_avg = sum(b[animal] for b in breakdowns[1]) / max(len(breakdowns[1]), 1)
        print(f"    {animal:7s} {a_avg:5.2f} | {b_avg:5.2f}  (Δ {a_avg-b_avg:+.2f})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary-a", required=True)
    p.add_argument("--weights-a", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--binary-b", required=True)
    p.add_argument("--weights-b", required=True)
    p.add_argument("--label-b", default="B")
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=20260420)
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--verify-each-move", action="store_true",
                   help="Hash-check after every move (slow); default verifies every 10th.")
    p.add_argument("--summarize", help="Read JSONL and print summary, skip running.")
    args = p.parse_args()

    if args.summarize:
        summarize(args.summarize, args.label_a, args.label_b)
        return

    signal.signal(signal.SIGINT, _on_sigint)

    seen = already_done(args.jsonl_out)
    print(f"[cross_bin_hh] {len(seen)} games already in {args.jsonl_out}; resuming.", flush=True)

    print(f"[cross_bin_hh] launching daemons", flush=True)
    daemon_a = Daemon(args.binary_a, args.weights_a, args.label_a)
    daemon_b = Daemon(args.binary_b, args.weights_b, args.label_b)

    os.makedirs(os.path.dirname(args.jsonl_out) or ".", exist_ok=True)

    t0 = time.time()
    n_done_now = 0
    try:
        for game_idx, owners in seat_assignments(args.num_games):
            if game_idx in seen:
                continue
            if _STOP["flag"]:
                break
            seed = args.seed_base + game_idx
            tg = time.time()
            try:
                result = play_game(daemon_a, daemon_b, owners, seed,
                                   verify_each_move=args.verify_each_move)
            except Exception as e:
                print(f"[cross_bin_hh] game {game_idx} FAILED: {e}", flush=True)
                # restart daemons; one likely crashed
                daemon_a.quit(); daemon_b.quit()
                daemon_a = Daemon(args.binary_a, args.weights_a, args.label_a)
                daemon_b = Daemon(args.binary_b, args.weights_b, args.label_b)
                continue

            elapsed = time.time() - tg
            rec = {
                "game_idx": game_idx,
                "seed": seed,
                "seat_owner": owners,
                "label_a": args.label_a,
                "label_b": args.label_b,
                "scores": result["scores"],
                "bonus": result["bonus"],
                "scores_with_bonus": result["scores_with_bonus"],
                "breakdown": result["breakdown"],
                "move_count": result["move_count"],
                "elapsed_s": elapsed,
            }
            with open(args.jsonl_out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done_now += 1
            done_total = len(seen) + n_done_now
            avg_time = (time.time() - t0) / n_done_now
            eta_min = avg_time * (args.num_games - done_total) / 60.0
            print(
                f"[g{game_idx:03d} owners={owners}] base={result['scores']} "
                f"+bonus={result['scores_with_bonus']} "
                f"({elapsed:.1f}s; avg {avg_time:.1f}s/game; ETA {eta_min:.1f} min)",
                flush=True,
            )
    finally:
        daemon_a.quit()
        daemon_b.quit()

    summarize(args.jsonl_out, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
