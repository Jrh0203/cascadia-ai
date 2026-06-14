"""Tandem training coordinator: run SA + split-head trainings synchronized per iteration.

The design (per user direction 2026-04-11):
- One self-play batch per iteration, shared between both trainings.
- Self-play uses the "best current model" (highest recent NNUE bench) for both
  player 0 and the opponents.
- Both trainings train in parallel on the identical merged file.
- Wait for both to finish before generating next iter's self-play.
- Quick-bench each new iter's weights and pick the winner as next iter's driver.

This is cleaner than two fully independent runs with `--opp-weights` cross-pollination:
- Halves self-play cost (one batch serves both trainings)
- Apples-to-apples training-data comparison
- Self-play quality is monotonically improving (always driven by the best model)
- Avoids stale-opponent issues

Output layout:
    tandem_iter{N}/
        self_play.bin        (MCV3)
        training.bin         (MCV3, self-play + MCE cache if available)
        sa_iter{N}.bin       (SA weights, version 1 file format)
        split_iter{N}.bin    (split-head weights, version 2 file format)
        bench_sa.log
        bench_split.log
    train_tandem.log         (stdout mirror)

Usage:
    python3 train_tandem.py --iterations 10 --self-play-games 100000 \\
        --epochs-per-iter 15 --lr 3e-05 \\
        --temperature-start 2.0 --temperature-end 0.1 \\
        --epsilon-start 0.3 --epsilon-end 0.05
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

CASCADIA_CLI = "./target/release/cascadia-cli"
TANDEM_ROOT = "tandem_runs"


def run_self_play(num_games, weights_path, out_path, epsilon=None, temperature=None):
    """Generate self-play data using the given weights.
    If temperature is set, uses softmax sampling; otherwise ε-greedy.
    Writes MCV3 format.
    """
    cmd = [
        CASCADIA_CLI, str(num_games),
        "--self-play",
        "--out", out_path,
        "--aux-targets",  # always on — MCV3 format for both trainings
    ]
    if weights_path and os.path.exists(weights_path):
        cmd.extend(["--weights", weights_path])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    else:
        cmd.extend(["--epsilon", str(epsilon if epsilon is not None else 0.1)])

    mode_str = (f"softmax(T={temperature})" if temperature is not None
                else f"ε={epsilon}")
    print(f"  [self-play] {num_games} games, {mode_str}, weights={weights_path or 'fresh'}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [self-play] FAILED: {r.stderr[-500:]}")
        return False
    # Extract self-play mean score from stdout
    for line in r.stdout.splitlines():
        if "Mean:" in line and "self-play" in r.stdout.lower():
            pass
    print(f"  [self-play] done in {time.time()-t0:.0f}s")
    return True


def merge_with_mce_cache(self_play_path, mce_cache_path, out_path):
    """Concatenate self-play file with the MCE cache if the cache is MCV3.
    Both must be MCV3 (magic b'MCV3') to be compatible.
    """
    with open(self_play_path, 'rb') as f:
        sp_data = f.read()
    sp_magic = sp_data[:4]
    with open(out_path, 'wb') as out:
        out.write(sp_data)

    if mce_cache_path and os.path.exists(mce_cache_path):
        with open(mce_cache_path, 'rb') as f:
            mce_data = f.read()
        mce_magic = mce_data[:4]
        if sp_magic == mce_magic == b'MCV3':
            with open(out_path, 'ab') as out:
                out.write(mce_data[4:])  # skip magic
            print(f"  [merge] self-play + MCE cache → {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")
            return
        else:
            print(f"  [merge] skip MCE cache (magic={mce_magic!r}, need MCV3)")
    print(f"  [merge] self-play only → {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")


def _build_train_cmd(variant, samples_path, init_weights, out_weights, args):
    cmd = [
        sys.executable, "-u", "train_pytorch.py", "value",
        "--samples", samples_path,
        "--epochs", str(args.epochs_per_iter),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--hidden1", str(args.hidden1),
        "--hidden2", str(args.hidden2),
        "--optimizer", args.optimizer,
        "--out", out_weights,
        "--no-augment",
        "--use-aux",
        "--aux-bear-weight", str(args.aux_bear_weight),
        "--aux-salmon-weight", str(args.aux_salmon_weight),
    ]
    if variant == "split":
        cmd.append("--split-value-head")
    if init_weights and os.path.exists(init_weights):
        cmd.extend(["--init-weights", init_weights])
    return cmd


def train_variant(variant, samples_path, init_weights, out_weights, args, logfile):
    """Launch a training subprocess for one variant (background). Returns Popen."""
    cmd = _build_train_cmd(variant, samples_path, init_weights, out_weights, args)
    logf = open(logfile, 'w')
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf


def train_variant_serial(variant, samples_path, init_weights, out_weights, args, logfile):
    """Run a training subprocess foreground (blocking). Returns exit code.

    Use this when --serial is set so the two variants don't fight for MPS GPU time.
    On Apple Silicon MPS, multi-process GPU sharing has unfair scheduling and causes
    catastrophic epoch-time variance (13× range on iter1). Serial execution is
    actually faster in wall time despite being 'sequential' because each epoch runs
    at predictable speed (~130s vs 78-1062s under contention).
    """
    cmd = _build_train_cmd(variant, samples_path, init_weights, out_weights, args)
    with open(logfile, 'w') as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return r.returncode


def quick_bench(weights_path, num_games=50, logfile=None):
    """Quick NNUE bench — returns mean base score or None on failure."""
    if not os.path.exists(weights_path):
        return None
    cmd = [CASCADIA_CLI, str(num_games), "--nnue", "--weights", weights_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if logfile:
        with open(logfile, 'w') as f:
            f.write(r.stdout)
    if r.returncode != 0:
        return None
    # First "Mean:" line is the base-score mean.
    m = re.search(r"Mean:\s+([\d.]+)", r.stdout)
    if m:
        return float(m.group(1))
    return None


def anneal(start, end, iteration, total_iterations):
    """Linear anneal from start (iter 1) to end (iter total_iterations)."""
    if total_iterations <= 1:
        return start
    progress = (iteration - 1) / (total_iterations - 1)
    return start + progress * (end - start)


def main():
    parser = argparse.ArgumentParser(description="Tandem SA + split-head training coordinator")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=100000)
    parser.add_argument("--epochs-per-iter", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--aux-bear-weight", type=float, default=0.3)
    parser.add_argument("--aux-salmon-weight", type=float, default=0.3)

    # Sampling schedules
    parser.add_argument("--temperature-start", type=float, default=2.0,
                        help="SA initial temperature (iter 1). SA variant uses this.")
    parser.add_argument("--temperature-end", type=float, default=0.1,
                        help="SA final temperature (last iter). Linear anneal.")
    parser.add_argument("--epsilon-start", type=float, default=0.3,
                        help="Fallback epsilon if temperature is None. Not used by SA — "
                             "the driver for self-play is always the best current model.")
    parser.add_argument("--epsilon-end", type=float, default=0.05)

    # Which variant drives self-play mode. "sa" means use softmax sampling with the
    # annealed temperature; "best" means use ε-greedy with annealed ε (safer default).
    parser.add_argument("--self-play-mode", default="sa", choices=["sa", "epsilon"],
                        help="sa = softmax sampling driven by best model; "
                             "epsilon = ε-greedy driven by best model")

    parser.add_argument("--mce-samples", default="mce_policy_samples.bin",
                        help="MCE cache to mix in (must be MCV3 to be included)")
    parser.add_argument("--init-weights", default=None,
                        help="Starting weights (both variants initialize from here)")
    parser.add_argument("--benchmark-games", type=int, default=50)
    parser.add_argument("--run-name", default="tandem",
                        help="Subdirectory name under tandem_runs/")
    parser.add_argument("--serial", action="store_true",
                        help="Run the two variant trainings sequentially instead of in parallel. "
                             "On MPS GPU this is actually faster in wall time because it avoids "
                             "multi-process GPU contention (13× epoch-time variance in parallel "
                             "mode). Recommended when both variants share the same GPU.")
    parser.add_argument("--iter-offset", type=int, default=0,
                        help="Start numbering iterations from offset+1. Use when continuing a "
                             "prior run (e.g. --iter-offset 2 to resume from iter3).")

    args = parser.parse_args()

    run_dir = Path(TANDEM_ROOT) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_tandem.log"
    with open(log_path, 'w') as f:
        f.write(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Args: {vars(args)}\n")

    print(f"=== Tandem training: SA + split-head ===")
    print(f"  Run dir:             {run_dir}")
    print(f"  Iterations:          {args.iterations} (starting at iter {args.iter_offset+1})")
    print(f"  Training mode:       {'SERIAL' if args.serial else 'PARALLEL'}")
    print(f"  Self-play games/iter: {args.self_play_games}")
    print(f"  Self-play driver:    {args.self_play_mode}")
    if args.self_play_mode == "sa":
        print(f"  T schedule:          {args.temperature_start} → {args.temperature_end}")
    else:
        print(f"  ε schedule:          {args.epsilon_start} → {args.epsilon_end}")
    print(f"  Architecture:        {args.hidden1}→{args.hidden2}")
    print(f"  LR:                  {args.lr}")
    print(f"  Epochs/iter:         {args.epochs_per_iter}")
    print()

    # The "best" model driving self-play. Initially whatever was passed in (or None).
    best_weights = args.init_weights
    sa_weights = None
    split_weights = None

    total_start = time.time()

    for step in range(1, args.iterations + 1):
        iteration = step + args.iter_offset
        iter_start = time.time()
        iter_dir = run_dir / f"iter{iteration:02d}"
        iter_dir.mkdir(exist_ok=True)

        print(f"{'='*66}")
        print(f"  ITERATION {iteration} (step {step}/{args.iterations})")
        print(f"{'='*66}")

        # 1. Self-play: one batch, driven by best_weights.
        # If self_play.bin already exists in the iter dir (e.g. copied from a prior
        # run you killed), reuse it. This lets you resume without re-paying the
        # self-play generation cost.
        self_play_path = str(iter_dir / "self_play.bin")

        # Anneal on the GLOBAL iter number (iter_offset + step), over the combined
        # run length. This lets a resumed run pick up the schedule where the prior
        # run left off — if original was iters 1..10 and we resume at iter 3, the
        # schedule still targets the same T at iter 10.
        global_total = args.iterations + args.iter_offset
        if args.self_play_mode == "sa":
            current_temp = anneal(args.temperature_start, args.temperature_end, iteration, global_total)
            current_eps = None
        else:
            current_eps = anneal(args.epsilon_start, args.epsilon_end, iteration, global_total)
            current_temp = None

        if os.path.exists(self_play_path) and os.path.getsize(self_play_path) > 1024:
            size_mb = os.path.getsize(self_play_path) / 1e6
            print(f"  [self-play] REUSING existing {self_play_path} ({size_mb:.0f} MB)")
        else:
            ok = run_self_play(
                num_games=args.self_play_games,
                weights_path=best_weights,
                out_path=self_play_path,
                epsilon=current_eps,
                temperature=current_temp,
            )
            if not ok:
                print("  [FATAL] self-play failed, stopping")
                return 1

        # 2. Merge self-play + MCE cache (if MCV3)
        merged_path = str(iter_dir / "training.bin")
        merge_with_mce_cache(self_play_path, args.mce_samples, merged_path)

        # 3. Launch both trainings (serial or parallel) on the same data.
        sa_out = str(iter_dir / f"sa_iter{iteration}.bin")
        split_out = str(iter_dir / f"split_iter{iteration}.bin")
        sa_log = str(iter_dir / "train_sa.log")
        split_log = str(iter_dir / "train_split.log")

        t_train = time.time()
        if args.serial:
            print(f"  [train] running SA on {merged_path} (serial)")
            t_sa = time.time()
            sa_rc = train_variant_serial(
                "sa", merged_path, sa_weights or best_weights, sa_out, args, sa_log,
            )
            print(f"  [train] SA done (exit={sa_rc}, {time.time()-t_sa:.0f}s)")
            print(f"  [train] running split on {merged_path} (serial)")
            t_split = time.time()
            split_rc = train_variant_serial(
                "split", merged_path, split_weights or best_weights, split_out, args, split_log,
            )
            print(f"  [train] split done (exit={split_rc}, {time.time()-t_split:.0f}s)")
        else:
            print(f"  [train] launching SA + split-head in parallel on {merged_path}")
            sa_proc, sa_logf = train_variant(
                "sa", merged_path, sa_weights or best_weights, sa_out, args, sa_log,
            )
            split_proc, split_logf = train_variant(
                "split", merged_path, split_weights or best_weights, split_out, args, split_log,
            )
            sa_rc = sa_proc.wait()
            split_rc = split_proc.wait()
            sa_logf.close()
            split_logf.close()
        print(f"  [train] SA exit={sa_rc}, split exit={split_rc}, "
              f"total_wall={time.time()-t_train:.0f}s")
        if sa_rc != 0 or split_rc != 0:
            print(f"  [WARN] at least one training failed — check logs at {iter_dir}")
            # Keep going: re-use prior weights for the failing variant next iter

        if sa_rc == 0:
            sa_weights = sa_out
        if split_rc == 0:
            split_weights = split_out

        # 4. Quick-bench each new weight file
        print(f"  [bench] running quick NNUE benches ({args.benchmark_games} games each)")
        bench_sa_log = str(iter_dir / "bench_sa.log")
        bench_split_log = str(iter_dir / "bench_split.log")
        sa_score = quick_bench(sa_out, args.benchmark_games, bench_sa_log)
        split_score = quick_bench(split_out, args.benchmark_games, bench_split_log)

        sa_str = f"{sa_score:.1f}" if sa_score is not None else "failed"
        split_str = f"{split_score:.1f}" if split_score is not None else "failed"
        print(f"  [bench] SA={sa_str}  split={split_str}")

        # 5. Pick winner as next iter's self-play driver.
        #    If either bench failed, prefer the other. If both failed, keep prior best.
        new_best = best_weights
        if sa_score is not None and split_score is not None:
            if sa_score >= split_score:
                new_best = sa_out
                print(f"  [pick] SA wins ({sa_score:.1f} ≥ {split_score:.1f}) → next self-play uses {sa_out}")
            else:
                new_best = split_out
                print(f"  [pick] split wins ({split_score:.1f} > {sa_score:.1f}) → next self-play uses {split_out}")
        elif sa_score is not None:
            new_best = sa_out
            print(f"  [pick] SA used (split failed) → {sa_out}")
        elif split_score is not None:
            new_best = split_out
            print(f"  [pick] split used (SA failed) → {split_out}")
        else:
            print(f"  [pick] both failed, keeping previous best {best_weights}")
        best_weights = new_best

        iter_elapsed = time.time() - iter_start
        print(f"  iter {iteration} complete in {iter_elapsed:.0f}s")
        print()

        # Append to log
        with open(log_path, 'a') as f:
            f.write(f"iter {iteration}: sa={sa_str}  split={split_str}  "
                    f"best={Path(best_weights).name if best_weights else '-'}  "
                    f"{iter_elapsed:.0f}s\n")
            f.flush()

    total_elapsed = time.time() - total_start
    print(f"{'='*66}")
    print(f"  ALL ITERATIONS COMPLETE in {total_elapsed:.0f}s ({total_elapsed/60:.0f} min)")
    print(f"  Final best weights: {best_weights}")
    print(f"{'='*66}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
