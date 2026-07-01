"""Re-label aux targets in an MCV2 training file to use PER-SPECIES UB targets.

The new aux target for each sample is the MAXIMUM possible bear/salmon score
INDEPENDENTLY (not constrained by other species). This decouples the aux signal
from the optimal-allocation's choice of which species to invest in.

For each sample:
  1. Decode features → current per-species cell counts (b, e, s, h, f) + turn
  2. Compute max bear pairs achievable: min(4, (current_b + r) / 2)
  3. New aux_bear = max bear pairs (0-4)
  4. Compute max salmon cells achievable: min(9, current_s + r)
  5. Convert to longest chain length proxy: min(7, max_salmon_cells)
     (Card A maxes salmon contribution at chain-7=26 pts; longer chains add via partition)
  6. New aux_salmon = max longest salmon chain length (0-7)
  7. Replace the existing aux_bear, aux_salmon fields

Rationale (vs the original v4 aux which used AI's actual achievement):
- Original aux: "predict what the AI WILL do" → network learns its own (poor) policy
- New aux: "predict what the AI COULD do" → network learns potential, regardless of policy

Output: a new MCV2 file with same value targets but UB-derived per-species aux targets.

Usage:
    python3 relabel_aux_to_ub.py --input training_merged_iter1.bin \\
        --output training_merged_iter1_ub.bin
"""

import argparse
import struct
import sys
from typing import Tuple


# ─── Greedy UB tables (mirror crates/cascadia-ai/src/greedy_ub.rs) ───

SALMON_BEST = [
    0, 2, 4, 7, 11, 15, 20, 26, 28, 30, 33, 37, 41, 46, 52, 54, 56, 59, 63, 67, 72,
]
ELK_BEST = [
    0, 2, 5, 9, 13, 15, 18, 22, 26, 28, 31, 35, 39, 41, 44, 48, 52, 54, 57, 61, 65,
]
BEAR_SCORE = [0, 4, 11, 19, 27]
HAWK_SCORE = [0, 2, 5, 8, 11, 14, 18, 22, 28]


def bear_score(n_cells: int) -> int:
    pairs = min(n_cells // 2, 4)
    return BEAR_SCORE[pairs]


def salmon_score(n_cells: int) -> int:
    return SALMON_BEST[min(n_cells, 20)]


def elk_score(n_cells: int) -> int:
    return ELK_BEST[min(n_cells, 20)]


def hawk_score(n_cells: int) -> int:
    return HAWK_SCORE[min(n_cells, 8)]


def fox_score(n_cells: int, num_other_species: int) -> int:
    if n_cells == 0:
        return 0
    per_fox = min(num_other_species + 1, 5)
    return n_cells * per_fox


def greedy_ub_argmax_from_state(
    cur_b: int, cur_e: int, cur_s: int, cur_h: int, cur_f: int, r: int
) -> Tuple[int, int, int, int, int]:
    """Returns (b, e, s, h, f) cell counts that maximize the greedy UB."""
    best = 0
    best_alloc = (cur_b, cur_e, cur_s, cur_h, cur_f)
    for db in range(min(r, 8) + 1):
        new_b = cur_b + db
        for ds in range(min(r - db, 9) + 1):
            new_s = cur_s + ds
            for dh in range(min(r - db - ds, 8) + 1):
                new_h = cur_h + dh
                r_ef = r - db - ds - dh
                for de in range(r_ef + 1):
                    df = r_ef - de
                    new_e = cur_e + de
                    new_f = cur_f + df
                    other = (
                        (1 if new_b > 0 else 0)
                        + (1 if new_e > 0 else 0)
                        + (1 if new_s > 0 else 0)
                        + (1 if new_h > 0 else 0)
                    )
                    score = (
                        bear_score(new_b)
                        + elk_score(new_e)
                        + salmon_score(new_s)
                        + hawk_score(new_h)
                        + fox_score(new_f, other)
                    )
                    if score > best:
                        best = score
                        best_alloc = (new_b, new_e, new_s, new_h, new_f)
    return best_alloc


# ─── Feature decoder (mirrors nnue.rs) ───

FEATURES_PER_CELL = 11
NUM_CELLS = 441
CELL_FEATURES = NUM_CELLS * FEATURES_PER_CELL  # 4851
TURN_FEATURE_OFFSET = CELL_FEATURES  # turn features start at 4851


def count_species_and_turn(feature_indices) -> Tuple[int, int, int, int, int, int]:
    """Decode feature indices into (b, e, s, h, f, turn).
    Wildlife indices in per-cell features (offset 0-4):
      0=Bear, 1=Elk, 2=Salmon, 3=Hawk, 4=Fox
    Turn feature: range [4851..4872] (21 turns), single one-hot at 4851+turn.
    """
    counts = [0, 0, 0, 0, 0]
    turn = 0
    for fi in feature_indices:
        if fi < CELL_FEATURES:
            offset = fi % FEATURES_PER_CELL
            if offset < 5:
                counts[offset] += 1
        elif TURN_FEATURE_OFFSET <= fi < TURN_FEATURE_OFFSET + 21:
            turn = fi - TURN_FEATURE_OFFSET
    return counts[0], counts[1], counts[2], counts[3], counts[4], turn


# ─── MCV2 read/write ───

MCV2_MAGIC = b"MCV2"


def relabel_file(in_path: str, out_path: str, max_samples: int = None,
                 verbose: bool = False):
    with open(in_path, "rb") as f:
        data = f.read()
    if data[:4] != MCV2_MAGIC:
        raise ValueError(f"Expected MCV2, got {data[:4]}")

    out = bytearray(MCV2_MAGIC)
    pos = 4
    n = len(data)
    n_samples = 0
    n_changed = 0
    sum_old_b = 0
    sum_new_b = 0
    sum_old_s = 0
    sum_new_s = 0

    # Cache UB lookups by (b, e, s, h, f, turn) — these are bounded
    cache = {}

    while pos + 2 <= n:
        if max_samples is not None and n_samples >= max_samples:
            break
        nf_bytes = data[pos : pos + 2]
        nf = struct.unpack("<H", nf_bytes)[0]
        if nf > 1024 or pos + 2 + nf * 2 + 4 + 8 > n:
            break
        feat_bytes = data[pos + 2 : pos + 2 + nf * 2]
        feats = struct.unpack(f"<{nf}H", feat_bytes)
        target_bytes = data[pos + 2 + nf * 2 : pos + 2 + nf * 2 + 4]
        old_aux_bear_bytes = data[pos + 2 + nf * 2 + 4 : pos + 2 + nf * 2 + 8]
        old_aux_salmon_bytes = data[pos + 2 + nf * 2 + 8 : pos + 2 + nf * 2 + 12]

        old_aux_bear = struct.unpack("<f", old_aux_bear_bytes)[0]
        old_aux_salmon = struct.unpack("<f", old_aux_salmon_bytes)[0]

        b, e, s, h, f, turn = count_species_and_turn(feats)
        # In Cascadia 4-player, AI plays 20 moves (turns 1..20). Turn feature is 0-indexed.
        # moves_remaining = 20 - turn (after turn 0, 19 moves remain... etc)
        # But turn here is the AFTERSTATE turn — the move just played.
        moves_remaining = max(0, 20 - (turn + 1))

        # Per-species INDEPENDENT max:
        #   - max bear pairs: spend ALL r remaining moves on bears, max 4 pairs
        #   - max salmon cells: spend ALL r remaining moves on salmon, max 9 (per game cap)
        #   - longest chain: capped at min(max_salmon_cells, 7) since chain-7 is 26pts
        max_bear_pairs = min(4, (b + moves_remaining) // 2)
        max_salmon_cells = min(9, s + moves_remaining)
        max_salmon_chain = min(7, max_salmon_cells)  # chain length proxy

        new_aux_bear = float(max_bear_pairs)
        new_aux_salmon = float(max_salmon_chain)

        if abs(new_aux_bear - old_aux_bear) > 0.01 or abs(new_aux_salmon - old_aux_salmon) > 0.01:
            n_changed += 1
        sum_old_b += old_aux_bear
        sum_new_b += new_aux_bear
        sum_old_s += old_aux_salmon
        sum_new_s += new_aux_salmon

        # Write the sample with new aux
        out += nf_bytes
        out += feat_bytes
        out += target_bytes
        out += struct.pack("<f", new_aux_bear)
        out += struct.pack("<f", new_aux_salmon)

        pos += 2 + nf * 2 + 4 + 8
        n_samples += 1
        if verbose and n_samples % 100000 == 0:
            print(f"  Processed {n_samples} samples...")

    with open(out_path, "wb") as f:
        f.write(bytes(out))

    print(f"Processed {n_samples} samples")
    print(f"Cache size: {len(cache)} unique (b,e,s,h,f,r) tuples")
    print(f"Changed: {n_changed} ({100*n_changed/max(1,n_samples):.1f}%)")
    print(f"Mean aux_bear:   old={sum_old_b/max(1,n_samples):.2f}  new={sum_new_b/max(1,n_samples):.2f}")
    print(f"Mean aux_salmon: old={sum_old_s/max(1,n_samples):.2f}  new={sum_new_s/max(1,n_samples):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    relabel_file(args.input, args.output, args.max_samples, args.verbose)


if __name__ == "__main__":
    main()
