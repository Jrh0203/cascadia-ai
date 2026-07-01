"""Evaluate a GNN checkpoint on held-out tile-token data.

Loads the current checkpoint (saved after the latest best val epoch), runs
inference on a test set, and reports:
  - Overall RMSE, MAE, R²
  - Accuracy broken down by game turn (how well does it predict at each phase?)
  - Correlation between predictions and actuals

This gives a fair measure of model quality without needing gameplay integration.
"""

import argparse
import math
import time

import numpy as np
import torch

from train_cnn import (
    HexGNN,
    NODE_FEATURES,
    TileGraphDataset,
    collate_graphs,
    load_tile_samples,
)
from torch.utils.data import DataLoader


def evaluate(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = HexGNN(
        node_in=NODE_FEATURES,
        hidden=args.hidden,
        n_layers=args.n_layers,
        global_dim=53,
    ).to(device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.checkpoint} ({n_params:,} params)")

    # Load test samples
    print(f"Loading {args.samples}...")
    t0 = time.time()
    raw = load_tile_samples(args.samples)
    if args.limit and len(raw) > args.limit:
        raw = raw[:args.limit]
    print(f"  {len(raw)} samples in {time.time()-t0:.1f}s")

    dataset = TileGraphDataset(raw)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_graphs)

    # Collect predictions, targets, and turn numbers (derived from num_tiles)
    all_preds = []
    all_targets = []
    all_num_tiles = []  # proxy for "turns played" — 3 = start, 23 = near end

    t0 = time.time()
    with torch.no_grad():
        for nf, ei, bi, gf, targets, nc in loader:
            nf = nf.to(device); ei = ei.to(device); bi = bi.to(device)
            gf = gf.to(device); targets = targets.to(device)
            preds = model(nf, ei, bi, gf, nc)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_num_tiles.extend(nc)
    inf_time = time.time() - t0

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    num_tiles = np.array(all_num_tiles)
    n = len(preds)

    print(f"\nInference: {n} samples in {inf_time:.1f}s ({n/inf_time:.0f} samples/sec)")

    # Overall metrics
    residuals = preds - targets
    mse = (residuals ** 2).mean()
    rmse = math.sqrt(mse)
    mae = np.abs(residuals).mean()

    # R² = 1 - SS_res / SS_tot
    ss_res = (residuals ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Pearson correlation
    pred_centered = preds - preds.mean()
    target_centered = targets - targets.mean()
    corr = (pred_centered * target_centered).sum() / (
        np.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum()) + 1e-12)

    print(f"\n═══ Overall Prediction Quality ═══")
    print(f"  RMSE:        {rmse:.3f}")
    print(f"  MAE:         {mae:.3f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Target mean: {targets.mean():.2f} (std {targets.std():.2f})")
    print(f"  Pred mean:   {preds.mean():.2f} (std {preds.std():.2f})")

    # Breakdown by game phase (num_tiles buckets)
    print(f"\n═══ Error by Game Phase ═══")
    print(f"  {'Phase':<20} {'N':>7} {'Actual':>9} {'Pred':>9} {'RMSE':>6} {'MAE':>6}")
    phases = [
        ("Early (3-7 tiles)",    (num_tiles >= 3)  & (num_tiles <= 7)),
        ("Mid-early (8-12)",     (num_tiles >= 8)  & (num_tiles <= 12)),
        ("Mid-late (13-17)",     (num_tiles >= 13) & (num_tiles <= 17)),
        ("Late (18-22)",         (num_tiles >= 18) & (num_tiles <= 22)),
        ("End (23)",             num_tiles == 23),
    ]
    for label, mask in phases:
        if mask.sum() == 0:
            continue
        t_m = targets[mask]; p_m = preds[mask]
        p_rmse = math.sqrt(((p_m - t_m) ** 2).mean())
        p_mae = np.abs(p_m - t_m).mean()
        print(f"  {label:<20} {mask.sum():>7d} "
              f"{t_m.mean():>9.2f} {p_m.mean():>9.2f} "
              f"{p_rmse:>6.2f} {p_mae:>6.2f}")

    # Error distribution
    print(f"\n═══ Error Distribution (predicted - actual) ═══")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for pct in percentiles:
        v = np.percentile(residuals, pct)
        print(f"  P{pct:>2}: {v:>+7.2f}")

    # Baselines for context
    print(f"\n═══ Baseline Comparisons ═══")
    mean_pred = targets.mean()
    mean_rmse = math.sqrt(((mean_pred - targets) ** 2).mean())
    print(f"  'Always predict mean' RMSE: {mean_rmse:.3f}  (our RMSE: {rmse:.3f}, "
          f"{'better' if rmse < mean_rmse else 'worse'} by {abs(mean_rmse - rmse):.2f})")

    # If target is large-ish, a linear baseline from num_tiles
    # (remaining = f(tiles_placed)) could be competitive for simple data
    x = num_tiles.astype(np.float32)
    # OLS: slope, intercept
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    linear_preds = coeffs[0] * x + coeffs[1]
    linear_rmse = math.sqrt(((linear_preds - targets) ** 2).mean())
    print(f"  Linear-in-tiles RMSE:       {linear_rmse:.3f}  (our RMSE: {rmse:.3f}, "
          f"{'better' if rmse < linear_rmse else 'worse'} by {abs(linear_rmse - rmse):.2f})")

    return rmse


def main():
    p = argparse.ArgumentParser(description="Evaluate a GNN checkpoint on tile-token data")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--samples", required=True, help="Path to TILE-format samples")
    p.add_argument("--limit", type=int, default=0, help="Use only first N samples (0=all)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
