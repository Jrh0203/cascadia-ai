"""Train an NNUE-style feedforward net using TRANSFORMER FEATURES.

Tests whether a sparse-feature MLP (no attention) can match the transformer
when given the SAME content features. This isolates "attention vs MLP"
controlling for feature representation.

Feature design (axis-decomposed position, per user's suggestion):
  Per-tile content (43 channels each):
    - 6 edges × 5 terrains = 30 (rotation-aware)
    - 6 wildlife states (none + 5 types)
    - 5 allowed_mask bits
    - 2 flags (keystone, has_wildlife)

  Axis × content cross (3 axes, 21 buckets each):
    For each placed tile at axial (q, r, s = -q-r), fire
        q_bucket * 43 + c   for each content c set
        r_bucket * 43 + c
        s_bucket * 43 + c
    Total: 21 * 43 * 3 = 2709 sparse binary features

  Globals (one-hot, same as transformer v2):
    Total: 785 features

  Grand total: 3494 features

Architecture: 3494 -> 512 -> 64 -> 1 (matches existing NNUE shape)

Loads TIL2 binary (same format as the transformer trainer).

Usage:
    python3 train_nnue_t.py --samples rich_tokens_50k.bin --epochs 30 --lr 1e-3
"""

import argparse
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─── Constants ───

NUM_TERRAIN_TYPES = 5
NUM_WILDLIFE_STATES = 6  # 0=none, 1-5=types
NEIGHBOR_WL_STATES = 7   # 0=no tile, 1-5=wildlife, 6=tile-no-wildlife
NEIGHBOR_TR_STATES = 6   # 0=no tile, 1-5=terrain
TERRAIN_EDGES = 6
TILE_BYTES = 23

GRID_DIM = 21       # q, r, s each fit in 21 buckets shifted by 10
HALF_GRID = 10

# Base per-tile content channels (43)
BASE_CONTENT_CHANNELS = (
    TERRAIN_EDGES * NUM_TERRAIN_TYPES   # 30: edge × terrain
    + NUM_WILDLIFE_STATES               # 6: wildlife
    + 5                                  # allowed bits
    + 2                                  # flags
)
assert BASE_CONTENT_CHANNELS == 43

# Adjacency channels per tile (78 = 6 dirs × 13 neighbor states)
ADJ_CONTENT_CHANNELS = (
    TERRAIN_EDGES * NEIGHBOR_WL_STATES   # 42: 6 dirs × 7 wildlife states
    + TERRAIN_EDGES * NEIGHBOR_TR_STATES  # 36: 6 dirs × 6 terrain states
)
assert ADJ_CONTENT_CHANNELS == 78


def feature_dims(include_adjacency):
    """Return (content_channels, axis_features, total_axis_features, num_features,
    q_offset, r_offset, s_offset, global_offset) for a given variant."""
    cc = BASE_CONTENT_CHANNELS + (ADJ_CONTENT_CHANNELS if include_adjacency else 0)
    axf = GRID_DIM * cc
    total_ax = 3 * axf
    nf = total_ax + GLOBAL_FEATURES
    return cc, axf, total_ax, nf, 0, axf, 2 * axf, total_ax

# Global one-hot bins
G_TURN_BINS = 21
G_TOKEN_BINS = 9
G_WL_COUNT_BINS = 25
G_HAB_BINS = 14
G_BAG_BINS = 25
G_TBAG_BINS = 30

GLOBAL_FEATURES = (
    G_TURN_BINS                 # 21
    + G_TOKEN_BINS              # 9
    + 5 * G_WL_COUNT_BINS       # 125
    + 5 * G_HAB_BINS            # 70
    + 5 * G_BAG_BINS            # 125
    + 5 * G_HAB_BINS            # 70
    + 4 * NUM_TERRAIN_TYPES     # 20
    + 4 * NUM_TERRAIN_TYPES     # 20
    + 4 * NUM_WILDLIFE_STATES   # 24
    + 5 * G_TBAG_BINS           # 150
    + 5 * G_TBAG_BINS           # 150
    + 1                         # overflow
)
assert GLOBAL_FEATURES == 785

# Backward-compat constants for the no-adjacency variant (used by NnueT default)
_DEFAULT_DIMS = feature_dims(include_adjacency=False)
CONTENT_CHANNELS = _DEFAULT_DIMS[0]
AXIS_FEATURES = _DEFAULT_DIMS[1]
TOTAL_AXIS_FEATURES = _DEFAULT_DIMS[2]
NUM_FEATURES = _DEFAULT_DIMS[3]
Q_AXIS_OFFSET = _DEFAULT_DIMS[4]
R_AXIS_OFFSET = _DEFAULT_DIMS[5]
S_AXIS_OFFSET = _DEFAULT_DIMS[6]
GLOBAL_OFFSET = _DEFAULT_DIMS[7]

MAX_TILES = 23


# ─── Data Loading ───

def load_tile_samples(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'TIL2':
        raise ValueError(f"Bad magic: {data[:4]!r}, expected TIL2")
    pos = 4
    tiles_list = []
    globals_list = []
    targets = []
    GLOBAL_BYTES = 45
    while pos < len(data):
        if pos + 1 > len(data):
            break
        num_tiles = data[pos]
        pos += 1
        if num_tiles == 0 or num_tiles > MAX_TILES:
            break
        tbs = num_tiles * TILE_BYTES
        if pos + tbs + GLOBAL_BYTES + 4 > len(data):
            break
        tile_data = np.frombuffer(data, dtype=np.uint8, count=tbs, offset=pos).reshape(num_tiles, TILE_BYTES)
        pos += tbs
        raw_global = np.frombuffer(data, dtype=np.uint8, count=GLOBAL_BYTES, offset=pos).copy()
        pos += GLOBAL_BYTES
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4
        tiles_list.append(tile_data.copy())
        globals_list.append(raw_global)
        targets.append(target)
    targets = np.array(targets, dtype=np.float32)
    print(f"Loaded {len(tiles_list)} samples from {path}")
    if tiles_list:
        print(f"  Tiles per sample: {min(len(t) for t in tiles_list)}-{max(len(t) for t in tiles_list)}")
        print(f"  Target range: [{targets.min():.1f}, {targets.max():.1f}], mean={targets.mean():.1f}")
    return tiles_list, globals_list, targets


def extract_tile_content_indices(tile_row, include_adjacency=False):
    """Given one tile's 23 raw bytes, return list of content channel indices.

    Without adjacency: indices in 0..43.
    With adjacency: indices in 0..121 (43 base + 78 adjacency).
    """
    indices = []
    # Edge terrain one-hots: 6 edges × 5 = 30
    for e in range(TERRAIN_EDGES):
        t = int(tile_row[e])
        if 0 <= t < NUM_TERRAIN_TYPES:
            indices.append(e * NUM_TERRAIN_TYPES + t)
    base = TERRAIN_EDGES * NUM_TERRAIN_TYPES  # 30
    # Wildlife (6 states one-hot)
    w = int(tile_row[6])
    if 0 <= w < NUM_WILDLIFE_STATES:
        indices.append(base + w)
    base += NUM_WILDLIFE_STATES  # 36
    # Allowed mask 5 bits
    am = int(tile_row[7])
    for b in range(5):
        if (am >> b) & 1:
            indices.append(base + b)
    base += 5  # 41
    # Flags (keystone, has_wildlife)
    f = int(tile_row[8])
    if f & 1:
        indices.append(base + 0)
    if (f >> 1) & 1:
        indices.append(base + 1)
    base += 2  # 43

    if include_adjacency:
        # Neighbor wildlife states: 6 dirs × 7 = 42 channels
        for d in range(TERRAIN_EDGES):
            nw = int(tile_row[11 + d])
            if 0 <= nw < NEIGHBOR_WL_STATES:
                indices.append(base + d * NEIGHBOR_WL_STATES + nw)
        base += TERRAIN_EDGES * NEIGHBOR_WL_STATES  # 85
        # Neighbor terrain states: 6 dirs × 6 = 36 channels
        for d in range(TERRAIN_EDGES):
            nt = int(tile_row[17 + d])
            if 0 <= nt < NEIGHBOR_TR_STATES:
                indices.append(base + d * NEIGHBOR_TR_STATES + nt)
        base += TERRAIN_EDGES * NEIGHBOR_TR_STATES  # 121
    return indices


def extract_features(tile_data, raw_global, include_adjacency=False):
    """Return sparse feature indices (np.int64 array) for one sample."""
    cc, axf, total_ax, nf, q_off, r_off, s_off, g_off = feature_dims(include_adjacency)
    feats = set()  # dedupe (multiple tiles may fire same axis-content pair)

    for tile_row in tile_data:
        q = int(np.int8(tile_row[9]))
        r = int(np.int8(tile_row[10]))
        s = -q - r
        qb = max(0, min(GRID_DIM - 1, q + HALF_GRID))
        rb = max(0, min(GRID_DIM - 1, r + HALF_GRID))
        sb = max(0, min(GRID_DIM - 1, s + HALF_GRID))

        content_idxs = extract_tile_content_indices(tile_row, include_adjacency)
        for c in content_idxs:
            feats.add(q_off + qb * cc + c)
            feats.add(r_off + rb * cc + c)
            feats.add(s_off + sb * cc + c)

    # Globals (one-hot)
    g = raw_global
    offset = g_off

    feats.add(offset + min(int(g[0]), G_TURN_BINS - 1)); offset += G_TURN_BINS
    feats.add(offset + min(int(g[1]), G_TOKEN_BINS - 1)); offset += G_TOKEN_BINS
    for i in range(5):
        feats.add(offset + min(int(g[2 + i]), G_WL_COUNT_BINS - 1)); offset += G_WL_COUNT_BINS
    for i in range(5):
        feats.add(offset + min(int(g[7 + i]), G_HAB_BINS - 1)); offset += G_HAB_BINS
    for i in range(5):
        feats.add(offset + min(int(g[12 + i]), G_BAG_BINS - 1)); offset += G_BAG_BINS
    for i in range(5):
        feats.add(offset + min(int(g[17 + i]), G_HAB_BINS - 1)); offset += G_HAB_BINS
    for i in range(4):
        v = int(g[22 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            feats.add(offset + v)
        offset += NUM_TERRAIN_TYPES
    for i in range(4):
        v = int(g[26 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            feats.add(offset + v)
        offset += NUM_TERRAIN_TYPES
    for i in range(4):
        v = int(g[30 + i])
        if 0 <= v < NUM_WILDLIFE_STATES:
            feats.add(offset + v)
        offset += NUM_WILDLIFE_STATES
    for i in range(5):
        feats.add(offset + min(int(g[34 + i]), G_TBAG_BINS - 1)); offset += G_TBAG_BINS
    for i in range(5):
        feats.add(offset + min(int(g[39 + i]), G_TBAG_BINS - 1)); offset += G_TBAG_BINS
    if g[44]:
        feats.add(offset)
    offset += 1
    assert offset == nf, f"global offset {offset} != num_features {nf}"

    return np.array(sorted(feats), dtype=np.int64)


# ─── Dataset ───

class PackedFeatures:
    """Memory-efficient packed storage of all sparse features.

    All feature indices live in one int32 array; offsets[i] gives the start
    of sample i's features. Total memory: ~50 features avg × N × 4 bytes
    + 8 bytes per offset. For 1M samples ≈ 200 MB total (vs ~5 GB for a
    Python list of small numpy arrays).
    """

    def __init__(self, tiles_list, globals_list, include_adjacency=False, verbose=True):
        n = len(tiles_list)
        flat = []
        offsets = np.zeros(n + 1, dtype=np.int64)
        t0 = time.time()
        for i, (t, g) in enumerate(zip(tiles_list, globals_list)):
            feats = extract_features(t, g, include_adjacency=include_adjacency)
            flat.append(feats.astype(np.int32))
            offsets[i + 1] = offsets[i] + len(feats)
            if verbose and (i + 1) % 100000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  Extracted {i+1}/{n} ({rate:.0f}/s)")
        self.feats = np.concatenate(flat).astype(np.int32)
        self.offsets = offsets
        if verbose:
            mb = (self.feats.nbytes + self.offsets.nbytes) / 1e6
            print(f"  Packed: {self.feats.nbytes/1e6:.1f} MB feats + "
                  f"{self.offsets.nbytes/1e6:.1f} MB offsets = {mb:.1f} MB "
                  f"(avg {self.feats.size/n:.1f} feats/sample)")

    def batch_indices(self, sample_idxs):
        """Build (feat_idx, samp_idx, batch_size) for a list of sample indices.

        Uses pre-allocated arrays + numpy operations only — no Python loops.
        """
        starts = self.offsets[sample_idxs]
        ends = self.offsets[sample_idxs + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        # Build feat_idx: concat self.feats[starts[k]:ends[k]] for each k
        feat_idx = np.empty(total, dtype=np.int32)
        pos = 0
        for k in range(len(sample_idxs)):
            n = lengths[k]
            feat_idx[pos:pos + n] = self.feats[starts[k]:starts[k] + n]
            pos += n
        # samp_idx: per-sample repeat
        samp_idx = np.repeat(np.arange(len(sample_idxs), dtype=np.int64), lengths)
        return feat_idx, samp_idx


class PackedDataset:
    """Wrapper around PackedFeatures + targets, optionally restricted to a
    subset of sample indices (without copying the underlying packed arrays)."""

    def __init__(self, packed, targets, sample_idx_map=None):
        self.packed = packed
        self.targets = targets.astype(np.float32)
        # If set, batch_idxs in [0, len(self)) are mapped through this array
        # to indices into packed.offsets. None → identity.
        self.sample_idx_map = sample_idx_map

    def __len__(self):
        return len(self.targets)


def _slice_packed(packed, sample_indices, sub_targets):
    """Build a PackedDataset view over `sample_indices` of `packed`."""
    return PackedDataset(packed, sub_targets,
                         sample_idx_map=sample_indices.astype(np.int64))


def packed_iterator(dataset, batch_size, shuffle, rng):
    """Yield batches as (feat_idx_tensor, samp_idx_tensor, targets_tensor, batch_size)."""
    n = len(dataset)
    if shuffle:
        order = rng.permutation(n)
    else:
        order = np.arange(n)
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        # Map to underlying packed indices if this is a subset
        if dataset.sample_idx_map is not None:
            packed_idxs = dataset.sample_idx_map[idxs]
        else:
            packed_idxs = idxs
        feat_idx, samp_idx = dataset.packed.batch_indices(packed_idxs)
        targets = dataset.targets[idxs]
        yield (
            torch.from_numpy(feat_idx).to(torch.int64),
            torch.from_numpy(samp_idx),
            torch.from_numpy(targets),
            len(idxs),
        )


# ─── Model ───

class NnueT(nn.Module):
    """Sparse-feature feedforward net (NNUE-style)."""

    def __init__(self, num_features=NUM_FEATURES, h1=512, h2=64, include_adjacency=False):
        super().__init__()
        self.num_features = num_features
        self.h1 = h1
        self.h2 = h2
        self.include_adjacency = include_adjacency

        # First layer: sparse embedding sum (num_features × h1)
        # Use nn.Embedding for efficient sparse lookup; sum across active features.
        self.embed = nn.Embedding(num_features, h1)
        self.bias1 = nn.Parameter(torch.zeros(h1))

        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward_sparse(self, feature_indices, sample_indices, batch_size):
        """Forward pass on sparse features.

        Args:
            feature_indices: [N_active] LongTensor
            sample_indices:  [N_active] LongTensor (which sample each feature belongs to)
            batch_size:      int

        Returns:
            [B] FloatTensor
        """
        # Lookup embeddings: [N_active, h1]
        embeds = self.embed(feature_indices)
        # Sum-aggregate per sample using scatter_add
        h1 = torch.zeros(batch_size, self.h1, device=embeds.device, dtype=embeds.dtype)
        h1.index_add_(0, sample_indices, embeds)
        h1 = h1 + self.bias1
        h1 = torch.relu(h1)
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2).squeeze(-1)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ─── Training ───

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, train_ds, val_ds, optimizer, scheduler, device,
          epochs, batch_size, checkpoint_path, save_every=False, rng=None,
          empty_cache_every=50):
    if rng is None:
        rng = np.random.default_rng(0)
    best_val = float('inf')
    criterion = nn.MSELoss()
    is_mps = (device.type == 'mps')
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_samples = 0
        batch_count = 0
        for feat_idx, samp_idx, targets, bs in packed_iterator(
                train_ds, batch_size, shuffle=True, rng=rng):
            feat_idx = feat_idx.to(device, non_blocking=True)
            samp_idx = samp_idx.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model.forward_sparse(feat_idx, samp_idx, bs)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * bs
            n_samples += bs
            batch_count += 1
            # Periodically release MPS cache to avoid leak-induced OOM
            if is_mps and batch_count % empty_cache_every == 0:
                torch.mps.empty_cache()
        train_loss = loss_sum / max(1, n_samples)
        train_rmse = math.sqrt(train_loss)

        val_rmse = 0.0
        if val_ds is not None:
            model.eval()
            vsum = 0.0
            vn = 0
            with torch.no_grad():
                for feat_idx, samp_idx, targets, bs in packed_iterator(
                        val_ds, batch_size, shuffle=False, rng=rng):
                    feat_idx = feat_idx.to(device, non_blocking=True)
                    samp_idx = samp_idx.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    preds = model.forward_sparse(feat_idx, samp_idx, bs)
                    loss = criterion(preds, targets)
                    vsum += loss.item() * bs
                    vn += bs
            val_loss = vsum / max(1, vn)
            val_rmse = math.sqrt(val_loss)
        else:
            val_loss = train_loss

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{epochs}  train_rmse={train_rmse:.3f}  "
              f"val_rmse={val_rmse:.3f}  lr={lr:.2e}  {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, checkpoint_path)
        if save_every:
            base, ext = os.path.splitext(checkpoint_path)
            save_checkpoint(model, f"{base}_epoch{epoch}{ext}")
    return best_val


def save_checkpoint(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_features': model.num_features,
            'h1': model.h1,
            'h2': model.h2,
            'include_adjacency': model.include_adjacency,
        },
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = NnueT(num_features=cfg['num_features'], h1=cfg['h1'], h2=cfg['h2'],
                  include_adjacency=cfg.get('include_adjacency', False))
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--h1', type=int, default=512)
    p.add_argument('--h2', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--out', default='nnue_t.pt')
    p.add_argument('--init-weights', default=None)
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--save-every-epoch', action='store_true')
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default=None,
                   help='Override device selection (default: auto)')
    p.add_argument('--include-adjacency', action='store_true',
                   help='Add 78 channels of per-cell adjacency to content '
                        '(NNUE-style interaction features)')
    args = p.parse_args()

    import sys
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device(args.device) if args.device else get_device()
    cc, axf, total_ax, nf, _, _, _, _ = feature_dims(args.include_adjacency)
    print(f"=== NNUE-T{'A' if args.include_adjacency else ''} "
          f"(transformer features, axis-decomposed"
          f"{', +adjacency' if args.include_adjacency else ''}) ===")
    print(f"  Device: {device}")
    print(f"  Content channels: {cc}, axis features: {total_ax}, globals: {GLOBAL_FEATURES}")
    print(f"  Total features: {nf}")
    print(f"  Architecture: {nf} → {args.h1} → {args.h2} → 1")
    print(f"  Training: epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print()

    print("Loading samples...")
    tiles_list, globals_list, targets = load_tile_samples(args.samples)
    if not tiles_list:
        print("ERROR: no samples")
        return

    print("Pre-extracting sparse features (packed)...")
    n = len(tiles_list)
    indices = np.random.default_rng(42).permutation(n)
    n_val = max(1, int(n * args.val_split)) if n > 10 else 0

    # Pack ALL features into one contiguous structure for memory efficiency
    packed_all = PackedFeatures(tiles_list, globals_list,
                                include_adjacency=args.include_adjacency, verbose=True)
    # Free the raw tile lists
    del tiles_list, globals_list

    if n_val > 0:
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        # Both datasets share the SAME packed features (just different sample indices).
        # We use a custom view to avoid copying the packed array.
        # Trick: build separate PackedFeatures via re-indexing the offsets.
        train_ds = _slice_packed(packed_all, train_idx, targets[train_idx])
        val_ds = _slice_packed(packed_all, val_idx, targets[val_idx])
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    else:
        train_ds = PackedDataset(packed_all, targets)
        val_ds = None
        print(f"  Train: {len(train_ds)}")

    if args.init_weights and os.path.exists(args.init_weights):
        print(f"  Loading {args.init_weights}")
        model = load_checkpoint(args.init_weights, device)
        if model.include_adjacency != args.include_adjacency:
            raise ValueError(f"Checkpoint include_adjacency={model.include_adjacency} "
                             f"!= --include-adjacency={args.include_adjacency}")
    else:
        model = NnueT(num_features=nf, h1=args.h1, h2=args.h2,
                      include_adjacency=args.include_adjacency)

    model = model.to(device)
    nparams = model.count_parameters()
    print(f"  Parameters: {nparams:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * batches_per_epoch
    warmup = int(0.05 * total_steps)
    scheduler = build_scheduler(optimizer, warmup, total_steps)
    print(f"  Steps: {total_steps}, warmup: {warmup}, batches/epoch: {batches_per_epoch}")
    print()

    rng = np.random.default_rng(0)
    t0 = time.time()
    best = train(model, train_ds, val_ds, optimizer, scheduler,
                 device, args.epochs, args.batch_size, args.out,
                 save_every=args.save_every_epoch, rng=rng)
    print()
    print(f"  Trained in {time.time()-t0:.1f}s, best RMSE: {math.sqrt(best):.3f}")
    print(f"  Saved to {args.out}")


if __name__ == '__main__':
    main()
