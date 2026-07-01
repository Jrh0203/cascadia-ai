"""GNN training for Cascadia board state evaluation (remaining score prediction).

Uses a Graph Neural Network with hex-topology message passing to predict how many
points remain to be scored from a given board state. Pure PyTorch implementation
with manual scatter operations -- no PyTorch Geometric or DGL dependency.

The hex grid is a natural graph: each placed tile has up to 6 neighbors. Message
passing along these edges captures adjacency patterns (bear pairs, elk lines,
salmon runs, hawk isolation) without the distortion of mapping hex -> square grid.

Data format: "TILE" binary format
    u8 num_tiles
    Per tile (11 bytes each):
        u8[6] terrain_triangles (terrain type per edge sector, 0-4)
        u8 wildlife_type (0-4 = placed wildlife, 255 = none)
        u8 allowed_mask (5 bits, which wildlife can be placed)
        u8 flags (bit 0 = keystone, bit 1 = has_wildlife)
        i8 q, i8 r (hex coordinates)
    u8 num_global
    f32[num_global] global_features
    f32 target (remaining score to predict)

Usage:
    # Train from scratch
    python3 train_cnn.py --samples tile_tokens.bin --epochs 30 --lr 1e-3

    # Resume from checkpoint
    python3 train_cnn.py --samples tile_tokens.bin --epochs 30 --init-weights gnn_weights.pt

    # Dry run with synthetic data
    python3 train_cnn.py --dry-run --epochs 5
"""

import argparse
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Hex geometry ───

# Two hex tiles at (q1,r1) and (q2,r2) are adjacent iff (dq,dr) is one of these.
HEX_NEIGHBOR_OFFSETS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def build_edge_index(coords):
    """Build undirected edge_index [2, E] from a list of (q, r) hex coordinates.

    Two tiles are neighbors if their (dq, dr) matches one of the 6 hex directions.
    Returns edges in both directions (undirected graph).
    """
    coord_to_idx = {}
    for i, (q, r) in enumerate(coords):
        coord_to_idx[(q, r)] = i

    src_list = []
    dst_list = []
    for i, (q, r) in enumerate(coords):
        for dq, dr in HEX_NEIGHBOR_OFFSETS:
            nq, nr = q + dq, r + dr
            j = coord_to_idx.get((nq, nr))
            if j is not None:
                src_list.append(i)
                dst_list.append(j)
                # Reverse direction added when we process j's neighbors

    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ─── Per-cell feature encoding ───

# Per-cell channels:
#   wildlife placed:    6 one-hot (0=none, 1=Bear, 2=Elk, 3=Salmon, 4=Hawk, 5=Fox)
#                       -- matches Rust encoding directly (no off-by-one)
#   terrain triangles:  6 edges × 5 terrain types = 30 one-hot
#                       -- full rotation-aware: which terrain faces each edge
#   allowed wildlife:   5 bits (which wildlife can be placed here)
#   keystone:           1 bit (single-terrain tile flag)
#   has_tile:           1 bit (always 1 for placed tiles)
#   position:           3 floats (hex cube coords q, r, s normalized to ~[-1,1])
# Total: 46
NODE_FEATURES = 46


def encode_tile_features(terrain_triangles, wildlife_type, allowed_mask, flags, q, r):
    """Encode a single tile into a feature vector of length NODE_FEATURES.

    Args:
        terrain_triangles: list of 6 u8 terrain values (0-4 per edge, rotation-aware)
        wildlife_type: u8 (0=none, 1=Bear, 2=Elk, 3=Salmon, 4=Hawk, 5=Fox)
        allowed_mask: u8 (5 bits, bit i = Wildlife i allowed)
        flags: u8 (bit 0 = keystone, bit 1 = has_wildlife)
        q, r: i8 hex axial coordinates

    Returns:
        numpy array of shape [NODE_FEATURES] (float32)
    """
    feat = np.zeros(NODE_FEATURES, dtype=np.float32)
    offset = 0

    # Wildlife: 6 one-hot (Rust writes 0=none, 1-5=types, matches layout directly)
    if 0 <= wildlife_type <= 5:
        feat[offset + wildlife_type] = 1.0
    offset += 6

    # Terrain triangles: 6 edges × 5 terrain types = 30 one-hot
    # Preserves full rotation info (which terrain is on which edge of the hex).
    for edge in range(6):
        t = terrain_triangles[edge]
        if 0 <= t < 5:
            feat[offset + edge * 5 + t] = 1.0
    offset += 30

    # Allowed wildlife mask: 5 bits
    for b in range(5):
        if allowed_mask & (1 << b):
            feat[offset + b] = 1.0
    offset += 5

    # Keystone flag
    feat[offset] = 1.0 if (flags & 1) else 0.0
    offset += 1

    # Has tile (always 1 for placed tiles in this feature set)
    feat[offset] = 1.0
    offset += 1

    # Position: hex cube coordinates normalized to approximately [-1, 1]
    # (21x21 board centered at origin, so |q|, |r| <= 10)
    feat[offset]     = q / 10.0
    feat[offset + 1] = r / 10.0
    feat[offset + 2] = -(q + r) / 10.0  # s = -q - r (cube coord constraint)
    offset += 3

    return feat


# ─── Data loading ───

def load_tile_samples(path):
    """Load samples from the TILE binary format.

    Returns:
        List of (node_features [N,23], coords [(q,r),...], global_features [G], target float)
    """
    with open(path, 'rb') as f:
        data = f.read()

    samples = []
    pos = 0
    # Skip magic header
    if data[:4] == b'TILE':
        pos = 4
    while pos < len(data):
        if pos + 1 > len(data):
            break

        # u8 num_tiles
        num_tiles = data[pos]
        pos += 1

        if num_tiles == 0 or num_tiles > 23:
            break

        # Per tile: 11 bytes each
        tile_bytes_needed = num_tiles * 11
        if pos + tile_bytes_needed > len(data):
            break

        node_features = np.zeros((num_tiles, NODE_FEATURES), dtype=np.float32)
        coords = []

        for t in range(num_tiles):
            base = pos + t * 11
            terrain_triangles = list(data[base:base + 6])
            wildlife_type = data[base + 6]
            allowed_mask = data[base + 7]
            flags = data[base + 8]
            q = struct.unpack_from('b', data, base + 9)[0]  # i8
            r = struct.unpack_from('b', data, base + 10)[0]  # i8

            node_features[t] = encode_tile_features(
                terrain_triangles, wildlife_type, allowed_mask, flags, q, r
            )
            coords.append((q, r))

        pos += tile_bytes_needed

        # Global features: fixed layout of u8 values (45 bytes total)
        # turn(1) + tokens(1) + wl_counts(5) + hab_sizes(5) + bag_rem(5) +
        # opp_hab(5) + mkt_t1(4) + mkt_t2(4) + mkt_wl(4) + tbag_t(5) +
        # tbag_w(5) + overflow(1) = 45 bytes
        GLOBAL_BYTES = 45
        NUM_GLOBAL_FLOAT = 53
        if pos + GLOBAL_BYTES + 4 > len(data):
            break
        raw_g = data[pos:pos + GLOBAL_BYTES]
        pos += GLOBAL_BYTES
        global_features = np.zeros(NUM_GLOBAL_FLOAT, dtype=np.float32)
        gi = 0
        global_features[gi] = raw_g[0] / 20.0;  gi += 1  # turn
        global_features[gi] = raw_g[1] / 8.0;   gi += 1  # tokens
        for i in range(5): global_features[gi] = raw_g[2+i] / 10.0;  gi += 1  # wl_counts
        for i in range(5): global_features[gi] = raw_g[7+i] / 13.0;  gi += 1  # hab_sizes
        for i in range(5): global_features[gi] = raw_g[12+i] / 20.0; gi += 1  # bag_remaining
        for i in range(5): global_features[gi] = raw_g[17+i] / 13.0; gi += 1  # opp_habitat
        for i in range(4): global_features[gi] = raw_g[22+i] / 5.0;  gi += 1  # market terrain1
        for i in range(4): global_features[gi] = raw_g[26+i] / 5.0;  gi += 1  # market terrain2
        for i in range(4): global_features[gi] = raw_g[30+i] / 5.0;  gi += 1  # market wildlife
        for i in range(5): global_features[gi] = raw_g[34+i] / 29.0; gi += 1  # tbag terrain
        for i in range(5): global_features[gi] = raw_g[39+i] / 29.0; gi += 1  # tbag wildlife
        global_features[gi] = float(raw_g[44]);  gi += 1  # overflow_used

        # f32 target
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4

        samples.append((node_features, coords, global_features, target))

    return samples


def generate_synthetic_data(n_samples=100):
    """Generate synthetic samples for --dry-run testing.

    Creates plausible-looking board states with random tiles, edges, and targets.
    """
    rng = np.random.default_rng(42)
    samples = []

    for _ in range(n_samples):
        num_tiles = rng.integers(3, 24)  # 3-23 tiles

        # Place tiles on a hex grid starting from (0,0) via BFS
        placed = {(0, 0)}
        frontier = [(0, 0)]
        while len(placed) < num_tiles:
            if not frontier:
                break
            q, r = frontier.pop(rng.integers(0, len(frontier)))
            for dq, dr in HEX_NEIGHBOR_OFFSETS:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in placed and len(placed) < num_tiles:
                    placed.add((nq, nr))
                    frontier.append((nq, nr))

        coords = list(placed)
        n = len(coords)
        node_features = np.zeros((n, NODE_FEATURES), dtype=np.float32)

        for i in range(n):
            # Random terrain triangles (mostly primary, sometimes dual)
            primary = rng.integers(0, 5)
            if rng.random() < 0.3:  # 30% dual-terrain
                secondary = (primary + rng.integers(1, 5)) % 5
                triangles = [primary] * 3 + [secondary] * 3
            else:
                triangles = [primary] * 6

            # Random wildlife: Rust encoding is 0=none, 1-5=types.
            # 70% placed (1-5), 30% empty (0).
            wildlife = int(rng.integers(1, 6)) if rng.random() < 0.7 else 0

            # Random allowed mask
            allowed = int(rng.integers(1, 32))

            # Flags
            flags = 0
            if rng.random() < 0.15:
                flags |= 1  # keystone
            if wildlife > 0:
                flags |= 2  # has_wildlife

            q_i, r_i = coords[i]
            node_features[i] = encode_tile_features(
                triangles, wildlife, allowed, flags, q_i, r_i
            )

        # Global features (~53 floats)
        num_global = 53
        global_features = rng.standard_normal(num_global).astype(np.float32)
        global_features[0] = num_tiles / 20.0  # turn-like feature

        # Target: remaining score (roughly 3-70 depending on game phase)
        target = float(70.0 - (num_tiles / 23.0) * 67.0 + rng.normal(0, 5))

        samples.append((node_features, coords, global_features, target))

    return samples


class TileGraphDataset(Dataset):
    """Dataset that stores pre-parsed graph samples.

    Each sample is a tuple of (node_features, edge_index, global_features, target).
    The edge_index is built from hex coordinates at load time.
    """

    def __init__(self, samples):
        """Args:
            samples: list of (node_features [N,23], coords, global_features [G], target)
        """
        self.data = []
        self.global_dim = 0

        for node_feat, coords, global_feat, target in samples:
            edge_index = build_edge_index(coords)
            self.data.append((
                torch.from_numpy(node_feat),          # [N, 23]
                edge_index,                            # [2, E]
                torch.from_numpy(global_feat),         # [G]
                torch.tensor(target, dtype=torch.float32),
            ))
            self.global_dim = max(self.global_dim, len(global_feat))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_graphs(batch):
    """Custom collate for variable-size graphs.

    Batches graphs by concatenating node features and offsetting edge indices.
    Returns:
        node_features: [total_nodes, C]
        edge_index: [2, total_edges] (with offsets applied)
        batch_idx: [total_nodes] (which graph each node belongs to)
        global_features: [B, G_max] (zero-padded)
        targets: [B]
        node_counts: [B] (number of nodes per graph, for pooling)
    """
    node_features_list = []
    edge_index_list = []
    batch_idx_list = []
    global_features_list = []
    targets_list = []
    node_counts = []

    # Find max global feature dim in this batch
    max_global = max(g.shape[0] for _, _, g, _ in batch)

    node_offset = 0
    for i, (nf, ei, gf, t) in enumerate(batch):
        n = nf.shape[0]
        node_features_list.append(nf)

        # Offset edge indices by cumulative node count
        if ei.shape[1] > 0:
            edge_index_list.append(ei + node_offset)
        else:
            edge_index_list.append(ei)

        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        node_counts.append(n)

        # Zero-pad global features to max dim
        if gf.shape[0] < max_global:
            padded = torch.zeros(max_global, dtype=torch.float32)
            padded[:gf.shape[0]] = gf
            global_features_list.append(padded)
        else:
            global_features_list.append(gf)

        targets_list.append(t)
        node_offset += n

    node_features = torch.cat(node_features_list, dim=0)
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    global_features = torch.stack(global_features_list, dim=0)
    targets = torch.stack(targets_list, dim=0)

    return node_features, edge_index, batch_idx, global_features, targets, node_counts


# ─── GNN Model ───

def scatter_mean(src, index, dim_size, dim=0):
    """Manual scatter_mean: average source values by target index.

    Equivalent to torch_scatter.scatter_mean but implemented with scatter_add_.

    Args:
        src: [E, C] source features (one per edge)
        index: [E] target node indices
        dim_size: number of target nodes
        dim: dimension to scatter along (always 0)

    Returns:
        [dim_size, C] mean-aggregated features
    """
    C = src.shape[1] if src.dim() > 1 else 1
    if src.dim() == 1:
        src = src.unsqueeze(1)

    # Sum
    out = torch.zeros(dim_size, C, device=src.device, dtype=src.dtype)
    idx_expanded = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_expanded, src)

    # Count
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    ones = torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(1), ones)

    # Mean (avoid division by zero for isolated nodes)
    count = count.clamp(min=1.0)
    return out / count


class HexGNNLayer(nn.Module):
    """Single message-passing layer for hex graph.

    Aggregates neighbor features via mean pooling, concatenates with self features,
    and applies a linear transform + ReLU.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] directed edges (src -> dst)

        Returns:
            [N, out_dim] updated node features
        """
        N = x.shape[0]

        if edge_index.shape[1] == 0:
            # No edges (isolated nodes): just use self features + zeros for neighbor
            combined = torch.cat([x, torch.zeros_like(x)], dim=-1)
            return F.relu(self.linear(combined))

        src, dst = edge_index[0], edge_index[1]

        # Gather source features and aggregate by destination (mean)
        neighbor_mean = scatter_mean(x[src], dst, dim_size=N)

        # Concatenate self + aggregated neighbor features
        combined = torch.cat([x, neighbor_mean], dim=-1)

        return F.relu(self.linear(combined))


class HexGNN(nn.Module):
    """Graph Neural Network for hex board evaluation.

    Architecture:
        Input: [N_placed, 23] node features + edge_index
        -> Linear(23, hidden) -> ReLU
        -> N x HexGNNLayer (message passing)
        -> Global mean pool over all nodes -> [hidden]
        -> concat(pool, global_features) -> [hidden + global_dim]
        -> Linear(hidden + global_dim, 64) -> ReLU -> Linear(64, 1)
        -> scalar prediction (remaining score)
    """

    def __init__(self, node_in=NODE_FEATURES, hidden=128, n_layers=3, global_dim=53):
        super().__init__()
        self.node_in = node_in
        self.hidden = hidden
        self.n_layers = n_layers
        self.global_dim = global_dim

        # Initial node embedding
        self.node_embed = nn.Linear(node_in, hidden)

        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            HexGNNLayer(hidden, hidden) for _ in range(n_layers)
        ])

        # Readout MLP: pooled graph features + global features -> scalar
        readout_in = hidden + global_dim
        self.readout_fc1 = nn.Linear(readout_in, 64)
        self.readout_fc2 = nn.Linear(64, 1)

    def forward(self, node_features, edge_index, batch_idx, global_features, node_counts):
        """
        Args:
            node_features: [total_nodes, 23] concatenated node features
            edge_index: [2, total_edges] batched edge indices (already offset)
            batch_idx: [total_nodes] graph membership for each node
            global_features: [B, global_dim] per-graph global features
            node_counts: list of int, nodes per graph

        Returns:
            [B] predicted remaining scores
        """
        B = global_features.shape[0]

        # Initial embedding
        x = F.relu(self.node_embed(node_features))  # [total_nodes, hidden]

        # Message passing
        for layer in self.gnn_layers:
            x = layer(x, edge_index)  # [total_nodes, hidden]

        # Global mean pool per graph
        pooled = scatter_mean(x, batch_idx, dim_size=B)  # [B, hidden]

        # Concatenate with global features
        combined = torch.cat([pooled, global_features], dim=-1)  # [B, hidden + global_dim]

        # Readout
        h = F.relu(self.readout_fc1(combined))  # [B, 64]
        return self.readout_fc2(h).squeeze(-1)  # [B]


# ─── Training ───

def train(args):
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    if args.dry_run:
        print("Dry run: generating 100 synthetic samples...")
        raw_samples = generate_synthetic_data(100)
    else:
        if not os.path.exists(args.samples):
            print(f"Error: {args.samples} not found")
            return
        print(f"Loading samples from {args.samples}...")
        t0 = time.time()
        raw_samples = load_tile_samples(args.samples)
        print(f"  Loaded {len(raw_samples)} samples in {time.time() - t0:.1f}s")

    if len(raw_samples) == 0:
        print("Error: no samples loaded")
        return

    # Determine global feature dimension from data
    global_dim = max(s[2].shape[0] for s in raw_samples)
    print(f"  Global feature dim: {global_dim}")

    # Train/val split (90/10)
    n_val = max(1, len(raw_samples) // 10)
    n_train = len(raw_samples) - n_val

    # Shuffle deterministically
    rng = np.random.default_rng(0)
    indices = rng.permutation(len(raw_samples))
    train_samples = [raw_samples[i] for i in indices[:n_train]]
    val_samples = [raw_samples[i] for i in indices[n_train:]]

    train_dataset = TileGraphDataset(train_samples)
    val_dataset = TileGraphDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_graphs,
    )

    # Target statistics for reporting
    all_targets = [s[3] for s in raw_samples]
    target_mean = np.mean(all_targets)
    target_std = np.std(all_targets)
    print(f"  Target stats: mean={target_mean:.2f}, std={target_std:.2f}, "
          f"range=[{min(all_targets):.1f}, {max(all_targets):.1f}]")
    print(f"  Train: {n_train}, Val: {n_val}")

    # Model
    model = HexGNN(
        node_in=NODE_FEATURES,
        hidden=args.hidden,
        n_layers=args.n_layers,
        global_dim=global_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {NODE_FEATURES} -> {args.hidden} (x{args.n_layers} GNN) "
          f"-> pool+global({global_dim}) -> 64 -> 1")
    print(f"  Parameters: {n_params:,}")

    # Load checkpoint
    if args.init_weights and os.path.exists(args.init_weights):
        checkpoint = torch.load(args.init_weights, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        print(f"  Loaded checkpoint: {args.init_weights}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'Train':>10} {'Val':>10} {'RMSE':>8} {'LR':>10} {'Time':>6}")
    print("-" * 55)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for nf, ei, bi, gf, targets, nc in train_loader:
            nf = nf.to(device)
            ei = ei.to(device)
            bi = bi.to(device)
            gf = gf.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(nf, ei, bi, gf, nc)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * targets.shape[0]
            train_count += targets.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for nf, ei, bi, gf, targets, nc in val_loader:
                nf = nf.to(device)
                ei = ei.to(device)
                bi = bi.to(device)
                gf = gf.to(device)
                targets = targets.to(device)

                preds = model(nf, ei, bi, gf, nc)
                loss = criterion(preds, targets)

                val_loss_sum += loss.item() * targets.shape[0]
                val_count += targets.shape[0]

        val_loss = val_loss_sum / max(val_count, 1)
        val_rmse = math.sqrt(val_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        scheduler.step()

        # Track best
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            improved = " *"

            # Save best checkpoint
            torch.save(model.state_dict(), args.out)

        print(f"{epoch+1:5d} {train_loss:10.4f} {val_loss:10.4f} {val_rmse:8.3f} "
              f"{lr:10.6f} {elapsed:5.1f}s{improved}")

    # Summary
    print("-" * 55)
    best_rmse = math.sqrt(best_val_loss)
    print(f"Best val loss: {best_val_loss:.4f} (RMSE {best_rmse:.3f}) at epoch {best_epoch}")
    print(f"Saved checkpoint: {args.out}")
    print(f"  {n_params:,} parameters")


def main():
    parser = argparse.ArgumentParser(
        description='GNN training for Cascadia board evaluation (remaining score prediction)'
    )
    parser.add_argument('--samples', default='tile_tokens.bin',
                        help='Training data file in TILE binary format')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension for GNN layers')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of GNN message-passing layers')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--out', default='gnn_weights.pt',
                        help='Output checkpoint path')
    parser.add_argument('--init-weights', default=None,
                        help='Load checkpoint to resume training')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test with synthetic data (100 samples)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
