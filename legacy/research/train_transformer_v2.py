"""Encoder-only transformer v2 for Cascadia, with NNUE-rich per-tile features.

Reads TIL2 binary format (23 bytes/tile) and trains a transformer that
processes variable-length tile sequences + RICH globals to predict remaining
score.

KEY DESIGN CHOICE: by default we use **per-cell learned position embedding**
(441-cell lookup, like a vision transformer over a hex grid) instead of
sequence-position encoding. This lets attention learn adjacency from position
without needing the explicit per-cell adjacency channels that NNUE uses.

Per-tile content features (default 43 channels):
  - 6 edges × 5 terrain types one-hot = 30
  - 6 wildlife states one-hot (none, 5 types) = 6
  - 5 allowed_mask bits
  - 2 flags (keystone, has_wildlife)
  TOTAL CONTENT = 43 floats per tile

Plus a learned cell-position embedding (441 × d_model lookup) added to the
projected tile token, giving the transformer rich spatial info.

If --include-adjacency is set, the 78 channels of per-cell adjacency
(6 dirs × 7 neighbor_wildlife + 6 dirs × 6 neighbor_terrain) are appended
as content features (total 121 channels per tile). This is the NNUE-style
inductive-bias variant.

Global features (rich one-hot, ~785 dims):
  - turn (21) + nature_tokens (9)
  - wildlife_counts (5 × 25 bins) = 125
  - largest_habitat (5 × 14 bins) = 70
  - bag_remaining (5 × 25 bins) = 125
  - opp_habitat (5 × 14 bins) = 70
  - market_terrain1/2 (4 × 5 each) = 40
  - market_wildlife (4 × 6) = 24
  - tbag_terrain (5 × 30) = 150
  - tbag_wildlife (5 × 30) = 150
  - overflow_used (1)
  TOTAL = 785

Architecture:
  Per tile: Linear(124 -> d_model)
  Sequence: [CLS, tile_1, ..., tile_N]
  -> N x TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
  -> CLS output
  -> concat(CLS[d_model], global[785]) -> Linear -> ReLU -> Linear -> scalar

Usage:
    python3 train_transformer_v2.py --samples rich_tile_tokens.bin --epochs 30 --lr 3e-4
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

NUM_TERRAIN_TYPES = 5     # 0-4 (Forest, Mountain, Prairie, Wetland, River)
NUM_WILDLIFE_TYPES = 6    # 0=none, 1-5=Bear/Elk/Salmon/Hawk/Fox
NEIGHBOR_WL_STATES = 7    # 0=no tile, 1-5=wildlife, 6=tile-no-wildlife
NEIGHBOR_TR_STATES = 6    # 0=no tile, 1-5=terrain
MAX_TILES = 23
TERRAIN_EDGES = 6
TILE_BYTES = 23

# Hex grid: Cascadia stores cells in a 21×21 axial grid (441 cells, ~half
# unreachable). We use a learned position embedding indexed by cell ID (0..440).
GRID_DIM = 21
NUM_CELLS = GRID_DIM * GRID_DIM  # 441

# Per-tile content channels (no adjacency by default; 43 channels)
TILE_CONTENT_DIM = (
    TERRAIN_EDGES * NUM_TERRAIN_TYPES   # 30: edge terrain one-hots
    + NUM_WILDLIFE_TYPES                # 6: wildlife one-hot
    + 5                                  # allowed_mask bits
    + 2                                  # keystone, has_wildlife
)
assert TILE_CONTENT_DIM == 43, f"expected 43, got {TILE_CONTENT_DIM}"

# Per-tile adjacency channels (opt-in; 78 channels)
TILE_ADJ_DIM = (
    TERRAIN_EDGES * NEIGHBOR_WL_STATES   # 42: neighbor wildlife
    + TERRAIN_EDGES * NEIGHBOR_TR_STATES # 36: neighbor terrain
)
assert TILE_ADJ_DIM == 78

# Global feature one-hot bin counts
G_TURN_BINS = 21
G_TOKEN_BINS = 9
G_WL_COUNT_BINS = 25
G_HAB_BINS = 14
G_BAG_BINS = 25
G_TBAG_BINS = 30

GLOBAL_FEAT_DIM = (
    G_TURN_BINS                 # 21
    + G_TOKEN_BINS              # 9
    + 5 * G_WL_COUNT_BINS       # 125
    + 5 * G_HAB_BINS            # 70
    + 5 * G_BAG_BINS            # 125
    + 5 * G_HAB_BINS            # 70: opp_habitat shares hab bins
    + 4 * NUM_TERRAIN_TYPES     # 20: market_terrain1
    + 4 * NUM_TERRAIN_TYPES     # 20: market_terrain2
    + 4 * NUM_WILDLIFE_TYPES    # 24: market_wildlife
    + 5 * G_TBAG_BINS           # 150: tbag_terrain
    + 5 * G_TBAG_BINS           # 150: tbag_wildlife
    + 1                         # overflow_used
)
assert GLOBAL_FEAT_DIM == 785, f"expected 785, got {GLOBAL_FEAT_DIM}"


# ─── Data Loading ───

def load_tile_samples(path):
    """Load samples from binary TIL2 format.

    Returns:
        tiles_list:   list of np.ndarray, each shape [num_tiles, 23] (raw bytes per tile)
        globals_list: list of np.ndarray, each shape [45] (raw global bytes)
        targets:      np.ndarray shape [N] (f32 targets)
    """
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    magic = data[:4]
    if magic != b'TIL2':
        raise ValueError(f"Bad magic: {magic!r}, expected b'TIL2'")
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

        tile_block_size = num_tiles * TILE_BYTES
        if pos + tile_block_size + GLOBAL_BYTES + 4 > len(data):
            break

        tile_data = np.frombuffer(data, dtype=np.uint8, count=tile_block_size, offset=pos)
        tile_data = tile_data.reshape(num_tiles, TILE_BYTES)
        pos += tile_block_size

        raw_global = np.frombuffer(data, dtype=np.uint8, count=GLOBAL_BYTES, offset=pos).copy()
        pos += GLOBAL_BYTES

        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4

        tiles_list.append(tile_data.copy())
        globals_list.append(raw_global)
        targets.append(target)

    targets = np.array(targets, dtype=np.float32)
    print(f"Loaded {len(tiles_list)} samples from {path}")
    if len(tiles_list) > 0:
        print(f"  Tiles per sample: {min(len(t) for t in tiles_list)}-{max(len(t) for t in tiles_list)}")
        print(f"  Tile content dim: {TILE_CONTENT_DIM} (+78 if --include-adjacency)")
        print(f"  Global feat dim: {GLOBAL_FEAT_DIM}")
        print(f"  Target range: [{targets.min():.1f}, {targets.max():.1f}], mean={targets.mean():.1f}")
    return tiles_list, globals_list, targets


def cell_id_from_qr(q, r):
    """Map (q, r) axial coordinates to a 0..NUM_CELLS-1 cell ID.

    The Rust side stores i8 q/r in the range roughly [-10, 10]; shift to
    [0, GRID_DIM-1] for indexing.
    """
    half = GRID_DIM // 2
    qi = np.clip(q.astype(np.int32) + half, 0, GRID_DIM - 1)
    ri = np.clip(r.astype(np.int32) + half, 0, GRID_DIM - 1)
    return qi * GRID_DIM + ri


def parse_tile_features(tile_raw, include_adjacency=False):
    """Convert raw tile bytes [num_tiles, 23] into:
      - content_feats: [num_tiles, TILE_CONTENT_DIM (+TILE_ADJ_DIM if adj)]
      - cell_ids: [num_tiles] int64 (0..NUM_CELLS-1)
    """
    num_tiles = tile_raw.shape[0]
    feat_dim = TILE_CONTENT_DIM + (TILE_ADJ_DIM if include_adjacency else 0)
    out = np.zeros((num_tiles, feat_dim), dtype=np.float32)

    # Layout of tile bytes (23):
    #   [0..6)   terrain_triangles (0-4)
    #   [6]      wildlife (0=none, 1-5)
    #   [7]      allowed_mask (5 bits)
    #   [8]      flags (bit0=keystone, bit1=has_wildlife)
    #   [9]      q (i8)
    #   [10]     r (i8)
    #   [11..17) neighbor_wildlife (0=no tile, 1-5, 6)
    #   [17..23) neighbor_terrain (0=no tile, 1-5)

    offset = 0

    # Edge terrain one-hots: 6 edges × 5 types = 30
    for e in range(TERRAIN_EDGES):
        terrain = tile_raw[:, e].astype(np.int64)
        valid = (terrain >= 0) & (terrain < NUM_TERRAIN_TYPES)
        rows = np.where(valid)[0]
        cols = offset + terrain[valid]
        out[rows, cols] = 1.0
        offset += NUM_TERRAIN_TYPES

    # Wildlife one-hot: 6 states
    wildlife = tile_raw[:, 6].astype(np.int64)
    valid = (wildlife >= 0) & (wildlife < NUM_WILDLIFE_TYPES)
    rows = np.where(valid)[0]
    cols = offset + wildlife[valid]
    out[rows, cols] = 1.0
    offset += NUM_WILDLIFE_TYPES

    # Allowed mask: 5 bits
    allowed_mask = tile_raw[:, 7]
    for b in range(5):
        out[:, offset + b] = ((allowed_mask >> b) & 1).astype(np.float32)
    offset += 5

    # Flags: keystone, has_wildlife
    flags = tile_raw[:, 8]
    out[:, offset + 0] = (flags & 1).astype(np.float32)
    out[:, offset + 1] = ((flags >> 1) & 1).astype(np.float32)
    offset += 2

    assert offset == TILE_CONTENT_DIM, f"content offset {offset} != {TILE_CONTENT_DIM}"

    if include_adjacency:
        # Neighbor wildlife one-hots: 6 dirs × 7 states = 42
        for d in range(TERRAIN_EDGES):
            nw = tile_raw[:, 11 + d].astype(np.int64)
            valid = (nw >= 0) & (nw < NEIGHBOR_WL_STATES)
            rows = np.where(valid)[0]
            cols = offset + nw[valid]
            out[rows, cols] = 1.0
            offset += NEIGHBOR_WL_STATES

        # Neighbor terrain one-hots: 6 dirs × 6 states = 36
        for d in range(TERRAIN_EDGES):
            nt = tile_raw[:, 17 + d].astype(np.int64)
            valid = (nt >= 0) & (nt < NEIGHBOR_TR_STATES)
            rows = np.where(valid)[0]
            cols = offset + nt[valid]
            out[rows, cols] = 1.0
            offset += NEIGHBOR_TR_STATES

    # Cell IDs from (q, r)
    q = tile_raw[:, 9].astype(np.int8)
    r = tile_raw[:, 10].astype(np.int8)
    cell_ids = cell_id_from_qr(q, r).astype(np.int64)

    return out, cell_ids


def parse_global_features(raw_global):
    """Convert raw 45-byte global into a [785] float one-hot vector.

    Layout of raw_global (45 bytes):
      [0]      turn (0-20)
      [1]      nature_tokens (0-8)
      [2..7)   wildlife_counts (5)
      [7..12)  largest_habitat (5)
      [12..17) bag_remaining (5)
      [17..22) opp_habitat (5)
      [22..26) market_terrain1 (4)
      [26..30) market_terrain2 (4)
      [30..34) market_wildlife (4)
      [34..39) tbag_terrain (5)
      [39..44) tbag_wildlife (5)
      [44]     overflow_used
    """
    g = raw_global
    out = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    offset = 0

    # turn: 21 bins
    turn = min(int(g[0]), G_TURN_BINS - 1)
    out[offset + turn] = 1.0
    offset += G_TURN_BINS

    # nature_tokens: 9 bins
    nt = min(int(g[1]), G_TOKEN_BINS - 1)
    out[offset + nt] = 1.0
    offset += G_TOKEN_BINS

    # wildlife_counts: 5 × 25 bins
    for i in range(5):
        v = min(int(g[2 + i]), G_WL_COUNT_BINS - 1)
        out[offset + v] = 1.0
        offset += G_WL_COUNT_BINS

    # largest_habitat: 5 × 14 bins
    for i in range(5):
        v = min(int(g[7 + i]), G_HAB_BINS - 1)
        out[offset + v] = 1.0
        offset += G_HAB_BINS

    # bag_remaining: 5 × 25 bins
    for i in range(5):
        v = min(int(g[12 + i]), G_BAG_BINS - 1)
        out[offset + v] = 1.0
        offset += G_BAG_BINS

    # opp_habitat: 5 × 14 bins
    for i in range(5):
        v = min(int(g[17 + i]), G_HAB_BINS - 1)
        out[offset + v] = 1.0
        offset += G_HAB_BINS

    # market_terrain1: 4 × 5
    for i in range(4):
        v = int(g[22 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            out[offset + v] = 1.0
        offset += NUM_TERRAIN_TYPES

    # market_terrain2: 4 × 5
    for i in range(4):
        v = int(g[26 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            out[offset + v] = 1.0
        offset += NUM_TERRAIN_TYPES

    # market_wildlife: 4 × 6 (0=none, 1-5)
    for i in range(4):
        v = int(g[30 + i])
        if 0 <= v < NUM_WILDLIFE_TYPES:
            out[offset + v] = 1.0
        offset += NUM_WILDLIFE_TYPES

    # tbag_terrain: 5 × 30
    for i in range(5):
        v = min(int(g[34 + i]), G_TBAG_BINS - 1)
        out[offset + v] = 1.0
        offset += G_TBAG_BINS

    # tbag_wildlife: 5 × 30
    for i in range(5):
        v = min(int(g[39 + i]), G_TBAG_BINS - 1)
        out[offset + v] = 1.0
        offset += G_TBAG_BINS

    # overflow_used
    out[offset] = float(g[44])
    offset += 1

    assert offset == GLOBAL_FEAT_DIM, f"offset {offset} != GLOBAL_FEAT_DIM {GLOBAL_FEAT_DIM}"
    return out


# ─── Dataset ───

class RichTileTokenDataset(Dataset):
    def __init__(self, tiles_list, globals_list, targets, include_adjacency=False):
        self.tiles_list = tiles_list
        self.globals_list = globals_list
        self.targets = targets
        self.include_adjacency = include_adjacency

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        return (self.tiles_list[idx], self.globals_list[idx],
                self.targets[idx], self.include_adjacency)


def collate_rich_tiles(batch):
    """Pad to max sequence length in batch (+1 for CLS).

    Returns:
        tile_feats:   FloatTensor [B, max_seq, content_dim]
        cell_ids:     LongTensor  [B, max_seq] (0 for CLS/pad — pad's pos embed
                                                gets masked out anyway)
        attn_mask:    BoolTensor  [B, max_seq]   -- True for padding
        global_feats: FloatTensor [B, 785]
        targets:      FloatTensor [B]
    """
    parsed_tiles = []
    parsed_cells = []
    globals_list = []
    targets_list = []
    lengths = []
    include_adj = batch[0][3]

    for tile_raw, global_raw, target, _ in batch:
        tile_feats, cell_ids = parse_tile_features(tile_raw, include_adjacency=include_adj)
        global_feats = parse_global_features(global_raw)
        parsed_tiles.append(tile_feats)
        parsed_cells.append(cell_ids)
        globals_list.append(global_feats)
        targets_list.append(target)
        lengths.append(tile_feats.shape[0] + 1)  # +1 for CLS

    max_len = max(lengths)
    B = len(batch)
    feat_dim = parsed_tiles[0].shape[1]

    tile_padded = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    cell_padded = torch.zeros(B, max_len, dtype=torch.long)
    attn_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = ignored

    for i in range(B):
        seq_len = lengths[i]
        tile_len = seq_len - 1
        # CLS at position 0 (zeros in tile_padded; replaced by cls embed in forward)
        # Position 0 gets cell_id=NUM_CELLS (special CLS bucket)
        cell_padded[i, 0] = NUM_CELLS  # CLS uses bucket NUM_CELLS
        tile_padded[i, 1:tile_len + 1] = torch.from_numpy(parsed_tiles[i])
        cell_padded[i, 1:tile_len + 1] = torch.from_numpy(parsed_cells[i])
        attn_mask[i, :seq_len] = False

    global_feats = torch.from_numpy(np.stack(globals_list))
    targets = torch.tensor(targets_list, dtype=torch.float32)

    return tile_padded, cell_padded, attn_mask, global_feats, targets


# ─── Model ───

class CascadiaTransformerV2(nn.Module):
    """Encoder-only transformer with rich content + cell-position embeddings.

    Spatial structure is provided via a learned per-cell embedding (441 cells +
    1 CLS bucket). Attention discovers adjacency from these positions instead
    of being given explicit neighbor features.
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=3, dim_feedforward=512,
                 dropout=0.1, num_global=GLOBAL_FEAT_DIM,
                 num_tile_feats=TILE_CONTENT_DIM, include_adjacency=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.num_global = num_global
        self.num_tile_feats = num_tile_feats
        self.include_adjacency = include_adjacency

        # Project tile content features to d_model
        self.tile_proj = nn.Linear(num_tile_feats, d_model)
        self.tile_norm = nn.LayerNorm(num_tile_feats)

        # Project rich globals (sparse one-hots)
        self.global_norm = nn.LayerNorm(num_global)
        self.global_proj = nn.Sequential(
            nn.Linear(num_global, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Learned CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Per-cell positional embedding: NUM_CELLS cells + 1 CLS bucket
        self.cell_embed = nn.Embedding(NUM_CELLS + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model + 128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.tile_proj.weight)
        nn.init.zeros_(self.tile_proj.bias)
        for m in [self.global_proj, self.head]:
            for sub in m:
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    nn.init.zeros_(sub.bias)
        nn.init.normal_(self.cell_embed.weight, std=0.02)

    def forward(self, tile_feats, cell_ids, attn_mask, global_feats):
        """
        Args:
            tile_feats:   [B, S, num_tile_feats] FloatTensor
            cell_ids:     [B, S] LongTensor (cell indices; NUM_CELLS for CLS)
            attn_mask:    [B, S] BoolTensor (True for padding)
            global_feats: [B, num_global] FloatTensor

        Returns:
            [B] FloatTensor — predicted remaining score
        """
        B, S, _ = tile_feats.shape

        tile_normed = self.tile_norm(tile_feats)
        tile_tokens = self.tile_proj(tile_normed)  # [B, S, d_model]

        # Replace position 0 with CLS embedding
        cls_expanded = self.cls_token.expand(B, -1, -1)
        tile_tokens = tile_tokens.clone()
        tile_tokens[:, 0:1, :] = cls_expanded

        # Add per-cell positional embeddings (CLS gets bucket NUM_CELLS)
        tile_tokens = tile_tokens + self.cell_embed(cell_ids)

        encoded = self.transformer(tile_tokens, src_key_padding_mask=attn_mask)
        encoded = self.layer_norm(encoded)
        cls_out = encoded[:, 0, :]  # [B, d_model]

        global_normed = self.global_norm(global_feats)
        global_proj = self.global_proj(global_normed)  # [B, 128]

        combined = torch.cat([cls_out, global_proj], dim=-1)
        output = self.head(combined).squeeze(-1)
        return output

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─── Training ───

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, train_loader, val_loader, optimizer, scheduler, device,
          epochs, checkpoint_path, save_every=False):
    best_val_loss = float('inf')
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for tile_feats, cell_ids, attn_mask, global_feats, targets in train_loader:
            tile_feats = tile_feats.to(device)
            cell_ids = cell_ids.to(device)
            attn_mask = attn_mask.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(tile_feats, cell_ids, attn_mask, global_feats)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = targets.shape[0]
            train_loss_sum += loss.item() * bs
            train_samples += bs

        train_loss = train_loss_sum / max(1, train_samples)
        train_rmse = math.sqrt(train_loss)

        val_loss = 0.0
        val_rmse = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for tile_feats, cell_ids, attn_mask, global_feats, targets in val_loader:
                    tile_feats = tile_feats.to(device)
                    cell_ids = cell_ids.to(device)
                    attn_mask = attn_mask.to(device)
                    global_feats = global_feats.to(device)
                    targets = targets.to(device)
                    preds = model(tile_feats, cell_ids, attn_mask, global_feats)
                    loss = criterion(preds, targets)
                    bs = targets.shape[0]
                    val_loss_sum += loss.item() * bs
                    val_samples += bs
            val_loss = val_loss_sum / max(1, val_samples)
            val_rmse = math.sqrt(val_loss)

        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        if val_loader is not None:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_rmse={train_rmse:.3f}  val_rmse={val_rmse:.3f}  "
                  f"lr={current_lr:.2e}  {elapsed:.1f}s")
        else:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_rmse={train_rmse:.3f}  "
                  f"lr={current_lr:.2e}  {elapsed:.1f}s")

        current_loss = val_loss if val_loader is not None else train_loss
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            save_checkpoint(model, checkpoint_path)

        if save_every:
            base, ext = os.path.splitext(checkpoint_path)
            save_checkpoint(model, f"{base}_epoch{epoch}{ext}")

    return best_val_loss


def save_checkpoint(model, path):
    config = {
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': model.n_layers,
        'dim_feedforward': model.dim_feedforward,
        'num_global': model.num_global,
        'num_tile_feats': model.num_tile_feats,
        'include_adjacency': model.include_adjacency,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CascadiaTransformerV2(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        num_global=config['num_global'],
        num_tile_feats=config['num_tile_feats'],
        include_adjacency=config.get('include_adjacency', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Cascadia transformer v2 (rich features)')
    parser.add_argument('--samples', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--dim-feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--out', type=str, default='transformer_v2.pt')
    parser.add_argument('--init-weights', type=str, default=None)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-every-epoch', action='store_true')
    parser.add_argument('--include-adjacency', action='store_true',
                        help='Use 78 channels of explicit per-cell adjacency '
                             '(NNUE-style inductive bias). Default: rely on '
                             'cell-position embedding + attention.')
    args = parser.parse_args()

    import sys
    sys.stdout.reconfigure(line_buffering=True)

    device = get_device()
    tile_feat_dim = TILE_CONTENT_DIM + (TILE_ADJ_DIM if args.include_adjacency else 0)
    print(f"=== Cascadia Transformer V2 (rich features) ===")
    print(f"  Device: {device}")
    print(f"  Architecture: d_model={args.d_model}, heads={args.n_heads}, "
          f"layers={args.n_layers}, ff={args.dim_feedforward}")
    print(f"  Tile feat dim: {tile_feat_dim} (content={TILE_CONTENT_DIM}"
          f"{', adj=78' if args.include_adjacency else ', no-adjacency'})")
    print(f"  Position: per-cell embedding (NUM_CELLS={NUM_CELLS} + 1 CLS bucket)")
    print(f"  Global feat dim: {GLOBAL_FEAT_DIM}")
    print(f"  Training: epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print()

    tiles_list, globals_list, targets = load_tile_samples(args.samples)
    if len(tiles_list) == 0:
        print("ERROR: No samples loaded.")
        return

    n = len(tiles_list)
    indices = np.random.default_rng(42).permutation(n)
    n_val = max(1, int(n * args.val_split)) if n > 10 else 0

    if n_val > 0:
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_dataset = RichTileTokenDataset(
            [tiles_list[i] for i in train_idx],
            [globals_list[i] for i in train_idx],
            targets[train_idx],
            include_adjacency=args.include_adjacency,
        )
        val_dataset = RichTileTokenDataset(
            [tiles_list[i] for i in val_idx],
            [globals_list[i] for i in val_idx],
            targets[val_idx],
            include_adjacency=args.include_adjacency,
        )
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = RichTileTokenDataset(tiles_list, globals_list, targets,
                                             include_adjacency=args.include_adjacency)
        val_dataset = None
        print(f"  Train: {len(train_dataset)} (no validation split)")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_rich_tiles, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_rich_tiles, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'), drop_last=False,
        )

    if args.init_weights and os.path.exists(args.init_weights):
        print(f"  Loading checkpoint: {args.init_weights}")
        model = load_checkpoint(args.init_weights, device)
    else:
        model = CascadiaTransformerV2(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward, dropout=args.dropout,
            num_tile_feats=tile_feat_dim,
            include_adjacency=args.include_adjacency,
        )

    model = model.to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print()

    t0 = time.time()
    best_loss = train(
        model, train_loader, val_loader, optimizer, scheduler,
        device, args.epochs, args.out, save_every=args.save_every_epoch,
    )
    total_time = time.time() - t0
    best_rmse = math.sqrt(best_loss) if best_loss < float('inf') else float('inf')

    print()
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best RMSE: {best_rmse:.3f}")
    print(f"  Checkpoint saved to: {args.out}")


if __name__ == '__main__':
    main()
