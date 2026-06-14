"""Encoder-only transformer for predicting remaining score in Cascadia.

Reads tile-token binary format (magic "TILE"), trains a transformer that
processes variable-length tile sequences + global features to predict how
many points remain to be scored (regression, same target as the NNUE).

Architecture:
  Per tile: terrain_embed(6 x embed_dim) + wildlife_embed + metadata -> Linear(d_model)
  Sequence: [CLS, tile_1, ..., tile_N]
  -> N x TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
  -> CLS output
  -> concat(CLS[d_model], global[num_global]) -> Linear -> ReLU -> Linear -> scalar

Usage:
    # Train from tile-token data
    python3 train_transformer.py --samples tile_tokens.bin --epochs 30 --lr 3e-4

    # Dry run (synthetic data, no file needed)
    python3 train_transformer.py --dry-run --epochs 5

    # Resume from checkpoint
    python3 train_transformer.py --samples tile_tokens.bin --init-weights transformer_weights.pt

    # All options
    python3 train_transformer.py --samples tile_tokens.bin --epochs 30 --lr 3e-4 \\
        --d-model 128 --n-heads 4 --n-layers 3 --batch-size 512 \\
        --out transformer_weights.pt
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

NUM_TERRAIN_TYPES = 5     # 0-4
NUM_WILDLIFE_TYPES = 6    # 0=none, 1-5=Bear/Elk/Salmon/Hawk/Fox
MAX_TILES = 23            # turns 1-20 start with 3 tiles, place one per turn
TERRAIN_EDGES = 6         # hex has 6 edges
TILE_METADATA_DIM = 10    # 5 allowed bits + 1 keystone + 1 has_wildlife + 3 position
DEFAULT_NUM_GLOBAL = 53   # global feature count (turn, tokens, wildlife counts, etc.)

# Binary format: per tile, 11 bytes
#   u8[6] terrain_triangles
#   u8    wildlife_type
#   u8    allowed_mask (5 bits)
#   u8    flags (bit 0=keystone, bit 1=has_wildlife)
#   i8    q, i8 r
TILE_BYTES = 11


# ─── Data Loading ───

def load_tile_samples(path):
    """Load samples from binary TILE format.

    Returns:
        tiles_list:   list of np.ndarray, each shape [num_tiles, 11] (raw bytes per tile)
        globals_list: list of np.ndarray, each shape [num_global] (f32 globals)
        targets:      np.ndarray shape [N] (f32 targets)
    """
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    magic = data[:4]
    if magic != b'TILE':
        raise ValueError(f"Bad magic: {magic!r}, expected b'TILE'")
    pos = 4

    tiles_list = []
    globals_list = []
    targets = []

    while pos < len(data):
        # Need at least 1 byte for num_tiles
        if pos + 1 > len(data):
            break
        num_tiles = data[pos]
        pos += 1

        if num_tiles == 0 or num_tiles > MAX_TILES:
            break

        tile_block_size = num_tiles * TILE_BYTES
        if pos + tile_block_size > len(data):
            break

        # Read raw tile bytes
        tile_data = np.frombuffer(data, dtype=np.uint8, count=tile_block_size, offset=pos)
        tile_data = tile_data.reshape(num_tiles, TILE_BYTES)
        pos += tile_block_size

        # Global features: fixed layout of u8 values (45 bytes total)
        # turn(1) + tokens(1) + wl_counts(5) + hab_sizes(5) + bag_rem(5) +
        # opp_hab(5) + mkt_t1(4) + mkt_t2(4) + mkt_wl(4) + tbag_t(5) +
        # tbag_w(5) + overflow(1) = 45 bytes
        GLOBAL_BYTES = 45
        if pos + GLOBAL_BYTES + 4 > len(data):
            break
        raw_global = np.frombuffer(data, dtype=np.uint8, count=GLOBAL_BYTES, offset=pos).copy()
        pos += GLOBAL_BYTES
        # Convert to float: normalize some fields
        global_feats = np.zeros(DEFAULT_NUM_GLOBAL, dtype=np.float32)
        g = raw_global
        gi = 0
        global_feats[gi] = g[0] / 20.0;  gi += 1  # turn (0-20 normalized)
        global_feats[gi] = g[1] / 8.0;   gi += 1  # tokens (0-8 normalized)
        for i in range(5): global_feats[gi] = g[2+i] / 10.0;  gi += 1  # wl_counts
        for i in range(5): global_feats[gi] = g[7+i] / 13.0;  gi += 1  # hab_sizes
        for i in range(5): global_feats[gi] = g[12+i] / 20.0; gi += 1  # bag_remaining
        for i in range(5): global_feats[gi] = g[17+i] / 13.0; gi += 1  # opp_habitat
        for i in range(4): global_feats[gi] = g[22+i] / 5.0;  gi += 1  # market terrain1
        for i in range(4): global_feats[gi] = g[26+i] / 5.0;  gi += 1  # market terrain2
        for i in range(4): global_feats[gi] = g[30+i] / 5.0;  gi += 1  # market wildlife
        for i in range(5): global_feats[gi] = g[34+i] / 29.0; gi += 1  # tbag terrain
        for i in range(5): global_feats[gi] = g[39+i] / 29.0; gi += 1  # tbag wildlife
        global_feats[gi] = float(g[44]);  gi += 1  # overflow_used

        # target (f32)
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4

        tiles_list.append(tile_data.copy())
        globals_list.append(global_feats)
        targets.append(target)

    targets = np.array(targets, dtype=np.float32)
    print(f"Loaded {len(tiles_list)} samples from {path}")
    if len(tiles_list) > 0:
        print(f"  Tiles per sample: {min(len(t) for t in tiles_list)}-{max(len(t) for t in tiles_list)}")
        print(f"  Global features: {len(globals_list[0])}")
        print(f"  Target range: [{targets.min():.1f}, {targets.max():.1f}], mean={targets.mean():.1f}")
    return tiles_list, globals_list, targets


def generate_synthetic_data(num_samples=100):
    """Generate synthetic tile-token data for dry-run testing."""
    rng = np.random.default_rng(42)
    tiles_list = []
    globals_list = []
    targets = []

    for _ in range(num_samples):
        num_tiles = rng.integers(3, MAX_TILES + 1)
        tile_data = np.zeros((num_tiles, TILE_BYTES), dtype=np.uint8)
        # terrain triangles: random 0-4
        tile_data[:, :6] = rng.integers(0, NUM_TERRAIN_TYPES, size=(num_tiles, 6), dtype=np.uint8)
        # wildlife type: 0-5
        tile_data[:, 6] = rng.integers(0, NUM_WILDLIFE_TYPES, size=num_tiles, dtype=np.uint8)
        # allowed mask: random 5-bit mask
        tile_data[:, 7] = rng.integers(0, 32, size=num_tiles, dtype=np.uint8)
        # flags: random 2 bits
        tile_data[:, 8] = rng.integers(0, 4, size=num_tiles, dtype=np.uint8)
        # q, r: hex coords in [-5, 5] stored as i8
        tile_data[:, 9] = rng.integers(-5, 6, size=num_tiles).astype(np.int8).view(np.uint8)
        tile_data[:, 10] = rng.integers(-5, 6, size=num_tiles).astype(np.int8).view(np.uint8)

        global_feats = rng.standard_normal(DEFAULT_NUM_GLOBAL).astype(np.float32)
        target = np.float32(rng.uniform(0, 70))

        tiles_list.append(tile_data)
        globals_list.append(global_feats)
        targets.append(target)

    targets = np.array(targets, dtype=np.float32)
    print(f"Generated {num_samples} synthetic samples")
    print(f"  Tiles per sample: {min(len(t) for t in tiles_list)}-{max(len(t) for t in tiles_list)}")
    print(f"  Global features: {DEFAULT_NUM_GLOBAL}")
    print(f"  Target range: [{targets.min():.1f}, {targets.max():.1f}], mean={targets.mean():.1f}")
    return tiles_list, globals_list, targets


# ─── Dataset ───

class TileTokenDataset(Dataset):
    """Variable-length tile token dataset.

    Each sample is (tile_features, global_features, target) where tile_features
    has shape [num_tiles, TILE_BYTES] and varies per sample.
    """

    def __init__(self, tiles_list, globals_list, targets):
        self.tiles_list = tiles_list
        self.globals_list = globals_list
        self.targets = targets

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, idx):
        return self.tiles_list[idx], self.globals_list[idx], self.targets[idx]


def parse_tile_features(tile_raw):
    """Convert raw tile bytes [num_tiles, 11] into model-ready tensors.

    Returns:
        terrain:   LongTensor [num_tiles, 6]  -- terrain IDs per edge (0-4)
        wildlife:  LongTensor [num_tiles]      -- wildlife type (0-5)
        metadata:  FloatTensor [num_tiles, 10] -- allowed(5) + keystone(1) + has_wildlife(1) + pos(3)
    """
    num_tiles = tile_raw.shape[0]

    terrain = tile_raw[:, :6].astype(np.int64)
    wildlife = tile_raw[:, 6].astype(np.int64)

    metadata = np.zeros((num_tiles, TILE_METADATA_DIM), dtype=np.float32)
    # Allowed wildlife bits (5 individual bits from the mask byte)
    allowed_mask = tile_raw[:, 7]
    for b in range(5):
        metadata[:, b] = ((allowed_mask >> b) & 1).astype(np.float32)
    # Keystone bit
    flags = tile_raw[:, 8]
    metadata[:, 5] = (flags & 1).astype(np.float32)
    # Has-wildlife bit
    metadata[:, 6] = ((flags >> 1) & 1).astype(np.float32)
    # Hex cube coordinates normalized to [-1, 1]
    q = tile_raw[:, 9].astype(np.int8).astype(np.float32) / 10.0
    r = tile_raw[:, 10].astype(np.int8).astype(np.float32) / 10.0
    s = -q - r  # cube coordinate constraint: q + r + s = 0
    metadata[:, 7] = q
    metadata[:, 8] = r
    metadata[:, 9] = s

    return (
        torch.from_numpy(terrain),
        torch.from_numpy(wildlife),
        torch.from_numpy(metadata),
    )


def collate_tiles(batch):
    """Custom collate for variable-length tile sequences.

    Pads to the maximum sequence length in the batch (+ 1 for CLS token).
    Returns attention mask where True = ignored (PyTorch transformer convention).

    Returns:
        terrain:      LongTensor  [B, max_seq, 6]
        wildlife:     LongTensor  [B, max_seq]
        metadata:     FloatTensor [B, max_seq, 10]
        attn_mask:    BoolTensor  [B, max_seq]  -- True for padding positions
        global_feats: FloatTensor [B, num_global]
        targets:      FloatTensor [B]
    """
    terrains = []
    wildlives = []
    metadatas = []
    globals_list = []
    targets_list = []
    lengths = []

    for tile_raw, global_feats, target in batch:
        terrain, wildlife, metadata = parse_tile_features(tile_raw)
        terrains.append(terrain)
        wildlives.append(wildlife)
        metadatas.append(metadata)
        globals_list.append(torch.from_numpy(global_feats))
        targets_list.append(target)
        # +1 for CLS token that the model will prepend
        lengths.append(terrain.shape[0] + 1)

    max_len = max(lengths)
    B = len(batch)
    num_global = globals_list[0].shape[0]

    # Allocate padded tensors
    terrain_padded = torch.zeros(B, max_len, TERRAIN_EDGES, dtype=torch.long)
    wildlife_padded = torch.zeros(B, max_len, dtype=torch.long)
    metadata_padded = torch.zeros(B, max_len, TILE_METADATA_DIM, dtype=torch.float32)
    attn_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = masked/ignored

    for i in range(B):
        seq_len = lengths[i]
        tile_len = seq_len - 1  # actual tiles (without CLS)
        # CLS is position 0 (zeros in terrain/wildlife/metadata) -- the model's
        # CLS embedding is learned and added in the forward pass; position 0
        # in these tensors is just a placeholder that gets overwritten.
        # Tiles start at position 1.
        terrain_padded[i, 1:tile_len + 1] = terrains[i]
        wildlife_padded[i, 1:tile_len + 1] = wildlives[i]
        metadata_padded[i, 1:tile_len + 1] = metadatas[i]
        attn_mask[i, :seq_len] = False  # real tokens are not masked

    global_feats = torch.stack(globals_list)
    targets = torch.tensor(targets_list, dtype=torch.float32)

    return terrain_padded, wildlife_padded, metadata_padded, attn_mask, global_feats, targets


# ─── Model ───

class CascadiaTransformer(nn.Module):
    """Encoder-only transformer for Cascadia board state evaluation.

    Input: variable-length sequence of tile tokens + global features.
    Output: scalar prediction of remaining score.
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=3, dim_feedforward=512,
                 dropout=0.1, num_global=DEFAULT_NUM_GLOBAL, terrain_embed_dim=16,
                 wildlife_embed_dim=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.num_global = num_global

        # Embedding layers for categorical features
        self.terrain_embed = nn.Embedding(NUM_TERRAIN_TYPES, terrain_embed_dim)
        self.wildlife_embed = nn.Embedding(NUM_WILDLIFE_TYPES, wildlife_embed_dim)

        # Project tile features to d_model
        # Input: 6 terrain embeddings + 1 wildlife embedding + 10 metadata floats
        tile_input_dim = TERRAIN_EDGES * terrain_embed_dim + wildlife_embed_dim + TILE_METADATA_DIM
        self.tile_proj = nn.Linear(tile_input_dim, d_model)

        # Learnable CLS token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding is not needed -- tiles have explicit hex
        # coordinates in their metadata. But we add a small learned embedding for
        # sequence position to help the transformer distinguish CLS from tiles.
        self.pos_embed = nn.Embedding(MAX_TILES + 1, d_model)  # +1 for CLS

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(d_model)

        # Final regression head: CLS output + global features -> scalar
        head_input_dim = d_model + num_global
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projections and head, small init for embeddings."""
        nn.init.xavier_uniform_(self.tile_proj.weight)
        nn.init.zeros_(self.tile_proj.bias)
        nn.init.normal_(self.terrain_embed.weight, std=0.02)
        nn.init.normal_(self.wildlife_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, terrain, wildlife, metadata, attn_mask, global_feats):
        """
        Args:
            terrain:      [B, S, 6] LongTensor  -- terrain IDs per edge
            wildlife:     [B, S]    LongTensor  -- wildlife type
            metadata:     [B, S, 10] FloatTensor -- allowed/flags/position
            attn_mask:    [B, S]    BoolTensor  -- True for padding
            global_feats: [B, G]    FloatTensor -- global features

        Returns:
            [B] FloatTensor -- predicted remaining score
        """
        B, S, _ = terrain.shape

        # Embed terrain: [B, S, 6] -> [B, S, 6, embed_dim] -> [B, S, 6*embed_dim]
        terrain_emb = self.terrain_embed(terrain)
        terrain_emb = terrain_emb.reshape(B, S, -1)

        # Embed wildlife: [B, S] -> [B, S, embed_dim]
        wildlife_emb = self.wildlife_embed(wildlife)

        # Concatenate all tile features: [B, S, tile_input_dim]
        tile_feats = torch.cat([terrain_emb, wildlife_emb, metadata], dim=-1)

        # Project to d_model: [B, S, d_model]
        tile_tokens = self.tile_proj(tile_feats)

        # Replace position 0 with the learned CLS embedding
        cls_expanded = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        tile_tokens = tile_tokens.clone()
        tile_tokens[:, 0:1, :] = cls_expanded

        # Add positional embeddings
        positions = torch.arange(S, device=terrain.device).unsqueeze(0).expand(B, -1)
        # Clamp to max position (shouldn't exceed, but be safe)
        positions = positions.clamp(max=MAX_TILES)
        tile_tokens = tile_tokens + self.pos_embed(positions)

        # Run transformer encoder
        # PyTorch expects src_key_padding_mask: True = ignore
        encoded = self.transformer(tile_tokens, src_key_padding_mask=attn_mask)
        encoded = self.layer_norm(encoded)

        # Extract CLS token output: [B, d_model]
        cls_out = encoded[:, 0, :]

        # Concatenate with global features and predict
        combined = torch.cat([cls_out, global_feats], dim=-1)
        output = self.head(combined).squeeze(-1)

        return output

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─── Training ───

def get_device():
    """Detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, train_loader, val_loader, optimizer, scheduler, device,
          epochs, checkpoint_path):
    """Training loop with validation and checkpointing."""
    best_val_loss = float('inf')
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for terrain, wildlife, metadata, attn_mask, global_feats, targets in train_loader:
            terrain = terrain.to(device)
            wildlife = wildlife.to(device)
            metadata = metadata.to(device)
            attn_mask = attn_mask.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(terrain, wildlife, metadata, attn_mask, global_feats)
            loss = criterion(preds, targets)
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_size = targets.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / max(1, train_samples)
        train_rmse = math.sqrt(train_loss)

        # ── Validate ──
        val_loss = 0.0
        val_rmse = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for terrain, wildlife, metadata, attn_mask, global_feats, targets in val_loader:
                    terrain = terrain.to(device)
                    wildlife = wildlife.to(device)
                    metadata = metadata.to(device)
                    attn_mask = attn_mask.to(device)
                    global_feats = global_feats.to(device)
                    targets = targets.to(device)

                    preds = model(terrain, wildlife, metadata, attn_mask, global_feats)
                    loss = criterion(preds, targets)

                    batch_size = targets.shape[0]
                    val_loss_sum += loss.item() * batch_size
                    val_samples += batch_size

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

        # Save checkpoint if validation improved (or every epoch if no val set)
        current_loss = val_loss if val_loader is not None else train_loss
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            save_checkpoint(model, checkpoint_path)

    return best_val_loss


def save_checkpoint(model, path):
    """Save model state dict + config for reproducible loading."""
    config = {
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': model.n_layers,
        'dim_feedforward': model.dim_feedforward,
        'num_global': model.num_global,
        'terrain_embed_dim': model.terrain_embed.embedding_dim,
        'wildlife_embed_dim': model.wildlife_embed.embedding_dim,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, path)


def load_checkpoint(path, device):
    """Load model from checkpoint, reconstructing architecture from saved config."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CascadiaTransformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        num_global=config['num_global'],
        terrain_embed_dim=config['terrain_embed_dim'],
        wildlife_embed_dim=config['wildlife_embed_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description='Train encoder-only transformer for Cascadia remaining-score prediction')
    parser.add_argument('--samples', type=str, default=None,
                        help='Path to tile-token binary data file (TILE format)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Peak learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay (default: 0.01)')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer hidden dimension (default: 128)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of transformer layers (default: 3)')
    parser.add_argument('--dim-feedforward', type=int, default=512,
                        help='Feedforward dimension in transformer (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size (default: 512)')
    parser.add_argument('--out', type=str, default='transformer_weights.pt',
                        help='Output checkpoint path (default: transformer_weights.pt)')
    parser.add_argument('--init-weights', type=str, default=None,
                        help='Load initial weights from a previous checkpoint')
    parser.add_argument('--val-split', type=float, default=0.05,
                        help='Fraction of data for validation (default: 0.05)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Use synthetic data (100 samples) to test the pipeline')
    parser.add_argument('--benchmark-games', type=int, default=0,
                        help='After training, run a benchmark (requires Rust integration)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes (default: 0 = main process)')
    args = parser.parse_args()

    if not args.dry_run and args.samples is None:
        parser.error("--samples is required unless --dry-run is set")

    # Force line-buffered stdout
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    device = get_device()
    print(f"=== Cascadia Transformer Training ===")
    print(f"  Device: {device}")
    print(f"  Architecture: d_model={args.d_model}, heads={args.n_heads}, "
          f"layers={args.n_layers}, ff={args.dim_feedforward}")
    print(f"  Training: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, weight_decay={args.weight_decay}")
    print()

    # ── Load or generate data ──
    if args.dry_run:
        tiles_list, globals_list, targets = generate_synthetic_data(100)
    else:
        tiles_list, globals_list, targets = load_tile_samples(args.samples)

    if len(tiles_list) == 0:
        print("ERROR: No samples loaded.")
        return

    num_global = len(globals_list[0])

    # ── Train/val split ──
    n = len(tiles_list)
    indices = np.random.default_rng(42).permutation(n)
    n_val = max(1, int(n * args.val_split)) if n > 10 else 0

    if n_val > 0:
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_dataset = TileTokenDataset(
            [tiles_list[i] for i in train_idx],
            [globals_list[i] for i in train_idx],
            targets[train_idx],
        )
        val_dataset = TileTokenDataset(
            [tiles_list[i] for i in val_idx],
            [globals_list[i] for i in val_idx],
            targets[val_idx],
        )
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = TileTokenDataset(tiles_list, globals_list, targets)
        val_dataset = None
        print(f"  Train: {len(train_dataset)} (no validation split)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tiles,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_tiles,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )

    # ── Build model ──
    if args.init_weights and os.path.exists(args.init_weights):
        print(f"  Loading checkpoint: {args.init_weights}")
        model = load_checkpoint(args.init_weights, device)
        print(f"  Loaded d_model={model.d_model}, layers={model.n_layers}, "
              f"heads={model.n_heads}")
    else:
        model = CascadiaTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_global=num_global,
        )

    model = model.to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print()

    # ── Train ──
    t0 = time.time()
    best_loss = train(
        model, train_loader, val_loader, optimizer, scheduler,
        device, args.epochs, args.out,
    )
    total_time = time.time() - t0
    best_rmse = math.sqrt(best_loss) if best_loss < float('inf') else float('inf')

    print()
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best RMSE: {best_rmse:.3f}")
    print(f"  Checkpoint saved to: {args.out}")

    # ── Benchmark ──
    if args.benchmark_games > 0:
        print()
        print(f"  Benchmark ({args.benchmark_games} games) requires Rust integration.")
        print(f"  To benchmark, export weights and run:")
        print(f"    cargo run --release --bin cascadia-cli -- {args.benchmark_games} "
              f"--transformer --weights {args.out}")


if __name__ == '__main__':
    main()
