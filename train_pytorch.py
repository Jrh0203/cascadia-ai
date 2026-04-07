"""PyTorch NNUE training with MPS (Apple Silicon GPU) acceleration.

Reads MCE policy samples from our binary format, trains the same NNUE
architecture, and exports weights compatible with the Rust inference code.

Usage:
    # Train from scratch (default 512->64)
    python3 train_pytorch.py --samples mce_policy_samples.bin --epochs 50 --lr 0.001

    # Train large net (1024->128)
    python3 train_pytorch.py --samples mce_policy_samples.bin --epochs 50 --lr 0.001 --hidden1 1024 --hidden2 128

    # Resume from existing weights
    python3 train_pytorch.py --samples mce_policy_samples.bin --epochs 50 --init-weights nnue_weights_mce93.bin

    # Export-only (convert Rust weights to PyTorch and back)
    python3 train_pytorch.py --init-weights nnue_weights_mce93.bin --epochs 0 --out nnue_test.bin
"""

import argparse
import os
import struct
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─── Load MCE samples from our binary format ───

def load_mce_samples(path):
    """Load samples from MCEP binary format. Returns (features_list, targets)."""
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    if data[:4] != b'MCEP':
        raise ValueError(f"Bad magic: {data[:4]}")
    pos = 4

    features_list = []
    targets = []
    while pos + 2 <= len(data):
        nf = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        if nf > 1024 or pos + nf * 2 + 4 > len(data):
            break
        feats = []
        for _ in range(nf):
            feats.append(struct.unpack_from('<H', data, pos)[0])
            pos += 2
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4
        features_list.append(feats)
        targets.append(target)

    print(f"Loaded {len(features_list)} samples from {path}")
    return features_list, targets


# ─── Sparse NNUE Dataset ───

# ─── Online Augmentation (rotation + translation) ───

GRID_DIM = 21
GRID_CENTER = 10

# Feature block boundaries (must match nnue.rs)
FPC = 11  # FEATURES_PER_CELL
CELL_END = 441 * FPC  # 4851
PHASE_END = CELL_END + 110  # 4961
WL_PAIR_STATES = 49
WL_PAIR_END = PHASE_END + 3 * WL_PAIR_STATES  # 5108
PATTERN_END = WL_PAIR_END + 89  # 5197
BAG_END = PATTERN_END + 55  # 5252
OPP_HAB_END = BAG_END + 55  # 5307
ALLOWED_WL_PC = 5
ALLOWED_END = OPP_HAB_END + 441 * ALLOWED_WL_PC  # 7512
EXT_WL_END = ALLOWED_END + 50  # 7562
TERRAIN_PAIR_STATES = 36
TERRAIN_PAIR_END = EXT_WL_END + 3 * TERRAIN_PAIR_STATES  # 7670


def build_cell_remap(dq, dr, rot):
    """Build cell index remapping table for translation (dq,dr) + rotation (0,1,2).
    Returns array of 441 entries: new_idx or -1 if out of bounds."""
    table = np.full(441, -1, dtype=np.int32)
    for idx in range(441):
        q = (idx // GRID_DIM) - GRID_CENTER
        r = (idx % GRID_DIM) - GRID_CENTER
        # Translate
        q2, r2 = q + dq, r + dr
        # Rotate
        if rot == 1:
            q2, r2 = -q2 - r2, q2
        elif rot == 2:
            q2, r2 = r2, -q2 - r2
        col = q2 + GRID_CENTER
        row = r2 + GRID_CENTER
        if 0 <= col < GRID_DIM and 0 <= row < GRID_DIM:
            table[idx] = col * GRID_DIM + row
        else:
            table[idx] = -1
    return table


# Pairwise pair swap table: PAIR_SWAP[dir_shift][dir] = should_swap
# Under rotation, some line directions become reversed, flipping the pair order.
PAIR_SWAP = [
    [False, False, False],  # dir_shift=0: identity
    [True,  True,  False],  # dir_shift=1 (120° CW): dirs 0,1 swap
    [False, True,  True],   # dir_shift=2 (240° CW): dirs 1,2 swap
]


def build_all_transforms():
    """Pre-compute all 75 transform tables (3 rotations × 25 translations)."""
    transforms = []
    for rot in range(3):
        for dq in range(-2, 3):
            for dr in range(-2, 3):
                if rot == 0 and dq == 0 and dr == 0:
                    continue  # skip identity
                table = build_cell_remap(dq, dr, rot)
                transforms.append((table, rot))
    # Add identity as first entry (no transform)
    identity = build_cell_remap(0, 0, 0)
    transforms.insert(0, (identity, 0))
    return transforms


# Pre-compute all transforms at module load
ALL_TRANSFORMS = build_all_transforms()


def apply_transform(features, cell_table, dir_shift):
    """Apply a rotation+translation transform to a sparse feature list.
    Returns transformed features or None if any cell goes out of bounds."""
    result = []
    for fi in features:
        if fi < CELL_END:
            cell_idx = fi // FPC
            offset = fi % FPC
            new_cell = cell_table[cell_idx]
            if new_cell < 0:
                return None
            result.append(new_cell * FPC + offset)
        elif fi < PHASE_END:
            result.append(fi)
        elif fi < WL_PAIR_END:
            rel = fi - PHASE_END
            d = rel // WL_PAIR_STATES
            ps = rel % WL_PAIR_STATES
            if PAIR_SWAP[dir_shift][d]:
                my, n = ps // 7, ps % 7
                ps = n * 7 + my
            result.append(PHASE_END + ((d + dir_shift) % 3) * WL_PAIR_STATES + ps)
        elif fi < PATTERN_END:
            result.append(fi)
        elif fi < OPP_HAB_END:
            result.append(fi)
        elif fi < ALLOWED_END:
            rel = fi - OPP_HAB_END
            cell_idx = rel // ALLOWED_WL_PC
            offset = rel % ALLOWED_WL_PC
            new_cell = cell_table[cell_idx]
            if new_cell < 0:
                return None
            result.append(OPP_HAB_END + new_cell * ALLOWED_WL_PC + offset)
        elif fi < EXT_WL_END:
            result.append(fi)
        elif fi < TERRAIN_PAIR_END:
            rel = fi - EXT_WL_END
            d = rel // TERRAIN_PAIR_STATES
            ps = rel % TERRAIN_PAIR_STATES
            if PAIR_SWAP[dir_shift][d]:
                my, n = ps // 6, ps % 6
                ps = n * 6 + my
            result.append(EXT_WL_END + ((d + dir_shift) % 3) * TERRAIN_PAIR_STATES + ps)
        else:
            result.append(fi)
    return result


class NNUEDatasetMCEP(Dataset):
    """Load from MCEP binary format with online augmentation.
    Each epoch, every sample gets a random rotation+translation applied."""
    def __init__(self, features_list, targets, num_features, augment=True):
        self.num_features = num_features
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.augment = augment
        # Store sparse features as numpy arrays for vectorized augmentation
        self.features_np = [np.array(f, dtype=np.int32) for f in features_list]

        # Pre-compute bit-packed numpy array for fast batch-level unpacking
        self.packed_np = None
        self._unpack_bits = torch.arange(8, dtype=torch.uint8)
        self.packed_width = (num_features + 7) // 8
        if not augment:
            print(f"  Bit-packing {len(features_list)} samples for fast loading...")
            t0 = time.time()
            self.packed_np = np.zeros((len(features_list), self.packed_width), dtype=np.uint8)
            for i, f in enumerate(self.features_np):
                for fi in f:
                    if fi < num_features:
                        self.packed_np[i, fi >> 3] |= (1 << (fi & 7))
            # Pre-compute targets as numpy too
            self.targets_np = np.array(targets, dtype=np.float32)
            print(f"  Done in {time.time()-t0:.1f}s ({self.packed_np.nbytes / 1e9:.1f} GB)")

        # Pre-compute feature index remapping tables for all transforms
        if augment:
            print(f"  Building {len(ALL_TRANSFORMS)} augmentation remap tables...")
            t0 = time.time()
            self.remap_tables = []
            for cell_table, rot in ALL_TRANSFORMS:
                remap = np.full(num_features, -1, dtype=np.int32)
                for old_fi in range(num_features):
                    result = _remap_single_feature(old_fi, cell_table, rot, num_features)
                    if result >= 0:
                        remap[old_fi] = result
                self.remap_tables.append(remap)
            print(f"  Done in {time.time()-t0:.1f}s ({len(self.remap_tables)} transforms)")


        print(f"  {len(features_list)} samples, augment={'75x' if augment else 'off'}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if not self.augment and self.packed_np is not None:
            # Return raw index — batch unpacking happens in collate_packed
            return idx, 0  # dummy second value

        dense = torch.zeros(self.num_features, dtype=torch.float32)
        feats = self.features_np[idx]
        feats = feats[feats < self.num_features]

        if self.augment:
            t_idx = torch.randint(len(self.remap_tables), (1,)).item()
            if t_idx == 0:
                dense[feats] = 1.0
            else:
                new_feats = self.remap_tables[t_idx][feats]
                if np.any(new_feats < 0):
                    dense[feats] = 1.0
                else:
                    dense[new_feats] = 1.0
        else:
            dense[feats] = 1.0

        return dense, self.targets[idx]

    def collate_packed(self, batch):
        """Batch-level bit unpacking — one tensor operation for the whole batch."""
        indices = torch.tensor([b[0] for b in batch], dtype=torch.long)
        # Grab packed rows as one numpy slice → torch tensor
        packed_batch = torch.from_numpy(self.packed_np[indices.numpy()])
        targets = torch.from_numpy(self.targets_np[indices.numpy()])
        # Unpack all bits at once: [batch, packed_width] → [batch, packed_width, 8] → [batch, packed_width*8]
        bits = packed_batch.unsqueeze(-1).bitwise_right_shift(self._unpack_bits).bitwise_and(1)
        dense = bits.reshape(len(batch), -1)[:, :self.num_features].float()
        return dense, targets


def _remap_single_feature(fi, cell_table, rot, num_features):
    """Remap a single feature index through a transform. Returns new index or -1."""
    dir_shift = rot
    if fi < CELL_END:
        cell_idx = fi // FPC
        offset = fi % FPC
        new_cell = cell_table[cell_idx]
        if new_cell < 0: return -1
        return new_cell * FPC + offset
    elif fi < PHASE_END:
        return fi
    elif fi < WL_PAIR_END:
        rel = fi - PHASE_END
        d = rel // WL_PAIR_STATES
        ps = rel % WL_PAIR_STATES
        if PAIR_SWAP[dir_shift][d]:
            my, n = ps // 7, ps % 7
            ps = n * 7 + my
        return PHASE_END + ((d + dir_shift) % 3) * WL_PAIR_STATES + ps
    elif fi < PATTERN_END:
        return fi
    elif fi < OPP_HAB_END:
        return fi
    elif fi < ALLOWED_END:
        rel = fi - OPP_HAB_END
        cell_idx = rel // ALLOWED_WL_PC
        offset = rel % ALLOWED_WL_PC
        new_cell = cell_table[cell_idx]
        if new_cell < 0: return -1
        return OPP_HAB_END + new_cell * ALLOWED_WL_PC + offset
    elif fi < EXT_WL_END:
        return fi
    elif fi < TERRAIN_PAIR_END:
        rel = fi - EXT_WL_END
        d = rel // TERRAIN_PAIR_STATES
        ps = rel % TERRAIN_PAIR_STATES
        if PAIR_SWAP[dir_shift][d]:
            my, n = ps // 6, ps % 6
            ps = n * 6 + my
        return EXT_WL_END + ((d + dir_shift) % 3) * TERRAIN_PAIR_STATES + ps
    else:
        return fi


class NNUEDatasetExported(Dataset):
    """Memory-mapped dataset from --export-pytorch binary format.
    File format: u32 num_samples, u32 num_features, then per sample:
    packed_bytes (ceil(num_features/8)) + f32 target."""
    def __init__(self, path):
        import mmap
        self.file = open(path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.num_samples = struct.unpack_from('<I', self.mm, 0)[0]
        self.num_features = struct.unpack_from('<I', self.mm, 4)[0]
        self.packed_width = (self.num_features + 7) // 8
        self.record_size = self.packed_width + 4  # packed features + f32 target
        self.data_offset = 8  # after header
        self._unpack_bits = torch.arange(8, dtype=torch.uint8)
        print(f"  Memory-mapped {self.num_samples} samples ({self.num_features} features, "
              f"{self.num_samples * self.record_size / 1e9:.1f} GB on disk)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        offset = self.data_offset + idx * self.record_size
        packed_bytes = self.mm[offset:offset + self.packed_width]
        target_bytes = self.mm[offset + self.packed_width:offset + self.record_size]
        packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8)
        target = struct.unpack('<f', target_bytes)[0]
        bits = packed.unsqueeze(-1).bitwise_right_shift(self._unpack_bits).bitwise_and(1)
        dense = bits.reshape(-1)[:self.num_features].float()
        return dense, torch.tensor(target, dtype=torch.float32)

    def __del__(self):
        self.mm.close()
        self.file.close()


def collate_sparse(batch):
    """Custom collate: convert sparse features to dense binary vectors."""
    features, targets = zip(*batch)
    num_features = batch[0][0].max().item() + 1  # rough estimate
    # Build dense binary matrix
    batch_size = len(features)
    # Find actual max feature index
    max_idx = max(f.max().item() for f in features if len(f) > 0)
    dense = torch.zeros(batch_size, max_idx + 1, dtype=torch.float32)
    for i, f in enumerate(features):
        dense[i].scatter_(0, f, 1.0)
    return dense, torch.stack(list(targets))


# ─── NNUE Model ───

class NNUE(nn.Module):
    def __init__(self, num_features, hidden1=512, hidden2=64):
        super().__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(num_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        # x is dense binary vector [batch, num_features]
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)


# ─── Weight I/O (compatible with Rust format) ───

def load_rust_weights(path, num_features, hidden1, hidden2):
    """Load weights from Rust NNUE binary format."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'NNUE':
            raise ValueError(f"Bad magic: {magic}")
        ver = f.read(4)
        data = f.read()

    floats = []
    for i in range(0, len(data), 4):
        floats.append(struct.unpack_from('<f', data, i)[0])

    # Detect how many features the file has
    rest_size = hidden1 + hidden1 * hidden2 + hidden2 + hidden2 + 1
    file_features = (len(floats) - rest_size) // hidden1

    pos = 0
    # W1: [file_features, hidden1] row-major
    w1 = np.array(floats[pos:pos + file_features * hidden1], dtype=np.float32).reshape(file_features, hidden1)
    pos += file_features * hidden1
    # Pad if file has fewer features
    if file_features < num_features:
        w1 = np.pad(w1, ((0, num_features - file_features), (0, 0)))

    b1 = np.array(floats[pos:pos + hidden1], dtype=np.float32)
    pos += hidden1

    # W2: [hidden1, hidden2] — but Rust file may have different hidden sizes
    file_h1 = hidden1  # assume same for now
    file_h2 = hidden2
    w2 = np.array(floats[pos:pos + file_h1 * file_h2], dtype=np.float32).reshape(file_h1, file_h2)
    pos += file_h1 * file_h2

    b2 = np.array(floats[pos:pos + file_h2], dtype=np.float32)
    pos += file_h2

    w3 = np.array(floats[pos:pos + file_h2], dtype=np.float32).reshape(1, file_h2)
    pos += file_h2

    b3 = np.array([floats[pos]], dtype=np.float32)

    return w1, b1, w2, b2, w3, b3


def save_rust_weights(path, model):
    """Save weights in Rust NNUE binary format."""
    with open(path, 'wb') as f:
        f.write(b'NNUE')
        f.write(struct.pack('<I', 1))

        # W1: [num_features, hidden1]
        w1 = model.fc1.weight.data.t().cpu().numpy()  # Linear stores [out, in], we want [in, out]
        f.write(w1.astype(np.float32).tobytes())
        f.write(model.fc1.bias.data.cpu().numpy().astype(np.float32).tobytes())

        # W2: [hidden1, hidden2]
        w2 = model.fc2.weight.data.t().cpu().numpy()
        f.write(w2.astype(np.float32).tobytes())
        f.write(model.fc2.bias.data.cpu().numpy().astype(np.float32).tobytes())

        # W3: [hidden2]
        w3 = model.fc3.weight.data.cpu().numpy().flatten()
        f.write(w3.astype(np.float32).tobytes())
        f.write(model.fc3.bias.data.cpu().numpy().astype(np.float32).tobytes())

    print(f"Saved weights to {path}")


def load_weights_into_model(model, w1, b1, w2, b2, w3, b3):
    """Load numpy weights into PyTorch model."""
    with torch.no_grad():
        # Linear weight is [out_features, in_features]
        model.fc1.weight.copy_(torch.from_numpy(w1.T[:model.fc1.weight.shape[0], :model.fc1.weight.shape[1]]))
        model.fc1.bias.copy_(torch.from_numpy(b1[:model.fc1.bias.shape[0]]))
        h1 = min(w2.shape[0], model.fc2.weight.shape[1])
        h2 = min(w2.shape[1], model.fc2.weight.shape[0])
        model.fc2.weight[:h2, :h1] = torch.from_numpy(w2[:h1, :h2].T)
        model.fc2.bias[:h2] = torch.from_numpy(b2[:h2])
        h2b = min(w3.shape[1], model.fc3.weight.shape[1])
        model.fc3.weight[0, :h2b] = torch.from_numpy(w3[0, :h2b])
        model.fc3.bias.copy_(torch.from_numpy(b3))


# ─── Training ───

def train(args):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data — use exported file if it exists, otherwise raw MCEP
    num_features = args.num_features
    exported_path = args.samples.replace('.bin', '') + '_exported.bin'
    if args.exported:
        exported_path = args.exported

    if os.path.exists(exported_path) and not args.samples.endswith('_exported.bin'):
        print(f"Using pre-exported augmented data: {exported_path}")
        dataset = NNUEDatasetExported(exported_path)
        num_features = dataset.num_features
    else:
        features_list, targets = load_mce_samples(args.samples)
        for f in features_list:
            for i in range(len(f)):
                if f[i] >= num_features:
                    f[i] = num_features - 1
        dataset = NNUEDatasetMCEP(features_list, targets, num_features, augment=not args.no_augment)

    collate_fn = dataset.collate_packed if (hasattr(dataset, 'packed_np') and dataset.packed_np is not None) else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Model
    model = NNUE(num_features, args.hidden1, args.hidden2).to(device)
    print(f"Architecture: {num_features} -> {args.hidden1} -> {args.hidden2} -> 1")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load initial weights if provided
    if args.init_weights:
        try:
            w1, b1, w2, b2, w3, b3 = load_rust_weights(
                args.init_weights, num_features, args.hidden1, args.hidden2)
            load_weights_into_model(model, w1, b1, w2, b2, w3, b3)
            print(f"Loaded initial weights from {args.init_weights}")
        except Exception as e:
            print(f"Warning: could not load weights: {e}. Starting fresh.")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    warmup_epochs = min(3, args.epochs)

    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        total_loss = 0.0
        num_samples = 0
        epoch_start = time.time()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]
            num_samples += batch_x.shape[0]

        if epoch >= warmup_epochs:
            scheduler.step()

        rmse = (total_loss / num_samples) ** 0.5
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs}: RMSE={rmse:.4f} lr={current_lr:.6f} ({elapsed:.1f}s)")

        # Checkpoint every epoch
        save_rust_weights(args.out, model)

    print(f"\nTraining complete. Final weights: {args.out}")


def collate_sparse_fixed(batch, num_features):
    """Collate sparse features into fixed-size dense binary vectors."""
    features, targets = zip(*batch)
    batch_size = len(features)
    dense = torch.zeros(batch_size, num_features, dtype=torch.float32)
    for i, f in enumerate(features):
        f_clamped = f[f < num_features]  # safety clamp
        if len(f_clamped) > 0:
            dense[i].scatter_(0, f_clamped, 1.0)
    return dense, torch.stack(list(targets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch NNUE training')
    parser.add_argument('--samples', default='mce_policy_samples.bin', help='Training data file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--num-features', type=int, default=7670)
    parser.add_argument('--init-weights', default=None, help='Initial weights (Rust NNUE format)')
    parser.add_argument('--exported', default=None, help='Pre-exported augmented data (from --export-pytorch)')
    parser.add_argument('--no-augment', action='store_true', help='Disable online augmentation')
    parser.add_argument('--out', default='nnue_weights_pytorch.bin', help='Output weights file')
    args = parser.parse_args()
    train(args)
