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

def load_mce_samples(path, return_aux=False):
    """Load samples from MCEP (v1) or MCV2 (v2 with aux targets) format.

    Returns:
        (features_list, targets) if return_aux=False (backward compat)
        (features_list, targets, aux_bear, aux_salmon) if return_aux=True
    """
    with open(path, 'rb') as f:
        data = f.read()

    pos = 0
    if data[:4] == b'MCV2':
        is_v2 = True
    elif data[:4] == b'MCEP':
        is_v2 = False
    else:
        raise ValueError(f"Bad magic: {data[:4]}")
    pos = 4
    extra = 8 if is_v2 else 0

    features_list = []
    targets = []
    aux_bear = []
    aux_salmon = []
    while pos + 2 <= len(data):
        nf = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        if nf > 1024 or pos + nf * 2 + 4 + extra > len(data):
            break
        feats = []
        for _ in range(nf):
            feats.append(struct.unpack_from('<H', data, pos)[0])
            pos += 2
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4
        if is_v2:
            ab = struct.unpack_from('<f', data, pos)[0]
            pos += 4
            asm = struct.unpack_from('<f', data, pos)[0]
            pos += 4
        else:
            ab = 0.0
            asm = 0.0
        features_list.append(feats)
        targets.append(target)
        aux_bear.append(ab)
        aux_salmon.append(asm)

    fmt = "MCV2" if is_v2 else "MCEP"
    print(f"Loaded {len(features_list)} samples from {path} ({fmt})")
    if return_aux:
        return features_list, targets, aux_bear, aux_salmon
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
    Each epoch, every sample gets a random rotation+translation applied.

    If aux_bear and aux_salmon are provided, __getitem__ returns 4-tuples
    (features, value_target, bear_target, salmon_target) for multi-task training.
    """
    def __init__(self, features_list, targets, num_features, augment=True,
                 aux_bear=None, aux_salmon=None):
        self.num_features = num_features
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.has_aux = aux_bear is not None and aux_salmon is not None
        if self.has_aux:
            self.aux_bear = torch.tensor(aux_bear, dtype=torch.float32)
            self.aux_salmon = torch.tensor(aux_salmon, dtype=torch.float32)
        self.augment = augment
        # Store sparse features as numpy arrays for vectorized augmentation
        self.features_np = [np.array(f, dtype=np.int32) for f in features_list]

        # Pre-compute bit-packed numpy array for fast batch-level unpacking
        self.packed_np = None
        self._unpack_bits = torch.arange(8, dtype=torch.uint8)
        self.packed_width = (num_features + 7) // 8
        if not augment:
            print(f"  Bit-packing {len(features_list)} samples for fast loading (chunked packbits)...")
            t0 = time.time()
            self.packed_np = np.zeros((len(features_list), self.packed_width), dtype=np.uint8)

            # Chunked dense+packbits approach: build a dense [chunk, num_features]
            # uint8 matrix in chunks, then use np.packbits which is fully vectorized
            # in C. For 2M samples × 10561 features in chunks of 10K = ~100MB per
            # chunk, ~200 chunks total. Much faster than np.bitwise_or.at.
            chunk_size = 10000
            n_samples = len(features_list)
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                dense = np.zeros((chunk_end - chunk_start, num_features), dtype=np.uint8)
                for i in range(chunk_start, chunk_end):
                    valid = self.features_np[i][self.features_np[i] < num_features]
                    dense[i - chunk_start, valid] = 1
                # bitorder='little' to match the Rust feature encoding (bit fi&7 in byte fi>>3)
                self.packed_np[chunk_start:chunk_end] = np.packbits(dense, axis=1, bitorder='little')

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
        idx_np = indices.numpy()
        # Grab packed rows as one numpy slice → torch tensor
        packed_batch = torch.from_numpy(self.packed_np[idx_np])
        targets = torch.from_numpy(self.targets_np[idx_np])
        # Unpack all bits at once: [batch, packed_width] → [batch, packed_width, 8] → [batch, packed_width*8]
        bits = packed_batch.unsqueeze(-1).bitwise_right_shift(self._unpack_bits).bitwise_and(1)
        dense = bits.reshape(len(batch), -1)[:, :self.num_features].float()
        if self.has_aux:
            aux_b = self.aux_bear[indices]
            aux_s = self.aux_salmon[indices]
            return dense, targets, aux_b, aux_s
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
        self.fc3 = nn.Linear(hidden2, 1)          # value head
        self.fc3_policy = nn.Linear(hidden2, 1)    # policy head
        # Auxiliary heads (v4 multi-task training):
        # fc3_aux_bear predicts final bear pair count
        # fc3_aux_salmon predicts final longest salmon chain length
        # These are TRAINING ONLY — discarded when saving Rust weights for inference.
        self.fc3_aux_bear = nn.Linear(hidden2, 1)
        self.fc3_aux_salmon = nn.Linear(hidden2, 1)
        # Small random init so gradients flow through shared layers
        for head in (self.fc3_policy, self.fc3_aux_bear, self.fc3_aux_salmon):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        # x is dense binary vector [batch, num_features]
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)

    def forward_dual(self, x):
        """Returns (value, policy_logit) — both [batch]."""
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        value = self.fc3(h2).squeeze(-1)
        policy = self.fc3_policy(h2).squeeze(-1)
        return value, policy

    def forward_multi(self, x):
        """Returns (value, aux_bear, aux_salmon) — all [batch]. For v4 multi-task training."""
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return (
            self.fc3(h2).squeeze(-1),
            self.fc3_aux_bear(h2).squeeze(-1),
            self.fc3_aux_salmon(h2).squeeze(-1),
        )


def save_policy_net_rust(path, model):
    """Save PolicyNet weights in a simple binary format for Rust loading.
    Format: magic 'PLCY', u32 version, u32 hidden1, u32 hidden2, then weights."""
    with open(path, 'wb') as f:
        f.write(b'PLCY')
        f.write(struct.pack('<I', 1))  # version
        h1 = model.fc1.weight.shape[0]
        h2 = model.fc2.weight.shape[0]
        nf = model.fc1.weight.shape[1]
        f.write(struct.pack('<I', nf))
        f.write(struct.pack('<I', h1))
        f.write(struct.pack('<I', h2))

        w1 = model.fc1.weight.data.t().cpu().numpy()
        f.write(w1.astype(np.float32).tobytes())
        f.write(model.fc1.bias.data.cpu().numpy().astype(np.float32).tobytes())

        w2 = model.fc2.weight.data.t().cpu().numpy()
        f.write(w2.astype(np.float32).tobytes())
        f.write(model.fc2.bias.data.cpu().numpy().astype(np.float32).tobytes())

        w3 = model.fc3.weight.data.cpu().numpy().flatten()
        f.write(w3.astype(np.float32).tobytes())
        f.write(model.fc3.bias.data.cpu().numpy().astype(np.float32).tobytes())

    print(f"Saved PolicyNet Rust weights to {path} ({nf}→{h1}→{h2}→1)")


class PolicyNet(nn.Module):
    """Separate policy network for candidate ranking. Independent from value NNUE."""
    def __init__(self, num_features, hidden1=256, hidden2=64):
        super().__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(num_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
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
    pos += 1

    # Policy head (optional — backward compatible)
    if pos + file_h2 + 1 <= len(floats):
        w3_policy = np.array(floats[pos:pos + file_h2], dtype=np.float32).reshape(1, file_h2)
        pos += file_h2
        b3_policy = np.array([floats[pos]], dtype=np.float32)
    else:
        w3_policy = np.zeros((1, file_h2), dtype=np.float32)
        b3_policy = np.zeros(1, dtype=np.float32)

    return w1, b1, w2, b2, w3, b3, w3_policy, b3_policy


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

        # W3: [hidden2] (value head)
        w3 = model.fc3.weight.data.cpu().numpy().flatten()
        f.write(w3.astype(np.float32).tobytes())
        f.write(model.fc3.bias.data.cpu().numpy().astype(np.float32).tobytes())

        # Policy head: w3_policy + b3_policy
        w3p = model.fc3_policy.weight.data.cpu().numpy().flatten()
        f.write(w3p.astype(np.float32).tobytes())
        f.write(model.fc3_policy.bias.data.cpu().numpy().astype(np.float32).tobytes())

    print(f"Saved weights to {path}")


def load_weights_into_model(model, w1, b1, w2, b2, w3, b3, w3_policy=None, b3_policy=None):
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
        # Policy head
        if w3_policy is not None:
            h2p = min(w3_policy.shape[1], model.fc3_policy.weight.shape[1])
            model.fc3_policy.weight[0, :h2p] = torch.from_numpy(w3_policy[0, :h2p])
            model.fc3_policy.bias.copy_(torch.from_numpy(b3_policy))


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
        # Load with aux targets if requested via --use-aux
        loaded = load_mce_samples(args.samples, return_aux=args.use_aux)
        if args.use_aux:
            features_list, targets, aux_bear, aux_salmon = loaded
            print(f"  Loaded aux targets: bear range [{min(aux_bear):.0f}..{max(aux_bear):.0f}], salmon range [{min(aux_salmon):.0f}..{max(aux_salmon):.0f}]")
        else:
            features_list, targets = loaded
            aux_bear = None
            aux_salmon = None
        for f in features_list:
            for i in range(len(f)):
                if f[i] >= num_features:
                    f[i] = num_features - 1
        dataset = NNUEDatasetMCEP(
            features_list, targets, num_features, augment=not args.no_augment,
            aux_bear=aux_bear, aux_salmon=aux_salmon,
        )

    collate_fn = dataset.collate_packed if (hasattr(dataset, 'packed_np') and dataset.packed_np is not None) else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Model
    model = NNUE(num_features, args.hidden1, args.hidden2).to(device)
    print(f"Architecture: {num_features} -> {args.hidden1} -> {args.hidden2} -> 1")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load initial weights if provided
    if args.init_weights:
        try:
            w1, b1, w2, b2, w3, b3, w3p, b3p = load_rust_weights(
                args.init_weights, num_features, args.hidden1, args.hidden2)
            load_weights_into_model(model, w1, b1, w2, b2, w3, b3, w3p, b3p)
            print(f"Loaded initial weights from {args.init_weights}")
        except Exception as e:
            print(f"Warning: could not load weights: {e}. Starting fresh.")

    # Optimizer + scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        warmup_epochs = min(3, args.epochs)
    else:
        # SGD — matches original Rust training behavior exactly
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = None
        warmup_epochs = 0

    criterion = nn.MSELoss()
    use_aux = args.use_aux and getattr(dataset, 'has_aux', False)
    if use_aux:
        print(f"  Multi-task training: value + bear (w={args.aux_bear_weight}) + salmon (w={args.aux_salmon_weight})")

    # Training loop
    for epoch in range(args.epochs):
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_b_loss = 0.0
        total_s_loss = 0.0
        num_samples = 0
        epoch_start = time.time()

        for batch in loader:
            if use_aux and len(batch) == 4:
                batch_x, batch_y, batch_b, batch_s = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_b = batch_b.to(device)
                batch_s = batch_s.to(device)
                value, aux_b, aux_s = model.forward_multi(batch_x)
                v_loss = criterion(value, batch_y)
                b_loss = criterion(aux_b, batch_b)
                s_loss = criterion(aux_s, batch_s)
                loss = v_loss + args.aux_bear_weight * b_loss + args.aux_salmon_weight * s_loss
                total_v_loss += v_loss.item() * batch_x.shape[0]
                total_b_loss += b_loss.item() * batch_x.shape[0]
                total_s_loss += s_loss.item() * batch_x.shape[0]
            else:
                batch_x, batch_y = batch[0], batch[1]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                total_v_loss += loss.item() * batch_x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]
            num_samples += batch_x.shape[0]

        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

        v_rmse = (total_v_loss / num_samples) ** 0.5
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start
        if use_aux:
            b_rmse = (total_b_loss / num_samples) ** 0.5
            s_rmse = (total_s_loss / num_samples) ** 0.5
            print(f"  Epoch {epoch+1}/{args.epochs}: V={v_rmse:.4f} B={b_rmse:.4f} S={s_rmse:.4f} lr={current_lr:.6f} ({elapsed:.1f}s)")
        else:
            print(f"  Epoch {epoch+1}/{args.epochs}: RMSE={v_rmse:.4f} lr={current_lr:.6f} ({elapsed:.1f}s)")

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


# ─── Policy Training (MCP2 format) ───

def load_policy_data(path):
    """Load MCP2 policy training data. Returns list of (candidates, value_target).
    Each candidate is (feature_indices, score)."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'MCP2':
        raise ValueError(f"Bad magic: {data[:4]}")
    pos = 4
    groups = []
    while pos + 6 <= len(data):
        k = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        vt = struct.unpack_from('<f', data, pos)[0]
        pos += 4
        candidates = []
        for _ in range(k):
            if pos + 2 > len(data): break
            nf = struct.unpack_from('<H', data, pos)[0]
            pos += 2
            if nf > 1024 or pos + nf * 2 + 4 > len(data): break
            feats = []
            for _ in range(nf):
                feats.append(struct.unpack_from('<H', data, pos)[0])
                pos += 2
            score = struct.unpack_from('<f', data, pos)[0]
            pos += 4
            candidates.append((feats, score))
        groups.append((candidates, vt))
    return groups


def train_policy(args):
    """Train dual-head (value + policy) from MCP2 policy data."""
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_features = args.num_features

    # Load policy data
    print(f"Loading policy data from {args.policy_data}...")
    groups = load_policy_data(args.policy_data)
    print(f"  {len(groups)} position groups")

    # Model
    model = NNUE(num_features, args.hidden1, args.hidden2).to(device)
    print(f"Architecture: {num_features} -> {args.hidden1} -> {args.hidden2} -> 1 (value + policy)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (policy head: {args.hidden2 + 1})")

    # Load initial weights
    if args.init_weights:
        try:
            w1, b1, w2, b2, w3, b3, w3p, b3p = load_rust_weights(
                args.init_weights, num_features, args.hidden1, args.hidden2)
            load_weights_into_model(model, w1, b1, w2, b2, w3, b3, w3p, b3p)
            print(f"Loaded initial weights from {args.init_weights}")
            # If policy head loaded as zeros, re-init with small random weights
            if model.fc3_policy.weight.data.abs().sum().item() < 1e-6:
                nn.init.xavier_uniform_(model.fc3_policy.weight)
                print("  Policy head was zero — re-initialized with Xavier")
        except Exception as e:
            print(f"Warning: could not load weights: {e}. Starting fresh.")

    # Flatten groups into per-candidate arrays for value training,
    # and build grouped indices for policy training
    all_features = []  # sparse feature lists
    all_value_targets = []
    all_scores = []
    group_ranges = []  # (start_idx, end_idx) for each group

    for candidates, value_target in groups:
        start = len(all_features)
        for feats, score in candidates:
            clamped = [f for f in feats if f < num_features]
            all_features.append(clamped)
            all_value_targets.append(value_target)
            all_scores.append(score)
        end = len(all_features)
        if end > start:
            group_ranges.append((start, end))

    n_samples = len(all_features)
    print(f"  {n_samples} total candidate afterstates across {len(group_ranges)} groups")
    print(f"  Avg candidates/group: {n_samples/max(len(group_ranges),1):.1f}")

    # Pre-compute bit-packed features for fast batch construction
    packed_width = (num_features + 7) // 8
    print(f"  Bit-packing {n_samples} samples ({n_samples * packed_width / 1e6:.0f} MB)...")
    t0 = time.time()
    packed_features = np.zeros((n_samples, packed_width), dtype=np.uint8)
    for i, f in enumerate(all_features):
        for fi in f:
            packed_features[i, fi >> 3] |= (1 << (fi & 7))
    all_value_targets_np = np.array(all_value_targets, dtype=np.float32)
    all_scores_np = np.array(all_scores, dtype=np.float32)
    _unpack_bits = torch.arange(8, dtype=torch.uint8)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Optimizer
    if args.freeze_shared:
        # Only train policy head, freeze everything else
        for p in model.fc1.parameters(): p.requires_grad = False
        for p in model.fc2.parameters(): p.requires_grad = False
        for p in model.fc3.parameters(): p.requires_grad = False
        optimizer = torch.optim.SGD([
            {'params': model.fc3_policy.parameters(), 'lr': args.lr},
        ])
        print("Freezing shared layers + value head — training ONLY policy head")
    elif getattr(args, 'joint', False):
        # Joint training: equal LR for all parameters (AlphaGo Zero style)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        print(f"Joint training: uniform LR={args.lr}")
    else:
        # Fine-tuning: higher LR for policy head, lower for shared layers
        shared_params = list(model.fc1.parameters()) + list(model.fc2.parameters())
        value_params = list(model.fc3.parameters())
        policy_params = list(model.fc3_policy.parameters())
        optimizer = torch.optim.SGD([
            {'params': shared_params, 'lr': args.lr * 0.1},
            {'params': value_params, 'lr': args.lr},
            {'params': policy_params, 'lr': args.lr * 10},
        ])
        print(f"Fine-tune LR: shared={args.lr*0.1:.6f}, value={args.lr:.6f}, policy={args.lr*10:.6f}")

    value_criterion = nn.MSELoss()
    policy_weight = args.policy_weight

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_correct = 0
        total_groups_seen = 0
        epoch_start = time.time()

        # Shuffle groups and process in batches
        perm = np.random.permutation(len(group_ranges))
        batch_size_groups = 64  # groups per optimizer step

        for batch_start in range(0, len(perm), batch_size_groups):
            batch_indices = perm[batch_start:batch_start + batch_size_groups]

            # Collect sample indices and group slices for this batch
            sample_indices = []
            batch_group_slices = []  # (offset_in_batch, k, group_range_idx)

            offset = 0
            for gi in batch_indices:
                start, end = group_ranges[gi]
                k = end - start
                if k < 2:
                    continue
                sample_indices.extend(range(start, end))
                batch_group_slices.append((offset, k, gi))
                offset += k

            if offset == 0:
                continue

            # Unpack bit-packed features to dense tensor (vectorized)
            packed_batch = torch.from_numpy(packed_features[sample_indices])
            bits = packed_batch.unsqueeze(-1).bitwise_right_shift(_unpack_bits).bitwise_and(1)
            dense = bits.reshape(offset, -1)[:, :num_features].float().to(device)
            vtargets = torch.from_numpy(all_value_targets_np[sample_indices]).to(device)

            # Forward all candidates at once
            values, policies = model.forward_dual(dense)

            # Value loss across all candidates
            v_loss = value_criterion(values, vtargets)

            # Policy loss: per-group cross-entropy, averaged
            p_loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_policy_groups = 0
            for (off, k, gi) in batch_group_slices:
                group_policies = policies[off:off + k]
                gstart, gend = group_ranges[gi]
                scores = torch.from_numpy(all_scores_np[gstart:gend]).to(device)

                target_dist = torch.softmax(scores / 2.0, dim=0)
                log_probs = torch.log_softmax(group_policies, dim=0)
                p_loss = p_loss - (target_dist * log_probs).sum()

                if group_policies.argmax() == scores.argmax():
                    total_correct += 1
                total_groups_seen += 1
                n_policy_groups += 1

            if n_policy_groups > 0:
                p_loss = p_loss / n_policy_groups

            loss = v_loss + policy_weight * p_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_value_loss += v_loss.item() * offset
            total_policy_loss += p_loss.item() * n_policy_groups

        value_rmse = (total_value_loss / max(len(all_features), 1)) ** 0.5
        avg_policy_loss = total_policy_loss / max(total_groups_seen, 1)
        accuracy = total_correct / max(total_groups_seen, 1) * 100
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs}: value_RMSE={value_rmse:.4f} "
              f"policy_loss={avg_policy_loss:.4f} top1_acc={accuracy:.1f}% ({elapsed:.1f}s)")

        save_rust_weights(args.out, model)

    print(f"\nPolicy training complete. Final weights: {args.out}")
    print(f"Top-1 accuracy: {accuracy:.1f}%")


def train_policy_standalone(args):
    """Train a standalone policy network (separate from value NNUE) using MCP2 data.
    Uses MCE scores as soft targets via cross-entropy."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_features = args.num_features

    print(f"Loading policy data from {args.policy_data}...")
    groups = load_policy_data(args.policy_data)
    print(f"  {len(groups)} position groups")

    # Model — separate PolicyNet, not the dual-head NNUE
    model = PolicyNet(num_features, args.hidden1, args.hidden2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PolicyNet: {num_features} -> {args.hidden1} -> {args.hidden2} -> 1")
    print(f"Parameters: {total_params:,}")

    # Load existing policy weights if provided
    if args.init_weights:
        try:
            state = torch.load(args.init_weights, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded policy weights from {args.init_weights}")
        except Exception as e:
            print(f"Warning: could not load policy weights: {e}. Starting fresh.")

    # Flatten groups
    all_features = []
    all_scores = []
    group_ranges = []

    for candidates, value_target in groups:
        start = len(all_features)
        for feats, score in candidates:
            clamped = [f for f in feats if f < num_features]
            all_features.append(clamped)
            all_scores.append(score)
        end = len(all_features)
        if end > start:
            group_ranges.append((start, end))

    n_samples = len(all_features)
    print(f"  {n_samples} candidate afterstates across {len(group_ranges)} groups")
    print(f"  Avg candidates/group: {n_samples/max(len(group_ranges),1):.1f}")

    # Bit-pack features
    packed_width = (num_features + 7) // 8
    print(f"  Bit-packing {n_samples} samples...")
    t0 = time.time()
    packed_features = np.zeros((n_samples, packed_width), dtype=np.uint8)
    for i, f in enumerate(all_features):
        for fi in f:
            packed_features[i, fi >> 3] |= (1 << (fi & 7))
    all_scores_np = np.array(all_scores, dtype=np.float32)
    _unpack_bits = torch.arange(8, dtype=torch.uint8)
    print(f"  Done in {time.time()-t0:.1f}s")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_size_groups = 128

    # Pre-compute padded tensors for vectorized loss computation
    # Find max group size and build padded score/index arrays
    max_k = max(end - start for start, end in group_ranges)
    print(f"  Max candidates per group: {max_k}, padding all groups to this size")

    # Pre-build padded score tensor and mask for all groups
    n_groups_total = len(group_ranges)
    all_padded_scores = np.full((n_groups_total, max_k), -1e9, dtype=np.float32)
    all_group_sizes = np.zeros(n_groups_total, dtype=np.int32)
    all_best_idx = np.zeros(n_groups_total, dtype=np.int64)
    # Map from group index to sample indices
    all_group_sample_indices = []
    for gi, (start, end) in enumerate(group_ranges):
        k = end - start
        all_group_sizes[gi] = k
        all_padded_scores[gi, :k] = all_scores_np[start:end]
        all_best_idx[gi] = np.argmax(all_scores_np[start:end])
        all_group_sample_indices.append(list(range(start, end)))

    all_padded_scores_t = torch.from_numpy(all_padded_scores)
    all_best_idx_t = torch.from_numpy(all_best_idx)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_top3_recall = 0
        total_top5_recall = 0
        total_top8_recall = 0
        total_groups_seen = 0
        epoch_start = time.time()

        perm = np.random.permutation(n_groups_total)

        for batch_start in range(0, len(perm), batch_size_groups):
            batch_gis = perm[batch_start:batch_start + batch_size_groups]
            # Filter groups with k < 2
            batch_gis = [gi for gi in batch_gis if all_group_sizes[gi] >= 2]
            if not batch_gis:
                continue

            n_batch = len(batch_gis)

            # Gather sample indices for all candidates, padded to max_k per group
            # Build a flat index array + reshape
            sample_indices = []
            for gi in batch_gis:
                indices = all_group_sample_indices[gi]
                k = len(indices)
                sample_indices.extend(indices)
                # Pad with index 0 (will be masked out)
                sample_indices.extend([0] * (max_k - k))

            # Unpack features: [n_batch * max_k, num_features]
            packed_batch = torch.from_numpy(packed_features[sample_indices])
            ubits = torch.arange(8, dtype=torch.uint8)
            bits = packed_batch.unsqueeze(-1).bitwise_right_shift(ubits).bitwise_and(1)
            dense = bits.reshape(n_batch * max_k, -1)[:, :num_features].float().to(device)

            # Forward: [n_batch * max_k]
            all_logits = model(dense)

            # Reshape to [n_batch, max_k]
            logits_2d = all_logits.reshape(n_batch, max_k)

            # Build mask: [n_batch, max_k], True where valid (no grad)
            sizes = torch.tensor([all_group_sizes[gi] for gi in batch_gis])
            mask = (torch.arange(max_k).unsqueeze(0) < sizes.unsqueeze(1)).to(device)

            # Scores: [n_batch, max_k] — from numpy, masked positions set to -inf
            scores_2d = torch.from_numpy(all_padded_scores[batch_gis].copy()).to(device)
            scores_2d = scores_2d.masked_fill(~mask, float('-inf'))

            # Target distribution: softmax over valid scores only
            target_dist = torch.softmax(scores_2d / 2.0, dim=1)

            # Log-softmax over logits (mask invalid positions with -inf)
            masked_logits = logits_2d.masked_fill(~mask, float('-inf'))
            log_probs = torch.log_softmax(masked_logits, dim=1)

            # Cross-entropy: only sum over valid positions (where mask is True)
            ce = -(target_dist * log_probs)
            ce = ce.masked_fill(~mask, 0.0)  # zero out NaN from 0 * -inf
            per_group_loss = ce.sum(dim=1)
            loss = per_group_loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * n_batch

            # Metrics (after backward, detached)
            with torch.no_grad():
                ml = logits_2d.masked_fill(~mask, float('-inf'))
                best_by_score = torch.tensor([all_best_idx[gi] for gi in batch_gis], device=device)
                policy_best = ml.argmax(dim=1)
                total_correct += (policy_best == best_by_score).sum().item()

                for topk_val, counter_name in [(3, 'total_top3_recall'), (5, 'total_top5_recall'), (8, 'total_top8_recall')]:
                    tk = min(topk_val, max_k)
                    _, top_indices = ml.topk(tk, dim=1)
                    hit = (top_indices == best_by_score.unsqueeze(1)).any(dim=1)
                    if counter_name == 'total_top3_recall':
                        total_top3_recall += hit.sum().item()
                    elif counter_name == 'total_top5_recall':
                        total_top5_recall += hit.sum().item()
                    else:
                        total_top8_recall += hit.sum().item()

                total_groups_seen += n_batch

        avg_loss = total_loss / max(total_groups_seen, 1)
        accuracy = total_correct / max(total_groups_seen, 1) * 100
        r3 = total_top3_recall / max(total_groups_seen, 1) * 100
        r5 = total_top5_recall / max(total_groups_seen, 1) * 100
        r8 = total_top8_recall / max(total_groups_seen, 1) * 100
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
              f"top1={accuracy:.1f}% r3={r3:.0f}% r5={r5:.0f}% r8={r8:.0f}% ({elapsed:.1f}s)")

        # Save both PyTorch checkpoint and Rust-compatible binary
        torch.save(model.state_dict(), args.out)
        rust_path = args.out.replace('.pt', '.bin')
        save_policy_net_rust(rust_path, model)

    print(f"\nStandalone policy training complete: {args.out}")
    print(f"Top-1 accuracy: {accuracy:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch NNUE training')
    subparsers = parser.add_subparsers(dest='command')

    # Value training (default)
    val_parser = subparsers.add_parser('value', help='Train value head from MCEP data')
    val_parser.add_argument('--samples', default='mce_policy_samples.bin', help='Training data file')
    val_parser.add_argument('--epochs', type=int, default=50)
    val_parser.add_argument('--lr', type=float, default=0.001)
    val_parser.add_argument('--batch-size', type=int, default=4096)
    val_parser.add_argument('--hidden1', type=int, default=512)
    val_parser.add_argument('--hidden2', type=int, default=64)
    val_parser.add_argument('--num-features', type=int, default=10561)
    val_parser.add_argument('--init-weights', default=None)
    val_parser.add_argument('--exported', default=None)
    val_parser.add_argument('--no-augment', action='store_true')
    val_parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    val_parser.add_argument('--out', default='nnue_weights_pytorch.bin')
    val_parser.add_argument('--use-aux', action='store_true',
                            help='Load aux targets from MCV2 file and train with multi-task loss')
    val_parser.add_argument('--aux-bear-weight', type=float, default=0.3)
    val_parser.add_argument('--aux-salmon-weight', type=float, default=0.3)

    # Policy training
    pol_parser = subparsers.add_parser('policy', help='Train policy+value from MCP2 data')
    pol_parser.add_argument('--policy-data', required=True, help='MCP2 policy training data')
    pol_parser.add_argument('--epochs', type=int, default=30)
    pol_parser.add_argument('--lr', type=float, default=0.0001)
    pol_parser.add_argument('--batch-size', type=int, default=64, help='Groups per batch')
    pol_parser.add_argument('--hidden1', type=int, default=512)
    pol_parser.add_argument('--hidden2', type=int, default=64)
    pol_parser.add_argument('--num-features', type=int, default=10561)
    pol_parser.add_argument('--init-weights', default=None)
    pol_parser.add_argument('--policy-weight', type=float, default=0.5, help='Weight for policy loss')
    pol_parser.add_argument('--freeze-shared', action='store_true', help='Only train policy head')
    pol_parser.add_argument('--joint', action='store_true', help='Joint training with uniform LR (AlphaGo Zero style)')
    pol_parser.add_argument('--out', default='nnue_weights_policy.bin')

    # Standalone policy network training
    sp_parser = subparsers.add_parser('policy-standalone', help='Train separate policy network')
    sp_parser.add_argument('--policy-data', required=True, help='MCP2 policy training data')
    sp_parser.add_argument('--epochs', type=int, default=30)
    sp_parser.add_argument('--lr', type=float, default=0.001)
    sp_parser.add_argument('--hidden1', type=int, default=256)
    sp_parser.add_argument('--hidden2', type=int, default=64)
    sp_parser.add_argument('--num-features', type=int, default=10561)
    sp_parser.add_argument('--init-weights', default=None, help='PyTorch checkpoint to resume from')
    sp_parser.add_argument('--out', default='policy_net.pt')

    args = parser.parse_args()
    if args.command == 'policy-standalone':
        train_policy_standalone(args)
    elif args.command == 'policy':
        train_policy(args)
    elif args.command == 'value':
        train(args)
    else:
        # Backward compatible — default to value training
        # Re-parse with old-style args
        parser2 = argparse.ArgumentParser(description='PyTorch NNUE training')
        parser2.add_argument('--samples', default='mce_policy_samples.bin')
        parser2.add_argument('--epochs', type=int, default=50)
        parser2.add_argument('--lr', type=float, default=0.001)
        parser2.add_argument('--batch-size', type=int, default=4096)
        parser2.add_argument('--hidden1', type=int, default=512)
        parser2.add_argument('--hidden2', type=int, default=64)
        parser2.add_argument('--num-features', type=int, default=10561)
        parser2.add_argument('--init-weights', default=None)
        parser2.add_argument('--exported', default=None)
        parser2.add_argument('--no-augment', action='store_true')
        parser2.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
        parser2.add_argument('--out', default='nnue_weights_pytorch.bin')
        args = parser2.parse_args()
        train(args)
