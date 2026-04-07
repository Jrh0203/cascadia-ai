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

class NNUEDataset(Dataset):
    def __init__(self, features_list, targets, num_features):
        self.num_features = num_features
        self.packed_width = (num_features + 7) // 8
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # Bit-pack: 324K × 959 bytes = ~300MB instead of 10GB
        print(f"  Bit-packing {len(features_list)} samples ({num_features} features)...")
        t0 = time.time()
        packed_np = np.zeros((len(features_list), self.packed_width), dtype=np.uint8)
        for i, f in enumerate(features_list):
            for fi in f:
                if fi < num_features:
                    packed_np[i, fi >> 3] |= (1 << (fi & 7))
        self.packed = torch.from_numpy(packed_np)
        print(f"  Done in {time.time()-t0:.1f}s ({self.packed.nbytes / 1e6:.0f} MB)")

        # Pre-compute unpack table for fast batch unpacking
        self._unpack_bits = torch.arange(8, dtype=torch.uint8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Unpack bits to float32 on the fly (only 7670 floats per sample)
        packed = self.packed[idx]
        bits = packed.unsqueeze(-1).bitwise_right_shift(self._unpack_bits).bitwise_and(1)
        dense = bits.reshape(-1)[:self.num_features].float()
        return dense, self.targets[idx]


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

    # Load data
    features_list, targets = load_mce_samples(args.samples)
    num_features = args.num_features

    # Clamp features to valid range
    for f in features_list:
        for i in range(len(f)):
            if f[i] >= num_features:
                f[i] = num_features - 1

    dataset = NNUEDataset(features_list, targets, num_features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
    parser.add_argument('--out', default='nnue_weights_pytorch.bin', help='Output weights file')
    args = parser.parse_args()
    train(args)
