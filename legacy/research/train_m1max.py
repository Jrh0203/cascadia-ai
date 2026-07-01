"""Dispatch training jobs to a remote M1 Max via SSH.

Mirrors the modal_collect.py train_modal pattern but uses plain SSH/rsync
instead of Modal. Designed for when you have a second Mac sitting idle.

Setup (one-time, on M1 Max):
    1. System Settings → Sharing → enable "Remote Login"
    2. Install Python + deps:
        brew install python@3.11
        pip3 install torch==2.4.0 numpy
    3. (Optional) Install Tailscale on both machines for cross-network access.

Setup (one-time, from M4):
    1. ssh-copy-id <you>@<m1max-host>       # key-based auth (no passwords)
    2. ssh <you>@<m1max-host> 'mkdir -p ~/cascadia'
    3. Set the host env var or pass --host:
        export M1MAX_HOST=m1max.local       # or Tailscale hostname
        export M1MAX_USER=<you>             # optional; defaults to $USER

Usage:
    # Sanity check — SSH works
    python3 train_m1max.py check

    # Push current working tree
    python3 train_m1max.py sync

    # Train GNN (50K greedy self-play)
    python3 train_m1max.py gnn --samples tile_tokens_50k.bin --epochs 30

    # Train transformer (same data)
    python3 train_m1max.py transformer --samples tile_tokens_50k.bin --epochs 30

    # Train NNUE (compare to MPS / Modal L4)
    python3 train_m1max.py nnue --samples training_merged_iter1.bin --epochs 15

    # Arbitrary remote command
    python3 train_m1max.py exec "python3 -c 'import torch; print(torch.backends.mps.is_available())'"
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Force line-buffered stdout so tee/logs update in real time.
sys.stdout.reconfigure(line_buffering=True)

# ─── Connection config ───

HOST = os.environ.get("M1MAX_HOST", "m1max.local")
USER = os.environ.get("M1MAX_USER", os.environ.get("USER", ""))
REMOTE_DIR = os.environ.get("M1MAX_REPO", "~/cascadia")

# rsync excludes: don't sync build artifacts or huge binary files by default.
# Users can still explicitly upload data files via upload_file().
RSYNC_EXCLUDES = [
    "target/",           # Rust build output
    "__pycache__/",
    ".venv/",
    "*.bin",             # weight files, training data — upload explicitly
    "*.log",
    "*.pt",              # PyTorch checkpoints
    ".git/",
    "tandem_runs/",
    "exit_work/",
    "iter_history/",
]


def _ssh_target():
    """Return the SSH target string: [user@]host."""
    return f"{USER}@{HOST}" if USER else HOST


def _log(msg):
    """Timestamped log line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Core SSH primitives ───

def check_connection(verbose=True):
    """Verify SSH reachability and basic remote environment."""
    target = _ssh_target()
    if verbose:
        _log(f"Checking connection to {target}:{REMOTE_DIR}")

    cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", target,
           f"echo OK; uname -a; python3 --version 2>&1; "
           f"test -d {REMOTE_DIR} && echo 'REPO_EXISTS' || echo 'REPO_MISSING'"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ SSH failed: {r.stderr.strip()}")
        print(f"    Try: ssh-copy-id {target}")
        return False
    if verbose:
        for line in r.stdout.strip().splitlines():
            print(f"  {line}")

    # Check PyTorch + MPS on remote
    cmd = ["ssh", target,
           "python3 -c 'import torch; "
           "print(f\"torch={torch.__version__}, mps={torch.backends.mps.is_available()}\")' "
           "2>&1 || echo 'torch_missing'"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(f"  {r.stdout.strip()}")
    return True


def sync_code(extra_excludes=None):
    """rsync the current working tree to the remote repo dir.

    Excludes build artifacts and binary files by default. Upload those
    explicitly with `upload_file()` when needed.
    """
    target = _ssh_target()
    _log(f"Syncing code → {target}:{REMOTE_DIR}")

    cmd = ["rsync", "-az", "--delete"]
    for e in RSYNC_EXCLUDES + (extra_excludes or []):
        cmd.extend(["--exclude", e])
    # Include specific binary/log files? Add --include BEFORE exclude if needed.
    cmd.extend([".", f"{target}:{REMOTE_DIR}/"])

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ rsync failed: {r.stderr.strip()}")
        return False
    _log(f"  done in {time.time()-t0:.1f}s")
    return True


def upload_file(local_path, remote_rel_path=None, show_progress=True):
    """Upload a single file to the remote repo dir via rsync.

    remote_rel_path defaults to the basename of local_path under REMOTE_DIR.
    """
    target = _ssh_target()
    local_path = str(local_path)
    if remote_rel_path is None:
        remote_rel_path = os.path.basename(local_path)
    remote_path = f"{REMOTE_DIR}/{remote_rel_path}"

    size_mb = os.path.getsize(local_path) / 1e6
    _log(f"Uploading {local_path} ({size_mb:.1f}MB) → {target}:{remote_path}")

    cmd = ["rsync", "-az"]
    if show_progress:
        cmd.append("--info=progress2")
    cmd.extend([local_path, f"{target}:{remote_path}"])

    t0 = time.time()
    r = subprocess.run(cmd)  # let progress stream through
    if r.returncode != 0:
        print(f"  ✗ upload failed")
        return False
    _log(f"  done in {time.time()-t0:.1f}s")
    return True


def download_file(remote_rel_path, local_path=None, show_progress=True):
    """Download a file from the remote repo dir.

    local_path defaults to the basename of remote_rel_path in the CWD.
    """
    target = _ssh_target()
    if local_path is None:
        local_path = os.path.basename(remote_rel_path)
    remote_path = f"{REMOTE_DIR}/{remote_rel_path}"

    _log(f"Downloading {target}:{remote_path} → {local_path}")

    cmd = ["rsync", "-az"]
    if show_progress:
        cmd.append("--info=progress2")
    cmd.extend([f"{target}:{remote_path}", local_path])

    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  ✗ download failed")
        return False
    size_mb = os.path.getsize(local_path) / 1e6
    _log(f"  done in {time.time()-t0:.1f}s ({size_mb:.1f}MB)")
    return True


def run_remote(command, stream=True, cwd=None):
    """Run a shell command on the remote host.

    stream=True pipes stdout/stderr live to local terminal (blocks until done).
    Returns the exit code.
    """
    target = _ssh_target()
    if cwd is None:
        cwd = REMOTE_DIR
    # Use bash -lc to pick up user's PATH (pyenv, homebrew, etc.)
    full_cmd = f"cd {cwd} && {command}"
    ssh_cmd = ["ssh", target, "bash -lc", repr(full_cmd)]
    # repr() handles shell escaping. Equivalent to: bash -lc '<full_cmd>'

    if not stream:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr

    # Stream mode
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


def run_remote_background(command, log_path, cwd=None):
    """Launch a remote command detached under nohup; return the remote PID.

    Output goes to {REMOTE_DIR}/{log_path}. Retrieve it later with download_file().
    This is the closest analog to Modal's .spawn() — fire-and-forget.
    """
    target = _ssh_target()
    if cwd is None:
        cwd = REMOTE_DIR
    remote_log = f"{cwd}/{log_path}"
    # Wrap with nohup + disown so the SSH session can disconnect.
    bg_cmd = (
        f"cd {cwd} && "
        f"nohup bash -lc {repr(command)} > {remote_log} 2>&1 & "
        f"echo $!"  # print PID
    )
    ssh_cmd = ["ssh", target, bg_cmd]
    r = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ background launch failed: {r.stderr}")
        return None
    pid = r.stdout.strip()
    _log(f"  remote PID {pid}, log → {target}:{remote_log}")
    return pid


def tail_remote_log(log_path, follow=True, cwd=None):
    """Stream the contents of a remote log file to local stdout.

    Useful for monitoring a background job started with run_remote_background().
    """
    target = _ssh_target()
    if cwd is None:
        cwd = REMOTE_DIR
    remote_log = f"{cwd}/{log_path}"
    flag = "-f" if follow else ""
    ssh_cmd = ["ssh", target, f"tail {flag} {remote_log}"]
    subprocess.run(ssh_cmd)


def remote_process_alive(pid):
    """Check whether a remote PID is still running. Returns bool."""
    target = _ssh_target()
    cmd = ["ssh", target, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return "alive" in r.stdout


def kill_remote(pid):
    """Kill a remote process by PID."""
    target = _ssh_target()
    subprocess.run(["ssh", target, f"kill {pid}"], capture_output=True)


# ─── High-level training orchestrators ───

def train_gnn(args):
    """Train the GNN on M1 Max. Uploads samples, runs train_cnn.py, downloads weights."""
    if not check_connection(verbose=False):
        print("SSH connection failed. Run `python3 train_m1max.py check` to debug.")
        return 1
    sync_code()

    remote_samples = os.path.basename(args.samples)
    upload_file(args.samples, remote_samples)

    remote_out = os.path.basename(args.out)
    cmd = (
        f"python3 -u train_cnn.py "
        f"--samples {remote_samples} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--hidden {args.hidden} "
        f"--n-layers {args.n_layers} "
        f"--batch-size {args.batch_size} "
        f"--out {remote_out}"
    )
    if args.init_weights:
        remote_init = os.path.basename(args.init_weights)
        upload_file(args.init_weights, remote_init)
        cmd += f" --init-weights {remote_init}"

    _log("Starting GNN training on M1 Max")
    rc = run_remote(cmd, stream=True)
    if rc != 0:
        print(f"Training failed (exit {rc})")
        return rc

    download_file(remote_out, args.out)
    _log(f"GNN training complete — weights saved to {args.out}")
    return 0


def train_transformer(args):
    """Train the transformer on M1 Max."""
    if not check_connection(verbose=False):
        print("SSH connection failed. Run `python3 train_m1max.py check` to debug.")
        return 1
    sync_code()

    remote_samples = os.path.basename(args.samples)
    upload_file(args.samples, remote_samples)

    remote_out = os.path.basename(args.out)
    cmd = (
        f"python3 -u train_transformer.py "
        f"--samples {remote_samples} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--d-model {args.d_model} "
        f"--n-heads {args.n_heads} "
        f"--n-layers {args.n_layers} "
        f"--batch-size {args.batch_size} "
        f"--out {remote_out}"
    )
    if args.init_weights:
        remote_init = os.path.basename(args.init_weights)
        upload_file(args.init_weights, remote_init)
        cmd += f" --init-weights {remote_init}"

    _log("Starting transformer training on M1 Max")
    rc = run_remote(cmd, stream=True)
    if rc != 0:
        print(f"Training failed (exit {rc})")
        return rc

    download_file(remote_out, args.out)
    _log(f"Transformer training complete — weights saved to {args.out}")
    return 0


def train_nnue(args):
    """Train the NNUE (train_pytorch.py) on M1 Max for comparison vs MPS / Modal."""
    if not check_connection(verbose=False):
        print("SSH connection failed. Run `python3 train_m1max.py check` to debug.")
        return 1
    sync_code()

    remote_samples = os.path.basename(args.samples)
    upload_file(args.samples, remote_samples)

    remote_out = os.path.basename(args.out)
    cmd = (
        f"python3 -u train_pytorch.py value "
        f"--samples {remote_samples} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--batch-size {args.batch_size} "
        f"--hidden1 {args.hidden1} "
        f"--hidden2 {args.hidden2} "
        f"--num-features {args.num_features} "
        f"--optimizer {args.optimizer} "
        f"--out {remote_out} "
        f"--no-augment"
    )
    if args.init_weights:
        remote_init = os.path.basename(args.init_weights)
        upload_file(args.init_weights, remote_init)
        cmd += f" --init-weights {remote_init}"

    _log("Starting NNUE training on M1 Max")
    rc = run_remote(cmd, stream=True)
    if rc != 0:
        print(f"Training failed (exit {rc})")
        return rc

    download_file(remote_out, args.out)
    _log(f"NNUE training complete — weights saved to {args.out}")
    return 0


# ─── CLI ───

def main():
    p = argparse.ArgumentParser(description="Dispatch training to a remote M1 Max.")
    p.add_argument("--host", default=None, help="Override M1MAX_HOST env var")
    p.add_argument("--user", default=None, help="Override M1MAX_USER env var")
    p.add_argument("--remote-dir", default=None, help="Override M1MAX_REPO env var")

    sub = p.add_subparsers(dest="cmd", required=True)

    # check
    sub.add_parser("check", help="Verify SSH connection + remote environment")

    # sync
    sub.add_parser("sync", help="rsync current tree to remote")

    # exec
    p_exec = sub.add_parser("exec", help="Run an arbitrary command on remote")
    p_exec.add_argument("command", help="Shell command to run (quote it)")

    # upload / download
    p_up = sub.add_parser("upload", help="Upload a file to remote repo dir")
    p_up.add_argument("path")
    p_up.add_argument("--to", default=None, help="remote relative path")

    p_dn = sub.add_parser("download", help="Download a file from remote repo dir")
    p_dn.add_argument("path", help="remote path (relative to repo)")
    p_dn.add_argument("--to", default=None, help="local path")

    # GNN training
    p_gnn = sub.add_parser("gnn", help="Train the GNN on M1 Max")
    p_gnn.add_argument("--samples", required=True)
    p_gnn.add_argument("--epochs", type=int, default=30)
    p_gnn.add_argument("--lr", type=float, default=0.001)
    p_gnn.add_argument("--hidden", type=int, default=128)
    p_gnn.add_argument("--n-layers", type=int, default=3)
    p_gnn.add_argument("--batch-size", type=int, default=256)
    p_gnn.add_argument("--init-weights", default=None)
    p_gnn.add_argument("--out", default="gnn_m1max.pt")

    # Transformer training
    p_tr = sub.add_parser("transformer", help="Train the transformer on M1 Max")
    p_tr.add_argument("--samples", required=True)
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--d-model", type=int, default=128)
    p_tr.add_argument("--n-heads", type=int, default=4)
    p_tr.add_argument("--n-layers", type=int, default=3)
    p_tr.add_argument("--batch-size", type=int, default=512)
    p_tr.add_argument("--init-weights", default=None)
    p_tr.add_argument("--out", default="transformer_m1max.pt")

    # NNUE training (big model, tests whether M1 Max helps)
    p_nn = sub.add_parser("nnue", help="Train the full NNUE on M1 Max")
    p_nn.add_argument("--samples", required=True)
    p_nn.add_argument("--epochs", type=int, default=15)
    p_nn.add_argument("--lr", type=float, default=0.0001)
    p_nn.add_argument("--batch-size", type=int, default=4096)
    p_nn.add_argument("--hidden1", type=int, default=512)
    p_nn.add_argument("--hidden2", type=int, default=64)
    p_nn.add_argument("--num-features", type=int, default=45260)
    p_nn.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p_nn.add_argument("--init-weights", default=None)
    p_nn.add_argument("--out", default="nnue_m1max.bin")

    args = p.parse_args()

    # Allow CLI overrides of connection config
    global HOST, USER, REMOTE_DIR
    if args.host:
        HOST = args.host
    if args.user:
        USER = args.user
    if args.remote_dir:
        REMOTE_DIR = args.remote_dir

    if args.cmd == "check":
        ok = check_connection()
        return 0 if ok else 1

    if args.cmd == "sync":
        ok = sync_code()
        return 0 if ok else 1

    if args.cmd == "exec":
        rc = run_remote(args.command, stream=True)
        return rc

    if args.cmd == "upload":
        ok = upload_file(args.path, args.to)
        return 0 if ok else 1

    if args.cmd == "download":
        ok = download_file(args.path, args.to)
        return 0 if ok else 1

    if args.cmd == "gnn":
        return train_gnn(args)
    if args.cmd == "transformer":
        return train_transformer(args)
    if args.cmd == "nnue":
        return train_nnue(args)

    print(f"Unknown command: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
