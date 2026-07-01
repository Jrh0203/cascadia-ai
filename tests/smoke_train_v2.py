#!/usr/bin/env python3
"""End-to-end MLX smoketest for the AZv2 trainer.

Generates a tiny AZD2 fixture via `cascadia-cli --az-collect --az-arch v2`,
trains a small v2 net for 1 epoch on it, asserts no NaN / finite KL, then
saves an AZR2 and loads it back through `cascadia-cli --az` to validate the
Rust-side reader.

Intended as the Phase-0 acceptance gate for the v2 pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _resolve_cli() -> Path:
    for candidate in [
        Path("./target/release/cascadia-cli"),
        Path("./target/debug/cascadia-cli"),
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "cascadia-cli not found — build first:\n"
        "  cargo build --release --features v4-opp,v5-feat,czero-feat,az-v2 --bin cascadia-cli"
    )


def _run(cmd: list[str], env: dict | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, env=env, check=False)
    if res.returncode != 0:
        raise SystemExit(f"command failed (rc={res.returncode}): {' '.join(cmd)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    # The v2 trainer lives at the repo root; add it to sys.path so this smoke
    # test can import it.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    cli = _resolve_cli()
    print(f"cli={cli}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        azd_path = td / "smoke_v2.azd"
        azr_path = td / "smoke_v2.azr"

        env = os.environ.copy()
        env["CASCADIA_SCORING_CARDS"] = "A,A,A,A,A"

        # 1. Collect a 2-game greedy bootstrap shard.
        _run(
            [
                str(cli),
                "2",
                "--az-collect",
                "--az-arch",
                "v2",
                "--out",
                str(azd_path),
                "--seed",
                "777",
                "--score-target",
                "with-bonus",
            ],
            env=env,
        )
        assert azd_path.exists(), "AZD2 file was not produced"
        print(f"AZD2 size: {azd_path.stat().st_size} bytes")

        # 2. Train 1 epoch with a tiny v2 model.
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from train_alphazero_mlx_v2 import (
            CascadiaAzNetV2,
            load_azd_v2,
            loss_fn,
            make_batch,
            save_azr_v2,
        )

        mx.random.seed(11)
        ds = load_azd_v2([azd_path])
        assert ds.size > 0, "loaded zero samples"
        # Phase 0.7 layout: 68 planes x 128 cells own, 3x68x128 opp, 10x32 entities.
        assert ds.inputs.shape[1] == 68, f"expected 68 input planes, got {ds.inputs.shape[1]}"
        assert ds.inputs.shape[2] == 128
        assert ds.opp_inputs.shape[1] == 3, (
            f"expected 3 opponent boards, got {ds.opp_inputs.shape[1]}"
        )
        assert ds.opp_inputs.shape[2:] == (68, 128), (
            f"opp boards wrong shape: {ds.opp_inputs.shape}"
        )
        assert ds.entities.shape[1] == 10, f"expected 10 entity tokens, got {ds.entities.shape[1]}"
        assert ds.entities.shape[2] == 32
        assert ds.aux_values.shape[1] == 16
        assert ds.phase_one_hot.shape[1] == 3
        # Phase one-hot is a valid distribution row-wise (sourced from globals
        # token at index 8, dims 0..3).
        assert np.all(np.isclose(ds.phase_one_hot.sum(axis=1), 1.0))
        np.testing.assert_allclose(ds.phase_one_hot, ds.entities[:, 8, 0:3], atol=1e-6)
        # Race-state token (index 9) carries 5 x 4-way rank one-hots in dims 0..20.
        for ti in range(5):
            rank_block = ds.entities[:, 9, ti * 4 : (ti + 1) * 4]
            row_sums = rank_block.sum(axis=1)
            assert np.all(np.isclose(row_sums, 1.0)), (
                f"race-state terrain {ti} rank one-hot rows must sum to 1"
            )
        # Opp entity slots (4..7) are zero at sample time; filled by trunk.
        assert np.allclose(ds.entities[:, 4:7, :], 0.0), (
            "opp entity slots should be zero — shared trunk fills them at forward time"
        )
        # Bias plane on own + each opp board: 1.0 on real cells, 0.0 on pad.
        assert np.allclose(ds.inputs[:, 0, :127], 1.0)
        assert np.allclose(ds.inputs[:, 0, 127], 0.0)
        # 4P AAAAA → all 3 opp boards are real (non-zero bias plane on real cells).
        assert np.allclose(ds.opp_inputs[:, :, 0, :127], 1.0)
        assert np.allclose(ds.opp_inputs[:, :, 0, 127], 0.0)

        model = CascadiaAzNetV2(
            channels=16,
            blocks=2,
            entity_dim=16,
            sab_blocks=1,
            heads=2,
            value_hidden=32,
            max_candidates=ds.max_candidates,
            c_puct=2.0,
        )
        opt = optim.Adam(learning_rate=1e-3)
        loss_and_grad = nn.value_and_grad(model, lambda m, b: loss_fn(m, b, 1.0, 0.3))

        # Single epoch over all samples.
        rng = np.random.default_rng(13)
        order = rng.permutation(ds.size)
        for start in range(0, ds.size, 16):
            idx = order[start : start + 16]
            batch = make_batch(ds, idx)
            loss, grads = loss_and_grad(model, batch)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            loss_val = float(np.array(loss))
            assert np.isfinite(loss_val), f"non-finite loss {loss_val}"
        print(f"trained 1 epoch on {ds.size} samples, final loss={loss_val:.4f}")

        save_azr_v2(model, azr_path)
        assert azr_path.exists() and azr_path.stat().st_size > 0

        # 3. Verify Rust can load the AZR2 and play 1 game.
        _run(
            [
                str(cli),
                "1",
                "--az",
                "--weights",
                str(azr_path),
                "--az-sims",
                "4",
                "--score-target",
                "with-bonus",
            ],
            env=env,
        )

    print("PHASE-0 SMOKETEST OK")


if __name__ == "__main__":
    main()
