from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_v3_mlx import campaign_train
from cascadia_v3_mlx.provenance import training_source_identity


def test_training_source_identity_is_complete_and_stable() -> None:
    first = training_source_identity()
    second = training_source_identity()
    assert first == second
    paths = {record["path"] for record in first["files"]}
    assert "python/cascadia_v3_mlx/campaign_train.py" in paths
    assert "python/cascadia_v3_mlx/cycle_train.py" in paths
    assert "python/cascadia_v3_mlx/provenance.py" in paths
    assert "python/cascadia_mlx/checkpoint.py" in paths
    assert len(first["blake3"]) == 64


def test_long_running_trainers_bound_mlx_free_buffer_cache() -> None:
    assert campaign_train.MLX_CACHE_LIMIT_BYTES == 512 * 1024**2
    assert campaign_train.LOSS_PROGRESS_EVERY_STEPS == 25


def test_training_storage_guard_trims_sparse_worker_disk_before_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = 1280 * 1024**2
    readings = iter(
        [
            SimpleNamespace(free=campaign_train.MIN_FREE_BYTES + planned - 1),
            SimpleNamespace(free=campaign_train.MIN_FREE_BYTES + planned + 1),
        ]
    )
    trims = 0

    def trim() -> bool:
        nonlocal trims
        trims += 1
        return True

    monkeypatch.setattr(campaign_train, "_campaign_bytes", lambda _root: 0)
    monkeypatch.setattr(campaign_train.shutil, "disk_usage", lambda _root: next(readings))
    monkeypatch.setattr(campaign_train, "_trim_sparse_worker_disk", trim)

    campaign_train._assert_storage(tmp_path, planned)

    assert trims == 1


def test_training_storage_guard_still_refuses_genuine_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = 1280 * 1024**2
    reading = SimpleNamespace(free=campaign_train.MIN_FREE_BYTES + planned - 1)
    monkeypatch.setattr(campaign_train, "_campaign_bytes", lambda _root: 0)
    monkeypatch.setattr(campaign_train.shutil, "disk_usage", lambda _root: reading)
    monkeypatch.setattr(campaign_train, "_trim_sparse_worker_disk", lambda: True)

    with pytest.raises(campaign_train.CampaignTrainingError, match="free_bytes"):
        campaign_train._assert_storage(tmp_path, planned)


def test_schedule_is_exactly_twelve_blocks_and_36m_per_origin(tmp_path: Path) -> None:
    path_to_module = Path(__file__).resolve().parents[2] / "tools/v3_training_schedule.py"
    spec = importlib.util.spec_from_file_location("v3_training_schedule_for_test", path_to_module)
    assert spec is not None and spec.loader is not None
    v3_training_schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3_training_schedule)

    value = v3_training_schedule.build(8192)
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(value))
    decoded = campaign_train._schedule(path)
    assert len(decoded["bootstrap"]["blocks"]) == 12
    assert sum(
        block["exposures"] for block in decoded["bootstrap"]["blocks"]
    ) == 36_000_000


def test_round_robin_preserves_each_source_without_duplication() -> None:
    values = list(campaign_train._round_robin(iter([1, 2, 3]), iter([10, 20])))
    assert values == [
        ("broad", 1),
        ("teacher", 10),
        ("broad", 2),
        ("teacher", 20),
        ("broad", 3),
    ]


def test_deterministic_shard_order_changes_by_block(tmp_path: Path) -> None:
    paths = []
    for index in range(20):
        path = tmp_path / f"{index:02}.v3g"
        path.write_bytes(bytes([index]))
        paths.append(path)
    first = campaign_train._ordered(paths, 73, 1, "broad")
    assert first == campaign_train._ordered(paths, 73, 1, "broad")
    assert first != campaign_train._ordered(paths, 73, 2, "broad")


def test_checkpoint_serving_integrity_is_durable_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def export(*_args: object, **_kwargs: object) -> dict[str, str]:
        output = _args[1]
        assert isinstance(output, Path)
        output.mkdir(parents=True)
        return {"weights_blake3": "a" * 64}

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"passed": true}')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(campaign_train, "export_quantized_bundle", export)
    monkeypatch.setattr(campaign_train.subprocess, "run", run)
    args = SimpleNamespace(
        checkpoint_integrity_games=2,
        checkpoint_integrity_first_seed=1_650_000,
        checkpoint_integrity_binary=tmp_path / "engine",
        feature_manifest=tmp_path / "features.json",
        run_dir=tmp_path / "run",
        origin="origin-1",
    )
    state = campaign_train.TrainerState(global_step=3, examples_seen=16_384)

    campaign_train._validate_checkpoint_serving(
        model=object(), args=args, state=state, run_manifest_blake3="b" * 64
    )
    campaign_train._validate_checkpoint_serving(
        model=object(), args=args, state=state, run_manifest_blake3="b" * 64
    )

    assert calls == 1
    receipt = json.loads(
        (args.run_dir / "checkpoint-integrity/000000016384.json").read_text()
    )
    assert receipt["passed"] is True
    assert receipt["weights_blake3"] == "a" * 64
    assert receipt["game_report"].endswith("000000016384.games.json")
    assert receipt["game_report_blake3"] == campaign_train._checksum(
        args.run_dir / "checkpoint-integrity/000000016384.games.json"
    )


def test_checkpoint_serving_integrity_stops_on_integer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def export(*_args: object, **_kwargs: object) -> dict[str, str]:
        output = _args[1]
        assert isinstance(output, Path)
        output.mkdir(parents=True)
        return {"weights_blake3": "c" * 64}

    monkeypatch.setattr(campaign_train, "export_quantized_bundle", export)
    monkeypatch.setattr(
        campaign_train.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "Error: AccumulatorOverflow"
        ),
    )
    args = SimpleNamespace(
        checkpoint_integrity_games=2,
        checkpoint_integrity_first_seed=1_650_000,
        checkpoint_integrity_binary=tmp_path / "engine",
        feature_manifest=tmp_path / "features.json",
        run_dir=tmp_path / "run",
        origin="origin-1",
    )
    state = campaign_train.TrainerState(global_step=9, examples_seen=5_000_000)

    with pytest.raises(campaign_train.CampaignTrainingError, match="serving integrity failed"):
        campaign_train._validate_checkpoint_serving(
            model=object(), args=args, state=state, run_manifest_blake3="d" * 64
        )
    receipt = json.loads(
        (args.run_dir / "checkpoint-integrity/000005000000.json").read_text()
    )
    assert receipt["passed"] is False
    assert "AccumulatorOverflow" in receipt["stderr"]


def test_online_swa_is_exact_resumable_and_storage_bounded(tmp_path: Path) -> None:
    class TinyModel:
        def __init__(self, values: list[float]) -> None:
            self.weight = mx.array(values, dtype=mx.float32)

        def parameters(self) -> dict[str, mx.array]:
            return {"weight": self.weight}

        def load_weights(self, path: str) -> None:
            self.weight = mx.load(path)["weight"]

    model = TinyModel([1.0, 3.0])
    first = campaign_train._save_swa_snapshot(tmp_path, model, 100)
    model.weight = mx.array([3.0, 7.0], dtype=mx.float32)
    second = campaign_train._save_swa_snapshot(tmp_path, model, 200)

    assert not first.exists()
    assert second.is_file()
    assert len(list((tmp_path / "swa").glob("average-*.safetensors"))) == 1
    np.testing.assert_array_equal(np.asarray(mx.load(second)["weight"]), [2.0, 5.0])
    final, count = campaign_train._average_swa(tmp_path, model)
    assert count == 2
    assert final.is_file()
    np.testing.assert_array_equal(np.asarray(model.weight), [2.0, 5.0])


def test_forward_swa_journal_is_replayed_without_double_counting(tmp_path: Path) -> None:
    class TinyModel:
        def __init__(self, values: list[float]) -> None:
            self.weight = mx.array(values, dtype=mx.float32)

        def parameters(self) -> dict[str, mx.array]:
            return {"weight": self.weight}

    model = TinyModel([1.0, 3.0])
    campaign_train._save_swa_snapshot(tmp_path, model, 28_802_240)
    model.weight = mx.array([3.0, 7.0], dtype=mx.float32)
    campaign_train._save_swa_snapshot(tmp_path, model, 29_700_560)
    model.weight = mx.array([5.0, 11.0], dtype=mx.float32)
    campaign_train._save_swa_snapshot(tmp_path, model, 30_601_680)
    model.weight = mx.array([7.0, 15.0], dtype=mx.float32)
    campaign_train._save_swa_snapshot(tmp_path, model, 31_500_000)

    replay_target = campaign_train._swa_replay_target(
        run_dir=tmp_path,
        checkpoint_examples=30_003_664,
        swa_start=28_800_000,
        swa_interval=900_000,
        batch_size=8_192,
        run_manifest_blake3="a" * 64,
        checkpoint_id="step-000003672-epoch-0000-batch-000000",
    )

    assert replay_target == 31_500_000
    state = json.loads((tmp_path / "swa/state.json").read_text())
    assert state["count"] == 4
    receipt = json.loads((tmp_path / "swa/replay-recovery.json").read_text())
    assert receipt["swa_event_target"] == 31_500_000
    assert receipt["recovery_rule"].startswith("exact-replay")


def test_operational_source_migration_changes_only_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    receipt_path = tmp_path / "source-migration.json"
    old_source = {
        "blake3": "1" * 64,
        "files": [{"path": "campaign_train.py", "bytes": 10, "blake3": "2" * 64}],
    }
    new_source = {
        "blake3": "3" * 64,
        "files": [{"path": "campaign_train.py", "bytes": 12, "blake3": "4" * 64}],
    }
    old = {"schema_id": "run", "origin": "o2", "training_source_identity": old_source}
    old["canonical_blake3"] = campaign_train.blake3.blake3(
        campaign_train._canonical(old)
    ).hexdigest()
    path.write_text(json.dumps(old))
    changed_files = [
        {
            "path": "campaign_train.py",
            "before_bytes": 10,
            "before_blake3": "2" * 64,
            "after_bytes": 12,
            "after_blake3": "4" * 64,
        }
    ]
    receipt_path.write_text(
        json.dumps(
            {
                "schema_id": "cascadia-v3-operational-source-migration-v1",
                "passed": True,
                "from_run_manifest_blake3": old["canonical_blake3"],
                "from_training_source_blake3": old_source["blake3"],
                "to_training_source_blake3": new_source["blake3"],
                "classification": "recovery-only-no-training-math-change",
                "changed_files": changed_files,
            }
        )
    )
    new = {"schema_id": "run", "origin": "o2", "training_source_identity": new_source}

    digest, accepted = campaign_train._bind_manifest_with_source_migration(
        path, new, True, receipt_path
    )

    assert old["canonical_blake3"] in accepted
    assert digest in accepted
    assert json.loads(path.read_text())["canonical_blake3"] == digest


def test_operational_source_migration_rejects_scientific_change(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    receipt_path = tmp_path / "source-migration.json"
    old_source = {"blake3": "1" * 64, "files": []}
    new_source = {"blake3": "3" * 64, "files": []}
    old = {
        "schema_id": "run",
        "base_learning_rate": 0.0015,
        "training_source_identity": old_source,
    }
    old["canonical_blake3"] = campaign_train.blake3.blake3(
        campaign_train._canonical(old)
    ).hexdigest()
    path.write_text(json.dumps(old))
    receipt_path.write_text(
        json.dumps(
            {
                "schema_id": "cascadia-v3-operational-source-migration-v1",
                "passed": True,
                "from_run_manifest_blake3": old["canonical_blake3"],
                "from_training_source_blake3": old_source["blake3"],
                "to_training_source_blake3": new_source["blake3"],
                "classification": "recovery-only-no-training-math-change",
                "changed_files": [],
            }
        )
    )
    changed = {
        "schema_id": "run",
        "base_learning_rate": 0.002,
        "training_source_identity": new_source,
    }

    with pytest.raises(campaign_train.CampaignTrainingError, match="scientific run contract"):
        campaign_train._bind_manifest_with_source_migration(
            path, changed, True, receipt_path
        )


def test_trainer_lock_rejects_duplicate_writer(tmp_path: Path) -> None:
    path = tmp_path / ".trainer.lock"
    first = campaign_train._acquire_trainer_lock(path)
    assert first is not None
    try:
        assert campaign_train._acquire_trainer_lock(path) is None
    finally:
        first.close()
    replacement = campaign_train._acquire_trainer_lock(path)
    assert replacement is not None
    replacement.close()
