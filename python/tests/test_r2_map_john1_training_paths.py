from pathlib import Path

import pytest
from cascadia_mlx import r2_map_train


def test_native_training_run_path_is_scoped_to_john1_campaign_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_root = tmp_path / "john1-primary" / "r2-map-v1"
    runs = campaign_root / "runs"
    runs.mkdir(parents=True)
    outside = tmp_path / "john2-worker"
    outside.mkdir()

    monkeypatch.setattr(r2_map_train, "CAMPAIGN_ROOT", campaign_root)
    monkeypatch.setattr(r2_map_train, "require_local_storage_authority", lambda: "john1")

    r2_map_train._require_local_john1_run_path(runs / "iteration-0001")
    with pytest.raises(ValueError, match="john1 root"):
        r2_map_train._require_local_john1_run_path(outside / "iteration-0001")
