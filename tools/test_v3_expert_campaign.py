from __future__ import annotations

from pathlib import Path

import v3_expert_campaign as campaign


def test_controller_lock_rejects_duplicate_launch(tmp_path: Path) -> None:
    path = tmp_path / "expert.lock"
    first = campaign._acquire_controller_lock(path)
    assert first is not None
    try:
        assert campaign._acquire_controller_lock(path) is None
    finally:
        first.close()
    replacement = campaign._acquire_controller_lock(path)
    assert replacement is not None
    replacement.close()


def test_cycle_phase_parser_accepts_registered_phases() -> None:
    assert campaign._cycle_from_phase("cycle-01-collecting") == 1
    assert campaign._cycle_from_phase("cycle-10-promotion") == 10
    assert campaign._cycle_from_phase("bootstrap_training") is None
