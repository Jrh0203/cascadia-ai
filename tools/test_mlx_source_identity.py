from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/mlx_source_identity.py"
SPEC = importlib.util.spec_from_file_location("mlx_source_identity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
identity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = identity
SPEC.loader.exec_module(identity)


def make_repository(root: Path) -> None:
    (root / "python/cascadia_mlx").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / "python/cascadia_mlx/a.py").write_text("A = 1\n")
    (root / "python/cascadia_mlx/b.py").write_text("B = 2\n")


def test_source_identity_is_content_deterministic(tmp_path: Path) -> None:
    make_repository(tmp_path)
    first = identity.collect_source_identity(tmp_path, host="john1")
    second = identity.collect_source_identity(tmp_path, host="john2")
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["entries"] == second["entries"]
    assert first["files"] == 4


def test_source_identity_changes_with_runtime_source(tmp_path: Path) -> None:
    make_repository(tmp_path)
    before = identity.collect_source_identity(tmp_path)
    (tmp_path / "python/cascadia_mlx/b.py").write_text("B = 3\n")
    after = identity.collect_source_identity(tmp_path)
    assert before["bundle_sha256"] != after["bundle_sha256"]
