from __future__ import annotations

import stat
from pathlib import Path

import pytest
import r2_map_w0_w5_validate as validate


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_remove_tree_handles_immutable_content_without_following_links(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("protected\n", encoding="ascii")
    outside.chmod(0o400)

    root = tmp_path / "cleanup"
    nested = root / "immutable"
    nested.mkdir(parents=True)
    payload = nested / "payload.json"
    payload.write_text("evidence\n", encoding="ascii")
    payload.chmod(0o400)
    nested.chmod(0o500)
    (root / "outside-link").symlink_to(outside)

    validate.remove_tree(root)
    assert not root.exists()
    assert outside.read_text(encoding="ascii") == "protected\n"
    assert _mode(outside) == 0o400


def test_cleanup_root_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "cleanup-link"
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(validate.ValidationError, match="cleanup root"):
        validate.remove_tree(root)
    assert root.is_symlink()
    assert target.is_dir()

    root.unlink()
    root.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(validate.ValidationError, match="cleanup root"):
        validate.remove_tree(root)
    assert root.is_symlink()
