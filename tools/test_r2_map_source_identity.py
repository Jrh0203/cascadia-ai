from __future__ import annotations

from pathlib import Path

import r2_map_source_identity as subject


def _repository(root: Path) -> None:
    for relative in subject.SOURCE_ROOTS:
        path = root / relative
        if relative in {"Cargo.toml", "Cargo.lock", "Makefile", "pyproject.toml", "uv.lock"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "source.txt").write_text(relative)


def test_identity_is_content_deterministic_and_changes_with_source(
    tmp_path: Path, monkeypatch
) -> None:
    _repository(tmp_path)
    monkeypatch.setattr(subject, "_git", lambda *_args: "test")
    first = subject.source_identity(tmp_path)
    second = subject.source_identity(tmp_path)
    assert first == second
    changed = tmp_path / "crates/cascadia-eval/source.txt"
    changed.write_text("changed")
    assert subject.source_identity(tmp_path)["v2_source_blake3"] != first["v2_source_blake3"]
