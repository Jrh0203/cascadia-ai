from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
EXTRACTION_GATES = (
    "tools/r2_map_python_boundary_gate.mk",
    "tools/r2_map_python_fixture_gate.mk",
    "tools/r2_map_rust_compile_gate.mk",
    "tools/r2_map_rust_p1_gate.mk",
    "tools/r2_map_rust_release_gate.mk",
    "tools/r2_map_rust_w4_target_gate.mk",
)


@pytest.mark.parametrize("relative", EXTRACTION_GATES)
def test_every_source_extraction_is_preverified_fresh_and_postverified(
    relative: str,
) -> None:
    source = (REPOSITORY / relative).read_text(encoding="utf-8")

    assert "$(TRANSACTION)/archive-verify.py" in source
    assert "$(TRANSACTION)/source-manifest.json" in source
    assert "$(TRANSACTION)/source.tar" in source
    assert "mkdir -p" not in source
    assert "$(WORKSPACE)/Cargo.toml:" not in source

    absent = source.index('test ! -e "$(WORKSPACE)"')
    archive_verify = source.index('"$(ARCHIVE_VERIFIER)" verify')
    create = source.index('mkdir -m 0700 "$(WORKSPACE)"')
    extract_command = 'umask 077 && COPYFILE_DISABLE=1 /usr/bin/tar -xf "$(SOURCE_ARCHIVE)"'
    assert source.count(extract_command) == 1
    extract = source.index(extract_command)
    extracted_tree_verify = source.index('"$(ARCHIVE_VERIFIER)" verify-tree')
    tree_verify = source.index("tools/r2_map_w0_w5_source_manifest.py --repository . verify")
    assert absent < archive_verify < create < extract < extracted_tree_verify < tree_verify


@pytest.mark.parametrize(
    "relative",
    (
        "tools/r2_map_python_boundary_gate.mk",
        "tools/r2_map_python_fixture_gate.mk",
    ),
)
def test_python_gates_verify_the_captured_v1_1_reference_manifest(
    relative: str,
) -> None:
    source = (REPOSITORY / relative).read_text(encoding="utf-8")

    assert "tools/r2_map_reference_panels.py" in source
    assert "--revision v1.1 verify" in source
    assert "r2-map-w0-reference-panel-manifest-v1.1.json" in source
    assert "--revision v1.1 render" not in source


def test_release_gate_is_serial_even_under_inherited_parallel_make_flags() -> None:
    source = (REPOSITORY / "tools/r2_map_rust_release_gate.mk").read_text(encoding="utf-8")

    assert ".NOTPARALLEL:" in source
    assert "release: test clippy fmt" in source


def test_python_boundary_owns_and_cleans_cross_host_pytest_and_cargo_state() -> None:
    source = (REPOSITORY / "tools/r2_map_python_boundary_gate.mk").read_text(encoding="utf-8")

    assert "PYTEST_BASETEMP := $(TMPDIR)/r2-map-python-boundary-pytest" in source
    assert '--basetemp "$(PYTEST_BASETEMP)"' in source
    assert "tmp_path_retention_policy=none" in source
    assert "trap cleanup_or_fail 0" in source
    assert "trap 'cleanup_or_fail; exit 130' 2" in source
    assert "trap - 0 1 2 15" in source
    assert '"$(ARCHIVE_VERIFIER)"' in source
    assert "cleanup-pytest;" in source
    assert "cleanup-pytest --parent" not in source
    assert "/bin/rm -rf" not in source
    assert 'test ! -e "$(PYTEST_BASETEMP)"' in source
    assert 'test ! -L "$(PYTEST_BASETEMP)"' in source
    assert '/usr/bin/find "$(TMPDIR)" -type l -print -quit' in source
    assert "exit $$status" in source
    assert "stable-aarch64-apple-darwin/bin" in source
    assert 'PATH="$(TOOLCHAIN):$$PATH"' in source
    assert 'CARGO_TARGET_DIR="$(CARGO_TARGET_DIR)"' in source
