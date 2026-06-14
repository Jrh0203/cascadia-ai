from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2_CRATES = (
    "cascadia-api",
    "cascadia-cli-v2",
    "cascadia-data",
    "cascadia-eval",
    "cascadia-game",
    "cascadia-model",
    "cascadia-provenance",
    "cascadia-search",
    "cascadia-sim",
)
RUST_SOURCE_DIRS = tuple(ROOT / f"crates/{name}/src" for name in V2_CRATES)


def source_lines(path: Path) -> int:
    return path.read_text().count("\n") + 1


def production_lines(path: Path) -> int:
    production = path.read_text().split("\n#[cfg(test)]", maxsplit=1)[0]
    return production.count("\n") + 1


def test_v2_cli_entrypoint_stays_dispatch_only() -> None:
    main = ROOT / "crates/cascadia-cli-v2/src/main.rs"
    lines = source_lines(main)
    assert lines <= 300, f"{main.relative_to(ROOT)} grew to {lines} lines"


def test_v2_rust_modules_keep_focused_ownership() -> None:
    oversized = {
        str(path.relative_to(ROOT)): production_lines(path)
        for directory in RUST_SOURCE_DIRS
        for path in directory.glob("*.rs")
        if production_lines(path) > 1_500
    }
    assert not oversized, (
        f"v2 modules exceeded 1,500 lines: {oversized}; split by ownership before extending them"
    )


def test_v1_remains_behind_the_explicit_legacy_boundary() -> None:
    for old_name in ("cascadia-ai", "cascadia-cli", "cascadia-core", "cascadia-web"):
        assert not (ROOT / f"crates/{old_name}").exists()
        assert (ROOT / f"legacy/crates/{old_name}").is_dir()

    production_manifests = [
        ROOT / f"crates/{name}/Cargo.toml" for name in V2_CRATES if name != "cascadia-differential"
    ]
    for manifest in production_manifests:
        assert "legacy/" not in manifest.read_text(), (
            f"{manifest.relative_to(ROOT)} imported the superseded v1 boundary"
        )


def test_make_recipes_use_resolved_project_tools() -> None:
    makefile = (ROOT / "Makefile").read_text().splitlines()
    bare_tools = [line for line in makefile if line.startswith(("\tcargo ", "\tuv ", "\tnpm "))]
    assert not bare_tools, (
        "Make recipes must use $(CARGO), $(UV), or $(NPM) so they work over minimal SSH PATHs"
    )
