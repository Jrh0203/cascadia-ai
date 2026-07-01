#!/usr/bin/env python3
"""Render or verify the immutable John2 W0/W5 source-transaction manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_ID = "cascadia.r2-map.w0-w5-source-manifest.v1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
OPEN_DOMAIN = "r2-map-open-reference-performance-100-v1"
OPEN_SEED_COUNT = 100
REFERENCE_MANIFEST_RELATIVE = (
    "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
)

ROOT_FILES = (
    ".python-version",
    "AGENTS.md",
    "Cargo.lock",
    "Cargo.toml",
    "pyproject.toml",
    "uv.lock",
)
TREE_ROOTS = (
    "crates",
    "legacy/crates",
    "python/cascadia_mlx",
    "python/tests",
    "tests/fixtures",
)
WEB_ROOT = "apps/web"
WEB_EXCLUDED_DIRECTORIES = frozenset(
    {
        "coverage",
        "dist",
        "node_modules",
        "playwright-report",
        "test-results",
    }
)
WEB_EXCLUDED_SUFFIXES = (".tsbuildinfo",)
EXACT_FILES = (
    "docs/v2/CLI_REFERENCE.md",
    "docs/v2/CLUSTER_DASHBOARD.md",
    "docs/v2/CLUSTER_SCHEDULER.md",
    "docs/v2/R2_MAP_DATASET_BRIDGE.md",
    "docs/v2/R2_MAP_EXPERIENCE_FORMAT.md",
    "docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md",
    "docs/v2/R2_MAP_JOHN2_REMOTE_STORAGE.md",
    "docs/v2/TRAINING.md",
    "docs/v2/decisions/0193-r2-map-deterministic-campaign-controller.md",
    "docs/v2/decisions/0194-r2-map-compact-on-demand-training-data.md",
    "docs/v2/decisions/0195-r2-map-john2-canonical-storage.md",
    "docs/v2/decisions/0196-r2-map-public-universal-market-legality.md",
    "docs/v2/decisions/0197-r2-map-rules-complete-token-capacity.md",
    "docs/v2/decisions/0198-r2-map-gate-order-adaptations.md",
    "docs/v2/decisions/0199-r2-map-manifest-driven-source-archive.md",
    "docs/v2/reports/r2-map-gate7-readmission-audit-v1.md",
    "docs/v2/reports/r2-map-john2-storage-migration-v1.md",
    "docs/v2/reports/r2-map-paired-gate-power-v1.json",
    "docs/v2/reports/r2-map-w0-gate-power-and-reference-panels-v1-preregistration.md",
    "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.json",
    "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json",
    "docs/v2/reports/r2-map-w0-v1-predecessor-transcript-recovery-v1.json",
    "docs/v2/reports/r2-map-w0-v1.1-readmission-preregistration.md",
    "docs/v2/reports/r2-map-w5-benchmark-dashboard-readmission-v1-preregistration.md",
    "docs/v2/reports/r2-map-w6-controller-readmission-v1-preregistration.md",
    "tools/cluster_experiment_ledger.py",
    "tools/cluster_research_queue.py",
    "tools/com.johnherrick.cascadia.dashboard.plist",
    "tools/com.johnherrick.cascadia.r2-map-dashboard-fetch.plist",
    "tools/r2_map_cargo_probe.mk",
    "tools/r2_map_compact_dataset.py",
    "tools/r2_map_dashboard_fetch.py",
    "tools/r2_map_expert_iteration.py",
    "tools/r2_map_headless_continuation_prompt.txt",
    "tools/r2_map_headless_resume.sh",
    "tools/r2_map_headless_resume_prompt.txt",
    "tools/r2_map_headless_turn.py",
    "tools/r2_map_host_recovery.py",
    "tools/r2_map_john1_packing_sweep.py",
    "tools/r2_map_john1_no_write_launcher.py",
    "tools/r2_map_john1_publish_write_attestation.py",
    "tools/r2_map_john1_train.py",
    "tools/r2_map_market_protocol_fixture.py",
    "tools/r2_map_max_width_service_smoke.py",
    "tools/r2_map_p1_resource_gate.py",
    "tools/r2_map_paired_gate_power.py",
    "tools/r2_map_reference_panels.py",
    "tools/r2_map_remote_rust_verify.sh",
    "tools/r2_map_remote_storage.py",
    "tools/r2_map_source_archive.py",
    "tools/r2_map_python_boundary_gate.mk",
    "tools/r2_map_python_fixture_gate.mk",
    "tools/r2_map_rust_compile_gate.mk",
    "tools/r2_map_rust_p1_gate.mk",
    "tools/r2_map_rust_release_gate.mk",
    "tools/r2_map_rust_w4_target_gate.mk",
    "tools/r2_map_w0_w5_source_manifest.py",
    "tools/r2_map_w0_w5_validate.py",
    "tools/r2_map_w5_dashboard_live.py",
    "tools/r2_map_w6_validate.py",
    "tools/test_r2_map_reference_panels.py",
    "tools/test_r2_map_source_archive.py",
    "tools/test_r2_map_source_extraction_gates.py",
    "tools/test_r2_map_dashboard_fetch.py",
    "tools/test_r2_map_headless_turn.py",
    "tools/test_r2_map_john1_packing_sweep.py",
    "tools/test_r2_map_john1_no_write_launcher.py",
    "tools/test_r2_map_paired_gate_power.py",
    "tools/test_r2_map_w0_w5_source_manifest.py",
    "tools/test_r2_map_w0_w5_validate.py",
    "tools/test_r2_map_w5_dashboard_live.py",
    "tools/test_r2_map_w6_validate.py",
)
EXECUTABLE_FILES = frozenset(
    {
        "tools/r2_map_compact_dataset.py",
        "tools/r2_map_reference_panels.py",
        "tools/r2_map_expert_iteration.py",
        "tools/r2_map_headless_resume.sh",
        "tools/r2_map_remote_rust_verify.sh",
        "tools/r2_map_source_archive.py",
        "tools/r2_map_w0_w5_source_manifest.py",
        "tools/r2_map_w0_w5_validate.py",
        "tools/r2_map_w5_dashboard_live.py",
        "tools/r2_map_w6_validate.py",
    }
)
RUNTIME_ENTRYPOINTS = (
    "tools/r2_map_compact_dataset.py",
    "tools/r2_map_dashboard_fetch.py",
    "tools/r2_map_expert_iteration.py",
    "tools/r2_map_headless_resume.sh",
    "tools/r2_map_headless_turn.py",
    "tools/r2_map_host_recovery.py",
    "tools/r2_map_john1_no_write_launcher.py",
    "tools/r2_map_john1_packing_sweep.py",
    "tools/r2_map_john1_publish_write_attestation.py",
    "tools/r2_map_john1_train.py",
    "tools/r2_map_market_protocol_fixture.py",
    "tools/r2_map_max_width_service_smoke.py",
    "tools/r2_map_p1_resource_gate.py",
    "tools/r2_map_reference_panels.py",
    "tools/r2_map_remote_storage.py",
    "tools/r2_map_source_archive.py",
    "tools/r2_map_w0_w5_source_manifest.py",
    "tools/r2_map_w0_w5_validate.py",
    "tools/r2_map_w5_dashboard_live.py",
    "tools/r2_map_w6_validate.py",
)
RUNTIME_GATE_FILES = (
    "tools/r2_map_cargo_probe.mk",
    "tools/r2_map_python_boundary_gate.mk",
    "tools/r2_map_python_fixture_gate.mk",
    "tools/r2_map_remote_rust_verify.sh",
    "tools/r2_map_rust_compile_gate.mk",
    "tools/r2_map_rust_p1_gate.mk",
    "tools/r2_map_rust_release_gate.mk",
    "tools/r2_map_rust_w4_target_gate.mk",
)
RUNTIME_PYTHON_ROOTS = ("python/cascadia_mlx",)
RUNTIME_RUST_ROOTS = ("crates", "legacy/crates")
RUNTIME_FIXTURE_ROOTS = ("tests/fixtures",)
RUNTIME_DEPENDENCY_LOCKS = ("Cargo.lock", "pyproject.toml", "uv.lock")
IGNORED_TREE_NAMES = frozenset({".DS_Store", "__pycache__"})
IGNORED_TREE_SUFFIXES = (".pyc", ".pyo")


class SourceManifestError(ValueError):
    """The source selection, identity, or frozen manifest is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(value: dict[str, Any]) -> str:
    content = dict(value)
    content.pop("document_sha256", None)
    return sha256_bytes(canonical_json(content))


def _require_regular(repository: Path, relative: str) -> Path:
    path = repository / relative
    try:
        details = path.lstat()
    except OSError as error:
        raise SourceManifestError(f"required source is absent: {relative}") from error
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise SourceManifestError(f"source is not a regular non-symlink file: {relative}")
    return path


def _iter_tree(
    repository: Path,
    relative_root: str,
    *,
    excluded_directories: frozenset[str] = frozenset(),
    excluded_suffixes: Sequence[str] = (),
) -> Iterable[str]:
    root = repository / relative_root
    try:
        details = root.lstat()
    except OSError as error:
        raise SourceManifestError(f"required source tree is absent: {relative_root}") from error
    if root.is_symlink() or not stat.S_ISDIR(details.st_mode):
        raise SourceManifestError(f"source tree is not a regular directory: {relative_root}")

    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            child_details = child.lstat()
            if child.is_symlink() or not stat.S_ISDIR(child_details.st_mode):
                relative = child.relative_to(repository).as_posix()
                raise SourceManifestError(f"source tree contains an unsafe directory: {relative}")
            if name not in excluded_directories and name not in IGNORED_TREE_NAMES:
                retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            if (
                name in IGNORED_TREE_NAMES
                or name.endswith(IGNORED_TREE_SUFFIXES)
                or name.endswith(tuple(excluded_suffixes))
            ):
                continue
            path = current_path / name
            details = path.lstat()
            relative = path.relative_to(repository).as_posix()
            if path.is_symlink() or not stat.S_ISREG(details.st_mode):
                raise SourceManifestError(f"source tree contains a non-regular file: {relative}")
            yield relative


def selected_paths(repository: Path) -> list[str]:
    repository = repository.resolve(strict=True)
    selected = set(ROOT_FILES)
    selected.update(EXACT_FILES)
    for relative_root in TREE_ROOTS:
        selected.update(_iter_tree(repository, relative_root))
    selected.update(
        _iter_tree(
            repository,
            WEB_ROOT,
            excluded_directories=WEB_EXCLUDED_DIRECTORIES,
            excluded_suffixes=WEB_EXCLUDED_SUFFIXES,
        )
    )
    for relative in sorted(selected):
        _require_regular(repository, relative)
    if not EXECUTABLE_FILES.issubset(selected):
        raise SourceManifestError("the source selection omits a required executable")
    return sorted(selected)


def _verify_reference_manifest(repository: Path) -> dict[str, Any]:
    path = _require_regular(repository, REFERENCE_MANIFEST_RELATIVE)
    try:
        manifest = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceManifestError(f"cannot read W0 v1.1 manifest: {error}") from error
    if (
        manifest.get("schema_id") != "cascadia.r2-map.reference-panel-manifest.v1.1"
        or manifest.get("contract_revision") != "sequential-public-market-v1.1"
    ):
        raise SourceManifestError("W0 v1.1 manifest has the wrong contract identity")
    open_panels = [
        panel
        for panel in manifest.get("panels", [])
        if isinstance(panel, dict) and panel.get("panel_id") == "open-performance-100"
    ]
    if len(open_panels) != 1:
        raise SourceManifestError("W0 v1.1 manifest has no unique open panel")
    definition = open_panels[0].get("definition", {})
    seeds = definition.get("seeds") if isinstance(definition, dict) else None
    if (
        not isinstance(definition, dict)
        or definition.get("seed_domain") != OPEN_DOMAIN
        or not isinstance(seeds, list)
        or len(seeds) != OPEN_SEED_COUNT
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != OPEN_SEED_COUNT
        or definition.get("seed_domain_changed") is not False
        or definition.get("predecessor_outcomes_opened") is not False
        or definition.get("old_v1_outcomes_reused") is not False
    ):
        raise SourceManifestError("W0 v1.1 open seed boundary differs")
    protected = manifest.get("protected_seed_domains")
    if not isinstance(protected, list) or not protected:
        raise SourceManifestError("W0 v1.1 protected-domain descriptors are absent")
    if any(
        not isinstance(item, dict)
        or item.get("opened") is not False
        or item.get("seed_material_present") is not False
        for item in protected
    ):
        raise SourceManifestError("a protected seed domain is opened")
    protected_handling = manifest.get("protected_seed_handling")
    if protected_handling != {
        "opening_authorized": False,
        "values_accepted_by_tool": False,
        "values_derivable_by_tool": False,
        "values_in_manifest": False,
    }:
        raise SourceManifestError("protected seed handling is not sealed")
    return manifest


def _runtime_closure(repository: Path, selected_paths_value: list[str]) -> dict[str, Any]:
    selected = set(selected_paths_value)
    required = (
        set(RUNTIME_ENTRYPOINTS)
        | set(RUNTIME_GATE_FILES)
        | set(RUNTIME_DEPENDENCY_LOCKS)
    )
    missing = sorted(required - selected)
    if missing:
        raise SourceManifestError(f"runtime closure omits required files: {missing}")

    local_tool_edges: set[tuple[str, str]] = set()
    for relative in RUNTIME_ENTRYPOINTS:
        if not relative.endswith(".py"):
            continue
        source = _require_regular(repository, relative).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as error:
            raise SourceManifestError(
                f"runtime entrypoint is not valid Python: {relative}"
            ) from error
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                root = name.split(".", 1)[0]
                candidate = f"tools/{root}.py"
                if root.startswith(("r2_map_", "cluster_")) and (
                    repository / candidate
                ).is_file():
                    if candidate not in selected:
                        raise SourceManifestError(
                            f"runtime tool import is outside the source closure: {candidate}"
                        )
                    local_tool_edges.add((relative, candidate))

    def rooted_paths(roots: tuple[str, ...], label: str) -> list[str]:
        values = sorted(
            relative
            for relative in selected_paths_value
            if any(relative == root or relative.startswith(f"{root}/") for root in roots)
        )
        if roots and not values:
            raise SourceManifestError(f"runtime {label} closure is empty")
        return values

    python_paths = rooted_paths(RUNTIME_PYTHON_ROOTS, "Python")
    rust_paths = rooted_paths(RUNTIME_RUST_ROOTS, "Rust")
    fixture_paths = rooted_paths(RUNTIME_FIXTURE_ROOTS, "fixture")

    def path_set_identity(values: list[str]) -> dict[str, Any]:
        return {
            "file_count": len(values),
            "paths_sha256": sha256_bytes(canonical_json(values)),
        }

    return {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.runtime-closure.v1",
        "entrypoints": list(RUNTIME_ENTRYPOINTS),
        "gate_files": list(RUNTIME_GATE_FILES),
        "local_tool_imports": [
            {"importer": importer, "dependency": dependency}
            for importer, dependency in sorted(local_tool_edges)
        ],
        "python": path_set_identity(python_paths),
        "rust": path_set_identity(rust_paths),
        "fixtures": path_set_identity(fixture_paths),
        "dependency_locks": list(RUNTIME_DEPENDENCY_LOCKS),
    }


def build_manifest(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    w0_manifest = _verify_reference_manifest(repository)
    selected = selected_paths(repository)
    files = []
    total_bytes = 0
    for relative in selected:
        path = _require_regular(repository, relative)
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "relative": relative,
                "size": size,
                "sha256": sha256_file(path),
                "mode": "0500" if relative in EXECUTABLE_FILES else "0400",
            }
        )
    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "source_contract": "manifest-driven-source-archive-w0-v1.1-w5-readmission-v3",
        "open_seed_domain": OPEN_DOMAIN,
        "open_seed_count": OPEN_SEED_COUNT,
        "protected_seed_values_opened": False,
        "w0_reference_manifest_sha256": sha256_file(
            repository / REFERENCE_MANIFEST_RELATIVE
        ),
        "w0_reference_manifest_canonical_sha256": w0_manifest["manifest_sha256"],
        "selection": {
            "root_files": list(ROOT_FILES),
            "tree_roots": list(TREE_ROOTS),
            "web_root": WEB_ROOT,
            "web_excluded_directories": sorted(WEB_EXCLUDED_DIRECTORIES),
            "web_excluded_suffixes": list(WEB_EXCLUDED_SUFFIXES),
            "exact_files": list(EXACT_FILES),
            "executable_files": sorted(EXECUTABLE_FILES),
        },
        "runtime_closure": _runtime_closure(repository, selected),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    result["document_sha256"] = document_sha256(result)
    return result


def render_manifest(repository: Path) -> str:
    return json.dumps(build_manifest(repository), indent=2, sort_keys=True) + "\n"


def verify_manifest(repository: Path, path: Path) -> dict[str, Any]:
    try:
        observed_bytes = path.read_bytes()
        observed = json.loads(observed_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceManifestError(f"cannot read source manifest: {error}") from error
    expected = build_manifest(repository)
    if observed != expected or observed_bytes != (
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    ).encode("ascii"):
        raise SourceManifestError("source manifest differs from deterministic selection")
    return {
        "valid": True,
        "document_sha256": expected["document_sha256"],
        "file_count": expected["file_count"],
        "total_bytes": expected["total_bytes"],
        "protected_seed_values_opened": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("render")
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "render":
            sys.stdout.write(render_manifest(arguments.repository))
        else:
            print(
                json.dumps(
                    verify_manifest(arguments.repository, arguments.manifest),
                    sort_keys=True,
                )
            )
    except (OSError, SourceManifestError, ValueError) as error:
        print(f"R2-MAP W0/W5 source manifest refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
