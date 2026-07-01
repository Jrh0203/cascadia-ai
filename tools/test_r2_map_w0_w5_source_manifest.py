from __future__ import annotations

import json
from pathlib import Path

import pytest
import r2_map_w0_w5_source_manifest as source_manifest


def _write(path: Path, content: bytes = b"source\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_compact_exporter_is_frozen_as_an_executable_source_object() -> None:
    assert "tools/r2_map_compact_dataset.py" in source_manifest.EXACT_FILES
    assert "tools/r2_map_compact_dataset.py" in source_manifest.EXECUTABLE_FILES


def test_training_contract_document_is_frozen_with_its_implementation() -> None:
    assert "docs/v2/TRAINING.md" in source_manifest.EXACT_FILES


def test_complete_source_keeps_both_w0_manifests_for_drift_rejection() -> None:
    assert (
        "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.json"
        in source_manifest.EXACT_FILES
    )
    assert (
        "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
        in source_manifest.EXACT_FILES
    )


def test_runtime_entrypoints_gates_and_dependency_locks_are_in_the_source_set() -> None:
    selected = set(source_manifest.ROOT_FILES) | set(source_manifest.EXACT_FILES)
    assert set(source_manifest.RUNTIME_ENTRYPOINTS).issubset(selected)
    assert set(source_manifest.RUNTIME_GATE_FILES).issubset(selected)
    assert set(source_manifest.RUNTIME_DEPENDENCY_LOCKS).issubset(selected)


def _repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(source_manifest, "ROOT_FILES", ("Cargo.toml",))
    monkeypatch.setattr(source_manifest, "TREE_ROOTS", ("src",))
    monkeypatch.setattr(source_manifest, "WEB_ROOT", "web")
    monkeypatch.setattr(source_manifest, "EXACT_FILES", ("w0-v1.1.json", "tool.py"))
    monkeypatch.setattr(source_manifest, "EXECUTABLE_FILES", frozenset({"tool.py"}))
    monkeypatch.setattr(source_manifest, "RUNTIME_ENTRYPOINTS", ())
    monkeypatch.setattr(source_manifest, "RUNTIME_GATE_FILES", ())
    monkeypatch.setattr(source_manifest, "RUNTIME_PYTHON_ROOTS", ())
    monkeypatch.setattr(source_manifest, "RUNTIME_RUST_ROOTS", ())
    monkeypatch.setattr(source_manifest, "RUNTIME_FIXTURE_ROOTS", ())
    monkeypatch.setattr(source_manifest, "RUNTIME_DEPENDENCY_LOCKS", ("Cargo.toml",))
    monkeypatch.setattr(source_manifest, "REFERENCE_MANIFEST_RELATIVE", "w0-v1.1.json")
    _write(repository / "Cargo.toml")
    _write(repository / "src/lib.rs")
    _write(repository / "src/__pycache__/ignored.pyc")
    _write(repository / "web/package.json")
    _write(repository / "web/dist/ignored.js")
    _write(repository / "web/compiler.tsbuildinfo")
    _write(repository / "tool.py")
    w0 = {
        "schema_id": "cascadia.r2-map.reference-panel-manifest.v1.1",
        "contract_revision": "sequential-public-market-v1.1",
        "manifest_sha256": "a" * 64,
        "panels": [
            {
                "panel_id": "open-performance-100",
                "definition": {
                    "seed_domain": source_manifest.OPEN_DOMAIN,
                    "seeds": list(range(100)),
                    "seed_domain_changed": False,
                    "predecessor_outcomes_opened": False,
                    "old_v1_outcomes_reused": False,
                },
            }
        ],
        "protected_seed_domains": [
            {"domain_id": "sealed", "opened": False, "seed_material_present": False}
        ],
        "protected_seed_handling": {
            "opening_authorized": False,
            "values_accepted_by_tool": False,
            "values_derivable_by_tool": False,
            "values_in_manifest": False,
        },
    }
    _write(repository / "w0-v1.1.json", json.dumps(w0).encode("ascii"))
    return repository


def test_manifest_is_exact_deterministic_and_excludes_build_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path, monkeypatch)
    first = source_manifest.build_manifest(repository)
    second = source_manifest.build_manifest(repository)
    assert first == second
    assert first["document_sha256"] == source_manifest.document_sha256(first)
    assert first["runtime_closure"] == {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.runtime-closure.v1",
        "entrypoints": [],
        "gate_files": [],
        "local_tool_imports": [],
        "python": {
            "file_count": 0,
            "paths_sha256": source_manifest.sha256_bytes(
                source_manifest.canonical_json([])
            ),
        },
        "rust": {
            "file_count": 0,
            "paths_sha256": source_manifest.sha256_bytes(
                source_manifest.canonical_json([])
            ),
        },
        "fixtures": {
            "file_count": 0,
            "paths_sha256": source_manifest.sha256_bytes(
                source_manifest.canonical_json([])
            ),
        },
        "dependency_locks": ["Cargo.toml"],
    }
    entries = {entry["relative"]: entry for entry in first["files"]}
    assert sorted(entries) == [
        "Cargo.toml",
        "src/lib.rs",
        "tool.py",
        "w0-v1.1.json",
        "web/package.json",
    ]
    assert entries["tool.py"]["mode"] == "0500"
    assert all(
        entry["mode"] == "0400" for name, entry in entries.items() if name != "tool.py"
    )


def test_verify_requires_byte_exact_canonical_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path, monkeypatch)
    rendered = source_manifest.render_manifest(repository)
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(rendered, encoding="ascii")
    assert source_manifest.verify_manifest(repository, manifest_path)["valid"] is True
    manifest_path.write_text(json.dumps(json.loads(rendered)), encoding="ascii")
    with pytest.raises(source_manifest.SourceManifestError, match="deterministic selection"):
        source_manifest.verify_manifest(repository, manifest_path)


def test_hidden_seed_boundary_or_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path, monkeypatch)
    # Exercise the safety primitive without requiring the full repository fixture.
    link = repository / "unsafe"
    link.symlink_to(repository / "Cargo.toml")
    with pytest.raises(source_manifest.SourceManifestError, match="non-symlink"):
        source_manifest._require_regular(repository, "unsafe")


def test_protected_domain_must_remain_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path, monkeypatch)
    manifest_path = repository / "w0-v1.1.json"
    value = json.loads(manifest_path.read_text(encoding="ascii"))
    value["protected_seed_domains"][0]["opened"] = True
    manifest_path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(source_manifest.SourceManifestError, match="protected seed"):
        source_manifest._verify_reference_manifest(repository)
