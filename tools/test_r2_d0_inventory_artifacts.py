from __future__ import annotations

import gzip
import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest
import r2_d0.storage as storage_module
from r2_d0.artifacts import (
    RegistryClient,
    acquire_core,
    atomic_install_bytes,
    frozen_homebrew_formula_projection,
    normalize_homebrew_formula_metadata,
    probe_context,
    scanner_oci_archive,
    smoke_oci_archive,
    validate_alpine_objects,
    validate_homebrew_formula_projection,
    validate_scanner_source_supply,
    verify_scanner_oci_archive,
)
from r2_d0.canonical import (
    CORE_IMAGE,
    FROZEN_RUNTIME,
    SCANNER_IMAGE,
    SMOKE_IMAGE,
    D0Error,
    canonical_json,
    sha256_bytes,
)
from r2_d0.inventory import (
    InventoryPolicy,
    compare_inventories,
    inventory_managed_homebrew_link,
    inventory_roots,
    podman_negative_control,
)
from r2_d0.storage import _campaign_apparent_size, _storage_write_probe


def _campaign_tree(tmp_path: Path) -> tuple[Path, int]:
    root = tmp_path / "r2-map-v1"
    (root / "cache/runs/run-a").mkdir(parents=True)
    (root / "cache/runs/run-b").mkdir(parents=True)
    (root / "build/build-a").mkdir(parents=True)
    (root / "toolchains/toolchain-a").mkdir(parents=True)
    return root, root.stat().st_dev


def test_campaign_size_accepts_only_contained_same_run_symlinks(tmp_path: Path) -> None:
    root, device = _campaign_tree(tmp_path)
    target = root / "cache/runs/run-a/payload.bin"
    target.write_bytes(b"payload")
    link = root / "cache/runs/run-a/payload-link"
    link.symlink_to("payload.bin")

    apparent, entries = _campaign_apparent_size(root, device)

    assert apparent == len(b"payload") + link.lstat().st_size
    assert entries == 10


@pytest.mark.parametrize(
    ("relative", "target", "message"),
    (
        (
            "cache/runs/run-b/cross-run",
            "../run-a/payload.bin",
            "escapes its audited run boundary",
        ),
        ("cache/runs/run-a/dangling", "missing.bin", "dangling symlink"),
        (
            "toolchains/escape",
            "../cache/runs/run-a/payload.bin",
            "escapes its audited run boundary",
        ),
    ),
)
def test_campaign_size_rejects_symlink_escape_and_dangling(
    tmp_path: Path,
    relative: str,
    target: str,
    message: str,
) -> None:
    root, device = _campaign_tree(tmp_path)
    (root / "cache/runs/run-a/payload.bin").write_bytes(b"payload")
    (root / relative).symlink_to(target)

    with pytest.raises(D0Error, match=message):
        _campaign_apparent_size(root, device)


def test_campaign_size_rejects_unauthorized_and_cross_device_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, device = _campaign_tree(tmp_path)
    unauthorized = root / "control-link"
    unauthorized.symlink_to("cache/runs/run-a")
    with pytest.raises(D0Error, match="unauthorized symlink"):
        _campaign_apparent_size(root, device)
    unauthorized.unlink()

    target = root / "toolchains/toolchain-a/payload.bin"
    target.write_bytes(b"payload")
    (root / "toolchains/payload-link").symlink_to("toolchain-a/payload.bin")
    real_stat = storage_module.os.stat

    def cross_device(path: object, *args: object, **kwargs: object):
        result = real_stat(path, *args, **kwargs)
        if Path(path) == target:
            return type("CrossDeviceStat", (), {"st_dev": device + 1})()
        return result

    monkeypatch.setattr(storage_module.os, "stat", cross_device)
    with pytest.raises(D0Error, match="crosses a device boundary"):
        _campaign_apparent_size(root, device)


def test_no_follow_inventory_records_symlink_and_hashes_regular_files(tmp_path: Path) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"payload")
    (root / "link").symlink_to(payload)
    report = inventory_roots([root], label="fixture")
    entries = {item["relative"]: item for item in report["roots"][0]["entries"]}
    assert entries["payload.bin"]["content_sha256"] == sha256_bytes(b"payload")
    assert entries["link"]["type"] == "symlink"
    assert entries["link"]["target"] == str(payload)


def test_large_file_inventory_uses_stable_samples(tmp_path: Path) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    large = root / "large.bin"
    large.write_bytes(bytes(range(256)) * 32)
    report = inventory_roots(
        [root],
        label="sampled",
        policy=InventoryPolicy(full_hash_limit=128, sample_bytes=64),
    )
    entry = next(item for item in report["roots"][0]["entries"] if item["relative"] == "large.bin")
    assert entry["content_sha256"] is None
    assert len(entry["sample_offsets"]) == 5
    assert len(entry["sample_sha256"]) == 64


def test_inventory_comparison_ignores_collection_time_but_detects_drift(tmp_path: Path) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"before")
    before = inventory_roots([root], label="before")
    unchanged = inventory_roots([root], label="after")
    assert compare_inventories(before, unchanged, label="stable")["status"] == "pass"
    payload.write_bytes(b"after")
    changed = inventory_roots([root], label="changed")
    assert compare_inventories(before, changed, label="drift")["status"] == "fail"


def test_inventory_rejects_duplicate_root_and_symlink_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "real"
    root.mkdir()
    with pytest.raises(D0Error, match="duplicated"):
        inventory_roots([root, root], label="duplicate")
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    root_report = inventory_roots([link], label="symlink-root")
    assert root_report["roots"][0]["entries"] == [
        {
            **root_report["roots"][0]["entries"][0],
            "relative": ".",
            "type": "symlink",
            "target": str(root),
        }
    ]
    with pytest.raises(D0Error, match="symlink"):
        inventory_roots([link / "child"], label="escape")


def test_homebrew_shaped_final_symlink_roots_are_recorded_without_traversal(
    tmp_path: Path,
) -> None:
    cellar = tmp_path / "Cellar/podman/5.5.2"
    cellar.mkdir(parents=True)
    executable = cellar / "bin/podman"
    executable.parent.mkdir()
    executable.write_bytes(b"podman")
    opt = tmp_path / "opt/podman"
    opt.parent.mkdir()
    opt.symlink_to(cellar, target_is_directory=True)
    binary = tmp_path / "bin/podman"
    binary.parent.mkdir()
    binary.symlink_to(executable)
    report = inventory_roots([opt, binary], label="homebrew-links")
    assert all(root["entries"][0]["type"] == "symlink" for root in report["roots"])
    assert report["totals"]["entries"] == 2


def _managed_buildx_fixture(tmp_path: Path) -> dict[str, object]:
    cellar = tmp_path / "Cellar"
    keg = cellar / "docker-buildx/0.35.0"
    managed_target = keg / "lib/docker"
    plugins = managed_target / "cli-plugins"
    plugins.mkdir(parents=True)
    binary = keg / "bin/docker-buildx"
    binary.parent.mkdir()
    binary.write_bytes(b"frozen-buildx")
    binary.chmod(0o555)
    receipt = keg / "INSTALL_RECEIPT.json"
    receipt.write_bytes(b'{"source":"bottle"}')
    public = tmp_path / "lib/docker"
    public.parent.mkdir()
    public.symlink_to("../Cellar/docker-buildx/0.35.0/lib/docker", target_is_directory=True)
    plugin = plugins / "docker-buildx"
    plugin.symlink_to("../../../bin/docker-buildx")
    return {
        "requested_path": public / "cli-plugins/docker-buildx",
        "managed_link": public,
        "cellar_root": cellar,
        "receipt_sha256": sha256_bytes(receipt.read_bytes()),
        "binary_sha256": sha256_bytes(binary.read_bytes()),
        "plugin": plugin,
        "managed_link_target": "../Cellar/docker-buildx/0.35.0/lib/docker",
        "requested_link_target": "../../../bin/docker-buildx",
    }


def _managed_buildx_inventory(fixture: dict[str, object]) -> dict[str, object]:
    return inventory_managed_homebrew_link(
        fixture["requested_path"],
        managed_link=fixture["managed_link"],
        cellar_root=fixture["cellar_root"],
        formula="docker-buildx",
        version="0.35.0",
        managed_target_relative=Path("lib/docker"),
        requested_suffix=Path("cli-plugins/docker-buildx"),
        installed_file_relative=Path("bin/docker-buildx"),
        managed_link_target=fixture["managed_link_target"],
        requested_link_target=fixture["requested_link_target"],
        install_receipt_sha256=fixture["receipt_sha256"],
        installed_file_sha256=fixture["binary_sha256"],
        label="fixture-buildx",
    )


def test_managed_homebrew_link_records_public_link_and_resolved_keg(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    report = _managed_buildx_inventory(fixture)
    assert report["status"] == "pass"
    assert report["identity_stable"] is True
    assert report["before"] == report["after"]
    assert report["before"]["public_symlink"]["type"] == "symlink"
    assert report["before"]["requested_symlink"]["type"] == "symlink"
    assert report["before"]["resolved_installed_file"]["content_sha256"] == fixture[
        "binary_sha256"
    ]


def test_managed_homebrew_link_rejects_absolute_public_target(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    managed = fixture["managed_link"]
    managed.unlink()
    absolute = str(tmp_path / "Cellar/docker-buildx/0.35.0/lib/docker")
    managed.symlink_to(absolute)
    fixture["managed_link_target"] = absolute
    with pytest.raises(D0Error, match="authorization differs"):
        _managed_buildx_inventory(fixture)


def test_managed_homebrew_link_rejects_keg_escape(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    plugin = fixture["plugin"]
    plugin.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"escape")
    escape_target = os.path.relpath(outside, plugin.parent)
    plugin.symlink_to(escape_target)
    fixture["requested_link_target"] = escape_target
    with pytest.raises(D0Error, match="escapes its pinned keg"):
        _managed_buildx_inventory(fixture)


def test_managed_homebrew_link_rejects_wrong_receipt_or_binary_hash(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    fixture["receipt_sha256"] = "0" * 64
    with pytest.raises(D0Error, match="receipt or installed file identity differs"):
        _managed_buildx_inventory(fixture)


def test_managed_homebrew_link_rejects_extra_symlink_ancestor(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    plugin = fixture["plugin"]
    plugins = plugin.parent
    plugin.unlink()
    plugins.rmdir()
    actual = plugins.with_name("actual-plugins")
    actual.mkdir()
    (actual / "docker-buildx").symlink_to("../../../bin/docker-buildx")
    plugins.symlink_to(actual, target_is_directory=True)
    with pytest.raises(D0Error, match="extra symlink"):
        _managed_buildx_inventory(fixture)


def test_managed_homebrew_link_rejects_cycle(tmp_path: Path) -> None:
    fixture = _managed_buildx_fixture(tmp_path)
    managed = fixture["managed_link"]
    managed.unlink()
    managed.symlink_to("docker")
    fixture["managed_link_target"] = "docker"
    with pytest.raises(D0Error, match="dangling or cyclic"):
        _managed_buildx_inventory(fixture)


def test_post_cleanup_podman_negative_control_is_semantic_not_old_vm_identity(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    config = home / ".config/containers"
    local = home / ".local/share/containers"
    for path, mode in (
        (config / "podman/machine/applehv", 0o755),
        (local / "podman/machine/applehv/cache", 0o755),
        (local / "cache", 0o700),
    ):
        path.mkdir(parents=True, mode=mode)
        path.chmod(mode)
    (config / "podman-connections.json").write_bytes(
        b'{"Connection":{},"Farm":{}}\n'
    )
    (config / "podman-connections.json.lock").write_bytes(b"")
    for path in (
        config / "podman-connections.json",
        config / "podman-connections.json.lock",
    ):
        path.chmod(0o644)
    for path in (
        config,
        config / "podman",
        config / "podman/machine",
        local,
        local / "podman",
        local / "podman/machine",
        local / "podman/machine/applehv",
    ):
        path.chmod(0o755)
    receipt = podman_negative_control(
        home,
        installation_paths=(tmp_path / "formula", tmp_path / "cli"),
    )
    assert receipt["status"] == "pass"
    assert receipt["semantic"]["machine_records"] == 0
    assert receipt["semantic"]["storage_payload_files"] == 0

    disk = local / "podman/machine/applehv/podman-machine-default-arm64.raw"
    disk.write_bytes(b"forbidden")
    with pytest.raises(D0Error, match="unexpected path"):
        podman_negative_control(
            home,
            installation_paths=(tmp_path / "formula", tmp_path / "cli"),
        )


def test_probe_context_is_exact_reproducible_and_project_free() -> None:
    first, first_receipt = probe_context()
    second, second_receipt = probe_context()
    assert first == second
    assert first_receipt == second_receipt
    assert first_receipt["project_code_present"] is False
    assert first_receipt["protected_seed_values_opened"] is False


def _synthetic_alpine(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes, bytes, bytes]:
    config = canonical_json({"architecture": "arm64", "os": "linux"})
    layer = b"synthetic-gzip-layer"
    config_digest = f"sha256:{sha256_bytes(config)}"
    layer_digest = f"sha256:{sha256_bytes(layer)}"
    manifest = canonical_json(
        {
            "schemaVersion": 2,
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": layer_digest,
                    "size": len(layer),
                }
            ],
        }
    )
    manifest_digest = f"sha256:{sha256_bytes(manifest)}"
    source_index = canonical_json(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    "digest": manifest_digest,
                    "platform": {"architecture": "arm64", "os": "linux", "variant": "v8"},
                }
            ],
        }
    )
    values = {
        "index_digest": f"sha256:{sha256_bytes(source_index)}",
        "manifest_digest": manifest_digest,
        "config_digest": config_digest,
        "layer_digest": layer_digest,
        "manifest_size": len(manifest),
        "config_size": len(config),
        "layer_size": len(layer),
    }
    for key, value in values.items():
        monkeypatch.setitem(SMOKE_IMAGE, key, value)
    return source_index, manifest, config, layer


def test_smoke_oci_archive_validates_every_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    objects = _synthetic_alpine(monkeypatch)
    validate_alpine_objects(*objects)
    archive, receipt = smoke_oci_archive(*objects)
    archive_again, receipt_again = smoke_oci_archive(*objects)
    assert archive == archive_again
    assert receipt == receipt_again
    assert receipt["archive_sha256"] == sha256_bytes(archive)
    changed = bytearray(objects[3])
    changed[0] ^= 1
    with pytest.raises(D0Error, match="layer digest"):
        validate_alpine_objects(*objects[:3], bytes(changed))


def test_registry_client_is_digest_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    objects = _synthetic_alpine(monkeypatch)
    responses = [canonical_json({"token": "token"}), *objects]
    requests: list[str] = []

    def opener(request: object) -> bytes:
        requests.append(request.full_url)  # type: ignore[attr-defined]
        return responses.pop(0)

    assert RegistryClient(opener).acquire() == objects
    assert requests[0].startswith("https://auth.docker.io/token?")
    assert all("library/alpine" in value for value in requests[1:])
    assert SMOKE_IMAGE["index_digest"] in requests[1]


def test_homebrew_metadata_regeneration_accepts_same_semantics_and_rejects_drift() -> None:
    formula = "colima"
    frozen = FROZEN_RUNTIME[formula]
    bottle_url = (
        f"https://ghcr.io/v2/homebrew/core/{formula}/blobs/sha256:{frozen['bottle_sha256']}"
    )
    metadata = {
        "name": formula,
        "tap": "homebrew/core",
        "license": frozen["license"],
        "versions": {"stable": frozen["version"]},
        "revision": frozen["revision"],
        "dependencies": list(frozen["dependencies"]),
        "ruby_source_path": frozen["formula_path"],
        "ruby_source_checksum": {"sha256": frozen["ruby_source_sha256"]},
        "urls": {
            "stable": {
                "url": frozen["source_url"],
                "tag": frozen["source_tag"],
                "revision": frozen["source_revision"],
                "checksum": frozen["source_checksum"],
            }
        },
        "bottle": {
            "stable": {
                "files": {
                    frozen["bottle_tag"]: {
                        "url": bottle_url,
                        "sha256": frozen["bottle_sha256"],
                    }
                }
            }
        },
        "tap_git_head": "f" * 40,
        "generated_date": "2099-01-01",
        "analytics": {"regenerated": True},
    }
    expected = frozen_homebrew_formula_projection(formula)
    assert normalize_homebrew_formula_metadata(canonical_json(metadata), formula) == expected
    projection = canonical_json(expected)
    assert validate_homebrew_formula_projection(projection, formula) == expected
    metadata["license"] = "GPL-3.0-only"
    with pytest.raises(D0Error, match="semantics drifted"):
        normalize_homebrew_formula_metadata(canonical_json(metadata), formula)


def test_atomic_install_and_core_acquisition_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"core-fixture"
    monkeypatch.setitem(CORE_IMAGE, "size", len(payload))
    monkeypatch.setitem(CORE_IMAGE, "sha256", sha256_bytes(payload))
    monkeypatch.setitem(CORE_IMAGE, "sha512", hashlib.sha512(payload).hexdigest())
    destination = tmp_path / "core.raw.gz"
    receipt = acquire_core(destination, opener=lambda _request: payload)
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o400
    assert receipt["sha256"] == sha256_bytes(payload)
    recovered = atomic_install_bytes(destination, payload)
    assert recovered["status"] == "already-installed"


def test_storage_write_probe_fsyncs_renames_rereads_and_cleans(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    receipt = _storage_write_probe(control, {"host": "john2", "volume": "internal-apfs"})
    assert receipt["status"] == "pass"
    assert receipt["atomic_rename"] is True
    assert receipt["no_follow_reread"] is True
    assert receipt["cleanup_unlink"] is True
    assert list(control.iterdir()) == []


def _synthetic_scanner(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, ...]:
    layer_output = io.BytesIO()
    scanner_payload = b"synthetic-static-scanner"
    with tarfile.open(fileobj=layer_output, mode="w|", format=tarfile.USTAR_FORMAT) as layer_tar:
        scanner = tarfile.TarInfo("bin/syft-scanner")
        scanner.size = len(scanner_payload)
        scanner.mode = 0o755
        scanner.uid = scanner.gid = scanner.mtime = 0
        layer_tar.addfile(scanner, io.BytesIO(scanner_payload))
    layer_plain = layer_output.getvalue()
    layer = gzip.compress(layer_plain, mtime=0)
    config = canonical_json(
        {
            "architecture": "arm64",
            "os": "linux",
            "config": {"Entrypoint": ["/bin/syft-scanner"]},
            "rootfs": {
                "type": "layers",
                "diff_ids": [f"sha256:{sha256_bytes(layer_plain)}"],
            },
        }
    )
    manifest = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": f"sha256:{sha256_bytes(config)}",
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": f"sha256:{sha256_bytes(layer)}",
                    "size": len(layer),
                }
            ],
        }
    )
    manifest_digest = f"sha256:{sha256_bytes(manifest)}"
    spdx = canonical_json(
        {
            "_type": "https://in-toto.io/Statement/v0.1",
            "subject": [{"digest": {"sha256": manifest_digest.split(":", 1)[1]}}],
            "predicateType": "https://spdx.dev/Document",
            "predicate": {"spdxVersion": "SPDX-2.3"},
        }
    )
    provenance = canonical_json(
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"digest": {"sha256": manifest_digest.split(":", 1)[1]}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "resolvedDependencies": [
                        {
                            "uri": (
                                "https://github.com/docker/buildkit-syft-scanner.git"
                                "#refs/tags/v1.11.0"
                            ),
                            "digest": {"sha1": "a" * 40},
                        }
                    ]
                }
            },
        }
    )
    attestation = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": "application/vnd.docker.attestation.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.empty.v1+json",
                "digest": f"sha256:{sha256_bytes(b'{}')}",
                "size": 2,
                "data": "e30=",
            },
            "layers": [
                {
                    "mediaType": "application/vnd.in-toto+json",
                    "digest": f"sha256:{sha256_bytes(spdx)}",
                    "size": len(spdx),
                    "annotations": {"in-toto.io/predicate-type": "https://spdx.dev/Document"},
                },
                {
                    "mediaType": "application/vnd.in-toto+json",
                    "digest": f"sha256:{sha256_bytes(provenance)}",
                    "size": len(provenance),
                    "annotations": {"in-toto.io/predicate-type": "https://slsa.dev/provenance/v1"},
                },
            ],
            "subject": {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": manifest_digest,
                "size": len(manifest),
            },
        }
    )
    attestation_digest = f"sha256:{sha256_bytes(attestation)}"
    source_index = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": manifest_digest,
                    "size": len(manifest),
                    "platform": {"architecture": "arm64", "os": "linux"},
                },
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": attestation_digest,
                    "size": len(attestation),
                    "platform": {"architecture": "unknown", "os": "unknown"},
                    "annotations": {
                        "vnd.docker.reference.digest": manifest_digest,
                        "vnd.docker.reference.type": "attestation-manifest",
                    },
                },
            ],
        }
    )
    replacements = {
        "source_revision": "a" * 40,
        "index_digest": f"sha256:{sha256_bytes(source_index)}",
        "index_size": len(source_index),
        "manifest_digest": manifest_digest,
        "manifest_size": len(manifest),
        "config_digest": f"sha256:{sha256_bytes(config)}",
        "config_size": len(config),
        "layer_digest": f"sha256:{sha256_bytes(layer)}",
        "layer_size": len(layer),
        "diff_id": f"sha256:{sha256_bytes(layer_plain)}",
        "attestation_manifest_digest": attestation_digest,
        "attestation_manifest_size": len(attestation),
        "attestation_config_digest": f"sha256:{sha256_bytes(b'{}')}",
        "attestation_config_size": 2,
        "spdx_digest": f"sha256:{sha256_bytes(spdx)}",
        "spdx_size": len(spdx),
        "provenance_digest": f"sha256:{sha256_bytes(provenance)}",
        "provenance_size": len(provenance),
    }
    for key, value in replacements.items():
        monkeypatch.setitem(SCANNER_IMAGE, key, value)
    return source_index, manifest, config, layer, attestation, spdx, provenance


def test_scanner_oci_binds_image_attestations_and_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = _synthetic_scanner(monkeypatch)
    archive, receipt = scanner_oci_archive(*objects)
    verified = verify_scanner_oci_archive(archive)
    assert verified["manifest_digest"] == SCANNER_IMAGE["manifest_digest"]
    assert verified["manifest_size"] == SCANNER_IMAGE["manifest_size"]
    assert verified["config_digest"] == SCANNER_IMAGE["config_digest"]
    assert verified["config_size"] == SCANNER_IMAGE["config_size"]
    assert verified["layer_digest"] == SCANNER_IMAGE["layer_digest"]
    assert verified["layer_size"] == SCANNER_IMAGE["layer_size"]
    assert receipt["spdx_digest"] == SCANNER_IMAGE["spdx_digest"]
    with pytest.raises(D0Error):
        verify_scanner_oci_archive(archive + b"tamper")
    changed = bytearray(objects[-1])
    changed[-2] ^= 1
    with pytest.raises(D0Error, match="provenance attestation identity"):
        scanner_oci_archive(*objects[:-1], bytes(changed))


def test_scanner_source_supply_binds_top_level_license_and_dockerfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    license_bytes = b"Apache License\n"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in (
            ("scanner/LICENSE", license_bytes),
            (
                "scanner/Dockerfile",
                b"# Apache License, Version 2.0\nCOPY scanner /bin/syft-scanner\n",
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    source = output.getvalue()
    monkeypatch.setitem(SCANNER_IMAGE, "source_archive_size", len(source))
    monkeypatch.setitem(SCANNER_IMAGE, "source_archive_sha256", sha256_bytes(source))
    monkeypatch.setitem(SCANNER_IMAGE, "license_size", len(license_bytes))
    monkeypatch.setitem(SCANNER_IMAGE, "license_sha256", sha256_bytes(license_bytes))
    assert validate_scanner_source_supply(source, license_bytes)["status"] == "pass"
    with pytest.raises(D0Error, match="source or license identity"):
        validate_scanner_source_supply(source, license_bytes + b"tamper")
