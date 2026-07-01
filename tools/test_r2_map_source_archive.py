from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest
import r2_map_source_archive as source_archive


def _set_xattr(path: Path, name: str = "com.openai.r2-map-test") -> None:
    subprocess.run(
        ["/usr/bin/xattr", "-w", name, "1", str(path)],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _manifest(entries: list[dict[str, object]]) -> bytes:
    value: dict[str, object] = {
        "schema_id": source_archive.MANIFEST_SCHEMA,
        "campaign_id": "test",
        "protected_seed_values_opened": False,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size"]) for entry in entries),
        "files": entries,
    }
    value["document_sha256"] = source_archive._sha256(source_archive.canonical_json(value))
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def _entry(relative: str, payload: bytes, mode: str = "0400") -> dict[str, object]:
    return {
        "relative": relative,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": mode,
    }


def _fixture_manifest(repository: Path) -> bytes:
    plain = b"manifest exact\n"
    executable = b"#!/bin/sh\nexit 0\n"
    (repository / "bin").mkdir(mode=0o700)
    (repository / "a.txt").write_bytes(plain)
    (repository / "bin/tool").write_bytes(executable)
    return _manifest([_entry("a.txt", plain), _entry("bin/tool", executable, "0500")])


def _archive_member(
    name: str,
    payload: bytes,
    *,
    mode: int = 0o400,
    pax_headers: dict[str, str] | None = None,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.mtime = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = pax_headers or {}
    return info


def _custom_archive(
    members: list[tuple[tarfile.TarInfo, bytes]],
    *,
    archive_format: int = tarfile.USTAR_FORMAT,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=archive_format) as archive:
        for info, payload in members:
            archive.addfile(info, io.BytesIO(payload) if info.isfile() else None)
    return output.getvalue()


def test_archive_is_deterministic_regular_only_and_tree_exact(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    manifest = _fixture_manifest(repository)
    first = io.BytesIO()
    second = io.BytesIO()
    source_archive.create_archive(repository, manifest, first)
    source_archive.create_archive(repository, manifest, second)
    assert first.getvalue() == second.getvalue()

    report = source_archive.verify_archive_bytes(manifest, first.getvalue())
    assert report["status"] == "valid"
    assert report["member_count"] == 2
    assert report["regular_only"] is True
    assert report["pax_or_extended_headers"] is False
    assert len(first.getvalue()) % source_archive.RECORD_SIZE == 0

    extracted = tmp_path / "extracted"
    extracted.mkdir(mode=0o700)
    previous_umask = os.umask(0o077)
    try:
        with tarfile.open(fileobj=io.BytesIO(first.getvalue()), mode="r:") as archive:
            archive.extractall(extracted)
    finally:
        os.umask(previous_umask)
    tree = source_archive.verify_tree(extracted, manifest)
    assert tree["status"] == "tree-valid"
    assert (extracted / "a.txt").stat().st_mode & 0o777 == 0o400
    assert (extracted / "bin/tool").stat().st_mode & 0o777 == 0o500


def test_rejects_v40_appledouble_regular_member_and_binds_audit_fixture(
    tmp_path: Path,
) -> None:
    audit = json.loads(
        Path("tests/fixtures/r2_map/rejected-v40-source-archive-characteristics.json").read_text()
    )
    assert audit["source_tar_regular_member_count"] == 887
    assert audit["source_manifest_file_count"] == 699
    assert audit["extra_member_count"] == 188
    assert audit["appledouble_magic_hex"] == "00051607"
    assert audit["mode_mismatch_count"] == 699
    assert (
        audit["transaction_manifest_raw_sha256"]
        == "e2d7ea4c3e5cffd7eff56d0c91b1c204def1fafc0506a8ee4c13cc41fbcaa817"
    )
    assert (
        audit["transaction_manifest_canonical_sha256"]
        == "b68a12fef70cac317b2ce12bf6da720af2aa2a60428602579724f10b2b4833e6"
    )
    assert (
        audit["commit_receipt_raw_sha256"]
        == "08d719a81e2f9d3b51238512e088f0f430af5fca5f70d7e2bcb03fbb5bdb20b3"
    )
    assert (
        audit["commit_receipt_canonical_sha256"]
        == "9e6c0fb988637f45230f213a34a49ee91fa0343ca899cdf7917f4c7550e487c4"
    )

    payload = b"source\n"
    manifest = _manifest([_entry("a.txt", payload)])
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        archive.addfile(_archive_member("a.txt", payload), io.BytesIO(payload))
        appledouble = bytes.fromhex(audit["appledouble_magic_hex"]) + bytes(
            audit["appledouble_member_bytes"] - 4
        )
        archive.addfile(
            _archive_member("._a.txt", appledouble),
            io.BytesIO(appledouble),
        )
    with pytest.raises(source_archive.SourceArchiveError, match="AppleDouble"):
        source_archive.verify_archive_bytes(manifest, output.getvalue())


def test_rejects_pax_xattr_control_header() -> None:
    payload = b"source\n"
    manifest = _manifest([_entry("a.txt", payload)])
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(
            _archive_member(
                "a.txt",
                payload,
                pax_headers={"SCHILY.xattr.com.apple.provenance": "opaque"},
            ),
            io.BytesIO(payload),
        )
    with pytest.raises(source_archive.SourceArchiveError, match="PAX/GNU/link/control"):
        source_archive.verify_archive_bytes(manifest, output.getvalue())


def test_rejects_mode_drift_extra_tree_files_and_xattrs(tmp_path: Path) -> None:
    payload = b"source\n"
    manifest = _manifest([_entry("a.txt", payload)])
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        archive.addfile(_archive_member("a.txt", payload, mode=0o644), io.BytesIO(payload))
    with pytest.raises(source_archive.SourceArchiveError, match="mode differs"):
        source_archive.verify_archive_bytes(manifest, output.getvalue())

    repository = tmp_path / "tree"
    repository.mkdir(mode=0o700)
    source = repository / "a.txt"
    source.write_bytes(payload)
    source.chmod(0o400)
    (repository / "extra").write_bytes(b"no\n")
    with pytest.raises(source_archive.SourceArchiveError, match="extra file"):
        source_archive.verify_tree(repository, manifest)
    (repository / "extra").unlink()
    source.chmod(0o600)
    _set_xattr(source)
    source.chmod(0o400)
    with pytest.raises(source_archive.SourceArchiveError, match="non-allowlisted"):
        source_archive.verify_tree(repository, manifest)


def test_rejects_order_duplicate_directory_link_gnu_and_metadata() -> None:
    first = b"a\n"
    second = b"b\n"
    manifest = _manifest([_entry("a", first), _entry("b", second)])

    wrong_order = _custom_archive(
        [(_archive_member("b", second), second), (_archive_member("a", first), first)]
    )
    with pytest.raises(source_archive.SourceArchiveError, match="path/order"):
        source_archive.verify_archive_bytes(manifest, wrong_order)

    duplicate = _custom_archive(
        [(_archive_member("a", first), first), (_archive_member("a", first), first)]
    )
    with pytest.raises(source_archive.SourceArchiveError, match="path/order"):
        source_archive.verify_archive_bytes(manifest, duplicate)

    directory = tarfile.TarInfo("b")
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o400
    directory.uid = directory.gid = directory.mtime = 0
    with pytest.raises(source_archive.SourceArchiveError, match="PAX/GNU/link/control"):
        source_archive.verify_archive_bytes(
            manifest,
            _custom_archive([(_archive_member("a", first), first), (directory, b"")]),
        )

    link = tarfile.TarInfo("b")
    link.type = tarfile.SYMTYPE
    link.linkname = "a"
    link.mode = 0o400
    link.uid = link.gid = link.mtime = 0
    with pytest.raises(source_archive.SourceArchiveError, match="PAX/GNU/link/control"):
        source_archive.verify_archive_bytes(
            manifest,
            _custom_archive([(_archive_member("a", first), first), (link, b"")]),
        )

    gnu_name = "g" * 110
    with pytest.raises(source_archive.SourceArchiveError, match=r"PAX/GNU|non-USTAR"):
        source_archive.verify_archive_bytes(
            _manifest([_entry(gnu_name[:100], first)]),
            _custom_archive(
                [(_archive_member(gnu_name, first), first)],
                archive_format=tarfile.GNU_FORMAT,
            ),
        )

    metadata = _archive_member("a", first)
    metadata.uid = 501
    metadata.mtime = 7
    with pytest.raises(source_archive.SourceArchiveError, match="metadata is not normalized"):
        source_archive.verify_archive_bytes(
            _manifest([_entry("a", first)]),
            _custom_archive([(metadata, first)]),
        )


def test_rejects_checksum_padding_trailing_records_and_control_paths(tmp_path: Path) -> None:
    payload = b"source\n"
    manifest = _manifest([_entry("a", payload)])
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    (repository / "a").write_bytes(payload)
    output = io.BytesIO()
    source_archive.create_archive(repository, manifest, output)
    valid = output.getvalue()

    checksum = bytearray(valid)
    checksum[0] ^= 1
    with pytest.raises(source_archive.SourceArchiveError, match="checksum"):
        source_archive.verify_archive_bytes(manifest, bytes(checksum))

    padding = bytearray(valid)
    padding[source_archive.BLOCK_SIZE + len(payload)] = 1
    with pytest.raises(source_archive.SourceArchiveError, match="block padding"):
        source_archive.verify_archive_bytes(manifest, bytes(padding))

    trailing = bytearray(valid)
    trailing[-1] = 1
    with pytest.raises(source_archive.SourceArchiveError, match="trailing records"):
        source_archive.verify_archive_bytes(manifest, bytes(trailing))

    with pytest.raises(source_archive.SourceArchiveError, match="canonical relative"):
        source_archive._safe_relative("line\nbreak")
    with pytest.raises(source_archive.SourceArchiveError, match="archive byte bound"):
        source_archive.load_manifest_bytes(
            _manifest(
                [
                    {
                        "relative": "huge",
                        "size": source_archive.MAX_ARCHIVE_BYTES,
                        "sha256": "0" * 64,
                        "mode": "0400",
                    }
                ]
            )
        )


def test_verify_tree_rejects_root_symlink_and_directory_xattrs(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    manifest = _fixture_manifest(repository)
    (repository / "a.txt").chmod(0o400)
    (repository / "bin/tool").chmod(0o500)
    alias = tmp_path / "alias"
    alias.symlink_to(repository, target_is_directory=True)
    with pytest.raises(source_archive.SourceArchiveError, match="direct regular directory"):
        source_archive.verify_tree(alias, manifest)

    _set_xattr(repository / "bin")
    with pytest.raises(source_archive.SourceArchiveError, match="non-allowlisted"):
        source_archive.verify_tree(repository, manifest)


def test_verify_tree_rejects_nonprivate_directory_mode(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    manifest = _fixture_manifest(repository)
    (repository / "a.txt").chmod(0o400)
    (repository / "bin/tool").chmod(0o500)
    (repository / "bin").chmod(0o755)
    with pytest.raises(source_archive.SourceArchiveError, match="mode is not private"):
        source_archive.verify_tree(repository, manifest)


def test_darwin_xattr_enumerator_handles_empty_nonempty_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "xattrs"
    path.write_bytes(b"x\n")
    monkeypatch.delattr(source_archive.os, "listxattr", raising=False)
    baseline = set(source_archive._xattr_names_no_follow(path))
    assert baseline.issubset(source_archive.ALLOWED_EXTRACTED_XATTRS)
    _set_xattr(path)
    assert set(source_archive._xattr_names_no_follow(path)) == baseline | {
        b"com.openai.r2-map-test"
    }
    with pytest.raises(source_archive.SourceArchiveError, match="cannot open path"):
        source_archive._xattr_names_no_follow(tmp_path / "absent")


def test_verify_tree_accepts_and_reports_only_host_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    manifest = _fixture_manifest(repository)
    (repository / "a.txt").chmod(0o400)
    (repository / "bin/tool").chmod(0o500)
    monkeypatch.setattr(
        source_archive,
        "_xattr_names_fd",
        lambda _descriptor, _label: (b"com.apple.provenance",),
    )
    report = source_archive.verify_tree(repository, manifest)
    assert report["provenance_path_count"] == 4
    assert report["unexpected_extended_attributes_absent"] is True


def test_create_rejects_root_and_parent_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (real / "a").write_bytes(b"a\n")
    manifest = _manifest([_entry("a", b"a\n")])
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(source_archive.SourceArchiveError, match="direct regular directory"):
        source_archive.create_archive(alias, manifest, io.BytesIO())

    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    (repository / "real").mkdir()
    (repository / "real/a").write_bytes(b"a\n")
    (repository / "parent").symlink_to(repository / "real", target_is_directory=True)
    parent_manifest = _manifest([_entry("parent/a", b"a\n")])
    with pytest.raises(source_archive.SourceArchiveError, match="parent cannot be opened"):
        source_archive.create_archive(repository, parent_manifest, io.BytesIO())


def test_manifest_rejects_file_directory_prefix_collision() -> None:
    with pytest.raises(source_archive.SourceArchiveError, match="path collision"):
        source_archive.load_manifest_bytes(_manifest([_entry("a", b"a\n"), _entry("a/b", b"b\n")]))


def test_cleanup_pytest_basetemp_is_bounded_no_follow_and_idempotent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    basetemp = parent / source_archive.PYTEST_BASETEMP_NAME
    nested = basetemp / "nested" / "readonly"
    nested.mkdir(parents=True, mode=0o700)
    model = nested / "model.bin"
    model.write_bytes(b"model\n")
    model.chmod(0o400)

    external_directory = tmp_path / "external"
    external_directory.mkdir(mode=0o700)
    external_payload = external_directory / "keep.txt"
    external_payload.write_bytes(b"outside\n")
    external_mode = stat.S_IMODE(external_payload.stat().st_mode)
    (nested / "external-link").symlink_to(external_directory, target_is_directory=True)
    (nested / "dangling-link").symlink_to(tmp_path / "missing-sensitive-target")
    hardlink = nested / "external-hardlink"
    os.link(external_payload, hardlink)

    nested.chmod(0o500)
    nested.parent.chmod(0o500)
    basetemp.chmod(0o500)

    report = source_archive.cleanup_pytest_basetemp(parent)
    assert report == {
        "schema_id": source_archive.CLEANUP_SCHEMA,
        "status": "removed",
        "basetemp_name": source_archive.PYTEST_BASETEMP_NAME,
        "removed": True,
        "directories_removed": 3,
        "regular_files_removed": 2,
        "regular_file_bytes_removed": len(b"model\n") + len(b"outside\n"),
        "symlinks_removed": 2,
        "no_links_followed": True,
    }
    with pytest.raises(FileNotFoundError):
        basetemp.lstat()
    assert external_payload.read_bytes() == b"outside\n"
    assert stat.S_IMODE(external_payload.stat().st_mode) == external_mode
    assert external_payload.stat().st_nlink == 1

    absent = source_archive.cleanup_pytest_basetemp(parent)
    assert absent["status"] == "absent"
    assert absent["removed"] is False


def test_cleanup_pytest_basetemp_rejects_root_links_and_special_files(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    basetemp = parent / source_archive.PYTEST_BASETEMP_NAME
    basetemp.symlink_to(external, target_is_directory=True)
    with pytest.raises(source_archive.SourceArchiveError, match="direct regular directory"):
        source_archive.cleanup_pytest_basetemp(parent)
    assert basetemp.is_symlink()
    assert external.is_dir()
    basetemp.unlink()

    basetemp.mkdir(mode=0o700)
    fifo = basetemp / "unsafe-fifo"
    os.mkfifo(fifo, 0o400)
    try:
        with pytest.raises(source_archive.SourceArchiveError, match="special file"):
            source_archive.cleanup_pytest_basetemp(parent)
        assert stat.S_ISFIFO(fifo.lstat().st_mode)
    finally:
        fifo.unlink()
        basetemp.rmdir()


def test_cleanup_command_derives_only_the_effective_absolute_tmpdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TMPDIR", raising=False)
    with pytest.raises(source_archive.SourceArchiveError, match="requires"):
        source_archive._gate_cleanup_parent()
    monkeypatch.setenv("TMPDIR", "relative/tmp")
    with pytest.raises(source_archive.SourceArchiveError, match="absolute"):
        source_archive._gate_cleanup_parent()
    monkeypatch.setenv("TMPDIR", f"{tmp_path}/")
    assert source_archive._gate_cleanup_parent() == tmp_path
    with pytest.raises(SystemExit):
        source_archive.parser().parse_args(["cleanup-pytest", "--parent", str(tmp_path)])
