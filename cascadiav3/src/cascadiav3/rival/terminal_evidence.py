"""Authenticate Rust-replayed terminal pairs before they enter an appeal.

The Rust verifier is the sole authority for canonical trajectory replay and
terminal-score differences.  This module is intentionally only an identity
join: it pins the verifier executable and immutable ledger bytes, invokes the
verifier without a shell or accelerator-visible environment, validates its
content-addressed receipt, rejoins preregistered identity fields to a frozen
``RootManifest`` and unit expectation, and checks outcome/file fields against
their separate authorities.

The v1 verifier authenticates CPU proxy trajectories only.  Consequently this
module can construct ``CPU_PROXY_REFERENCE`` rows for contract plumbing, but
it cannot construct production or multifidelity evidence and cannot emit a
scientific preference label.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from .appeals import (
    EvidenceDomain,
    HighOnlyHRow,
    SelectionRow,
    _verified_high_only_h_row,
    _verified_selection_row,
)
from .manifest import (
    ACTION_CONTENT_ID_PREFIX,
    CANDIDATE_OCCURRENCE_ID_PREFIX,
    RootManifest,
    require_externally_pinned_root_manifest,
)
from .panel_plan import PanelPlanError, TerminalUnitExpectation
from .schema import (
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    canonical_json_bytes,
    require_exact_keys,
    require_sha256,
    sha256_hex,
)

VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID = "cascadiav3.rival_verified_terminal_pair_receipt.v1"
TERMINAL_PAIR_VERIFIER_CONTRACT_ID = "cascadia-rival.verify-terminal-pair.v1"
MAX_VERIFIER_STDOUT_BYTES = 1024 * 1024
MAX_VERIFIER_STDERR_BYTES = 256 * 1024
MAX_TERMINAL_PAIR_LEDGER_BYTES = 64 * 1024 * 1024

_RECEIPT_FIELDS = (
    "schema_id",
    "verifier_contract_id",
    "verifier_executable_sha256",
    "ledger_file_sha256",
    "pair_sha256",
    "parent_manifest_sha256",
    "ruleset_identity_sha256",
    "source_game_identity_sha256",
    "scenario_sampler_identity_sha256",
    "continuation_policy_identity_sha256",
    "policy_rng_factory_identity_sha256",
    "source_public_root_id",
    "source_rules_menu_hash",
    "source_candidate_menu_hash",
    "panel_id",
    "unit_index",
    "fidelity",
    "target_seat",
    "challenger_branch_ordinal",
    "incumbent_candidate_occurrence_id",
    "challenger_candidate_occurrence_id",
    "incumbent_action_content_id",
    "challenger_action_content_id",
    "incumbent_post_action_memory_sha256",
    "challenger_post_action_memory_sha256",
    "incumbent_world_redetermination_seed_sha256",
    "challenger_world_redetermination_seed_sha256",
    "target_score_difference",
    "proxy_policy",
    "beta_cv_required",
    "receipt_sha256",
)


class TerminalEvidenceError(ValueError):
    """Raised when terminal evidence is unverified, substituted, or unstable."""


def _qualified_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise TerminalEvidenceError(f"{field} must use the 'sha256:' wire")
    try:
        require_sha256(value, field)
    except ValueError as exc:
        raise TerminalEvidenceError(str(exc)) from exc
    return value


def _namespaced(value: Any, field: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise TerminalEvidenceError(f"{field} must use namespace {prefix!r}")
    try:
        require_sha256(value.removeprefix(prefix), field)
    except ValueError as exc:
        raise TerminalEvidenceError(str(exc)) from exc
    return value


def _bounded_int(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerminalEvidenceError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise TerminalEvidenceError(f"{field} is outside [{minimum}, {maximum}]")
    return value


def _file_sha256(
    path: Path,
    *,
    maximum_bytes: int | None = None,
    field: str = "file",
) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            if maximum_bytes is not None:
                observed_size = os.fstat(handle.fileno()).st_size
                if observed_size > maximum_bytes:
                    raise TerminalEvidenceError(
                        f"{field} exceeds the Rust contract byte limit of "
                        f"{maximum_bytes}: observed {observed_size}"
                    )
            observed_bytes = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                observed_bytes += len(chunk)
                if maximum_bytes is not None and observed_bytes > maximum_bytes:
                    raise TerminalEvidenceError(
                        f"{field} exceeds the Rust contract byte limit of "
                        f"{maximum_bytes} while being hashed"
                    )
                digest.update(chunk)
    except TerminalEvidenceError:
        raise
    except OSError as exc:
        raise TerminalEvidenceError(f"could not hash {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _strict_json_object(data: bytes, field: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TerminalEvidenceError(f"{field} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalEvidenceError(f"{field} is not one UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise TerminalEvidenceError(f"{field} must be one JSON object")
    return value


def _stable_regular_file(path: Path, field: str, *, executable: bool = False) -> os.stat_result:
    if path.is_symlink():
        raise TerminalEvidenceError(f"{field} may not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise TerminalEvidenceError(f"could not stat {field} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise TerminalEvidenceError(f"{field} must be a regular file")
    if executable and metadata.st_mode & 0o111 == 0:
        raise TerminalEvidenceError(f"{field} is not executable")
    return metadata


@dataclass(frozen=True)
class _ExecutableSnapshotIdentity:
    device: int
    inode: int
    size: int
    mode: int


def _open_regular_file_no_follow(
    path: Path,
    field: str,
    *,
    executable: bool = False,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TerminalEvidenceError(
            f"could not open {field} {path} without following links: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise TerminalEvidenceError(f"{field} must be a regular file")
        if executable and metadata.st_mode & 0o111 == 0:
            raise TerminalEvidenceError(f"{field} is not executable")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, metadata


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    duplicate = os.dup(descriptor)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        for chunk in iter(lambda: os.read(duplicate, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(duplicate)
    return "sha256:" + digest.hexdigest()


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing executable snapshot")
        view = view[written:]


def _snapshot_executable(
    source: Path,
    *,
    expected_sha256: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path, _ExecutableSnapshotIdentity]:
    """Copy one stable, no-follow source FD into a private immutable identity path."""

    owner = tempfile.TemporaryDirectory(prefix="cascadia-rival-verifier-")
    directory = Path(owner.name)
    try:
        os.chmod(directory, 0o700)
        source_descriptor, source_before = _open_regular_file_no_follow(
            source,
            "terminal verifier executable",
            executable=True,
        )
        staging = directory / ".snapshot-staging"
        staging_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            snapshot_descriptor = os.open(staging, staging_flags, 0o700)
            try:
                os.fchmod(snapshot_descriptor, 0o700)
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(source_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    _write_all(snapshot_descriptor, chunk)
                os.fsync(snapshot_descriptor)
            finally:
                os.close(snapshot_descriptor)
            source_after = os.fstat(source_descriptor)
        finally:
            os.close(source_descriptor)

        stable_fields_before = (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        )
        stable_fields_after = (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        )
        if stable_fields_before != stable_fields_after:
            raise TerminalEvidenceError(
                "terminal verifier executable changed while its snapshot was created"
            )
        executable_sha256 = "sha256:" + digest.hexdigest()
        if executable_sha256 != expected_sha256:
            raise TerminalEvidenceError(
                "terminal verifier executable does not match the frozen manifest"
            )

        snapshot = directory / f"verifier-{digest.hexdigest()}"
        try:
            os.link(staging, snapshot, follow_symlinks=False)
        except OSError as exc:
            raise TerminalEvidenceError(
                f"could not publish terminal verifier executable snapshot: {exc}"
            ) from exc
        staging.unlink()
        directory_descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        descriptor, metadata = _open_regular_file_no_follow(
            snapshot,
            "terminal verifier executable snapshot",
            executable=True,
        )
        try:
            snapshot_sha256 = _fd_sha256(descriptor)
        finally:
            os.close(descriptor)
        if snapshot_sha256 != executable_sha256:
            raise TerminalEvidenceError(
                "published terminal verifier executable snapshot failed its content check"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_nlink != 1:
            raise TerminalEvidenceError(
                "terminal verifier executable snapshot is not private and singly linked"
            )
        return (
            owner,
            snapshot,
            _ExecutableSnapshotIdentity(
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                stat.S_IMODE(metadata.st_mode),
            ),
        )
    except BaseException:
        owner.cleanup()
        raise


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - OS invariant failure
        raise TerminalEvidenceError("could not reap killed Rust terminal verifier") from exc


def _run_bounded_verifier(
    command: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, bytes, bytes]:
    """Run without a shell while bounding both output pipes during collection."""

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            bufsize=0,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise TerminalEvidenceError(f"Rust terminal verifier did not start: {exc}") from exc
    if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
        _terminate_and_reap(process)
        raise TerminalEvidenceError("Rust terminal verifier pipes were not created")

    streams = {
        process.stdout.fileno(): ("stdout", process.stdout, MAX_VERIFIER_STDOUT_BYTES),
        process.stderr.fileno(): ("stderr", process.stderr, MAX_VERIFIER_STDERR_BYTES),
    }
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout_seconds
    try:
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_and_reap(process)
                raise TerminalEvidenceError("Rust terminal verifier did not complete: timed out")
            events = selector.select(remaining)
            if not events:
                _terminate_and_reap(process)
                raise TerminalEvidenceError("Rust terminal verifier did not complete: timed out")
            for key, _mask in events:
                descriptor = key.fd
                name, _stream, limit = streams[descriptor]
                try:
                    chunk = os.read(descriptor, min(65536, limit - len(captured[name]) + 1))
                except BlockingIOError:
                    continue
                except OSError as exc:
                    _terminate_and_reap(process)
                    raise TerminalEvidenceError(
                        f"could not read Rust terminal verifier {name}: {exc}"
                    ) from exc
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                captured[name].extend(chunk)
                if len(captured[name]) > limit:
                    _terminate_and_reap(process)
                    raise TerminalEvidenceError(f"Rust terminal verifier {name} exceeded its limit")

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            _terminate_and_reap(process)
            raise TerminalEvidenceError("Rust terminal verifier did not complete: timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _terminate_and_reap(process)
            raise TerminalEvidenceError(
                "Rust terminal verifier did not complete: timed out"
            ) from exc
        return returncode, bytes(captured["stdout"]), bytes(captured["stderr"])
    except BaseException:
        if process.poll() is None:
            _terminate_and_reap(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


@dataclass(frozen=True)
class VerifiedTerminalPairEvidence:
    """A fully joined Rust receipt; construction is private to the verifier."""

    expectation: TerminalUnitExpectation
    ledger_file_sha256: str
    pair_sha256: str
    receipt_sha256: str
    target_score_difference: int
    world_redetermination_seed_sha256s: tuple[str, str]
    evidence_domain: EvidenceDomain
    _validation_capability: object | None = dataclass_field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.expectation, TerminalUnitExpectation):
            raise TerminalEvidenceError(
                "verified terminal evidence requires a TerminalUnitExpectation"
            )
        for name, identity in (
            ("ledger_file_sha256", self.ledger_file_sha256),
            ("pair_sha256", self.pair_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            _qualified_sha256(identity, name)
        if not isinstance(self.target_score_difference, int) or isinstance(
            self.target_score_difference, bool
        ):
            raise TerminalEvidenceError("target_score_difference must be an integer")
        if (
            not isinstance(self.world_redetermination_seed_sha256s, tuple)
            or len(self.world_redetermination_seed_sha256s) != 2
        ):
            raise TerminalEvidenceError(
                "verified terminal evidence requires exactly two "
                "world-redetermination seed commitments"
            )
        for index, identity in enumerate(self.world_redetermination_seed_sha256s):
            _qualified_sha256(identity, f"world_redetermination_seed_sha256s[{index}]")
        if len(set(self.world_redetermination_seed_sha256s)) != 2:
            raise TerminalEvidenceError(
                "verified terminal evidence repeats a world-redetermination seed commitment"
            )
        if self.evidence_domain is not EvidenceDomain.CPU_PROXY_REFERENCE:
            raise TerminalEvidenceError(
                "the pre-GPU Rust adapter can issue only CPU proxy evidence"
            )
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind="VerifiedTerminalPairEvidence",
                    content_sha256=_verified_terminal_evidence_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise TerminalEvidenceError(str(exc)) from exc

    def as_selection_row(self) -> SelectionRow:
        require_validated_terminal_pair_evidence(self)
        if self.expectation.panel_kind != "S":
            raise TerminalEvidenceError("only an S receipt can become a selection row")
        return _verified_selection_row(
            unit_id=self.expectation.unit_id,
            challenger_id=self.expectation.challenger_candidate_occurrence_id,
            selection_score=float(self.target_score_difference),
            rng_key=(f"cascadiav3.rival_verified_selection_rng.v1:{self.receipt_sha256}"),
            world_redetermination_seed_sha256s=(self.world_redetermination_seed_sha256s),
            evidence_domain=self.evidence_domain,
            receipt_sha256=self.receipt_sha256,
        )

    def as_high_only_h_row(self) -> HighOnlyHRow:
        require_validated_terminal_pair_evidence(self)
        if self.expectation.panel_kind != "H":
            raise TerminalEvidenceError("only an H receipt can become a high-only H row")
        return _verified_high_only_h_row(
            unit_id=self.expectation.unit_id,
            challenger_id=self.expectation.challenger_candidate_occurrence_id,
            high_difference=float(self.target_score_difference),
            physical_key=(f"cascadiav3.rival_independent_proxy_pair.v1:{self.pair_sha256}"),
            inner_rng_keys=(
                f"cascadiav3.rival_verified_proxy_inner_rng.v1:{self.ledger_file_sha256}",
            ),
            world_redetermination_seed_sha256s=(self.world_redetermination_seed_sha256s),
            evidence_domain=self.evidence_domain,
            receipt_sha256=self.receipt_sha256,
        )


def _verified_terminal_evidence_runtime_fingerprint(
    evidence: VerifiedTerminalPairEvidence,
) -> str:
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_verified_terminal_evidence_runtime.v1",
            "expectation": {
                "unit_id": evidence.expectation.unit_id,
                "panel_kind": evidence.expectation.panel_kind,
                "panel_id": evidence.expectation.panel_id,
                "unit_index": evidence.expectation.unit_index,
                "fidelity": evidence.expectation.fidelity,
                "target_seat": evidence.expectation.target_seat,
                "challenger_candidate_occurrence_id": (
                    evidence.expectation.challenger_candidate_occurrence_id
                ),
                "challenger_action_content_id": (evidence.expectation.challenger_action_content_id),
                "incumbent_post_action_memory_sha256": (
                    evidence.expectation.incumbent_post_action_memory_sha256
                ),
                "challenger_post_action_memory_sha256": (
                    evidence.expectation.challenger_post_action_memory_sha256
                ),
            },
            "ledger_file_sha256": evidence.ledger_file_sha256,
            "pair_sha256": evidence.pair_sha256,
            "receipt_sha256": evidence.receipt_sha256,
            "target_score_difference": evidence.target_score_difference,
            "world_redetermination_seed_sha256s": (evidence.world_redetermination_seed_sha256s),
            "evidence_domain": evidence.evidence_domain,
        }
    )


def _seal_verified_terminal_pair_evidence(
    evidence: VerifiedTerminalPairEvidence,
) -> VerifiedTerminalPairEvidence:
    return replace(
        evidence,
        _validation_capability=_issue_validation_capability(
            "VerifiedTerminalPairEvidence",
            _verified_terminal_evidence_runtime_fingerprint(evidence),
        ),
    )


def require_validated_terminal_pair_evidence(
    evidence: VerifiedTerminalPairEvidence,
) -> None:
    if not isinstance(evidence, VerifiedTerminalPairEvidence):
        raise TerminalEvidenceError(
            "terminal row construction requires verified terminal-pair evidence"
        )
    evidence.__post_init__()
    try:
        _require_validation_capability(
            evidence._validation_capability,
            artifact_kind="VerifiedTerminalPairEvidence",
            content_sha256=_verified_terminal_evidence_runtime_fingerprint(evidence),
        )
    except RivalSchemaError as exc:
        raise TerminalEvidenceError(str(exc)) from exc


class RustTerminalVerifier:
    """Invoke one manifest-pinned Rust verifier and join its receipt exactly."""

    def __init__(
        self,
        *,
        executable: str | Path,
        manifest: RootManifest,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(manifest, RootManifest):
            raise TerminalEvidenceError("terminal verifier requires a RootManifest")
        try:
            require_externally_pinned_root_manifest(manifest)
        except RivalSchemaError as exc:
            raise TerminalEvidenceError(
                f"terminal verifier requires an externally byte/content-pinned RootManifest: {exc}"
            ) from exc
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.0 < float(timeout_seconds) <= 3600.0
        ):
            raise TerminalEvidenceError("timeout_seconds must be in (0, 3600]")
        if (
            manifest.inference_mode != "high_fidelity_only"
            or manifest.multifidelity_claim
            or manifest.beta_cv != 0.0
            or manifest.expected_l != 0
        ):
            raise TerminalEvidenceError(
                "the v1 independent proxy verifier is admissible only for the "
                "beta-zero high-fidelity-only control design"
            )
        path = Path(executable)
        if manifest.terminal_verifier_contract_id != TERMINAL_PAIR_VERIFIER_CONTRACT_ID:
            raise TerminalEvidenceError("manifest pins an unsupported verifier contract")
        self.source_executable = path.absolute()
        (
            self._snapshot_owner,
            self.executable,
            self._snapshot_identity,
        ) = _snapshot_executable(
            path,
            expected_sha256=manifest.terminal_verifier_executable_sha256,
        )
        self.executable_sha256 = manifest.terminal_verifier_executable_sha256
        self.manifest = manifest
        self.timeout_seconds = float(timeout_seconds)

    def close(self) -> None:
        """Remove the private executable snapshot owned by this verifier."""

        owner = getattr(self, "_snapshot_owner", None)
        if owner is not None:
            owner.cleanup()
            del self._snapshot_owner

    def __enter__(self) -> RustTerminalVerifier:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _check_executable_snapshot(self) -> os.stat_result:
        descriptor, metadata = _open_regular_file_no_follow(
            self.executable,
            "terminal verifier executable snapshot",
            executable=True,
        )
        try:
            executable_sha256 = _fd_sha256(descriptor)
            metadata_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity = self._snapshot_identity
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_nlink,
        ) != (identity.device, identity.inode, identity.size, identity.mode, 1):
            raise TerminalEvidenceError(
                "terminal verifier executable snapshot identity was substituted"
            )
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_size,
            metadata_after.st_mtime_ns,
            metadata_after.st_ctime_ns,
        ) or executable_sha256 != self.executable_sha256:
            raise TerminalEvidenceError(
                "terminal verifier executable snapshot content was substituted"
            )
        return metadata

    def verify(
        self,
        ledger_path: str | Path,
        *,
        expectation: TerminalUnitExpectation,
        expected_pair_sha256: str,
    ) -> VerifiedTerminalPairEvidence:
        try:
            expectation.validate(self.manifest)
        except PanelPlanError as exc:
            raise TerminalEvidenceError(str(exc)) from exc
        expected_pair_sha256 = _qualified_sha256(expected_pair_sha256, "expected_pair_sha256")
        ledger = Path(ledger_path)
        before_stat = _stable_regular_file(ledger, "terminal-pair ledger")
        if before_stat.st_size > MAX_TERMINAL_PAIR_LEDGER_BYTES:
            raise TerminalEvidenceError(
                "terminal-pair ledger exceeds the Rust contract byte limit of "
                f"{MAX_TERMINAL_PAIR_LEDGER_BYTES}: observed {before_stat.st_size}"
            )
        before_sha256 = _file_sha256(
            ledger,
            maximum_bytes=MAX_TERMINAL_PAIR_LEDGER_BYTES,
            field="terminal-pair ledger",
        )
        executable_before_stat = self._check_executable_snapshot()

        environment = {
            "CASCADIA_CPU_ONLY_TESTS": "1",
            "CASCADIA_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "LC_ALL": "C",
            "PATH": os.defpath,
        }
        command = [
            str(self.executable),
            "verify-terminal-pair",
            str(ledger.absolute()),
            expected_pair_sha256,
            "sha256:" + self.manifest.content_sha256,
        ]
        returncode, stdout, stderr = _run_bounded_verifier(
            command,
            environment=environment,
            timeout_seconds=self.timeout_seconds,
        )
        if returncode != 0:
            reason = stderr.decode("utf-8", errors="replace").strip()
            raise TerminalEvidenceError(
                f"Rust terminal verifier rejected the ledger (exit {returncode}): {reason[:4096]}"
            )
        if stderr:
            raise TerminalEvidenceError("successful Rust terminal verification wrote to stderr")

        after_stat = _stable_regular_file(ledger, "terminal-pair ledger")
        after_sha256 = _file_sha256(
            ledger,
            maximum_bytes=MAX_TERMINAL_PAIR_LEDGER_BYTES,
            field="terminal-pair ledger",
        )
        executable_after_stat = self._check_executable_snapshot()
        if (before_stat.st_dev, before_stat.st_ino, before_stat.st_size) != (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
        ) or before_sha256 != after_sha256:
            raise TerminalEvidenceError("terminal-pair ledger changed during verification")
        if (
            executable_before_stat.st_dev,
            executable_before_stat.st_ino,
            executable_before_stat.st_mtime_ns,
            executable_before_stat.st_ctime_ns,
        ) != (
            executable_after_stat.st_dev,
            executable_after_stat.st_ino,
            executable_after_stat.st_mtime_ns,
            executable_after_stat.st_ctime_ns,
        ):
            raise TerminalEvidenceError(
                "terminal verifier executable snapshot changed during execution"
            )

        receipt = _strict_json_object(stdout, "Rust terminal receipt")
        self._validate_receipt(
            receipt,
            ledger_file_sha256=before_sha256,
            expectation=expectation,
        )
        if receipt["pair_sha256"] != expected_pair_sha256:
            raise TerminalEvidenceError("Rust terminal receipt differs from expected pair pin")
        evidence = VerifiedTerminalPairEvidence(
            expectation=expectation,
            ledger_file_sha256=before_sha256,
            pair_sha256=receipt["pair_sha256"],
            receipt_sha256=receipt["receipt_sha256"],
            target_score_difference=receipt["target_score_difference"],
            world_redetermination_seed_sha256s=(
                receipt["incumbent_world_redetermination_seed_sha256"],
                receipt["challenger_world_redetermination_seed_sha256"],
            ),
            evidence_domain=EvidenceDomain.CPU_PROXY_REFERENCE,
            _validation_capability=None,
        )
        return _seal_verified_terminal_pair_evidence(evidence)

    def _validate_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        ledger_file_sha256: str,
        expectation: TerminalUnitExpectation,
    ) -> None:
        try:
            require_exact_keys(
                receipt,
                required=_RECEIPT_FIELDS,
                where="verified terminal-pair receipt",
            )
        except ValueError as exc:
            raise TerminalEvidenceError(str(exc)) from exc
        if receipt["schema_id"] != VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID:
            raise TerminalEvidenceError("Rust terminal receipt schema was substituted")
        if receipt["verifier_contract_id"] != TERMINAL_PAIR_VERIFIER_CONTRACT_ID:
            raise TerminalEvidenceError("Rust terminal verifier contract was substituted")
        digest_fields = (
            "verifier_executable_sha256",
            "ledger_file_sha256",
            "pair_sha256",
            "parent_manifest_sha256",
            "ruleset_identity_sha256",
            "source_game_identity_sha256",
            "scenario_sampler_identity_sha256",
            "continuation_policy_identity_sha256",
            "policy_rng_factory_identity_sha256",
            "panel_id",
            "incumbent_post_action_memory_sha256",
            "challenger_post_action_memory_sha256",
            "incumbent_world_redetermination_seed_sha256",
            "challenger_world_redetermination_seed_sha256",
            "receipt_sha256",
        )
        for field in digest_fields:
            _qualified_sha256(receipt[field], field)
        _namespaced(
            receipt["source_public_root_id"],
            "source_public_root_id",
            "cascadiav3.rival_public_root.v1:sha256:",
        )
        _namespaced(
            receipt["source_rules_menu_hash"],
            "source_rules_menu_hash",
            "cascadiav3.rival_rules_menu.v1:sha256:",
        )
        _namespaced(
            receipt["source_candidate_menu_hash"],
            "source_candidate_menu_hash",
            "cascadiav3.rival_incumbent_menu.v1:sha256:",
        )
        for field in (
            "incumbent_candidate_occurrence_id",
            "challenger_candidate_occurrence_id",
        ):
            _namespaced(receipt[field], field, CANDIDATE_OCCURRENCE_ID_PREFIX)
        for field in ("incumbent_action_content_id", "challenger_action_content_id"):
            _namespaced(receipt[field], field, ACTION_CONTENT_ID_PREFIX)
        _bounded_int(receipt["unit_index"], "unit_index", minimum=0, maximum=2**32 - 1)
        _bounded_int(receipt["target_seat"], "target_seat", minimum=0, maximum=3)
        _bounded_int(
            receipt["challenger_branch_ordinal"],
            "challenger_branch_ordinal",
            minimum=0,
            maximum=2**16 - 1,
        )
        _bounded_int(
            receipt["target_score_difference"],
            "target_score_difference",
            minimum=-(2**31),
            maximum=2**31 - 1,
        )
        if receipt["fidelity"] not in {"low", "high"}:
            raise TerminalEvidenceError("unknown terminal receipt fidelity")
        if receipt["proxy_policy"] is not True:
            raise TerminalEvidenceError(
                "the v1 verifier contract cannot authenticate production evidence"
            )
        if isinstance(receipt["beta_cv_required"], bool) or receipt["beta_cv_required"] != 0:
            raise TerminalEvidenceError(
                "the v1 independent proxy receipt requires beta_cv_required=0"
            )

        content = {key: receipt[key] for key in _RECEIPT_FIELDS[:-1]}
        expected_receipt_sha256 = (
            "sha256:" + hashlib.sha256(canonical_json_bytes(content)).hexdigest()
        )
        if receipt["receipt_sha256"] != expected_receipt_sha256:
            raise TerminalEvidenceError("Rust terminal receipt content hash mismatch")

        challenger_ordinals = tuple(
            index
            for index, candidate in enumerate(self.manifest.candidate_selection_entries)
            if candidate.candidate_action_occurrence_id
            == expectation.challenger_candidate_occurrence_id
        )
        if len(challenger_ordinals) != 1:
            raise TerminalEvidenceError(
                "the preregistered challenger must occur exactly once in the frozen candidate menu"
            )
        challenger_branch_ordinal = challenger_ordinals[0]
        if challenger_branch_ordinal > 2**16 - 1:
            raise TerminalEvidenceError(
                "the preregistered challenger branch ordinal exceeds the Rust u16 wire"
            )

        expected = {
            "verifier_executable_sha256": self.executable_sha256,
            "ledger_file_sha256": ledger_file_sha256,
            "parent_manifest_sha256": "sha256:" + self.manifest.content_sha256,
            "ruleset_identity_sha256": self.manifest.ruleset_identity,
            "source_game_identity_sha256": self.manifest.source_game_identity_sha256,
            "scenario_sampler_identity_sha256": self.manifest.sampler_identity,
            "continuation_policy_identity_sha256": self.manifest.incumbent_policy_identity,
            "policy_rng_factory_identity_sha256": (self.manifest.policy_rng_factory_identity),
            "source_public_root_id": self.manifest.root_id,
            "source_rules_menu_hash": self.manifest.rules_menu_hash,
            "source_candidate_menu_hash": self.manifest.incumbent_menu_hash,
            "panel_id": expectation.panel_id,
            "unit_index": expectation.unit_index,
            "fidelity": expectation.fidelity,
            "target_seat": expectation.target_seat,
            "incumbent_candidate_occurrence_id": (self.manifest.incumbent_candidate_occurrence_id),
            "challenger_candidate_occurrence_id": (expectation.challenger_candidate_occurrence_id),
            "challenger_branch_ordinal": challenger_branch_ordinal,
            "incumbent_action_content_id": self.manifest.incumbent_action_id,
            "challenger_action_content_id": expectation.challenger_action_content_id,
            "incumbent_post_action_memory_sha256": (
                expectation.incumbent_post_action_memory_sha256
            ),
            "challenger_post_action_memory_sha256": (
                expectation.challenger_post_action_memory_sha256
            ),
        }
        mismatches = [field for field, value in expected.items() if receipt[field] != value]
        if mismatches:
            raise TerminalEvidenceError(
                "Rust terminal receipt does not join to the frozen manifest/unit: "
                + ", ".join(sorted(mismatches))
            )


@dataclass(frozen=True)
class TerminalEvidenceReference:
    """Replay handle for one journaled verifier receipt and its source ledger."""

    receipt_sha256: str
    pair_sha256: str
    ledger_path: Path
    expectation: TerminalUnitExpectation


class RustTerminalRowResolver:
    """Concrete journal resolver that replays the Rust evidence on every read.

    The journal intentionally stores receipt identities, not trusted numeric
    authority.  This resolver dereferences each identity to its immutable pair
    ledger and re-runs :class:`RustTerminalVerifier`; a missing, moved,
    substituted, or mutated ledger therefore makes journal replay fail closed.
    The v1 resolver supports only S and high-only H proxy rows.
    """

    def __init__(
        self,
        *,
        verifier: RustTerminalVerifier,
        references: tuple[TerminalEvidenceReference, ...],
    ) -> None:
        if not isinstance(verifier, RustTerminalVerifier):
            raise TerminalEvidenceError("row resolver requires a RustTerminalVerifier")
        by_receipt: dict[str, TerminalEvidenceReference] = {}
        for reference in references:
            if not isinstance(reference, TerminalEvidenceReference):
                raise TerminalEvidenceError(
                    "row resolver references must be TerminalEvidenceReference values"
                )
            receipt = _qualified_sha256(reference.receipt_sha256, "reference.receipt_sha256")
            pair = _qualified_sha256(reference.pair_sha256, "reference.pair_sha256")
            if receipt in by_receipt:
                raise TerminalEvidenceError("duplicate terminal receipt reference")
            expectation = reference.expectation
            expectation.validate(verifier.manifest)
            ledger = Path(reference.ledger_path).absolute()
            _stable_regular_file(ledger, "terminal-pair ledger")
            by_receipt[receipt] = TerminalEvidenceReference(
                receipt,
                pair,
                ledger,
                expectation,
            )
        if not by_receipt:
            raise TerminalEvidenceError("row resolver requires at least one receipt reference")
        self.verifier = verifier
        self._by_receipt = by_receipt
        self.root_id = verifier.manifest.root_id
        self.deployment_design_sha256 = verifier.manifest.deployment_design_sha256
        self.manifest_content_sha256 = "sha256:" + verifier.manifest.content_sha256

    def __call__(
        self, row_kind: str, serialized_row: Mapping[str, Any]
    ) -> SelectionRow | HighOnlyHRow:
        receipt_raw = serialized_row.get("evidence_receipt_sha256")
        receipt = _qualified_sha256(receipt_raw, "evidence_receipt_sha256")
        try:
            reference = self._by_receipt[receipt]
        except KeyError as exc:
            raise TerminalEvidenceError(
                "journal row references an unregistered terminal receipt"
            ) from exc
        evidence = self.verifier.verify(
            reference.ledger_path,
            expectation=reference.expectation,
            expected_pair_sha256=reference.pair_sha256,
        )
        if evidence.receipt_sha256 != receipt:
            raise TerminalEvidenceError(
                "replayed terminal receipt differs from the journal reference"
            )
        if row_kind == "selection":
            return evidence.as_selection_row()
        if row_kind == "h_high_only":
            return evidence.as_high_only_h_row()
        raise TerminalEvidenceError(
            "the v1 independent proxy resolver cannot authenticate L or multifidelity H rows"
        )


def receipt_identity(record: Mapping[str, Any]) -> str:
    """Validate and return a terminal receipt's canonical self-hash.

    The Rust wire hashes the exact receipt fields except ``receipt_sha256``.
    This helper validates that canonical self-hash but does not authenticate
    the receipt's external joins, so it is deliberately not a row constructor.
    """

    try:
        require_exact_keys(record, required=_RECEIPT_FIELDS, where="terminal receipt")
    except ValueError as exc:
        raise TerminalEvidenceError(str(exc)) from exc
    declared = _qualified_sha256(record["receipt_sha256"], "receipt_sha256")
    content = {key: record[key] for key in _RECEIPT_FIELDS[:-1]}
    expected = "sha256:" + hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if declared != expected:
        raise TerminalEvidenceError("terminal receipt content hash mismatch")
    return declared
