"""Canonical D0 documents, frozen identities, and strict schema validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, TypedDict

CAMPAIGN_ID = "r2-map-expert-iteration-v1"
D0_RUN_ID = "d0-runtime-bootstrap-20260618-v1"
BOOTSTRAP_PACKET_SCHEMA = "cascadia.r2-map.d0-public-key-bootstrap-packet.v1"
LEGACY_WORK_PACKET_SCHEMA = "cascadia.r2-map.d0-runtime-work-packet.v9"
WORK_PACKET_SCHEMA = "cascadia.r2-map.d0-runtime-work-packet.v10"
SIGNATURE_SCHEMA = "cascadia.r2-map.d0-openssh-signature.v1"
INVENTORY_SCHEMA = "cascadia.r2-map.d0-no-follow-inventory.v1"
LEDGER_COMPARISON_SCHEMA = "cascadia.r2-map.d0-ledger-comparison.v1"
HOST_REPORT_SCHEMA = "cascadia.r2-map.d0-host-report.v4"
PUBLIC_KEY_NAMESPACE = "cascadia-r2-map-d0-v1"
REJECTED_SOURCE_FREEZE_MANIFEST_SHA256 = (
    "969264c12868fcfd819f7197affeef947f827ef9c8a1697fa91fd605b54725aa"
)
REJECTED_HELPER_ARCHIVE_SHA256 = "2b3c335bcdd8281994ef72e102fbafc7da35658f46a7a407a857d7be5290a4c2"

IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
OPENSSH_FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{43}\Z")


class D0Error(ValueError):
    """A D0 packet, artifact, inventory, or operation violated its contract."""


class ArtifactIdentityDocument(TypedDict):
    name: str
    size: int
    sha256: str
    source: str


class BootstrapHelperDocument(TypedDict):
    sha256: str
    size: int
    entrypoint: str


class PublicKeyIdentityDocument(TypedDict):
    algorithm: Literal["ssh-ed25519"]
    fingerprint: str
    openssh_sha256: str
    namespace: str


class BootstrapDestinationsDocument(TypedDict):
    helper: str
    public_key: str
    receipt: str


class BootstrapPacketDocument(TypedDict):
    schema_id: str
    schema_version: int
    campaign_id: str
    run_id: str
    host: Literal["john1", "john2", "john3"]
    issued_unix_ms: int
    expires_unix_ms: int
    helper: BootstrapHelperDocument
    public_key: PublicKeyIdentityDocument
    destinations: BootstrapDestinationsDocument
    protected_seed_values_opened: Literal[False]
    packet_sha256: str


class WorkPathsDocument(TypedDict):
    campaign_root: str
    colima_home: str
    colima_cache_home: str
    docker_config: str
    homebrew_cache: str
    homebrew_logs: str
    homebrew_temp: str
    core_image: str
    smoke_oci: str
    scanner_oci: str
    scanner_license: str
    scanner_source_archive: str
    homebrew_closure: str
    runtime_supply: str
    runtime_supply_inbox: str
    pending_root: str
    control_inbox: str
    output_root: str


class WorkLimitsDocument(TypedDict):
    runtime_max_bytes: int
    runtime_max_free_fraction_ppm: int
    vm_cpu: int
    vm_memory_gib: int
    host_reserve_gib: int
    root_disk_gib: int
    data_disk_gib: int
    output_max_bytes: int
    timeout_seconds: int


class WorkPolicyDocument(TypedDict):
    goal_sha256: str
    plan_sha256: str
    runbook_sha256: str


class SmokeSourceDocument(TypedDict):
    repository: str
    tag: str
    index_digest: str
    manifest_digest: str
    config_digest: str
    layer_digest: str
    manifest_size: int
    config_size: int
    layer_size: int


class WorkArtifactsDocument(TypedDict):
    core_image: ArtifactIdentityDocument
    smoke_source: SmokeSourceDocument
    smoke_oci: ArtifactIdentityDocument | None
    scanner_source: dict[str, Any]
    scanner_oci: ArtifactIdentityDocument | None
    scanner_license: ArtifactIdentityDocument
    scanner_source_archive: ArtifactIdentityDocument
    homebrew_closure: ArtifactIdentityDocument | None
    runtime_supply: ArtifactIdentityDocument | None
    probe_context: ArtifactIdentityDocument
    bottles: list[ArtifactIdentityDocument]


class PredecessorDocument(TypedDict):
    cycle_id: Literal["qualification", "final-live"]
    host: Literal["john1", "john2", "john3"]
    phase: Literal["preflight", "install", "start", "verify", "rollback", "postflight"]
    operation: str
    status: Literal["pass", "fail", "rolled-back"]
    packet_sha256: str
    report_sha256: str
    bundle_sha256: str
    bundle_size: int
    manifest_sha256: str
    materialization_receipt_sha256: str
    finished_unix_ms: int
    receipt_relative: str


class WorkPacketDocument(TypedDict):
    schema_id: str
    schema_version: int
    campaign_id: str
    run_id: str
    cycle_id: Literal["qualification", "final-live"]
    host: Literal["john1", "john2", "john3"]
    role: Literal["builder-worker", "worker"]
    phase: Literal["preflight", "install", "start", "verify", "rollback", "postflight"]
    issued_unix_ms: int
    expires_unix_ms: int
    policy: WorkPolicyDocument
    helper_sha256: str
    public_key_fingerprint: str
    paths: WorkPathsDocument
    limits: WorkLimitsDocument
    artifacts: WorkArtifactsDocument
    allowed_operations: list[str]
    predecessors: list[PredecessorDocument]
    protected_seed_values_opened: Literal[False]
    packet_sha256: str


class SignatureBundleDocument(TypedDict):
    schema_id: str
    schema_version: int
    algorithm: Literal["openssh-ed25519"]
    namespace: str
    signer_identity: str
    public_key_fingerprint: str
    public_key_sha256: str
    payload_sha256: str
    signature_armored: str
    signature_sha256: str
    bundle_sha256: str


class HostReportDocument(TypedDict):
    schema_id: str
    schema_version: int
    campaign_id: str
    run_id: str
    cycle_id: Literal["qualification", "final-live"]
    host: Literal["john1", "john2", "john3"]
    role: Literal["builder-worker", "worker"]
    phase: Literal["preflight", "install", "start", "verify", "rollback", "postflight"]
    operation: str
    packet_sha256: str
    started_unix_ms: int
    finished_unix_ms: int
    status: Literal["pass", "fail", "rolled-back"]
    evidence: dict[str, Any]
    protected_seed_values_opened: Literal[False]
    project_code_executed: Literal[False]
    report_sha256: str


def canonical_json(value: Any) -> bytes:
    """Return the frozen ASCII canonical-JSON representation."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise D0Error("value cannot be represented as canonical JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_sha256(value: Mapping[str, Any], field: str) -> str:
    semantic = {key: item for key, item in value.items() if key != field}
    return sha256_bytes(canonical_json(semantic))


def load_canonical_json(value: bytes, *, maximum: int, label: str) -> dict[str, Any]:
    if len(value) > maximum:
        raise D0Error(f"{label} exceeds {maximum} bytes")
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"{label} is not valid JSON") from error
    if not isinstance(decoded, dict) or canonical_json(decoded) != value:
        raise D0Error(f"{label} is not canonical JSON")
    return decoded


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise D0Error(f"{label} fields differ: missing={missing!r} extra={extra!r}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise D0Error(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise D0Error(f"{label} is not a lowercase SHA-256")
    return value


def _oci_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or OCI_DIGEST.fullmatch(value) is None:
        raise D0Error(f"{label} is not a sha256 OCI digest")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise D0Error(f"{label} is not a valid integer")
    return value


def safe_relative(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise D0Error(f"{label} is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise D0Error(f"{label} is not a canonical relative path")
    return value


def absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise D0Error(f"{label} is not an absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise D0Error(f"{label} is not a canonical absolute path")
    return value


@dataclass(frozen=True)
class ArtifactIdentity:
    name: str
    size: int
    sha256: str
    source: str

    @classmethod
    def from_value(cls, value: Any, label: str) -> ArtifactIdentity:
        if not isinstance(value, dict):
            raise D0Error(f"{label} is not an object")
        _exact_keys(value, {"name", "size", "sha256", "source"}, label)
        return cls(
            name=_identifier(value["name"], f"{label}.name"),
            size=_positive_int(value["size"], f"{label}.size"),
            sha256=_sha256(value["sha256"], f"{label}.sha256"),
            source=absolute_path(value["source"], f"{label}.source")
            if str(value["source"]).startswith("/")
            else _safe_https(value["source"], f"{label}.source"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256, "source": self.source}


def _safe_https(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("https://")
        or any(character.isspace() for character in value)
        or len(value) > 2048
    ):
        raise D0Error(f"{label} is not an HTTPS URL")
    return value


FROZEN_RUNTIME = {
    "colima": {
        "version": "0.10.3",
        "license": "MIT",
        "revision": 0,
        "dependencies": ("lima",),
        "formula_path": "Formula/c/colima.rb",
        "source_url": "https://github.com/abiosoft/colima.git",
        "source_tag": "v0.10.3",
        "source_revision": "00f6c297e92a82c04a4ab507db0a61435650d7e8",
        "source_checksum": None,
        "metadata_size": 4052,
        "metadata_sha256": "17fb0507f710c6792cd21d9d69e038c5c133b469a252267e50bdbfe494591c03",
        "ruby_source_sha256": "389b1d33cf7a692ccd392209311e8a9295fce03a9851b0baa83a432836e1731a",
        "bottle_size": 4_152_160,
        "bottle_sha256": "a9dfd1fa0a4aee62fef75974f39f174e4da774f7ba495c43dd0bcc23633381b8",
        "bottle_tag": "arm64_tahoe",
    },
    "lima": {
        "version": "2.1.2",
        "license": "Apache-2.0",
        "revision": 0,
        "dependencies": (),
        "formula_path": "Formula/l/lima.rb",
        "source_url": "https://github.com/lima-vm/lima/archive/refs/tags/v2.1.2.tar.gz",
        "source_tag": None,
        "source_revision": None,
        "source_checksum": "23fa5f4621e355236a10200c4e4f61eae9f69c805c57a107247847b51522ab8a",
        "metadata_size": 3925,
        "metadata_sha256": "af0746d9d4583b04c2639314530bfdf5883f062dd53f3b69d3549a9484c32da8",
        "ruby_source_sha256": "b7771388bdf131764307ba075d1bff47900a8fe284eee149e31eb845dc943403",
        "bottle_size": 37_444_585,
        "bottle_sha256": "b762e573046db099d16a730ac5b0561ad61b823a337d73c0528750ca2d4f9bd6",
        "bottle_tag": "arm64_tahoe",
    },
    "docker": {
        "version": "29.5.3",
        "license": "Apache-2.0",
        "revision": 0,
        "dependencies": (),
        "formula_path": "Formula/d/docker.rb",
        "source_url": "https://github.com/docker/cli.git",
        "source_tag": "v29.5.3",
        "source_revision": "d1c06ef6b41d88d76866aea43c246cd7c63d04fa",
        "source_checksum": None,
        "metadata_size": 3942,
        "metadata_sha256": "d2d88d3860ac5c870097db226f200b18fda8ef9d20a26227c082b084c04aa720",
        "ruby_source_sha256": "b93f0df01c523b3bb1540f13672e8c9ae584c63c094aa32faa40293f48bb4adb",
        "bottle_size": 9_250_741,
        "bottle_sha256": "bc5abed82384f4456e06b53bea84b71b0f6c0f5dbc249c44b727cb8e2b87510c",
        "bottle_tag": "arm64_tahoe",
    },
    "docker-buildx": {
        "version": "0.35.0",
        "license": "Apache-2.0",
        "revision": 0,
        "dependencies": (),
        "formula_path": "Formula/d/docker-buildx.rb",
        "source_url": "https://github.com/docker/buildx/archive/refs/tags/v0.35.0.tar.gz",
        "source_tag": None,
        "source_revision": None,
        "source_checksum": "790e4eb0c98da49c60d2c94cebcd3f1658cd7aca3be82093fcb19b9c1d0ac06b",
        "metadata_size": 4134,
        "metadata_sha256": "b65dd90d58775e3d810a4b24e5fb82b528badcc71d1daf2ee828a475a85dc0a5",
        "ruby_source_sha256": "77501cb839ae3ac9d735df25bc6eea9678954dc4dabcef4e90a06493f26dc40f",
        "bottle_size": 20_912_619,
        "bottle_sha256": "bb8a00f55798493e9fa48fedd4b5d4fcb4e1c7b3d20451a97c88015320ae77de",
        "bottle_tag": "arm64_tahoe",
        "managed_link_target": "../Cellar/docker-buildx/0.35.0/lib/docker",
        "plugin_link_target": "../../../bin/docker-buildx",
        "install_receipt_sha256": (
            "ed19750933871c41da0e332c147fb55e9ed8bc348ce373040234f6d052f6fa74"
        ),
        "installed_entrypoint_sha256": (
            "8d50dd2ab46d37b57f6cb41f31ee64ebdfd20ea402e3ffbe26b7c1ff42d3ca7e"
        ),
    },
}
FROZEN_HOMEBREW_TAP_HEAD = "6afd4901447db7721cec82a5ec46cedb9c3c2e8f"
FROZEN_FORMULA_GENERATED_DATE = "2026-06-18"
FROZEN_HOMEBREW = {
    "version_line": "Homebrew 6.0.1",
    "executable": "/opt/homebrew/bin/brew",
    "executable_size": 8_671,
    "executable_sha256": "91c16722fc0be515162583a2cb77bc5aa6f317a9f28e2fb883c83690e2b7ea81",
    "repository": "/opt/homebrew",
    "repository_git_head": "109191be4988470b51a60a5ef1998520aa24c01b",
    "repository_origin": "https://github.com/Homebrew/brew",
}

# The packet authorizes one exact action surface. Each runtime host owns an
# independent infrastructure-only qualification chain. Nothing here admits
# project source or project execution; that remains a three-host aggregate
# barrier. John2 alone receives acquisition, build, and BuildKit authority.
# Every cross-host control or result edge terminates at John1; workers never
# receive peer credentials and never publish canonical campaign state.
COMMON_RUNTIME_INSTALL_OPERATIONS = ("materialize-runtime-supply", "install-runtime")
JOHN2_RUNTIME_INSTALL_OPERATIONS = (
    "acquire-core",
    "acquire-homebrew-artifacts",
    "acquire-scanner",
    "acquire-smoke",
    "render-runtime-supply",
    "render-probe-context",
    "install-runtime",
)
JOHN3_RUNTIME_INSTALL_OPERATIONS = (*COMMON_RUNTIME_INSTALL_OPERATIONS,)
INSTALL_OPERATIONS_BY_HOST = {
    "john1": COMMON_RUNTIME_INSTALL_OPERATIONS,
    "john2": JOHN2_RUNTIME_INSTALL_OPERATIONS,
    "john3": JOHN3_RUNTIME_INSTALL_OPERATIONS,
}
JOHN2_ONLY_CAPABILITIES = frozenset(
    {
        "acquire-scanner",
        "acquire-core",
        "acquire-homebrew-artifacts",
        "acquire-smoke",
        "build-oci",
        "build-reproducibility-epoch",
        "buildkit-probe",
        "render-probe-context",
        "render-runtime-supply",
        "render-source-context",
    }
)

OPERATION_MATRIX: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("john1", "preflight"): (("preflight-audit",),),
    ("john1", "install"): tuple((item,) for item in COMMON_RUNTIME_INSTALL_OPERATIONS),
    ("john1", "start"): (("start-runtime",),),
    ("john1", "verify"): (("verify-runtime",),),
    ("john1", "rollback"): (("rollback-runtime",),),
    ("john1", "postflight"): (("postflight-audit",),),
    ("john2", "preflight"): (("preflight-audit",),),
    ("john2", "install"): tuple((item,) for item in JOHN2_RUNTIME_INSTALL_OPERATIONS),
    ("john2", "start"): (("start-runtime",),),
    ("john2", "verify"): (("buildkit-probe", "verify-runtime"),),
    ("john2", "rollback"): (("rollback-runtime",),),
    ("john2", "postflight"): (("postflight-audit",),),
    ("john3", "preflight"): (("preflight-audit",),),
    ("john3", "install"): tuple((item,) for item in JOHN3_RUNTIME_INSTALL_OPERATIONS),
    ("john3", "start"): (("start-runtime",),),
    ("john3", "verify"): (("verify-runtime",),),
    ("john3", "rollback"): (("rollback-runtime",),),
    ("john3", "postflight"): (("postflight-audit",),),
}


def primary_operation(host: str, phase: str, operations: Sequence[str]) -> str:
    """Return the one report-producing operation for an exact matrix entry."""

    normalized = tuple(operations)
    if normalized not in OPERATION_MATRIX.get((host, phase), ()):
        raise D0Error("work packet operation set differs from the host-phase matrix")
    return "verify-runtime" if "verify-runtime" in normalized else normalized[-1]


CORE_IMAGE = {
    "name": "ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz",
    "size": 332_354_401,
    "sha256": "1fc0354f4f99734ce3886628cc7af8b0437c1a1d391b126bd09cba0df35ee53f",
    "sha512": (
        "32242674b046b5057e60c4aba334b51e3665f05412cda89ed081cc2de153ae5c4"
        "1f6b105b5c442cbe48d78e2cc21e9ba1950e406b6fb4fc2fd1dd2259240abbd"
    ),
    "url": (
        "https://github.com/abiosoft/colima-core/releases/download/v0.10.4/"
        "ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz"
    ),
}

SMOKE_IMAGE = {
    "repository": "library/alpine",
    "tag": "3.22.1",
    "index_digest": "sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1",
    "manifest_digest": "sha256:4562b419adf48c5f3c763995d6014c123b3ce1d2e0ef2613b189779caa787192",
    "config_digest": "sha256:02f8efbefad605a169e89926147edd0676646263268f303c6fb3cdfdbc4a9612",
    "layer_digest": "sha256:6e174226ea690ced550e5641249a412cdbefd2d09871f3e64ab52137a54ba606",
    "manifest_size": 1025,
    "config_size": 597,
    "layer_size": 4_130_750,
}

SCANNER_IMAGE = {
    "repository": "docker/buildkit-syft-scanner",
    "tag": "stable-1",
    "version": "v1.11.0",
    "source_revision": "d88056b4e5b61d0ca037340df91be47d343b4386",
    "license": "Apache-2.0",
    "license_url": (
        "https://raw.githubusercontent.com/docker/buildkit-syft-scanner/"
        "d88056b4e5b61d0ca037340df91be47d343b4386/LICENSE"
    ),
    "license_size": 11_357,
    "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "source_archive_url": (
        "https://github.com/docker/buildkit-syft-scanner/archive/refs/tags/v1.11.0.tar.gz"
    ),
    "source_archive_size": 63_641_725,
    "source_archive_sha256": ("75c3edcb10dd5829473f80b1e49c931c6b07ff9618fdd819c13f2e5beda04682"),
    "index_digest": "sha256:79e7b013cbec16bbb436f312819a49a4a57752b2270c1a9332ae1a10fcc82a68",
    "index_size": 4_648,
    "manifest_digest": "sha256:860305b3d1667c35142f11f6e9485e322c1c6173702a0831dc68739a34847f2d",
    "manifest_size": 481,
    "config_digest": "sha256:1b1140a57649a9e09e5492ee74230f997d5b40ee1d9d3c53712aa4825678a02d",
    "config_size": 801,
    "layer_digest": "sha256:255b27b900f93498bf8a20c658d82d2a34b8163713329e8d9b076a3b1d69361f",
    "layer_size": 43_158_689,
    "diff_id": "sha256:c40a46be0aa7222e3bad0cce74e1a379be15a61a37b227f9b83726edd7bb48b9",
    "attestation_manifest_digest": (
        "sha256:9e01281ba29c3dd27d010fa5052a34238393c13e4f6bc094d9f52eec5bd61fe5"
    ),
    "attestation_manifest_size": 1_110,
    "attestation_config_digest": (
        "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    ),
    "attestation_config_size": 2,
    "spdx_digest": "sha256:80a9062f92b9163a1339f1b75a9cdac6c36a9bf6a319a9b4c6554923d6ca61fe",
    "spdx_size": 402_976,
    "provenance_digest": (
        "sha256:46887a099b32d0f53ac921e269ce1d2c74dec8bc89d38ba185a0451fb3e4a303"
    ),
    "provenance_size": 22_628,
}

PROBE_DOCKERFILE = (
    b"FROM scratch\n"
    b"COPY probe.txt /probe.txt\n"
    b'LABEL org.opencontainers.image.title="cascadia-r2-d0-buildkit-probe"\n'
)
PROBE_PAYLOAD = b"cascadia-r2-d0-buildkit-probe-v1\n"
PROBE_DOCKERFILE_SHA256 = "6833706c19eb06366795fcf8687dcac2a5847bdfaf4dd191a8cc0d2a14a33f1d"
PROBE_PAYLOAD_SHA256 = "f0bc36d07e69267a2225ec9b4b5fd00d1099015ef211172d8345851f5c134e23"
PROBE_ARCHIVE_SIZE = 10_240
PROBE_ARCHIVE_SHA256 = "bb61a97afa096f1af029226404dc12a6d22d0be5700d66d5401cc8c35c8df5db"

COLIMA_CONFIG = b"""cpu: 10
memory: 14
rootDisk: 5
disk: 13
arch: aarch64
runtime: docker
vmType: vz
rosetta: false
nestedVirtualization: false
binfmt: false
mounts: null
mountInotify: false
forwardAgent: false
sshConfig: false
autoActivate: false
portForwarder: none
kubernetes:
  enabled: false
"""

DOCKER_CONFIG_JOHN2 = (
    canonical_json({"auths": {}, "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]})
    + b"\n"
)
DOCKER_CONFIG_WORKER = canonical_json({"auths": {}}) + b"\n"
ACTIVE_ROOT = "/Users/johnherrick/cascadia-bench/r2-map-v1"
JOHN2_ARCHIVE_ROOT = "/Users/john2/cascadia-bench/r2-map-archive-v1"
JOHN2_RUNTIME_ROOT = "/Users/john2/.local/share/cascadia-r2"
JOHN3_RUNTIME_ROOT = "/Users/john3/.local/share/cascadia-r2"
PATH_CONTRACT = {
    "john1": {
        "campaign_root": ACTIVE_ROOT,
        "colima_home": "/Users/johnherrick/.local/share/cascadia-r2/colima",
        "colima_cache_home": "/Users/johnherrick/Library/Caches/cascadia-r2/colima",
        "docker_config": "/Users/johnherrick/.config/cascadia-r2/docker",
        "homebrew_cache": "/Users/johnherrick/.local/share/cascadia-r2/homebrew/cache",
        "homebrew_logs": "/Users/johnherrick/.local/share/cascadia-r2/homebrew/logs",
        "homebrew_temp": "/Users/johnherrick/.local/share/cascadia-r2/homebrew/temp",
        "core_image": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/"
            "ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz"
        ),
        "smoke_oci": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/alpine-3.22.1-arm64.oci.tar"
        ),
        "scanner_oci": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/unused-scanner.oci.tar"
        ),
        "scanner_license": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/unused-scanner.LICENSE"
        ),
        "scanner_source_archive": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/unused-scanner-source.tar.gz"
        ),
        "homebrew_closure": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/unused-homebrew-closure.tar"
        ),
        "runtime_supply": (
            "/Users/johnherrick/.local/share/cascadia-r2/bootstrap/worker-runtime-supply-v1.tar"
        ),
        "runtime_supply_inbox": (
            "/Users/johnherrick/.local/share/cascadia-r2/supply-inbox/worker-runtime-supply-v1.tar"
        ),
        "pending_root": f"/Users/johnherrick/.local/share/cascadia-r2/results/{D0_RUN_ID}",
        "control_inbox": "/Users/johnherrick/.local/share/cascadia-r2/control-inbox",
        "output_root": f"{ACTIVE_ROOT}/reports/infrastructure/{D0_RUN_ID}/john1",
    },
    "john2": {
        "campaign_root": ACTIVE_ROOT,
        "colima_home": "/Users/john2/.local/share/cascadia-r2/colima",
        "colima_cache_home": "/Users/john2/Library/Caches/cascadia-r2/colima",
        "docker_config": "/Users/john2/.config/cascadia-r2/docker",
        "homebrew_cache": "/Users/john2/.local/share/cascadia-r2/homebrew/cache",
        "homebrew_logs": "/Users/john2/.local/share/cascadia-r2/homebrew/logs",
        "homebrew_temp": "/Users/john2/.local/share/cascadia-r2/homebrew/temp",
        "core_image": (
            f"{JOHN2_RUNTIME_ROOT}/bootstrap/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz"
        ),
        "smoke_oci": f"{JOHN2_RUNTIME_ROOT}/bootstrap/alpine-3.22.1-arm64.oci.tar",
        "scanner_oci": f"{JOHN2_RUNTIME_ROOT}/bootstrap/buildkit-syft-scanner-v1.11.0.oci.tar",
        "scanner_license": f"{JOHN2_RUNTIME_ROOT}/bootstrap/buildkit-syft-scanner-v1.11.0.LICENSE",
        "scanner_source_archive": (
            f"{JOHN2_RUNTIME_ROOT}/bootstrap/buildkit-syft-scanner-v1.11.0-source.tar.gz"
        ),
        "homebrew_closure": f"{JOHN2_RUNTIME_ROOT}/bootstrap/homebrew-john3-arm64-tahoe-v1.tar",
        "runtime_supply": f"{JOHN2_RUNTIME_ROOT}/bootstrap/worker-runtime-supply-v1.tar",
        "runtime_supply_inbox": f"{JOHN2_RUNTIME_ROOT}/supply-inbox/unused.tar",
        "pending_root": f"{JOHN2_RUNTIME_ROOT}/results/{D0_RUN_ID}",
        "control_inbox": f"{JOHN2_RUNTIME_ROOT}/control-inbox",
        "output_root": f"{JOHN2_RUNTIME_ROOT}/results/{D0_RUN_ID}",
    },
    "john3": {
        "campaign_root": ACTIVE_ROOT,
        "colima_home": "/Users/john3/.local/share/cascadia-r2/colima",
        "colima_cache_home": "/Users/john3/Library/Caches/cascadia-r2/colima",
        "docker_config": "/Users/john3/.config/cascadia-r2/docker",
        "homebrew_cache": "/Users/john3/.local/share/cascadia-r2/homebrew/cache",
        "homebrew_logs": "/Users/john3/.local/share/cascadia-r2/homebrew/logs",
        "homebrew_temp": "/Users/john3/.local/share/cascadia-r2/homebrew/temp",
        "core_image": (
            "/Users/john3/.local/share/cascadia-r2/bootstrap/"
            "ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz"
        ),
        "smoke_oci": "/Users/john3/.local/share/cascadia-r2/bootstrap/alpine-3.22.1-arm64.oci.tar",
        "scanner_oci": "/Users/john3/.local/share/cascadia-r2/bootstrap/unused-scanner.oci.tar",
        "scanner_license": "/Users/john3/.local/share/cascadia-r2/bootstrap/unused-scanner.LICENSE",
        "scanner_source_archive": (
            "/Users/john3/.local/share/cascadia-r2/bootstrap/unused-scanner-source.tar.gz"
        ),
        "homebrew_closure": (
            "/Users/john3/.local/share/cascadia-r2/bootstrap/unused-homebrew-closure.tar"
        ),
        "runtime_supply": (
            "/Users/john3/.local/share/cascadia-r2/bootstrap/worker-runtime-supply-v1.tar"
        ),
        "runtime_supply_inbox": (
            "/Users/john3/.local/share/cascadia-r2/supply-inbox/worker-runtime-supply-v1.tar"
        ),
        "pending_root": f"{JOHN3_RUNTIME_ROOT}/results/{D0_RUN_ID}",
        "control_inbox": f"{JOHN3_RUNTIME_ROOT}/control-inbox",
        "output_root": f"{JOHN3_RUNTIME_ROOT}/results/{D0_RUN_ID}",
    },
}


def validate_bootstrap_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D0Error("bootstrap packet is not an object")
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "host",
        "issued_unix_ms",
        "expires_unix_ms",
        "helper",
        "public_key",
        "destinations",
        "protected_seed_values_opened",
        "packet_sha256",
    }
    _exact_keys(value, required, "bootstrap packet")
    if (
        value["schema_id"] != BOOTSTRAP_PACKET_SCHEMA
        or value["schema_version"] != 1
        or value["campaign_id"] != CAMPAIGN_ID
        or value["host"] not in {"john1", "john2", "john3"}
        or value["protected_seed_values_opened"] is not False
    ):
        raise D0Error("bootstrap packet identity differs")
    if _identifier(value["run_id"], "run_id") != D0_RUN_ID:
        raise D0Error("bootstrap run ID differs")
    issued = _positive_int(value["issued_unix_ms"], "issued_unix_ms")
    expires = _positive_int(value["expires_unix_ms"], "expires_unix_ms")
    if expires <= issued or expires - issued > 24 * 60 * 60 * 1000:
        raise D0Error("bootstrap packet validity window differs")
    helper = value["helper"]
    if not isinstance(helper, dict):
        raise D0Error("helper identity is not an object")
    _exact_keys(helper, {"sha256", "size", "entrypoint"}, "helper")
    _sha256(helper["sha256"], "helper.sha256")
    _positive_int(helper["size"], "helper.size")
    safe_relative(helper["entrypoint"], "helper.entrypoint")
    public = value["public_key"]
    if not isinstance(public, dict):
        raise D0Error("public key identity is not an object")
    _exact_keys(public, {"algorithm", "fingerprint", "openssh_sha256", "namespace"}, "public_key")
    if public["algorithm"] != "ssh-ed25519" or public["namespace"] != PUBLIC_KEY_NAMESPACE:
        raise D0Error("public key contract differs")
    if (
        not isinstance(public["fingerprint"], str)
        or OPENSSH_FINGERPRINT.fullmatch(public["fingerprint"]) is None
    ):
        raise D0Error("public key fingerprint differs")
    _sha256(public["openssh_sha256"], "public_key.openssh_sha256")
    destinations = value["destinations"]
    if not isinstance(destinations, dict):
        raise D0Error("bootstrap destinations are not an object")
    _exact_keys(destinations, {"helper", "public_key", "receipt"}, "destinations")
    absolute_path(destinations["helper"], "destinations.helper")
    absolute_path(destinations["public_key"], "destinations.public_key")
    user_home = {
        "john1": "/Users/johnherrick",
        "john2": "/Users/john2",
        "john3": "/Users/john3",
    }[value["host"]]
    if destinations != {
        "helper": f"{user_home}/.local/libexec/cascadia-r2-d0/v1",
        "public_key": f"{user_home}/.config/cascadia-r2-d0/public-key",
        "receipt": f"{user_home}/.config/cascadia-r2-d0/bootstrap-receipt.json",
    }:
        raise D0Error("bootstrap destinations differ from the owner-private contract")
    if value["packet_sha256"] != document_sha256(value, "packet_sha256"):
        raise D0Error("bootstrap packet SHA-256 differs")
    return value


def validate_work_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D0Error("work packet is not an object")
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "role",
        "phase",
        "issued_unix_ms",
        "expires_unix_ms",
        "policy",
        "helper_sha256",
        "public_key_fingerprint",
        "paths",
        "limits",
        "artifacts",
        "allowed_operations",
        "predecessors",
        "protected_seed_values_opened",
        "packet_sha256",
    }
    legacy = (
        value.get("schema_id") == LEGACY_WORK_PACKET_SCHEMA and value.get("schema_version") == 9
    )
    current = value.get("schema_id") == WORK_PACKET_SCHEMA and value.get("schema_version") == 10
    if current:
        required.add("helper_transitions")
    _exact_keys(value, required, "work packet")
    if (
        not (legacy or current)
        or value["campaign_id"] != CAMPAIGN_ID
        or value["cycle_id"] not in {"qualification", "final-live"}
        or value["host"] not in {"john1", "john2", "john3"}
        or value["role"] not in {"builder-worker", "worker"}
        or value["phase"]
        not in {"preflight", "install", "start", "verify", "rollback", "postflight"}
        or value["protected_seed_values_opened"] is not False
    ):
        raise D0Error("work packet identity differs")
    if (value["host"], value["role"]) not in {
        ("john1", "worker"),
        ("john2", "builder-worker"),
        ("john3", "worker"),
    }:
        raise D0Error("work packet host role differs")
    if _identifier(value["run_id"], "run_id") != D0_RUN_ID:
        raise D0Error("work packet run ID differs")
    issued = _positive_int(value["issued_unix_ms"], "issued_unix_ms")
    expires = _positive_int(value["expires_unix_ms"], "expires_unix_ms")
    if expires <= issued or expires - issued > 24 * 60 * 60 * 1000:
        raise D0Error("work packet validity window differs")
    policy = value["policy"]
    if not isinstance(policy, dict):
        raise D0Error("policy is not an object")
    _exact_keys(policy, {"goal_sha256", "plan_sha256", "runbook_sha256"}, "policy")
    for key, digest in policy.items():
        _sha256(digest, f"policy.{key}")
    _sha256(value["helper_sha256"], "helper_sha256")
    if value["helper_sha256"] == REJECTED_HELPER_ARCHIVE_SHA256:
        raise D0Error("work packet targets the explicitly rejected obsolete helper archive")
    if (
        not isinstance(value["public_key_fingerprint"], str)
        or OPENSSH_FINGERPRINT.fullmatch(value["public_key_fingerprint"]) is None
    ):
        raise D0Error("work packet public key fingerprint differs")
    if current:
        _validate_embedded_helper_transitions(
            value["helper_transitions"],
            helper_sha256=value["helper_sha256"],
            public_key_fingerprint=value["public_key_fingerprint"],
        )
    _validate_paths(value["paths"], value["host"])
    _validate_limits(value["limits"])
    operations = value["allowed_operations"]
    if not isinstance(operations, list) or not operations:
        raise D0Error("allowed operations are absent")
    normalized = [_identifier(item, "allowed operation") for item in operations]
    if normalized != sorted(set(normalized)):
        raise D0Error("allowed operations are not sorted and unique")
    unauthorized = JOHN2_ONLY_CAPABILITIES.intersection(normalized)
    if value["host"] != "john2" and unauthorized:
        raise D0Error(
            f"{value['host']} packet authorizes John2-only capabilities: {sorted(unauthorized)!r}"
        )
    operation = primary_operation(value["host"], value["phase"], normalized)
    _validate_artifacts(
        value["artifacts"],
        value["host"],
        value["paths"],
        value["phase"],
        normalized,
    )
    _validate_predecessors(
        value["predecessors"],
        host=value["host"],
        phase=value["phase"],
        cycle_id=value["cycle_id"],
        issued_unix_ms=issued,
        current_operation=operation,
    )
    if value["packet_sha256"] != document_sha256(value, "packet_sha256"):
        raise D0Error("work packet SHA-256 differs")
    return value


def _validate_embedded_helper_transitions(
    value: Any,
    *,
    helper_sha256: str,
    public_key_fingerprint: str,
) -> None:
    """Validate the signed transition documents bound into a v10 packet.

    Signature authenticity is checked at execution time against the installed
    campaign public key.  The packet validator still binds every document and
    signature byte semantically and rejects malformed, discontinuous, or
    wrong-target chains before a packet can be signed.
    """

    if not isinstance(value, list):
        raise D0Error("work packet helper transitions are not an array")
    # Kept as a local import because aggregate imports this canonical module.
    from .aggregate import validate_helper_transition

    transitions: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise D0Error("work packet helper transition entry is not an object")
        _exact_keys(item, {"document", "signature"}, f"helper_transitions[{index}]")
        document = validate_helper_transition(item["document"])
        signature = validate_signature_bundle(
            item["signature"], payload_sha256=sha256_bytes(canonical_json(document))
        )
        if signature["public_key_fingerprint"] != public_key_fingerprint:
            raise D0Error("work packet helper transition signer differs")
        transitions.append(document)
    if [item["chain_index"] for item in transitions] != list(range(1, len(transitions) + 1)):
        raise D0Error("work packet helper transition chain order differs")
    for index in range(len(transitions) - 1):
        before, after = transitions[index], transitions[index + 1]
        if (
            before["to_helper_sha256"] != after["from_helper_sha256"]
            or before["to_plan_sha256"] != after["from_plan_sha256"]
            or before["to_plan_file_sha256"] != after["from_plan_file_sha256"]
            or (
                "previous_transition_sha256" in after
                and after["previous_transition_sha256"] != before["transition_sha256"]
            )
        ):
            raise D0Error("work packet helper transition chain continuity differs")
    if transitions and transitions[-1]["to_helper_sha256"] != helper_sha256:
        raise D0Error("work packet helper transition target differs")


def _validate_predecessors(
    value: Any,
    *,
    host: str,
    phase: str,
    cycle_id: str,
    issued_unix_ms: int,
    current_operation: str,
) -> None:
    if not isinstance(value, list):
        raise D0Error("work packet predecessors are not an array")
    previous_finished = 0
    seen_final_live = False
    seen_packets: set[str] = set()
    seen_reports: set[str] = set()
    normalized: list[tuple[str, str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise D0Error("predecessor binding is not an object")
        _exact_keys(
            item,
            {
                "phase",
                "cycle_id",
                "host",
                "operation",
                "status",
                "packet_sha256",
                "report_sha256",
                "bundle_sha256",
                "bundle_size",
                "manifest_sha256",
                "materialization_receipt_sha256",
                "finished_unix_ms",
                "receipt_relative",
            },
            f"predecessors[{index}]",
        )
        predecessor_host = item["host"]
        if predecessor_host not in {"john1", "john2", "john3"}:
            raise D0Error("predecessor host differs")
        prior_phase = item["phase"]
        predecessor_cycle = item["cycle_id"]
        if predecessor_cycle not in {"qualification", "final-live"}:
            raise D0Error("predecessor cycle differs")
        if predecessor_cycle == "final-live":
            seen_final_live = True
        elif seen_final_live:
            raise D0Error("predecessor cycles are not monotonic")
        if cycle_id == "qualification" and predecessor_cycle != "qualification":
            raise D0Error("qualification packet binds a future cycle")
        if prior_phase not in {
            "preflight",
            "install",
            "start",
            "verify",
            "rollback",
            "postflight",
        }:
            raise D0Error("predecessor phase differs")
        operation = _identifier(item["operation"], "predecessor operation")
        expected_supply_source = {
            "john1": ("john2", "render-runtime-supply"),
            "john3": ("john1", "materialize-runtime-supply"),
        }.get(host)
        cross_host_dependency = (
            phase == "install"
            and current_operation == "materialize-runtime-supply"
            and expected_supply_source is not None
            and predecessor_host == expected_supply_source[0]
            and predecessor_cycle == cycle_id
            and prior_phase == "install"
            and operation == expected_supply_source[1]
            and item["status"] == "pass"
        )
        if predecessor_host != host and not cross_host_dependency:
            raise D0Error("runtime qualification predecessor crosses a host boundary")
        permitted = OPERATION_MATRIX.get((predecessor_host, prior_phase), ())
        if not any(
            operation == primary_operation(predecessor_host, prior_phase, option)
            for option in permitted
        ):
            raise D0Error("predecessor operation differs from the host-phase matrix")
        if item["status"] not in {"pass", "fail", "rolled-back"}:
            raise D0Error("predecessor status differs")
        packet_digest = _sha256(item["packet_sha256"], "predecessor packet SHA-256")
        report_digest = _sha256(item["report_sha256"], "predecessor report SHA-256")
        _sha256(item["bundle_sha256"], "predecessor bundle SHA-256")
        _sha256(item["manifest_sha256"], "predecessor manifest SHA-256")
        _sha256(
            item["materialization_receipt_sha256"],
            "predecessor materialization receipt SHA-256",
        )
        _positive_int(item["bundle_size"], "predecessor bundle size")
        finished = _positive_int(item["finished_unix_ms"], "predecessor finished time")
        if finished < previous_finished or finished > issued_unix_ms:
            raise D0Error("predecessor times are not monotonic and historical")
        if packet_digest in seen_packets or report_digest in seen_reports:
            raise D0Error("predecessor bindings are duplicated")
        expected_relative = (
            f"receipts/{report_digest}"
            if predecessor_host == host
            else f"dependencies/{predecessor_host}/{report_digest}"
        )
        if safe_relative(item["receipt_relative"], "predecessor receipt") != expected_relative:
            raise D0Error("predecessor receipt path is not digest-derived")
        previous_finished = finished
        seen_packets.add(packet_digest)
        seen_reports.add(report_digest)
        if predecessor_host == host and predecessor_cycle == cycle_id:
            normalized.append((prior_phase, operation, item["status"]))
    _validate_cycle_barrier(value, host=host, cycle_id=cycle_id, current_phase=phase)
    _validate_phase_chain(normalized, host=host, current_phase=phase)
    _validate_required_dependencies(
        value,
        host=host,
        cycle_id=cycle_id,
        current_phase=phase,
        current_operation=current_operation,
    )
    if current_operation == "materialize-runtime-supply":
        expected_source = {
            "john1": ("john2", "render-runtime-supply"),
            "john3": ("john1", "materialize-runtime-supply"),
        }.get(host)
        supply_dependencies = [
            item
            for item in value
            if expected_source is not None
            and item["host"] == expected_source[0]
            and item["cycle_id"] == cycle_id
            and item["phase"] == "install"
            and item["operation"] == expected_source[1]
            and item["status"] == "pass"
        ]
        if len(supply_dependencies) != 1:
            raise D0Error("runtime-supply materialization lacks its exact direct dependency")


def _validate_cycle_barrier(
    bindings: Sequence[Mapping[str, Any]],
    *,
    host: str,
    cycle_id: str,
    current_phase: str,
) -> None:
    if cycle_id == "qualification":
        if current_phase == "preflight" and bindings:
            raise D0Error("qualification preflight has predecessor state")
        return
    qualification = [
        item for item in bindings if item["host"] == host and item["cycle_id"] == "qualification"
    ]
    expected_operation = "postflight-audit"
    if not qualification or (
        qualification[-1]["phase"],
        qualification[-1]["operation"],
        qualification[-1]["status"],
    ) != ("postflight", expected_operation, "pass"):
        raise D0Error("final-live cycle lacks a passing qualification postflight barrier")


def _validate_required_dependencies(
    bindings: Sequence[Mapping[str, Any]],
    *,
    host: str,
    cycle_id: str,
    current_phase: str,
    current_operation: str,
) -> None:
    passed = {
        item["operation"]
        for item in bindings
        if item["host"] == host and item["cycle_id"] == cycle_id and item["status"] == "pass"
    }
    install_order = INSTALL_OPERATIONS_BY_HOST[host]
    if current_phase == "rollback":
        host_install = [
            (item["operation"], item["status"])
            for item in bindings
            if item["host"] == host and item["cycle_id"] == cycle_id and item["phase"] == "install"
        ]
        observed = [operation for operation, _status in host_install]
        if observed != list(install_order[: len(observed)]):
            raise D0Error(f"{host} rollback does not bind a valid install prefix")
        if any(status == "fail" for _operation, status in host_install[:-1]):
            raise D0Error("rollback chain contains a nonterminal failed install")
    if current_phase == "install" and current_operation in install_order:
        required = set(install_order[: install_order.index(current_operation)])
        missing = required - passed
        if missing:
            raise D0Error(f"{host} install chain omits required predecessors: {sorted(missing)!r}")
    if current_phase in {"start", "verify"}:
        required = set(install_order)
        missing = required - passed
        if missing:
            raise D0Error(f"{host} phase chain omits required D0 dependencies: {sorted(missing)!r}")


def _validate_phase_chain(
    chain: Sequence[tuple[str, str, str]],
    *,
    host: str,
    current_phase: str,
) -> None:
    if current_phase == "preflight":
        if chain:
            raise D0Error("preflight packet has predecessor state")
        return
    if not chain or chain[0] != (
        "preflight",
        "preflight-audit",
        "pass",
    ):
        raise D0Error("phase chain does not begin with a passing preflight")
    prior_phase = "preflight"
    prior_status = "pass"
    for index, (next_phase, _operation, status) in enumerate(chain[1:], start=1):
        allowed = {
            "preflight": {"install"},
            "install": {"install", "start", "rollback"},
            "start": {"verify", "rollback"},
            "verify": {"rollback"},
            "rollback": {"rollback", "postflight"},
            "postflight": {"install", "rollback"},
        }[prior_phase]
        if next_phase not in allowed:
            raise D0Error("predecessor phase transition differs")
        if prior_status == "fail" and next_phase != "rollback":
            raise D0Error("a failed phase may only transition to rollback")
        if prior_phase == "rollback" and prior_status != "rolled-back" and next_phase != "rollback":
            raise D0Error("rollback predecessor did not restore the baseline")
        if next_phase == "rollback":
            if status == "fail":
                if index != len(chain) - 1 or current_phase != "rollback":
                    raise D0Error("failed rollback predecessor is not terminal before retry")
            elif status != "rolled-back":
                raise D0Error("rollback report status differs")
        elif status == "fail" and (index != len(chain) - 1 or current_phase != "rollback"):
            raise D0Error("failed predecessor is not terminal before rollback")
        elif status not in {"pass", "fail"}:
            raise D0Error("non-rollback predecessor is not passing")
        prior_phase, prior_status = next_phase, status
    allowed_current = {
        "preflight": {"install"},
        "install": {"install", "start", "rollback"},
        "start": {"verify", "rollback"},
        "verify": {"rollback"},
        "rollback": {"rollback", "postflight"},
        "postflight": {"install", "rollback"},
    }[prior_phase]
    if current_phase not in allowed_current:
        raise D0Error("current packet phase does not follow its predecessor chain")
    if prior_status == "fail" and current_phase != "rollback":
        raise D0Error("current packet does not roll back the failed phase")


def _validate_paths(value: Any, host: str) -> None:
    if not isinstance(value, dict):
        raise D0Error("paths are not an object")
    required = {
        "campaign_root",
        "colima_home",
        "colima_cache_home",
        "docker_config",
        "homebrew_cache",
        "homebrew_logs",
        "homebrew_temp",
        "core_image",
        "smoke_oci",
        "scanner_oci",
        "scanner_license",
        "scanner_source_archive",
        "homebrew_closure",
        "runtime_supply",
        "runtime_supply_inbox",
        "pending_root",
        "control_inbox",
        "output_root",
    }
    _exact_keys(value, required, "paths")
    for key, path in value.items():
        absolute_path(path, f"paths.{key}")
    if "/Volumes/John_1" in canonical_json(value).decode("ascii"):
        raise D0Error("paths reference the retired SSD")
    if value != PATH_CONTRACT[host]:
        raise D0Error("paths differ from the exact host storage contract")


def _validate_limits(value: Any) -> None:
    if not isinstance(value, dict):
        raise D0Error("limits are not an object")
    required = {
        "runtime_max_bytes",
        "runtime_max_free_fraction_ppm",
        "vm_cpu",
        "vm_memory_gib",
        "host_reserve_gib",
        "root_disk_gib",
        "data_disk_gib",
        "output_max_bytes",
        "timeout_seconds",
    }
    _exact_keys(value, required, "limits")
    for key, item in value.items():
        _positive_int(item, f"limits.{key}")
    if (
        value["runtime_max_bytes"] != 20 * 1024**3
        or value["runtime_max_free_fraction_ppm"] != 250_000
        or value["vm_cpu"] != 10
        or value["vm_memory_gib"] != 14
        or value["host_reserve_gib"] != 2
        or value["vm_memory_gib"] + value["host_reserve_gib"] != 16
        or value["root_disk_gib"] != 5
        or value["data_disk_gib"] != 13
        or value["output_max_bytes"] != 1024**3
        or value["timeout_seconds"] != 3600
    ):
        raise D0Error("frozen runtime limits differ")


def _john2_artifact_pending(
    host: str,
    phase: str,
    operations: Sequence[str],
    producer: str,
) -> bool:
    """Return whether a John2 acquisition-prefix packet may omit an output."""

    if host != "john2" or phase != "install":
        return False
    operation = primary_operation(host, phase, operations)
    order = INSTALL_OPERATIONS_BY_HOST[host]
    try:
        return order.index(operation) <= order.index(producer)
    except ValueError as error:
        raise D0Error("John2 progressive artifact producer differs") from error


def _dynamic_artifact_pending(
    host: str,
    phase: str,
    operations: Sequence[str],
    producer: str,
) -> bool:
    """Return whether a derived artifact must still be absent at this node.

    Derived outputs are deliberately null through the transaction that creates
    them.  Worker hosts never acquire the builder-only scanner.  The worker
    runtime-supply identity is an authenticated input to materialization, while
    its extracted smoke image and Homebrew closure become available only after
    materialization succeeds.
    """

    if phase == "preflight":
        return True
    if host == "john2":
        if phase != "install":
            return False
        john2_producer = "render-runtime-supply" if producer == "homebrew-closure" else producer
        return _john2_artifact_pending(host, phase, operations, john2_producer)
    if producer == "acquire-scanner":
        return True
    if producer == "render-runtime-supply":
        return False
    if phase == "install":
        operation = primary_operation(host, phase, operations)
        return operation == "materialize-runtime-supply"
    return False


def _derived_artifact(
    value: Any,
    *,
    field: str,
    host: str,
    phase: str,
    operations: Sequence[str],
    producer: str,
) -> ArtifactIdentity | None:
    """Validate exact progressive nullability for one derived artifact."""

    pending = _dynamic_artifact_pending(host, phase, operations, producer)
    if value is None:
        if not pending:
            raise D0Error(f"{field} is absent after its production boundary")
        return None
    if pending:
        raise D0Error(f"{field} appears before its production boundary")
    return ArtifactIdentity.from_value(value, f"artifacts.{field}")


def _validate_artifacts(
    value: Any,
    host: str,
    paths: Mapping[str, Any],
    phase: str,
    operations: Sequence[str],
) -> None:
    if not isinstance(value, dict):
        raise D0Error("artifacts are not an object")
    _exact_keys(
        value,
        {
            "core_image",
            "smoke_source",
            "smoke_oci",
            "scanner_source",
            "scanner_oci",
            "scanner_license",
            "scanner_source_archive",
            "homebrew_closure",
            "runtime_supply",
            "probe_context",
            "bottles",
        },
        "artifacts",
    )
    core = ArtifactIdentity.from_value(value["core_image"], "artifacts.core_image")
    probe = ArtifactIdentity.from_value(value["probe_context"], "artifacts.probe_context")
    smoke_source = value["smoke_source"]
    if smoke_source != SMOKE_IMAGE:
        raise D0Error("Alpine source descriptor identity differs")
    smoke = _derived_artifact(
        value["smoke_oci"],
        field="smoke_oci",
        host=host,
        phase=phase,
        operations=operations,
        producer="acquire-smoke",
    )
    if value["scanner_source"] != SCANNER_IMAGE:
        raise D0Error("BuildKit scanner source descriptor identity differs")
    scanner = _derived_artifact(
        value["scanner_oci"],
        field="scanner_oci",
        host=host,
        phase=phase,
        operations=operations,
        producer="acquire-scanner",
    )
    scanner_license = ArtifactIdentity.from_value(
        value["scanner_license"], "artifacts.scanner_license"
    )
    scanner_source_archive = ArtifactIdentity.from_value(
        value["scanner_source_archive"], "artifacts.scanner_source_archive"
    )
    closure = _derived_artifact(
        value["homebrew_closure"],
        field="homebrew_closure",
        host=host,
        phase=phase,
        operations=operations,
        producer="homebrew-closure",
    )
    supply = _derived_artifact(
        value["runtime_supply"],
        field="runtime_supply",
        host=host,
        phase=phase,
        operations=operations,
        producer="render-runtime-supply",
    )
    if core != ArtifactIdentity(
        name="colima-core-v0.10.4",
        size=CORE_IMAGE["size"],
        sha256=CORE_IMAGE["sha256"],
        source=CORE_IMAGE["url"],
    ):
        raise D0Error("Colima core artifact identity differs")
    if (
        (smoke is not None and smoke.name != "alpine-3.22.1-arm64-oci")
        or (smoke is not None and smoke.source != paths["smoke_oci"])
        or (scanner is not None and scanner.name != "buildkit-syft-scanner-v1.11.0-arm64-oci")
        or (scanner is not None and scanner.source != paths["scanner_oci"])
        or scanner_license
        != ArtifactIdentity(
            name="buildkit-syft-scanner-v1.11.0-license",
            size=SCANNER_IMAGE["license_size"],
            sha256=SCANNER_IMAGE["license_sha256"],
            source=SCANNER_IMAGE["license_url"],
        )
        or scanner_source_archive
        != ArtifactIdentity(
            name="buildkit-syft-scanner-v1.11.0-source",
            size=SCANNER_IMAGE["source_archive_size"],
            sha256=SCANNER_IMAGE["source_archive_sha256"],
            source=SCANNER_IMAGE["source_archive_url"],
        )
        or probe.name != "d0-buildkit-probe"
        or probe.size != PROBE_ARCHIVE_SIZE
        or probe.sha256 != PROBE_ARCHIVE_SHA256
        or probe.source != f"{paths['output_root']}/probe-context.tar"
        or (closure is not None and closure.source != paths["homebrew_closure"])
        or (supply is not None and supply.name != "worker-runtime-supply-v1")
        or (supply is not None and supply.source != paths["runtime_supply"])
    ):
        raise D0Error("smoke or probe artifact path identity differs")
    bottles = value["bottles"]
    if not isinstance(bottles, list) or not bottles:
        raise D0Error("bottle identities are absent")
    names: list[str] = []
    for index, bottle in enumerate(bottles):
        identity = ArtifactIdentity.from_value(bottle, f"bottles[{index}]")
        names.append(identity.name)
        frozen = FROZEN_RUNTIME.get(identity.name)
        if (
            frozen is None
            or identity.sha256 != frozen["bottle_sha256"]
            or identity.size != frozen["bottle_size"]
            or identity.source
            != (
                f"https://ghcr.io/v2/homebrew/core/{identity.name}/blobs/"
                f"sha256:{frozen['bottle_sha256']}"
            )
        ):
            raise D0Error("bottle identity differs from the frozen runtime")
    if names != sorted(set(names)):
        raise D0Error("bottle identities are not sorted and unique")
    expected = {
        "john1": {"colima", "docker", "lima"},
        "john2": {"colima", "docker", "docker-buildx", "lima"},
        "john3": {"colima", "docker", "lima"},
    }[host]
    if set(names) != expected:
        raise D0Error("bottle set differs from the host role")


def render_document(specification: Mapping[str, Any], *, kind: str) -> bytes:
    value = dict(specification)
    digest_field = "packet_sha256"
    if digest_field in value:
        raise D0Error(f"{digest_field} is renderer-owned")
    value[digest_field] = document_sha256(value, digest_field)
    if kind == "bootstrap":
        validate_bootstrap_packet(value)
    elif kind == "work":
        validate_work_packet(value)
    else:
        raise D0Error("unknown packet kind")
    return canonical_json(value)


def validate_signature_bundle(value: Any, *, payload_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D0Error("signature bundle is not an object")
    required = {
        "schema_id",
        "schema_version",
        "algorithm",
        "namespace",
        "signer_identity",
        "public_key_fingerprint",
        "public_key_sha256",
        "payload_sha256",
        "signature_armored",
        "signature_sha256",
        "bundle_sha256",
    }
    _exact_keys(value, required, "signature bundle")
    if (
        value["schema_id"] != SIGNATURE_SCHEMA
        or value["schema_version"] != 1
        or value["algorithm"] != "openssh-ed25519"
        or value["namespace"] != PUBLIC_KEY_NAMESPACE
        or value["signer_identity"] != "cascadia-r2-map-d0"
        or value["payload_sha256"] != payload_sha256
    ):
        raise D0Error("signature bundle identity differs")
    if (
        not isinstance(value["public_key_fingerprint"], str)
        or OPENSSH_FINGERPRINT.fullmatch(value["public_key_fingerprint"]) is None
    ):
        raise D0Error("signature fingerprint differs")
    _sha256(value["public_key_sha256"], "public_key_sha256")
    _sha256(value["signature_sha256"], "signature_sha256")
    signature = value["signature_armored"]
    if (
        not isinstance(signature, str)
        or not signature.startswith("-----BEGIN SSH SIGNATURE-----\n")
        or not signature.endswith("-----END SSH SIGNATURE-----\n")
        or sha256_bytes(signature.encode("ascii")) != value["signature_sha256"]
    ):
        raise D0Error("armored signature differs")
    if value["bundle_sha256"] != document_sha256(value, "bundle_sha256"):
        raise D0Error("signature bundle SHA-256 differs")
    return value


def validate_host_report(
    value: Any,
    *,
    packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D0Error("host report is not an object")
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "run_id",
        "cycle_id",
        "host",
        "role",
        "phase",
        "operation",
        "packet_sha256",
        "started_unix_ms",
        "finished_unix_ms",
        "status",
        "evidence",
        "protected_seed_values_opened",
        "project_code_executed",
        "report_sha256",
    }
    _exact_keys(value, required, "host report")
    if (
        value["schema_id"] != HOST_REPORT_SCHEMA
        or value["schema_version"] != 4
        or value["campaign_id"] != CAMPAIGN_ID
        or value["run_id"] != D0_RUN_ID
        or value["cycle_id"] not in {"qualification", "final-live"}
        or value["host"] not in {"john1", "john2", "john3"}
        or value["role"] not in {"builder-worker", "worker"}
        or value["phase"]
        not in {"preflight", "install", "start", "verify", "rollback", "postflight"}
        or _identifier(value["operation"], "host report operation") != value["operation"]
        or value["status"] not in {"pass", "fail", "rolled-back"}
        or not isinstance(value["started_unix_ms"], int)
        or isinstance(value["started_unix_ms"], bool)
        or not isinstance(value["finished_unix_ms"], int)
        or isinstance(value["finished_unix_ms"], bool)
        or value["finished_unix_ms"] < value["started_unix_ms"]
        or not isinstance(value["evidence"], dict)
        or value["protected_seed_values_opened"] is not False
        or value["project_code_executed"] is not False
        or value["report_sha256"] != document_sha256(value, "report_sha256")
    ):
        raise D0Error("host report identity differs")
    if packet is not None and (
        any(
            value[field] != packet[field]
            for field in (
                "campaign_id",
                "run_id",
                "cycle_id",
                "host",
                "role",
                "phase",
                "packet_sha256",
            )
        )
        or value["operation"]
        != primary_operation(packet["host"], packet["phase"], packet["allowed_operations"])
        or value["started_unix_ms"] < packet["issued_unix_ms"]
        or value["started_unix_ms"] > packet["expires_unix_ms"]
        or value["finished_unix_ms"] > packet["expires_unix_ms"]
        or value["finished_unix_ms"] - value["started_unix_ms"]
        > packet["limits"]["timeout_seconds"] * 1000
        or (value["phase"] == "rollback" and value["status"] not in {"rolled-back", "fail"})
        or (value["phase"] != "rollback" and value["status"] == "rolled-back")
    ):
        raise D0Error("host report and work packet bindings differ")
    return value


def strict_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise D0Error(f"{label} is not an array")
    return value
