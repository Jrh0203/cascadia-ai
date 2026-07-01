from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import r2_d0.authorization as authorization_module
import r2_d0.cli as cli_module
import r2_d0_helper_migration as migration_module
from r2_d0.aggregate import (
    _validate_helper_history,
    validate_operation_evidence,
    verify_helper_transitions,
)
from r2_d0.bundle import verify_result_bundle
from r2_d0.canonical import D0Error, canonical_json, document_sha256, load_canonical_json
from r2_d0.closure import validate_materialization_receipt
from r2_d0.signing import load_public_key, sign_stdin, signature_bytes, verify_stdin
from r2_d0_helper_migration import (
    _current_epoch_predecessors,
    render_transition,
    render_transition_finalization,
)
from r2_d0_predecessor_transfer import _validate_authorization
from r2_d0_render_successor import _validate_transfer_provenance, parser, render

CAMPAIGN = Path("/Users/johnherrick/cascadia-bench/r2-map-v1/control/d0-v8-campaign")
REPORT_ROOT = Path(
    "/Users/johnherrick/cascadia-bench/r2-map-v1/reports/infrastructure/"
    "d0-runtime-bootstrap-20260618-v1/john2/receipts"
)
FAILED = CAMPAIGN / "ready/qualification/john2/install/acquire-scanner"
PUBLIC_KEY = CAMPAIGN / "plan/campaign-public-key"
PRIVATE_KEY = CAMPAIGN / "private/campaign-ed25519"
V9F_HELPER = "253fefe8c65854811fcd2506e1c95058e7f74601ecb23b57140e5196f2a66432"
V9G_HELPER = "b5b014eb8164adc7c18e8d63459ec3b34901a95c2686b3e1b01da9f8e9caada4"
V9H_HELPER = "7b4ad571d4a0cb19700fc8a42d36fbb012e3c5e03ea9eb7a247a0452eb5d38a4"
V9I_HELPER = "9f8078467f2b82b058f457086a9d07442875e0200337bf8cf0c8b23ad8c64b5e"
V9J_HELPER = "576028dc96ed3f62ae764645d41e12cbd3fb9d945f2ec39e5183654579b3dc51"
RESEARCH_PLAN_SHA256 = "fc3c5cdcbfe86602734f6dcc850f3023f616524040c23d8e621d88ad725a04a7"
REPORTS = (
    (
        "preflight",
        "preflight-audit",
        "6d91b7e86f907c444387f7600a3caf8e8336f9744e332880153a28532a5dd5dd",
    ),
    (
        "install",
        "acquire-core",
        "fc235e75de14099102e6cebf5e69af6b880bc7046d977950ec6499268d68c23b",
    ),
    (
        "install",
        "acquire-homebrew-artifacts",
        "46b69249492168b2274d9174b92f1a6164b36c8adb118367846cfab74ca578a6",
    ),
)


def test_successor_parser_preserves_multi_operation_matrix_order() -> None:
    arguments = parser().parse_args(
        [
            "--base-packet",
            "/tmp/base.json",
            "--predecessor-bundle",
            "/tmp/bundle.tar",
            "--materialization-receipt",
            "/tmp/receipt.json",
            "--cycle",
            "qualification",
            "--phase",
            "verify",
            "--operation",
            "buildkit-probe",
            "--operation",
            "verify-runtime",
            "--public-key",
            "/tmp/public-key",
            "--private-key",
            "/tmp/private-key",
            "--output-root",
            "/tmp/out",
        ]
    )
    assert arguments.operation == ["buildkit-probe", "verify-runtime"]


TRANSITIONS = (
    (
        CAMPAIGN / "migration-v9d/transitions/v9c-to-v9d.json",
        CAMPAIGN / "migration-v9d/transitions/v9c-to-v9d-signature.json",
    ),
    (
        CAMPAIGN / "migration-v9d/transitions/v9d-to-v9e.json",
        CAMPAIGN / "migration-v9d/transitions/v9d-to-v9e-signature.json",
    ),
    (
        CAMPAIGN / "migration-v9f/transitions/v9e-to-v9f.json",
        CAMPAIGN / "migration-v9f/transitions/v9e-to-v9f-signature.json",
    ),
)
TRANSITIONS_TO_V9G = (
    *TRANSITIONS,
    (
        CAMPAIGN / "migration-v9g/transitions/v9f-to-v9g.json",
        CAMPAIGN / "migration-v9g/transitions/v9f-to-v9g-signature.json",
    ),
)


def _epoch_item(helper: str, suffix: str) -> dict[str, object]:
    return {
        "cycle_id": "qualification",
        "host": "john2",
        "phase": "install",
        "operation": f"operation-{suffix}",
        "old_helper_sha256": helper,
        "packet_sha256": suffix * 64,
        "report_sha256": suffix * 64,
        "bundle_sha256": suffix * 64,
    }


def test_transition_epoch_projection_excludes_already_transitioned_predecessors() -> None:
    lineage = {
        "old_helper_sha256": V9J_HELPER,
        "accepted_predecessors": [
            _epoch_item(V9I_HELPER, "a"),
            _epoch_item(V9J_HELPER, "b"),
            _epoch_item(V9H_HELPER, "c"),
        ],
    }
    assert _current_epoch_predecessors(lineage) == [lineage["accepted_predecessors"][1]]


def test_transition_epoch_projection_rejects_duplicate_current_epoch_work() -> None:
    current = _epoch_item(V9J_HELPER, "d")
    lineage = {
        "old_helper_sha256": V9J_HELPER,
        "accepted_predecessors": [current, copy.deepcopy(current)],
    }
    with pytest.raises(D0Error, match="current helper epoch lineage is duplicated"):
        _current_epoch_predecessors(lineage)


def test_real_john1_to_john3_dependency_transfer_is_namespace_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = "91b6b9f7557731dd31794246f810890aedb4747e4c7d9d87fd3df10a25424bb3"
    canonical_root = (
        Path("/Users/johnherrick/cascadia-bench/r2-map-v1/reports/infrastructure/")
        / "d0-runtime-bootstrap-20260618-v1/john1/receipts"
        / report
    )
    transfer_root = (
        CAMPAIGN
        / "transfers/qualification/john3/install/materialize-runtime-supply"
        / report
    )
    required = [
        canonical_root / "bundle.tar",
        canonical_root / "materialization-receipt.json",
        transfer_root / "authorization.json",
        transfer_root / "authorization-signature.json",
        transfer_root / "target-materialization-receipt.json",
    ]
    assert not [str(path) for path in required if not path.is_file()]

    authorization_bytes = (transfer_root / "authorization.json").read_bytes()
    authorization = load_canonical_json(
        authorization_bytes,
        maximum=1024 * 1024,
        label="live cross-host transfer authorization",
    )
    monkeypatch.setattr(
        "r2_d0_predecessor_transfer.time.time_ns",
        lambda: (authorization["issued_unix_ms"] + 1) * 1_000_000,
    )
    assert _validate_authorization(authorization) == authorization
    assert authorization["destination_relative"] == f"dependencies/john1/{report}"

    bundle = (canonical_root / "bundle.tar").read_bytes()
    verification = verify_result_bundle(bundle, public_key=load_public_key(PUBLIC_KEY))
    canonical_bytes = (canonical_root / "materialization-receipt.json").read_bytes()
    canonical_receipt = validate_materialization_receipt(json.loads(canonical_bytes))
    target_receipt = validate_materialization_receipt(
        json.loads((transfer_root / "target-materialization-receipt.json").read_bytes())
    )
    arguments = {
        "canonical_acceptance_bytes": canonical_bytes,
        "canonical_acceptance": canonical_receipt,
        "target_receipt": target_receipt,
        "authorization": authorization,
        "packet": verification["packet"],
        "report": verification["report"],
        "manifest_sha256": verification["manifest"]["manifest_sha256"],
        "bundle_sha256": verification["archive_sha256"],
        "bundle_size": verification["archive_size"],
        "target_host": "john3",
    }
    _validate_transfer_provenance(**arguments)

    wrong_namespace = copy.deepcopy(authorization)
    wrong_namespace["destination_relative"] = f"receipts/{report}"
    wrong_namespace["destination"] = (
        f"{wrong_namespace['target_output_root']}/{wrong_namespace['destination_relative']}"
    )
    wrong_namespace["authorization_sha256"] = document_sha256(
        wrong_namespace, "authorization_sha256"
    )
    with pytest.raises(D0Error, match="destination"):
        _validate_authorization(wrong_namespace)

    wrong_source = copy.deepcopy(authorization)
    wrong_source["source_host"] = "john2"
    wrong_source["destination_relative"] = f"dependencies/john2/{report}"
    wrong_source["destination"] = (
        f"{wrong_source['target_output_root']}/{wrong_source['destination_relative']}"
    )
    wrong_source["authorization_sha256"] = document_sha256(
        wrong_source, "authorization_sha256"
    )
    assert _validate_authorization(wrong_source) == wrong_source
    wrong_arguments = dict(arguments)
    wrong_arguments["authorization"] = wrong_source
    with pytest.raises(D0Error, match="provenance"):
        _validate_transfer_provenance(**wrong_arguments)


def _paths() -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    bundles = []
    target_receipts = []
    canonical_receipts = []
    transfer_authorizations = []
    transfer_signatures = []
    for phase, operation, report in REPORTS:
        relative = (
            f"transfers/qualification/john2/preflight/{report}"
            if phase == "preflight"
            else f"transfers/qualification/john2/{phase}/{operation}/{report}"
        )
        transfer = CAMPAIGN / relative
        canonical = REPORT_ROOT / report
        bundles.append(canonical / "bundle.tar")
        target_receipts.append(transfer / "target-materialization-receipt.json")
        canonical_receipts.append(canonical / "materialization-receipt.json")
        transfer_authorizations.append(transfer / "authorization.json")
        transfer_signatures.append(transfer / "authorization-signature.json")
    return (
        bundles,
        target_receipts,
        canonical_receipts,
        transfer_authorizations,
        transfer_signatures,
    )


def test_real_v9c_predecessors_authorize_v10_scanner_without_hash_domain_confusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = [FAILED / "work-packet.json", PUBLIC_KEY, PRIVATE_KEY]
    required.extend(path for pair in TRANSITIONS for path in pair)
    required.extend(path for group in _paths() for path in group)
    assert not [str(path) for path in required if not path.is_file()]

    incident_path = CAMPAIGN / "migration-v9d/collision/incident.json"
    incident_signature_path = CAMPAIGN / "migration-v9d/collision/incident-signature.json"
    incident_payload = incident_path.read_bytes()
    incident = load_canonical_json(
        incident_payload, maximum=1024 * 1024, label="collision incident"
    )
    incident_signature = load_canonical_json(
        incident_signature_path.read_bytes(),
        maximum=1024 * 1024,
        label="collision incident signature",
    )
    verify_stdin(load_public_key(PUBLIC_KEY), incident_payload, incident_signature)

    bundles, target, canonical, authorizations, authorization_signatures = _paths()
    output = tmp_path / "rendered-scanner-v10"
    receipt = render(
        Namespace(
            base_packet=FAILED / "work-packet.json",
            predecessor_bundle=bundles,
            materialization_receipt=target,
            canonical_acceptance_receipt=canonical,
            predecessor_transfer_authorization=authorizations,
            predecessor_transfer_signature=authorization_signatures,
            artifacts=None,
            cycle="qualification",
            phase="install",
            operation="acquire-scanner",
            helper_sha256=V9F_HELPER,
            helper_transition=[item[0] for item in TRANSITIONS],
            helper_transition_signature=[item[1] for item in TRANSITIONS],
            public_key=PUBLIC_KEY,
            private_key=PRIVATE_KEY,
            output_root=output,
        )
    )
    packet = json.loads((output / "work-packet.json").read_bytes())
    assert packet["schema_version"] == 10
    assert packet["packet_sha256"] == receipt["packet_sha256"]
    assert packet["policy"]["plan_sha256"] == RESEARCH_PLAN_SHA256
    execution_plans = {
        transition["document"][field]
        for transition in packet["helper_transitions"]
        for field in ("from_plan_sha256", "to_plan_sha256")
    }
    assert RESEARCH_PLAN_SHA256 not in execution_plans
    assert [item["document"]["transition_sha256"] for item in packet["helper_transitions"]] == [
        "1864164af309169dbd71edbb4a4427d46ad88b58839e968a37121e1011290520",
        "b086e0290b04bb9fc37a7068c092bdcfd0aafa33c8a8f8a87c26dd7eb57f87aa",
        "0073de46d86dadca42eae94d25cb0558e969a6199664c88a7285cc0c99351967",
    ]
    predecessor_reports = {item["report_sha256"] for item in packet["predecessors"]}
    assert predecessor_reports <= set(incident["accepted_report_sha256s"])
    assert not predecessor_reports.intersection(incident["quarantined_report_sha256s"])

    mirrored_output = tmp_path / "john2-output"
    for binding, bundle, target_receipt in zip(
        packet["predecessors"], bundles, target, strict=True
    ):
        destination = mirrored_output / binding["receipt_relative"]
        destination.mkdir(parents=True)
        (destination / "bundle.tar").write_bytes(bundle.read_bytes())
        (destination / "materialization-receipt.json").write_bytes(target_receipt.read_bytes())
    production_output = packet["paths"]["output_root"]
    real_path = Path
    monkeypatch.setattr(
        cli_module,
        "Path",
        lambda value: mirrored_output if str(value) == production_output else real_path(value),
    )
    monkeypatch.setattr(
        cli_module,
        "_active_helper_identity",
        lambda: {"archive_sha256": V9F_HELPER},
    )
    monkeypatch.setitem(authorization_module.HOST_USERS, "john2", "johnherrick")
    args = Namespace(
        packet=output / "work-packet.json",
        signature=output / "work-packet-signature.json",
        public_key=PUBLIC_KEY,
        control_envelope=None,
        execute=False,
    )
    authorized = cli_module._authorized(args, phase="install", operation="acquire-scanner")
    assert authorized["policy"]["plan_sha256"] == RESEARCH_PLAN_SHA256
    assert {item["report_sha256"] for item in args._predecessor_reports} == predecessor_reports

    # A transition-domain hash substituted into packet policy is a policy
    # change, never a valid way to satisfy helper-plan lineage.
    changed = dict(packet)
    changed["policy"] = dict(packet["policy"])
    changed["policy"]["plan_sha256"] = packet["helper_transitions"][-1]["document"][
        "to_plan_sha256"
    ]
    changed.pop("packet_sha256")
    from r2_d0.canonical import render_document
    from r2_d0.signing import sign_stdin, signature_bytes

    changed_payload = render_document(changed, kind="work")
    changed_path = tmp_path / "cross-domain-packet.json"
    changed_signature = tmp_path / "cross-domain-signature.json"
    changed_path.write_bytes(changed_payload)
    changed_signature.write_bytes(signature_bytes(sign_stdin(PRIVATE_KEY, changed_payload)))
    args.packet = changed_path
    args.signature = changed_signature
    with pytest.raises(D0Error, match="authorization lineage"):
        cli_module._authorized(args, phase="install", operation="acquire-scanner")


def test_real_acquisition_chain_renders_and_preclaim_authorizes_acquire_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanner_report = "bea56fbd88f03856263125b0e4239eee60a872c662b5d443ac91712283c1e69d"
    scanner_bundle = REPORT_ROOT / scanner_report / "bundle.tar"
    scanner_transfer = (
        CAMPAIGN / "transfers/qualification/john2/install/acquire-scanner" / scanner_report
    )
    scanner_verification = verify_result_bundle(
        scanner_bundle.read_bytes(), public_key=load_public_key(PUBLIC_KEY)
    )
    assert scanner_verification["packet"]["artifacts"]["scanner_oci"] is None
    validate_operation_evidence(scanner_verification["packet"], scanner_verification["report"])
    # Core, Homebrew artifacts, and scanner are all byte-exact real acquisition
    # outputs. Recompute each report's pass semantics before rendering.
    bundles, target, canonical, authorizations, authorization_signatures = _paths()
    for bundle in bundles:
        verification = verify_result_bundle(
            bundle.read_bytes(), public_key=load_public_key(PUBLIC_KEY)
        )
        validate_operation_evidence(verification["packet"], verification["report"])
    bundles.append(scanner_bundle)
    target.append(scanner_transfer / "target-materialization-receipt.json")
    canonical.append(REPORT_ROOT / scanner_report / "materialization-receipt.json")
    authorizations.append(scanner_transfer / "authorization.json")
    authorization_signatures.append(scanner_transfer / "authorization-signature.json")

    output = tmp_path / "rendered-smoke-v10"
    render(
        Namespace(
            base_packet=(
                CAMPAIGN / "ready-v9g/qualification/john2/install/acquire-scanner/work-packet.json"
            ),
            predecessor_bundle=bundles,
            materialization_receipt=target,
            canonical_acceptance_receipt=canonical,
            predecessor_transfer_authorization=authorizations,
            predecessor_transfer_signature=authorization_signatures,
            artifacts=(
                CAMPAIGN / "artifact-projections/qualification/john2/after-acquire-scanner.json"
            ),
            cycle="qualification",
            phase="install",
            operation="acquire-smoke",
            helper_sha256=None,
            helper_transition=[],
            helper_transition_signature=[],
            public_key=PUBLIC_KEY,
            private_key=PRIVATE_KEY,
            output_root=output,
        )
    )
    packet = json.loads((output / "work-packet.json").read_bytes())
    assert packet["helper_sha256"] == V9G_HELPER
    assert packet["artifacts"]["scanner_oci"] == {
        "name": "buildkit-syft-scanner-v1.11.0-arm64-oci",
        "sha256": "11cefa740ddac876bd915a943770f62b8e98a598cd8171122409a27db9e4c8ef",
        "size": 43601920,
        "source": packet["paths"]["scanner_oci"],
    }
    assert packet["artifacts"]["smoke_oci"] is None

    mirrored_output = tmp_path / "john2-output"
    for binding, bundle, target_receipt in zip(
        packet["predecessors"], bundles, target, strict=True
    ):
        destination = mirrored_output / binding["receipt_relative"]
        destination.mkdir(parents=True)
        (destination / "bundle.tar").write_bytes(bundle.read_bytes())
        (destination / "materialization-receipt.json").write_bytes(target_receipt.read_bytes())
    production_output = packet["paths"]["output_root"]
    real_path = Path
    monkeypatch.setattr(
        cli_module,
        "Path",
        lambda value: mirrored_output if str(value) == production_output else real_path(value),
    )
    monkeypatch.setattr(
        cli_module,
        "_active_helper_identity",
        lambda: {"archive_sha256": V9G_HELPER},
    )
    monkeypatch.setitem(authorization_module.HOST_USERS, "john2", "johnherrick")
    args = Namespace(
        packet=output / "work-packet.json",
        signature=output / "work-packet-signature.json",
        public_key=PUBLIC_KEY,
        control_envelope=None,
        execute=False,
    )
    authorized = cli_module._authorized(args, phase="install", operation="acquire-smoke")
    assert authorized["packet_sha256"] == packet["packet_sha256"]
    assert len(args._predecessor_reports) == 4


def test_real_scanner_tail_terminally_finalizes_v9g_to_v9h_and_authorizes_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close the real omitted scanner at the rotation cutoff, then use it."""

    # Freeze rendering inside the already signed authorization window and
    # after all three migration receipts. This keeps the live fixture stable.
    monkeypatch.setattr(migration_module.time, "time_ns", lambda: 1_781_847_040_000_000_000)
    hosts = ("john1", "john2", "john3")
    base_path = tmp_path / "v9g-to-v9h-provisional.json"
    base = render_transition(
        Namespace(
            lineage=CAMPAIGN / "migration-v9h/no-work-lineage.json",
            chain_index=5,
            collision_incident=CAMPAIGN / "migration-v9d/collision/incident.json",
            collision_incident_signature=(
                CAMPAIGN / "migration-v9d/collision/incident-signature.json"
            ),
            old_plan=(
                CAMPAIGN
                / (
                    "plan-v9g-superseded-"
                    "6c1ad81a5d24d0f132ba94f24eed684e866120ba685df5795abdaf9b8631d7ef/"
                    "execution-plan.json"
                )
            ),
            new_plan=(
                CAMPAIGN
                / (
                    "plan-v9h-superseded-"
                    "f99719c0b05fcbcb085d0bdfc29a0fc4b90664a664110839ff0b40036d3284eb/"
                    "execution-plan.json"
                )
            ),
            public_key=PUBLIC_KEY,
            accepted_bundle=[],
            migration_authorization=[
                CAMPAIGN / f"migration-v9h/rotation-v1/{host}/authorization.json" for host in hosts
            ],
            migration_authorization_signature=[
                CAMPAIGN / f"migration-v9h/rotation-v1/{host}/authorization-signature.json"
                for host in hosts
            ],
            migration_receipt=[
                CAMPAIGN / f"migration-v9h/final-receipts/{host}/migration-receipt.json"
                for host in hosts
            ],
            old_bootstrap_record=[
                CAMPAIGN / f"bootstrap-records/v9g-b5b014eb8164adc7/{host}/bootstrap-record.json"
                for host in hosts
            ],
            old_bootstrap_signature=[
                CAMPAIGN
                / (f"bootstrap-records/v9g-b5b014eb8164adc7/{host}/bootstrap-record-signature.json")
                for host in hosts
            ],
            new_bootstrap_record=[
                CAMPAIGN / f"bootstrap-records/v9h-7b4ad571d4a0cb19/{host}/bootstrap-record.json"
                for host in hosts
            ],
            new_bootstrap_signature=[
                CAMPAIGN
                / (f"bootstrap-records/v9h-7b4ad571d4a0cb19/{host}/bootstrap-record-signature.json")
                for host in hosts
            ],
            out=base_path,
        )
    )
    assert base["accepted_transactions"] == []
    base_signature_path = tmp_path / "v9g-to-v9h-provisional-signature.json"
    base_signature_path.write_bytes(
        signature_bytes(sign_stdin(PRIVATE_KEY, base_path.read_bytes()))
    )

    scanner_report = "bea56fbd88f03856263125b0e4239eee60a872c662b5d443ac91712283c1e69d"
    scanner_root = REPORT_ROOT / scanner_report
    scanner_transfer = (
        CAMPAIGN / "transfers/qualification/john2/install/acquire-scanner" / scanner_report
    )
    final_path = tmp_path / "v9g-to-v9h-final.json"
    final = render_transition_finalization(
        Namespace(
            base_transition=base_path,
            base_transition_signature=base_signature_path,
            previous_transition=TRANSITIONS_TO_V9G[-1][0],
            previous_transition_signature=TRANSITIONS_TO_V9G[-1][1],
            old_plan=(
                CAMPAIGN
                / (
                    "plan-v9g-superseded-"
                    "6c1ad81a5d24d0f132ba94f24eed684e866120ba685df5795abdaf9b8631d7ef/"
                    "execution-plan.json"
                )
            ),
            public_key=PUBLIC_KEY,
            collision_incident=CAMPAIGN / "migration-v9d/collision/incident.json",
            collision_incident_signature=(
                CAMPAIGN / "migration-v9d/collision/incident-signature.json"
            ),
            migration_receipt=[
                CAMPAIGN / f"migration-v9h/final-receipts/{host}/migration-receipt.json"
                for host in hosts
            ],
            tail_bundle=[scanner_root / "bundle.tar"],
            canonical_receipt=[scanner_root / "materialization-receipt.json"],
            target_receipt=[scanner_transfer / "target-materialization-receipt.json"],
            out=final_path,
        )
    )
    assert final["terminal"] is True
    assert final["base_transition_sha256"] == base["transition_sha256"]
    assert final["tail_transaction_count"] == 1
    assert final["tail_transactions"][0]["report_sha256"] == scanner_report
    final_signature_path = tmp_path / "v9g-to-v9h-final-signature.json"
    final_signature_path.write_bytes(
        signature_bytes(sign_stdin(PRIVATE_KEY, final_path.read_bytes()))
    )

    transition_paths = (
        *TRANSITIONS_TO_V9G,
        (
            CAMPAIGN / "migration-v9h/transitions/v9g-to-v9h.json",
            CAMPAIGN / "migration-v9h/transitions/v9g-to-v9h-signature.json",
        ),
        (
            CAMPAIGN / "migration-v9i/transitions/v9h-to-v9i.json",
            CAMPAIGN / "migration-v9i/transitions/v9h-to-v9i-signature.json",
        ),
        (
            CAMPAIGN / "migration-v9j/transitions/v9i-to-v9j.json",
            CAMPAIGN / "migration-v9j/transitions/v9i-to-v9j-signature.json",
        ),
    )
    transition_pairs = [
        (path.read_bytes(), json.loads(signature.read_bytes()))
        for path, signature in transition_paths
    ]
    verified_transitions = verify_helper_transitions(
        transition_pairs, public_key=load_public_key(PUBLIC_KEY)
    )
    assert [item["chain_index"] for item in verified_transitions] == [1, 2, 3, 4, 5, 6, 7]
    assert verified_transitions[-3]["tail_transaction_count"] == 1
    assert verified_transitions[-2]["tail_transaction_count"] == 0
    assert verified_transitions[-1]["tail_transaction_count"] == 0

    changed_last = dict(verified_transitions[-1])
    changed_last["previous_transition_sha256"] = "0" * 64
    changed_last["transition_sha256"] = document_sha256(changed_last, "transition_sha256")
    changed_payload = canonical_json(changed_last)
    changed_pairs = [
        *transition_pairs[:-1],
        (changed_payload, sign_stdin(PRIVATE_KEY, changed_payload)),
    ]
    with pytest.raises(D0Error, match="chain continuity"):
        verify_helper_transitions(changed_pairs, public_key=load_public_key(PUBLIC_KEY))

    bundles, target, canonical, authorizations, authorization_signatures = _paths()
    bundles.append(scanner_root / "bundle.tar")
    target.append(scanner_transfer / "target-materialization-receipt.json")
    canonical.append(scanner_root / "materialization-receipt.json")
    authorizations.append(scanner_transfer / "authorization.json")
    authorization_signatures.append(scanner_transfer / "authorization-signature.json")
    output = tmp_path / "rendered-smoke-v9h"
    render(
        Namespace(
            base_packet=(
                CAMPAIGN / "ready-v9g/qualification/john2/install/acquire-scanner/work-packet.json"
            ),
            predecessor_bundle=bundles,
            materialization_receipt=target,
            canonical_acceptance_receipt=canonical,
            predecessor_transfer_authorization=authorizations,
            predecessor_transfer_signature=authorization_signatures,
            artifacts=(
                CAMPAIGN / "artifact-projections/qualification/john2/after-acquire-scanner.json"
            ),
            cycle="qualification",
            phase="install",
            operation="acquire-smoke",
            helper_sha256=V9J_HELPER,
            helper_transition=[item[0] for item in transition_paths],
            helper_transition_signature=[item[1] for item in transition_paths],
            public_key=PUBLIC_KEY,
            private_key=PRIVATE_KEY,
            output_root=output,
        )
    )
    packet = json.loads((output / "work-packet.json").read_bytes())
    assert packet["helper_sha256"] == V9J_HELPER
    assert packet["helper_transitions"][-3]["document"]["tail_transaction_count"] == 1
    assert packet["helper_transitions"][-2]["document"]["tail_transaction_count"] == 0
    assert packet["helper_transitions"][-1]["document"]["tail_transaction_count"] == 0

    mirrored_output = tmp_path / "john2-v9h-output"
    for binding, bundle, target_receipt in zip(  # noqa: B905 -- Python 3.9.
        packet["predecessors"], bundles, target
    ):
        destination = mirrored_output / binding["receipt_relative"]
        destination.mkdir(parents=True)
        (destination / "bundle.tar").write_bytes(bundle.read_bytes())
        (destination / "materialization-receipt.json").write_bytes(target_receipt.read_bytes())
    production_output = packet["paths"]["output_root"]
    real_path = Path
    monkeypatch.setattr(
        cli_module,
        "Path",
        lambda value: mirrored_output if str(value) == production_output else real_path(value),
    )
    monkeypatch.setattr(
        cli_module,
        "_active_helper_identity",
        lambda: {"archive_sha256": V9J_HELPER},
    )
    monkeypatch.setitem(authorization_module.HOST_USERS, "john2", "johnherrick")
    args = Namespace(
        packet=output / "work-packet.json",
        signature=output / "work-packet-signature.json",
        public_key=PUBLIC_KEY,
        control_envelope=None,
        execute=False,
    )
    assert (
        cli_module._authorized(args, phase="install", operation="acquire-smoke")["packet_sha256"]
        == packet["packet_sha256"]
    )

    # Dry-run the aggregate mixed-helper gate over every transition's exact
    # accepted set. The terminal transition must contribute the real scanner.
    transactions = []
    for transition in verified_transitions:
        for item in transition["accepted_transactions"]:
            transactions.append(
                (
                    {
                        "helper_sha256": transition["from_helper_sha256"],
                        "cycle_id": item["cycle_id"],
                        "host": item["host"],
                        "packet_sha256": item["packet_sha256"],
                    },
                    {
                        "phase": item["phase"],
                        "operation": item["operation"],
                        "report_sha256": item["report_sha256"],
                    },
                    {
                        "archive_sha256": item["bundle_sha256"],
                        "archive_size": item["bundle_size"],
                        "manifest_sha256": item["manifest_sha256"],
                        "started_unix_ms": item["finished_unix_ms"],
                        "finished_unix_ms": item["finished_unix_ms"],
                    },
                )
            )
    summaries = _validate_helper_history(
        transactions, verified_transitions[-1]["new_bootstraps"], verified_transitions
    )
    assert summaries[-3]["accepted_transaction_count"] == 1
    assert summaries[-2]["accepted_transaction_count"] == 0
    assert summaries[-1]["accepted_transaction_count"] == 0
