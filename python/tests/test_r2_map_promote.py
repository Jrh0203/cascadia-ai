from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import blake3
import pytest
from cascadia_mlx.r2_map_contracts import (
    Phase,
    TransitionError,
    canonical_json_bytes,
    new_campaign_state,
    read_state,
    transition_state,
    write_state,
)
from cascadia_mlx.r2_map_promote import (
    HARD_GATE_NAMES,
    MODEL_POINTERS_SCHEMA,
    OPPONENT_POOL_SCHEMA,
    PromotionError,
    PromotionValidators,
    apply_registered_gate,
    register_fixed_250_gate,
    register_verified_candidate,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
WORK_ITEM_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/r2_map/focal-work-item-provenance-v4.json"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_shared_rust_work_item_golden_freezes_promotion_provenance_fields() -> None:
    item = json.loads(WORK_ITEM_GOLDEN.read_text())
    assert item["schema_id"] == "cascadia.r2-map.focal-work-item.v4"
    assert item["work_item_id"] == "pair-0000"
    assert item["contract_blake3"] == "a" * 64
    assert item["contract_sha256"] == "b" * 64
    assert item["opponent_field_blake3"] == "c" * 64
    assert item["opponent_field_sha256"] == "d" * 64
    assert all(len(reference["receipt_blake3"]) == 64 for reference in item["pair_receipts"])


def _content_blake3(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return blake3.blake3(canonical_json_bytes(payload)).hexdigest()


def _fake_validators(seen: list[Path] | None = None) -> PromotionValidators:
    def checkpoint(path: str | Path, **_kwargs: Any) -> tuple[dict[str, Any], None, None]:
        path = Path(path)
        if seen is not None:
            seen.append(path)
        manifest = json.loads((path / "checkpoint.json").read_text())
        if manifest["checkpoint_id"] != path.name:
            raise PromotionError("fake checkpoint identity differs")
        return manifest, None, None

    def receipt(path: str | Path, *, checkpoint_path: str | Path) -> dict[str, Any]:
        value = json.loads(Path(path).read_text())
        if value["checkpoint_id"] != Path(checkpoint_path).name:
            raise PromotionError("fake verification receipt differs")
        return value

    return PromotionValidators(checkpoint=checkpoint, receipt=receipt)


def _fake_checkpoint(
    root: Path,
    *,
    run_name: str,
    checkpoint_id: str,
    identity_digit: str,
) -> tuple[Path, Path, Path]:
    run = root / "runs" / run_name
    checkpoint = run / "checkpoints" / checkpoint_id
    manifest = {
        "schema_version": 2,
        "schema_id": "r2-map-checkpoint-v2",
        "checkpoint_id": checkpoint_id,
        "manifest_identity_blake3": identity_digit * 64,
    }
    _write_json(checkpoint / "checkpoint.json", manifest)
    verification = run / "verifications" / f"{checkpoint_id}.json"
    _write_json(
        verification,
        {"checkpoint_id": checkpoint_id, "verification_id": identity_digit * 64},
    )
    return run, checkpoint, verification


def _replace_state(root: Path, current: dict[str, Any], proposed: dict[str, Any]) -> None:
    write_state(root / "control/campaign-state.json", proposed, expected_current=current)


def _initial_state(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True)
    state = new_campaign_state(now="2026-06-18T00:00:00.000Z")
    write_state(root / "control/campaign-state.json", state)
    return state


def _advance_bootstrap_gate(root: Path, candidate_sha256: str) -> dict[str, Any]:
    state = read_state(root / "control/campaign-state.json")
    for phase, kwargs in (
        (Phase.BOOTSTRAP_GENERATING, {}),
        (
            Phase.BOOTSTRAP_VALIDATED,
            {
                "generation_manifest_sha256": DIGEST_A,
                "completed_shard_hosts": ["john1", "john2", "john3"],
            },
        ),
        (Phase.BOOTSTRAP_TRAINING, {}),
        (
            Phase.BOOTSTRAP_CANDIDATE_GATE,
            {"candidate_checkpoint_sha256": candidate_sha256},
        ),
    ):
        proposed = transition_state(state, phase, reason=f"test advance to {phase}", **kwargs)
        _replace_state(root, state, proposed)
        state = proposed
    return state


def _advance_round_gate(root: Path, candidate_sha256: str) -> dict[str, Any]:
    state = read_state(root / "control/campaign-state.json")
    for phase, kwargs in (
        (Phase.ROUND_ALLOCATED, {}),
        (Phase.GENERATING, {}),
        (
            Phase.LOCAL_SHARDS_COMPLETE,
            {"completed_shard_hosts": ["john1", "john2", "john3"]},
        ),
        (
            Phase.COLLECTED_AND_VALIDATED,
            {"generation_manifest_sha256": DIGEST_B},
        ),
        (Phase.TRAINING_AND_BENCHMARKING, {}),
        (
            Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
            {"candidate_checkpoint_sha256": candidate_sha256},
        ),
        (Phase.PAIRED_CANDIDATE_GATE, {}),
    ):
        proposed = transition_state(state, phase, reason=f"test advance to {phase}", **kwargs)
        _replace_state(root, state, proposed)
        state = proposed
    return state


def _distribution(count: int) -> dict[str, Any]:
    return {"count": count, "mean": 90.0, "confidence_95": [89.0, 91.0]}


def _report(
    *,
    benchmark_id: str,
    candidate: str,
    control: str,
    classification: str,
    contract_blake3: str,
    opponent_field_blake3: str,
    contract_sha256: str,
    opponent_field_sha256: str,
    peak_rss_bytes: int = 100_000,
) -> dict[str, Any]:
    delta = {
        "promote": (1.0, [0.2, 1.8]),
        "reject": (-1.0, [-1.8, -0.2]),
        "inconclusive": (0.1, [-0.5, 0.7]),
    }[classification]
    work_items = []
    for pair_index in range(250):
        work_item = f"pair-{pair_index:04}"
        work_items.append(
            {
                "schema_version": 4,
                "schema_id": "cascadia.r2-map.focal-work-item.v4",
                "contract_blake3": contract_blake3,
                "opponent_field_blake3": opponent_field_blake3,
                "contract_sha256": contract_sha256,
                "opponent_field_sha256": opponent_field_sha256,
                "work_item_id": work_item,
                "stage": "development",
                "pairs": 1,
                "physical_games": 2,
                "pair_receipts": [
                    {"pair_index": pair_index, "receipt_blake3": f"{pair_index:064x}"}
                ],
                "peak_rss_bytes": peak_rss_bytes,
                "maximum_swap_delta_bytes": 0,
                "all_clean_shutdowns": True,
                "all_pinecone_conservation_checks_passed": True,
                "summed_game_seconds": 10.0,
                "summed_checkpoint_load_seconds": 1.0,
            }
        )
    arm = {
        "base_total": _distribution(250),
        "pinecones": {"conservation_valid_games": 250},
    }
    return {
        "schema_version": 4,
        "schema_id": "cascadia.r2-map.focal-report.v4",
        "benchmark_id": benchmark_id,
        "contract_blake3": contract_blake3,
        "opponent_field_blake3": opponent_field_blake3,
        "contract_sha256": contract_sha256,
        "opponent_field_sha256": opponent_field_sha256,
        "work_items": work_items,
        "result": {
            "kind": "development",
            "statistics": {
                "schema_version": 1,
                "protocol_id": "r2-map-focal-paired-v1",
                "stage": "development",
                "strength_outputs_blinded": False,
                "pairs": 250,
                "physical_games": 500,
                "candidate_checkpoint_id": candidate,
                "control_checkpoint_id": control,
                "candidate": arm,
                "control": arm,
                "paired_delta": {
                    "base_total": {
                        "count": 250,
                        "mean": delta[0],
                        "confidence_95": delta[1],
                    }
                },
                "classification": classification,
            },
        },
    }


def _focal_inputs(
    *, benchmark_id: str, candidate: str, control: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_id = f"field-{benchmark_id}"
    contract = {
        "schema_version": 4,
        "schema_id": "cascadia.r2-map.focal-contract.v4",
        "benchmark_id": benchmark_id,
        "stage": "development",
        "pair_count": 250,
        "execution_partition": {"kind": "scheduler-managed-pairs"},
        "candidate_checkpoint_id": candidate,
        "control_checkpoint_id": control,
        "opponent_field_manifest_id": manifest_id,
        "inference_settings_id": "r2-map-reference-exhaustive-v1",
    }
    assignments = []
    for pair_index in range(250):
        focal_seat = pair_index % 4
        assignments.append(
            {
                "pair_index": pair_index,
                "game_seed": 10_000 + pair_index,
                "seed_domain_id": f"promotion-{pair_index:04}",
                "focal_seat": focal_seat,
                "opponents": [
                    {"seat": seat, "checkpoint_id": "greedy-v1"}
                    for seat in range(4)
                    if seat != focal_seat
                ],
            }
        )
    opponent_field = {
        "schema_version": 4,
        "schema_id": "cascadia.r2-map.opponent-field.v4",
        "manifest_id": manifest_id,
        "assignments": assignments,
    }
    return contract, opponent_field


def _rust_struct_blake3(value: dict[str, Any]) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _all_gates() -> dict[str, bool]:
    return dict.fromkeys(HARD_GATE_NAMES, True)


def _limits() -> dict[str, int | bool]:
    return {
        "max_peak_rss_bytes": 1_000_000,
        "max_swap_delta_bytes": 0,
        "require_clean_shutdown": True,
    }


def _register_candidate(
    root: Path,
    *,
    logical_id: str,
    round_index: int | None,
    physical_id: str,
    identity_digit: str,
    validators: PromotionValidators,
) -> tuple[Path, str, str]:
    run, checkpoint, receipt = _fake_checkpoint(
        root,
        run_name=f"run-{physical_id}",
        checkpoint_id=physical_id,
        identity_digit=identity_digit,
    )
    manifest_sha256 = hashlib.sha256((checkpoint / "checkpoint.json").read_bytes()).hexdigest()
    benchmark_id = f"benchmark-{logical_id}"
    registration = register_verified_candidate(
        campaign_root=root,
        checkpoint_path=checkpoint,
        run_dir=run,
        verification_receipt_path=receipt,
        logical_candidate_id=logical_id,
        round_index=round_index,
        benchmark_id=benchmark_id,
        validators=validators,
        now="2026-06-18T00:01:00.000Z",
    )
    return registration, manifest_sha256, benchmark_id


def _register_gate(
    root: Path,
    *,
    candidate_registration: Path,
    benchmark_id: str,
    candidate: str,
    control: str,
    classification: str,
    validators: PromotionValidators,
    gate_results: dict[str, bool] | None = None,
    peak_rss_bytes: int = 100_000,
    mutate_report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path]:
    benchmark_root = root / "benchmarks" / benchmark_id
    report_path = benchmark_root / "reports/focal-benchmark.json"
    contract_path = benchmark_root / "contract.json"
    opponent_field_path = benchmark_root / "opponent-field.json"
    contract, opponent_field = _focal_inputs(
        benchmark_id=benchmark_id, candidate=candidate, control=control
    )
    _write_json(contract_path, contract)
    _write_json(opponent_field_path, opponent_field)
    contract = json.loads(contract_path.read_text())
    opponent_field = json.loads(opponent_field_path.read_text())
    report = _report(
        benchmark_id=benchmark_id,
        candidate=candidate,
        control=control,
        classification=classification,
        contract_blake3=_rust_struct_blake3(contract),
        opponent_field_blake3=_rust_struct_blake3(opponent_field),
        contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        opponent_field_sha256=hashlib.sha256(opponent_field_path.read_bytes()).hexdigest(),
        peak_rss_bytes=peak_rss_bytes,
    )
    if mutate_report is not None:
        mutate_report(report)
    _write_json(report_path, report)
    pointers_path = root / "control/incumbent-promoted-pointers.json"
    pool_hash = None
    if pointers_path.exists():
        pool_hash = json.loads(pointers_path.read_text())["opponent_pool"]["manifest_blake3"]
    gate = register_fixed_250_gate(
        campaign_root=root,
        candidate_registration_path=candidate_registration,
        focal_report_path=report_path,
        focal_contract_path=contract_path,
        opponent_field_path=opponent_field_path,
        gate_results=_all_gates() if gate_results is None else gate_results,
        resource_limits=_limits(),
        opponent_pool_manifest_blake3=pool_hash,
        validators=validators,
        now="2026-06-18T00:02:00.000Z",
    )
    return gate, report_path


def _bootstrap(root: Path, validators: PromotionValidators) -> None:
    _initial_state(root)
    candidate, manifest_sha256, benchmark_id = _register_candidate(
        root,
        logical_id="bootstrap-candidate",
        round_index=None,
        physical_id="checkpoint-bootstrap",
        identity_digit="1",
        validators=validators,
    )
    _advance_bootstrap_gate(root, manifest_sha256)
    gate, _ = _register_gate(
        root,
        candidate_registration=candidate,
        benchmark_id=benchmark_id,
        candidate="bootstrap-candidate",
        control="greedy-v1",
        classification="promote",
        validators=validators,
    )
    result = apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )
    assert result["incumbent_checkpoint_id"] == "C[0]"


def _round(
    root: Path,
    validators: PromotionValidators,
    classification: str,
) -> tuple[Path, Path, Path]:
    candidate, manifest_sha256, benchmark_id = _register_candidate(
        root,
        logical_id="T[0]",
        round_index=0,
        physical_id="checkpoint-round-0",
        identity_digit="2",
        validators=validators,
    )
    _advance_round_gate(root, manifest_sha256)
    gate, report = _register_gate(
        root,
        candidate_registration=candidate,
        benchmark_id=benchmark_id,
        candidate="T[0]",
        control="C[0]",
        classification=classification,
        validators=validators,
    )
    return candidate, gate, report


def test_promote_updates_atomic_roles_and_builds_historical_pool(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, gate, _ = _round(root, validators, "promote")

    outcome = apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )

    state = read_state(root / "control/campaign-state.json")
    pointers = json.loads((root / "control/incumbent-promoted-pointers.json").read_text())
    pool_path = root / pointers["opponent_pool"]["path"]
    pool = json.loads(pool_path.read_text())
    assert outcome["classification"] == "promote"
    assert state["incumbent_checkpoint_id"] == "C[1]"
    assert pointers["schema_id"] == MODEL_POINTERS_SCHEMA
    assert pointers["incumbent"] == pointers["promoted"]
    assert pointers["incumbent"]["policy_id"] == "C[1]"
    assert [entry["policy_id"] for entry in pool["entries"]] == ["greedy-v1", "C[0]"]
    assert pool["schema_id"] == OPPONENT_POOL_SCHEMA
    assert "C[1]" not in {entry["policy_id"] for entry in pool["entries"]}


@pytest.mark.parametrize("classification", ["reject", "inconclusive"])
def test_non_promote_consumes_only_round_and_preserves_incumbent_pool(
    tmp_path: Path, classification: str
) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    pointer_path = root / "control/incumbent-promoted-pointers.json"
    pointer_before = pointer_path.read_bytes()
    pointers = json.loads(pointer_before)
    pool_path = root / pointers["opponent_pool"]["path"]
    pool_before = pool_path.read_bytes()
    candidate, gate, _ = _round(root, validators, classification)

    outcome = apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )

    state = read_state(root / "control/campaign-state.json")
    assert outcome["classification"] == classification
    assert state["phase"] == Phase.CANDIDATE_REJECTED.value
    assert state["incumbent_checkpoint_id"] == "C[0]"
    assert pointer_path.read_bytes() == pointer_before
    assert pool_path.read_bytes() == pool_before


def test_hard_resource_failure_is_mechanically_rejected(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, manifest_sha256, benchmark_id = _register_candidate(
        root,
        logical_id="T[0]",
        round_index=0,
        physical_id="checkpoint-round-0",
        identity_digit="2",
        validators=validators,
    )
    _advance_round_gate(root, manifest_sha256)
    gates = _all_gates()
    gates["memory"] = False
    gates["resource"] = False
    gate, _ = _register_gate(
        root,
        candidate_registration=candidate,
        benchmark_id=benchmark_id,
        candidate="T[0]",
        control="C[0]",
        classification="reject",
        validators=validators,
        gate_results=gates,
        peak_rss_bytes=2_000_000,
    )
    outcome = apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )
    assert outcome["classification"] == "reject"
    assert read_state(root / "control/campaign-state.json")["phase"] == (
        Phase.CANDIDATE_REJECTED.value
    )


def test_report_and_checkpoint_tamper_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, gate, report = _round(root, validators, "promote")
    report.write_text(report.read_text() + " ")
    with pytest.raises(PromotionError, match="report changed"):
        apply_registered_gate(
            campaign_root=root,
            candidate_registration_path=candidate,
            gate_registration_path=gate,
            validators=validators,
        )
    assert read_state(root / "control/campaign-state.json")["phase"] == (
        Phase.PAIRED_CANDIDATE_GATE.value
    )

    # Restore report exactly, then corrupt the explicitly registered checkpoint.
    report.write_text(report.read_text()[:-1])
    registration = json.loads(candidate.read_text())
    manifest = root / registration["checkpoint_path"] / "checkpoint.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(PromotionError, match="candidate bytes changed"):
        apply_registered_gate(
            campaign_root=root,
            candidate_registration_path=candidate,
            gate_registration_path=gate,
            validators=validators,
        )


@pytest.mark.parametrize("location", ["report", "work-item"])
def test_exact_contract_file_hash_drift_fails_before_gate_registration(
    tmp_path: Path, location: str
) -> None:
    root = tmp_path / f"campaign-{location}"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, manifest_sha256, benchmark_id = _register_candidate(
        root,
        logical_id="T[0]",
        round_index=0,
        physical_id="checkpoint-round-0",
        identity_digit="2",
        validators=validators,
    )
    _advance_round_gate(root, manifest_sha256)

    def mutate(report: dict[str, Any]) -> None:
        target = report if location == "report" else report["work_items"][0]
        target["contract_sha256"] = "0" * 64

    with pytest.raises(PromotionError, match=r"bind|coverage"):
        _register_gate(
            root,
            candidate_registration=candidate,
            benchmark_id=benchmark_id,
            candidate="T[0]",
            control="C[0]",
            classification="promote",
            validators=validators,
            mutate_report=mutate,
        )


def test_identity_drift_and_checkpoint_hunting_are_rejected_or_avoided(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    seen: list[Path] = []
    validators = _fake_validators(seen)
    _bootstrap(root, validators)
    candidate, gate, _ = _round(root, validators, "promote")
    registered = json.loads(candidate.read_text())
    registered_checkpoint = root / registered["checkpoint_path"]
    seen.clear()
    _fake_checkpoint(
        root,
        run_name=registered_checkpoint.parents[1].name,
        checkpoint_id="checkpoint-better-looking",
        identity_digit="9",
    )
    apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )
    assert seen and set(seen) == {registered_checkpoint}

    # Re-signing a drifted gate does not make it agree with the registered candidate.
    root2 = tmp_path / "campaign-drift"
    _bootstrap(root2, validators)
    candidate2, gate2, _ = _round(root2, validators, "promote")
    gate_value = json.loads(gate2.read_text())
    gate_value["candidate_checkpoint_id"] = "T[999]"
    gate_value["registration_blake3"] = _content_blake3(gate_value, "registration_blake3")
    _write_json(gate2, gate_value)
    with pytest.raises(PromotionError, match="candidate registration differ"):
        apply_registered_gate(
            campaign_root=root2,
            candidate_registration_path=candidate2,
            gate_registration_path=gate2,
            validators=validators,
        )


def test_duplicate_opponent_pool_identity_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    pointers_path = root / "control/incumbent-promoted-pointers.json"
    pointers = json.loads(pointers_path.read_text())
    pool_path = root / pointers["opponent_pool"]["path"]
    pool = json.loads(pool_path.read_text())
    pool["entries"].append({"kind": "greedy", "policy_id": "greedy-v1"})
    pool["manifest_blake3"] = _content_blake3(pool, "manifest_blake3")
    _write_json(pool_path, pool)
    pointers["opponent_pool"]["manifest_blake3"] = pool["manifest_blake3"]
    pointers["pointers_blake3"] = _content_blake3(pointers, "pointers_blake3")
    _write_json(pointers_path, pointers)
    candidate, manifest_sha256, benchmark_id = _register_candidate(
        root,
        logical_id="T[0]",
        round_index=0,
        physical_id="checkpoint-round-0",
        identity_digit="2",
        validators=validators,
    )
    _advance_round_gate(root, manifest_sha256)
    with pytest.raises(PromotionError, match="not unique"):
        _register_gate(
            root,
            candidate_registration=candidate,
            benchmark_id=benchmark_id,
            candidate="T[0]",
            control="C[0]",
            classification="promote",
            validators=validators,
        )


def test_crash_after_state_commit_recovers_pointer_bundle(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, gate, _ = _round(root, validators, "promote")
    pointer_path = root / "control/incumbent-promoted-pointers.json"
    old_pointer = pointer_path.read_bytes()

    def crash(stage: str) -> None:
        if stage == "state-committed":
            raise RuntimeError("synthetic power loss")

    with pytest.raises(RuntimeError, match="power loss"):
        apply_registered_gate(
            campaign_root=root,
            candidate_registration_path=candidate,
            gate_registration_path=gate,
            validators=validators,
            fault_injector=crash,
        )
    assert read_state(root / "control/campaign-state.json")["incumbent_checkpoint_id"] == "C[1]"
    assert pointer_path.read_bytes() == old_pointer

    outcome = apply_registered_gate(
        campaign_root=root,
        candidate_registration_path=candidate,
        gate_registration_path=gate,
        validators=validators,
    )
    pointers = json.loads(pointer_path.read_text())
    assert outcome["classification"] == "promote"
    assert pointers["incumbent"]["policy_id"] == "C[1]"
    assert pointers["incumbent"] == pointers["promoted"]


def test_state_revision_conflict_never_updates_model_pointers(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    validators = _fake_validators()
    _bootstrap(root, validators)
    candidate, gate, _ = _round(root, validators, "promote")
    pointer_path = root / "control/incumbent-promoted-pointers.json"
    old_pointer = pointer_path.read_bytes()

    def concurrent_rejection(stage: str) -> None:
        if stage != "before-state-cas":
            return
        current = read_state(root / "control/campaign-state.json")
        rejected = transition_state(
            current,
            Phase.CANDIDATE_REJECTED,
            reason="synthetic concurrent controller won CAS",
        )
        write_state(
            root / "control/campaign-state.json",
            rejected,
            expected_current=current,
        )

    with pytest.raises(TransitionError, match="changed before compare-and-swap"):
        apply_registered_gate(
            campaign_root=root,
            candidate_registration_path=candidate,
            gate_registration_path=gate,
            validators=validators,
            fault_injector=concurrent_rejection,
        )
    assert pointer_path.read_bytes() == old_pointer
    assert read_state(root / "control/campaign-state.json")["phase"] == (
        Phase.CANDIDATE_REJECTED.value
    )
