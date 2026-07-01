from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cascadia_mlx.r2_map_contracts as contracts
import pytest
from cascadia_mlx.r2_map_contracts import (
    ALLOWED_HOSTS,
    CAMPAIGN_ID,
    CAMPAIGN_RELATIVE_PATH,
    CAMPAIGN_ROOT,
    CAMPAIGN_STATE_JSON_SCHEMA,
    DECISION_LOG_JSON_SCHEMA,
    PHASE_HOST_INTENTS,
    STORAGE_HOST,
    STORAGE_SUPERSESSION_GENESIS_JSON_SCHEMA,
    DecisionLogError,
    HostIntent,
    Phase,
    StateValidationError,
    StorageContract,
    StoragePreflightError,
    TransitionError,
    append_decision,
    content_sha256,
    new_campaign_state,
    new_storage_supersession_genesis,
    preflight_storage,
    read_decision_log,
    read_state,
    transition_state,
    validate_state,
    validate_storage_supersession_genesis,
    validate_transition,
    write_state,
    write_storage_supersession_genesis,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _storage(tmp_path: Path, *, min_free: int = 1) -> tuple[StorageContract, dict[str, str]]:
    volume = tmp_path / "john1-disk"
    relative = Path("Users/johnherrick/cascadia-bench/r2-map-v1")
    root = volume / relative
    (root / "tmp/cargo-target").mkdir(parents=True)
    root.chmod(0o700)
    contract = StorageContract(
        expected_host="john1-test",
        expected_volume=volume,
        campaign_relative_path=relative,
        campaign_root=root,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        min_free_bytes=min_free,
        campaign_budget_bytes=10_000_000,
        per_run_budget_bytes=1_000_000,
    )
    environment = {
        "TMPDIR": str(root / "tmp"),
        "CARGO_TARGET_DIR": str(root / "tmp/cargo-target"),
    }
    return contract, environment


def _preflight(contract: StorageContract, environment: dict[str, str], **kwargs: object) -> dict:
    return preflight_storage(
        contract=contract,
        environ=environment,
        mount_checker=lambda path: path == contract.expected_volume,
        disk_usage=lambda _path: SimpleNamespace(total=20_000_000, used=0, free=20_000_000),
        current_host_id=contract.expected_host,
        **kwargs,
    )


def _through_bootstrap() -> dict:
    state = new_campaign_state(now="2026-06-18T00:00:00.000Z")
    state = transition_state(
        state,
        Phase.BOOTSTRAP_GENERATING,
        reason="contracts frozen",
        now="2026-06-18T00:00:01.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_VALIDATED,
        reason="all bootstrap shards validated",
        generation_manifest_sha256=HASH_A,
        completed_shard_hosts=ALLOWED_HOSTS,
        now="2026-06-18T00:00:02.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_TRAINING,
        reason="bootstrap corpus admitted",
        now="2026-06-18T00:00:03.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        reason="bootstrap candidate independently verified",
        candidate_checkpoint_sha256=HASH_B,
        now="2026-06-18T00:00:04.000Z",
    )
    return transition_state(
        state,
        Phase.INCUMBENT_PROMOTED,
        reason="bootstrap candidate passed its gate",
        now="2026-06-18T00:00:05.000Z",
    )


def _through_training() -> dict:
    state = _through_bootstrap()
    state = transition_state(
        state, Phase.ROUND_ALLOCATED, reason="allocate R[0]", now="2026-06-18T00:00:06.000Z"
    )
    state = transition_state(
        state, Phase.GENERATING, reason="start 45 minute window", now="2026-06-18T00:00:07.000Z"
    )
    state = transition_state(
        state,
        Phase.LOCAL_SHARDS_COMPLETE,
        reason="all workers atomically closed shards",
        completed_shard_hosts=ALLOWED_HOSTS,
        now="2026-06-18T00:00:08.000Z",
    )
    state = transition_state(
        state,
        Phase.COLLECTED_AND_VALIDATED,
        reason="john1 validated merged manifest",
        generation_manifest_sha256=HASH_A,
        now="2026-06-18T00:00:09.000Z",
    )
    return transition_state(
        state,
        Phase.TRAINING_AND_BENCHMARKING,
        reason="launch blocking training and prior-checkpoint benchmark",
        now="2026-06-18T00:00:10.000Z",
    )


def test_frozen_campaign_identity_and_schemas_exclude_john4() -> None:
    assert CAMPAIGN_ID == "r2-map-expert-iteration-v1"
    assert STORAGE_HOST == "john1"
    assert Path("Users/johnherrick/cascadia-bench/r2-map-v1") == CAMPAIGN_RELATIVE_PATH
    assert Path("/Users/johnherrick/cascadia-bench/r2-map-v1") == CAMPAIGN_ROOT
    assert set(ALLOWED_HOSTS) == {"john1", "john2", "john3"}
    serialized = json.dumps(
        {
            "state": CAMPAIGN_STATE_JSON_SCHEMA,
            "decisions": DECISION_LOG_JSON_SCHEMA,
            "storage_genesis": STORAGE_SUPERSESSION_GENESIS_JSON_SCHEMA,
            "intents": PHASE_HOST_INTENTS,
        },
        default=str,
    ).casefold()
    assert "john4" not in serialized


def test_storage_supersession_genesis_anchors_full_legacy_hashes_and_new_chain(
    tmp_path: Path,
) -> None:
    state = new_campaign_state(now="2026-06-18T00:00:00.000Z")
    legacy_state = "87d6" + "a" * 56 + "7582"
    legacy_head = "a7c6" + "b" * 56 + "0dd8"
    authorization = "c" * 64
    genesis = new_storage_supersession_genesis(
        legacy_campaign_state_sha256=legacy_state,
        legacy_decision_head_sha256=legacy_head,
        canonical_state=state,
        authorization_sha256=authorization,
        now="2026-06-18T01:00:00.000Z",
    )
    assert genesis["legacy_campaign_state_sha256"] == legacy_state
    assert genesis["legacy_decision_head_sha256"] == legacy_head
    assert genesis["canonical_campaign_state_sha256"] == state["state_sha256"]
    assert genesis["legacy_storage_host"] == "john2"
    assert genesis["canonical_storage_host"] == "john1"
    path = tmp_path / "decision-log.jsonl"
    write_storage_supersession_genesis(path, genesis)
    with pytest.raises(DecisionLogError, match="refusing to replace"):
        write_storage_supersession_genesis(path, genesis)
    decision = append_decision(
        path,
        actor="root-orchestrator",
        triggering_evidence=["user superseded the former SSD directive"],
        alternatives_considered=["copy legacy tree", "use canonical John1 storage"],
        chosen_action="use only canonical John1 storage and retain legacy evidence immutably",
        affected_artifacts=["control/campaign-state.json", "control/decision-log.jsonl"],
        rollback_path="require a new explicit user storage directive",
        state=state,
        now="2026-06-18T01:00:01.000Z",
    )
    assert decision["sequence"] == 1
    assert decision["previous_decision_sha256"] == genesis["decision_sha256"]
    entries = read_decision_log(path)
    assert entries[0]["storage_supersession_genesis"] is True
    assert entries[1]["decision_sha256"] == decision["decision_sha256"]

    tampered = dict(genesis)
    tampered["legacy_decision_head_sha256"] = "d" * 64
    with pytest.raises(DecisionLogError, match="hash differs"):
        validate_storage_supersession_genesis(tampered)


def test_storage_preflight_accepts_only_authoritative_john1_paths(tmp_path: Path) -> None:
    contract, environment = _storage(tmp_path)
    output = contract.campaign_root / "datasets/bootstrap/part-0.replay"
    proof = _preflight(contract, environment, configured_paths={"output": output})
    assert proof["campaign_root"] == str(contract.campaign_root.resolve())
    assert proof["storage_host"] == contract.expected_host
    assert proof["configured_paths"] == {"output": str(output.resolve())}
    assert proof["atomic_rename_fsync"] is True
    assert proof["atomic_probe_directory"] == str(
        (contract.campaign_root / "tmp").resolve()
    )
    assert not list(contract.campaign_root.glob(".r2map-preflight-*"))
    assert not list((contract.campaign_root / "tmp").glob(".r2map-preflight-*"))


def test_storage_preflight_accepts_registered_build_target_on_john1(
    tmp_path: Path,
) -> None:
    contract, environment = _storage(tmp_path)
    cargo_target = contract.campaign_root / "build/run-controller/cargo-target"
    cargo_target.mkdir(parents=True)
    environment["CARGO_TARGET_DIR"] = str(cargo_target)
    proof = _preflight(contract, environment)
    assert proof["cargo_target_dir"] == str(cargo_target.resolve())


def test_storage_preflight_rejects_nested_or_external_apfs_workspace(tmp_path: Path) -> None:
    contract, environment = _storage(tmp_path)
    with pytest.raises(StoragePreflightError, match="nested or external APFS"):
        preflight_storage(
            contract=contract,
            environ=environment,
            current_host_id=contract.expected_host,
            apfs_workspace_spec=object(),
        )


@pytest.mark.parametrize("observed_host", ["john2", "john3", None])
def test_storage_preflight_rejects_non_storage_hosts(
    tmp_path: Path, observed_host: str | None
) -> None:
    contract, environment = _storage(tmp_path)
    with pytest.raises(StoragePreflightError, match="authoritative campaign storage is remote"):
        preflight_storage(
            contract=contract,
            environ=environment,
            current_host_id=observed_host,
        )


@pytest.mark.parametrize(
    ("root", "message"),
    [
        (
            Path("/Volumes/John_1/cascadia-cluster/r2-map-v1/new-run"),
            "frozen legacy evidence",
        ),
        (Path("/Users/john2/cascadia-bench/r2-map-v1/new-run"), "frozen legacy evidence"),
    ],
)
def test_storage_preflight_rejects_frozen_legacy_storage(
    root: Path, message: str
) -> None:
    contract = StorageContract(
        expected_host="john1-test",
        expected_volume=root.parent,
        campaign_relative_path=Path(root.name),
        campaign_root=root,
        min_free_bytes=1,
        campaign_budget_bytes=1,
        per_run_budget_bytes=1,
    )
    with pytest.raises(StoragePreflightError, match=message):
        preflight_storage(contract=contract, current_host_id=contract.expected_host)


@pytest.mark.parametrize("failure", ["missing-mount", "read-only", "low-space"])
def test_storage_preflight_fails_closed_for_unsafe_volume(tmp_path: Path, failure: str) -> None:
    contract, environment = _storage(tmp_path, min_free=100)
    options: dict = {
        "contract": contract,
        "environ": environment,
        "current_host_id": contract.expected_host,
        "mount_checker": lambda _path: failure != "missing-mount",
        "writable_checker": lambda _path: failure != "read-only",
        "disk_usage": lambda _path: SimpleNamespace(
            total=1_000,
            used=950 if failure == "low-space" else 0,
            free=50 if failure == "low-space" else 1_000,
        ),
    }
    with pytest.raises(StoragePreflightError):
        preflight_storage(**options)


def test_storage_preflight_rejects_wrong_volume_and_internal_fallback(tmp_path: Path) -> None:
    contract, environment = _storage(tmp_path)
    wrong_root = tmp_path / "other-volume/cascadia-cluster/r2-map-v1"
    wrong_root.mkdir(parents=True)
    wrong = StorageContract(
        expected_host=contract.expected_host,
        expected_volume=contract.expected_volume,
        campaign_relative_path=contract.campaign_relative_path,
        campaign_root=wrong_root,
        expected_uid=contract.expected_uid,
        expected_gid=contract.expected_gid,
        required_mode=contract.required_mode,
        min_free_bytes=1,
        campaign_budget_bytes=1_000_000,
        per_run_budget_bytes=1_000_000,
    )
    with pytest.raises(StoragePreflightError, match="exact required path"):
        preflight_storage(
            contract=wrong,
            environ={
                "TMPDIR": str(wrong_root / "tmp"),
                "CARGO_TARGET_DIR": str(wrong_root / "tmp/cargo-target"),
            },
            mount_checker=lambda _path: True,
            current_host_id=contract.expected_host,
        )

    internal_fallback = tmp_path / "internal-disk/new-output.bin"
    with pytest.raises(StoragePreflightError, match="escapes campaign root"):
        _preflight(
            contract,
            environment,
            configured_paths={"forbidden_internal_fallback": internal_fallback},
        )
    assert not internal_fallback.exists()


def test_storage_preflight_rejects_parent_escape_symlink_and_bad_environment(
    tmp_path: Path,
) -> None:
    contract, environment = _storage(tmp_path)
    parent_escape = contract.campaign_root / "datasets/../outside.bin"
    with pytest.raises(StoragePreflightError, match=r"forbidden '\.\.'"):
        _preflight(contract, environment, configured_paths={"escape": parent_escape})

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = contract.campaign_root / "datasets"
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoragePreflightError, match="symlink"):
        _preflight(
            contract,
            environment,
            configured_paths={"symlink_escape": symlink / "data.bin"},
        )

    symlink.unlink()
    bad_environment = dict(environment)
    bad_environment["TMPDIR"] = str(tmp_path / "internal-tmp")
    with pytest.raises(StoragePreflightError, match="TMPDIR"):
        _preflight(contract, bad_environment)


def test_budget_scan_does_not_follow_unconfigured_convenience_symlink(tmp_path: Path) -> None:
    contract, environment = _storage(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large.bin").write_bytes(b"x" * 1_000_000)
    (contract.campaign_root / "tmp/pytest-current").symlink_to(outside, target_is_directory=True)
    proof = _preflight(contract, environment)
    assert proof["campaign_bytes"] < 1_000_000


def test_compact_status_preflight_never_walks_bulk_campaign_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, environment = _storage(tmp_path)

    def reject_walk(*_args: object, **_kwargs: object):
        raise AssertionError("compact publisher must not scan campaign descendants")

    monkeypatch.setattr(contracts.os, "walk", reject_walk)
    proof = _preflight(contract, environment, measure_campaign_bytes=False)
    assert proof["campaign_bytes"] is None
    assert proof["campaign_bytes_measured"] is False


def test_budget_scan_tolerates_descendant_removed_after_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, environment = _storage(tmp_path)
    vanishing = contract.campaign_root / "tmp/cargo-target/vanishing.o"
    vanishing.write_bytes(b"transient")
    original_lstat = Path.lstat
    injected = False

    def racing_lstat(path: Path):
        nonlocal injected
        if path == vanishing and not injected:
            injected = True
            vanishing.unlink()
            raise FileNotFoundError(vanishing)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    proof = _preflight(contract, environment)
    assert injected is True
    assert proof["atomic_rename_fsync"] is True


def test_storage_preflight_enforces_run_and_campaign_budgets(tmp_path: Path) -> None:
    contract, environment = _storage(tmp_path)
    with pytest.raises(StoragePreflightError, match="requested run budget"):
        _preflight(contract, environment, expected_run_bytes=contract.per_run_budget_bytes + 1)
    budget_contract = StorageContract(
        expected_host=contract.expected_host,
        expected_volume=contract.expected_volume,
        campaign_relative_path=contract.campaign_relative_path,
        campaign_root=contract.campaign_root,
        expected_uid=contract.expected_uid,
        expected_gid=contract.expected_gid,
        required_mode=contract.required_mode,
        min_free_bytes=1,
        campaign_budget_bytes=1,
        per_run_budget_bytes=100,
    )
    (contract.campaign_root / "payload").write_bytes(b"too large")
    with pytest.raises(StoragePreflightError, match="campaign uses"):
        _preflight(budget_contract, environment)


def test_bootstrap_and_first_round_legal_path_is_hash_chained() -> None:
    state = _through_training()
    assert state["promotion_index"] == 0
    assert state["round_index"] == 0
    assert state["incumbent_checkpoint_id"] == "C[0]"
    assert state["generation_dataset_id"] == "G[0]"
    assert state["candidate_checkpoint_id"] == "T[0]"
    assert state["host_intents"] == {
        "john1": HostIntent.TRAIN,
        "john2": HostIntent.BENCHMARK,
        "john3": HostIntent.BENCHMARK,
    }
    assert state["revision"] == 10
    assert state["previous_state_sha256"] != state["state_sha256"]

    state = transition_state(
        state,
        Phase.CANDIDATE_VERIFIED_BENCHMARK_COMPLETE,
        reason="training and longitudinal benchmark terminal",
        candidate_checkpoint_sha256=HASH_B,
    )
    state = transition_state(state, Phase.PAIRED_CANDIDATE_GATE, reason="start paired gate")
    promoted = transition_state(
        state, Phase.INCUMBENT_PROMOTED, reason="fixed gate classified promote"
    )
    assert promoted["promotion_index"] == 1
    assert promoted["incumbent_checkpoint_id"] == "C[1]"
    assert promoted["incumbent_checkpoint_sha256"] == HASH_B


def test_illegal_transitions_and_incomplete_shards_are_rejected() -> None:
    initial = new_campaign_state()
    with pytest.raises(TransitionError, match="illegal campaign transition"):
        transition_state(initial, Phase.BOOTSTRAP_TRAINING, reason="skip generation")
    generating = transition_state(
        initial, Phase.BOOTSTRAP_GENERATING, reason="start bootstrap generation"
    )
    with pytest.raises(TransitionError, match="complete john1/john2/john3 shards"):
        transition_state(
            generating,
            Phase.BOOTSTRAP_VALIDATED,
            reason="one shard missing",
            generation_manifest_sha256=HASH_A,
            completed_shard_hosts=("john1", "john2"),
        )
    training = _through_training()
    with pytest.raises(TransitionError, match="illegal campaign transition"):
        transition_state(training, Phase.GENERATING, reason="forbidden overlap")


def test_state_tampering_and_john4_are_rejected_even_with_recomputed_hash() -> None:
    state = new_campaign_state()
    state["host_intents"] = {**state["host_intents"], "john4": "generate"}
    state["state_sha256"] = content_sha256(state, hash_field="state_sha256")
    with pytest.raises(StateValidationError, match="john4"):
        validate_state(state)


def test_legal_phase_name_cannot_smuggle_noncanonical_state_changes() -> None:
    current = _through_bootstrap()
    proposed = transition_state(current, Phase.ROUND_ALLOCATED, reason="allocate R[0]")
    proposed["incumbent_checkpoint_sha256"] = HASH_A
    proposed["state_sha256"] = content_sha256(proposed, hash_field="state_sha256")
    validate_state(proposed)
    with pytest.raises(TransitionError, match="not the canonical result"):
        validate_transition(current, proposed)


def test_atomic_state_compare_and_swap_rejects_stale_writer(tmp_path: Path) -> None:
    path = tmp_path / "control/campaign-state.json"
    initial = new_campaign_state()
    write_state(path, initial)
    first = transition_state(
        initial, Phase.BOOTSTRAP_GENERATING, reason="first writer starts bootstrap"
    )
    write_state(path, first, expected_current=initial)
    assert read_state(path) == first
    stale = transition_state(
        initial, Phase.BOOTSTRAP_GENERATING, reason="stale writer duplicates bootstrap"
    )
    with pytest.raises(TransitionError, match="changed before compare-and-swap"):
        write_state(path, stale, expected_current=initial)


def test_decision_log_is_append_only_fsynced_and_hash_chained(tmp_path: Path) -> None:
    path = tmp_path / "control/decision-log.jsonl"
    state = new_campaign_state()
    first = append_decision(
        path,
        actor="john1-owner",
        triggering_evidence=["authoritative John1 disk endpoint is mandatory and verified"],
        alternatives_considered=["inline checks in every caller", "shared fail-closed contract"],
        chosen_action="centralize storage and state contracts in a dependency-light module",
        affected_artifacts=["python/cascadia_mlx/r2_map_contracts.py"],
        rollback_path="remove the new module before any campaign state advances",
        state=state,
        now="2026-06-18T00:01:00.000Z",
    )
    second = append_decision(
        path,
        actor="john1-owner",
        triggering_evidence=["controller state needs crash-safe compare-and-swap"],
        alternatives_considered=["mutable state without a hash chain", "atomic hash-chained state"],
        chosen_action="use atomic rename plus previous-state hash binding",
        affected_artifacts=["tools/r2_map_expert_iteration.py"],
        rollback_path="restore state from the last verified JSON document",
        state=state,
        now="2026-06-18T00:02:00.000Z",
    )
    entries = read_decision_log(path)
    assert entries == [first, second]
    assert second["previous_decision_sha256"] == first["decision_sha256"]

    lines = path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["chosen_action"] = "tampered"
    path.write_text(json.dumps(tampered) + "\n" + lines[1] + "\n")
    with pytest.raises(DecisionLogError, match="hash"):
        read_decision_log(path)


def test_scientific_change_requires_amended_authorization(tmp_path: Path) -> None:
    path = tmp_path / "control/decision-log.jsonl"
    arguments = {
        "actor": "john1-owner",
        "triggering_evidence": ["hypothetical science change"],
        "alternatives_considered": ["keep frozen contract"],
        "chosen_action": "change a scientific invariant",
        "affected_artifacts": ["campaign contract"],
        "rollback_path": "retain frozen contract",
        "decision_kind": "scientific-contract-amendment",
    }
    with pytest.raises(DecisionLogError, match="amended authorization"):
        append_decision(path, **arguments)
    entry = append_decision(path, authorization_sha256=HASH_A, **arguments)
    assert entry["authorization_sha256"] == HASH_A


def test_pre_contract_genesis_decision_is_preserved_and_anchors_v1_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control/decision-log.jsonl"
    path.parent.mkdir(parents=True)
    genesis = {
        "schema_version": 1,
        "timestamp": "2026-06-18T04:36:07Z",
        "campaign_id": CAMPAIGN_ID,
        "decision_id": "initial-control-plane-wave-v1",
        "trigger": "goal execution began",
        "alternatives": ["sequential implementation", "partitioned owner agents"],
        "chosen_action": "partition owner work",
        "affected_artifacts": ["campaign root"],
        "rollback": "stop before contracts-ready advances",
        "scientific_contract_changed": False,
    }
    raw_genesis = json.dumps(genesis, separators=(",", ":"))
    path.write_text(raw_genesis + "\n")
    entry = append_decision(
        path,
        actor="john1-owner",
        triggering_evidence=["schemas are now frozen"],
        alternatives_considered=["rewrite genesis", "preserve and anchor genesis"],
        chosen_action="preserve genesis and hash-anchor its exact line bytes",
        affected_artifacts=["control/decision-log.jsonl"],
        rollback_path="retain the genesis-only log",
    )
    records = read_decision_log(path)
    expected_anchor = hashlib.sha256(raw_genesis.encode()).hexdigest()
    assert records[0]["legacy_genesis"] is True
    assert records[0]["decision_sha256"] == expected_anchor
    assert entry["sequence"] == 1
    assert entry["previous_decision_sha256"] == expected_anchor


def test_legacy_decision_is_rejected_after_v1_entry(tmp_path: Path) -> None:
    path = tmp_path / "decision-log.jsonl"
    append_decision(
        path,
        actor="john1-owner",
        triggering_evidence=["first"],
        alternatives_considered=["a"],
        chosen_action="b",
        affected_artifacts=["c"],
        rollback_path="d",
    )
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp": "now",
                    "campaign_id": CAMPAIGN_ID,
                    "decision_id": "late-legacy",
                    "trigger": "bad",
                    "alternatives": ["bad"],
                    "chosen_action": "bad",
                    "affected_artifacts": ["bad"],
                    "rollback": "bad",
                    "scientific_contract_changed": False,
                }
            )
            + "\n"
        )
    with pytest.raises(DecisionLogError, match="genesis line"):
        read_decision_log(path)
