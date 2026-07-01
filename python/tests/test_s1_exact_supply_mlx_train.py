from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARM_INPUT_CONTRACTS,
    ARMS,
    CATALOG_BLAKE3,
    NORMALIZATION_CONTRACT,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
    S1ExactSupplyRanker,
    parameter_count,
    parameter_layout_blake3,
)
from cascadia_mlx.s1_exact_supply_mlx_train import (
    S1ExactSupplyTrainingConfig,
    S1ExactSupplyTrainingProtocol,
    _canonical_blake3,
    _ranking_training_config,
    _write_control_lock,
    validate_launch_controls,
)


def _config(tmp_path: Path, arm: str = ARMS[0]) -> S1ExactSupplyTrainingConfig:
    return S1ExactSupplyTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        cache=tmp_path / "cache",
        run_dir=tmp_path / "run",
        output=tmp_path / "report.json",
        authorization=tmp_path / "authorization.json",
        preflight=tmp_path / "preflight.json",
        arm=arm,
        model=S1ExactSupplyModelConfig(arm=arm),
    )


def test_training_protocol_rejects_data_optimizer_capacity_and_warm_start_drift(
    tmp_path: Path,
) -> None:
    _config(tmp_path).validate()
    with pytest.raises(ValueError, match="protocol drifted"):
        S1ExactSupplyTrainingProtocol(seed=1).validate()
    with pytest.raises(ValueError, match="only the frozen open"):
        S1ExactSupplyTrainingConfig(
            **{
                **_config(tmp_path).__dict__,
                "additional_train_datasets": (tmp_path / "extra",),
            }
        ).validate()
    with pytest.raises(ValueError, match="warm starts"):
        S1ExactSupplyTrainingConfig(
            **{
                **_config(tmp_path).__dict__,
                "init_model_dir": tmp_path / "warm",
            }
        ).validate()
    with pytest.raises(ValueError, match="optimizer"):
        S1ExactSupplyTrainingConfig(
            **{
                **_config(tmp_path).__dict__,
                "learning_rate": 2e-4,
            }
        ).validate()


def test_shared_trainer_projection_excludes_launch_only_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    projected = _ranking_training_config(config)
    assert projected.train_dataset == config.train_dataset
    assert projected.validation_dataset == config.validation_dataset
    assert projected.model == config.model
    assert not hasattr(projected, "cache")
    assert not hasattr(projected, "authorization")
    assert not hasattr(projected, "preflight")
    assert not hasattr(projected, "output")


def _write_launch_controls(tmp_path: Path, arm: str) -> tuple[Path, Path]:
    protocol = S1ExactSupplyTrainingProtocol().to_dict()
    counts = {
        candidate: parameter_count(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=candidate))
        )
        for candidate in ARMS
    }
    layouts = {
        candidate: parameter_layout_blake3(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=candidate))
        )
        for candidate in ARMS
    }
    authorization_identity = {
        "cache_id": "a" * 64,
        "catalog_blake3": CATALOG_BLAKE3,
        "protocol": protocol,
        "protocol_blake3": _canonical_blake3(protocol),
        "normalization": NORMALIZATION_CONTRACT,
        "arm_input_contracts": ARM_INPUT_CONTRACTS,
        "collision_witness_id": "b" * 64,
        "authorized_arms": list(ARMS),
        "independent_replay_role": "independent-replay-control",
        "cross_arm_parameter_counts": counts,
        "cross_arm_parameter_layout_blake3": layouts,
    }
    authorization = {
        "schema_version": 1,
        "experiment_id": "exact-semantic-supply-learned-comparison-v1",
        "protocol_id": "s1-exact-semantic-supply-mlx-comparison-v1",
        "adr": "0147",
        "approved": True,
        "identity": authorization_identity,
        "authorization_id": _canonical_blake3(authorization_identity),
    }
    preflight_identity = {
        "authorization_id": authorization["authorization_id"],
        "cache_id": "a" * 64,
        "arm": arm,
        "host": {
            ARMS[0]: "john1",
            ARMS[1]: "john2",
            ARMS[2]: "john3",
        }[arm],
        "cross_arm_parameter_counts": counts,
        "cross_arm_parameter_layout_blake3": layouts,
    }
    checks = {
        "immutable_bundle_verified": True,
        "authorization_verified": True,
        "cache_verified": True,
        "dataset_manifests_verified": True,
        "apple_silicon_verified": True,
        "mlx_gpu_verified": True,
        "python_bytecode_disabled": True,
        "host_assignment_verified": True,
        "production_training_started": False,
    }
    preflight = {
        "schema_version": 1,
        "experiment_id": authorization["experiment_id"],
        "protocol_id": authorization["protocol_id"],
        "adr": authorization["adr"],
        "scientific_identity": preflight_identity,
        "preflight_id": _canonical_blake3(preflight_identity),
        "checks": checks,
    }
    authorization_path = tmp_path / "authorization.json"
    preflight_path = tmp_path / "preflight.json"
    authorization_path.write_text(json.dumps(authorization))
    preflight_path.write_text(json.dumps(preflight))
    return authorization_path, preflight_path


def test_launch_controls_bind_arm_host_cache_authorization_and_python_b(
    tmp_path: Path,
) -> None:
    authorization, preflight = _write_launch_controls(tmp_path, ARMS[1])
    controls = validate_launch_controls(
        authorization,
        preflight,
        arm=ARMS[1],
        cache_id="a" * 64,
        collision_witness_id="b" * 64,
    )
    assert controls["preflight"]["scientific_identity"]["host"] == "john2"
    value = json.loads(preflight.read_text())
    value["checks"]["python_bytecode_disabled"] = False
    preflight.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="preflight"):
        validate_launch_controls(
            authorization,
            preflight,
            arm=ARMS[1],
            cache_id="a" * 64,
            collision_witness_id="b" * 64,
        )


def test_control_lock_freezes_normalization_collision_and_capacity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for root in (config.train_dataset, config.validation_dataset):
        root.mkdir()
        (root / "dataset.json").write_text("{}\n")
    cache_manifest = tmp_path / "cache.json"
    cache_manifest.write_text("{}\n")
    authorization, preflight = _write_launch_controls(tmp_path, ARMS[0])
    controls = validate_launch_controls(
        authorization,
        preflight,
        arm=ARMS[0],
        cache_id="a" * 64,
        collision_witness_id="b" * 64,
    )
    cache = SimpleNamespace(
        cache_id="a" * 64,
        manifest_path=cache_manifest,
        manifest={"collision_witness": {"witness_id": "b" * 64}},
    )
    counts = dict.fromkeys(ARMS, FROZEN_PARAMETER_COUNT)
    layouts = {
        arm: parameter_layout_blake3(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        )
        for arm in ARMS
    }
    _write_control_lock(config, cache, controls, counts, layouts)
    lock = json.loads((config.run_dir / "s1-control-lock.json").read_text())
    assert lock["normalization"] == NORMALIZATION_CONTRACT
    assert lock["input_contract"] == ARM_INPUT_CONTRACTS[ARMS[0]]
    assert lock["collision_witness_id"] == "b" * 64
    assert set(lock["cross_arm_parameter_counts"].values()) == {
        FROZEN_PARAMETER_COUNT
    }
    assert len(set(lock["cross_arm_parameter_layout_blake3"].values())) == 1
