"""Authorized, resumable ADR 0147 training for one exact-supply arm."""

from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.ranking_train import (
    GroupedRankingAdapter,
    RankingTrainingConfig,
    train_ranking,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ADR_ID,
    ARM_INPUT_CONTRACTS,
    ARMS,
    CATALOG_BLAKE3,
    EXPERIMENT_ID,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    NORMALIZATION_CONTRACT,
    PROTOCOL_ID,
    S1ExactSupplyCache,
    randomly_transform_s1_exact_supply_batch,
)
from cascadia_mlx.s1_exact_supply_mlx_metrics import (
    benchmark_s1_exact_supply,
    evaluate_s1_exact_supply,
)
from cascadia_mlx.s1_exact_supply_mlx_model import (
    FROZEN_PARAMETER_COUNT,
    S1ExactSupplyModelConfig,
    S1ExactSupplyRanker,
    parameter_count,
    parameter_layout_blake3,
    s1_exact_supply_loss,
    score_s1_exact_supply_batch,
)

TRAINING_SEED = 2026061707
EPOCHS = 30
GROUP_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 250
VALIDATION_PATIENCE = 6
ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
}
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
}


@dataclass(frozen=True)
class S1ExactSupplyTrainingProtocol:
    """All scientific constants held equal across C0, T1, and T2."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    epochs: int = EPOCHS
    group_batch_size: int = GROUP_BATCH_SIZE
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    validation_patience: int = VALIDATION_PATIENCE
    augmentation: str = "uniform-per-group-full-d6-rust-contract-ids-0-through-11"

    def validate(self) -> None:
        if self != S1ExactSupplyTrainingProtocol():
            raise ValueError("ADR 0147 training protocol drifted")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class S1ExactSupplyTrainingConfig:
    """One production arm invocation bound to explicit launch controls."""

    train_dataset: Path
    validation_dataset: Path
    cache: Path
    run_dir: Path
    output: Path
    authorization: Path
    preflight: Path
    arm: str
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = EPOCHS
    group_batch_size: int = GROUP_BATCH_SIZE
    maximum_actions_per_batch: int = GRADED_ORACLE_PACKED_ACTION_LIMIT
    maximum_group_actions: int = GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    seed: int = TRAINING_SEED
    checkpoint_steps: int = CHECKPOINT_STEPS
    validation_patience: int = VALIDATION_PATIENCE
    resume: bool = False
    model: S1ExactSupplyModelConfig = field(default_factory=S1ExactSupplyModelConfig)

    def validate(self) -> None:
        S1ExactSupplyTrainingProtocol().validate()
        if self.arm not in ARMS or self.model.arm != self.arm:
            raise ValueError("S1 training arm and model routing must agree")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("ADR 0147 permits only the frozen open train and validation rows")
        if self.init_model_dir is not None:
            raise ValueError("ADR 0147 prohibits warm starts")
        if (
            self.epochs != EPOCHS
            or self.group_batch_size != GROUP_BATCH_SIZE
            or self.maximum_actions_per_batch != GRADED_ORACLE_PACKED_ACTION_LIMIT
            or self.maximum_group_actions != GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS
            or self.learning_rate != LEARNING_RATE
            or self.weight_decay != WEIGHT_DECAY
            or self.seed != TRAINING_SEED
            or self.checkpoint_steps != CHECKPOINT_STEPS
            or self.validation_patience != VALIDATION_PATIENCE
        ):
            raise ValueError("ADR 0147 optimizer, examples, or schedule drifted")
        self.model.validate()


def s1_exact_supply_adapter(
    cache: S1ExactSupplyCache,
    arm: str,
) -> GroupedRankingAdapter:
    """Bind one arm to the shared rows, objective, D6 schedule, and evaluator."""
    return GroupedRankingAdapter(
        kind="s1-exact-supply-ranking",
        dataset_factory=lambda path: cache.bind_dataset(path, arm=arm),
        model_factory=lambda values: S1ExactSupplyRanker(
            S1ExactSupplyModelConfig.from_dict(values)
        ),
        new_model=S1ExactSupplyRanker,
        load_promoted=_warm_start_forbidden,
        loss=s1_exact_supply_loss,
        score_batch=score_s1_exact_supply_batch,
        augment_batch=randomly_transform_s1_exact_supply_batch,
        evaluate=evaluate_s1_exact_supply,
        selection_metric="mean_top64_retained_r4800_regret",
        accuracy_metric="top64_r4800_winner_recall",
        tertiary_metric="r4800_value_mae",
        batch_kwargs={
            "maximum_actions_per_batch": GRADED_ORACLE_PACKED_ACTION_LIMIT,
            "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        },
    )


def train_s1_exact_supply(config: S1ExactSupplyTrainingConfig) -> dict[str, Any]:
    """Train and evaluate one arm only after immutable launch controls pass."""
    config.validate()
    cache = S1ExactSupplyCache(config.cache, require_complete=True)
    controls = validate_launch_controls(
        config.authorization,
        config.preflight,
        arm=config.arm,
        cache_id=cache.cache_id,
        collision_witness_id=cache.manifest["collision_witness"]["witness_id"],
    )
    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    if "gpu" not in str(mx.default_device()).lower():
        raise ValueError("ADR 0147 production training requires the MLX GPU")
    parameter_counts = {
        arm: parameter_count(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        )
        for arm in ARMS
    }
    if len(set(parameter_counts.values())) != 1:
        raise ValueError("S1 model parameter budget differs across arms")
    if set(parameter_counts.values()) != {FROZEN_PARAMETER_COUNT}:
        raise ValueError("S1 model parameter budget drifted from the frozen count")
    parameter_layouts = {
        arm: parameter_layout_blake3(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=arm))
        )
        for arm in ARMS
    }
    if len(set(parameter_layouts.values())) != 1:
        raise ValueError("S1 model parameter layout differs across arms")
    _write_control_lock(
        config,
        cache,
        controls,
        parameter_counts,
        parameter_layouts,
    )

    adapter = s1_exact_supply_adapter(cache, config.arm)
    training_report = train_ranking(
        _ranking_training_config(config),
        adapter=adapter,
    )
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        config.run_dir,
        pointer="best",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_factory=lambda values: S1ExactSupplyRanker(
            S1ExactSupplyModelConfig.from_dict(values)
        ),
    )
    validation = cache.bind_dataset(config.validation_dataset, arm=config.arm)
    metrics = evaluate_s1_exact_supply(model, validation, config.group_batch_size)
    performance = benchmark_s1_exact_supply(model, validation)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": config.arm,
        "host": _normalize_host(socket.gethostname().split(".")[0]),
        "cache_id": cache.cache_id,
        "authorization_id": controls["authorization"]["authorization_id"],
        "preflight_id": controls["preflight"]["preflight_id"],
        "protocol": S1ExactSupplyTrainingProtocol().to_dict(),
        "normalization": NORMALIZATION_CONTRACT,
        "input_contract": ARM_INPUT_CONTRACTS[config.arm],
        "collision_witness_id": cache.manifest["collision_witness"]["witness_id"],
        "model": {
            "config": model.config.to_dict(),
            "parameter_count": parameter_count(model),
            "cross_arm_parameter_counts": parameter_counts,
            "parameter_layout_blake3": parameter_layout_blake3(model),
            "cross_arm_parameter_layout_blake3": parameter_layouts,
            "parameter_count_scope": "all-trainable-scalars",
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
            "model_blake3": _checksum(checkpoint / "model.safetensors"),
        },
        "training_report_blake3": _checksum(config.run_dir / "final-report.json"),
        "training": training_report,
        "metrics": metrics,
        "performance": performance,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_read": False,
        },
        "claims": {
            "offline_comparison_complete": True,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    report["scientific_identity"] = {
        key: report[key]
        for key in (
            "experiment_id",
            "protocol_id",
            "adr",
            "arm",
            "host",
            "cache_id",
            "authorization_id",
            "preflight_id",
            "protocol",
            "normalization",
            "input_contract",
            "collision_witness_id",
            "model",
            "checkpoint",
            "metrics",
            "performance",
            "information_boundary",
        )
    }
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    _write_json_atomic(config.output, report)
    _write_json_atomic(config.run_dir / "s1-arm-report.json", report)
    return report


def validate_launch_controls(
    authorization_path: Path,
    preflight_path: Path,
    *,
    arm: str,
    cache_id: str,
    collision_witness_id: str,
) -> dict[str, dict[str, Any]]:
    """Fail closed unless an explicit authorization and matching host preflight exist."""
    authorization = _read_json(authorization_path, "S1 authorization")
    identity = authorization.get("identity")
    parameter_counts = {
        candidate: parameter_count(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=candidate))
        )
        for candidate in ARMS
    }
    parameter_layouts = {
        candidate: parameter_layout_blake3(
            S1ExactSupplyRanker(S1ExactSupplyModelConfig(arm=candidate))
        )
        for candidate in ARMS
    }
    protocol = S1ExactSupplyTrainingProtocol().to_dict()
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("protocol_id") != PROTOCOL_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(identity, dict)
        or _canonical_blake3(identity) != authorization.get("authorization_id")
        or identity.get("cache_id") != cache_id
        or identity.get("catalog_blake3") != CATALOG_BLAKE3
        or identity.get("protocol") != protocol
        or identity.get("protocol_blake3") != _canonical_blake3(protocol)
        or identity.get("normalization") != NORMALIZATION_CONTRACT
        or identity.get("arm_input_contracts") != ARM_INPUT_CONTRACTS
        or identity.get("collision_witness_id") != collision_witness_id
        or identity.get("authorized_arms") != list(ARMS)
        or identity.get("independent_replay_role")
        != "independent-replay-control"
        or identity.get("cross_arm_parameter_counts") != parameter_counts
        or identity.get("cross_arm_parameter_layout_blake3")
        != parameter_layouts
    ):
        raise ValueError("S1 production authorization is missing, stale, or invalid")

    preflight = _read_json(preflight_path, "S1 host preflight")
    preflight_identity = preflight.get("scientific_identity")
    checks = preflight.get("checks")
    if (
        preflight.get("schema_version") != 1
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("adr") != ADR_ID
        or not isinstance(preflight_identity, dict)
        or _canonical_blake3(preflight_identity) != preflight.get("preflight_id")
        or preflight_identity.get("authorization_id") != authorization["authorization_id"]
        or preflight_identity.get("cache_id") != cache_id
        or preflight_identity.get("arm") != arm
        or preflight_identity.get("host") != ARM_HOSTS[arm]
        or preflight_identity.get("cross_arm_parameter_counts")
        != parameter_counts
        or preflight_identity.get("cross_arm_parameter_layout_blake3")
        != parameter_layouts
        or not isinstance(checks, dict)
        or any(
            checks.get(field) is not True
            for field in (
                "immutable_bundle_verified",
                "authorization_verified",
                "cache_verified",
                "dataset_manifests_verified",
                "apple_silicon_verified",
                "mlx_gpu_verified",
                "python_bytecode_disabled",
                "host_assignment_verified",
            )
        )
        or checks.get("production_training_started") is not False
    ):
        raise ValueError("S1 host preflight is missing, stale, or invalid")
    return {
        "authorization": authorization,
        "preflight": preflight,
    }


def _ranking_training_config(
    config: S1ExactSupplyTrainingConfig,
) -> RankingTrainingConfig:
    """Project S1 controls onto the shared trainer's exact manifest schema."""
    return RankingTrainingConfig(
        train_dataset=config.train_dataset,
        validation_dataset=config.validation_dataset,
        run_dir=config.run_dir,
        additional_train_datasets=config.additional_train_datasets,
        regression_validation_datasets=config.regression_validation_datasets,
        init_model_dir=config.init_model_dir,
        epochs=config.epochs,
        group_batch_size=config.group_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
        checkpoint_steps=config.checkpoint_steps,
        validation_patience=config.validation_patience,
        resume=config.resume,
        model=config.model,
    )


def _write_control_lock(
    config: S1ExactSupplyTrainingConfig,
    cache: S1ExactSupplyCache,
    controls: dict[str, dict[str, Any]],
    parameter_counts: dict[str, int],
    parameter_layouts: dict[str, str],
) -> None:
    lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": config.arm,
        "cache_id": cache.cache_id,
        "cache_manifest_blake3": _checksum(cache.manifest_path),
        "train_manifest_blake3": _checksum(config.train_dataset / "dataset.json"),
        "validation_manifest_blake3": _checksum(
            config.validation_dataset / "dataset.json"
        ),
        "authorization_id": controls["authorization"]["authorization_id"],
        "preflight_id": controls["preflight"]["preflight_id"],
        "protocol": S1ExactSupplyTrainingProtocol().to_dict(),
        "model": config.model.to_dict(),
        "normalization": NORMALIZATION_CONTRACT,
        "input_contract": ARM_INPUT_CONTRACTS[config.arm],
        "collision_witness_id": cache.manifest["collision_witness"]["witness_id"],
        "cross_arm_parameter_counts": parameter_counts,
        "parameter_layout_blake3": parameter_layouts[config.arm],
        "cross_arm_parameter_layout_blake3": parameter_layouts,
    }
    lock["lock_id"] = _canonical_blake3(lock)
    path = config.run_dir / "s1-control-lock.json"
    if path.exists():
        if _read_json(path, "S1 control lock") != lock:
            raise ValueError("S1 resume control lock drifted")
        return
    config.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, lock)


def _warm_start_forbidden(_path: Path) -> S1ExactSupplyRanker:
    raise ValueError("ADR 0147 prohibits warm starts")


def _normalize_host(value: str) -> str:
    short = value.removesuffix(".local")
    return HOST_ALIASES.get(short, short)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-if-present", action="store_true")
    args = parser.parse_args()
    resume = args.resume or (
        args.resume_if_present and (args.run_dir / "latest.json").is_file()
    )
    report = train_s1_exact_supply(
        S1ExactSupplyTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            cache=args.cache,
            run_dir=args.run_dir,
            output=args.output,
            authorization=args.authorization,
            preflight=args.preflight,
            arm=args.arm,
            resume=resume,
            model=S1ExactSupplyModelConfig(arm=args.arm),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
