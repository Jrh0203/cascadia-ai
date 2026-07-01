"""Immutable authorization and host preflight for ADR 0188."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cascadia_mlx.o1_ranking_intent_cache import ARMS
from cascadia_mlx.o1_ranking_model import (
    o1_ranking_loss,
    parameter_tensor_blake3,
)
from cascadia_mlx.o1_ranking_protocol import (
    EXPERIMENT_ID,
    LEARNING_RATE,
    TRAINING_SEED,
    WAVE_HOSTS,
    WEIGHT_DECAY,
)
from cascadia_mlx.o1_ranking_train import (
    O1RankingTrainingConfig,
    cross_arm_initialization,
    experiment_authorization_identity,
    initialize_model,
    input_identity,
    intent_batch_blake3,
    load_exact_r2_model,
    load_experiment_surfaces,
    require_production_runtime,
    runtime_identity,
    scientific_batch_blake3,
    verify_zero_init_prediction_parity,
    warm_start_identity,
)
from cascadia_mlx.run_manifest import source_provenance


def build_authorization(
    config: O1RankingTrainingConfig,
    *,
    output: Path,
) -> dict[str, Any]:
    """Verify the complete open bundle and authorize both cluster waves."""
    mx.set_default_device(mx.gpu)
    require_production_runtime(runtime_identity())
    source = source_provenance(Path(__file__).resolve().parents[2])
    surfaces = load_experiment_surfaces(config, verify_checksums=True)
    warm_start = warm_start_identity(config.warm_start_checkpoint)
    cross_arm = cross_arm_initialization(config.warm_start_checkpoint)
    identity = experiment_authorization_identity(
        config=config,
        surfaces=surfaces,
        warm_start=warm_start,
        cross_arm=cross_arm,
        source=source,
    )
    authorization = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "approved": True,
        "identity": identity,
        "authorization_id": _canonical_blake3(identity),
        "checks": {
            "complete_open_train_verified": True,
            "complete_open_validation_verified": True,
            "cohort_checksums_verified": True,
            "afterstate_checksums_and_model_inputs_verified": True,
            "intent_checksums_and_semantics_verified": True,
            "exact_r2_warm_start_verified": True,
            "cross_arm_initialization_verified": True,
            "sealed_test_untouched": True,
            "gameplay_not_run": True,
        },
    }
    _write_json_atomic(output, authorization)
    return authorization


def run_preflight(
    config: O1RankingTrainingConfig,
    *,
    authorization_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify one host, routed arm, deterministic update, and exact parity."""
    mx.set_default_device(mx.gpu)
    runtime = runtime_identity()
    require_production_runtime(runtime)
    expected_host = WAVE_HOSTS[config.wave][config.arm]
    if runtime["host"] != expected_host:
        raise ValueError(
            f"{config.wave} preflight for {config.arm} requires "
            f"{expected_host}, not {runtime['host']}"
        )
    source = source_provenance(Path(__file__).resolve().parents[2])
    surfaces = load_experiment_surfaces(config, verify_checksums=True)
    warm_start = warm_start_identity(config.warm_start_checkpoint)
    cross_arm = cross_arm_initialization(config.warm_start_checkpoint)
    expected_identity = experiment_authorization_identity(
        config=config,
        surfaces=surfaces,
        warm_start=warm_start,
        cross_arm=cross_arm,
        source=source,
    )
    authorization = _read_json(
        authorization_path,
        "O1 ranking authorization",
    )
    authorization_id = _canonical_blake3(expected_identity)
    if (
        authorization.get("approved") is not True
        or authorization.get("identity") != expected_identity
        or authorization.get("authorization_id") != authorization_id
    ):
        raise ValueError("O1 preflight authorization is stale or malformed")

    base = load_exact_r2_model(config.warm_start_checkpoint)
    first = initialize_model(config.arm, config.warm_start_checkpoint)
    parity = verify_zero_init_prediction_parity(
        base,
        first,
        surfaces.validation,
    )
    batch = surfaces.train.deterministic_training_batch(
        step=0,
        seed=TRAINING_SEED,
        groups_per_step=4,
    )
    first_result = _one_step(first, batch)
    second = initialize_model(config.arm, config.warm_start_checkpoint)
    second_result = _one_step(second, batch)
    if first_result != second_result:
        raise ValueError("O1 preflight one-step replay is not deterministic")
    checks = {
        "mlx_gpu_verified": True,
        "complete_open_bundle_verified": True,
        "open_data_only_verified": True,
        "warm_start_verified": True,
        "cross_arm_initialization_verified": True,
        "zero_init_prediction_parity_verified": True,
        "smoke_replay_verified": True,
        "candidate_cohort_alignment_verified": True,
        "routed_intent_alignment_verified": True,
        "sealed_test_untouched": True,
        "production_training_started": False,
    }
    identity = {
        "authorization_id": authorization_id,
        "arm": config.arm,
        "wave": config.wave,
        "host": runtime["host"],
        "runtime": runtime,
        "source_blake3": source["v2_source_blake3"],
        "input_bundle_id": _canonical_blake3(
            input_identity(config, surfaces)
        ),
        "warm_start_id": warm_start["warm_start_id"],
        "cross_arm_initialization_id": cross_arm[
            "cross_arm_initialization_id"
        ],
        "zero_init_prediction_blake3": parity["prediction_blake3"],
        "cohort_batch_blake3": scientific_batch_blake3(
            batch,
            surfaces.cohort,
        ),
        "intent_batch_blake3": intent_batch_blake3(batch),
        "one_step_replay": first_result,
    }
    preflight = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "arm": config.arm,
        "wave": config.wave,
        "identity": identity,
        "preflight_id": _canonical_blake3(identity),
        "checks": checks,
    }
    _write_json_atomic(output, preflight)
    return preflight


def _one_step(model: object, batch: object) -> dict[str, Any]:
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, o1_ranking_loss)
    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    return {
        "loss_f64_hex": float(loss.item()).hex(),
        "adapter_parameter_tensor_blake3": parameter_tensor_blake3(model),
    }


def _base_config(args: argparse.Namespace) -> O1RankingTrainingConfig:
    return O1RankingTrainingConfig(
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        r3_cache=args.r3_cache,
        s1_cache=args.s1_cache,
        cohort=args.cohort,
        afterstates=args.afterstates,
        intent_cache=args.intent_cache,
        warm_start_checkpoint=args.warm_start_checkpoint,
        run_dir=Path("."),
        output=Path("."),
        arm=getattr(args, "arm", ARMS[0]),
        wave=getattr(args, "wave", "primary"),
        smoke_steps=1,
    )


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--r3-cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--afterstates", type=Path, required=True)
    parser.add_argument("--intent-cache", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorize or preflight ADR 0188 cluster training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    _add_inputs(authorize)
    authorize.add_argument("--output", type=Path, required=True)
    preflight = subparsers.add_parser("preflight")
    _add_inputs(preflight)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--arm", choices=ARMS, required=True)
    preflight.add_argument(
        "--wave",
        choices=tuple(WAVE_HOSTS),
        required=True,
    )
    preflight.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    config = _base_config(args)
    if args.command == "authorize":
        report = build_authorization(config, output=args.output)
        result = {
            "authorization_id": report["authorization_id"],
            "output": str(args.output.resolve()),
        }
    else:
        report = run_preflight(
            config,
            authorization_path=args.authorization,
            output=args.output,
        )
        result = {
            "preflight_id": report["preflight_id"],
            "arm": report["arm"],
            "wave": report["wave"],
            "output": str(args.output.resolve()),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
