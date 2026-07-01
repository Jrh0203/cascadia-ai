"""Per-decision failure atlas for completed ADR 0150 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ADR_ID,
    ARMS,
    CONTROL_ARM,
    EXPERIMENT_ID,
    LOW_SUPPLY_MAX_UNSEEN,
    PROTOCOL_ID,
    R3_LOCAL_PATCH_TOKEN,
    R3_TOKEN_TYPE_COUNT,
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_metrics import (
    CANDIDATE_CHUNK,
    _confidence_set,
    _retained_regret,
    _stable_ranking,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

ATLAS_SCHEMA_VERSION = 1
ATLAS_KIND = "r3-action-edit-validation-failure-atlas-v1"


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, values: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = blake3.blake3()
    with temporary.open("wb") as handle:
        for value in values:
            payload = (
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode()
            digest.update(payload)
            handle.write(payload)
    os.replace(temporary, path)
    return digest.hexdigest()


def _normalize_host(host: str) -> str:
    lowered = host.lower()
    for known in ("john1", "john2", "john3", "john4"):
        if known in lowered:
            return known
    return host


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one deterministic validation cohort."""
    if not records:
        return {
            "groups": 0,
            "top64_winner_recall": 0.0,
            "top64_confidence_set_coverage_95": 0.0,
            "mean_top64_retained_r4800_regret": 0.0,
        }
    groups = len(records)
    winner_ranks = [float(record["winner_rank"]) for record in records]
    confidence_ranks = [
        float(record["best_confidence_set_rank"])
        for record in records
        if record["best_confidence_set_rank"] is not None
    ]
    action_counts = [float(record["candidate_count"]) for record in records]
    winner_tokens = [float(record["winner_token_count"]) for record in records]
    winner_global_tokens = [
        float(record["winner_global_token_count"]) for record in records
    ]
    return {
        "groups": groups,
        "top64_winner_recall": sum(record["winner_recalled_top64"] for record in records)
        / groups,
        "top64_confidence_set_coverage_95": sum(
            record["confidence_set_covered_top64"] for record in records
        )
        / groups,
        "mean_top64_retained_r4800_regret": float(
            np.mean([record["top64_retained_r4800_regret"] for record in records])
        ),
        "winner_rank": {
            "p50": _quantile(winner_ranks, 0.50),
            "p90": _quantile(winner_ranks, 0.90),
            "p99": _quantile(winner_ranks, 0.99),
            "maximum": int(max(winner_ranks)),
        },
        "best_confidence_set_rank": {
            "observed": len(confidence_ranks),
            "p50": _quantile(confidence_ranks, 0.50),
            "p90": _quantile(confidence_ranks, 0.90),
            "p99": _quantile(confidence_ranks, 0.99),
            "maximum": int(max(confidence_ranks)) if confidence_ranks else None,
        },
        "candidate_count": {
            "mean": float(np.mean(action_counts)),
            "p50": _quantile(action_counts, 0.50),
            "p90": _quantile(action_counts, 0.90),
            "p99": _quantile(action_counts, 0.99),
            "maximum": int(max(action_counts)),
        },
        "winner_token_count": {
            "mean": float(np.mean(winner_tokens)),
            "p50": _quantile(winner_tokens, 0.50),
            "p90": _quantile(winner_tokens, 0.90),
            "p99": _quantile(winner_tokens, 0.99),
            "maximum": int(max(winner_tokens)),
        },
        "winner_global_token_count": {
            "mean": float(np.mean(winner_global_tokens)),
            "p50": _quantile(winner_global_tokens, 0.50),
            "p90": _quantile(winner_global_tokens, 0.90),
            "p99": _quantile(winner_global_tokens, 0.99),
            "maximum": int(max(winner_global_tokens)),
        },
    }


def _action_width_bucket(count: int) -> str:
    if count <= 512:
        return "0001-0512"
    if count <= 2048:
        return "0513-2048"
    if count <= 4096:
        return "2049-4096"
    return "4097-plus"


def _global_token_bucket(count: int) -> str:
    if count <= 16:
        return "00-16"
    if count <= 32:
        return "17-32"
    if count <= 48:
        return "33-48"
    return "49-plus"


def _local_token_counts(
    candidate_features: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    type_indices = np.argmax(
        candidate_features[..., :R3_TOKEN_TYPE_COUNT],
        axis=-1,
    )
    return np.sum(
        candidate_mask & (type_indices == R3_LOCAL_PATCH_TOKEN - 1),
        axis=1,
    ).astype(np.int64)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build fixed, interpretable strata without fitting post-hoc thresholds."""
    strata: dict[str, dict[str, list[dict[str, Any]]]] = {
        "phase": defaultdict(list),
        "action_width": defaultdict(list),
        "low_supply": defaultdict(list),
        "independent_draft_winner": defaultdict(list),
        "winner_global_token_count": defaultdict(list),
    }
    for record in records:
        strata["phase"][record["phase"]].append(record)
        strata["action_width"][_action_width_bucket(record["candidate_count"])].append(record)
        strata["low_supply"][str(record["low_supply"]).lower()].append(record)
        strata["independent_draft_winner"][
            str(record["independent_draft_winner"]).lower()
        ].append(record)
        strata["winner_global_token_count"][
            _global_token_bucket(record["winner_global_token_count"])
        ].append(record)
    return {
        "all": summarize_records(records),
        "strata": {
            dimension: {
                key: summarize_records(values)
                for key, values in sorted(groups.items())
            }
            for dimension, groups in strata.items()
        },
    }


def build_failure_atlas(
    *,
    model: R3ActionEditRanker,
    dataset: Any,
    arm: str,
    candidate_chunk: int = CANDIDATE_CHUNK,
) -> list[dict[str, Any]]:
    """Score every validation decision once and retain diagnostic group facts."""
    if arm not in ARMS or candidate_chunk <= 0:
        raise ValueError("failure-atlas arm or chunk is invalid")
    model.eval()
    records: list[dict[str, Any]] = []
    for row in range(dataset.group_count):
        batch = dataset.batch([row], arm=arm, transform_ids=[0])
        mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)[0]
        count = int(mask.sum())
        if count <= 0:
            raise ValueError("failure-atlas validation group has no candidates")
        parent = model.encode_parent(batch)
        mx.eval(parent)
        score_chunks: list[np.ndarray] = []
        uncertainty_chunks: list[np.ndarray] = []
        for start in range(0, count, candidate_chunk):
            end = min(start + candidate_chunk, count)
            prediction = model.predict(
                batch,
                candidate_slice=slice(start, end),
                parent_state=parent,
            )
            mx.eval(prediction.scores, prediction.standard_errors)
            score_chunks.append(np.asarray(prediction.scores)[0])
            uncertainty_chunks.append(np.asarray(prediction.standard_errors)[0])
        scores = np.concatenate(score_chunks)
        uncertainties = np.concatenate(uncertainty_chunks)
        hashes = np.asarray(batch.base.action_hash)[0, :count]
        winner = int(np.asarray(batch.base.selected_index)[0])
        r4800 = np.asarray(batch.base.r4800_mean)[0, :count]
        r4800_stddev = np.asarray(batch.base.r4800_stddev)[0, :count]
        r4800_samples = np.asarray(batch.base.r4800_samples)[0, :count]
        r4800_mask = np.asarray(batch.base.r4800_mask)[0, :count]
        if winner >= count or not r4800_mask[winner]:
            raise ValueError("failure-atlas winner lacks an R4800 label")
        ranking = _stable_ranking(scores, hashes)
        rank_by_candidate = np.empty(count, dtype=np.int64)
        rank_by_candidate[ranking] = np.arange(count, dtype=np.int64)
        confidence = _confidence_set(
            r4800,
            r4800_stddev,
            r4800_samples,
            r4800_mask,
            winner,
        )
        confidence_candidates = np.flatnonzero(confidence)
        best_confidence_rank = (
            int(np.min(rank_by_candidate[confidence_candidates]))
            if len(confidence_candidates)
            else None
        )
        retained = ranking[: min(64, count)]
        token_counts = np.asarray(batch.candidate_token_counts)[0, :count].astype(np.int64)
        if arm == CONTROL_ARM:
            local_counts = np.zeros(count, dtype=np.int64)
            global_counts = token_counts.copy()
        else:
            token_mask = np.asarray(batch.candidate_token_mask, dtype=np.bool_)[0, :count]
            candidate_features = np.asarray(batch.candidate_token_features)[
                0,
                :count,
                :,
            ]
            local_counts = _local_token_counts(candidate_features, token_mask)
            global_counts = token_counts - local_counts
        turn = int(np.asarray(batch.base.turn)[0])
        independent = int(np.asarray(batch.base.draft_kind)[0, winner]) == 1
        record = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "row": row,
            "group_id": int(np.asarray(batch.base.group_id)[0]),
            "game_index": int(np.asarray(batch.base.game_index)[0]),
            "turn": turn,
            "personal_turn": int(np.asarray(batch.base.personal_turn)[0]),
            "phase": "early" if turn < 27 else "middle" if turn < 54 else "late",
            "low_supply": 81 - turn <= LOW_SUPPLY_MAX_UNSEEN,
            "independent_draft_winner": independent,
            "candidate_count": count,
            "winner_index": winner,
            "winner_rank": int(rank_by_candidate[winner]),
            "winner_recalled_top64": bool(np.any(retained == winner)),
            "confidence_set_covered_top64": bool(np.any(confidence[retained])),
            "best_confidence_set_rank": best_confidence_rank,
            "confidence_set_size": int(np.sum(confidence)),
            "top64_retained_r4800_regret": _retained_regret(
                retained,
                r4800,
                r4800_mask,
            ),
            "winner_prediction": float(scores[winner]),
            "winner_uncertainty": float(uncertainties[winner]),
            "winner_r4800": float(r4800[winner]),
            "predicted_top1_index": int(ranking[0]),
            "predicted_top1_r4800": (
                float(r4800[ranking[0]]) if r4800_mask[ranking[0]] else None
            ),
            "winner_token_count": int(token_counts[winner]),
            "winner_local_token_count": int(local_counts[winner]),
            "winner_global_token_count": int(global_counts[winner]),
            "candidate_token_count_mean": float(np.mean(token_counts)),
            "candidate_token_count_p99": float(np.quantile(token_counts, 0.99)),
            "candidate_token_count_maximum": int(np.max(token_counts)),
            "nonfinite_scores": int(np.sum(~np.isfinite(scores))),
            "nonfinite_uncertainties": int(
                np.sum(~np.isfinite(uncertainties) | (uncertainties <= 0))
            ),
        }
        records.append(record)
    return records


def _latest_checkpoint(run_dir: Path) -> Path:
    latest = _read_json(run_dir / "latest.json", "latest checkpoint pointer")
    checkpoint_name = latest.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError("latest checkpoint pointer is malformed")
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_dir():
        raise ValueError(f"latest checkpoint is absent: {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--s1-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-chunk", type=int, default=CANDIDATE_CHUNK)
    args = parser.parse_args()

    origin = _read_json(args.origin_report, "origin report")
    authorization = _read_json(args.authorization, "authorization")
    arm = origin.get("arm")
    if (
        origin.get("experiment_id") != EXPERIMENT_ID
        or origin.get("protocol_id") != PROTOCOL_ID
        or origin.get("adr") != ADR_ID
        or origin.get("mode") != "production"
        or arm not in ARMS
    ):
        raise ValueError("origin report is not a production ADR 0150 arm")
    identity = authorization.get("identity")
    if (
        authorization.get("approved") is not True
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or not isinstance(identity, dict)
    ):
        raise ValueError("ADR 0150 authorization is invalid")
    expected_open_data = identity.get("open_data_verification")
    if not isinstance(expected_open_data, dict):
        raise ValueError("ADR 0150 open-data proof is absent")
    proof_id = open_data_verification_id(expected_open_data)
    if proof_id != identity.get("open_data_verification_id"):
        raise ValueError("ADR 0150 open-data proof digest differs")

    cache = R3ActionEditMlxCache(
        args.cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    s1_cache = S1ExactSupplyCache(
        args.s1_cache,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    observed_open_data = open_data_verification_identity(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
    )
    if (
        observed_open_data != expected_open_data
        or open_data_verification_id(observed_open_data) != proof_id
    ):
        raise ValueError("failure-atlas open-data identity differs from authorization")
    validation = cache.bind_dataset(
        args.validation_dataset,
        s1_cache=s1_cache,
        verify_dataset_checksums=False,
        preverified_open_data_proof_id=proof_id,
    )
    checkpoint = _latest_checkpoint(args.run_dir)
    checkpoint_identity = origin["checkpoint"]
    if (
        _checksum(checkpoint / "checkpoint.json")
        != checkpoint_identity["manifest_blake3"]
        or _checksum(checkpoint / "model.safetensors")
        != checkpoint_identity["model_blake3"]
    ):
        raise ValueError("failure-atlas checkpoint bytes differ from origin")

    mx.set_default_device(mx.gpu)
    model, _optimizer, state, loaded_checkpoint = load_latest_checkpoint_with_factory(
        args.run_dir,
        learning_rate=1e-4,
        weight_decay=1e-4,
        model_factory=lambda values: R3ActionEditRanker(
            R3ActionEditModelConfig.from_dict(values)
        ),
    )
    if (
        loaded_checkpoint.resolve() != checkpoint.resolve()
        or state.global_step != origin["optimization"]["global_step"]
        or model.config.arm != arm
    ):
        raise ValueError("failure-atlas loaded checkpoint identity differs")
    records = build_failure_atlas(
        model=model,
        dataset=validation,
        arm=str(arm),
        candidate_chunk=args.candidate_chunk,
    )
    if (
        len(records) != 240
        or sum(record["candidate_count"] for record in records) != 860_203
        or any(record["nonfinite_scores"] for record in records)
        or any(record["nonfinite_uncertainties"] for record in records)
    ):
        raise ValueError("failure-atlas validation coverage is incomplete")
    details_blake3 = _write_jsonl_atomic(args.details_jsonl, records)
    aggregates = aggregate_records(records)
    worst = sorted(
        records,
        key=lambda record: (
            not record["confidence_set_covered_top64"],
            record["top64_retained_r4800_regret"],
            record["winner_rank"],
        ),
        reverse=True,
    )[:20]
    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "atlas_kind": ATLAS_KIND,
        "arm": arm,
        "origin_host": origin["host"],
        "analysis_host": _normalize_host(socket.gethostname().split(".")[0]),
        "origin_report_id": origin["report_id"],
        "authorization_id": authorization["authorization_id"],
        "checkpoint": {
            "manifest_blake3": checkpoint_identity["manifest_blake3"],
            "model_blake3": checkpoint_identity["model_blake3"],
            "global_step": int(state.global_step),
        },
        "open_data_verification_id": proof_id,
        "validation_groups": len(records),
        "validation_candidates": sum(record["candidate_count"] for record in records),
        "candidate_chunk": args.candidate_chunk,
        "details_blake3": details_blake3,
        "classifier_eligible": False,
        "scientific_use": "post-classification-diagnostic-only",
    }
    report = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        **scientific_identity,
        "scientific_identity": scientific_identity,
        "aggregates": aggregates,
        "worst_decisions": worst,
    }
    report["report_id"] = _canonical_blake3(report)
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "arm": arm,
                "analysis_host": report["analysis_host"],
                "report_id": report["report_id"],
                "all": aggregates["all"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
