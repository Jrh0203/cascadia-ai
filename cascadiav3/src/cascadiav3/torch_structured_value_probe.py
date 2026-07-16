"""Held-out action-conditioned score-decomposition representation probe.

This probe is deliberately upstream of a serving implementation. It asks one
bounded question: does the incumbent's frozen selected-action representation
predict real terminal wildlife/habitat/Nature components better than the
incumbent scalar/value heads on an untouched seed block? A positive result
only authorizes building an exact-grounded action-decomposition schema; it is
not gameplay or promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .expert_tensor_shards import ExpertTensorShard, SHARD_VERSION_V3
from .torch_inference_bridge import _load_model, resolve_checkpoint_path

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
DEFAULT_RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
DEFAULT_MIN_RELATIVE_RMSE_GAIN = 0.10
SCORE_CATEGORIES = ("wildlife", "habitat", "nature_tokens")
ACTIVE_SEAT_FEATURE_ATOL = 1.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _seed_interval(seed_domain: str) -> tuple[int, int]:
    match = re.search(r"first_seed=(\d+),seed_count=(\d+)", seed_domain)
    if match is None:
        raise ValueError(f"cannot parse seed interval from {seed_domain!r}")
    first = int(match.group(1))
    count = int(match.group(2))
    if count <= 0:
        raise ValueError("seed count must be positive")
    return first, first + count


def _validate_blocks(
    manifest: Path,
    manifest_payload: dict[str, Any],
    weights: Path,
    shards: dict[str, ExpertTensorShard],
) -> dict[str, Any]:
    manifest_hash = _sha256(manifest)
    weights_hash = _sha256(weights)
    source_revisions: set[str] = set()
    intervals: dict[str, tuple[int, int]] = {}
    for name, shard in shards.items():
        metadata = shard.metadata
        if shard.version != SHARD_VERSION_V3:
            raise ValueError(f"{name} block is not a v3 tensor shard")
        if metadata.get("ruleset_id") != RULESET_ID:
            raise ValueError(f"{name} block has the wrong ruleset")
        if metadata.get("scientific_eligibility") != "gumbel_selfplay_expert_iteration":
            raise ValueError(f"{name} block is not training-eligible")
        if metadata.get("filter") is not None:
            raise ValueError(f"{name} block must be an exact unfiltered action menu")
        source_revisions.add(str(metadata.get("source_revision", "")))
        intervals[name] = _seed_interval(str(metadata.get("seed_domain", "")))
        teacher = metadata.get("teacher_model", {})
        teacher_manifest = teacher.get("manifest", {})
        teacher_weights = teacher.get("weights", {})
        if teacher_manifest.get("sha256") != manifest_hash:
            raise ValueError(f"{name} teacher manifest does not match the probed checkpoint")
        if teacher_weights.get("sha256") != weights_hash:
            raise ValueError(f"{name} teacher weights do not match the probed checkpoint")
    if len(source_revisions) != 1 or "" in source_revisions:
        raise ValueError("probe blocks must share one exact source revision")
    names = list(intervals)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_start, left_end = intervals[left]
            right_start, right_end = intervals[right]
            if max(left_start, right_start) < min(left_end, right_end):
                raise ValueError(f"seed blocks overlap: {left} and {right}")
    config = manifest_payload.get("config", manifest_payload)
    return {
        "ruleset_id": RULESET_ID,
        "source_revision": next(iter(source_revisions)),
        "checkpoint_tag": manifest_payload.get("checkpoint_tag"),
        "checkpoint_step": manifest_payload.get("step"),
        "model_name": config.get("model_name"),
        "q_quantiles": int(config.get("q_quantiles", 1)),
        "seed_intervals": {name: list(interval) for name, interval in intervals.items()},
    }


def _selected_relation_ids(example: dict[str, Any]) -> np.ndarray:
    token_count = int(example["tokens"].shape[0])
    selected = int(example["selected_action_index"])
    action_count = int(example["actions"].shape[0])
    if not 0 <= selected < action_count:
        raise ValueError("selected action index is outside the action menu")
    source = token_count + selected
    combined_count = token_count + action_count
    relation_ids = np.zeros((combined_count,), dtype=np.uint8)
    seen_targets: set[int] = set()
    for raw_source, raw_target, raw_relation in example["relation_edges"]:
        edge_source = int(raw_source)
        if edge_source != source:
            continue
        target = int(raw_target)
        relation = int(raw_relation)
        if not 0 <= target < combined_count:
            raise ValueError("selected action relation target is outside the combined sequence")
        if target in seen_targets and int(relation_ids[target]) != relation:
            raise ValueError("selected action has conflicting relation ids for one target")
        if not 0 <= relation < 256:
            raise ValueError("relation id does not fit the uint8 serving contract")
        seen_targets.add(target)
        relation_ids[target] = relation
    return relation_ids


def _selected_record(example: dict[str, Any]) -> dict[str, Any]:
    selected = int(example["selected_action_index"])
    actions = example["actions"]
    selected_action = np.asarray(actions[selected], dtype=np.float32)
    seat_position = float(selected_action[0]) * 3.0
    active_seat = int(round(seat_position))
    if not 0 <= active_seat < 4 or abs(seat_position - active_seat) > ACTIVE_SEAT_FEATURE_ATOL:
        raise ValueError("cannot recover active seat from the action feature contract")
    if not np.allclose(
        actions[:, 0], active_seat / 3.0, rtol=0.0, atol=ACTIVE_SEAT_FEATURE_ATOL
    ):
        raise ValueError("action menu contains inconsistent active-seat features")
    categories = np.asarray(example["score_decomposition"][:, active_seat], dtype=np.float32)
    final_score = float(example["final_score_vector"][active_seat])
    if categories.shape != (3,) or not math.isclose(
        float(categories.sum()), final_score, rel_tol=0.0, abs_tol=1.0e-5
    ):
        raise ValueError("active-seat score decomposition does not sum to final score")
    return {
        "tokens": np.asarray(example["tokens"], dtype=np.float32),
        "action_count": int(actions.shape[0]),
        "action": selected_action,
        "relation_ids": _selected_relation_ids(example),
        "active_seat": active_seat,
        "categories": categories,
        "final_score": final_score,
        "exact_afterstate": float(example["exact_afterstate_score_active"][selected]),
        "teacher_q": float(example["target_q"][selected]),
    }


def _extract_block(model: Any, shard: ExpertTensorShard, device: Any, batch_size: int) -> dict[str, Any]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    records: list[dict[str, Any]] = []
    excluded_exact = 0
    for index in range(len(shard)):
        example = shard.example(index)
        if bool(example.get("exact_endgame", False)):
            excluded_exact += 1
            continue
        records.append(_selected_record(example))
    if not records:
        raise ValueError("probe block has no non-exact records")

    features: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            max_tokens = max(record["tokens"].shape[0] for record in chunk)
            max_actions = max(record["action_count"] for record in chunk)
            tokens = torch.zeros(
                (len(chunk), max_tokens, model.config.token_feature_dim), dtype=torch.float32
            )
            token_mask = torch.zeros((len(chunk), max_tokens), dtype=torch.bool)
            actions = torch.zeros(
                (len(chunk), 1, model.config.action_feature_dim), dtype=torch.float32
            )
            action_mask = torch.ones((len(chunk), 1), dtype=torch.bool)
            relation_tail = torch.zeros(
                (len(chunk), 1, max_tokens + max_actions), dtype=torch.uint8
            )
            active_seats = torch.zeros((len(chunk),), dtype=torch.long)
            exact_afterstate = torch.zeros((len(chunk),), dtype=torch.float32)
            teacher_q = torch.zeros((len(chunk),), dtype=torch.float32)
            final_score = torch.zeros((len(chunk),), dtype=torch.float32)
            target_categories = torch.zeros((len(chunk), 3), dtype=torch.float32)
            for row, record in enumerate(chunk):
                token_count = int(record["tokens"].shape[0])
                action_count = int(record["action_count"])
                tokens[row, :token_count] = torch.as_tensor(record["tokens"])
                token_mask[row, :token_count] = True
                actions[row, 0] = torch.as_tensor(record["action"])
                raw_relations = torch.as_tensor(record["relation_ids"])
                relation_tail[row, 0, :token_count] = raw_relations[:token_count]
                relation_tail[row, 0, max_tokens : max_tokens + action_count] = raw_relations[
                    token_count : token_count + action_count
                ]
                active_seats[row] = int(record["active_seat"])
                exact_afterstate[row] = float(record["exact_afterstate"])
                teacher_q[row] = float(record["teacher_q"])
                final_score[row] = float(record["final_score"])
                target_categories[row] = torch.as_tensor(record["categories"])
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            relation_tail = relation_tail.to(device)
            active_seats = active_seats.to(device)
            root_h, decoded, _ = model.encode_action_queries(
                tokens,
                token_mask,
                actions,
                action_mask,
                relation_tail=relation_tail,
            )
            q_raw = model.q_head(decoded[:, 0])
            predicted_score_to_go = (
                q_raw.mean(dim=-1) if model.config.q_quantiles > 1 else q_raw.squeeze(-1)
            )
            rows = torch.arange(len(chunk), device=device)
            root_decomposition = model.score_head(root_h).view(-1, 3, 4)[
                rows, :, active_seats
            ]
            root_value = model.value_head(root_h)[rows, active_seats]
            selected_model_q = exact_afterstate.to(device) + predicted_score_to_go
            features.append(decoded[:, 0].float().cpu().numpy())
            categories.append(target_categories.numpy())
            baselines.append(
                torch.stack(
                    (
                        final_score,
                        root_decomposition.sum(dim=1).float().cpu(),
                        root_value.float().cpu(),
                        selected_model_q.float().cpu(),
                        teacher_q,
                    ),
                    dim=1,
                ).numpy()
            )
    return {
        "features": np.concatenate(features),
        "categories": np.concatenate(categories),
        "baselines": np.concatenate(baselines),
        "records": len(records),
        "excluded_exact_endgame_records": excluded_exact,
    }


def _normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1.0e-6] = 1.0
    return mean, scale


def _design(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = (features.astype(np.float64) - mean) / scale
    return np.concatenate((normalized, np.ones((len(features), 1))), axis=1)


def _fit_ridge(design: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> np.ndarray:
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ValueError("ridge lambda must be finite and positive")
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge_lambda
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ targets.astype(np.float64))


def _error_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("error summary requires a finite non-empty vector")
    return {
        "n": len(error),
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "error_sd": float(error.std(ddof=1)) if len(error) > 1 else 0.0,
    }


def run_probe(
    *,
    manifest: Path,
    fit_tensor: Path,
    selection_tensor: Path,
    validation_tensor: Path,
    device_name: str,
    batch_size: int,
    ridge_lambdas: tuple[float, ...] = DEFAULT_RIDGE_LAMBDAS,
    min_relative_rmse_gain: float = DEFAULT_MIN_RELATIVE_RMSE_GAIN,
) -> dict[str, Any]:
    import torch

    if not ridge_lambdas or any(not math.isfinite(value) or value <= 0 for value in ridge_lambdas):
        raise ValueError("ridge lambda grid must contain only finite positive values")
    if len(set(ridge_lambdas)) != len(ridge_lambdas):
        raise ValueError("ridge lambda grid contains duplicates")
    if not 0.0 <= min_relative_rmse_gain < 1.0:
        raise ValueError("minimum relative RMSE gain must be in [0, 1)")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    weights = resolve_checkpoint_path(
        manifest_payload["weights"], manifest_path=manifest, checkpoint_path=manifest
    )
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = _load_model(
        manifest,
        manifest_path=manifest,
        manifest_payload=manifest_payload,
        device_name=str(device),
    )
    model.eval()
    paths = {
        "fit": fit_tensor,
        "selection": selection_tensor,
        "validation": validation_tensor,
    }
    shards = {name: ExpertTensorShard(path) for name, path in paths.items()}
    try:
        identity = _validate_blocks(manifest, manifest_payload, weights, shards)
        blocks = {
            name: _extract_block(model, shard, device, batch_size)
            for name, shard in shards.items()
        }
    finally:
        for shard in shards.values():
            shard.close()
    if min(block["records"] for block in blocks.values()) < 500:
        raise ValueError("each probe block needs at least 500 non-exact records")

    fit = blocks["fit"]
    selection = blocks["selection"]
    validation = blocks["validation"]
    mean, scale = _normalization(fit["features"])
    fit_design = _design(fit["features"], mean, scale)
    selection_design = _design(selection["features"], mean, scale)
    lambda_results: list[dict[str, float]] = []
    for ridge_lambda in ridge_lambdas:
        coefficients = _fit_ridge(fit_design, fit["categories"], ridge_lambda)
        prediction = selection_design @ coefficients
        lambda_results.append(
            {
                "lambda": ridge_lambda,
                "category_mse": float(np.mean((prediction - selection["categories"]) ** 2)),
                "sum_rmse": float(
                    np.sqrt(
                        np.mean(
                            (prediction.sum(axis=1) - selection["categories"].sum(axis=1)) ** 2
                        )
                    )
                ),
            }
        )
    selected_lambda = min(lambda_results, key=lambda item: item["category_mse"])["lambda"]
    combined_features = np.concatenate((fit["features"], selection["features"]))
    combined_targets = np.concatenate((fit["categories"], selection["categories"]))
    mean, scale = _normalization(combined_features)
    coefficients = _fit_ridge(
        _design(combined_features, mean, scale), combined_targets, selected_lambda
    )
    prediction = _design(validation["features"], mean, scale) @ coefficients
    target_categories = validation["categories"]
    baselines = validation["baselines"]
    heldout = {
        "action_conditioned_category_sum": _error_stats(
            prediction.sum(axis=1), target_categories.sum(axis=1)
        ),
        "root_decomposition_sum": _error_stats(baselines[:, 1], baselines[:, 0]),
        "root_value": _error_stats(baselines[:, 2], baselines[:, 0]),
        "selected_model_q": _error_stats(baselines[:, 3], baselines[:, 0]),
        "selected_teacher_q": _error_stats(baselines[:, 4], baselines[:, 0]),
        "category": {
            category: _error_stats(prediction[:, index], target_categories[:, index])
            for index, category in enumerate(SCORE_CATEGORIES)
        },
    }
    baseline_names = (
        "root_decomposition_sum",
        "root_value",
        "selected_model_q",
        "selected_teacher_q",
    )
    best_baseline_name = min(baseline_names, key=lambda name: heldout[name]["rmse"])
    best_baseline_rmse = float(heldout[best_baseline_name]["rmse"])
    candidate_rmse = float(heldout["action_conditioned_category_sum"]["rmse"])
    relative_gain = 1.0 - candidate_rmse / best_baseline_rmse
    gate_pass = relative_gain >= min_relative_rmse_gain
    return {
        "status": "pass",
        "scientific_eligibility": "offline_representation_preflight_only",
        "identity": identity,
        "device": str(device),
        "execution": {
            "batch_size": batch_size,
            "action_query_mode": "selected_only_with_complete_full_menu_relation_row",
            "cgab_fused": bool(model.cgab.fused),
        },
        "blocks": {
            name: {
                "records": block["records"],
                "excluded_exact_endgame_records": block["excluded_exact_endgame_records"],
            }
            for name, block in blocks.items()
        },
        "ridge": {
            "selection_metric": "category_mse",
            "lambda_results": lambda_results,
            "selected_lambda": selected_lambda,
            "feature_standardization": "fit_only_for_selection_then_fit_plus_selection_for_refit",
        },
        "heldout": heldout,
        "gate": {
            "best_baseline": best_baseline_name,
            "best_baseline_rmse": best_baseline_rmse,
            "candidate_rmse": candidate_rmse,
            "relative_rmse_gain": relative_gain,
            "minimum_relative_rmse_gain": min_relative_rmse_gain,
            "pass": gate_pass,
            "recommendation": (
                "build_exact_grounded_action_decomposition_schema"
                if gate_pass
                else "stop_structured_value_branch"
            ),
        },
        "inputs": {
            "manifest": _artifact(manifest),
            "weights": _artifact(weights),
            "tensors": {name: _artifact(path) for name, path in paths.items()},
        },
        "caveat": (
            "The ridge head predicts direct final categories from one frozen selected-action "
            "embedding. It is not exact-grounded, has no counterfactual category labels, and "
            "cannot be served. Passing only authorizes building the real schema and head."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    heldout = report["heldout"]
    gate = report["gate"]
    lines = [
        "# Action-Conditioned Structured-Value Preflight",
        "",
        f"- Rules: `{report['identity']['ruleset_id']}`",
        f"- Source: `{report['identity']['source_revision']}`",
        f"- Device: `{report['device']}`",
        f"- CGAB fused: `{report['execution']['cgab_fused']}`",
        f"- Non-exact records: fit `{report['blocks']['fit']['records']}`, "
        f"selection `{report['blocks']['selection']['records']}`, "
        f"validation `{report['blocks']['validation']['records']}`",
        f"- Selected ridge lambda: `{report['ridge']['selected_lambda']}`",
        "",
        "| Predictor | Held-out RMSE | MAE | Bias | Error SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "action_conditioned_category_sum",
        "root_decomposition_sum",
        "root_value",
        "selected_model_q",
        "selected_teacher_q",
    ):
        metrics = heldout[name]
        lines.append(
            f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
            f"{metrics['bias']:+.4f} | {metrics['error_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Gate: `{gate['pass']}`; relative RMSE gain "
            f"`{gate['relative_rmse_gain']:.2%}` versus `{gate['best_baseline']}` "
            f"(required `{gate['minimum_relative_rmse_gain']:.2%}`).",
            f"Recommendation: `{gate['recommendation']}`.",
            "",
            report["caveat"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fit-tensor", required=True)
    parser.add_argument("--selection-tensor", required=True)
    parser.add_argument("--validation-tensor", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ridge-lambdas",
        default=",".join(str(value) for value in DEFAULT_RIDGE_LAMBDAS),
    )
    parser.add_argument(
        "--min-relative-rmse-gain", type=float, default=DEFAULT_MIN_RELATIVE_RMSE_GAIN
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    ridge_lambdas = tuple(float(value) for value in args.ridge_lambdas.split(",") if value)
    report = run_probe(
        manifest=Path(args.manifest),
        fit_tensor=Path(args.fit_tensor),
        selection_tensor=Path(args.selection_tensor),
        validation_tensor=Path(args.validation_tensor),
        device_name=args.device,
        batch_size=args.batch_size,
        ridge_lambdas=ridge_lambdas,
        min_relative_rmse_gain=args.min_relative_rmse_gain,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
