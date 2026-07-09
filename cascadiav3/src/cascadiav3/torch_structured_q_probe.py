"""Held-out gate for an exact-grounded structured-Q checkpoint.

The probe compares a structured candidate with its frozen incumbent on v4
roots. Category metrics use only the selected real trajectory; completed-Q
metrics use every q-valid retained action. Exact-endgame rows are excluded
from the primary gate because their zero residual would make the result look
artificially easy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .expert_tensor_shards import SHARD_VERSION_V4, ExpertTensorCorpus, collate_expert_tensor_examples
from .torch_benchmark_stats import paired_delta_stats
from .torch_inference_bridge import _config_from_payload, _load_model, resolve_checkpoint_path

CATEGORIES = ("wildlife", "habitat", "nature_tokens")
MIN_SELECTED_RMSE_IMPROVEMENT = 0.10
MAX_ALL_Q_RMSE_RATIO = 1.05
MAX_Q_REGRET_INCREASE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(manifest: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    weights = resolve_checkpoint_path(
        str(payload["weights"]),
        manifest_path=manifest,
        checkpoint_path=manifest,
    )
    return payload, weights, {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "manifest_bytes": manifest.stat().st_size,
        "weights": str(weights),
        "weights_sha256": _sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "checkpoint_tag": payload.get("checkpoint_tag"),
        "step": payload.get("step"),
        "config": payload.get("config", {}),
    }


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("error summary requires finite observations")
    return {
        "n": int(errors.size),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "error_sd": float(np.std(errors)),
    }


def summarize_structured_q_observations(
    *,
    target_components: np.ndarray,
    candidate_components: np.ndarray,
    real_final: np.ndarray,
    candidate_selected: np.ndarray,
    incumbent_selected: np.ndarray,
    teacher_selected: np.ndarray,
    candidate_all_q_errors: np.ndarray,
    incumbent_all_q_errors: np.ndarray,
    candidate_q_regret: np.ndarray,
    incumbent_q_regret: np.ndarray,
    candidate_q_top1: np.ndarray,
    incumbent_q_top1: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    target_components = np.asarray(target_components, dtype=np.float64)
    candidate_components = np.asarray(candidate_components, dtype=np.float64)
    if target_components.shape != candidate_components.shape or target_components.ndim != 2:
        raise ValueError("component targets/predictions must share [records, categories] shape")
    if target_components.shape[1] != len(CATEGORIES):
        raise ValueError("structured-Q probe requires three score categories")
    arrays = {
        "real_final": np.asarray(real_final, dtype=np.float64),
        "candidate_selected": np.asarray(candidate_selected, dtype=np.float64),
        "incumbent_selected": np.asarray(incumbent_selected, dtype=np.float64),
        "teacher_selected": np.asarray(teacher_selected, dtype=np.float64),
        "candidate_all_q_errors": np.asarray(candidate_all_q_errors, dtype=np.float64),
        "incumbent_all_q_errors": np.asarray(incumbent_all_q_errors, dtype=np.float64),
        "candidate_q_regret": np.asarray(candidate_q_regret, dtype=np.float64),
        "incumbent_q_regret": np.asarray(incumbent_q_regret, dtype=np.float64),
        "candidate_q_top1": np.asarray(candidate_q_top1, dtype=np.float64),
        "incumbent_q_top1": np.asarray(incumbent_q_top1, dtype=np.float64),
    }
    record_count = target_components.shape[0]
    for name in (
        "real_final",
        "candidate_selected",
        "incumbent_selected",
        "teacher_selected",
        "candidate_q_regret",
        "incumbent_q_regret",
        "candidate_q_top1",
        "incumbent_q_top1",
    ):
        if arrays[name].shape != (record_count,):
            raise ValueError(f"{name} must contain one value per root")
    if record_count < 2:
        raise ValueError("structured-Q verdict requires at least two non-exact roots")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("structured-Q observations must be finite")
    if arrays["candidate_all_q_errors"].shape != arrays["incumbent_all_q_errors"].shape:
        raise ValueError("candidate/incumbent completed-Q observations must align")
    if not np.allclose(target_components.sum(axis=1), arrays["real_final"], rtol=0.0, atol=1e-4):
        raise ValueError("target components do not sum to real final score")
    if not np.allclose(
        candidate_components.sum(axis=1),
        arrays["candidate_selected"],
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("candidate components do not sum to selected final Q")
    if (arrays["candidate_q_regret"] < -1e-6).any() or (
        arrays["incumbent_q_regret"] < -1e-6
    ).any():
        raise ValueError("completed-Q regret must be nonnegative")
    if not all(
        np.isin(arrays[name], (0.0, 1.0)).all()
        for name in ("candidate_q_top1", "incumbent_q_top1")
    ):
        raise ValueError("completed-Q top1 observations must be binary")

    component_error = candidate_components - target_components
    candidate_error = arrays["candidate_selected"] - arrays["real_final"]
    incumbent_error = arrays["incumbent_selected"] - arrays["real_final"]
    teacher_error = arrays["teacher_selected"] - arrays["real_final"]
    candidate_summary = _error_summary(candidate_error)
    incumbent_summary = _error_summary(incumbent_error)
    teacher_summary = _error_summary(teacher_error)
    best_baseline_name, best_baseline = min(
        (("incumbent_model_q", incumbent_summary), ("selected_teacher_q", teacher_summary)),
        key=lambda item: item[1]["rmse"],
    )
    improvement = 1.0 - candidate_summary["rmse"] / best_baseline["rmse"]
    best_baseline_error = (
        incumbent_error if best_baseline_name == "incumbent_model_q" else teacher_error
    )
    paired_abs_delta = np.abs(candidate_error) - np.abs(best_baseline_error)
    paired = paired_delta_stats(paired_abs_delta.tolist(), seed=seed)

    candidate_all_q = _error_summary(arrays["candidate_all_q_errors"])
    incumbent_all_q = _error_summary(arrays["incumbent_all_q_errors"])
    if incumbent_all_q["rmse"] <= 0.0:
        all_q_ratio = 1.0 if candidate_all_q["rmse"] <= 0.0 else float("inf")
    else:
        all_q_ratio = candidate_all_q["rmse"] / incumbent_all_q["rmse"]
    candidate_regret = float(np.mean(arrays["candidate_q_regret"]))
    incumbent_regret = float(np.mean(arrays["incumbent_q_regret"]))
    gates = {
        "selected_rmse_improvement_at_least_10pct": improvement
        >= MIN_SELECTED_RMSE_IMPROVEMENT,
        "paired_absolute_error_ci_excludes_zero": float(paired["t_ci_high"]) < 0.0,
        "all_q_rmse_within_5pct": all_q_ratio <= MAX_ALL_Q_RMSE_RATIO,
        "q_regret_increase_at_most_0_05": candidate_regret - incumbent_regret
        <= MAX_Q_REGRET_INCREASE,
    }
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "thresholds": {
            "minimum_selected_rmse_improvement": MIN_SELECTED_RMSE_IMPROVEMENT,
            "maximum_all_q_rmse_ratio": MAX_ALL_Q_RMSE_RATIO,
            "maximum_q_regret_increase": MAX_Q_REGRET_INCREASE,
        },
        "gates": gates,
        "selected_real_outcome": {
            "candidate": candidate_summary,
            "incumbent_model_q": incumbent_summary,
            "selected_teacher_q": teacher_summary,
            "best_baseline": best_baseline_name,
            "candidate_rmse_improvement_fraction": improvement,
            "candidate_minus_best_baseline_absolute_error": paired,
        },
        "components": {
            category: _error_summary(component_error[:, index])
            for index, category in enumerate(CATEGORIES)
        }
        | {"overall": _error_summary(component_error.reshape(-1))},
        "completed_q": {
            "candidate": candidate_all_q,
            "incumbent": incumbent_all_q,
            "candidate_to_incumbent_rmse_ratio": all_q_ratio,
            "candidate_mean_regret": candidate_regret,
            "incumbent_mean_regret": incumbent_regret,
            "candidate_top1": float(np.mean(arrays["candidate_q_top1"])),
            "incumbent_top1": float(np.mean(arrays["incumbent_q_top1"])),
        },
    }


def _validate_contract(
    *,
    candidate_payload: dict[str, Any],
    incumbent_artifact: dict[str, Any],
    shards: ExpertTensorCorpus,
) -> dict[str, Any]:
    candidate_config = candidate_payload.get("config", {})
    if not bool(candidate_config.get("q_decomposition", False)):
        raise ValueError("candidate checkpoint does not enable q_decomposition")
    if bool(incumbent_artifact["config"].get("q_decomposition", False)):
        raise ValueError("incumbent checkpoint must use the legacy monolithic Q head")
    normalized_candidate = _config_from_payload(candidate_payload).to_dict()
    normalized_incumbent = _config_from_payload(
        {"config": incumbent_artifact["config"]}
    ).to_dict()
    ignored_config_fields = {"model_name", "q_decomposition", "q_quantiles"}
    trunk_mismatches = sorted(
        key
        for key in normalized_candidate
        if key not in ignored_config_fields
        and normalized_candidate[key] != normalized_incumbent[key]
    )
    if trunk_mismatches:
        raise ValueError(f"candidate/incumbent trunk config mismatch: {trunk_mismatches}")
    source_revisions: set[str] = set()
    rulesets: set[str] = set()
    search_contracts: set[str] = set()
    action_surfaces: list[dict[str, Any] | None] = []
    for shard in shards.shards:
        if shard.version != SHARD_VERSION_V4:
            raise ValueError(f"structured-Q probe requires v4 shards: {shard.path}")
        metadata = shard.metadata
        if metadata.get("scientific_eligibility") != "gumbel_selfplay_expert_iteration":
            raise ValueError(f"structured-Q probe shard is not training eligible: {shard.path}")
        teacher = metadata.get("teacher_model", {})
        if teacher.get("manifest", {}).get("sha256") != incumbent_artifact["manifest_sha256"]:
            raise ValueError(f"shard teacher manifest does not match incumbent: {shard.path}")
        if teacher.get("weights", {}).get("sha256") != incumbent_artifact["weights_sha256"]:
            raise ValueError(f"shard teacher weights do not match incumbent: {shard.path}")
        source_revisions.add(str(metadata.get("source_revision", "")))
        rulesets.add(str(metadata.get("ruleset_id", "")))
        search_contracts.add(json.dumps(metadata.get("search"), sort_keys=True))
        action_surfaces.append(metadata.get("filter"))
    if len(source_revisions) != 1 or "" in source_revisions:
        raise ValueError("probe shards must share one non-empty source revision")
    if len(rulesets) != 1 or "" in rulesets:
        raise ValueError("probe shards must share one non-empty ruleset")
    if len(search_contracts) != 1 or "null" in search_contracts:
        raise ValueError("probe shards must share one complete search contract")
    return {
        "source_revision": next(iter(source_revisions)),
        "ruleset_id": next(iter(rulesets)),
        "action_surfaces": action_surfaces,
    }


def run_probe(
    *,
    candidate_manifest: Path,
    incumbent_manifest: Path,
    shard_paths: list[Path],
    device_name: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidate_payload, _, candidate_artifact = _artifact(candidate_manifest)
    incumbent_payload, _, incumbent_artifact = _artifact(incumbent_manifest)
    corpus = ExpertTensorCorpus(shard_paths)
    try:
        contract = _validate_contract(
            candidate_payload=candidate_payload,
            incumbent_artifact=incumbent_artifact,
            shards=corpus,
        )
        device = torch.device(
            device_name
            if device_name != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        candidate = _load_model(
            candidate_manifest,
            manifest_path=candidate_manifest,
            manifest_payload=candidate_payload,
            device_name=str(device),
        )
        incumbent = _load_model(
            incumbent_manifest,
            manifest_path=incumbent_manifest,
            manifest_payload=incumbent_payload,
            device_name=str(device),
        )
        target_components: list[np.ndarray] = []
        candidate_components: list[np.ndarray] = []
        real_final: list[np.ndarray] = []
        candidate_selected: list[np.ndarray] = []
        incumbent_selected: list[np.ndarray] = []
        teacher_selected: list[np.ndarray] = []
        candidate_all_q_errors: list[np.ndarray] = []
        incumbent_all_q_errors: list[np.ndarray] = []
        candidate_q_regret: list[float] = []
        incumbent_q_regret: list[float] = []
        candidate_q_top1: list[float] = []
        incumbent_q_top1: list[float] = []
        exact_rows = 0

        with torch.inference_mode():
            for start in range(0, len(corpus), batch_size):
                examples = corpus.examples(list(range(start, min(len(corpus), start + batch_size))))
                batch = collate_expert_tensor_examples(examples)
                tensor_batch = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in batch.items()
                }
                args = (
                    tensor_batch["tokens"],
                    tensor_batch["token_mask"],
                    tensor_batch["actions"],
                    tensor_batch["action_mask"],
                )
                kwargs = {
                    "relation_ids": tensor_batch.get("relation_ids"),
                    "relation_tail": tensor_batch.get("relation_tail"),
                }
                candidate_out = candidate(*args, **kwargs)
                incumbent_out = incumbent(*args, **kwargs)
                if "q_score_to_go_components" not in candidate_out:
                    raise ValueError("candidate model did not emit structured Q components")
                rows = torch.arange(len(examples), device=device)
                selected = tensor_batch["selected_action_index"]
                active = tensor_batch["active_seat"]
                non_exact = ~tensor_batch["exact_endgame"].to(torch.bool)
                exact_rows += int((~non_exact).sum().item())
                if not non_exact.any():
                    continue
                rows_kept = rows[non_exact]
                selected_kept = selected[non_exact]
                active_kept = active[non_exact]
                target_component = tensor_batch["target_score"][
                    rows_kept, :, active_kept
                ]
                after_component = tensor_batch[
                    "exact_afterstate_score_decomposition_active"
                ][rows_kept, selected_kept]
                predicted_component = after_component + candidate_out[
                    "q_score_to_go_components"
                ][rows_kept, selected_kept]
                exact_scalar = tensor_batch["exact_afterstate_score_active"]
                candidate_final_q = exact_scalar + candidate_out["q"]
                incumbent_final_q = exact_scalar + incumbent_out["q"]
                target_q = tensor_batch["target_q"]
                q_valid = tensor_batch["q_valid"] & tensor_batch["action_mask"]
                final = tensor_batch["target_value"][rows_kept, active_kept]

                target_components.append(target_component.cpu().numpy())
                candidate_components.append(predicted_component.cpu().numpy())
                real_final.append(final.cpu().numpy())
                candidate_selected.append(
                    candidate_final_q[rows_kept, selected_kept].cpu().numpy()
                )
                incumbent_selected.append(
                    incumbent_final_q[rows_kept, selected_kept].cpu().numpy()
                )
                teacher_selected.append(target_q[rows_kept, selected_kept].cpu().numpy())

                for row in rows_kept.tolist():
                    valid = q_valid[row]
                    candidate_values = candidate_final_q[row, valid]
                    incumbent_values = incumbent_final_q[row, valid]
                    targets = target_q[row, valid]
                    if targets.numel() == 0:
                        continue
                    candidate_all_q_errors.append((candidate_values - targets).cpu().numpy())
                    incumbent_all_q_errors.append((incumbent_values - targets).cpu().numpy())
                    best = int(targets.argmax().item())
                    candidate_choice = int(candidate_values.argmax().item())
                    incumbent_choice = int(incumbent_values.argmax().item())
                    candidate_q_regret.append(float((targets[best] - targets[candidate_choice]).item()))
                    incumbent_q_regret.append(float((targets[best] - targets[incumbent_choice]).item()))
                    candidate_q_top1.append(float(candidate_choice == best))
                    incumbent_q_top1.append(float(incumbent_choice == best))

        observations = summarize_structured_q_observations(
            target_components=np.concatenate(target_components),
            candidate_components=np.concatenate(candidate_components),
            real_final=np.concatenate(real_final),
            candidate_selected=np.concatenate(candidate_selected),
            incumbent_selected=np.concatenate(incumbent_selected),
            teacher_selected=np.concatenate(teacher_selected),
            candidate_all_q_errors=np.concatenate(candidate_all_q_errors),
            incumbent_all_q_errors=np.concatenate(incumbent_all_q_errors),
            candidate_q_regret=np.asarray(candidate_q_regret),
            incumbent_q_regret=np.asarray(incumbent_q_regret),
            candidate_q_top1=np.asarray(candidate_q_top1),
            incumbent_q_top1=np.asarray(incumbent_q_top1),
            seed=seed,
        )
        return {
            **observations,
            "probe": "exact_grounded_structured_q_v1",
            "device": str(device),
            "batch_size": batch_size,
            "record_count": len(corpus),
            "non_exact_record_count": len(corpus) - exact_rows,
            "exact_record_count_excluded": exact_rows,
            "contract": contract,
            "candidate": candidate_artifact,
            "incumbent": incumbent_artifact,
            "shards": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in shard_paths
            ],
        }
    finally:
        corpus.close()


def _markdown(report: dict[str, Any]) -> str:
    selected = report["selected_real_outcome"]
    completed = report["completed_q"]
    lines = [
        "# Exact-grounded structured-Q held-out gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Non-exact roots: {report['non_exact_record_count']}",
        f"- Candidate selected-final RMSE: {selected['candidate']['rmse']:.4f}",
        f"- Best baseline ({selected['best_baseline']}) RMSE: "
        f"{selected[selected['best_baseline']]['rmse']:.4f}",
        f"- Relative RMSE improvement: {selected['candidate_rmse_improvement_fraction']:.2%}",
        f"- All-Q RMSE ratio candidate/incumbent: "
        f"{completed['candidate_to_incumbent_rmse_ratio']:.4f}",
        f"- Q regret candidate/incumbent: {completed['candidate_mean_regret']:.4f} / "
        f"{completed['incumbent_mean_regret']:.4f}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report["gates"].items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--incumbent-manifest", type=Path, required=True)
    parser.add_argument("--shards", required=True, help="Comma-separated v4 NPZ shards")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = run_probe(
        candidate_manifest=args.candidate_manifest,
        incumbent_manifest=args.incumbent_manifest,
        shard_paths=[Path(part) for part in args.shards.split(",") if part],
        device_name=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
