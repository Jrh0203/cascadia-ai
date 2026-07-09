"""Exact full-menu candidate-recall probe for policy-only checkpoint updates.

Unlike training-time filtered tensors, this tool requires unfiltered v3
shards and scores every legal action. Action rows are chunked while the state
encoding is reused, so large late-game menus remain memory bounded. The result
is an offline routing gate, never gameplay or promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .expert_tensor_shards import ExpertTensorCorpus, SHARD_VERSION_V3
from .torch_benchmark_stats import paired_delta_stats
from .torch_inference_bridge import _load_model, resolve_checkpoint_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_checkpoint(manifest: Path, device_name: str):  # type: ignore[no-untyped-def]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    weights = resolve_checkpoint_path(
        payload["weights"],
        manifest_path=manifest,
        checkpoint_path=manifest,
    )
    model = _load_model(
        manifest,
        manifest_path=manifest,
        manifest_payload=payload,
        device_name=device_name,
    )
    model.set_cgab_fused(True)
    model.eval()
    return model, payload, weights


def _full_menu_inputs(example: dict[str, Any], device):  # type: ignore[no-untyped-def]
    import numpy as np
    import torch

    tokens_np = np.array(example["tokens"], dtype=np.float32, copy=True)
    actions_np = np.array(example["actions"], dtype=np.float32, copy=True)
    token_count = int(tokens_np.shape[0])
    action_count = int(actions_np.shape[0])
    relation_tail = torch.zeros(
        (action_count, token_count + action_count),
        dtype=torch.uint8,
    )
    edges = np.asarray(example["relation_edges"], dtype=np.int64)
    if edges.size:
        action_edges = edges[edges[:, 0] >= token_count]
        if action_edges.size:
            rows = torch.from_numpy(action_edges[:, 0] - token_count)
            columns = torch.from_numpy(action_edges[:, 1])
            values = torch.from_numpy(action_edges[:, 2]).to(dtype=torch.uint8)
            relation_tail[rows, columns] = values
    tokens = torch.from_numpy(tokens_np).unsqueeze(0).to(device)
    actions = torch.from_numpy(actions_np).unsqueeze(0).to(device)
    token_mask = torch.ones((1, token_count), dtype=torch.bool, device=device)
    action_mask = torch.ones((1, action_count), dtype=torch.bool, device=device)
    return tokens, token_mask, actions, action_mask, relation_tail.unsqueeze(0).to(device)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finalize_checkpoint(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    confidence = [row for row in rows if row["confidence_qualified"]]
    oracle_regrets = [row["oracle_regret"] for row in rows if row["oracle_regret"] is not None]
    top1_regrets = [row["top1_regret"] for row in rows if row["top1_regret"] is not None]
    return {
        "root_count": len(rows),
        "candidate_top_k": top_k,
        "global_completed_q_best_coverage": _mean(
            [float(row["best_covered"]) for row in rows]
        ),
        "confidence_qualified_root_count": len(confidence),
        "confidence_qualified_best_coverage": _mean(
            [float(row["best_covered"]) for row in confidence]
        ),
        "valid_completed_q_candidate_rate": _mean(
            [float(row["has_valid_q_candidate"]) for row in rows]
        ),
        "mean_candidate_oracle_regret": _mean(oracle_regrets),
        "candidate_oracle_regret_root_count": len(oracle_regrets),
        "top1_accuracy": _mean([float(row["top1_correct"]) for row in rows]),
        "top1_q_valid_rate": _mean([float(row["top1_q_valid"]) for row in rows]),
        "mean_top1_completed_q_regret": _mean(top1_regrets),
        "top1_regret_root_count": len(top1_regrets),
    }


def _paired_numeric(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    key: str,
    *,
    seed: int,
    confidence_only: bool = False,
) -> dict[str, Any]:
    deltas = []
    for left, right in zip(baseline, candidate):
        if confidence_only and not left["confidence_qualified"]:
            continue
        if left[key] is None or right[key] is None:
            continue
        deltas.append(float(right[key]) - float(left[key]))
    return paired_delta_stats(deltas, seed=seed)


def _discordance(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    key: str,
    *,
    confidence_only: bool = False,
) -> dict[str, int]:
    pairs = [
        (bool(left[key]), bool(right[key]))
        for left, right in zip(baseline, candidate)
        if not confidence_only or left["confidence_qualified"]
    ]
    return {
        "candidate_only": sum(right and not left for left, right in pairs),
        "baseline_only": sum(left and not right for left, right in pairs),
        "both": sum(left and right for left, right in pairs),
        "neither": sum(not left and not right for left, right in pairs),
    }


def _comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    return {
        "candidate_minus_baseline_best_coverage": _paired_numeric(
            baseline, candidate, "best_covered", seed=20260740 + seed_offset
        ),
        "candidate_minus_baseline_confidence_best_coverage": _paired_numeric(
            baseline,
            candidate,
            "best_covered",
            seed=20260741 + seed_offset,
            confidence_only=True,
        ),
        "candidate_minus_baseline_candidate_oracle_regret": _paired_numeric(
            baseline, candidate, "oracle_regret", seed=20260742 + seed_offset
        ),
        "candidate_minus_baseline_top1_accuracy": _paired_numeric(
            baseline, candidate, "top1_correct", seed=20260743 + seed_offset
        ),
        "candidate_minus_baseline_top1_regret": _paired_numeric(
            baseline, candidate, "top1_regret", seed=20260744 + seed_offset
        ),
        "best_coverage_discordance": _discordance(baseline, candidate, "best_covered"),
        "confidence_best_coverage_discordance": _discordance(
            baseline,
            candidate,
            "best_covered",
            confidence_only=True,
        ),
        "top1_discordance": _discordance(baseline, candidate, "top1_correct"),
    }


def run_probe(
    *,
    baseline_manifest: Path,
    candidate_manifest: Path,
    tensors: list[Path],
    device_name: str,
    action_chunk_size: int,
    top_k: int,
    max_records: int,
    min_margin: float,
    min_snr: float,
    require_prior_top_k_parity: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if action_chunk_size <= 0 or top_k <= 0:
        raise ValueError("action_chunk_size and top_k must be positive")
    if max_records < 0:
        raise ValueError("max_records must be nonnegative")
    if min_margin < 0.0 or min_snr < 0.0:
        raise ValueError("confidence thresholds must be nonnegative")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    baseline_model, baseline_payload, baseline_weights = _load_checkpoint(
        baseline_manifest, str(device)
    )
    candidate_model, candidate_payload, candidate_weights = _load_checkpoint(
        candidate_manifest, str(device)
    )
    baseline_config = baseline_payload.get("config", {})
    candidate_config = candidate_payload.get("config", {})
    for key in ("token_feature_dim", "action_feature_dim", "d_model", "model_size"):
        if baseline_config.get(key) != candidate_config.get(key):
            raise ValueError(f"checkpoint config mismatch for {key}")

    corpus = ExpertTensorCorpus(tensors)
    try:
        if corpus.schema_ids() != [SHARD_VERSION_V3]:
            raise ValueError("policy candidate probe requires only v3 shards")
        metadata = [shard.metadata for shard in corpus.shards]
        if any("filter" in item or "relation_tail" in item for item in metadata):
            raise ValueError("policy candidate probe requires unfiltered full-menu v3 shards")
        if any(item.get("scientific_eligibility") != "gumbel_selfplay_expert_iteration" for item in metadata):
            raise ValueError("policy candidate probe requires training-eligible v3 shards")
        source_revisions = sorted({str(item["source_revision"]) for item in metadata})
        ruleset_ids = sorted({str(item["ruleset_id"]) for item in metadata})
        if len(source_revisions) != 1 or len(ruleset_ids) != 1:
            raise ValueError("probe shards must share one source revision and ruleset")
        record_count = len(corpus) if max_records == 0 else min(len(corpus), max_records)
        if record_count <= 0:
            raise ValueError("policy candidate probe requires at least one record")

        observations: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
        prior_parity_eligible = 0
        prior_parity_matches = 0
        prior_top_k_overlap_total = 0
        prior_top_k_overlap_min = top_k
        prior_best_coverage_agreements = 0
        prior_mismatches: list[dict[str, Any]] = []
        skipped_without_valid_q = 0
        exact_endgame_roots = 0
        with torch.inference_mode():
            for index in range(record_count):
                example = corpus.example(index)
                tokens, token_mask, actions, action_mask, relation_tail = _full_menu_inputs(
                    example, device
                )
                logits_by_name = {
                    "baseline": baseline_model.policy_logits_chunked(
                        tokens,
                        token_mask,
                        actions,
                        action_mask,
                        relation_tail=relation_tail,
                        action_chunk_size=action_chunk_size,
                    ).squeeze(0).float().cpu(),
                    "candidate": candidate_model.policy_logits_chunked(
                        tokens,
                        token_mask,
                        actions,
                        action_mask,
                        relation_tail=relation_tail,
                        action_chunk_size=action_chunk_size,
                    ).squeeze(0).float().cpu(),
                }
                target_q = torch.as_tensor(np.array(example["target_q"], copy=True), dtype=torch.float32)
                q_valid = torch.as_tensor(np.array(example["q_valid"], copy=True), dtype=torch.bool)
                valid_count = int(q_valid.sum())
                if valid_count == 0:
                    skipped_without_valid_q += 1
                    continue
                exact_endgame = bool(example.get("exact_endgame", False))
                exact_endgame_roots += int(exact_endgame)
                global_best_q, global_best = target_q.masked_fill(~q_valid, -torch.inf).max(dim=0)

                confidence_qualified = False
                if valid_count >= 2:
                    top_values, top_indices = target_q.masked_fill(~q_valid, -torch.inf).topk(2)
                    q_count = torch.as_tensor(np.array(example["q_count"], copy=True), dtype=torch.float32)
                    q_variance = torch.as_tensor(
                        np.array(example["q_variance"], copy=True), dtype=torch.float32
                    )
                    best_count = q_count[top_indices[0]]
                    second_count = q_count[top_indices[1]]
                    margin = top_values[0] - top_values[1]
                    se_sq = (
                        q_variance[top_indices[0]].clamp_min(0.0) / best_count.clamp_min(1.0)
                        + q_variance[top_indices[1]].clamp_min(0.0)
                        / second_count.clamp_min(1.0)
                    )
                    if float(se_sq) > 0.0:
                        snr = float(margin / torch.sqrt(se_sq))
                    else:
                        snr = float("inf") if float(margin) > 0.0 else 0.0
                    confidence_qualified = bool(
                        float(best_count) >= 2.0
                        and float(second_count) >= 2.0
                        and float(margin) >= min_margin
                        and snr >= min_snr
                    )

                action_count = int(target_q.shape[0])
                candidate_count = min(top_k, action_count)
                baseline_top = logits_by_name["baseline"].topk(candidate_count).indices
                if not exact_endgame:
                    prior_parity_eligible += 1
                    priors = torch.as_tensor(np.array(example["priors"], copy=True))
                    prior_top = priors.topk(candidate_count).indices
                    baseline_set = set(baseline_top.tolist())
                    prior_set = set(prior_top.tolist())
                    overlap = len(baseline_set & prior_set)
                    exact_set_match = baseline_set == prior_set
                    prior_parity_matches += int(exact_set_match)
                    prior_top_k_overlap_total += overlap
                    prior_top_k_overlap_min = min(prior_top_k_overlap_min, overlap)
                    coverage_agrees = (int(global_best) in baseline_set) == (
                        int(global_best) in prior_set
                    )
                    prior_best_coverage_agreements += int(coverage_agrees)
                    if not exact_set_match:
                        sorted_priors = priors.topk(min(candidate_count + 1, action_count)).values
                        boundary_gap = (
                            float(sorted_priors[candidate_count - 1] - sorted_priors[candidate_count])
                            if action_count > candidate_count
                            else None
                        )
                        prior_mismatches.append(
                            {
                                "record_index": index,
                                "action_count": action_count,
                                "top_k_overlap_count": overlap,
                                "completed_q_best_coverage_agrees": coverage_agrees,
                                "stored_prior_boundary_gap": boundary_gap,
                            }
                        )

                for name, logits in logits_by_name.items():
                    candidate_indices = logits.topk(candidate_count).indices
                    candidate_mask = torch.zeros(action_count, dtype=torch.bool)
                    candidate_mask[candidate_indices] = True
                    candidate_q_mask = candidate_mask & q_valid
                    has_valid_q_candidate = bool(candidate_q_mask.any())
                    candidate_best_q = target_q.masked_fill(~candidate_q_mask, -torch.inf).max()
                    top1 = int(logits.argmax())
                    top1_q_valid = bool(q_valid[top1])
                    observations[name].append(
                        {
                            "best_covered": bool(candidate_mask[global_best]),
                            "has_valid_q_candidate": has_valid_q_candidate,
                            "oracle_regret": (
                                float((global_best_q - candidate_best_q).clamp_min(0.0))
                                if has_valid_q_candidate
                                else None
                            ),
                            "top1_correct": top1 == int(global_best),
                            "top1_q_valid": top1_q_valid,
                            "top1_regret": (
                                float((global_best_q - target_q[top1]).clamp_min(0.0))
                                if top1_q_valid
                                else None
                            ),
                            "confidence_qualified": confidence_qualified,
                            "exact_endgame": exact_endgame,
                        }
                    )
                del tokens, token_mask, actions, action_mask, relation_tail
    finally:
        corpus.close()

    baseline = observations["baseline"]
    candidate = observations["candidate"]
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("probe produced no aligned valid-Q observations")
    prior_parity_rate = (
        prior_parity_matches / prior_parity_eligible if prior_parity_eligible else None
    )
    if require_prior_top_k_parity and prior_parity_matches != prior_parity_eligible:
        raise ValueError(
            "baseline full-menu top-K does not reproduce stored generator priors: "
            f"{prior_parity_matches}/{prior_parity_eligible}"
        )
    non_exact_baseline = [row for row in baseline if not row["exact_endgame"]]
    non_exact_candidate = [row for row in candidate if not row["exact_endgame"]]
    return {
        "status": "pass",
        "schema_id": "cascadiav3.policy_candidate_probe.v2",
        "scientific_eligibility": "exact_full_menu_offline_policy_routing_only_not_gameplay",
        "action_surface": "unfiltered_full_legal_menu",
        "ruleset_id": ruleset_ids[0],
        "source_revision": source_revisions[0],
        "record_count": record_count,
        "valid_q_root_count": len(baseline),
        "skipped_without_valid_q": skipped_without_valid_q,
        "exact_endgame_root_count": exact_endgame_roots,
        "confidence_gate": {
            "min_samples_per_action": 2,
            "min_margin": min_margin,
            "min_snr": min_snr,
        },
        "candidate_top_k": top_k,
        "action_chunk_size": action_chunk_size,
        "fused_cgab": True,
        "generator_prior_parity": {
            "non_exact_root_count": prior_parity_eligible,
            "exact_top_k_set_match_count": prior_parity_matches,
            "exact_top_k_set_match_rate": prior_parity_rate,
            "mean_top_k_action_overlap_rate": (
                prior_top_k_overlap_total / (prior_parity_eligible * top_k)
                if prior_parity_eligible
                else None
            ),
            "minimum_top_k_action_overlap_rate": (
                prior_top_k_overlap_min / top_k if prior_parity_eligible else None
            ),
            "completed_q_best_coverage_agreement_rate": (
                prior_best_coverage_agreements / prior_parity_eligible
                if prior_parity_eligible
                else None
            ),
            "mismatches": prior_mismatches,
            "required": require_prior_top_k_parity,
        },
        "baseline": _finalize_checkpoint(baseline, top_k=top_k),
        "candidate": _finalize_checkpoint(candidate, top_k=top_k),
        "comparison": _comparison(baseline, candidate, seed_offset=0),
        "non_exact_policy_roots": {
            "baseline": _finalize_checkpoint(non_exact_baseline, top_k=top_k),
            "candidate": _finalize_checkpoint(non_exact_candidate, top_k=top_k),
            "comparison": _comparison(
                non_exact_baseline,
                non_exact_candidate,
                seed_offset=100,
            ),
        },
        "checkpoints": {
            "baseline": {
                "manifest": _artifact(baseline_manifest),
                "weights": _artifact(baseline_weights),
                "checkpoint_tag": baseline_payload.get("checkpoint_tag"),
                "step": baseline_payload.get("step"),
                "config": baseline_config,
            },
            "candidate": {
                "manifest": _artifact(candidate_manifest),
                "weights": _artifact(candidate_weights),
                "checkpoint_tag": candidate_payload.get("checkpoint_tag"),
                "step": candidate_payload.get("step"),
                "config": candidate_config,
            },
        },
        "tensors": [_artifact(path) for path in tensors],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline"]
    candidate = report["candidate"]
    comparison = report["comparison"]
    coverage = comparison["candidate_minus_baseline_best_coverage"]
    confidence_coverage = comparison["candidate_minus_baseline_confidence_best_coverage"]
    regret = comparison["candidate_minus_baseline_candidate_oracle_regret"]
    lines = [
        "# Exact Full-Menu Policy Candidate Probe",
        "",
        f"Records: `{report['record_count']}`",
        f"Valid-Q roots: `{report['valid_q_root_count']}`",
        f"Candidate top-K: `{report['candidate_top_k']}`",
        f"Generator-prior top-K parity: `{report['generator_prior_parity']['exact_top_k_set_match_rate']:.2%}`",
        "",
        "| Metric | Baseline | Candidate | Paired candidate-baseline |",
        "|---|---:|---:|---:|",
        f"| Global completed-Q best coverage | {baseline['global_completed_q_best_coverage']:.2%} | "
        f"{candidate['global_completed_q_best_coverage']:.2%} | {coverage['mean']:+.2%} |",
        f"| Confidence-qualified best coverage | {baseline['confidence_qualified_best_coverage']:.2%} | "
        f"{candidate['confidence_qualified_best_coverage']:.2%} | {confidence_coverage['mean']:+.2%} |",
        f"| Mean candidate oracle regret | {baseline['mean_candidate_oracle_regret']:.4f} | "
        f"{candidate['mean_candidate_oracle_regret']:.4f} | {regret['mean']:+.4f} |",
        f"| Top-1 accuracy | {baseline['top1_accuracy']:.2%} | {candidate['top1_accuracy']:.2%} | "
        f"{comparison['candidate_minus_baseline_top1_accuracy']['mean']:+.2%} |",
        "",
        "Offline routing evidence only; paired gameplay is required for strength.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-margin", type=float, default=0.25)
    parser.add_argument("--min-snr", type=float, default=1.0)
    parser.add_argument("--require-prior-top-k-parity", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    report = run_probe(
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
        tensors=args.tensor,
        device_name=args.device,
        action_chunk_size=args.action_chunk_size,
        top_k=args.top_k,
        max_records=args.max_records,
        min_margin=args.min_margin,
        min_snr=args.min_snr,
        require_prior_top_k_parity=args.require_prior_top_k_parity,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out is not None:
        write_markdown(report, args.summary_out)
    print(
        json.dumps(
            {
                "generator_prior_parity": report["generator_prior_parity"],
                "baseline": report["baseline"],
                "candidate": report["candidate"],
                "comparison": report["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
