"""Held-out policy probe for pairwise-comparator checkpoints.

This is an offline routing gate, never gameplay evidence. It evaluates only
roots whose top-two completed-Q comparison has at least two samples per action
and clears the configured SNR threshold. Established logits, pairwise Borda,
and their sum are compared on exactly the same v3 roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .expert_tensor_shards import ExpertTensorCorpus, SHARD_VERSION_V3, collate_expert_tensor_examples
from .torch_benchmark_stats import paired_delta_stats
from .torch_inference_bridge import _load_model, resolve_checkpoint_path
from .torch_train_cascadiaformer import _add_pairwise_supervision, _move_to_device


POLICY_MODES = ("logits", "pairwise-borda", "logits-plus-pairwise")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _empty_mode_totals() -> dict[str, float]:
    return {"correct": 0.0, "regret": 0.0, "count": 0.0}


def _finalize_mode(totals: dict[str, float]) -> dict[str, float | int | None]:
    count = int(totals["count"])
    return {
        "root_count": count,
        "top1_accuracy": totals["correct"] / count if count else None,
        "mean_completed_q_regret": totals["regret"] / count if count else None,
    }


def _compare_mode_to_logits(
    baseline: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("paired policy-mode observations must have equal lengths")
    top1_deltas = [right[0] - left[0] for left, right in zip(baseline, candidate)]
    regret_deltas = [right[1] - left[1] for left, right in zip(baseline, candidate)]
    baseline_only = sum(left[0] > right[0] for left, right in zip(baseline, candidate))
    candidate_only = sum(right[0] > left[0] for left, right in zip(baseline, candidate))
    both_correct = sum(left[0] > 0.0 and right[0] > 0.0 for left, right in zip(baseline, candidate))
    both_wrong = len(baseline) - baseline_only - candidate_only - both_correct
    return {
        "root_count": len(baseline),
        "top1_delta_candidate_minus_logits": paired_delta_stats(
            top1_deltas,
            seed=bootstrap_seed,
        ),
        "regret_delta_candidate_minus_logits": paired_delta_stats(
            regret_deltas,
            seed=bootstrap_seed + 1,
        ),
        "top1_discordance": {
            "candidate_only_correct": candidate_only,
            "logits_only_correct": baseline_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
    }


def run_probe(
    *,
    manifest: Path,
    tensors: list[Path],
    device_name: str,
    batch_size: int,
    max_records: int,
    policy_top_k: int = 16,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_records < 0:
        raise ValueError("max_records must be nonnegative")
    if policy_top_k <= 1:
        raise ValueError("policy_top_k must be greater than one")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    config_payload = manifest_payload.get("config", {})
    if not bool(config_payload.get("pairwise_comparator", False)):
        raise ValueError("pairwise policy probe requires a comparator checkpoint")
    weights = resolve_checkpoint_path(
        manifest_payload["weights"],
        manifest_path=manifest,
        checkpoint_path=manifest,
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

    corpus = ExpertTensorCorpus(tensors)
    try:
        if corpus.schema_ids() != [SHARD_VERSION_V3]:
            raise ValueError("pairwise policy probe requires only v3 shards")
        record_count = len(corpus) if max_records == 0 else min(len(corpus), max_records)
        if record_count <= 0:
            raise ValueError("pairwise policy probe requires at least one record")
        metadata = [shard.metadata for shard in corpus.shards]
        source_revisions = sorted({str(item["source_revision"]) for item in metadata})
        ruleset_ids = sorted({str(item["ruleset_id"]) for item in metadata})
        if len(source_revisions) != 1 or len(ruleset_ids) != 1:
            raise ValueError("probe shards must share one source revision and ruleset")
        filter_contracts = [item.get("filter") for item in metadata if item.get("filter")]
        exact_full_menu = not filter_contracts

        mode_totals = {mode: _empty_mode_totals() for mode in POLICY_MODES}
        mode_observations: dict[str, list[tuple[float, float]]] = {
            mode: [] for mode in POLICY_MODES
        }
        pair_count = 0
        pair_correct = 0.0
        pair_weight = 0.0
        pair_weighted_correct = 0.0
        pair_weighted_loss = 0.0
        exact_endgame_roots = 0
        eligible_policy_roots = 0
        global_best_in_candidate_roots = 0
        candidate_coverage_roots = 0

        with torch.inference_mode():
            for start in range(0, record_count, batch_size):
                indices = list(range(start, min(start + batch_size, record_count)))
                host_batch = collate_expert_tensor_examples(corpus.examples(indices))
                _add_pairwise_supervision(host_batch, model.config)
                batch = _move_to_device(host_batch, device)
                outputs = model(
                    batch["tokens"],
                    batch["token_mask"],
                    batch["actions"],
                    batch["action_mask"],
                    relation_ids=batch.get("relation_ids"),
                    relation_tail=batch.get("relation_tail"),
                    pairwise_root_indices=batch["pairwise_root_indices"],
                    pairwise_left_indices=batch["pairwise_left_indices"],
                    pairwise_right_indices=batch["pairwise_right_indices"],
                    return_pairwise_borda=True,
                    pairwise_borda_top_k=policy_top_k,
                )

                pair_logits = outputs["pairwise_logits"].float()
                pair_targets = batch["pairwise_targets"]
                pair_weights = batch["pairwise_weights"]
                if pair_logits.numel():
                    pair_count += int(pair_logits.numel())
                    correct = ((pair_logits >= 0.0) == (pair_targets >= 0.5)).to(torch.float32)
                    pair_correct += float(correct.sum().cpu())
                    pair_weight += float(pair_weights.sum().cpu())
                    pair_weighted_correct += float((correct * pair_weights).sum().cpu())
                    losses = F.binary_cross_entropy_with_logits(
                        pair_logits,
                        pair_targets,
                        reduction="none",
                    )
                    pair_weighted_loss += float((losses * pair_weights).sum().cpu())

                exact = batch.get("exact_endgame")
                if exact is not None:
                    exact_endgame_roots += int(exact.sum().cpu())
                q_valid = batch["q_valid"] & batch["action_mask"]
                candidate_mask = outputs["pairwise_borda_mask"] & q_valid
                target_q = batch["target_q"]
                q_count = batch["target_q_count"]
                q_variance = batch["target_q_variance"]
                valid_counts = candidate_mask.sum(dim=1)
                top_values, top_indices = target_q.masked_fill(
                    ~candidate_mask,
                    -torch.inf,
                ).topk(2, dim=1)
                best = top_indices[:, 0]
                second = top_indices[:, 1]
                rows = torch.arange(target_q.shape[0], device=device)
                roots_with_candidates = valid_counts >= 1
                global_best = target_q.masked_fill(~q_valid, -torch.inf).argmax(dim=1)
                global_best_in_candidate_roots += int(
                    (
                        candidate_mask[rows, global_best]
                        & roots_with_candidates
                    ).sum().cpu()
                )
                candidate_coverage_roots += int(roots_with_candidates.sum().cpu())
                best_count = q_count[rows, best]
                second_count = q_count[rows, second]
                margin = top_values[:, 0] - top_values[:, 1]
                se_sq = (
                    q_variance[rows, best].clamp_min(0.0) / best_count.clamp_min(1.0)
                    + q_variance[rows, second].clamp_min(0.0) / second_count.clamp_min(1.0)
                )
                snr = torch.where(
                    se_sq > 0.0,
                    margin / torch.sqrt(se_sq.clamp_min(torch.finfo(torch.float32).tiny)),
                    torch.where(margin > 0.0, torch.full_like(margin, torch.inf), torch.zeros_like(margin)),
                )
                eligible = (
                    (valid_counts >= 2)
                    & (best_count >= 2.0)
                    & (second_count >= 2.0)
                    & torch.isfinite(margin)
                    & (margin >= float(model.config.pairwise_min_margin))
                    & (snr >= float(model.config.pairwise_min_snr))
                )
                eligible_policy_roots += int(eligible.sum().cpu())
                if not eligible.any():
                    continue
                policy_logits = {
                    "logits": outputs["logits"].float(),
                    "pairwise-borda": outputs["pairwise_borda_logits"].float(),
                    "logits-plus-pairwise": (
                        outputs["logits"] + outputs["pairwise_borda_logits"]
                    ).float(),
                }
                best_q = top_values[:, 0]
                for mode, logits in policy_logits.items():
                    predicted = logits.masked_fill(~candidate_mask, -torch.inf).argmax(dim=1)
                    selected_q = target_q[rows, predicted]
                    correct = ((predicted == best) & eligible).to(torch.float32)
                    regret = (best_q - selected_q).clamp_min(0.0)
                    mode_totals[mode]["correct"] += float(
                        correct.sum().cpu()
                    )
                    mode_totals[mode]["regret"] += float(
                        regret.masked_select(eligible).sum().cpu()
                    )
                    mode_totals[mode]["count"] += float(eligible.sum().cpu())
                    mode_observations[mode].extend(
                        zip(
                            correct.masked_select(eligible).cpu().tolist(),
                            regret.masked_select(eligible).cpu().tolist(),
                        )
                    )
    finally:
        corpus.close()

    if eligible_policy_roots == 0 or pair_count == 0:
        raise ValueError("probe found no confidence-qualified policy roots/pairs")
    return {
        "status": "pass",
        "schema_id": "cascadiav3.pairwise_policy_probe.v4",
        "scientific_eligibility": (
            "exact_full_menu_offline_policy_routing_only_not_gameplay"
            if exact_full_menu
            else "filtered_action_surface_offline_routing_only_not_gameplay"
        ),
        "action_surface": {
            "exact_full_legal_menu": exact_full_menu,
            "filter_contracts": filter_contracts,
            "warning": (
                None
                if exact_full_menu
                else "policy top-K is computed inside the filtered tensor, not the full legal menu"
            ),
        },
        "ruleset_id": ruleset_ids[0],
        "source_revision": source_revisions[0],
        "record_count": record_count,
        "exact_endgame_root_count": exact_endgame_roots,
        "eligible_policy_root_count": eligible_policy_roots,
        "confidence_gate": {
            "min_samples_per_action": 2,
            "min_margin": float(model.config.pairwise_min_margin),
            "min_snr": float(model.config.pairwise_min_snr),
            "max_pairs_per_root": int(model.config.pairwise_max_pairs_per_root),
            "incumbent_policy_top_k": policy_top_k,
        },
        "candidate_set": {
            "root_count": candidate_coverage_roots,
            "global_completed_q_best_covered_count": global_best_in_candidate_roots,
            "global_completed_q_best_coverage": (
                global_best_in_candidate_roots / candidate_coverage_roots
                if candidate_coverage_roots
                else None
            ),
        },
        "pairwise": {
            "directed_pair_count": pair_count,
            "accuracy": pair_correct / pair_count,
            "weighted_accuracy": pair_weighted_correct / pair_weight,
            "weighted_binary_cross_entropy": pair_weighted_loss / pair_weight,
        },
        "policy_modes": {mode: _finalize_mode(totals) for mode, totals in mode_totals.items()},
        "comparisons_vs_logits": {
            mode: _compare_mode_to_logits(
                mode_observations["logits"],
                mode_observations[mode],
                bootstrap_seed=20260709 + index * 10,
            )
            for index, mode in enumerate(POLICY_MODES[1:], start=1)
        },
        "checkpoint": {
            "manifest": _artifact(manifest),
            "weights": _artifact(weights),
            "checkpoint_tag": manifest_payload.get("checkpoint_tag"),
            "step": manifest_payload.get("step"),
            "config": config_payload,
        },
        "tensors": [_artifact(path) for path in tensors],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pairwise Policy Probe",
        "",
        f"Records: `{report['record_count']}`",
        f"Exact full legal action surface: `{report['action_surface']['exact_full_legal_menu']}`",
        f"Eligible policy roots: `{report['eligible_policy_root_count']}`",
        f"Incumbent policy candidate cap: `{report['confidence_gate']['incumbent_policy_top_k']}`",
        "Global completed-Q best covered by candidate set: "
        f"`{report['candidate_set']['global_completed_q_best_coverage']:.2%}`",
        f"Directed pairs: `{report['pairwise']['directed_pair_count']}`",
        f"Pairwise accuracy: `{report['pairwise']['accuracy']:.2%}`",
        f"Weighted pairwise accuracy: `{report['pairwise']['weighted_accuracy']:.2%}`",
        "",
        "| Policy mode | Top-1 | Mean completed-Q regret |",
        "|---|---:|---:|",
    ]
    for mode in POLICY_MODES:
        metrics = report["policy_modes"][mode]
        accuracy = metrics["top1_accuracy"]
        regret = metrics["mean_completed_q_regret"]
        lines.append(
            f"| {mode} | {accuracy:.2%} | {regret:.4f} |"
            if accuracy is not None and regret is not None
            else f"| {mode} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "| Candidate vs logits | Top-1 delta | 95% bootstrap CI | Regret delta | 95% bootstrap CI | Candidate-only / logits-only correct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in POLICY_MODES[1:]:
        comparison = report["comparisons_vs_logits"][mode]
        top1 = comparison["top1_delta_candidate_minus_logits"]
        regret = comparison["regret_delta_candidate_minus_logits"]
        discordance = comparison["top1_discordance"]
        lines.append(
            f"| {mode} | {top1['mean']:+.2%} | "
            f"[{top1['bootstrap_ci_low']:+.2%}, {top1['bootstrap_ci_high']:+.2%}] | "
            f"{regret['mean']:+.4f} | "
            f"[{regret['bootstrap_ci_low']:+.4f}, {regret['bootstrap_ci_high']:+.4f}] | "
            f"{discordance['candidate_only_correct']} / {discordance['logits_only_correct']} |"
        )
    lines.extend(["", "Offline routing evidence only; this is not gameplay or promotion evidence."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--policy-top-k", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    report = run_probe(
        manifest=args.manifest,
        tensors=args.tensor,
        device_name=args.device,
        batch_size=args.batch_size,
        max_records=args.max_records,
        policy_top_k=args.policy_top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_out is not None:
        write_markdown(report, args.summary_out)
    print(
        json.dumps(
            {
                "pairwise": report["pairwise"],
                "policy_modes": report["policy_modes"],
                "comparisons_vs_logits": report["comparisons_vs_logits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
