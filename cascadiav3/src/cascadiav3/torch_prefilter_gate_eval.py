"""Learned source-gating prefilter evaluation.

This is a strict follow-up to the source-union negative result. It trains a
small serving-safe combiner over source scores/ranks using only train-shard
labels, selects by an inner validation split, and evaluates once on the held-out
validation shard.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .torch_prefilter_blend_eval import _collect_rows
from .torch_prefilter_eval import _serving_decision


def source_gate_feature_names(source_names: tuple[str, ...]) -> list[str]:
    names = []
    for source in source_names:
        names.extend(
            [
                f"{source}_zscore",
                f"{source}_rank_score",
                f"{source}_top4",
                f"{source}_top8",
                f"{source}_top16",
            ]
        )
    names.extend(["source_mean", "source_max", "source_min", "source_std", "source_range", "source_top16_votes"])
    return names


def _rank_scores(scores):  # type: ignore[no-untyped-def]
    import torch

    action_count = scores.shape[0]
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    positions = torch.arange(action_count, dtype=torch.float32)
    ranks[order] = positions
    denom = max(1, action_count - 1)
    return 1.0 - ranks / float(denom), ranks


def _row_features(row: dict[str, Any], source_names: tuple[str, ...]):  # type: ignore[no-untyped-def]
    import torch

    source_columns = []
    top16_votes = None
    for source in source_names:
        scores = row["sources"][source].to(torch.float32)
        zscores = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1.0e-6)
        rank_score, ranks = _rank_scores(scores)
        top4 = (ranks < 4).to(torch.float32)
        top8 = (ranks < 8).to(torch.float32)
        top16 = (ranks < 16).to(torch.float32)
        top16_votes = top16 if top16_votes is None else top16_votes + top16
        source_columns.append(torch.stack([zscores, rank_score, top4, top8, top16], dim=1))
    source_block = torch.cat(source_columns, dim=1)
    zscore_columns = [source_block[:, index * 5] for index in range(len(source_names))]
    zstack = torch.stack(zscore_columns, dim=1)
    aggregate = torch.stack(
        [
            zstack.mean(dim=1),
            zstack.max(dim=1).values,
            zstack.min(dim=1).values,
            zstack.std(dim=1, unbiased=False),
            zstack.max(dim=1).values - zstack.min(dim=1).values,
            top16_votes if top16_votes is not None else torch.zeros_like(zstack[:, 0]),
        ],
        dim=1,
    )
    return torch.cat([source_block, aggregate], dim=1)


def _rows_to_tensors(rows: list[dict[str, Any]], source_names: tuple[str, ...], *, k: int):  # type: ignore[no-untyped-def]
    import torch

    if not rows:
        raise ValueError("cannot build tensors from an empty row set")
    features = torch.stack([_row_features(row, source_names) for row in rows], dim=0)
    target_q = torch.stack([row["target_q"].to(torch.float32) for row in rows], dim=0)
    ranked = torch.argsort(target_q, dim=1, descending=True)
    ranks = torch.empty_like(ranked)
    positions = torch.arange(ranked.shape[1])[None, :].expand_as(ranked)
    ranks.scatter_(1, ranked, positions)
    target_topk = (ranks < k).to(torch.float32)
    return features, target_q, target_topk


def _metrics_from_scores(target_q, scores, *, k: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch

    total = int(target_q.shape[0])
    hits = 0
    regret_sum = 0.0
    selected_regret_sum = 0.0
    oracle_q_sum = 0.0
    for row in range(total):
        q = target_q[row]
        pred = scores[row]
        teacher_best = int(torch.argmax(q).item())
        best_q = float(q[teacher_best].item())
        ranked = torch.argsort(pred, descending=True)
        retained = ranked[:k]
        retained_best_q = float(q[retained].max().item())
        hits += int(teacher_best in retained.tolist())
        regret_sum += best_q - retained_best_q
        selected_regret_sum += best_q - float(q[int(ranked[0].item())].item())
        oracle_q_sum += retained_best_q
    return {
        "roots": total,
        "prefilter": {
            str(k): {
                "recall": hits / total if total else 0.0,
                "mean_oracle_regret": regret_sum / total if total else 0.0,
                "mean_oracle_q": oracle_q_sum / total if total else 0.0,
            }
        },
        "mean_selected_regret": selected_regret_sum / total if total else 0.0,
    }


def _source_baseline_metrics(rows: list[dict[str, Any]], source: str, *, k: int) -> dict[str, Any]:
    import torch

    target_q = torch.stack([row["target_q"].to(torch.float32) for row in rows], dim=0)
    scores = torch.stack([row["sources"][source].to(torch.float32) for row in rows], dim=0)
    return _metrics_from_scores(target_q, scores, k=k)


def _selection_key(metrics: dict[str, Any], *, k: int, min_recall: float, max_oracle_regret: float):
    row = metrics["prefilter"][str(k)]
    recall = float(row["recall"])
    regret = float(row["mean_oracle_regret"])
    passes = recall >= min_recall and regret <= max_oracle_regret
    return (int(passes), recall, -regret, -float(metrics["mean_selected_regret"]))


def _gate_loss(scores, target_q, target_topk, *, k: int, pairwise_weight: float):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(scores, target_topk)
    ranked = torch.argsort(target_q, dim=1, descending=True)
    ranks = torch.empty_like(ranked)
    positions = torch.arange(ranked.shape[1], device=ranked.device)[None, :].expand_as(ranked)
    ranks.scatter_(1, ranked, positions)
    positive = ranks < k
    negative = ~positive
    pos_scores = scores[:, :, None]
    neg_scores = scores[:, None, :]
    pair_mask = positive[:, :, None] & negative[:, None, :]
    pair_mask_f = pair_mask.to(scores.dtype)
    q_gap = (target_q[:, :, None] - target_q[:, None, :]).clamp_min(0.0)
    q_gap_scale = (q_gap * pair_mask_f).sum(dim=(1, 2), keepdim=True)
    q_gap_scale = q_gap_scale / pair_mask_f.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    pair_weight = ((q_gap / q_gap_scale.clamp_min(1.0)).clamp(0.0, 3.0) + 0.25) * pair_mask_f
    pairwise = (F.softplus(0.10 - (pos_scores - neg_scores)) * pair_weight).sum()
    pairwise = pairwise / pair_weight.sum().clamp_min(1.0)
    return bce + pairwise_weight * pairwise


def run_gate_eval(
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device_name: str,
    k: int,
    min_recall: float,
    max_oracle_regret: float,
    seed: int,
    steps: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    pairwise_weight: float,
    train_fraction: float,
    eval_interval: int,
    experiment_id: str,
) -> dict[str, Any]:
    import torch
    from torch import nn

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_rows = _collect_rows(train_path, checkpoint, batch_size=batch_size, device=device)
    val_rows = _collect_rows(val_path, checkpoint, batch_size=batch_size, device=device)
    source_names = tuple(train_rows[0]["source_names"]) if train_rows else ()
    feature_names = source_gate_feature_names(source_names)
    features, target_q, target_topk = _rows_to_tensors(train_rows, source_names, k=k)
    heldout_features, heldout_q, _ = _rows_to_tensors(val_rows, source_names, k=k)
    features = features.to(device)
    target_q = target_q.to(device)
    target_topk = target_topk.to(device)
    heldout_features = heldout_features.to(device)
    heldout_q = heldout_q.to(device)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(features.shape[0], generator=generator)
    train_cut = max(1, min(features.shape[0] - 1, int(features.shape[0] * train_fraction)))
    fit_idx = order[:train_cut].to(device)
    tune_idx = order[train_cut:].to(device)

    model = nn.Sequential(
        nn.Linear(features.shape[-1], hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    best_key: tuple[Any, ...] | None = None
    loss_tail: list[float] = []
    for step in range(1, steps + 1):
        batch_positions = torch.randint(0, fit_idx.shape[0], (batch_size,), device=device)
        batch_idx = fit_idx[batch_positions]
        optimizer.zero_grad(set_to_none=True)
        scores = model(features[batch_idx]).squeeze(-1)
        loss = _gate_loss(
            scores,
            target_q[batch_idx],
            target_topk[batch_idx],
            k=k,
            pairwise_weight=pairwise_weight,
        )
        loss.backward()
        optimizer.step()
        loss_tail.append(float(loss.detach().cpu()))
        if len(loss_tail) > 20:
            loss_tail.pop(0)
        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                tune_scores = model(features[tune_idx]).squeeze(-1)
            tune_metrics = _metrics_from_scores(target_q[tune_idx].detach().cpu(), tune_scores.detach().cpu(), k=k)
            key = _selection_key(tune_metrics, k=k, min_recall=min_recall, max_oracle_regret=max_oracle_regret)
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                best_metrics = tune_metrics
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fit_scores = model(features[fit_idx]).squeeze(-1)
        tune_scores = model(features[tune_idx]).squeeze(-1)
        heldout_scores = model(heldout_features).squeeze(-1)
    fit_metrics = _metrics_from_scores(target_q[fit_idx].detach().cpu(), fit_scores.detach().cpu(), k=k)
    tune_metrics = _metrics_from_scores(target_q[tune_idx].detach().cpu(), tune_scores.detach().cpu(), k=k)
    heldout_metrics = _metrics_from_scores(heldout_q.detach().cpu(), heldout_scores.detach().cpu(), k=k)
    return {
        "status": "pass",
        "scientific_eligibility": "dry_run_prefilter_gate_eval",
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_experiment_id": checkpoint["report"].get("experiment_id"),
        "train": str(train_path),
        "val": str(val_path),
        "source_names": list(source_names),
        "feature_names": feature_names,
        "config": {
            "k": k,
            "seed": seed,
            "steps": steps,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "pairwise_weight": pairwise_weight,
            "train_fraction": train_fraction,
            "eval_interval": eval_interval,
            "fit_roots": int(fit_idx.shape[0]),
            "tune_roots": int(tune_idx.shape[0]),
            "heldout_roots": int(heldout_features.shape[0]),
        },
        "training": {
            "best_step": best_step,
            "best_tune_metrics": best_metrics,
            "loss_tail": loss_tail,
        },
        "metrics": {
            "fit": fit_metrics,
            "tune": tune_metrics,
            "heldout": heldout_metrics,
            "heldout_serving_decision": _serving_decision(
                heldout_metrics,
                k_values=[k],
                min_recall=min_recall,
                max_oracle_regret=max_oracle_regret,
            ),
        },
        "heldout_source_baselines": {
            source: _source_baseline_metrics(val_rows, source, k=k)
            for source in source_names
        },
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    k = report["config"]["k"]
    fit = report["metrics"]["fit"]["prefilter"][str(k)]
    tune = report["metrics"]["tune"]["prefilter"][str(k)]
    heldout = report["metrics"]["heldout"]["prefilter"][str(k)]
    lines = [
        "# CRT Wide-32 R16p20 Learned Source-Gate Eval",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## Result",
        "",
        "| Split | K=16 recall | K=16 oracle regret |",
        "|---|---:|---:|",
        f"| Fit | {fit['recall']:.4f} | {fit['mean_oracle_regret']:.4f} |",
        f"| Tune | {tune['recall']:.4f} | {tune['mean_oracle_regret']:.4f} |",
        f"| Held-out validation | {heldout['recall']:.4f} | {heldout['mean_oracle_regret']:.4f} |",
        "",
        "## Baselines On Held-out Validation",
        "",
        "| Source | K=16 recall | K=16 oracle regret |",
        "|---|---:|---:|",
    ]
    for source, metrics in report["heldout_source_baselines"].items():
        row = metrics["prefilter"][str(k)]
        lines.append(f"| {source} | {row['recall']:.4f} | {row['mean_oracle_regret']:.4f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Held-out validation passes: `{report['metrics']['heldout_serving_decision']['passes']}`",
            f"- Best selected step: `{report['training']['best_step']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_train.jsonl")
    parser.add_argument("--val", default="cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl")
    parser.add_argument("--checkpoint", default="cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-oracle-regret", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--experiment-id", default="crt-wide32-r16p20-semantic-learned-source-gate-v1")
    parser.add_argument("--out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_learned_source_gate.json")
    parser.add_argument("--summary-out", default="cascadiav3/reports/crt_wide32_r16p20_semantic_learned_source_gate_summary.md")
    args = parser.parse_args()

    result = run_gate_eval(
        Path(args.train),
        Path(args.val),
        Path(args.checkpoint),
        batch_size=args.batch_size,
        device_name=args.device,
        k=args.k,
        min_recall=args.min_recall,
        max_oracle_regret=args.max_oracle_regret,
        seed=args.seed,
        steps=args.steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pairwise_weight=args.pairwise_weight,
        train_fraction=args.train_fraction,
        eval_interval=args.eval_interval,
        experiment_id=args.experiment_id,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(result, summary_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
