"""Validate CascadiaFormer warm-start and exact-resume semantics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .torch_train_cascadiaformer import _checkpoint_member_path, run_training


def _load_weights(manifest_path: Path) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = _checkpoint_member_path(manifest_path, "weights")
    if payload.get("weights_format") == "safetensors" or weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu", weights_only=False)


def _weights_match(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str | None]:  # type: ignore[no-untyped-def]
    import torch

    if set(left) != set(right):
        return False, "state_dict_keys"
    for key in sorted(left):
        if not torch.equal(left[key].cpu(), right[key].cpu()):
            return False, key
    return True, None


def _run(
    *,
    dataset: Path,
    checkpoint_dir: Path,
    out: Path,
    metrics: Path,
    steps: int,
    batch_size: int,
    seed: int,
    model_size: str,
    objective: str,
    init_manifest: Path | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    return run_training(
        [dataset],
        [dataset],
        train_format="npz",
        val_format="npz",
        model_size=model_size,
        steps=steps,
        batch_size=batch_size,
        lr=1.0e-4,
        weight_decay=0.05,
        device_name="cpu",
        seed=seed,
        grad_accum=1,
        warmup_fraction=0.02,
        checkpoint_dir=checkpoint_dir,
        metrics_jsonl=metrics,
        out=out,
        overfit_one_batch=False,
        val_max_batches=1,
        swa_fraction=0.20,
        objective=objective,
        selection_metric="locked_val_total",
        selection_mode="min",
        init_manifest=init_manifest,
        resume=resume,
    )


def _mutated_manifest(original: Path, destination: Path, mismatch: str) -> Path:
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["weights"] = str(_checkpoint_member_path(original, "weights"))
    payload["state"] = str(_checkpoint_member_path(original, "state"))
    identity = dict(payload["resume_identity"])
    if mismatch == "dataset":
        identity["dataset_manifests"] = {"train": [{"path": "different"}], "val": [{"path": "different"}]}
    elif mismatch == "model_size":
        identity["model_size"] = "different"
    elif mismatch == "batch_size":
        identity["batch_size"] = int(identity["batch_size"]) + 1
    elif mismatch == "seed":
        identity["seed"] = int(identity["seed"]) + 1
    elif mismatch == "objective":
        identity["objective"] = "different"
    elif mismatch == "source_hash":
        source_hashes = dict(identity["source_hashes"])
        source_hashes["trainer"] = "different"
        identity["source_hashes"] = source_hashes
    else:
        identity[mismatch] = "different"
    payload["resume_identity"] = identity
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init-manifest")
    parser.add_argument("--steps-before-resume", type=int, default=4)
    parser.add_argument("--steps-after-resume", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "S", "M"])
    parser.add_argument("--objective", default="search-improved-greedy-retention")
    parser.add_argument("--expect-exact-final-weight-match", action="store_true")
    parser.add_argument("--expect-mismatch-refusal", default="")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    if args.steps_before_resume >= args.steps_after_resume:
        parser.error("--steps-before-resume must be less than --steps-after-resume")

    with tempfile.TemporaryDirectory(prefix="cascadiav3-resume-") as tmp_raw:
        tmp = Path(tmp_raw)
        full_dir = tmp / "full"
        partial_dir = tmp / "partial"
        resumed_dir = tmp / "resumed"
        init_dir = tmp / "init"
        full = _run(
            dataset=dataset,
            checkpoint_dir=full_dir,
            out=tmp / "full.json",
            metrics=tmp / "full.metrics.jsonl",
            steps=args.steps_after_resume,
            batch_size=args.batch_size,
            seed=args.seed,
            model_size=args.model_size,
            objective=args.objective,
        )
        partial = _run(
            dataset=dataset,
            checkpoint_dir=partial_dir,
            out=tmp / "partial.json",
            metrics=tmp / "partial.metrics.jsonl",
            steps=args.steps_before_resume,
            batch_size=args.batch_size,
            seed=args.seed,
            model_size=args.model_size,
            objective=args.objective,
        )
        partial_manifest = partial_dir / f"step_{args.steps_before_resume:07d}.manifest.json"
        resumed = _run(
            dataset=dataset,
            checkpoint_dir=resumed_dir,
            out=tmp / "resumed.json",
            metrics=tmp / "resumed.metrics.jsonl",
            steps=args.steps_after_resume,
            batch_size=args.batch_size,
            seed=args.seed,
            model_size=args.model_size,
            objective=args.objective,
            resume=partial_manifest,
        )

        full_final = full_dir / f"step_{args.steps_after_resume:07d}.manifest.json"
        resumed_final = resumed_dir / f"step_{args.steps_after_resume:07d}.manifest.json"
        weights_match, mismatch_key = _weights_match(_load_weights(full_final), _load_weights(resumed_final))
        if args.expect_exact_final_weight_match and not weights_match:
            raise AssertionError(f"resumed weights differ from uninterrupted weights at {mismatch_key}")

        init_report = None
        init_manifest = Path(args.init_manifest) if args.init_manifest else partial["best_val_checkpoint_manifest"]
        if init_manifest:
            init_report = _run(
                dataset=dataset,
                checkpoint_dir=init_dir,
                out=tmp / "init.json",
                metrics=tmp / "init.metrics.jsonl",
                steps=1,
                batch_size=args.batch_size,
                seed=args.seed,
                model_size=args.model_size,
                objective=args.objective,
                init_manifest=Path(init_manifest),
            )

        refused: dict[str, str] = {}
        for mismatch in [item.strip() for item in args.expect_mismatch_refusal.split(",") if item.strip()]:
            mutated = _mutated_manifest(partial_manifest, tmp / f"mutated-{mismatch}.manifest.json", mismatch)
            try:
                _run(
                    dataset=dataset,
                    checkpoint_dir=tmp / f"bad-{mismatch}",
                    out=tmp / f"bad-{mismatch}.json",
                    metrics=tmp / f"bad-{mismatch}.metrics.jsonl",
                    steps=args.steps_after_resume,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    model_size=args.model_size,
                    objective=args.objective,
                    resume=mutated,
                )
            except ValueError as exc:
                refused[mismatch] = str(exc)
            else:
                raise AssertionError(f"resume mismatch {mismatch!r} was not refused")

        report = {
            "status": "pass",
            "dataset": str(dataset),
            "steps_before_resume": args.steps_before_resume,
            "steps_after_resume": args.steps_after_resume,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_size": args.model_size,
            "objective": args.objective,
            "full_report_status": full["status"],
            "partial_report_status": partial["status"],
            "resumed_report_status": resumed["status"],
            "init_report_status": init_report["status"] if init_report else None,
            "exact_final_weight_match": weights_match,
            "mismatch_key": mismatch_key,
            "refused_mismatches": refused,
        }

    # Ensure no temp path survives through report consumers.
    report["temp_artifacts_removed"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
