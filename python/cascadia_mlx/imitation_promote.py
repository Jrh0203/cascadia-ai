"""Promote a test-qualified full-legal imitation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import blake3

from cascadia_mlx.imitation_model import ImitationModelConfig, SharedStateActionRanker
from cascadia_mlx.promote import PROMOTION_SCHEMA_VERSION, PromotionError


def promote_imitation(run_dir: str | Path, output: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise PromotionError(f"promotion target already exists: {output}")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        final_report = json.loads((run_dir / "final-report.json").read_text())
        best = json.loads((run_dir / "best.json").read_text())
        test_report = json.loads((run_dir / "test-report.json").read_text())
        checkpoint = run_dir / "checkpoints" / best["checkpoint"]
        checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise PromotionError(f"imitation run is not promotion-ready: {error}") from error
    if run.get("kind") != "canonical-action-imitation":
        raise PromotionError("run is not a canonical action-imitation experiment")
    if not test_report.get("passed"):
        raise PromotionError("untouched test imitation gates did not pass")
    if test_report.get("checkpoint") != best["checkpoint"]:
        raise PromotionError("test report does not evaluate the selected best checkpoint")
    initial_loss = float(final_report["initial_validation"]["selection_loss"])
    best_loss = float(best["selection_loss"])
    if not best_loss < initial_loss:
        raise PromotionError("best validation selection loss did not improve over initialization")

    model_source = checkpoint / "model.safetensors"
    expected = checkpoint_manifest["files"]["model.safetensors"]
    if (
        model_source.stat().st_size != expected["bytes"]
        or _checksum(model_source) != expected["blake3"]
    ):
        raise PromotionError("best imitation checkpoint failed integrity validation")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        model_target = temporary / "model.safetensors"
        shutil.copyfile(model_source, model_target)
        artifact = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "status": "promoted",
            "kind": "canonical-action-imitation",
            "model_config": checkpoint_manifest["model_config"],
            "model": {
                "file": model_target.name,
                "bytes": model_target.stat().st_size,
                "blake3": _checksum(model_target),
            },
            "source_run": str(run_dir),
            "source_checkpoint": checkpoint.name,
            "best_ranking_loss": best_loss,
            "validation": best["validation"],
            "initial_validation": final_report["initial_validation"],
            "test": test_report,
            "run_manifest": run,
        }
        (temporary / "model.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_promoted_imitation_model(
    model_dir: str | Path,
) -> SharedStateActionRanker:
    model_dir = Path(model_dir)
    try:
        manifest = json.loads((model_dir / "model.json").read_text())
        model_info = manifest["model"]
        model_path = model_dir / model_info["file"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot read promoted imitation model: {error}") from error
    if (
        manifest.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or manifest.get("status") != "promoted"
        or manifest.get("kind") != "canonical-action-imitation"
    ):
        raise PromotionError("unsupported or incomplete promoted imitation model")
    try:
        if (
            model_path.stat().st_size != model_info["bytes"]
            or _checksum(model_path) != model_info["blake3"]
        ):
            raise PromotionError("promoted imitation model failed integrity validation")
    except OSError as error:
        raise PromotionError(f"cannot read promoted imitation weights: {error}") from error
    model = SharedStateActionRanker(ImitationModelConfig.from_dict(manifest["model_config"]))
    model.load_weights(str(model_path))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(promote_imitation(args.run_dir, args.output))


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
