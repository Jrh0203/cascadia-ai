"""Atomically promote the best validated checkpoint into a model artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import blake3

from cascadia_mlx.model import EntitySetValueModel, ModelConfig

PROMOTION_SCHEMA_VERSION = 1


class PromotionError(ValueError):
    """Raised when a run is incomplete or cannot be promoted honestly."""


def promote(run_dir: str | Path, output: str | Path) -> Path:
    """Package the run's best checkpoint with complete provenance."""
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise PromotionError(f"promotion target already exists: {output}")
    try:
        run = json.loads((run_dir / "run.json").read_text())
        final_report = json.loads((run_dir / "final-report.json").read_text())
        best = json.loads((run_dir / "best.json").read_text())
        checkpoint = run_dir / "checkpoints" / best["checkpoint"]
        checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise PromotionError(f"run is not promotion-ready: {error}") from error
    if final_report.get("best_validation_mae") is None or "validation" not in best:
        raise PromotionError("run has no validation result")
    model_source = checkpoint / "model.safetensors"
    expected = checkpoint_manifest["files"]["model.safetensors"]
    if (
        model_source.stat().st_size != expected["bytes"]
        or _checksum(model_source) != expected["blake3"]
    ):
        raise PromotionError("best checkpoint model failed integrity validation")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        model_target = temporary / "model.safetensors"
        shutil.copyfile(model_source, model_target)
        artifact = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "status": "promoted",
            "model_config": checkpoint_manifest["model_config"],
            "model": {
                "file": model_target.name,
                "bytes": model_target.stat().st_size,
                "blake3": _checksum(model_target),
            },
            "source_run": str(run_dir),
            "source_checkpoint": checkpoint.name,
            "best_validation_mae": best["validation"]["total_mae"],
            "validation": best["validation"],
            "final_validation": final_report["validation"],
            "run_manifest": run,
        }
        (temporary / "model.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_promoted_model(model_dir: str | Path) -> EntitySetValueModel:
    """Verify and load a standalone promoted model artifact."""
    model_dir = Path(model_dir)
    try:
        manifest = json.loads((model_dir / "model.json").read_text())
        model_info = manifest["model"]
        model_path = model_dir / model_info["file"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot read promoted model: {error}") from error
    if (
        manifest.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or manifest.get("status") != "promoted"
    ):
        raise PromotionError("unsupported or incomplete promoted model")
    try:
        if (
            model_path.stat().st_size != model_info["bytes"]
            or _checksum(model_path) != model_info["blake3"]
        ):
            raise PromotionError("promoted model failed integrity validation")
    except OSError as error:
        raise PromotionError(f"cannot read promoted weights: {error}") from error
    model = EntitySetValueModel(ModelConfig.from_dict(manifest["model_config"]))
    model.load_weights(str(model_path))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = promote(args.run_dir, args.output)
    print(path)


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
