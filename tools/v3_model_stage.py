"""Content-checked V3 model identity and Bacalhau bundle staging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
from cascadia_cluster import InputReference


class ModelStageError(ValueError):
    """A V3 serving bundle is invalid or cannot be staged safely."""


def _digest(path: Path) -> str:
    value = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_identity(directory: Path) -> str:
    value = json.loads((directory / "model.json").read_text())
    weights = directory / str(value.get("weights_file", ""))
    if (
        not weights.is_file()
        or value.get("weights_blake3") != _digest(weights)
        or not value.get("serving_compatible")
    ):
        raise ModelStageError(f"V3 serving bundle is invalid: {directory}")
    return f"{value['architecture_id']}:{value['checkpoint_id']}:{value['weights_blake3']}"


@dataclass(frozen=True)
class StagedModel:
    references: tuple[InputReference, InputReference]
    materialized_directory: str
    bundle_spec: dict[str, dict[str, str]]


def stage_model(store: Any, directory: Path, name: str) -> StagedModel:
    portable = "abcdefghijklmnopqrstuvwxyz0123456789-"
    if not name or any(character not in portable for character in name):
        raise ModelStageError("model stage name is not portable")
    model_identity(directory)
    manifest = json.loads((directory / "model.json").read_text())
    weights = directory / manifest["weights_file"]
    manifest_reference = store.stage_file(
        directory / "model.json", target=f"/inputs/model-bundles/{name}/manifest"
    )
    weights_reference = store.stage_file(
        weights, target=f"/inputs/model-bundles/{name}/weights"
    )
    destination = f"/tmp/cascadia-models/{name}"
    return StagedModel(
        references=(manifest_reference, weights_reference),
        materialized_directory=destination,
        bundle_spec={
            destination: {
                "manifest": manifest_reference.mounted_path,
                "weights": weights_reference.mounted_path,
            }
        },
    )


def bundle_environment(stages: list[StagedModel]) -> str:
    value: dict[str, dict[str, str]] = {}
    for stage in stages:
        overlap = value.keys() & stage.bundle_spec.keys()
        if overlap:
            raise ModelStageError(f"duplicate model materialization destination: {overlap}")
        value.update(stage.bundle_spec)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
