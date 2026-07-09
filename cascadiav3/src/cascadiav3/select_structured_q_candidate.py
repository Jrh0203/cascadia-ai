"""Select one structured-Q head-only arm on the disjoint selection block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_OBJECTIVE = "gumbel-selfplay-structured-q"
EXPECTED_METRIC = "locked_val_q_decomposition"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_candidate(arms: dict[str, Path]) -> dict[str, Any]:
    if len(arms) < 2:
        raise ValueError("structured-Q selection requires at least two arms")
    normalized: list[dict[str, Any]] = []
    reference_datasets: dict[str, Any] | None = None
    reference_source_hashes: dict[str, Any] | None = None
    for label, report_path in sorted(arms.items()):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "pass":
            raise ValueError(f"arm {label} did not complete successfully")
        if report.get("objective") != EXPECTED_OBJECTIVE:
            raise ValueError(f"arm {label} has the wrong objective")
        if report.get("selection_metric") != EXPECTED_METRIC or report.get(
            "selection_mode"
        ) != "min":
            raise ValueError(f"arm {label} has the wrong selection contract")
        if not report.get("q_decomposition_head_only") or not report.get("config", {}).get(
            "q_decomposition"
        ):
            raise ValueError(f"arm {label} is not structured head-only training")
        if report.get("q_component_initialization") != "equal_split_of_loaded_legacy_q":
            raise ValueError(f"arm {label} did not preserve the incumbent Q initialization")
        if report.get("schema_ids") != ["cascadiav3.expert_tensor_shard.v4"]:
            raise ValueError(f"arm {label} did not use only v4 shards")
        metric = float(report.get("best_selection_metric_value", math.nan))
        if not math.isfinite(metric):
            raise ValueError(f"arm {label} has a non-finite selection metric")
        datasets = report.get("dataset_manifests")
        source_hashes = report.get("source_hashes")
        if reference_datasets is None:
            reference_datasets = datasets
            reference_source_hashes = source_hashes
        elif datasets != reference_datasets or source_hashes != reference_source_hashes:
            raise ValueError("structured-Q arms do not share exact data and source hashes")
        checkpoint_dir = Path(str(report["checkpoint_dir"]))
        manifest = checkpoint_dir / "best_locked_val.manifest.json"
        if not manifest.is_absolute():
            manifest = report_path.parent / manifest
            if not manifest.exists():
                manifest = Path(str(report["checkpoint_dir"])) / "best_locked_val.manifest.json"
        if not manifest.exists():
            raise ValueError(f"arm {label} best checkpoint manifest is missing: {manifest}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        weights = manifest.parent / str(manifest_payload["weights"])
        if not weights.exists():
            raise ValueError(f"arm {label} best checkpoint weights are missing: {weights}")
        normalized.append(
            {
                "label": label,
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "learning_rate": float(report["optimizer"]["lr"]),
                "selection_metric": metric,
                "best_step": int(manifest_payload["step"]),
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
                "weights": str(weights),
                "weights_sha256": _sha256(weights),
            }
        )
    normalized.sort(key=lambda arm: (arm["selection_metric"], arm["learning_rate"], arm["label"]))
    chosen = normalized[0]
    return {
        "status": "pass",
        "selection_metric": EXPECTED_METRIC,
        "selection_mode": "min",
        "tie_break": "lower_learning_rate_then_label",
        "chosen": chosen,
        "arms": normalized,
        "dataset_manifests": reference_datasets,
        "source_hashes": reference_source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="label=training-report.json (repeat at least twice)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms: dict[str, Path] = {}
    for raw in args.arm:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path or label in arms:
            raise ValueError(f"invalid or duplicate --arm {raw!r}")
        arms[label] = Path(path)
    report = select_candidate(arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": report["chosen"]["label"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
