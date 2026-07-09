"""Fail-closed cross-shard audit for raw exact-grounded structured-Q data."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .expert_tensor_shards import SHARD_VERSION_V4, ExpertTensorShard

ELIGIBILITY = "gumbel_selfplay_expert_iteration"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 digest") from exc
    return value.lower()


@dataclass(frozen=True)
class SeedDomain:
    first_seed: int
    seed_count: int
    plies_per_seed: int
    mode: str

    @property
    def last_seed(self) -> int:
        return self.first_seed + self.seed_count - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_seed": self.first_seed,
            "last_seed": self.last_seed,
            "seed_count": self.seed_count,
            "plies_per_seed": self.plies_per_seed,
            "mode": self.mode,
        }


def parse_seed_domain(raw: Any) -> SeedDomain:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("seed_domain must be a non-empty string")
    fields: dict[str, str] = {}
    for entry in raw.split(","):
        key, separator, value = entry.strip().partition("=")
        if not separator or not key or not value:
            raise ValueError(f"malformed seed_domain entry {entry!r}")
        if key in fields:
            raise ValueError(f"duplicate seed_domain field {key!r}")
        fields[key] = value
    required = ("first_seed", "seed_count", "plies_per_seed", "mode")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"seed_domain missing fields: {missing}")
    try:
        first_seed = int(fields["first_seed"])
        seed_count = int(fields["seed_count"])
        plies_per_seed = int(fields["plies_per_seed"])
    except ValueError as exc:
        raise ValueError("seed_domain numeric fields must be integers") from exc
    if first_seed < 0 or seed_count <= 0 or plies_per_seed <= 0:
        raise ValueError("seed_domain requires nonnegative first_seed and positive counts")
    return SeedDomain(
        first_seed=first_seed,
        seed_count=seed_count,
        plies_per_seed=plies_per_seed,
        mode=fields["mode"],
    )


def _teacher_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    teacher = metadata.get("teacher_model")
    if not isinstance(teacher, dict):
        raise ValueError("structured-Q shard requires teacher_model metadata")
    manifest = teacher.get("manifest")
    weights = teacher.get("weights")
    if not isinstance(manifest, dict) or not isinstance(weights, dict):
        raise ValueError("teacher_model requires manifest and weights artifacts")
    return {
        "manifest_sha256": _require_sha256(manifest.get("sha256"), "teacher manifest"),
        "manifest_bytes": int(manifest.get("bytes", -1)),
        "weights_sha256": _require_sha256(weights.get("sha256"), "teacher weights"),
        "weights_bytes": int(weights.get("bytes", -1)),
        "checkpoint_tag": teacher.get("checkpoint_tag"),
        "step": teacher.get("step"),
        "model_name": teacher.get("model_name"),
        "model_size": teacher.get("model_size"),
    }


def _contract(metadata: dict[str, Any]) -> dict[str, Any]:
    search = metadata.get("search")
    execution = metadata.get("execution")
    if not isinstance(search, dict) or not isinstance(execution, dict):
        raise ValueError("structured-Q shard requires search and execution metadata")
    return {
        "schema_id": metadata.get("schema_id"),
        "scientific_eligibility": metadata.get("scientific_eligibility"),
        "ruleset_id": metadata.get("ruleset_id"),
        "source_revision": metadata.get("source_revision"),
        "search": search,
        "execution": execution,
        "teacher": _teacher_contract(metadata),
    }


def _audit_one(label: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"structured-Q shard is missing: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"structured-Q sidecar manifest is missing: {manifest_path}")
    shard_sha256 = _sha256(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checksum") != shard_sha256:
        raise ValueError(f"sidecar checksum mismatch for {label}")

    shard = ExpertTensorShard(path)
    try:
        if shard.version != SHARD_VERSION_V4:
            raise ValueError(f"{label} is not a schema-v4 structured-Q shard")
        metadata = shard.metadata
        if metadata.get("scientific_eligibility") != ELIGIBILITY:
            raise ValueError(f"{label} is not eligible expert-iteration data")
        if manifest.get("metadata") != metadata:
            raise ValueError(f"sidecar metadata mismatch for {label}")
        if manifest.get("schema_id") != shard.version or manifest.get("version") != shard.version:
            raise ValueError(f"sidecar schema mismatch for {label}")
        if manifest.get("scientific_eligibility") != ELIGIBILITY:
            raise ValueError(f"sidecar eligibility mismatch for {label}")
        if manifest.get("seed_domain") != metadata.get("seed_domain"):
            raise ValueError(f"sidecar seed domain mismatch for {label}")
        if manifest.get("record_count") != len(shard):
            raise ValueError(f"sidecar record count mismatch for {label}")
        if manifest.get("total_action_count") != int(shard.actions.shape[0]):
            raise ValueError(f"sidecar action count mismatch for {label}")
        domain = parse_seed_domain(metadata.get("seed_domain"))
        expected_records = domain.seed_count * domain.plies_per_seed
        if len(shard) != expected_records:
            raise ValueError(
                f"{label} record count {len(shard)} != seed-domain expectation {expected_records}"
            )
        if domain.mode != "gumbel_selfplay_tensor_corpus":
            raise ValueError(f"{label} has the wrong seed-domain mode")

        exact = np.asarray(shard.exact_endgame, dtype=bool)
        exact_turns = int(metadata["search"].get("exact_endgame_turns", -1))
        expected_exact = 4 * domain.seed_count if exact_turns == 1 else 0
        if exact_turns not in (0, 1) or int(exact.sum()) != expected_exact:
            raise ValueError(
                f"{label} exact-row count {int(exact.sum())} != expected {expected_exact}"
            )

        offsets = np.asarray(shard.action_offsets, dtype=np.int64)
        selected = np.asarray(shard.selected_action_index, dtype=np.int64)
        selected_global = offsets[:-1] + selected
        q_valid = np.asarray(shard.q_valid, dtype=bool)
        if not q_valid[selected_global].all():
            raise ValueError(f"{label} contains a selected action without a valid Q target")
        q_delta = np.abs(
            np.asarray(shard.target_q, dtype=np.float64)
            - np.asarray(shard.exact_afterstate_score_active, dtype=np.float64)
            - np.asarray(shard.target_score_to_go, dtype=np.float64)
        )
        q_error = float(q_delta[q_valid].max(initial=0.0))
        if q_error > 1.0e-4:
            raise ValueError(f"{label} completed-Q identity error {q_error} exceeds 1e-4")

        after_components = np.asarray(
            shard.exact_afterstate_score_decomposition_active,
            dtype=np.float64,
        )
        after_error = float(
            np.abs(
                after_components.sum(axis=1)
                - np.asarray(shard.exact_afterstate_score_active, dtype=np.float64)
            ).max(initial=0.0)
        )
        active = np.asarray(shard.active_seat, dtype=np.int64)
        terminal_components = np.asarray(shard.score_decomposition, dtype=np.float64)[
            np.arange(len(shard)), :, active
        ]
        terminal_scores = np.asarray(shard.final_score_vector, dtype=np.float64)[
            np.arange(len(shard)), active
        ]
        terminal_error = float(
            np.abs(terminal_components.sum(axis=1) - terminal_scores).max(initial=0.0)
        )
        return {
            "label": label,
            "path": str(path),
            "sha256": shard_sha256,
            "bytes": path.stat().st_size,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "records": len(shard),
            "actions": int(shard.actions.shape[0]),
            "q_valid_actions": int(q_valid.sum()),
            "exact_rows": int(exact.sum()),
            "seed_domain": domain.to_dict(),
            "max_abs_q_identity_error": q_error,
            "max_abs_afterstate_component_error": after_error,
            "max_abs_terminal_component_error": terminal_error,
            "contract": _contract(metadata),
        }
    finally:
        shard.close()


def _assert_disjoint(rows: list[dict[str, Any]]) -> None:
    intervals = sorted(
        (
            int(row["seed_domain"]["first_seed"]),
            int(row["seed_domain"]["last_seed"]),
            str(row["label"]),
        )
        for row in rows
    )
    for (_, previous_last, previous_label), (current_first, _, current_label) in zip(
        intervals,
        intervals[1:],
    ):
        if current_first <= previous_last:
            raise ValueError(
                f"seed overlap between {previous_label} and {current_label} at {current_first}"
            )


def audit_shards(
    shards: dict[str, Path],
    *,
    excluded_shards: dict[str, Path] | None = None,
    expected_source_revision: str | None = None,
    expected_teacher_manifest_sha256: str | None = None,
    expected_teacher_weights_sha256: str | None = None,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one structured-Q shard is required")
    excluded_shards = excluded_shards or {}
    duplicate_labels = set(shards).intersection(excluded_shards)
    if duplicate_labels:
        raise ValueError(f"duplicate primary/excluded labels: {sorted(duplicate_labels)}")
    primary = [_audit_one(label, path) for label, path in sorted(shards.items())]
    excluded = [
        _audit_one(label, path) for label, path in sorted(excluded_shards.items())
    ]
    all_rows = primary + excluded
    reference = all_rows[0]["contract"]
    for row in all_rows[1:]:
        if row["contract"] != reference:
            raise ValueError(f"cross-shard contract mismatch for {row['label']}")
    _assert_disjoint(all_rows)

    if (
        expected_source_revision is not None
        and reference["source_revision"] != expected_source_revision
    ):
        raise ValueError("structured-Q source revision does not match expectation")
    teacher = reference["teacher"]
    if (
        expected_teacher_manifest_sha256 is not None
        and teacher["manifest_sha256"] != expected_teacher_manifest_sha256
    ):
        raise ValueError("structured-Q teacher manifest does not match expectation")
    if (
        expected_teacher_weights_sha256 is not None
        and teacher["weights_sha256"] != expected_teacher_weights_sha256
    ):
        raise ValueError("structured-Q teacher weights do not match expectation")

    return {
        "status": "pass",
        "audit": "raw_structured_q_cross_shard_v1",
        "contract": reference,
        "primary": primary,
        "excluded_shards": excluded,
        "totals": {
            "shards": len(primary),
            "seeds": sum(int(row["seed_domain"]["seed_count"]) for row in primary),
            "records": sum(int(row["records"]) for row in primary),
            "actions": sum(int(row["actions"]) for row in primary),
            "q_valid_actions": sum(int(row["q_valid_actions"]) for row in primary),
            "exact_rows": sum(int(row["exact_rows"]) for row in primary),
        },
    }


def _labeled_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path or label in result:
            raise ValueError(f"invalid or duplicate labeled path {raw!r}")
        result[label] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, help="label=raw-v4.npz")
    parser.add_argument(
        "--exclude-shard",
        action="append",
        default=[],
        help="label=raw-v4.npz; validate contract and prove seed disjointness",
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-teacher-manifest-sha256")
    parser.add_argument("--expected-teacher-weights-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit_shards(
        _labeled_paths(args.shard),
        excluded_shards=_labeled_paths(args.exclude_shard),
        expected_source_revision=args.expected_source_revision,
        expected_teacher_manifest_sha256=args.expected_teacher_manifest_sha256,
        expected_teacher_weights_sha256=args.expected_teacher_weights_sha256,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
