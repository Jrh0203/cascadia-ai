"""Build the hash-pinned D1 training views (preregistration 2026-07-16 09:00).

The raw corpora stay immutable; this tool derives two masked training
views:

**Base view** (``--base`` + ``--base-out``): the Stage A generation shard
with the cheap search supervision masked at every tranche root — the
per-action ``q_valid`` rows are zeroed and a per-record ``policy_valid``
array is added (zero at tranche roots). Realized outcomes stay valid: the
trajectory genuinely happened. Selected roots are located through the
decisions ledger, whose (seed, ply) rows map one-to-one onto the packed
record order (seed-ascending, ply-ascending — the exporter sorts per-seed
shards by seed before merging).

**D1 view** (``--d1`` + ``--d1-out``): the relabel shard with a
per-record ``outcome_valid`` array of zeros — its outcome fields carry
the behavior trajectory's realized result, which must not be trained on
again at the same roots (EfficientZero stale-target correction). The
fresh aggregated search fields stay fully valid.

Every output embeds a ``view`` metadata block with the source shard
SHA-256, the mask SHA-256, and counts, so a training manifest pins the
exact view bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path: Path) -> set[tuple[int, int]]:
    roots = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            roots.add((int(row["seed"]), int(row["ply"])))
    if not roots:
        raise ValueError(f"{path}: empty probe-roots mask")
    return roots


def ledger_record_order(path: Path) -> list[tuple[int, int]]:
    """(seed, ply) in packed-record order: seed-ascending, ply-ascending."""
    by_seed: dict[int, list[int]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "gumbel_decision":
                continue
            by_seed.setdefault(int(row["seed"]), []).append(int(row["ply"]))
    order: list[tuple[int, int]] = []
    for seed in sorted(by_seed):
        plies = by_seed[seed]
        if sorted(plies) != plies:
            raise ValueError(f"ledger seed {seed} plies are not ascending")
        if len(set(plies)) != len(plies):
            raise ValueError(f"ledger seed {seed} has duplicate plies")
        order.extend((seed, ply) for ply in plies)
    if not order:
        raise ValueError(f"{path}: no gumbel_decision rows")
    return order


def _load_arrays(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as npz:
        return {name: npz[name].copy() for name in npz.files}


def _scalar_string(value: Any) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _write_view(arrays: dict[str, Any], metadata: dict[str, Any], out_path: Path) -> None:
    import numpy as np

    arrays = dict(arrays)
    arrays["metadata_json"] = np.array(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype=np.str_
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    # Explicit file handle: np.savez must not append ".npz" to the tmp name.
    # savez (not savez_compressed) keeps members ZIP_STORED and therefore
    # memory-mappable for the trainer loader.
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(out_path)


def build_base_view(
    base_path: Path, decisions_path: Path, mask_path: Path, out_path: Path
) -> dict[str, Any]:
    import numpy as np

    mask = read_mask(mask_path)
    order = ledger_record_order(decisions_path)
    arrays = _load_arrays(base_path)
    record_count = int(arrays["selected_action_index"].shape[0])
    if len(order) != record_count:
        raise ValueError(
            f"ledger rows ({len(order)}) != shard records ({record_count}); "
            "the ledger must be the same generation run's sidecar"
        )
    index_of = {root: index for index, root in enumerate(order)}
    missing = sorted(mask - set(order))
    if missing:
        raise ValueError(
            f"{len(missing)} tranche roots absent from the ledger (first: {missing[0]})"
        )
    selected = sorted(index_of[root] for root in mask)

    action_offsets = arrays["action_offsets"]
    q_valid = arrays["q_valid"].copy()
    policy_valid = np.ones((record_count,), dtype=np.uint8)
    for index in selected:
        start = int(action_offsets[index])
        end = int(action_offsets[index + 1])
        q_valid[start:end] = 0
        policy_valid[index] = 0
    arrays["q_valid"] = q_valid
    arrays["policy_valid"] = policy_valid

    metadata = json.loads(_scalar_string(arrays["metadata_json"]))
    metadata["view"] = {
        "type": "d1_base_view_masked_stale_search",
        "view_version": 1,
        "source_shard": str(base_path),
        "source_shard_sha256": _sha256(base_path),
        "mask": str(mask_path),
        "mask_sha256": _sha256(mask_path),
        "masked_roots": len(selected),
        "masked_fields": ["per_action_Q (via q_valid)", "improved_policy (via policy_valid)"],
        "outcome_fields": "retained (realized trajectory)",
    }
    del arrays["metadata_json"]
    _write_view(arrays, metadata, out_path)
    report = {
        "view": "base",
        "records": record_count,
        "masked_roots": len(selected),
        "out": str(out_path),
        "out_sha256": _sha256(out_path),
    }
    return report


def build_d1_view(
    d1_path: Path, audit_path: Path, mask_path: Path, out_path: Path
) -> dict[str, Any]:
    import numpy as np

    mask = read_mask(mask_path)
    audit_roots: list[tuple[int, int]] = []
    with audit_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "d1_repeat_audit":
                continue
            audit_roots.append((int(row["seed"]), int(row["ply"])))
    if not audit_roots:
        raise ValueError(f"{audit_path}: no d1_repeat_audit rows")
    extra = sorted(set(audit_roots) - mask)
    if extra:
        raise ValueError(
            f"{len(extra)} relabeled roots are outside the tranche mask (first: {extra[0]})"
        )
    arrays = _load_arrays(d1_path)
    record_count = int(arrays["selected_action_index"].shape[0])
    if len(audit_roots) != record_count:
        raise ValueError(
            f"audit rows ({len(audit_roots)}) != relabel records ({record_count})"
        )
    coverage = len(set(audit_roots)) / len(mask)

    arrays["outcome_valid"] = np.zeros((record_count,), dtype=np.uint8)
    arrays["policy_valid"] = np.ones((record_count,), dtype=np.uint8)

    metadata = json.loads(_scalar_string(arrays["metadata_json"]))
    metadata["view"] = {
        "type": "d1_relabel_view_masked_outcomes",
        "view_version": 1,
        "source_shard": str(d1_path),
        "source_shard_sha256": _sha256(d1_path),
        "mask": str(mask_path),
        "mask_sha256": _sha256(mask_path),
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "mask_coverage": coverage,
        "masked_fields": ["final_score_vector/score/rank (via outcome_valid)"],
        "search_fields": "fresh repeat-aggregated teacher targets, fully valid",
    }
    del arrays["metadata_json"]
    _write_view(arrays, metadata, out_path)
    return {
        "view": "d1",
        "records": record_count,
        "mask_coverage": coverage,
        "out": str(out_path),
        "out_sha256": _sha256(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="Stage A generation shard")
    parser.add_argument("--decisions", type=Path, help="generation decisions ledger")
    parser.add_argument("--tranche", type=Path, required=True, help="tranche probe-roots mask")
    parser.add_argument("--base-out", type=Path)
    parser.add_argument("--d1", type=Path, help="relabel training-records shard")
    parser.add_argument("--audit", type=Path, help="relabel repeat-audit sidecar")
    parser.add_argument("--d1-out", type=Path)
    args = parser.parse_args()
    reports = []
    if args.base or args.base_out:
        if not (args.base and args.decisions and args.base_out):
            raise SystemExit("base view requires --base, --decisions and --base-out")
        reports.append(build_base_view(args.base, args.decisions, args.tranche, args.base_out))
    if args.d1 or args.d1_out:
        if not (args.d1 and args.audit and args.d1_out):
            raise SystemExit("d1 view requires --d1, --audit and --d1-out")
        reports.append(build_d1_view(args.d1, args.audit, args.tranche, args.d1_out))
    if not reports:
        raise SystemExit("nothing to do: pass --base/--base-out and/or --d1/--d1-out")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
