"""Validate packed expert tensor semantic invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .expert_tensor_shards import ExpertTensorShard


def _filter_drop_count(metadata: dict[str, Any]) -> int:
    filter_meta = metadata.get("filter")
    if isinstance(filter_meta, dict):
        return int(filter_meta.get("selected_action_dropped_count", 0))
    return int(metadata.get("selected_action_dropped_count", 0))


def validate_shard(
    path: Path,
    *,
    require_selected_action_dropped_count: int | None,
    require_q_equals_afterstate_plus_score_to_go: bool,
    tolerance: float,
) -> dict[str, Any]:
    import numpy as np

    shard = ExpertTensorShard(path)
    try:
        action_counts = shard.action_offsets[1:] - shard.action_offsets[:-1]
        invalid_selected = [
            int(index)
            for index, (selected, action_count) in enumerate(zip(shard.selected_action_index, action_counts))
            if int(selected) < 0 or int(selected) >= int(action_count)
        ]
        if invalid_selected:
            raise AssertionError(f"selected_action_index out of bounds for roots {invalid_selected[:10]}")

        max_abs_q_error = 0.0
        if require_q_equals_afterstate_plus_score_to_go:
            valid = shard.q_valid.astype(bool)
            delta = np.abs(
                shard.target_q.astype(np.float64)
                - (
                    shard.exact_afterstate_score_active.astype(np.float64)
                    + shard.target_score_to_go.astype(np.float64)
                )
            )
            if valid.any():
                max_abs_q_error = float(delta[valid].max(initial=0.0))
            if max_abs_q_error > tolerance:
                raise AssertionError(
                    f"target_q invariant failed: max abs error {max_abs_q_error} > {tolerance}"
                )

        selected_action_dropped_count = _filter_drop_count(shard.metadata)
        if (
            require_selected_action_dropped_count is not None
            and selected_action_dropped_count != require_selected_action_dropped_count
        ):
            raise AssertionError(
                "selected_action_dropped_count "
                f"{selected_action_dropped_count} != required {require_selected_action_dropped_count}"
            )

        return {
            "status": "pass",
            "path": str(path),
            "record_count": len(shard),
            "total_action_count": int(shard.actions.shape[0]),
            "selected_action_dropped_count": selected_action_dropped_count,
            "max_abs_q_invariant_error": max_abs_q_error,
            "q_invariant_tolerance": tolerance,
            "q_semantics": "target_q == exact_afterstate_score_active + target_score_to_go",
        }
    finally:
        shard.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--report")
    parser.add_argument("--require-selected-action-dropped-count", type=int)
    parser.add_argument("--require-q-equals-afterstate-plus-score-to-go", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()
    report = validate_shard(
        Path(args.shard),
        require_selected_action_dropped_count=args.require_selected_action_dropped_count,
        require_q_equals_afterstate_plus_score_to_go=args.require_q_equals_afterstate_plus_score_to_go,
        tolerance=args.tolerance,
    )
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
