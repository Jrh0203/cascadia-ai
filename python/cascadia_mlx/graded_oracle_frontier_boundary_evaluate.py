"""Open-split evaluation for the ADR 0092 boundary-ranking pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascadia_mlx.graded_oracle_frontier_boundary_train import EXPERIMENT_ID
from cascadia_mlx.graded_oracle_frontier_target_curriculum_evaluate import (
    evaluate_frontier_open_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_frontier_open_pilot(
        run_dir=args.run_dir,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        expected_kind="graded-oracle-frontier-boundary-ranking",
        experiment_id=EXPERIMENT_ID,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
