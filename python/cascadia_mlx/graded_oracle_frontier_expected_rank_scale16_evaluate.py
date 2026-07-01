"""Selected-checkpoint open evaluation for ADR 0101."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascadia_mlx.graded_oracle_frontier_expected_rank_evaluate import (
    _write_json,
    evaluate_selected_expected_rank,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    EXPERIMENT_ID,
    STUDENT_TEMPERATURE,
    TARGET_SCALE,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16_train import (
    RUN_KIND,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_selected_expected_rank(
        run_dir=args.run_dir,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        train_cache=args.train_cache,
        validation_cache=args.validation_cache,
        experiment_id=EXPERIMENT_ID,
        target_scale=TARGET_SCALE,
        student_temperature=STUDENT_TEMPERATURE,
        expected_run_kind=RUN_KIND,
    )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
