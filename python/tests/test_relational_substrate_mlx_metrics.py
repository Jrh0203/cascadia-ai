from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from cascadia_mlx import relational_substrate_mlx_metrics as metrics_module
from cascadia_mlx.relational_substrate_mlx_metrics import (
    evaluate_relational_substrate,
)


def test_strategic_opportunity_mean_uses_elk_salmon_and_hawk(
    monkeypatch,
) -> None:
    observed = {}

    def fake_evaluate(*_args, **kwargs):
        observed["row_subsets"] = kwargs["row_subsets"]
        return {
            "subsets": {
                "elk_opportunity": {
                    "top64_r4800_winner_recall": 0.75
                },
                "salmon_opportunity": {
                    "top64_r4800_winner_recall": 0.60
                },
                "hawk_opportunity": {
                    "top64_r4800_winner_recall": 0.90
                },
                "bear_opportunity": {
                    "top64_r4800_winner_recall": 0.50
                },
            }
        }

    monkeypatch.setattr(
        metrics_module,
        "evaluate_r3_action_edit",
        fake_evaluate,
    )
    dataset = SimpleNamespace(
        opportunity_rows={
            "elk": np.asarray([0, 2]),
            "salmon": np.asarray([1]),
            "hawk": np.asarray([2]),
            "bear": np.asarray([3]),
        },
        parent_token_statistics=lambda _arm: {"groups": 4},
        derivative_statistics=lambda _arm: {"enabled": False},
    )

    report = evaluate_relational_substrate(
        object(),
        dataset,
        arm="c0-exact-r2",
    )

    assert set(observed["row_subsets"]) == {
        "elk_opportunity",
        "salmon_opportunity",
        "hawk_opportunity",
        "bear_opportunity",
    }
    assert report["strategic_opportunity_recall"]["primary_mean"] == 0.75
    assert (
        report["strategic_opportunity_recall"]["bear_diagnostic"]
        == 0.50
    )
