"""R1.4 Stage 1 preregistered trainer flags: V1b, V2, C1, T0.

Preregistration: `cascadiav3/EXPERIMENT_LOG.md` 2026-07-13 23:45 ("Stage 1
arms"); design memo `docs/v3/R1_4_DENSIFICATION_DESIGN.md` sections 4-5.

The cardinal constraint under test: with none of the new flags set, the
trainer is BIT-IDENTICAL to the pre-Stage-1 trainer. `GOLDEN_GUMBEL` /
`GOLDEN_EXPERT` below were captured by running `build_loss_fixture` through
`_loss_components` of the pre-Stage-1 working tree (commit ce63c3da plus the
uncommitted 2026-07-13 docs, torch 2.13.0 CPU) BEFORE any Stage 1 edit; the
default-config test asserts exact float equality against them.

Torch-dependent tests skip cleanly when torch is unavailable (the repo's
standard `_require_torch` pattern); the C1 argparse tests and the numpy
pairing test run without torch.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TOKEN_FEATURE_DIM = 41  # PUBLIC_TOKEN_FEATURE_DIM

# Captured from the pre-Stage-1 `_loss_components` on `build_loss_fixture`
# (see module docstring). Keys cover every loss component and retention
# metric the trainer aggregates. "path_consistency" is asserted separately
# (the key did not exist pre-Stage-1; its flag-off value must be exactly 0).
GOLDEN_GUMBEL: dict[str, float] = {
    "total": 15.536055564880371,
    "policy": 1.0279959440231323,
    "weighted_policy": 1.0279959440231323,
    "q": 3.226492166519165,
    "score_to_go_q": 3.226492166519165,
    "final_q_regret": 0.0,
    "value": 25.54166603088379,
    "score": 1.8656662702560425,
    "rank": 1.3893547058105469,
    "uncertainty": 0.2909727096557617,
    "greedy_policy": 1.1954425573349,
    "greedy_margin": 0.25999999046325684,
    "pairwise": 0.0,
    "policy_recall": -0.0,
    "q_decomposition": 0.0,
    "teacher_top1": 0.3333333432674408,
    "greedy_top1": 0.0,
    "mean_teacher_rank": 2.3333332538604736,
    "mean_greedy_rank": 3.0,
    "teacher_advantage_over_greedy": 0.20666630566120148,
    "pairwise_accuracy": 0.0,
    "pairwise_examples": 0.0,
    "pairwise_mean_snr": 0.0,
    "policy_best_top16": 1.0,
    "policy_confident_best_top16": -0.0,
    "policy_confident_best_top16_correct": -0.0,
    "policy_recall_examples": -0.0,
}
GOLDEN_EXPERT: dict[str, float] = {
    "total": 10.967745780944824,
    "policy": 1.096893548965454,
    "weighted_policy": 1.0850111246109009,
    "q": 3.226492166519165,
    "score_to_go_q": 3.226492166519165,
    "final_q_regret": 0.0,
    "value": 25.54166603088379,
    "score": 1.8656662702560425,
    "rank": 1.3893547058105469,
    "uncertainty": 0.2909727096557617,
    "greedy_policy": 1.1954425573349,
    "greedy_margin": 0.5099999904632568,
    "pairwise": 0.0,
    "policy_recall": -0.0,
    "q_decomposition": 0.0,
    "teacher_top1": 0.3333333432674408,
    "greedy_top1": 0.0,
    "mean_teacher_rank": 2.3333332538604736,
    "mean_greedy_rank": 3.0,
    "teacher_advantage_over_greedy": 0.20666630566120148,
    "pairwise_accuracy": 0.0,
    "pairwise_examples": 0.0,
    "pairwise_mean_snr": 0.0,
    "policy_best_top16": 1.0,
    "policy_confident_best_top16": -0.0,
    "policy_confident_best_top16_correct": -0.0,
    "policy_recall_examples": -0.0,
}


def build_loss_fixture(*, improved_policy: bool):  # type: ignore[no-untyped-def]
    """Deterministic, arithmetic-only outputs/batch fixture (no RNG, no model).

    Values are arange-based so golden floats are stable across processes.
    """
    import torch

    batch_size, action_count = 3, 4

    def grid(*shape: int, scale: float, shift: float):  # type: ignore[no-untyped-def]
        numel = 1
        for dim in shape:
            numel *= dim
        return torch.arange(numel, dtype=torch.float32).reshape(*shape) * scale + shift

    outputs = {
        "logits": grid(batch_size, action_count, scale=0.13, shift=-0.4),
        "q": grid(batch_size, action_count, scale=0.21, shift=-0.9),
        "uncertainty": torch.nn.functional.softplus(
            grid(batch_size, action_count, scale=0.05, shift=-0.2)
        ),
        "value_vector": grid(batch_size, 4, scale=0.5, shift=60.0),
        "rank_logits": grid(batch_size, 4, 4, scale=0.07, shift=-0.3),
        "score_decomposition": grid(batch_size, 3, 4, scale=0.11, shift=5.0),
    }
    action_mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False], [True, True, False, False]]
    )
    batch = {
        "action_mask": action_mask,
        "q_valid": torch.tensor(
            [[True, True, False, False], [True, False, True, False], [False, True, False, False]]
        ),
        "selected_action_index": torch.tensor([0, 1, 1]),
        "greedy_action_index": torch.tensor([0, 0, 0]),
        "target_q": grid(batch_size, action_count, scale=0.31, shift=58.0),
        "target_score_to_go": grid(batch_size, action_count, scale=0.17, shift=3.0),
        "target_q_variance": grid(batch_size, action_count, scale=0.09, shift=0.5),
        "target_q_count": grid(batch_size, action_count, scale=1.0, shift=2.0),
        "exact_afterstate_score_active": grid(batch_size, action_count, scale=0.23, shift=55.0),
        "target_value": grid(batch_size, 4, scale=1.0, shift=62.0),
        "target_rank": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2]]),
        "target_score": grid(batch_size, 3, 4, scale=0.13, shift=6.0),
        "has_improved_policy": improved_policy,
    }
    if improved_policy:
        raw = grid(batch_size, action_count, scale=0.19, shift=0.05).masked_fill(
            ~action_mask, 0.0
        )
        batch["improved_policy"] = raw / raw.sum(dim=1, keepdim=True)
        batch["search_root_value"] = torch.tensor([70.0, 64.5, 61.25])
    return outputs, batch


def build_phase_tokens(seat_tile_pairs):  # type: ignore[no-untyped-def]
    """Packed public-token rows carrying the analyzer's phase proxy.

    One active-player row per record (kind one-hot ``player`` at column 0,
    relative_seat == 0) storing owner_seat / OWNER_SEAT_SCALE and
    tile_count / TILE_COUNT_SCALE, plus one non-player row. A pair of
    ``(None, None)`` produces a record with NO recoverable active-player row.
    """
    import torch

    from cascadiav3.analyze_label_density import (
        OWNER_SEAT_SCALE,
        TILE_COUNT_SCALE,
        TOKEN_COL_KIND_PLAYER,
        TOKEN_COL_OWNER_SEAT,
        TOKEN_COL_RELATIVE_SEAT,
        TOKEN_COL_TILE_COUNT,
    )

    batch_size = len(seat_tile_pairs)
    tokens = torch.zeros((batch_size, 2, TOKEN_FEATURE_DIM), dtype=torch.float32)
    token_mask = torch.ones((batch_size, 2), dtype=torch.bool)
    for row, (seat, tile_count) in enumerate(seat_tile_pairs):
        if seat is None:
            continue
        tokens[row, 0, TOKEN_COL_KIND_PLAYER] = 1.0
        tokens[row, 0, TOKEN_COL_OWNER_SEAT] = seat / OWNER_SEAT_SCALE
        tokens[row, 0, TOKEN_COL_RELATIVE_SEAT] = 0.0
        tokens[row, 0, TOKEN_COL_TILE_COUNT] = tile_count / TILE_COUNT_SCALE
        # Non-player token row (e.g. a habitat tile): player kind stays 0.
        tokens[row, 1, TOKEN_COL_TILE_COUNT] = tile_count / TILE_COUNT_SCALE
    return tokens, token_mask


class _TorchCase(unittest.TestCase):
    def _require_torch(self):  # type: ignore[no-untyped-def]
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch unavailable")


class DefaultBitIdentityTest(_TorchCase):
    """Flag-off behavior must match the pre-Stage-1 trainer exactly."""

    def _components(self, *, improved_policy: bool, objective: str, **kwargs):  # type: ignore[no-untyped-def]
        from cascadiav3.torch_train_cascadiaformer import (
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=improved_policy)
        return _loss_components(outputs, batch, loss_weights_for_objective(objective), **kwargs)

    def _assert_matches_golden(self, losses, golden: dict[str, float]) -> None:  # type: ignore[no-untyped-def]
        for key, expected in golden.items():
            self.assertEqual(
                float(losses[key]),
                expected,
                msg=f"loss component {key!r} drifted from the pre-Stage-1 trainer",
            )
        self.assertEqual(float(losses["path_consistency"]), 0.0)

    def test_gumbel_selfplay_components_match_pre_stage1_goldens(self) -> None:
        self._require_torch()
        losses = self._components(improved_policy=True, objective="gumbel-selfplay")
        self._assert_matches_golden(losses, GOLDEN_GUMBEL)

    def test_expert_objective_components_match_pre_stage1_goldens(self) -> None:
        self._require_torch()
        losses = self._components(improved_policy=False, objective="expert")
        self._assert_matches_golden(losses, GOLDEN_EXPERT)

    def test_default_options_and_none_options_are_equivalent(self) -> None:
        self._require_torch()
        from cascadiav3.torch_train_cascadiaformer import Stage1TargetOptions

        base = self._components(improved_policy=True, objective="gumbel-selfplay")
        with_none = self._components(
            improved_policy=True, objective="gumbel-selfplay", options=None
        )
        with_defaults = self._components(
            improved_policy=True, objective="gumbel-selfplay", options=Stage1TargetOptions()
        )
        for key, value in base.items():
            self.assertEqual(float(value), float(with_none[key]))
            self.assertEqual(float(value), float(with_defaults[key]))

    def test_default_loss_weights_keep_new_term_off(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            loss_weights_for_objective,
        )

        self.assertEqual(LossWeights().path_consistency, 0.0)
        for objective in (
            "expert",
            "gumbel-selfplay",
            "gumbel-selfplay-structured-q",
            "gumbel-selfplay-pairwise",
            "gumbel-policy-recall",
            "k32-greedy-retention",
            "pure-greedy-retention",
            "search-improved-greedy-retention",
        ):
            self.assertEqual(loss_weights_for_objective(objective).path_consistency, 0.0)

    def test_default_model_value_head_contract_unchanged(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        config = config_for_size("tiny")
        self.assertEqual(config.value_quantiles, 0)
        torch.manual_seed(11)
        model = build_cascadiaformer(config).eval()
        self.assertEqual(model.value_head.out_features, 4)
        with torch.inference_mode():
            outputs = model(
                torch.randn(2, 6, config.token_feature_dim),
                torch.ones(2, 6, dtype=torch.bool),
                torch.randn(2, 5, config.action_feature_dim),
                torch.ones(2, 5, dtype=torch.bool),
            )
        self.assertNotIn("value_quantile_values", outputs)
        self.assertEqual(outputs["value_vector"].shape, (2, 4))


class ValueTargetSearchMixTest(_TorchCase):
    """V1b: phase-gated search-value mixing into the ACTIVE seat's target."""

    def test_mixes_only_active_seat_and_only_at_or_above_min_tiles(self) -> None:
        self._require_torch()
        import torch
        import torch.nn.functional as F

        from cascadiav3.torch_train_cascadiaformer import (
            Stage1TargetOptions,
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        # Record 0: seat 1, 12 tiles (below the gate) -> pure outcome.
        # Record 1: seat 2, 13 tiles (at the gate)    -> mixed target.
        # Record 2: no recoverable player row          -> pure outcome.
        batch["tokens"], batch["token_mask"] = build_phase_tokens(
            [(1, 12), (2, 13), (None, None)]
        )
        options = Stage1TargetOptions(
            value_target_search_mix=0.5, value_target_search_mix_min_tiles=13
        )
        losses = _loss_components(
            outputs, batch, loss_weights_for_objective("gumbel-selfplay"), options=options
        )

        expected_target = batch["target_value"].clone()
        expected_target[1, 2] = 0.5 * expected_target[1, 2] + 0.5 * batch["search_root_value"][1]
        expected = F.mse_loss(outputs["value_vector"], expected_target)
        self.assertEqual(float(losses["value"]), float(expected))
        # And the mixed loss really differs from the unmixed one.
        unmixed = F.mse_loss(outputs["value_vector"], batch["target_value"])
        self.assertNotEqual(float(losses["value"]), float(unmixed))

    def test_lambda_zero_is_inert_even_with_tokens_present(self) -> None:
        self._require_torch()
        import torch.nn.functional as F

        from cascadiav3.torch_train_cascadiaformer import (
            Stage1TargetOptions,
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        batch["tokens"], batch["token_mask"] = build_phase_tokens([(0, 20), (1, 20), (2, 20)])
        losses = _loss_components(
            outputs,
            batch,
            loss_weights_for_objective("gumbel-selfplay"),
            options=Stage1TargetOptions(value_target_search_mix=0.0),
        )
        expected = F.mse_loss(outputs["value_vector"], batch["target_value"])
        self.assertEqual(float(losses["value"]), float(expected))

    def test_packed_active_seat_takes_precedence_over_token_seat(self) -> None:
        self._require_torch()
        import torch
        import torch.nn.functional as F

        from cascadiav3.torch_train_cascadiaformer import (
            Stage1TargetOptions,
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        batch["tokens"], batch["token_mask"] = build_phase_tokens([(1, 15), (1, 15), (1, 15)])
        batch["active_seat"] = torch.tensor([3, 0, 2])  # v4 packed seat wins
        options = Stage1TargetOptions(value_target_search_mix=0.25)
        losses = _loss_components(
            outputs, batch, loss_weights_for_objective("gumbel-selfplay"), options=options
        )
        expected_target = batch["target_value"].clone()
        for row, seat in enumerate((3, 0, 2)):
            expected_target[row, seat] = (
                0.75 * expected_target[row, seat] + 0.25 * batch["search_root_value"][row]
            )
        expected = F.mse_loss(outputs["value_vector"], expected_target)
        self.assertEqual(float(losses["value"]), float(expected))

    def test_requires_search_root_value(self) -> None:
        self._require_torch()
        from cascadiav3.torch_train_cascadiaformer import (
            Stage1TargetOptions,
            _loss_components,
            loss_weights_for_objective,
        )

        outputs, batch = build_loss_fixture(improved_policy=False)
        batch["tokens"], batch["token_mask"] = build_phase_tokens([(0, 15), (1, 15), (2, 15)])
        with self.assertRaisesRegex(ValueError, "search_root_value"):
            _loss_components(
                outputs,
                batch,
                loss_weights_for_objective("expert"),
                options=Stage1TargetOptions(value_target_search_mix=0.5),
            )


class ValueQuantileHeadTest(_TorchCase):
    """V2: K-quantile distributional value head with pinball loss."""

    def _forward(self, value_quantiles: int):  # type: ignore[no-untyped-def]
        import torch
        from dataclasses import replace

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size

        config = replace(config_for_size("tiny"), value_quantiles=value_quantiles)
        torch.manual_seed(23)
        model = build_cascadiaformer(config).eval()
        with torch.inference_mode():
            outputs = model(
                torch.randn(2, 6, config.token_feature_dim),
                torch.ones(2, 6, dtype=torch.bool),
                torch.randn(2, 5, config.action_feature_dim),
                torch.ones(2, 5, dtype=torch.bool),
            )
        return model, outputs

    def test_head_outputs_4k_and_scalar_value_is_quantile_mean(self) -> None:
        self._require_torch()
        import torch

        model, outputs = self._forward(8)
        self.assertEqual(model.value_head.out_features, 4 * 8)
        self.assertEqual(outputs["value_quantile_values"].shape, (2, 4, 8))
        torch.testing.assert_close(
            outputs["value_vector"],
            outputs["value_quantile_values"].mean(dim=-1),
            rtol=0.0,
            atol=0.0,
        )

    def test_pinball_loss_exact_on_hand_fixture(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        # K=2 quantiles at levels 0.25 / 0.75; residual r = target - prediction.
        # pinball = mean over K of max(tau*r, (tau-1)*r).
        quantiles = torch.stack(
            [batch["target_value"] - 2.0, batch["target_value"] + 4.0], dim=-1
        )
        outputs["value_quantile_values"] = quantiles
        outputs["value_vector"] = quantiles.mean(dim=-1)
        losses = _loss_components(outputs, batch, LossWeights())
        # Residuals are constant: +2.0 for the tau=0.25 quantile, -4.0 for
        # tau=0.75. pinball(0.25, +2) = 0.5; pinball(0.75, -4) = 1.0.
        self.assertEqual(float(losses["value"]), (0.5 + 1.0) / 2.0)

    def test_default_zero_keeps_scalar_value_loss(self) -> None:
        self._require_torch()
        import torch.nn.functional as F

        from cascadiav3.torch_train_cascadiaformer import LossWeights, _loss_components

        outputs, batch = build_loss_fixture(improved_policy=True)
        losses = _loss_components(outputs, batch, LossWeights())
        expected = F.mse_loss(outputs["value_vector"], batch["target_value"])
        self.assertEqual(float(losses["value"]), float(expected))

    def test_init_skip_mismatched_covers_reshaped_value_head(self) -> None:
        self._require_torch()
        import io
        from contextlib import redirect_stdout
        from dataclasses import replace

        import torch

        from cascadiav3.torch_cascadiaformer import build_cascadiaformer, config_for_size
        from cascadiav3.torch_train_cascadiaformer import _load_weights_from_manifest

        torch.manual_seed(31)
        scalar_model = build_cascadiaformer(config_for_size("tiny"))
        torch.manual_seed(37)
        quantile_model = build_cascadiaformer(
            replace(config_for_size("tiny"), value_quantiles=8)
        )
        fresh_value_head = quantile_model.value_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "weights.pt"
            torch.save(scalar_model.state_dict(), weights_path)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"weights": "weights.pt"}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                _load_weights_from_manifest(quantile_model, manifest_path)
            with redirect_stdout(io.StringIO()) as captured:
                _load_weights_from_manifest(
                    quantile_model, manifest_path, skip_mismatched=True
                )
        self.assertIn("value_head", captured.getvalue())
        # Trunk warm-started; the reshaped value head stays at fresh init.
        torch.testing.assert_close(
            quantile_model.token_proj.weight, scalar_model.token_proj.weight
        )
        torch.testing.assert_close(quantile_model.value_head.weight, fresh_value_head)


class AuxWeightFlagTest(unittest.TestCase):
    """C1: short aux-weight spellings flow into LossWeights like --policy-weight."""

    def _captured_loss_weights(self, argv: list[str]):  # type: ignore[no-untyped-def]
        import io
        from contextlib import redirect_stdout

        import cascadiav3.torch_train_cascadiaformer as trainer

        captured: dict[str, object] = {}

        def fake_run_training(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return {"status": "pass"}

        with mock.patch.object(trainer, "run_training", fake_run_training):
            with mock.patch(
                "sys.argv",
                ["torch_train_cascadiaformer.py", "--train", "t.jsonl", "--val", "v.jsonl"]
                + argv,
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(trainer.main(), 0)
        return captured

    def test_new_weight_flags_override_objective_loss_weights(self) -> None:
        captured = self._captured_loss_weights(
            [
                "--objective",
                "gumbel-selfplay",
                "--score-weight",
                "0.2",
                "--rank-weight",
                "0.08",
                "--uncertainty-weight",
                "0.04",
                "--value-weight",
                "1.5",
                "--q-weight",
                "0.75",
            ]
        )
        weights = captured["loss_weights"]
        self.assertEqual(weights.score, 0.2)
        self.assertEqual(weights.rank, 0.08)
        self.assertEqual(weights.uncertainty, 0.04)
        self.assertEqual(weights.value, 1.5)
        self.assertEqual(weights.q, 0.75)
        # Untouched fields keep the objective's values.
        self.assertEqual(weights.policy, 1.0)
        self.assertEqual(weights.path_consistency, 0.0)

    def test_legacy_loss_weight_spellings_still_work(self) -> None:
        captured = self._captured_loss_weights(
            ["--score-loss-weight", "0.3", "--q-loss-weight", "0.9"]
        )
        weights = captured["loss_weights"]
        self.assertEqual(weights.score, 0.3)
        self.assertEqual(weights.q, 0.9)

    def test_stage1_flags_default_off_and_thread_through(self) -> None:
        captured = self._captured_loss_weights([])
        self.assertEqual(captured["value_quantiles"], 0)
        self.assertEqual(captured["value_target_search_mix"], 0.0)
        self.assertEqual(captured["value_target_search_mix_min_tiles"], 13)
        self.assertEqual(captured["loss_weights"].path_consistency, 0.0)

        captured = self._captured_loss_weights(
            [
                "--value-quantiles",
                "8",
                "--value-target-search-mix",
                "0.5",
                "--value-target-search-mix-min-tiles",
                "13",
                "--path-consistency-weight",
                "0.1",
            ]
        )
        self.assertEqual(captured["value_quantiles"], 8)
        self.assertEqual(captured["value_target_search_mix"], 0.5)
        self.assertEqual(captured["value_target_search_mix_min_tiles"], 13)
        self.assertEqual(captured["loss_weights"].path_consistency, 0.1)


class PathConsistencyTest(_TorchCase):
    """T0: value(t) vs stop-gradient search_root_value at the same seat's next
    root (t+4 in packed order), same-game guarded."""

    def test_shard_pairing_only_matches_same_game_records(self) -> None:
        import numpy as np

        from cascadiav3.torch_train_cascadiaformer import _shard_path_consistency_pairs

        # Two games of 8 records each; seats cycle 0..3; final score vectors
        # are bit-identical within a game and differ across games.
        game_a = np.tile(np.array([60.0, 70.0, 80.0, 90.0], dtype=np.float32), (8, 1))
        game_b = np.tile(np.array([61.0, 71.0, 81.0, 91.0], dtype=np.float32), (8, 1))
        final_scores = np.concatenate([game_a, game_b])
        search_root_value = np.arange(16, dtype=np.float32) * 0.5 + 50.0
        active_seat = np.tile(np.arange(4, dtype=np.int64), 4)
        targets, valid = _shard_path_consistency_pairs(
            final_scores, search_root_value, active_seat
        )
        # Records 0-3 of each game pair with records 4-7 of the SAME game;
        # records 4-7 of game A must NOT pair into game B; the tail of the
        # shard has no successor.
        expected_valid = np.array([True] * 4 + [False] * 4 + [True] * 4 + [False] * 4)
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(targets[:4], search_root_value[4:8])
        np.testing.assert_array_equal(targets[8:12], search_root_value[12:16])

    def test_shard_pairing_guards_on_active_seat_when_packed(self) -> None:
        import numpy as np

        from cascadiav3.torch_train_cascadiaformer import _shard_path_consistency_pairs

        final_scores = np.tile(np.array([60.0, 70.0, 80.0, 90.0], dtype=np.float32), (8, 1))
        search_root_value = np.arange(8, dtype=np.float32)
        broken_seats = np.array([0, 1, 2, 3, 1, 1, 2, 3], dtype=np.int64)
        _, valid = _shard_path_consistency_pairs(
            final_scores, search_root_value, broken_seats
        )
        # Record 0 pairs with record 4, but the seat does not match (0 vs 1).
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1]))

    def test_corpus_arrays_never_pair_across_shards(self) -> None:
        self._require_torch()
        import numpy as np

        from cascadiav3.torch_train_cascadiaformer import _corpus_path_consistency_arrays

        def shard(scores, srv):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                path=Path("synthetic.npz"),
                final_score_vector=scores,
                search_root_value=srv,
                active_seat=np.tile(np.arange(4, dtype=np.int64), scores.shape[0] // 4),
            )

        scores = np.tile(np.array([60.0, 70.0, 80.0, 90.0], dtype=np.float32), (8, 1))
        corpus = SimpleNamespace(
            shards=[
                shard(scores, np.arange(8, dtype=np.float32)),
                shard(scores, np.arange(8, dtype=np.float32) + 100.0),
            ]
        )
        targets, valid = _corpus_path_consistency_arrays(corpus)
        self.assertEqual(targets.shape, (16,))
        # Last 4 records of shard 0 stay unpaired even though shard 1 opens
        # with bit-identical final scores.
        self.assertEqual(valid[:8].tolist(), [True] * 4 + [False] * 4)
        self.assertEqual(valid[8:].tolist(), [True] * 4 + [False] * 4)
        self.assertEqual(targets[:4].tolist(), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(targets[8:12].tolist(), [104.0, 105.0, 106.0, 107.0])

    def test_corpus_arrays_require_search_root_value(self) -> None:
        self._require_torch()
        import numpy as np

        from cascadiav3.torch_train_cascadiaformer import _corpus_path_consistency_arrays

        corpus = SimpleNamespace(
            shards=[
                SimpleNamespace(
                    path=Path("v1.npz"),
                    final_score_vector=np.zeros((8, 4), dtype=np.float32),
                    search_root_value=None,
                    active_seat=None,
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "search_root_value"):
            _corpus_path_consistency_arrays(corpus)

    def test_loss_value_exact_and_masked_to_valid_pairs(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        batch["tokens"], batch["token_mask"] = build_phase_tokens([(0, 10), (1, 10), (2, 10)])
        batch["path_consistency_target"] = torch.tensor([61.0, 65.0, 59.0])
        batch["path_consistency_valid"] = torch.tensor([True, False, True])
        weights = LossWeights(path_consistency=0.1)
        losses = _loss_components(outputs, batch, weights)
        # Active-seat value predictions: record 0 seat 0 -> 60.0; record 2
        # seat 2 -> 65.0. Valid residuals: (60-61)^2 = 1, (65-59)^2 = 36.
        expected = (1.0 + 36.0) / 2.0
        self.assertEqual(float(losses["path_consistency"]), expected)
        # The weighted term participates in the total (float32 rounding on
        # the summed total, so almost-equal; the component itself is exact).
        weights_off = LossWeights(path_consistency=0.0)
        losses_off = _loss_components(outputs, batch, weights_off)
        self.assertAlmostEqual(
            float(losses["total"]) - float(losses_off["total"]),
            0.1 * expected,
            places=4,
        )

    def test_stop_gradient_no_grad_flows_to_next_root_side(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import (
            LossWeights,
            _loss_components,
        )

        outputs, batch = build_loss_fixture(improved_policy=True)
        batch["tokens"], batch["token_mask"] = build_phase_tokens([(0, 10), (1, 10), (2, 10)])
        value_leaf = outputs["value_vector"].clone().requires_grad_(True)
        outputs["value_vector"] = value_leaf
        target_leaf = torch.tensor([61.0, 65.0, 59.0], requires_grad=True)
        batch["path_consistency_target"] = target_leaf
        batch["path_consistency_valid"] = torch.tensor([True, True, True])
        weights = LossWeights(
            policy=0.0,
            q=0.0,
            value=0.0,
            score=0.0,
            rank=0.0,
            uncertainty=0.0,
            path_consistency=1.0,
        )
        losses = _loss_components(outputs, batch, weights)
        losses["total"].backward()
        # The t+4 side is a collated constant behind a stop-gradient: no grad.
        self.assertIsNone(target_leaf.grad)
        # Gradient reaches ONLY each record's active seat on the t side.
        grad = value_leaf.grad
        assert grad is not None
        nonzero = (grad != 0.0)
        expected = torch.zeros_like(nonzero)
        for row, seat in enumerate((0, 1, 2)):
            expected[row, seat] = True
        self.assertTrue(bool((nonzero == expected).all()))

    def test_attach_path_consistency_selects_batch_rows(self) -> None:
        self._require_torch()
        import torch

        from cascadiav3.torch_train_cascadiaformer import _attach_path_consistency

        targets = torch.arange(10, dtype=torch.float32)
        valid = torch.tensor([index % 2 == 0 for index in range(10)])
        batch: dict[str, object] = {}
        _attach_path_consistency(batch, (targets, valid), [7, 2, 4])
        self.assertEqual(batch["path_consistency_target"].tolist(), [7.0, 2.0, 4.0])
        self.assertEqual(batch["path_consistency_valid"].tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
