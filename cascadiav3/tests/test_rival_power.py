"""Fault-injection tests for symbolic work, memory, and immutable power artifacts."""

import tempfile
import unittest
from pathlib import Path

from cascadiav3.rival.power import (
    NO_FINITE_HOURS,
    CertifiedStratumRange,
    HypotheticalThroughput,
    MemoryAssumption,
    PowerEnvelopeError,
    PowerEnvelopeSpec,
    build_power_envelope,
    write_power_envelope,
)
from cascadiav3.rival.schema import RivalSchemaError


def power_spec() -> PowerEnvelopeSpec:
    return PowerEnvelopeSpec(
        envelope_id="symbolic:work-model",
        source_revision="revision:cpu-only",
        certified_ranges=(CertifiedStratumRange("late", "sha256:" + "a" * 64, 2.0, 2.0),),
        candidate_count=4,
        finite_training_family_count=1,
        one_seat_family_count=1,
        certified_potential_appeals=10,
        selection_units_per_candidate=2,
        delta_game=0.05,
        n_h_grid=(4, 8),
        n_l_grid=(8,),
        covariance_grid=(0.0,),
        variance_high_assumption=1.0,
        variance_low_h_assumption=1.0,
        variance_low_l_assumption=1.0,
        target_gap_grid=(0.0, 100.0),
        activation_frequency_grid=(0.25,),
        timeout_rate_grid=(0.0,),
        practical_margin=0.25,
        target_confirmed_roots=10,
        calibration_root_requirement=10,
        throughput_assumptions=(
            HypotheticalThroughput("optimistic", 0.1, 1.0, 0.25, 0.5, 8, "synthetic"),
            HypotheticalThroughput("central", 0.2, 2.0, 0.5, 1.0, 4, "synthetic"),
            HypotheticalThroughput("pessimistic", 0.4, 4.0, 1.0, 2.0, 2, "synthetic"),
        ),
        memory_assumptions=(
            MemoryAssumption("roomy", 16.0, 1.0, 2.0, "synthetic"),
            MemoryAssumption("tight", 4.0, 1.0, 2.0, "synthetic"),
        ),
    )


class RivalPowerWorkModelTest(unittest.TestCase):
    def test_cost_scales_with_allocation_and_memory_caps_parallelism(self) -> None:
        rows = build_power_envelope(power_spec())["rows"]
        selected = [
            row
            for row in rows
            if row["throughput_scenario"] == "optimistic"
            and row["target_gap_assumption"] == 100.0
            and row["memory_scenario"] == "roomy"
        ]
        by_n_h = {row["n_h"]: row for row in selected}
        self.assertGreater(
            by_n_h[8]["hypothetical_root_work_seconds"],
            by_n_h[4]["hypothetical_root_work_seconds"],
        )
        self.assertLess(
            by_n_h[8]["roots_per_hour_assumption"],
            by_n_h[4]["roots_per_hour_assumption"],
        )
        roomy = next(
            row
            for row in rows
            if row["throughput_scenario"] == "optimistic"
            and row["target_gap_assumption"] == 100.0
            and row["memory_scenario"] == "roomy"
            and row["n_h"] == 4
        )
        tight = next(
            row
            for row in rows
            if row["throughput_scenario"] == "optimistic"
            and row["target_gap_assumption"] == 100.0
            and row["memory_scenario"] == "tight"
            and row["n_h"] == 4
        )
        self.assertGreater(
            roomy["hypothetical_effective_parallel_roots"],
            tight["hypothetical_effective_parallel_roots"],
        )
        self.assertGreater(roomy["roots_per_hour_assumption"], tight["roots_per_hour_assumption"])

    def test_nonresolving_allocations_report_no_finite_hours(self) -> None:
        rows = build_power_envelope(power_spec())["rows"]
        unresolved = [row for row in rows if row["target_gap_assumption"] == 0.0]
        self.assertTrue(unresolved)
        self.assertTrue(
            all(
                row["required_attempted_roots_assuming_independent_timeouts"] == NO_FINITE_HOURS
                and row["hypothetical_hours_not_decision_grade"] == NO_FINITE_HOURS
                for row in unresolved
            )
        )

    def test_immutable_writer_never_replaces_an_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "power.json"
            write_power_envelope(target, power_spec())
            original = target.read_bytes()
            with self.assertRaisesRegex(RivalSchemaError, "refusing to replace"):
                write_power_envelope(target, power_spec())
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(any(path.suffix == ".tmp" for path in target.parent.iterdir()))

    def test_candidate_set_must_have_a_challenger(self) -> None:
        invalid = PowerEnvelopeSpec(**{**power_spec().__dict__, "candidate_count": 1})
        with self.assertRaisesRegex(PowerEnvelopeError, "incumbent and challenger"):
            build_power_envelope(invalid)

    def test_symbolic_inputs_are_exactly_typed_canonical_and_hash_bound(self) -> None:
        original = power_spec()
        cases = (
            ({"delta_game": "0.05"}, "delta_game"),
            ({"n_h_grid": (8, 4)}, "ascending canonical order"),
            ({"covariance_grid": (0.0, 0.0)}, "unique"),
            (
                {"certified_ranges": (CertifiedStratumRange("late", "a" * 64, 2.0, 2.0),)},
                "sha256",
            ),
            (
                {"memory_assumptions": (MemoryAssumption("roomy", "16", 1.0, 2.0, "synthetic"),)},
                "memory assumptions",
            ),
            (
                {
                    "throughput_assumptions": (
                        original.throughput_assumptions[0],
                        original.throughput_assumptions[0],
                        original.throughput_assumptions[2],
                    )
                },
                "exactly one",
            ),
        )
        for changes, reason in cases:
            invalid = PowerEnvelopeSpec(**{**original.__dict__, **changes})
            with self.subTest(changes=changes), self.assertRaisesRegex(PowerEnvelopeError, reason):
                build_power_envelope(invalid)


if __name__ == "__main__":
    unittest.main()
