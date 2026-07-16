"""Default-deny and no-device-discovery tests for Rival preflight."""

import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cascadiav3.rival.preflight import (
    PermitExpectation,
    PreflightError,
    preflight_validate_only,
    validate_expectation_fixture,
    validate_future_gpu_permit,
)
from cascadiav3.rival.preflight import (
    main as preflight_main,
)
from cascadiav3.rival.schema import (
    RIVAL_GPU_PERMIT_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parent / "fixtures" / "rival" / "preflight_fixture.json"


def expectation() -> PermitExpectation:
    return PermitExpectation(
        phase="p2a_cost_probe",
        requested_accelerator="cuda",
        source_revision="revision:abc",
        source_digest=HASH_A,
        command_sha256=HASH_A,
        preregistration_sha256=HASH_A,
        requested_gpu_hours=2.0,
    )


def permit() -> dict[str, object]:
    return attach_content_hash(
        {
            "schema_id": RIVAL_GPU_PERMIT_SCHEMA_ID,
            "permit_id": "permit:test",
            "phase": "p2a_cost_probe",
            "authority": "john_explicit_gpu_authorization",
            "source_revision": "revision:abc",
            "source_digest": HASH_A,
            "command_sha256": HASH_A,
            "preregistration_sha256": HASH_A,
            "allowed_device": "cuda",
            "max_gpu_hours": 3.0,
            "issued_at": "2026-07-16T10:00:00Z",
            "expires_at": "2026-07-16T14:00:00Z",
        }
    )


def rehash(record: dict[str, object]) -> dict[str, object]:
    value = deepcopy(record)
    value.pop("content_sha256", None)
    return attach_content_hash(value)


class RivalPreflightTest(unittest.TestCase):
    def test_cli_does_not_confuse_a_preloaded_torch_module_with_its_own_import(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"torch": object()}),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                preflight_main(
                    [
                        "--fixture",
                        str(FIXTURE),
                        "--device",
                        "cpu",
                        "--validate-only",
                    ]
                ),
                0,
            )

    def test_cli_distinguishes_expected_lock_from_invalid_contract(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                preflight_main(
                    [
                        "--fixture",
                        str(FIXTURE),
                        "--device",
                        "cpu",
                        "--validate-only",
                    ]
                ),
                0,
            )
        with TemporaryDirectory() as temporary:
            duplicate = Path(temporary) / "duplicate.json"
            duplicate.write_text('{"schema_id":"x","schema_id":"y"}\n', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    preflight_main(
                        [
                            "--fixture",
                            str(duplicate),
                            "--device",
                            "cpu",
                            "--validate-only",
                        ]
                    ),
                    2,
                )

    def test_expectation_fixture_is_exact_hash_pinned_and_non_coercive(self) -> None:
        original = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(validate_expectation_fixture(original).requested_gpu_hours, 1.0)
        for value in (True, "1.0"):
            with self.subTest(value=value):
                changed = deepcopy(original)
                changed.pop("content_sha256")
                changed["requested_gpu_hours"] = value
                with self.assertRaises(RivalSchemaError):
                    validate_expectation_fixture(attach_content_hash(changed))
        changed = deepcopy(original)
        changed["requested_gpu_hours"] = 2.0
        with self.assertRaisesRegex(RivalSchemaError, "content_sha256 mismatch"):
            validate_expectation_fixture(changed)
        changed = deepcopy(original)
        changed.pop("content_sha256")
        changed["unknown"] = "forbidden"
        with self.assertRaises(RivalSchemaError):
            validate_expectation_fixture(attach_content_hash(changed))

    def test_missing_permit_is_denied_and_never_queries_device(self) -> None:
        report = preflight_validate_only(
            device="cpu", permit=None, expectation=expectation(), now=NOW
        )
        self.assertEqual(report["status"], "DENIED")
        self.assertEqual(report["reason_code"], "PERMIT_MISSING")
        self.assertFalse(report["accelerator_phase_enabled"])

    def test_cuda_mps_and_auto_are_rejected_before_permit_access(self) -> None:
        for device in ("cuda", "mps", "auto"):
            with (
                self.subTest(device=device),
                self.assertRaisesRegex(PreflightError, "only explicit --device cpu"),
            ):
                preflight_validate_only(device=device, permit=None, expectation=None, now=NOW)

    def test_even_a_valid_future_permit_cannot_unlock_pre_gpu_source(self) -> None:
        report = preflight_validate_only(
            device="cpu", permit=permit(), expectation=expectation(), now=NOW
        )
        self.assertEqual(report["status"], "DENIED")
        self.assertEqual(report["reason_code"], "PRE_GPU_PHASE_LOCKED")
        self.assertFalse(report["accelerator_phase_enabled"])

    def test_wrong_revision_command_preregistration_phase_device_budget_and_time_reject(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object], PermitExpectation, datetime]] = []
        for field in ("source_revision", "command_sha256", "preregistration_sha256"):
            changed = permit()
            changed[field] = "revision:wrong" if field == "source_revision" else HASH_B
            cases.append((field, rehash(changed), expectation(), NOW))
        changed_phase = permit()
        changed_phase["phase"] = "p3"
        cases.append(("phase", rehash(changed_phase), expectation(), NOW))
        changed_device = permit()
        changed_device["allowed_device"] = "mps"
        cases.append(("device", rehash(changed_device), expectation(), NOW))
        over_budget = expectation()
        over_budget = PermitExpectation(**{**over_budget.__dict__, "requested_gpu_hours": 4.0})
        cases.append(("budget", permit(), over_budget, NOW))
        cases.append(("expired", permit(), expectation(), datetime(2026, 7, 16, 15, 0, tzinfo=UTC)))
        cases.append(("naive-now", permit(), expectation(), datetime(2026, 7, 16, 12, 0)))
        for name, record, expected, now in cases:
            with self.subTest(name=name), self.assertRaises(RivalSchemaError):
                validate_future_gpu_permit(record, expectation=expected, now=now)

    def test_preflight_import_in_fresh_interpreter_leaves_torch_absent(self) -> None:
        env = dict(os.environ)
        command = "import sys; import cascadiav3.rival.preflight; print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
