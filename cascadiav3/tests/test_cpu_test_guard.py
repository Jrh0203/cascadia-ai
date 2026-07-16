"""Proof that the CPU-only test boundary runs before device imports/queries."""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from cascadiav3.cpu_test_guard import (
    CASCADIA_DEVICE_ENV,
    CPU_ONLY_TESTS_ENV,
    CUDA_VISIBLE_DEVICES_ENV,
    CpuOnlyTestViolation,
    assert_cpu_only_test_environment,
    cpu_only_tests_enabled,
    require_cpu_test_device,
)


class CpuOnlyTestGuardTest(unittest.TestCase):
    def test_enablement_is_strict_and_fail_closed(self) -> None:
        self.assertFalse(cpu_only_tests_enabled({}))
        self.assertFalse(cpu_only_tests_enabled({CPU_ONLY_TESTS_ENV: "0"}))
        self.assertTrue(cpu_only_tests_enabled({CPU_ONLY_TESTS_ENV: "1"}))
        with self.assertRaisesRegex(ValueError, "must be unset, 0, or 1"):
            cpu_only_tests_enabled({CPU_ONLY_TESTS_ENV: "true"})

    def test_cpu_allowed_but_cuda_mps_and_auto_rejected(self) -> None:
        environ = {CPU_ONLY_TESTS_ENV: "1"}
        self.assertEqual(require_cpu_test_device(" CPU ", environ=environ), "cpu")
        for requested in ("cuda", "mps", "auto", "cuda:0"):
            with self.subTest(requested=requested), self.assertRaises(CpuOnlyTestViolation):
                require_cpu_test_device(requested, environ=environ)

    def test_disabled_guard_is_inert(self) -> None:
        self.assertEqual(require_cpu_test_device("CUDA", environ={}), "CUDA")
        self.assertEqual(require_cpu_test_device(" auto ", environ={}), " auto ")

    def test_validation_environment_requires_all_cpu_controls(self) -> None:
        valid = {
            CPU_ONLY_TESTS_ENV: "1",
            CASCADIA_DEVICE_ENV: "cpu",
            CUDA_VISIBLE_DEVICES_ENV: "",
        }
        self.assertIsNone(assert_cpu_only_test_environment(valid))
        invalid_cases = {
            "missing test boundary": {
                CASCADIA_DEVICE_ENV: "cpu",
                CUDA_VISIBLE_DEVICES_ENV: "",
            },
            "implicit device": {
                CPU_ONLY_TESTS_ENV: "1",
                CUDA_VISIBLE_DEVICES_ENV: "",
            },
            "auto device": {
                **valid,
                CASCADIA_DEVICE_ENV: "auto",
            },
            "implicit CUDA visibility": {
                CPU_ONLY_TESTS_ENV: "1",
                CASCADIA_DEVICE_ENV: "cpu",
            },
            "visible CUDA ordinal": {
                **valid,
                CUDA_VISIBLE_DEVICES_ENV: "0",
            },
        }
        for label, environment in invalid_cases.items():
            with self.subTest(label=label), self.assertRaises(CpuOnlyTestViolation):
                assert_cpu_only_test_environment(environment)

    def test_bridge_probe_rejects_before_torch_import(self) -> None:
        from cascadiav3 import torch_bridge_throughput_probe as probe

        original_import = builtins.__import__
        imported_torch: list[str] = []

        def audit_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch" or name.startswith("torch."):
                imported_torch.append(name)
                raise AssertionError("Torch import crossed CPU-only guard")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.dict(os.environ, {CPU_ONLY_TESTS_ENV: "1"}, clear=False),
            mock.patch.object(builtins, "__import__", audit_import),
            self.assertRaises(CpuOnlyTestViolation),
        ):
            probe.run_probe(
                manifest_path=Path("never-read.json"),
                roots_path=Path("never-read.jsonl"),
                batch_sizes=[1],
                warmup_iterations=1,
                measured_iterations=2,
                device_name="cuda",
                arms=["eager"],
                source_revision=None,
            )
        self.assertEqual(imported_torch, [])

    def test_every_guarded_bridge_and_benchmark_entry_rejects_before_torch(self) -> None:
        from cascadiav3 import torch_inference_bridge as bridge
        from cascadiav3 import torch_model_throughput_benchmark as benchmark

        guarded_calls = {
            "load_model": lambda: bridge._load_model(
                Path("never-read.pt"),
                manifest_path=None,
                manifest_payload=None,
                device_name="cuda",
            ),
            "model_eval_batch": lambda: bridge._model_eval_batch(
                None,
                [],
                device_name="mps",
            ),
            "serve": lambda: bridge.serve(
                checkpoint=None,
                manifest=None,
                allow_dry_run_fallback=True,
                device_name="auto",
            ),
            "benchmark_sync": lambda: benchmark._sync_device("cuda"),
            "benchmark_clear": lambda: benchmark._clear_device("mps"),
            "benchmark_run": lambda: benchmark.run_benchmark(
                roots_path=Path("never-read.jsonl"),
                manifests=[],
                synthetic_model_sizes=[],
                batch_sizes=[1],
                warmup_iterations=1,
                measured_iterations=1,
                device_name="cuda:0",
                baseline_label=None,
                source_revision=None,
            ),
        }
        original_import = builtins.__import__
        for label, call in guarded_calls.items():
            imported_torch: list[str] = []

            def audit_import(  # type: ignore[no-untyped-def]
                name,
                *args,
                _imports=imported_torch,
                **kwargs,
            ):
                if name == "torch" or name.startswith("torch."):
                    _imports.append(name)
                    raise AssertionError("Torch import crossed CPU-only guard")
                return original_import(name, *args, **kwargs)

            with (
                self.subTest(label=label),
                mock.patch.dict(os.environ, {CPU_ONLY_TESTS_ENV: "1"}, clear=False),
                mock.patch.object(builtins, "__import__", audit_import),
                self.assertRaises(CpuOnlyTestViolation),
            ):
                call()
            self.assertEqual(imported_torch, [])

    def test_trainer_rejects_auto_before_torch_import(self) -> None:
        from cascadiav3.torch_train_cascadiaformer import run_training

        original_import = builtins.__import__
        imported_torch: list[str] = []

        def audit_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch" or name.startswith("torch."):
                imported_torch.append(name)
                raise AssertionError("Torch import crossed CPU-only guard")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.dict(os.environ, {CPU_ONLY_TESTS_ENV: "1"}, clear=False),
            mock.patch.object(builtins, "__import__", audit_import),
            self.assertRaises(CpuOnlyTestViolation),
        ):
            run_training(
                [],
                [],
                train_format="npz",
                val_format="npz",
                model_size="tiny",
                steps=1,
                batch_size=1,
                lr=1.0e-3,
                weight_decay=0.0,
                device_name="auto",
                seed=1,
                grad_accum=1,
                warmup_fraction=0.1,
                checkpoint_dir=Path("never-written"),
                metrics_jsonl=Path("never-written.jsonl"),
                out=Path("never-written.json"),
                overfit_one_batch=False,
                val_max_batches=1,
                swa_fraction=0.5,
            )
        self.assertEqual(imported_torch, [])

    def test_held_rival_trainer_path_rejects_before_torch_or_filesystem(self) -> None:
        from cascadiav3.rival.training_view import RivalTrainerIntegrationHeld
        from cascadiav3.torch_train_cascadiaformer import run_training

        original_import = builtins.__import__
        imported_torch: list[str] = []

        def audit_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch" or name.startswith("torch."):
                imported_torch.append(name)
                raise AssertionError("Torch import crossed held Rival trainer boundary")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        CPU_ONLY_TESTS_ENV: "1",
                        CASCADIA_DEVICE_ENV: "cpu",
                        CUDA_VISIBLE_DEVICES_ENV: "",
                    },
                    clear=False,
                ),
                mock.patch.object(builtins, "__import__", audit_import),
                self.assertRaises(RivalTrainerIntegrationHeld),
            ):
                run_training(
                    [],
                    [],
                    train_format="npz",
                    val_format="npz",
                    model_size="tiny",
                    steps=1,
                    batch_size=1,
                    lr=1.0e-3,
                    weight_decay=0.0,
                    device_name="cpu",
                    seed=1,
                    grad_accum=1,
                    warmup_fraction=0.1,
                    checkpoint_dir=root / "checkpoints",
                    metrics_jsonl=root / "metrics.jsonl",
                    out=root / "report.json",
                    overfit_one_batch=False,
                    val_max_batches=1,
                    swa_fraction=0.5,
                    rival_preference_training=True,
                )
            self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(imported_torch, [])

    def test_held_rival_cli_is_wired_in_a_fresh_torch_audited_process(self) -> None:
        source = textwrap.dedent(
            """
            import builtins
            import sys
            from pathlib import Path

            original_import = builtins.__import__

            def audit_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError(f"Torch import crossed held boundary: {name}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = audit_import
            from cascadiav3.rival.training_view import RivalTrainerIntegrationHeld
            from cascadiav3.torch_train_cascadiaformer import main

            root = Path(sys.argv[1])
            sys.argv = [
                "torch_train_cascadiaformer",
                "--train", "never-read.npz",
                "--val", "never-read.npz",
                "--out", str(root / "report.json"),
                "--metrics-jsonl", str(root / "metrics.jsonl"),
                "--checkpoint-dir", str(root / "checkpoints"),
                "--rival-preference-training",
            ]
            try:
                main()
            except RivalTrainerIntegrationHeld:
                pass
            else:
                raise AssertionError("held CLI flag did not fail closed")
            if list(root.iterdir()):
                raise AssertionError("held CLI path mutated the filesystem")
            print("FRESH_RIVAL_HOLD_OK")
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment.update(
                {
                    CPU_ONLY_TESTS_ENV: "1",
                    CASCADIA_DEVICE_ENV: "cpu",
                    CUDA_VISIBLE_DEVICES_ENV: "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, "-c", source, directory],
                check=True,
                capture_output=True,
                env=environment,
                text=True,
            )
        self.assertEqual(completed.stdout.strip(), "FRESH_RIVAL_HOLD_OK")

    def test_pre_gpu_validator_has_no_accelerator_probe_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "cascadiav3/scripts/validate_rival_pre_gpu.sh"
        source = script.read_text(encoding="utf-8")
        for forbidden in (
            "nvidia-smi",
            "system_profiler",
            "torch.cuda",
            "torch.backends.mps",
            "torch.mps",
            "is_available(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
