from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import v3_training_benchmark as benchmark


@dataclass(frozen=True)
class _Reference:
    mounted_path: str
    target: str


class _Store:
    def stage_file(self, path: Path, *, target: str) -> _Reference:
        return _Reference(f"{target}/{path.name}", target)


def _model(directory: Path) -> Path:
    directory.mkdir()
    weights = directory / "weights.v3q"
    weights.write_bytes(b"parent-model")
    (directory / "model.json").write_text(
        json.dumps(
            {
                "architecture_id": "cascadia-v3-sfnnv13-r7",
                "checkpoint_id": "parent",
                "weights_file": weights.name,
                "weights_blake3": benchmark._digest(weights),
                "serving_compatible": True,
            }
        )
    )
    return directory


def test_training_benchmark_builds_two_disjoint_ten_cpu_worker_payloads(
    tmp_path: Path,
) -> None:
    jobs = benchmark.build_jobs(_Store(), _model(tmp_path / "model"), games_per_worker=3_000)
    assert [job.key for job in jobs] == ["parent-benchmark-0", "parent-benchmark-1"]
    assert [job.args[job.args.index("--first-game-index") + 1] for job in jobs] == [
        "3000000000",
        "3000003000",
    ]
    assert all(job.environment["RAYON_NUM_THREADS"] == "10" for job in jobs)
    assert all("CASCADIA_MODEL_BUNDLES_JSON" in job.environment for job in jobs)
    assert all(job.application_metadata["scientific_training_eligible"] == "false" for job in jobs)
    assert all(job.inputs[0].mounted_path == "/inputs/v1/qualified-v1.bin" for job in jobs)


def test_benchmark_summary_combines_exact_histogram_and_anatomy() -> None:
    summary = benchmark._benchmark_summary(
        {
            "games": 10,
            "score_sum": 950,
            "score_090": 2,
            "score_095": 6,
            "score_100": 2,
            "wildlife_bear": 100,
            "wildlife_elk": 110,
            "terrain_forest": 60,
            "terrain_river": 70,
            "nature_tokens_sum": 30,
            "pinecones_sum": 30,
        }
    )
    assert summary["mean"] == 95
    assert (summary["p10"], summary["p50"], summary["p90"]) == (90, 95, 100)
    assert summary["wildlife_means"] == {"bear": 10, "elk": 11}
    assert summary["pinecones_mean"] == 3


def test_quantiles_use_nearest_rank_for_small_samples() -> None:
    histogram = {67: 1, 74: 1}
    assert benchmark._quantile(histogram, 0.10) == 67
    assert benchmark._quantile(histogram, 0.50) == 67
    assert benchmark._quantile(histogram, 0.90) == 74
