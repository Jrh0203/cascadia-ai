"""Frozen graph and quantization contracts shared with the V3 Rust crate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

ARCHITECTURE_ID = "cascadia-v3-sfnnv13-radius7-1024x32x32-v1"
FEATURE_SCHEMA_ID = "cascadia-v3-radius7-nnue-features-v1"
BASE_FEATURE_ROWS = 11_769
GLOBAL_BASE = 10_066
OPPORTUNITY_FEATURE_ROWS = 83_284
OPPORTUNITY_TRAINING_FACTOR_ROWS = 1_624
TRANSFORM_WIDTH = 1_024
POOL_HALF = 512
PHASE_BUCKETS = 8
FC0_OUTPUTS = 32
FC0_NONLINEAR = 31
FC1_INPUTS = 62
FC1_OUTPUTS = 32
FEATURE_SCALE = 256
DENSE_SCALE = 64
OUTPUT_SCALE = 16
MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class V3MlxConfig:
    architecture_id: str = ARCHITECTURE_ID
    feature_schema_id: str = FEATURE_SCHEMA_ID
    base_feature_rows: int = BASE_FEATURE_ROWS
    opportunity_feature_rows: int = OPPORTUNITY_FEATURE_ROWS
    opportunity_training_factor_rows: int = OPPORTUNITY_TRAINING_FACTOR_ROWS
    transform_width: int = TRANSFORM_WIDTH
    phase_buckets: int = PHASE_BUCKETS
    fc0_outputs: int = FC0_OUTPUTS
    fc1_inputs: int = FC1_INPUTS
    fc1_outputs: int = FC1_OUTPUTS
    feature_scale: int = FEATURE_SCALE
    dense_scale: int = DENSE_SCALE
    output_scale: int = OUTPUT_SCALE
    qat: bool = True

    def validate(self) -> None:
        if self.architecture_id != ARCHITECTURE_ID:
            raise ValueError("unsupported V3 architecture")
        if self.feature_schema_id != FEATURE_SCHEMA_ID:
            raise ValueError("unsupported V3 feature schema")
        if self.base_feature_rows != BASE_FEATURE_ROWS:
            raise ValueError("V3 base feature width drifted")
        if not 80_000 <= self.opportunity_feature_rows <= 95_000:
            raise ValueError("V3 opportunity catalog must contain 80K-95K rows")
        if not 1 <= self.opportunity_training_factor_rows < self.opportunity_feature_rows:
            raise ValueError("V3 opportunity training-factor width is invalid")
        if (
            self.transform_width != TRANSFORM_WIDTH
            or self.phase_buckets != PHASE_BUCKETS
            or self.fc0_outputs != FC0_OUTPUTS
            or self.fc1_inputs != FC1_INPUTS
            or self.fc1_outputs != FC1_OUTPUTS
        ):
            raise ValueError("V3 SFNNv13 layer dimensions drifted")
        if (self.feature_scale, self.dense_scale, self.output_scale) != (
            FEATURE_SCALE,
            DENSE_SCALE,
            OUTPUT_SCALE,
        ):
            raise ValueError("V3 quantization scales drifted")
        if not self.qat:
            raise ValueError("V3 requires quantization-aware training from step zero")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> V3MlxConfig:
        config = cls(**values)
        config.validate()
        return config
