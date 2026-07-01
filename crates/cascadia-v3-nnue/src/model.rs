use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    FullOpportunitiesCatalog, Result, V3Error, V3FeatureSchemaManifest,
    schema::{BASE_FEATURE_ROWS, V3_FEATURE_SCHEMA_ID},
};

pub const TRANSFORM_WIDTH: usize = 1_024;
pub const POOL_HALF: usize = 512;
pub const PHASE_BUCKETS: usize = 8;
pub const FC0_OUTPUTS: usize = 32;
pub const FC0_NONLINEAR: usize = 31;
pub const FC1_INPUTS: usize = FC0_NONLINEAR * 2;
pub const FC1_OUTPUTS: usize = 32;
pub const MODEL_MAGIC: &[u8; 8] = b"CSV3Q01\0";
pub const MODEL_FORMAT_VERSION: u16 = 1;
pub const MODEL_ARCHITECTURE_ID: &str = "cascadia-v3-sfnnv13-radius7-1024x32x32-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct V3ModelScales {
    pub feature_transformer: i32,
    pub dense: i32,
    pub output: i32,
}

impl Default for V3ModelScales {
    fn default() -> Self {
        Self {
            feature_transformer: 256,
            dense: 64,
            output: 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBackend {
    Scalar,
    Neon,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantizedEvaluation {
    pub raw_output_units: i32,
    pub score: f32,
}

/// Exact integer intermediates used to diagnose and certify cross-backend
/// inference. This is deliberately part of the public V3 contract: a parity
/// failure must identify the first divergent layer rather than merely report a
/// different final score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedEvaluationTrace {
    pub pooled: Vec<i16>,
    pub fc0: Vec<i32>,
    pub fc1: Vec<i16>,
    pub dense_output_units: i32,
    pub skip_output_units: i32,
    pub direct_output_units: i32,
    pub evaluation: QuantizedEvaluation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V3ModelManifest {
    pub schema_version: u16,
    pub architecture_id: String,
    pub feature_schema_id: String,
    pub feature_schema_blake3: String,
    pub opportunity_catalog_blake3: String,
    pub opportunity_training_factor_rows: u32,
    pub opportunity_training_factor_blake3: String,
    pub training_factors_coalesced: bool,
    pub base_feature_rows: u32,
    pub opportunity_feature_rows: u32,
    pub transformer_width: u16,
    pub phase_buckets: u8,
    pub fc0_outputs: u8,
    pub fc1_inputs: u8,
    pub fc1_outputs: u8,
    pub scales: V3ModelScales,
    pub training_origin: String,
    pub training_run_manifest_blake3: Option<String>,
    pub checkpoint_id: String,
    pub weights_file: String,
    pub weights_blake3: String,
    pub serving_compatible: bool,
}

impl V3ModelManifest {
    pub fn validate(&self, model: &QuantizedV3Model) -> Result<()> {
        if self.schema_version != MODEL_FORMAT_VERSION
            || self.architecture_id != MODEL_ARCHITECTURE_ID
            || self.feature_schema_id != V3_FEATURE_SCHEMA_ID
            || self.base_feature_rows as usize != BASE_FEATURE_ROWS
            || self.opportunity_feature_rows as usize != FullOpportunitiesCatalog::global().len()
            || self.opportunity_training_factor_rows as usize
                != FullOpportunitiesCatalog::global().training_factor_len()
            || self.opportunity_training_factor_blake3
                != FullOpportunitiesCatalog::global().training_factor_checksum()
            || !self.training_factors_coalesced
            || self.transformer_width as usize != TRANSFORM_WIDTH
            || self.phase_buckets as usize != PHASE_BUCKETS
            || self.fc0_outputs as usize != FC0_OUTPUTS
            || self.fc1_inputs as usize != FC1_INPUTS
            || self.fc1_outputs as usize != FC1_OUTPUTS
            || self.scales != model.scales
            || !self.serving_compatible
            || self
                .training_run_manifest_blake3
                .as_ref()
                .is_some_and(|digest| {
                    digest.len() != 64 || !digest.bytes().all(|value| value.is_ascii_hexdigit())
                })
        {
            return Err(V3Error::InvalidModel(
                "model manifest architecture contract is invalid".to_owned(),
            ));
        }
        let schema = V3FeatureSchemaManifest::build()?;
        if self.feature_schema_blake3 != schema.canonical_blake3
            || self.opportunity_catalog_blake3 != schema.opportunity_catalog_blake3
        {
            return Err(V3Error::InvalidModel(
                "model manifest feature identity is invalid".to_owned(),
            ));
        }
        model.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedV3Model {
    pub scales: V3ModelScales,
    pub transformer_bias: Vec<i16>,
    pub base_transformer: Vec<i16>,
    pub opportunity_transformer: Vec<i8>,
    pub direct_potential: Vec<i16>,
    pub fc0_weights: Vec<i8>,
    pub fc0_biases: Vec<i32>,
    pub fc1_weights: Vec<i8>,
    pub fc1_biases: Vec<i32>,
    pub fc2_weights: Vec<i8>,
    pub fc2_biases: Vec<i32>,
}

impl QuantizedV3Model {
    pub fn zeroed() -> Self {
        let opportunity_rows = FullOpportunitiesCatalog::global().len();
        Self {
            scales: V3ModelScales::default(),
            transformer_bias: vec![0; TRANSFORM_WIDTH],
            base_transformer: vec![0; BASE_FEATURE_ROWS * TRANSFORM_WIDTH],
            opportunity_transformer: vec![0; opportunity_rows * TRANSFORM_WIDTH],
            direct_potential: vec![0; BASE_FEATURE_ROWS * PHASE_BUCKETS],
            fc0_weights: vec![0; PHASE_BUCKETS * TRANSFORM_WIDTH * FC0_OUTPUTS],
            fc0_biases: vec![0; PHASE_BUCKETS * FC0_OUTPUTS],
            fc1_weights: vec![0; PHASE_BUCKETS * FC1_INPUTS * FC1_OUTPUTS],
            fc1_biases: vec![0; PHASE_BUCKETS * FC1_OUTPUTS],
            fc2_weights: vec![0; PHASE_BUCKETS * FC1_OUTPUTS],
            fc2_biases: vec![0; PHASE_BUCKETS],
        }
    }

    /// Deterministic bounded weights for engineering parity and smoke tests.
    /// These weights are not a scientific training origin.
    pub fn engineering_smoke(seed: u64) -> Self {
        let mut model = Self::zeroed();
        fill_i16(&mut model.transformer_bias, seed ^ 0x01, 3);
        fill_i16(&mut model.base_transformer, seed ^ 0x02, 5);
        fill_i8(&mut model.opportunity_transformer, seed ^ 0x03, 3);
        fill_i16(&mut model.direct_potential, seed ^ 0x04, 3);
        fill_i8(&mut model.fc0_weights, seed ^ 0x05, 5);
        fill_i32(&mut model.fc0_biases, seed ^ 0x06, 31);
        fill_i8(&mut model.fc1_weights, seed ^ 0x07, 5);
        fill_i32(&mut model.fc1_biases, seed ^ 0x08, 31);
        fill_i8(&mut model.fc2_weights, seed ^ 0x09, 5);
        fill_i32(&mut model.fc2_biases, seed ^ 0x0a, 31);
        model
    }

    pub fn validate(&self) -> Result<()> {
        let opportunity_rows = FullOpportunitiesCatalog::global().len();
        let lengths = [
            (
                "transformer bias",
                self.transformer_bias.len(),
                TRANSFORM_WIDTH,
            ),
            (
                "base transformer",
                self.base_transformer.len(),
                BASE_FEATURE_ROWS * TRANSFORM_WIDTH,
            ),
            (
                "opportunity transformer",
                self.opportunity_transformer.len(),
                opportunity_rows * TRANSFORM_WIDTH,
            ),
            (
                "direct potential",
                self.direct_potential.len(),
                BASE_FEATURE_ROWS * PHASE_BUCKETS,
            ),
            (
                "fc0 weights",
                self.fc0_weights.len(),
                PHASE_BUCKETS * TRANSFORM_WIDTH * FC0_OUTPUTS,
            ),
            (
                "fc0 biases",
                self.fc0_biases.len(),
                PHASE_BUCKETS * FC0_OUTPUTS,
            ),
            (
                "fc1 weights",
                self.fc1_weights.len(),
                PHASE_BUCKETS * FC1_INPUTS * FC1_OUTPUTS,
            ),
            (
                "fc1 biases",
                self.fc1_biases.len(),
                PHASE_BUCKETS * FC1_OUTPUTS,
            ),
            (
                "fc2 weights",
                self.fc2_weights.len(),
                PHASE_BUCKETS * FC1_OUTPUTS,
            ),
            ("fc2 biases", self.fc2_biases.len(), PHASE_BUCKETS),
        ];
        for (name, actual, expected) in lengths {
            if actual != expected {
                return Err(V3Error::InvalidModel(format!(
                    "{name} has {actual} values; expected {expected}"
                )));
            }
        }
        if self.scales.feature_transformer <= 0 || self.scales.dense <= 0 || self.scales.output <= 0
        {
            return Err(V3Error::InvalidModel(
                "quantization scales must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn base_row(&self, feature: u32) -> Result<&[i16]> {
        let feature = feature as usize;
        if feature >= BASE_FEATURE_ROWS {
            return Err(V3Error::InvalidFeature(format!(
                "base feature {feature} is out of range"
            )));
        }
        let start = feature * TRANSFORM_WIDTH;
        Ok(&self.base_transformer[start..start + TRANSFORM_WIDTH])
    }

    pub fn opportunity_row(&self, feature: u32) -> Result<&[i8]> {
        let feature = feature as usize;
        let rows = FullOpportunitiesCatalog::global().len();
        if feature >= rows {
            return Err(V3Error::InvalidFeature(format!(
                "opportunity feature {feature} is out of range"
            )));
        }
        let start = feature * TRANSFORM_WIDTH;
        Ok(&self.opportunity_transformer[start..start + TRANSFORM_WIDTH])
    }

    pub fn evaluate_accumulators(
        &self,
        own: &[i16],
        field: &[i16],
        phase_bucket: u8,
        direct_potential: i32,
    ) -> Result<QuantizedEvaluation> {
        self.validate_accumulator_inputs(own, field, phase_bucket)?;
        let mut pooled = [0i16; TRANSFORM_WIDTH];
        product_pool_accumulators(own, field, self.scales.feature_transformer, &mut pooled);

        let phase = usize::from(phase_bucket);
        let fc0_weight_start = phase * TRANSFORM_WIDTH * FC0_OUTPUTS;
        let fc0_bias_start = phase * FC0_OUTPUTS;
        let mut fc0 = dense_i16_i8_32(
            &pooled,
            &self.fc0_weights[fc0_weight_start..][..TRANSFORM_WIDTH * FC0_OUTPUTS],
            &self.fc0_biases[fc0_bias_start..][..FC0_OUTPUTS],
        );
        for value in &mut fc0 {
            *value = rounded_div(*value, self.scales.dense);
        }

        let skip = fc0[FC0_OUTPUTS - 1];
        let mut fc1_input = [0i16; FC1_INPUTS];
        for index in 0..FC0_NONLINEAR {
            let clipped = fc0[index].clamp(0, self.scales.feature_transformer - 1) as i16;
            fc1_input[index] = ((i32::from(clipped) * i32::from(clipped))
                / (self.scales.feature_transformer - 1)) as i16;
            fc1_input[FC0_NONLINEAR + index] = clipped;
        }

        let fc1_weight_start = phase * FC1_INPUTS * FC1_OUTPUTS;
        let fc1_bias_start = phase * FC1_OUTPUTS;
        let fc1_wide = dense_i16_i8_32(
            &fc1_input,
            &self.fc1_weights[fc1_weight_start..][..FC1_INPUTS * FC1_OUTPUTS],
            &self.fc1_biases[fc1_bias_start..][..FC1_OUTPUTS],
        );
        let mut fc1 = [0i16; FC1_OUTPUTS];
        for (target, value) in fc1.iter_mut().zip(fc1_wide) {
            *target = rounded_div(value, self.scales.dense)
                .clamp(0, self.scales.feature_transformer - 1) as i16;
        }

        let fc2_start = phase * FC1_OUTPUTS;
        let mut output = self.fc2_biases[phase];
        for (index, activation) in fc1.iter().copied().enumerate() {
            output += i32::from(activation) * i32::from(self.fc2_weights[fc2_start + index]);
        }
        let output_divisor =
            self.scales.dense * self.scales.feature_transformer / self.scales.output;
        let dense_output_units = rounded_div(output, output_divisor);
        let skip_output_units =
            rounded_div(skip * self.scales.output, self.scales.feature_transformer);
        let raw_output_units = dense_output_units
            .checked_add(skip_output_units)
            .and_then(|value| value.checked_add(direct_potential))
            .ok_or(V3Error::AccumulatorOverflow)?;
        Ok(QuantizedEvaluation {
            raw_output_units,
            score: raw_output_units as f32 / self.scales.output as f32,
        })
    }

    /// Evaluate exact wide feature sums through the same clipped activation
    /// domain used by QAT. Cascadia has hundreds of active opportunity rows,
    /// so its pre-activation sum can legitimately exceed chess-style i16
    /// storage even though every value is clipped to [0, feature_scale] before
    /// product pooling. Clipping here is mathematically identical to the MLX
    /// graph and avoids lossy or order-dependent accumulator saturation.
    pub fn evaluate_wide_accumulators(
        &self,
        own: &[i32],
        field: &[i32],
        phase_bucket: u8,
        direct_potential: i32,
    ) -> Result<QuantizedEvaluation> {
        let (own, field) = self.clip_wide_accumulators(own, field, phase_bucket)?;
        self.evaluate_accumulators(&own, &field, phase_bucket, direct_potential)
    }

    pub fn trace_accumulators(
        &self,
        own: &[i16],
        field: &[i16],
        phase_bucket: u8,
        direct_potential: i32,
    ) -> Result<QuantizedEvaluationTrace> {
        self.validate_accumulator_inputs(own, field, phase_bucket)?;
        let mut pooled = vec![0i16; TRANSFORM_WIDTH];
        for index in 0..POOL_HALF {
            pooled[index] = product_pool(
                own[index],
                own[index + POOL_HALF],
                self.scales.feature_transformer,
            );
            pooled[index + POOL_HALF] = product_pool(
                field[index],
                field[index + POOL_HALF],
                self.scales.feature_transformer,
            );
        }
        let phase = usize::from(phase_bucket);
        let fc0_weight_start = phase * TRANSFORM_WIDTH * FC0_OUTPUTS;
        let fc0_bias_start = phase * FC0_OUTPUTS;
        let mut fc0 = [0i32; FC0_OUTPUTS];
        for (output, result) in fc0.iter_mut().enumerate() {
            let mut value = self.fc0_biases[fc0_bias_start + output];
            for (input, activation) in pooled.iter().copied().enumerate() {
                value += i32::from(activation)
                    * i32::from(self.fc0_weights[fc0_weight_start + input * FC0_OUTPUTS + output]);
            }
            *result = rounded_div(value, self.scales.dense);
        }
        let skip = fc0[FC0_OUTPUTS - 1];
        let mut fc1_input = [0i16; FC1_INPUTS];
        for index in 0..FC0_NONLINEAR {
            let clipped = fc0[index].clamp(0, self.scales.feature_transformer - 1) as i16;
            fc1_input[index] = ((i32::from(clipped) * i32::from(clipped))
                / (self.scales.feature_transformer - 1)) as i16;
            fc1_input[FC0_NONLINEAR + index] = clipped;
        }
        let fc1_weight_start = phase * FC1_INPUTS * FC1_OUTPUTS;
        let fc1_bias_start = phase * FC1_OUTPUTS;
        let mut fc1 = [0i16; FC1_OUTPUTS];
        for (output, result) in fc1.iter_mut().enumerate() {
            let mut value = self.fc1_biases[fc1_bias_start + output];
            for (input, activation) in fc1_input.iter().copied().enumerate() {
                value += i32::from(activation)
                    * i32::from(self.fc1_weights[fc1_weight_start + input * FC1_OUTPUTS + output]);
            }
            *result = rounded_div(value, self.scales.dense)
                .clamp(0, self.scales.feature_transformer - 1) as i16;
        }
        let fc2_start = phase * FC1_OUTPUTS;
        let mut output = self.fc2_biases[phase];
        for (index, activation) in fc1.iter().copied().enumerate() {
            output += i32::from(activation) * i32::from(self.fc2_weights[fc2_start + index]);
        }
        let output_divisor =
            self.scales.dense * self.scales.feature_transformer / self.scales.output;
        let dense_output_units = rounded_div(output, output_divisor);
        let skip_output_units =
            rounded_div(skip * self.scales.output, self.scales.feature_transformer);
        output = dense_output_units
            .checked_add(skip_output_units)
            .and_then(|value| value.checked_add(direct_potential))
            .ok_or(V3Error::AccumulatorOverflow)?;
        Ok(QuantizedEvaluationTrace {
            pooled,
            fc0: fc0.to_vec(),
            fc1: fc1.to_vec(),
            dense_output_units,
            skip_output_units,
            direct_output_units: direct_potential,
            evaluation: QuantizedEvaluation {
                raw_output_units: output,
                score: output as f32 / self.scales.output as f32,
            },
        })
    }

    pub fn trace_wide_accumulators(
        &self,
        own: &[i32],
        field: &[i32],
        phase_bucket: u8,
        direct_potential: i32,
    ) -> Result<QuantizedEvaluationTrace> {
        let (own, field) = self.clip_wide_accumulators(own, field, phase_bucket)?;
        self.trace_accumulators(&own, &field, phase_bucket, direct_potential)
    }

    fn clip_wide_accumulators(
        &self,
        own: &[i32],
        field: &[i32],
        phase_bucket: u8,
    ) -> Result<([i16; TRANSFORM_WIDTH], [i16; TRANSFORM_WIDTH])> {
        if own.len() != TRANSFORM_WIDTH || field.len() != TRANSFORM_WIDTH || phase_bucket >= 8 {
            return Err(V3Error::InvalidModel(
                "wide accumulator inputs do not match the V3 architecture".to_owned(),
            ));
        }
        let mut clipped_own = [0i16; TRANSFORM_WIDTH];
        let mut clipped_field = [0i16; TRANSFORM_WIDTH];
        for (target, value) in clipped_own.iter_mut().zip(own) {
            *target = (*value).clamp(0, self.scales.feature_transformer) as i16;
        }
        for (target, value) in clipped_field.iter_mut().zip(field) {
            *target = (*value).clamp(0, self.scales.feature_transformer) as i16;
        }
        Ok((clipped_own, clipped_field))
    }

    fn validate_accumulator_inputs(
        &self,
        own: &[i16],
        field: &[i16],
        phase_bucket: u8,
    ) -> Result<()> {
        if own.len() != TRANSFORM_WIDTH || field.len() != TRANSFORM_WIDTH || phase_bucket >= 8 {
            return Err(V3Error::InvalidModel(
                "accumulator width or phase bucket is invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn save_bundle(
        &self,
        directory: &Path,
        training_origin: impl Into<String>,
        checkpoint_id: impl Into<String>,
    ) -> Result<V3ModelManifest> {
        self.validate()?;
        fs::create_dir_all(directory)?;
        let weights_file = "weights.v3q";
        let weights_path = directory.join(weights_file);
        let temporary = temporary_path(&weights_path);
        let mut output = File::create(&temporary)?;
        self.write_binary(&mut output)?;
        output.sync_all()?;
        fs::rename(&temporary, &weights_path)?;
        let weights_blake3 = checksum_file(&weights_path)?;
        let schema = V3FeatureSchemaManifest::build()?;
        let manifest = V3ModelManifest {
            schema_version: MODEL_FORMAT_VERSION,
            architecture_id: MODEL_ARCHITECTURE_ID.to_owned(),
            feature_schema_id: V3_FEATURE_SCHEMA_ID.to_owned(),
            feature_schema_blake3: schema.canonical_blake3,
            opportunity_catalog_blake3: schema.opportunity_catalog_blake3,
            opportunity_training_factor_rows: schema.opportunity_training_factor_rows,
            opportunity_training_factor_blake3: schema.opportunity_training_factor_blake3,
            training_factors_coalesced: true,
            base_feature_rows: BASE_FEATURE_ROWS as u32,
            opportunity_feature_rows: FullOpportunitiesCatalog::global().len() as u32,
            transformer_width: TRANSFORM_WIDTH as u16,
            phase_buckets: PHASE_BUCKETS as u8,
            fc0_outputs: FC0_OUTPUTS as u8,
            fc1_inputs: FC1_INPUTS as u8,
            fc1_outputs: FC1_OUTPUTS as u8,
            scales: self.scales,
            training_origin: training_origin.into(),
            training_run_manifest_blake3: None,
            checkpoint_id: checkpoint_id.into(),
            weights_file: weights_file.to_owned(),
            weights_blake3,
            serving_compatible: true,
        };
        manifest.validate(self)?;
        write_json_atomic(&directory.join("model.json"), &manifest)?;
        Ok(manifest)
    }

    pub fn load_bundle(directory: &Path) -> Result<(Self, V3ModelManifest)> {
        let manifest: V3ModelManifest =
            serde_json::from_reader(File::open(directory.join("model.json"))?)?;
        let weights_path = directory.join(&manifest.weights_file);
        if checksum_file(&weights_path)? != manifest.weights_blake3 {
            return Err(V3Error::ChecksumMismatch(
                weights_path.display().to_string(),
            ));
        }
        let mut input = File::open(weights_path)?;
        let model = Self::read_binary(&mut input)?;
        manifest.validate(&model)?;
        Ok((model, manifest))
    }

    pub fn write_binary(&self, output: &mut impl Write) -> Result<()> {
        output.write_all(MODEL_MAGIC)?;
        output.write_all(&MODEL_FORMAT_VERSION.to_le_bytes())?;
        output.write_all(&(BASE_FEATURE_ROWS as u32).to_le_bytes())?;
        output.write_all(&(FullOpportunitiesCatalog::global().len() as u32).to_le_bytes())?;
        output.write_all(&(TRANSFORM_WIDTH as u16).to_le_bytes())?;
        output.write_all(&[
            PHASE_BUCKETS as u8,
            FC0_OUTPUTS as u8,
            FC1_INPUTS as u8,
            FC1_OUTPUTS as u8,
        ])?;
        for scale in [
            self.scales.feature_transformer,
            self.scales.dense,
            self.scales.output,
        ] {
            output.write_all(&scale.to_le_bytes())?;
        }
        write_i16s(output, &self.transformer_bias)?;
        write_i16s(output, &self.base_transformer)?;
        output.write_all(bytemuck_i8(&self.opportunity_transformer))?;
        write_i16s(output, &self.direct_potential)?;
        output.write_all(bytemuck_i8(&self.fc0_weights))?;
        write_i32s(output, &self.fc0_biases)?;
        output.write_all(bytemuck_i8(&self.fc1_weights))?;
        write_i32s(output, &self.fc1_biases)?;
        output.write_all(bytemuck_i8(&self.fc2_weights))?;
        write_i32s(output, &self.fc2_biases)?;
        Ok(())
    }

    pub fn read_binary(input: &mut impl Read) -> Result<Self> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MODEL_MAGIC {
            return Err(V3Error::InvalidModel("invalid V3 weight magic".to_owned()));
        }
        let version = read_u16(input)?;
        let base_rows = read_u32(input)? as usize;
        let opportunity_rows = read_u32(input)? as usize;
        let width = read_u16(input)? as usize;
        let mut architecture = [0u8; 4];
        input.read_exact(&mut architecture)?;
        if version != MODEL_FORMAT_VERSION
            || base_rows != BASE_FEATURE_ROWS
            || opportunity_rows != FullOpportunitiesCatalog::global().len()
            || width != TRANSFORM_WIDTH
            || architecture
                != [
                    PHASE_BUCKETS as u8,
                    FC0_OUTPUTS as u8,
                    FC1_INPUTS as u8,
                    FC1_OUTPUTS as u8,
                ]
        {
            return Err(V3Error::InvalidModel(
                "V3 weight header does not match the compiled architecture".to_owned(),
            ));
        }
        let scales = V3ModelScales {
            feature_transformer: read_i32(input)?,
            dense: read_i32(input)?,
            output: read_i32(input)?,
        };
        let transformer_bias = read_i16s(input, TRANSFORM_WIDTH)?;
        let base_transformer = read_i16s(input, BASE_FEATURE_ROWS * TRANSFORM_WIDTH)?;
        let opportunity_transformer = read_i8s(input, opportunity_rows * TRANSFORM_WIDTH)?;
        let direct_potential = read_i16s(input, BASE_FEATURE_ROWS * PHASE_BUCKETS)?;
        let fc0_weights = read_i8s(input, PHASE_BUCKETS * TRANSFORM_WIDTH * FC0_OUTPUTS)?;
        let fc0_biases = read_i32s(input, PHASE_BUCKETS * FC0_OUTPUTS)?;
        let fc1_weights = read_i8s(input, PHASE_BUCKETS * FC1_INPUTS * FC1_OUTPUTS)?;
        let fc1_biases = read_i32s(input, PHASE_BUCKETS * FC1_OUTPUTS)?;
        let fc2_weights = read_i8s(input, PHASE_BUCKETS * FC1_OUTPUTS)?;
        let fc2_biases = read_i32s(input, PHASE_BUCKETS)?;
        let mut trailing = [0u8; 1];
        if input.read(&mut trailing)? != 0 {
            return Err(V3Error::InvalidModel(
                "V3 weight file contains trailing bytes".to_owned(),
            ));
        }
        let model = Self {
            scales,
            transformer_bias,
            base_transformer,
            opportunity_transformer,
            direct_potential,
            fc0_weights,
            fc0_biases,
            fc1_weights,
            fc1_biases,
            fc2_weights,
            fc2_biases,
        };
        model.validate()?;
        Ok(model)
    }
}

fn product_pool_accumulators(own: &[i16], field: &[i16], scale: i32, output: &mut [i16]) {
    debug_assert_eq!(own.len(), TRANSFORM_WIDTH);
    debug_assert_eq!(field.len(), TRANSFORM_WIDTH);
    debug_assert_eq!(output.len(), TRANSFORM_WIDTH);
    #[cfg(target_arch = "aarch64")]
    if scale == 256 {
        use std::arch::aarch64::{
            vcombine_s16, vdupq_n_s16, vdupq_n_s32, vget_high_s16, vget_low_s16, vld1q_s16,
            vmaxq_s16, vminq_s16, vminq_s32, vmull_s16, vqmovn_s32, vshrq_n_s32, vst1q_s16,
        };
        unsafe {
            let zero = vdupq_n_s16(0);
            let maximum = vdupq_n_s16(256);
            let pooled_maximum = vdupq_n_s32(255);
            for (source, destination) in [(own, 0usize), (field, POOL_HALF)] {
                for offset in (0..POOL_HALF).step_by(8) {
                    let left = vminq_s16(
                        vmaxq_s16(vld1q_s16(source.as_ptr().add(offset)), zero),
                        maximum,
                    );
                    let right = vminq_s16(
                        vmaxq_s16(vld1q_s16(source.as_ptr().add(offset + POOL_HALF)), zero),
                        maximum,
                    );
                    let low = vminq_s32(
                        vshrq_n_s32::<8>(vmull_s16(vget_low_s16(left), vget_low_s16(right))),
                        pooled_maximum,
                    );
                    let high = vminq_s32(
                        vshrq_n_s32::<8>(vmull_s16(vget_high_s16(left), vget_high_s16(right))),
                        pooled_maximum,
                    );
                    vst1q_s16(
                        output.as_mut_ptr().add(destination + offset),
                        vcombine_s16(vqmovn_s32(low), vqmovn_s32(high)),
                    );
                }
            }
        }
        return;
    }
    for index in 0..POOL_HALF {
        output[index] = product_pool(own[index], own[index + POOL_HALF], scale);
        output[index + POOL_HALF] = product_pool(field[index], field[index + POOL_HALF], scale);
    }
}

#[cfg(target_arch = "aarch64")]
fn dense_i16_i8_32(inputs: &[i16], weights: &[i8], biases: &[i32]) -> [i32; 32] {
    use std::arch::aarch64::{
        int16x8_t, int32x4_t, vaddq_s32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16,
        vld1q_s8, vld1q_s32, vmovl_s8, vmull_n_s16, vst1q_s32,
    };

    debug_assert_eq!(weights.len(), inputs.len() * 32);
    debug_assert_eq!(biases.len(), 32);
    unsafe {
        let mut sums: [int32x4_t; 8] =
            std::array::from_fn(|chunk| vld1q_s32(biases.as_ptr().add(chunk * 4)));
        for (input, activation) in inputs.iter().copied().enumerate() {
            let row = weights.as_ptr().add(input * 32);
            let first = vld1q_s8(row);
            let second = vld1q_s8(row.add(16));
            let widened: [int16x8_t; 4] = [
                vmovl_s8(vget_low_s8(first)),
                vmovl_s8(vget_high_s8(first)),
                vmovl_s8(vget_low_s8(second)),
                vmovl_s8(vget_high_s8(second)),
            ];
            for (chunk, values) in widened.into_iter().enumerate() {
                sums[chunk * 2] = vaddq_s32(
                    sums[chunk * 2],
                    vmull_n_s16(vget_low_s16(values), activation),
                );
                sums[chunk * 2 + 1] = vaddq_s32(
                    sums[chunk * 2 + 1],
                    vmull_n_s16(vget_high_s16(values), activation),
                );
            }
        }
        let mut output = [0i32; 32];
        for (chunk, values) in sums.into_iter().enumerate() {
            vst1q_s32(output.as_mut_ptr().add(chunk * 4), values);
        }
        output
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dense_i16_i8_32(inputs: &[i16], weights: &[i8], biases: &[i32]) -> [i32; 32] {
    debug_assert_eq!(weights.len(), inputs.len() * 32);
    debug_assert_eq!(biases.len(), 32);
    let mut output = [0i32; 32];
    output.copy_from_slice(biases);
    for (input, activation) in inputs.iter().copied().enumerate() {
        for (value, weight) in output.iter_mut().zip(&weights[input * 32..][..32]) {
            *value += i32::from(activation) * i32::from(*weight);
        }
    }
    output
}

fn product_pool(left: i16, right: i16, scale: i32) -> i16 {
    let left = i32::from(left).clamp(0, scale);
    let right = i32::from(right).clamp(0, scale);
    ((left * right) / scale).clamp(0, scale - 1) as i16
}

fn rounded_div(value: i32, divisor: i32) -> i32 {
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

fn mix(index: usize, seed: u64) -> u64 {
    let mut value = seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fill_i8(values: &mut [i8], seed: u64, span: u64) {
    let center = (span / 2) as i8;
    for (index, value) in values.iter_mut().enumerate() {
        *value = (mix(index, seed) % span) as i8 - center;
    }
}

fn fill_i16(values: &mut [i16], seed: u64, span: u64) {
    let center = (span / 2) as i16;
    for (index, value) in values.iter_mut().enumerate() {
        *value = (mix(index, seed) % span) as i16 - center;
    }
}

fn fill_i32(values: &mut [i32], seed: u64, span: u64) {
    let center = (span / 2) as i32;
    for (index, value) in values.iter_mut().enumerate() {
        *value = (mix(index, seed) % span) as i32 - center;
    }
}

fn bytemuck_i8(values: &[i8]) -> &[u8] {
    // i8 and u8 have identical layout and alignment.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len()) }
}

fn write_i16s(output: &mut impl Write, values: &[i16]) -> Result<()> {
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i32s(output: &mut impl Write, values: &[i32]) -> Result<()> {
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_u16(input: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    input.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(input: &mut impl Read) -> Result<i32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_i8s(input: &mut impl Read, count: usize) -> Result<Vec<i8>> {
    let mut bytes = vec![0u8; count];
    input.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|value| value as i8).collect())
}

fn read_i16s(input: &mut impl Read, count: usize) -> Result<Vec<i16>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0u8; 2];
        input.read_exact(&mut bytes)?;
        values.push(i16::from_le_bytes(bytes));
    }
    Ok(values)
}

fn read_i32s(input: &mut impl Read, count: usize) -> Result<Vec<i32>> {
    (0..count).map(|_| read_i32(input)).collect()
}

fn checksum_file(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn temporary_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}", std::process::id()))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = temporary_path(path);
    let mut output = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_model_has_exact_architecture() {
        QuantizedV3Model::zeroed().validate().unwrap();
    }

    #[test]
    fn scalar_network_is_deterministic() {
        let model = QuantizedV3Model::zeroed();
        let own = vec![0i16; TRANSFORM_WIDTH];
        let field = vec![0i16; TRANSFORM_WIDTH];
        assert_eq!(
            model.evaluate_accumulators(&own, &field, 0, 0).unwrap(),
            model.evaluate_accumulators(&own, &field, 0, 0).unwrap()
        );
    }

    #[test]
    fn serving_kernel_is_bit_identical_to_diagnostic_trace() {
        let model = QuantizedV3Model::engineering_smoke(0x5eed);
        let own = (0..TRANSFORM_WIDTH)
            .map(|index| (mix(index, 17) % 513) as i16 - 128)
            .collect::<Vec<_>>();
        let field = (0..TRANSFORM_WIDTH)
            .map(|index| (mix(index, 29) % 513) as i16 - 128)
            .collect::<Vec<_>>();
        for phase in 0..PHASE_BUCKETS as u8 {
            let direct = i32::from(phase) * 13 - 41;
            assert_eq!(
                model
                    .evaluate_accumulators(&own, &field, phase, direct)
                    .unwrap(),
                model
                    .trace_accumulators(&own, &field, phase, direct)
                    .unwrap()
                    .evaluation
            );
        }
    }
}
