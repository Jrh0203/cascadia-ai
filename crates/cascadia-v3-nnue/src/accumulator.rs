use crate::{
    ActiveFeature, InferenceBackend, QuantizedEvaluation, QuantizedEvaluationTrace,
    QuantizedV3Model, Result, V3Error, V3FeatureSet, V3OwnFeatureSet,
    model::{PHASE_BUCKETS, TRANSFORM_WIDTH},
    schema::GLOBAL_BASE,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatorUndo {
    features: V3FeatureSet,
    own: Vec<i32>,
    field: Vec<i32>,
    direct: [i32; PHASE_BUCKETS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3AccumulatorStack {
    features: V3FeatureSet,
    own: Vec<i32>,
    field: Vec<i32>,
    direct: [i32; PHASE_BUCKETS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedOwnAccumulator {
    own_base: Vec<ActiveFeature>,
    own_opportunities: Vec<ActiveFeature>,
    own_wide: Vec<i32>,
    direct: [i32; PHASE_BUCKETS],
    phase_bucket: u8,
}

impl PreparedOwnAccumulator {
    pub fn own_base_features(&self) -> &[ActiveFeature] {
        &self.own_base
    }

    pub fn evaluate(
        &self,
        model: &QuantizedV3Model,
        prepared_field: &[i32],
    ) -> Result<QuantizedEvaluation> {
        model.evaluate_wide_accumulators(
            &self.own_wide,
            prepared_field,
            self.phase_bucket,
            self.direct[usize::from(self.phase_bucket)],
        )
    }

    pub fn evaluate_fork(
        &self,
        model: &QuantizedV3Model,
        next: &V3OwnFeatureSet,
        prepared_field: &[i32],
        _backend: InferenceBackend,
    ) -> Result<QuantizedEvaluation> {
        if prepared_field.len() != TRANSFORM_WIDTH {
            return Err(V3Error::InvalidFeature(
                "prepared field accumulator has the wrong width".to_owned(),
            ));
        }
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile = *PROFILE.get_or_init(|| {
            std::env::var_os("CASCADIA_V3_PROFILE_CANDIDATE_ACCUMULATOR").is_some()
        });
        let started = std::time::Instant::now();
        let base_rows = if profile {
            transition_row_count(&self.own_base, &next.own_base)
        } else {
            0
        };
        let opportunity_rows = if profile {
            transition_row_count(&self.own_opportunities, &next.own_opportunities)
        } else {
            0
        };
        let mut own_wide = [0i32; TRANSFORM_WIDTH];
        own_wide.copy_from_slice(&self.own_wide);
        apply_combined_feature_transitions_wide(
            model,
            &mut own_wide,
            &self.own_base,
            &next.own_base,
            &self.own_opportunities,
            &next.own_opportunities,
        )?;
        let transition_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let mut direct = self.direct;
        apply_direct_transition(model, &mut direct, &self.own_base, &next.own_base)?;
        let narrow_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let evaluation = model.evaluate_wide_accumulators(
            &own_wide,
            prepared_field,
            next.phase_bucket,
            direct[usize::from(next.phase_bucket)],
        )?;
        if profile {
            static PRINTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                eprintln!(
                    "V3_CANDIDATE_ACCUMULATOR_PROFILE {}",
                    serde_json::json!({
                        "base_rows": base_rows,
                        "opportunity_rows": opportunity_rows,
                        "transition_seconds": transition_seconds,
                        "narrow_seconds": narrow_seconds,
                        "dense_seconds": started.elapsed().as_secs_f64(),
                    })
                );
            }
        }
        Ok(evaluation)
    }
}

fn transition_row_count(before: &[ActiveFeature], after: &[ActiveFeature]) -> usize {
    let mut left = 0usize;
    let mut right = 0usize;
    let mut changed = 0usize;
    while left < before.len() || right < after.len() {
        match (before.get(left), after.get(right)) {
            (Some(old), Some(new)) if old.index == new.index => {
                changed += usize::from(old.count != new.count);
                left += 1;
                right += 1;
            }
            (Some(old), Some(new)) if old.index < new.index => {
                changed += 1;
                left += 1;
            }
            (Some(_), Some(_)) => {
                changed += 1;
                right += 1;
            }
            (Some(_), None) => {
                changed += 1;
                left += 1;
            }
            (None, Some(_)) => {
                changed += 1;
                right += 1;
            }
            (None, None) => break,
        }
    }
    changed
}

fn apply_combined_feature_transitions_wide(
    model: &QuantizedV3Model,
    accumulator: &mut [i32],
    before_base: &[ActiveFeature],
    after_base: &[ActiveFeature],
    before_opportunities: &[ActiveFeature],
    after_opportunities: &[ActiveFeature],
) -> Result<()> {
    apply_feature_transition_wide(model, accumulator, before_base, after_base, false)?;
    apply_feature_transition_wide(
        model,
        accumulator,
        before_opportunities,
        after_opportunities,
        true,
    )
}

fn apply_feature_transition_wide(
    model: &QuantizedV3Model,
    accumulator: &mut [i32],
    before: &[ActiveFeature],
    after: &[ActiveFeature],
    opportunity: bool,
) -> Result<()> {
    let mut left = 0usize;
    let mut right = 0usize;
    while left < before.len() || right < after.len() {
        match (before.get(left), after.get(right)) {
            (Some(old), Some(new)) if old.index == new.index => {
                let multiplier = i32::from(new.count) - i32::from(old.count);
                if multiplier != 0 {
                    add_feature_wide(model, accumulator, old.index, multiplier, opportunity)?;
                }
                left += 1;
                right += 1;
            }
            (Some(old), Some(new)) if old.index < new.index => {
                add_feature_wide(
                    model,
                    accumulator,
                    old.index,
                    -i32::from(old.count),
                    opportunity,
                )?;
                left += 1;
            }
            (Some(_), Some(new)) => {
                add_feature_wide(
                    model,
                    accumulator,
                    new.index,
                    i32::from(new.count),
                    opportunity,
                )?;
                right += 1;
            }
            (Some(old), None) => {
                add_feature_wide(
                    model,
                    accumulator,
                    old.index,
                    -i32::from(old.count),
                    opportunity,
                )?;
                left += 1;
            }
            (None, Some(new)) => {
                add_feature_wide(
                    model,
                    accumulator,
                    new.index,
                    i32::from(new.count),
                    opportunity,
                )?;
                right += 1;
            }
            (None, None) => break,
        }
    }
    Ok(())
}

fn add_feature_wide(
    model: &QuantizedV3Model,
    accumulator: &mut [i32],
    feature: u32,
    multiplier: i32,
    opportunity: bool,
) -> Result<()> {
    if opportunity {
        add_i8_row_wide(accumulator, model.opportunity_row(feature)?, multiplier);
    } else {
        add_i16_row_wide(accumulator, model.base_row(feature)?, multiplier);
    }
    Ok(())
}

fn add_features_wide(
    model: &QuantizedV3Model,
    accumulator: &mut [i32],
    features: &[ActiveFeature],
    opportunity: bool,
) -> Result<()> {
    for feature in features {
        add_feature_wide(
            model,
            accumulator,
            feature.index,
            i32::from(feature.count),
            opportunity,
        )?;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn add_i8_row_wide(target: &mut [i32], row: &[i8], multiplier: i32) {
    use std::arch::aarch64::{
        vaddq_s32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1q_s8, vld1q_s32,
        vmlaq_n_s32, vmovl_s8, vmovl_s16, vst1q_s32, vsubq_s32,
    };
    for offset in (0..TRANSFORM_WIDTH).step_by(16) {
        unsafe {
            let packed = vld1q_s8(row.as_ptr().add(offset));
            let low = vmovl_s8(vget_low_s8(packed));
            let high = vmovl_s8(vget_high_s8(packed));
            let weights = [
                vmovl_s16(vget_low_s16(low)),
                vmovl_s16(vget_high_s16(low)),
                vmovl_s16(vget_low_s16(high)),
                vmovl_s16(vget_high_s16(high)),
            ];
            for (chunk, weights) in weights.into_iter().enumerate() {
                let lane = target.as_mut_ptr().add(offset + chunk * 4);
                let current = vld1q_s32(lane);
                let updated = match multiplier {
                    1 => vaddq_s32(current, weights),
                    -1 => vsubq_s32(current, weights),
                    _ => vmlaq_n_s32(current, weights, multiplier),
                };
                vst1q_s32(lane, updated);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn add_i8_row_wide(target: &mut [i32], row: &[i8], multiplier: i32) {
    for (target, weight) in target.iter_mut().zip(row) {
        *target += i32::from(*weight) * multiplier;
    }
}

#[cfg(target_arch = "aarch64")]
fn add_i16_row_wide(target: &mut [i32], row: &[i16], multiplier: i32) {
    use std::arch::aarch64::{
        vaddq_s32, vget_high_s16, vget_low_s16, vld1q_s16, vld1q_s32, vmlaq_n_s32, vmovl_s16,
        vst1q_s32, vsubq_s32,
    };
    for offset in (0..TRANSFORM_WIDTH).step_by(8) {
        unsafe {
            let packed = vld1q_s16(row.as_ptr().add(offset));
            let weights = [
                vmovl_s16(vget_low_s16(packed)),
                vmovl_s16(vget_high_s16(packed)),
            ];
            for (chunk, weights) in weights.into_iter().enumerate() {
                let lane = target.as_mut_ptr().add(offset + chunk * 4);
                let current = vld1q_s32(lane);
                let updated = match multiplier {
                    1 => vaddq_s32(current, weights),
                    -1 => vsubq_s32(current, weights),
                    _ => vmlaq_n_s32(current, weights, multiplier),
                };
                vst1q_s32(lane, updated);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn add_i16_row_wide(target: &mut [i32], row: &[i16], multiplier: i32) {
    for (target, weight) in target.iter_mut().zip(row) {
        *target += i32::from(*weight) * multiplier;
    }
}

impl V3AccumulatorStack {
    pub fn new(
        model: &QuantizedV3Model,
        features: V3FeatureSet,
        backend: InferenceBackend,
    ) -> Result<Self> {
        features.validate()?;
        model.validate()?;
        let _ = backend;
        let mut own_wide = model
            .transformer_bias
            .iter()
            .map(|value| i32::from(*value))
            .collect::<Vec<_>>();
        let mut field_wide = own_wide.clone();
        add_features_wide(model, &mut own_wide, &features.own_base, false)?;
        add_features_wide(model, &mut own_wide, &features.own_opportunities, true)?;
        add_features_wide(model, &mut field_wide, &features.field_base, false)?;
        add_features_wide(model, &mut field_wide, &features.field_opportunities, true)?;
        let direct = direct_potential(model, &features.own_base)?;
        Ok(Self {
            features,
            own: own_wide,
            field: field_wide,
            direct,
        })
    }

    pub fn features(&self) -> &V3FeatureSet {
        &self.features
    }

    pub fn own_accumulator(&self) -> &[i32] {
        &self.own
    }

    pub fn field_accumulator(&self) -> &[i32] {
        &self.field
    }

    pub fn evaluate(&self, model: &QuantizedV3Model) -> Result<QuantizedEvaluation> {
        let phase = usize::from(self.features.phase_bucket);
        let direct = self.direct[phase];
        model.evaluate_wide_accumulators(&self.own, &self.field, self.features.phase_bucket, direct)
    }

    /// Prepare the opponent-field accumulator once for all placement siblings
    /// of a draft. Own-board candidates never mutate this shared slice.
    pub fn prepare_field_fork(
        &self,
        model: &QuantizedV3Model,
        next_opportunities: &[ActiveFeature],
        _backend: InferenceBackend,
    ) -> Result<Vec<i32>> {
        let mut field = self.field.clone();
        apply_feature_transition_wide(
            model,
            &mut field,
            &self.features.field_opportunities,
            next_opportunities,
            true,
        )?;
        Ok(field)
    }

    /// Materialize the invariant own-board work for one tile placement. Its
    /// wildlife siblings then apply only their small exact sparse delta.
    pub fn prepare_own_fork(
        &self,
        model: &QuantizedV3Model,
        next: &V3FeatureSet,
        _backend: InferenceBackend,
    ) -> Result<PreparedOwnAccumulator> {
        let mut own = self.own.clone();
        apply_combined_feature_transitions_wide(
            model,
            &mut own,
            &self.features.own_base,
            &next.own_base,
            &self.features.own_opportunities,
            &next.own_opportunities,
        )?;
        Ok(PreparedOwnAccumulator {
            own_base: next.own_base.clone(),
            own_opportunities: next.own_opportunities.clone(),
            own_wide: own,
            direct: direct_potential(model, &next.own_base)?,
            phase_bucket: next.phase_bucket,
        })
    }

    /// Evaluate one independently constructed afterstate while reusing a
    /// draft-level field accumulator. This applies exactly the same sparse own
    /// transitions as `transition_in_place` without cloning feature vectors or
    /// repeating invariant field work for every legal action.
    pub fn evaluate_own_fork(
        &self,
        model: &QuantizedV3Model,
        next: &V3FeatureSet,
        prepared_field: &[i32],
        _backend: InferenceBackend,
    ) -> Result<QuantizedEvaluation> {
        if prepared_field.len() != TRANSFORM_WIDTH {
            return Err(V3Error::InvalidFeature(
                "prepared field accumulator has the wrong width".to_owned(),
            ));
        }
        let mut own = self.own.clone();
        apply_feature_transition_wide(
            model,
            &mut own,
            &self.features.own_base,
            &next.own_base,
            false,
        )?;
        apply_feature_transition_wide(
            model,
            &mut own,
            &self.features.own_opportunities,
            &next.own_opportunities,
            true,
        )?;
        let direct = direct_potential(model, &next.own_base)?;
        model.evaluate_wide_accumulators(
            &own,
            prepared_field,
            next.phase_bucket,
            direct[usize::from(next.phase_bucket)],
        )
    }

    pub fn trace(&self, model: &QuantizedV3Model) -> Result<QuantizedEvaluationTrace> {
        let phase = usize::from(self.features.phase_bucket);
        model.trace_wide_accumulators(
            &self.own,
            &self.field,
            self.features.phase_bucket,
            self.direct[phase],
        )
    }

    /// Transition to another exact public feature set and return an atomic
    /// snapshot that restores the complete parent accumulator.
    pub fn transition(
        &mut self,
        model: &QuantizedV3Model,
        next: V3FeatureSet,
        backend: InferenceBackend,
    ) -> Result<AccumulatorUndo> {
        let undo = AccumulatorUndo {
            features: self.features.clone(),
            own: self.own.clone(),
            field: self.field.clone(),
            direct: self.direct,
        };
        self.transition_in_place(model, next, backend)?;
        Ok(undo)
    }

    /// Apply an exact sparse transition without retaining an undo snapshot.
    /// This is the allocation-bounded path for independent afterstate forks;
    /// callers clone a verified parent stack once per parallel candidate.
    pub fn transition_in_place(
        &mut self,
        model: &QuantizedV3Model,
        next: V3FeatureSet,
        backend: InferenceBackend,
    ) -> Result<()> {
        next.validate()?;
        let _ = backend;
        let mut own_wide = self.own.clone();
        let mut field_wide = self.field.clone();
        apply_combined_feature_transitions_wide(
            model,
            &mut own_wide,
            &self.features.own_base,
            &next.own_base,
            &self.features.own_opportunities,
            &next.own_opportunities,
        )?;
        apply_combined_feature_transitions_wide(
            model,
            &mut field_wide,
            &self.features.field_base,
            &next.field_base,
            &self.features.field_opportunities,
            &next.field_opportunities,
        )?;
        self.direct = direct_potential(model, &next.own_base)?;
        self.own = own_wide;
        self.field = field_wide;
        self.features = next;
        Ok(())
    }

    pub fn undo(&mut self, undo: AccumulatorUndo) {
        self.features = undo.features;
        self.own = undo.own;
        self.field = undo.field;
        self.direct = undo.direct;
    }

    pub fn verify_reconstruction(
        &self,
        model: &QuantizedV3Model,
        backend: InferenceBackend,
    ) -> Result<()> {
        let rebuilt = Self::new(model, self.features.clone(), backend)?;
        if rebuilt.own != self.own || rebuilt.field != self.field || rebuilt.direct != self.direct {
            return Err(V3Error::InvalidFeature(
                "incremental accumulator differs from full reconstruction".to_owned(),
            ));
        }
        Ok(())
    }
}

fn direct_potential(
    model: &QuantizedV3Model,
    features: &[ActiveFeature],
) -> Result<[i32; PHASE_BUCKETS]> {
    let mut direct = [0i32; PHASE_BUCKETS];
    for feature in features
        .iter()
        .filter(|feature| (feature.index as usize) < GLOBAL_BASE)
    {
        let start = feature.index as usize * PHASE_BUCKETS;
        for (phase, value) in direct.iter_mut().enumerate() {
            *value = value
                .checked_add(
                    i32::from(model.direct_potential[start + phase]) * i32::from(feature.count),
                )
                .ok_or(V3Error::AccumulatorOverflow)?;
        }
    }
    Ok(direct)
}

fn apply_direct_transition(
    model: &QuantizedV3Model,
    direct: &mut [i32; PHASE_BUCKETS],
    before: &[ActiveFeature],
    after: &[ActiveFeature],
) -> Result<()> {
    let mut left = 0usize;
    let mut right = 0usize;
    while left < before.len() || right < after.len() {
        let (feature, multiplier, advance_left, advance_right) =
            match (before.get(left), after.get(right)) {
                (Some(old), Some(new)) if old.index == new.index => (
                    old.index,
                    i32::from(new.count) - i32::from(old.count),
                    true,
                    true,
                ),
                (Some(old), Some(new)) if old.index < new.index => {
                    (old.index, -i32::from(old.count), true, false)
                }
                (Some(_), Some(new)) => (new.index, i32::from(new.count), false, true),
                (Some(old), None) => (old.index, -i32::from(old.count), true, false),
                (None, Some(new)) => (new.index, i32::from(new.count), false, true),
                (None, None) => break,
            };
        if feature as usize >= GLOBAL_BASE {
            break;
        }
        if multiplier != 0 {
            let start = feature as usize * PHASE_BUCKETS;
            for (phase, value) in direct.iter_mut().enumerate() {
                *value = value
                    .checked_add(i32::from(model.direct_potential[start + phase]) * multiplier)
                    .ok_or(V3Error::AccumulatorOverflow)?;
            }
        }
        left += usize::from(advance_left);
        right += usize::from(advance_right);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use crate::encode_public_features;

    use super::*;

    #[test]
    fn scalar_and_neon_accumulators_are_identical() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(11),
        )
        .unwrap();
        let features = encode_public_features(&game.public_state(), 0).unwrap();
        let model = QuantizedV3Model::engineering_smoke(10);
        let scalar =
            V3AccumulatorStack::new(&model, features.clone(), InferenceBackend::Scalar).unwrap();
        let neon = V3AccumulatorStack::new(&model, features, InferenceBackend::Neon).unwrap();
        assert_eq!(scalar, neon);
        assert_eq!(
            scalar.evaluate(&model).unwrap(),
            neon.evaluate(&model).unwrap()
        );
    }

    #[test]
    fn transition_and_undo_restore_parent_exactly() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(12),
        )
        .unwrap();
        let model = QuantizedV3Model::zeroed();
        let parent = encode_public_features(&game.public_state(), game.current_player()).unwrap();
        let mut stack = V3AccumulatorStack::new(&model, parent, InferenceBackend::Scalar).unwrap();
        let original = stack.clone();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let action = game.legal_turn_actions(&prelude).unwrap().remove(0);
        game.apply(&action).unwrap();
        let next = encode_public_features(&game.public_state(), 0).unwrap();
        let undo = stack
            .transition(&model, next, InferenceBackend::Scalar)
            .unwrap();
        stack
            .verify_reconstruction(&model, InferenceBackend::Scalar)
            .unwrap();
        stack.undo(undo);
        assert_eq!(stack, original);
    }

    #[test]
    fn wide_accumulator_is_exact_through_the_clipped_activation_domain() {
        let model = QuantizedV3Model::engineering_smoke(91);
        let mut own = vec![40_000i32; TRANSFORM_WIDTH];
        let mut field = vec![-40_000i32; TRANSFORM_WIDTH];
        for index in (0..TRANSFORM_WIDTH).step_by(2) {
            own[index] = -40_000;
            field[index] = 40_000;
        }
        let clipped_own = own
            .iter()
            .map(|value| (*value).clamp(0, model.scales.feature_transformer) as i16)
            .collect::<Vec<_>>();
        let clipped_field = field
            .iter()
            .map(|value| (*value).clamp(0, model.scales.feature_transformer) as i16)
            .collect::<Vec<_>>();
        assert_eq!(
            model
                .evaluate_wide_accumulators(&own, &field, 3, 17)
                .unwrap(),
            model
                .evaluate_accumulators(&clipped_own, &clipped_field, 3, 17)
                .unwrap()
        );
    }
}
