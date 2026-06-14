use std::collections::BTreeMap;

use cascadia_data::{
    ActionPositionRecord, DatasetSplit, PositionRecord, PublicBeamValueDatasetManifest,
    PublicBeamValueRecord, PublicBeamValueTeacherConfig,
};
use cascadia_game::{GameConfig, GameState, MarketPrelude, score_board};
use cascadia_search::{
    PublicBeamValueProbeConfig, SearchError, evaluate_public_beam_value_batches,
};
use cascadia_sim::{
    PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, select_pattern_action, strategy_rng,
};
use serde::Serialize;

use crate::ranking_data::ranking_group_id;

pub(crate) fn public_beam_value_probe_config(
    blueprint: PatternAwareConfig,
) -> Result<PublicBeamValueProbeConfig, SearchError> {
    PublicBeamValueProbeConfig {
        blueprint,
        wildlife_candidate_limit: 2,
        beam_width: 16,
        final_personal_turns: 5,
        determinizations_per_batch: 8,
        batches: 2,
    }
    .validate()
}

pub(crate) fn public_beam_value_teacher(
    blueprint: PatternAwareConfig,
) -> PublicBeamValueTeacherConfig {
    PublicBeamValueTeacherConfig {
        strategy_id: "public-beam-state-value-observability-v1-r8x2-b16-w2-20260611".to_owned(),
        trajectory_strategy_id: blueprint.strategy_id(),
        final_personal_turns: 5,
        recorded_personal_turns: vec![5, 4, 3, 2],
        determinizations_per_batch: 8,
        batches: 2,
        immediate_candidates: blueprint.immediate_candidate_limit,
        habitat_candidates: blueprint.habitat_candidate_limit,
        bear_candidates: blueprint.bear_candidate_limit,
        wildlife_candidates: 2,
        future_market_draws: blueprint.future_market_draws,
        beam_width: 16,
        seed_schema: "public-state-hash-domain-separated-v1".to_owned(),
    }
}

pub(crate) fn collect_public_beam_value_game(
    split: DatasetSplit,
    game_index: u64,
    blueprint: PatternAwareConfig,
    probe_config: PublicBeamValueProbeConfig,
) -> Result<Vec<PublicBeamValueRecord>, Box<dyn std::error::Error>> {
    let seed = split.game_seed(game_index);
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
    let mut blueprint_rngs = (0..usize::from(game.config().player_count))
        .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
        .collect::<Vec<_>>();
    let recorded_personal_turns = [5, 4, 3, 2];
    let mut records = Vec::new();
    let mut groups = 0usize;

    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let personal_turns = game.turns_remaining_for_player(active_seat);
        if recorded_personal_turns.contains(&personal_turns) {
            let values = evaluate_public_beam_value_batches(&game, probe_config)?;
            let candidate_count = u16::try_from(values.len())?;
            let group_id = ranking_group_id(split, game_index, turn, active_seat);
            let current_base_score =
                score_board(&game.boards()[active_seat], game.config().scoring_cards).base_total;
            let public_position_hash =
                *blake3::hash(&PositionRecord::observe(&game, 0).to_bytes()).as_bytes();
            for (candidate_index, value) in values.into_iter().enumerate() {
                let [batch_a_mean, batch_b_mean]: [f64; 2] =
                    value.batch_means.try_into().map_err(|_| {
                        std::io::Error::other("public beam evaluator did not return two batches")
                    })?;
                let [batch_a_stddev, batch_b_stddev]: [f64; 2] =
                    value.batch_stddevs.try_into().map_err(|_| {
                        std::io::Error::other("public beam evaluator did not return two deviations")
                    })?;
                let action_hash = *blake3::hash(&serde_json::to_vec(&value.action)?).as_bytes();
                records.push(PublicBeamValueRecord {
                    group_id,
                    candidate_index: u16::try_from(candidate_index)?,
                    candidate_count,
                    current_base_score,
                    batch_a_mean: batch_a_mean as f32,
                    batch_b_mean: batch_b_mean as f32,
                    batch_a_stddev: batch_a_stddev as f32,
                    batch_b_stddev: batch_b_stddev as f32,
                    public_position_hash,
                    action_hash,
                    input: ActionPositionRecord::observe(
                        &game,
                        &value.action,
                        game_index,
                        u16::try_from(value.immediate_rank)?,
                        value.immediate_score,
                    )?,
                });
            }
            groups += 1;
            eprintln!(
                "public beam-value probe game {game_index}: group {groups}/16, seat {active_seat}, personal turns {personal_turns}, {candidate_count} candidates"
            );
        }

        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let action =
            select_pattern_action(&game, &prelude, blueprint, &mut blueprint_rngs[active_seat])?;
        game.apply(&action)?;
    }
    if game.completed_turns() != 80 || groups != 16 {
        return Err(std::io::Error::other(format!(
            "public beam-value trajectory expected 80 decisions and 16 groups, got {} and {groups}",
            game.completed_turns()
        ))
        .into());
    }
    Ok(records)
}

#[derive(Debug, Serialize)]
struct PublicBeamValueProbeGates {
    candidate_value_correlation_minimum: f64,
    centered_advantage_correlation_minimum: f64,
    top_action_agreement_minimum: f64,
    mean_top_action_regret_maximum: f64,
    candidate_value_correlation_passed: bool,
    centered_advantage_correlation_passed: bool,
    top_action_agreement_passed: bool,
    mean_top_action_regret_passed: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PublicBeamValueProbeReport {
    protocol_id: &'static str,
    dataset_id: String,
    feature_schema: String,
    target_schema: String,
    games: usize,
    groups: usize,
    candidates: usize,
    candidate_value_correlation: f64,
    centered_advantage_correlation: f64,
    top_action_agreements: usize,
    top_action_agreement: f64,
    mean_top_action_regret: f64,
    max_top_action_regret: f64,
    mean_absolute_batch_difference: f64,
    mean_absolute_centered_difference: f64,
    mean_within_group_value_range: f64,
    mean_candidate_batch_stddev: f64,
    gates: PublicBeamValueProbeGates,
    passed: bool,
    elapsed_seconds: f64,
}

impl PublicBeamValueProbeReport {
    pub(crate) fn to_markdown(&self) -> String {
        format!(
            "# Public Beam-State Value Observability\n\n\
             - Dataset: `{}`\n\
             - Games / groups / candidates: {} / {} / {}\n\
             - Candidate-value correlation: {:.4} (gate >= {:.2})\n\
             - Centered-advantage correlation: {:.4} (gate >= {:.2})\n\
             - Top-action agreement: {}/{} = {:.2}% (gate >= {:.0}%)\n\
             - Mean top-action regret: {:.4} (gate <= {:.2})\n\
             - Maximum top-action regret: {:.4}\n\
             - Mean absolute batch difference: {:.4}\n\
             - Mean absolute centered difference: {:.4}\n\
             - Mean within-group value range: {:.4}\n\
             - Mean candidate batch standard deviation: {:.4}\n\
             - Runtime: {:.3}s\n\
             - Verdict: **{}**\n",
            self.dataset_id,
            self.games,
            self.groups,
            self.candidates,
            self.candidate_value_correlation,
            self.gates.candidate_value_correlation_minimum,
            self.centered_advantage_correlation,
            self.gates.centered_advantage_correlation_minimum,
            self.top_action_agreements,
            self.groups,
            self.top_action_agreement * 100.0,
            self.gates.top_action_agreement_minimum * 100.0,
            self.mean_top_action_regret,
            self.gates.mean_top_action_regret_maximum,
            self.max_top_action_regret,
            self.mean_absolute_batch_difference,
            self.mean_absolute_centered_difference,
            self.mean_within_group_value_range,
            self.mean_candidate_batch_stddev,
            self.elapsed_seconds,
            if self.passed { "PASS" } else { "FAIL" },
        )
    }
}

pub(crate) fn summarize_public_beam_value_probe(
    manifest: &PublicBeamValueDatasetManifest,
    records: &[PublicBeamValueRecord],
    elapsed_seconds: f64,
) -> Result<PublicBeamValueProbeReport, Box<dyn std::error::Error>> {
    if manifest.completed_games != 2 || manifest.total_groups != 32 || records.is_empty() {
        return Err(std::io::Error::other(format!(
            "frozen public beam-value probe expected 2 games and 32 groups, got {} and {}",
            manifest.completed_games, manifest.total_groups
        ))
        .into());
    }
    let mut grouped = BTreeMap::<u64, Vec<&PublicBeamValueRecord>>::new();
    for record in records {
        grouped.entry(record.group_id).or_default().push(record);
    }

    let mut batch_a = Vec::with_capacity(records.len());
    let mut batch_b = Vec::with_capacity(records.len());
    let mut centered_a = Vec::with_capacity(records.len());
    let mut centered_b = Vec::with_capacity(records.len());
    let mut absolute_batch_differences = Vec::with_capacity(records.len());
    let mut absolute_centered_differences = Vec::with_capacity(records.len());
    let mut within_group_ranges = Vec::with_capacity(grouped.len());
    let mut candidate_stddevs = Vec::with_capacity(records.len() * 2);
    let mut agreements = 0usize;
    let mut regrets = Vec::with_capacity(grouped.len());

    for group in grouped.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let mean_a = group
            .iter()
            .map(|record| f64::from(record.batch_a_mean))
            .sum::<f64>()
            / group.len() as f64;
        let mean_b = group
            .iter()
            .map(|record| f64::from(record.batch_b_mean))
            .sum::<f64>()
            / group.len() as f64;
        let top_a = group
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .batch_a_mean
                    .total_cmp(&right.1.batch_a_mean)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .expect("validated group is nonempty");
        let top_b = group
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .batch_b_mean
                    .total_cmp(&right.1.batch_b_mean)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .expect("validated group is nonempty");
        agreements += usize::from(top_a == top_b);
        let best_a = f64::from(group[top_a].batch_a_mean);
        let best_b = f64::from(group[top_b].batch_b_mean);
        regrets.push(
            ((best_b - f64::from(group[top_a].batch_b_mean))
                + (best_a - f64::from(group[top_b].batch_a_mean)))
                / 2.0,
        );
        let minimum = group
            .iter()
            .map(|record| f64::from(record.batch_a_mean + record.batch_b_mean) / 2.0)
            .min_by(f64::total_cmp)
            .expect("validated group is nonempty");
        let maximum = group
            .iter()
            .map(|record| f64::from(record.batch_a_mean + record.batch_b_mean) / 2.0)
            .max_by(f64::total_cmp)
            .expect("validated group is nonempty");
        within_group_ranges.push(maximum - minimum);

        for record in group {
            let a = f64::from(record.batch_a_mean);
            let b = f64::from(record.batch_b_mean);
            batch_a.push(a);
            batch_b.push(b);
            centered_a.push(a - mean_a);
            centered_b.push(b - mean_b);
            absolute_batch_differences.push((a - b).abs());
            absolute_centered_differences.push(((a - mean_a) - (b - mean_b)).abs());
            candidate_stddevs.push(f64::from(record.batch_a_stddev));
            candidate_stddevs.push(f64::from(record.batch_b_stddev));
        }
    }

    let candidate_value_correlation = pearson_correlation(&batch_a, &batch_b)?;
    let centered_advantage_correlation = pearson_correlation(&centered_a, &centered_b)?;
    let top_action_agreement = agreements as f64 / grouped.len() as f64;
    let mean_top_action_regret = arithmetic_mean(&regrets);
    let gates = PublicBeamValueProbeGates {
        candidate_value_correlation_minimum: 0.60,
        centered_advantage_correlation_minimum: 0.50,
        top_action_agreement_minimum: 0.50,
        mean_top_action_regret_maximum: 0.50,
        candidate_value_correlation_passed: candidate_value_correlation >= 0.60,
        centered_advantage_correlation_passed: centered_advantage_correlation >= 0.50,
        top_action_agreement_passed: top_action_agreement >= 0.50,
        mean_top_action_regret_passed: mean_top_action_regret <= 0.50,
    };
    let passed = gates.candidate_value_correlation_passed
        && gates.centered_advantage_correlation_passed
        && gates.top_action_agreement_passed
        && gates.mean_top_action_regret_passed;
    Ok(PublicBeamValueProbeReport {
        protocol_id: "cascadia-v2-public-beam-value-observability-v1",
        dataset_id: manifest.dataset_id.clone(),
        feature_schema: manifest.feature_schema.clone(),
        target_schema: manifest.target_schema.clone(),
        games: manifest.completed_games,
        groups: grouped.len(),
        candidates: records.len(),
        candidate_value_correlation,
        centered_advantage_correlation,
        top_action_agreements: agreements,
        top_action_agreement,
        mean_top_action_regret,
        max_top_action_regret: regrets
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        mean_absolute_batch_difference: arithmetic_mean(&absolute_batch_differences),
        mean_absolute_centered_difference: arithmetic_mean(&absolute_centered_differences),
        mean_within_group_value_range: arithmetic_mean(&within_group_ranges),
        mean_candidate_batch_stddev: arithmetic_mean(&candidate_stddevs),
        gates,
        passed,
        elapsed_seconds,
    })
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if left.len() != right.len() || left.len() < 2 {
        return Err(
            std::io::Error::other("Pearson correlation requires equal nontrivial samples").into(),
        );
    }
    let left_mean = arithmetic_mean(left);
    let right_mean = arithmetic_mean(right);
    let covariance = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>();
    let left_variance = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_variance = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        return Err(std::io::Error::other("Pearson correlation has zero variance").into());
    }
    Ok(covariance / denominator)
}

fn arithmetic_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
