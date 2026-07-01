use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    ActionPositionRecord, COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION,
    COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING,
    COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION, CounterfactualAdvantageCandidate,
    CounterfactualAdvantageDatasetManifest, CounterfactualAdvantageRecord, DatasetSplit,
    PositionRecord, read_counterfactual_advantage_shard_records,
    validate_counterfactual_advantage_dataset,
};
use cascadia_game::{
    GameConfig, GameSeed, GameState, RuleError, ScoreBreakdown, TurnAction, score_board, score_game,
};
use cascadia_provenance::checksum_file;
use cascadia_search::{HabitatCandidateLookaheadStrategy, RolloutCandidate};
use clap::ValueEnum;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CounterfactualCandidateSelectionArg {
    Nearest,
    Stratified,
}

impl CounterfactualCandidateSelectionArg {
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Nearest => COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION,
            Self::Stratified => COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION,
        }
    }
}

struct AdvantageSnapshot {
    state: GameState,
    parent: PositionRecord,
    current: ScoreBreakdown,
    public_supply: cascadia_game::PublicSupply,
    ranked: Vec<RolloutCandidate>,
    selected_index: usize,
}

pub fn collect_counterfactual_advantage_game(
    strategy: &HabitatCandidateLookaheadStrategy,
    split: DatasetSplit,
    game_index: u64,
    groups_per_game: usize,
    samples_per_candidate: usize,
    candidate_selection: CounterfactualCandidateSelectionArg,
) -> Result<Vec<CounterfactualAdvantageRecord>, Box<dyn Error>> {
    if groups_per_game == 0
        || !80usize.is_multiple_of(groups_per_game)
        || samples_per_candidate == 0
        || samples_per_candidate > 16
    {
        return Err("counterfactual-advantage collection shape is invalid".into());
    }
    let stride = 80 / groups_per_game;
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut snapshots = Vec::with_capacity(groups_per_game);
    while !game.is_game_over() {
        let turn = usize::from(game.completed_turns());
        let active_seat = game.current_player();
        let (ranked, source_action) = strategy.rank_and_select_deterministic(&game)?;
        if turn.is_multiple_of(stride) {
            if ranked.len() < 4 {
                return Err("H6 frontier contains fewer than four candidates".into());
            }
            let staged = game.preview_market_prelude(&source_action.prelude())?;
            let selected = without_prelude(source_action.clone());
            let ranked = ranked
                .into_iter()
                .map(|mut candidate| {
                    candidate.action = without_prelude(candidate.action);
                    candidate
                })
                .collect::<Vec<_>>();
            let retained = retain_candidates(&ranked, &selected, candidate_selection)?;
            snapshots.push(AdvantageSnapshot {
                state: staged.clone(),
                parent: PositionRecord::observe(&staged, game_index),
                current: score_board(&staged.boards()[active_seat], staged.config().scoring_cards),
                public_supply: staged.public_supply(),
                ranked: retained,
                selected_index: 0,
            });
        }
        game.apply(&source_action)?;
    }
    if snapshots.len() != groups_per_game || game.completed_turns() != 80 {
        return Err("counterfactual-advantage source trajectory is incomplete".into());
    }

    snapshots
        .into_iter()
        .map(|snapshot| {
            collect_snapshot(strategy, split, game_index, samples_per_candidate, snapshot)
        })
        .collect()
}

fn retain_candidates(
    ranked: &[RolloutCandidate],
    selected: &TurnAction,
    selection: CounterfactualCandidateSelectionArg,
) -> Result<Vec<RolloutCandidate>, Box<dyn Error>> {
    let selected_candidate = ranked
        .iter()
        .find(|candidate| candidate.action == *selected)
        .cloned()
        .ok_or("H6 selected action is absent from its ranked frontier")?;
    let mut remaining = Vec::with_capacity(ranked.len().saturating_sub(1));
    for candidate in ranked {
        if candidate.action != *selected
            && !remaining
                .iter()
                .any(|retained: &RolloutCandidate| retained.action == candidate.action)
        {
            remaining.push(candidate.clone());
        }
    }
    if remaining.len() < 3 {
        return Err("H6 frontier contains fewer than four distinct candidates".into());
    }
    let positions = alternative_positions(remaining.len(), selection);
    let mut retained = Vec::with_capacity(4);
    retained.push(selected_candidate);
    retained.extend(positions.map(|index| remaining[index].clone()));
    Ok(retained)
}

fn alternative_positions(
    remaining_count: usize,
    selection: CounterfactualCandidateSelectionArg,
) -> [usize; 3] {
    debug_assert!(remaining_count >= 3);
    match selection {
        CounterfactualCandidateSelectionArg::Nearest => [0, 1, 2],
        CounterfactualCandidateSelectionArg::Stratified => {
            [0, (remaining_count - 1) / 2, remaining_count - 1]
        }
    }
}

fn without_prelude(mut action: TurnAction) -> TurnAction {
    action.replace_three_of_a_kind = false;
    action.wildlife_wipes.clear();
    action
}

fn collect_snapshot(
    strategy: &HabitatCandidateLookaheadStrategy,
    split: DatasetSplit,
    game_index: u64,
    samples_per_candidate: usize,
    snapshot: AdvantageSnapshot,
) -> Result<CounterfactualAdvantageRecord, Box<dyn Error>> {
    let turn = u16::from(snapshot.parent.turn);
    let sample_seeds = (0..samples_per_candidate)
        .map(|sample_index| {
            counterfactual_advantage_sample_seed(split, game_index, turn, sample_index)
        })
        .collect::<Vec<_>>();
    let job_count = snapshot.ranked.len() * samples_per_candidate;
    let jobs = (0..job_count)
        .into_par_iter()
        .map(|job| {
            let candidate_index = job / samples_per_candidate;
            let sample_index = job % samples_per_candidate;
            let score = collect_conditioned_continuation(
                strategy,
                game_index,
                turn,
                candidate_index,
                sample_index,
                sample_seeds[sample_index],
                &snapshot,
            )?;
            Ok((candidate_index, sample_index, score))
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(std::io::Error::other)?;

    let mut samples =
        vec![vec![ScoreBreakdown::default(); samples_per_candidate]; snapshot.ranked.len()];
    for (candidate_index, sample_index, score) in jobs {
        samples[candidate_index][sample_index] = score;
    }
    let candidates = snapshot
        .ranked
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let serialized_action = serde_json::to_vec(&candidate.action)?;
            Ok(CounterfactualAdvantageCandidate::new(
                *blake3::hash(&serialized_action).as_bytes(),
                candidate.mean_leaf_score,
                candidate.leaf_score_stddev,
                ActionPositionRecord::observe(
                    &snapshot.state,
                    &candidate.action,
                    game_index,
                    u16::try_from(candidate.immediate_rank)?,
                    candidate.immediate_score,
                )?,
                &samples[index],
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    CounterfactualAdvantageRecord::new(
        counterfactual_advantage_group_id(split, game_index, turn),
        snapshot.selected_index,
        snapshot.current,
        snapshot.public_supply,
        snapshot.parent,
        &sample_seeds,
        candidates,
    )
    .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn collect_conditioned_continuation(
    strategy: &HabitatCandidateLookaheadStrategy,
    game_index: u64,
    turn: u16,
    candidate_index: usize,
    sample_index: usize,
    base_seed: GameSeed,
    snapshot: &AdvantageSnapshot,
) -> Result<ScoreBreakdown, String> {
    let mut attempt = 0u64;
    loop {
        let seed = conditioned_continuation_seed(base_seed, attempt);
        let mut continuation = snapshot.state.clone();
        continuation.redeterminize_hidden(seed);
        if continuation.public_supply() != snapshot.public_supply {
            return Err("public supply changed under hidden redetermination".to_owned());
        }
        match continuation.apply(&snapshot.ranked[candidate_index].action) {
            Ok(()) => {}
            Err(RuleError::WildlifeBagEmpty) => {
                attempt = next_conditioning_attempt(attempt)?;
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "game {game_index} source turn {turn} candidate {candidate_index} sample {sample_index} initial apply failed: {error}"
                ));
            }
        }

        let mut rejected = false;
        while !continuation.is_game_over() {
            let action = match strategy.select_action_deterministic(&continuation) {
                Ok(action) => action,
                Err(error) if error.is_unstable_market_exhaustion() => {
                    rejected = true;
                    break;
                }
                Err(error) => {
                    return Err(format!(
                        "game {game_index} source turn {turn} candidate {candidate_index} sample {sample_index} continuation turn {} selection failed: {error}",
                        continuation.completed_turns()
                    ));
                }
            };
            match continuation.apply(&action) {
                Ok(()) => {}
                Err(RuleError::WildlifeBagEmpty) => {
                    rejected = true;
                    break;
                }
                Err(error) => {
                    return Err(format!(
                        "game {game_index} source turn {turn} candidate {candidate_index} sample {sample_index} continuation turn {} apply failed: {error}",
                        continuation.completed_turns()
                    ));
                }
            }
        }
        if rejected {
            attempt = next_conditioning_attempt(attempt)?;
            continue;
        }
        return Ok(score_game(&continuation)[usize::from(snapshot.parent.active_seat)]);
    }
}

fn conditioned_continuation_seed(base_seed: GameSeed, attempt: u64) -> GameSeed {
    if attempt == 0 {
        return base_seed;
    }
    let mut hasher = Hasher::new();
    hasher.update(COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING.as_bytes());
    hasher.update(&base_seed.0);
    hasher.update(&attempt.to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn next_conditioning_attempt(attempt: u64) -> Result<u64, String> {
    attempt.checked_add(1).ok_or_else(|| {
        "stable-market trajectory rejection exhausted the deterministic seed space".to_owned()
    })
}

pub fn audit_counterfactual_advantage_dataset(
    dataset: &Path,
    manifest: &CounterfactualAdvantageDatasetManifest,
    estimator_samples: usize,
) -> Result<Value, Box<dyn Error>> {
    validate_counterfactual_advantage_dataset(dataset, manifest)?;
    let mut records = Vec::with_capacity(manifest.total_groups);
    for shard in &manifest.shards {
        records.extend(read_counterfactual_advantage_shard_records(
            dataset,
            manifest.split,
            &manifest.teacher,
            shard,
        )?);
    }
    records.sort_unstable_by_key(|record| (record.parent.game_index, record.parent.turn));
    let full_count = manifest.teacher.samples_per_candidate;
    validate_estimator_samples(estimator_samples, full_count)?;
    let full_centered = records
        .iter()
        .map(|record| centered_means(record, full_count))
        .collect::<Vec<_>>();
    let full_means = records
        .iter()
        .map(|record| candidate_means(record, full_count))
        .collect::<Vec<_>>();
    let group_ranges = full_means
        .iter()
        .map(|values| {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let group_standard_deviations = full_centered
        .iter()
        .map(|values| sample_stddev(values))
        .collect::<Vec<_>>();
    let advantage_standard_errors = records
        .iter()
        .flat_map(|record| {
            (0..4).map(|candidate_index| {
                let samples = centered_samples(record, candidate_index, full_count);
                sample_stddev(&samples) / (full_count as f64).sqrt()
            })
        })
        .collect::<Vec<_>>();
    let source_regrets = records
        .iter()
        .zip(&full_means)
        .map(|(record, values)| {
            values[best_index(values)] - values[usize::from(record.selected_index)]
        })
        .collect::<Vec<_>>();

    let prefixes = audit_prefix_counts(full_count)
        .into_iter()
        .map(|count| {
            let prefix_centered = records
                .iter()
                .map(|record| centered_means(record, count))
                .collect::<Vec<_>>();
            let prefix_means = records
                .iter()
                .map(|record| candidate_means(record, count))
                .collect::<Vec<_>>();
            let drifts = prefix_centered
                .iter()
                .zip(&full_centered)
                .flat_map(|(prefix, full)| {
                    prefix
                        .iter()
                        .zip(full)
                        .map(|(prefix, full)| (prefix - full).abs())
                })
                .collect::<Vec<_>>();
            let (pairwise_accuracy, pair_count) =
                pairwise_accuracy(&prefix_centered, &full_centered);
            let top_agreements = prefix_means
                .iter()
                .zip(&full_means)
                .filter(|(prefix, full)| best_index(prefix) == best_index(full))
                .count();
            let regrets = prefix_means
                .iter()
                .zip(&full_means)
                .map(|(prefix, full)| full[best_index(full)] - full[best_index(prefix)])
                .collect::<Vec<_>>();
            (
                count.to_string(),
                json!({
                    "centered_mean_absolute_drift": mean(&drifts),
                    "centered_p90_absolute_drift": percentile(&drifts, 0.90),
                    "centered_maximum_absolute_drift": drifts.iter().copied().fold(0.0, f64::max),
                    "centered_correlation": pearson(
                        &flatten(&prefix_centered),
                        &flatten(&full_centered),
                    ),
                    "pairwise_accuracy": pairwise_accuracy,
                    "pairwise_comparisons": pair_count,
                    "top_action_agreements": top_agreements,
                    "top_action_agreement": top_agreements as f64 / records.len() as f64,
                    "mean_top_action_regret": mean(&regrets),
                    "p90_top_action_regret": percentile(&regrets, 0.90),
                    "maximum_top_action_regret": regrets.iter().copied().fold(0.0, f64::max),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    let phase_bins = [
        ("turns-1-5", 1usize, 5usize),
        ("turns-6-10", 6, 10),
        ("turns-11-15", 11, 15),
        ("turns-16-20", 16, 20),
    ]
    .into_iter()
    .map(|(name, first, last)| {
        let indices = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                let personal_turn = usize::from(record.parent.turn) / 4 + 1;
                (first..=last).contains(&personal_turn).then_some(index)
            })
            .collect::<Vec<_>>();
        let estimator = (full_count >= estimator_samples).then(|| {
            let drifts = indices
                .iter()
                .flat_map(|index| {
                    let prefix = centered_means(&records[*index], estimator_samples);
                    prefix
                        .into_iter()
                        .zip(full_centered[*index])
                        .map(|(prefix, full)| (prefix - full).abs())
                })
                .collect::<Vec<_>>();
            let agreements = indices
                .iter()
                .filter(|index| {
                    best_index(&candidate_means(&records[**index], estimator_samples))
                        == best_index(&full_means[**index])
                })
                .count();
            json!({
                "centered_mean_absolute_drift": mean(&drifts),
                "top_action_agreement": agreements as f64 / indices.len() as f64,
            })
        });
        (
            name.to_owned(),
            json!({
                "groups": indices.len(),
                "estimator": estimator,
            }),
        )
    })
    .collect::<serde_json::Map<_, _>>();

    let collection_seconds = manifest.collection_milliseconds as f64 / 1000.0;
    let continuations_per_second = if collection_seconds > 0.0 {
        manifest.total_continuations as f64 / collection_seconds
    } else {
        0.0
    };
    let projected_seconds = if continuations_per_second > 0.0 {
        (160 * manifest.teacher.groups_per_game * 4 * estimator_samples) as f64
            / continuations_per_second
    } else {
        f64::INFINITY
    };
    let substantive =
        manifest.completed_games == 2 && manifest.teacher.groups_per_game == 16 && full_count == 16;
    let estimator_key = estimator_samples.to_string();
    let estimator = prefixes.get(&estimator_key);
    let r8 = prefixes.get("8");
    let gates = substantive.then(|| {
        let estimator_mae = metric(estimator, "centered_mean_absolute_drift", f64::INFINITY);
        let estimator_correlation = metric(estimator, "centered_correlation", 0.0);
        let estimator_pairwise = metric(estimator, "pairwise_accuracy", 0.0);
        let estimator_top_agreement = metric(estimator, "top_action_agreement", 0.0);
        let estimator_regret = metric(estimator, "mean_top_action_regret", f64::INFINITY);
        let mut gates = BTreeMap::from([
            ("integrity".to_owned(), true),
            (
                format!("r{estimator_samples}_centered_mae_at_most_0_50"),
                estimator_mae <= 0.50,
            ),
            (
                format!("r{estimator_samples}_centered_correlation_at_least_0_80"),
                estimator_correlation >= 0.80,
            ),
            (
                format!("r{estimator_samples}_pairwise_accuracy_at_least_0_80"),
                estimator_pairwise >= 0.80,
            ),
            (
                format!("r{estimator_samples}_top_agreement_at_least_0_65"),
                estimator_top_agreement >= 0.65,
            ),
            (
                format!("r{estimator_samples}_mean_regret_at_most_0_50"),
                estimator_regret <= 0.50,
            ),
            (
                "mean_group_range_at_least_1_50".to_owned(),
                mean(&group_ranges) >= 1.50,
            ),
            (
                "mean_advantage_se_at_most_0_75".to_owned(),
                mean(&advantage_standard_errors) <= 0.75,
            ),
            (
                "projected_corpus_at_most_12_hours".to_owned(),
                projected_seconds <= 12.0 * 3600.0,
            ),
        ]);
        if estimator_samples > 8 {
            let r8_top_agreement = metric(r8, "top_action_agreement", f64::INFINITY);
            let r8_regret = metric(r8, "mean_top_action_regret", f64::NEG_INFINITY);
            gates.insert(
                format!("r{estimator_samples}_top_agreement_greater_than_r8"),
                estimator_top_agreement > r8_top_agreement,
            );
            gates.insert(
                format!("r{estimator_samples}_mean_regret_no_greater_than_r8"),
                estimator_regret <= r8_regret,
            );
        }
        gates
    });
    let passed = gates
        .as_ref()
        .map(|values| values.values().all(|value| *value));
    Ok(json!({
        "schema_version": 2,
        "dataset": dataset.canonicalize()?.display().to_string(),
        "dataset_manifest_blake3": checksum_file(&dataset.join("dataset.json"))?,
        "dataset_id": manifest.dataset_id,
        "candidate_selection": manifest.teacher.candidate_selection_id(),
        "split": manifest.split.id(),
        "games": manifest.completed_games,
        "groups": records.len(),
        "candidates_per_group": 4,
        "estimator_samples": estimator_samples,
        "samples_per_candidate": full_count,
        "total_candidates": manifest.total_candidates,
        "total_continuations": manifest.total_continuations,
        "collection": {
            "seconds": collection_seconds,
            "continuations_per_second": continuations_per_second,
            "projected_160_game_estimator_seconds": projected_seconds,
            "projected_160_game_estimator_hours": projected_seconds / 3600.0,
        },
        "r16_target": {
            "mean_group_value_range": mean(&group_ranges),
            "p90_group_value_range": percentile(&group_ranges, 0.90),
            "mean_group_centered_standard_deviation": mean(&group_standard_deviations),
            "mean_centered_advantage_standard_error": mean(&advantage_standard_errors),
            "p90_centered_advantage_standard_error": percentile(&advantage_standard_errors, 0.90),
            "mean_absolute_centered_advantage": mean(
                &flatten(&full_centered).into_iter().map(f64::abs).collect::<Vec<_>>()
            ),
        },
        "h6_shallow_selection": {
            "mean_regret_to_r16_best": mean(&source_regrets),
            "p90_regret_to_r16_best": percentile(&source_regrets, 0.90),
            "exact_r16_best_agreement": source_regrets.iter().filter(|value| **value == 0.0).count()
                as f64 / records.len() as f64,
        },
        "prefixes": prefixes,
        "phase_bins": phase_bins,
        "substantive": substantive,
        "gates": gates,
        "passed": passed,
    }))
}

pub fn render_counterfactual_advantage_markdown(report: &Value) -> String {
    let estimator_samples = report["estimator_samples"].as_u64().unwrap_or(8);
    let estimator_key = estimator_samples.to_string();
    let estimator = &report["prefixes"][&estimator_key];
    let target = &report["r16_target"];
    let collection = &report["collection"];
    let reference = format!("R{}", report["samples_per_candidate"].as_u64().unwrap_or(0));
    let title = if estimator_samples == 12 {
        "R12 Rank-Stratified Estimator Audit"
    } else if report["candidate_selection"].as_str()
        == Some(COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION)
    {
        "Rank-Stratified Counterfactual Contrast Audit"
    } else {
        "Same-Decision Counterfactual Advantage Target Audit"
    };
    let gates = report["gates"].as_object();
    let failed = gates
        .into_iter()
        .flat_map(|values| values.iter())
        .filter_map(|(name, passed)| (!passed.as_bool().unwrap_or(false)).then_some(name.as_str()))
        .collect::<Vec<_>>();
    format!(
        "# {title}\n\n\
         - Dataset: `{}`\n\
         - Games / groups / candidates / continuations: {} / {} / {} / {}\n\
         - R{estimator_samples} centered MAE to {reference}: {:.4}\n\
         - R{estimator_samples} centered correlation: {:.4}\n\
         - R{estimator_samples} pairwise accuracy: {:.2}%\n\
         - R{estimator_samples} exact top-action agreement: {:.2}%\n\
         - R{estimator_samples} mean top-action regret: {:.4}\n\
         - Mean {reference} group range: {:.4}\n\
         - Mean {reference} centered-advantage SE: {:.4}\n\
         - Projected 160-game R{estimator_samples} corpus: {:.2} hours\n\
         - Failed gates: {}\n\
         - Verdict: **{}**\n",
        report["dataset_id"].as_str().unwrap_or("unknown"),
        report["games"].as_u64().unwrap_or(0),
        report["groups"].as_u64().unwrap_or(0),
        report["total_candidates"].as_u64().unwrap_or(0),
        report["total_continuations"].as_u64().unwrap_or(0),
        estimator["centered_mean_absolute_drift"]
            .as_f64()
            .unwrap_or(f64::NAN),
        estimator["centered_correlation"]
            .as_f64()
            .unwrap_or(f64::NAN),
        100.0 * estimator["pairwise_accuracy"].as_f64().unwrap_or(f64::NAN),
        100.0
            * estimator["top_action_agreement"]
                .as_f64()
                .unwrap_or(f64::NAN),
        estimator["mean_top_action_regret"]
            .as_f64()
            .unwrap_or(f64::NAN),
        target["mean_group_value_range"]
            .as_f64()
            .unwrap_or(f64::NAN),
        target["mean_centered_advantage_standard_error"]
            .as_f64()
            .unwrap_or(f64::NAN),
        collection["projected_160_game_estimator_hours"]
            .as_f64()
            .unwrap_or(f64::NAN),
        if failed.is_empty() {
            "none".to_owned()
        } else {
            failed.join(", ")
        },
        if report["passed"].as_bool() == Some(true) {
            "PASS"
        } else if report["passed"].is_null() {
            "IMPLEMENTATION SMOKE ONLY"
        } else {
            "FAIL"
        },
    )
}

pub fn write_text_atomic(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&temp, text)?;
    fs::rename(temp, path)?;
    Ok(())
}

fn counterfactual_advantage_sample_seed(
    split: DatasetSplit,
    game_index: u64,
    turn: u16,
    sample_index: usize,
) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-counterfactual-advantage-v1");
    hasher.update(split.id().as_bytes());
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    hasher.update(&(sample_index as u64).to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn counterfactual_advantage_group_id(split: DatasetSplit, game_index: u64, turn: u16) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-counterfactual-advantage-group-v1");
    hasher.update(split.id().as_bytes());
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("fixed hash"),
    ) | 1
}

fn total(components: &[u16; 11]) -> u16 {
    components.iter().sum()
}

fn candidate_means(record: &CounterfactualAdvantageRecord, count: usize) -> [f64; 4] {
    std::array::from_fn(|candidate| {
        record.candidates[candidate].sample_finals[..count]
            .iter()
            .map(|components| f64::from(total(components)))
            .sum::<f64>()
            / count as f64
    })
}

fn centered_means(record: &CounterfactualAdvantageRecord, count: usize) -> [f64; 4] {
    let values = candidate_means(record, count);
    let center = mean(&values);
    values.map(|value| value - center)
}

fn centered_samples(
    record: &CounterfactualAdvantageRecord,
    candidate_index: usize,
    count: usize,
) -> Vec<f64> {
    (0..count)
        .map(|sample| {
            let center = record
                .candidates
                .iter()
                .map(|candidate| f64::from(total(&candidate.sample_finals[sample])))
                .sum::<f64>()
                / 4.0;
            f64::from(total(
                &record.candidates[candidate_index].sample_finals[sample],
            )) - center
        })
        .collect()
}

fn flatten(values: &[[f64; 4]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect()
}

fn best_index(values: &[f64; 4]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn pairwise_accuracy(prefix: &[[f64; 4]], full: &[[f64; 4]]) -> (f64, usize) {
    let mut correct = 0;
    let mut comparisons = 0;
    for (prefix, full) in prefix.iter().zip(full) {
        for left in 0..4 {
            for right in left + 1..4 {
                let truth = (full[left] - full[right]).signum();
                if truth == 0.0 {
                    continue;
                }
                comparisons += 1;
                correct += usize::from((prefix[left] - prefix[right]).signum() == truth);
            }
        }
    }
    (
        if comparisons == 0 {
            0.0
        } else {
            correct as f64 / comparisons as f64
        },
        comparisons,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    (values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return 0.0;
    }
    let left_mean = mean(left);
    let right_mean = mean(right);
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = left - left_mean;
        let right = right - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    if left_variance == 0.0 || right_variance == 0.0 {
        0.0
    } else {
        covariance / (left_variance * right_variance).sqrt()
    }
}

fn metric(value: Option<&Value>, key: &str, fallback: f64) -> f64 {
    value
        .and_then(|value| value[key].as_f64())
        .unwrap_or(fallback)
}

fn validate_estimator_samples(
    estimator_samples: usize,
    available_samples: usize,
) -> Result<(), Box<dyn Error>> {
    if !matches!(estimator_samples, 8 | 12)
        || (available_samples >= 8 && estimator_samples > available_samples)
    {
        return Err(format!(
            "estimator samples must be R8 or R12 and fit the substantive R{available_samples} dataset"
        )
        .into());
    }
    Ok(())
}

fn audit_prefix_counts(available_samples: usize) -> Vec<usize> {
    [1, 2, 4, 8, 12, 16]
        .into_iter()
        .filter(|count| *count <= available_samples)
        .collect()
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, MarketSlot, Rotation};

    use super::*;

    #[test]
    fn shared_sample_seed_is_deterministic_and_domain_separated() {
        let left = counterfactual_advantage_sample_seed(DatasetSplit::Validation, 7, 20, 3);
        let same = counterfactual_advantage_sample_seed(DatasetSplit::Validation, 7, 20, 3);
        let other_sample = counterfactual_advantage_sample_seed(DatasetSplit::Validation, 7, 20, 4);
        let other_split = counterfactual_advantage_sample_seed(DatasetSplit::Test, 7, 20, 3);
        assert_eq!(left, same);
        assert_ne!(left, other_sample);
        assert_ne!(left, other_split);
    }

    #[test]
    fn conditioned_continuation_seed_preserves_attempt_zero_and_versions_retries() {
        let base = GameSeed::from_u64(51);
        assert_eq!(conditioned_continuation_seed(base, 0), base);
        assert_eq!(
            conditioned_continuation_seed(base, 1),
            conditioned_continuation_seed(base, 1)
        );
        assert_ne!(
            conditioned_continuation_seed(base, 1),
            conditioned_continuation_seed(base, 2)
        );
    }

    #[test]
    fn centered_values_sum_to_zero() {
        let values = [80.0, 82.0, 85.0, 81.0];
        let center = mean(&values);
        let centered = values.map(|value| value - center);
        assert!(mean(&centered).abs() < 1e-12);
        assert_eq!(best_index(&centered), 2);
    }

    #[test]
    fn estimator_sample_contract_admits_r8_and_r12_only() {
        validate_estimator_samples(8, 16).unwrap();
        validate_estimator_samples(12, 16).unwrap();
        validate_estimator_samples(8, 2).unwrap();
        assert!(validate_estimator_samples(0, 16).is_err());
        assert!(validate_estimator_samples(9, 16).is_err());
        assert!(validate_estimator_samples(12, 8).is_err());
    }

    #[test]
    fn audit_prefixes_include_r12_without_changing_legacy_prefixes() {
        assert_eq!(audit_prefix_counts(8), vec![1, 2, 4, 8]);
        assert_eq!(audit_prefix_counts(16), vec![1, 2, 4, 8, 12, 16]);
    }

    #[test]
    fn nearest_retention_keeps_selected_then_next_three_distinct_actions() {
        let (ranked, selected) = ranked_fixture();
        let retained = retain_candidates(
            &ranked,
            &selected,
            CounterfactualCandidateSelectionArg::Nearest,
        )
        .unwrap();
        assert_eq!(retained_ranks(&retained), [2, 0, 1, 3]);
    }

    #[test]
    fn stratified_retention_keeps_selected_high_median_and_low_actions() {
        let (ranked, selected) = ranked_fixture();
        let retained = retain_candidates(
            &ranked,
            &selected,
            CounterfactualCandidateSelectionArg::Stratified,
        )
        .unwrap();
        assert_eq!(retained_ranks(&retained), [2, 0, 4, 7]);
        assert_eq!(
            alternative_positions(3, CounterfactualCandidateSelectionArg::Stratified),
            [0, 1, 2]
        );
    }

    fn ranked_fixture() -> (Vec<RolloutCandidate>, TurnAction) {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(49),
        )
        .unwrap();
        let ranked = game.boards()[0]
            .frontier()
            .iter()
            .take(8)
            .enumerate()
            .map(|(rank, coord)| RolloutCandidate {
                action: TurnAction::paired(MarketSlot::ZERO, *coord, Rotation::ZERO),
                immediate_rank: rank,
                immediate_score: rank as u16,
                mean_leaf_score: rank as f64,
                leaf_score_stddev: 0.0,
            })
            .collect::<Vec<_>>();
        let selected = ranked[2].action.clone();
        (ranked, selected)
    }

    fn retained_ranks(retained: &[RolloutCandidate]) -> [usize; 4] {
        retained
            .iter()
            .map(|candidate| candidate.immediate_rank)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
