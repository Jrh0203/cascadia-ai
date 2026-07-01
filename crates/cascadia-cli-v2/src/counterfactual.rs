use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    CounterfactualValueDatasetManifest, CounterfactualValueRecord, DatasetSplit, PositionRecord,
    read_counterfactual_value_shard_records, validate_counterfactual_value_dataset,
};
use cascadia_game::{GameConfig, GameSeed, GameState, score_board, score_game};
use cascadia_provenance::checksum_file;
use cascadia_search::HabitatCandidateLookaheadStrategy;
use rayon::prelude::*;
use serde_json::{Value, json};

pub fn collect_counterfactual_value_game(
    strategy: &HabitatCandidateLookaheadStrategy,
    split: DatasetSplit,
    game_index: u64,
    samples_per_state: usize,
) -> Result<Vec<CounterfactualValueRecord>, Box<dyn Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut snapshots = Vec::with_capacity(80);
    while !game.is_game_over() {
        let active_seat = game.current_player();
        snapshots.push((
            game.clone(),
            PositionRecord::observe(&game, game_index),
            score_board(&game.boards()[active_seat], game.config().scoring_cards),
            game.public_supply(),
        ));
        let action = strategy.select_action_deterministic(&game)?;
        game.apply(&action)?;
    }
    let factual_scores = score_game(&game);
    snapshots
        .into_iter()
        .map(|(state, position, current, public_supply)| {
            let factual_final = factual_scores[usize::from(position.active_seat)];
            let samples = (0..samples_per_state)
                .into_par_iter()
                .map(|sample_index| {
                    let seed = counterfactual_sample_seed(
                        split,
                        game_index,
                        u16::from(position.turn),
                        sample_index,
                    );
                    let mut continuation = state.clone();
                    continuation.redeterminize_hidden(seed);
                    if continuation.public_supply() != public_supply {
                        return Err("public supply changed under hidden redetermination".to_owned());
                    }
                    while !continuation.is_game_over() {
                        let action = strategy
                            .select_action_deterministic(&continuation)
                            .map_err(|error| error.to_string())?;
                        continuation
                            .apply(&action)
                            .map_err(|error| error.to_string())?;
                    }
                    Ok((
                        seed,
                        score_game(&continuation)[usize::from(position.active_seat)],
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(std::io::Error::other)?;
            CounterfactualValueRecord::new(
                position,
                current,
                factual_final,
                public_supply,
                &samples,
            )
            .map_err(|error| Box::new(error) as Box<dyn Error>)
        })
        .collect()
}

pub fn audit_counterfactual_value_dataset(
    dataset: &Path,
    manifest: &CounterfactualValueDatasetManifest,
) -> Result<Value, Box<dyn Error>> {
    validate_counterfactual_value_dataset(dataset, manifest)?;
    let mut records = Vec::with_capacity(manifest.total_records);
    for shard in &manifest.shards {
        records.extend(read_counterfactual_value_shard_records(
            dataset,
            manifest.split,
            &manifest.teacher,
            shard,
        )?);
    }
    records.sort_unstable_by_key(|record| (record.position.game_index, record.position.turn));
    let full_count = manifest.teacher.samples_per_state;
    let full_means = records
        .iter()
        .map(|record| sample_mean(record, full_count))
        .collect::<Vec<_>>();
    let standard_deviations = records
        .iter()
        .map(|record| sample_standard_deviation(record, full_count))
        .collect::<Vec<_>>();
    let standard_errors = standard_deviations
        .iter()
        .map(|value| value / (full_count as f64).sqrt())
        .collect::<Vec<_>>();
    let factual_errors = records
        .iter()
        .zip(&full_means)
        .map(|(record, mean)| (f64::from(total(&record.factual_final)) - mean).abs())
        .collect::<Vec<_>>();
    let factual_totals = records
        .iter()
        .map(|record| f64::from(total(&record.factual_final)))
        .collect::<Vec<_>>();
    let prefixes = [1, 2, 4, 8, 16]
        .into_iter()
        .filter(|count| *count <= full_count)
        .map(|count| {
            let drifts = records
                .iter()
                .zip(&full_means)
                .map(|(record, full)| (sample_mean(record, count) - full).abs())
                .collect::<Vec<_>>();
            let pairwise = pairwise_metrics(&records, count, full_count);
            (
                count.to_string(),
                json!({
                    "mean_absolute_drift": mean(&drifts),
                    "p90_absolute_drift": percentile(&drifts, 0.90),
                    "maximum_absolute_drift": drifts.iter().copied().fold(0.0, f64::max),
                    "within_round_pairwise": pairwise,
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
                let personal_turn = usize::from(record.position.turn) / 4 + 1;
                (first..=last).contains(&personal_turn).then_some(index)
            })
            .collect::<Vec<_>>();
        let errors = indices
            .iter()
            .map(|index| standard_errors[*index])
            .collect::<Vec<_>>();
        let r8_drift = (full_count >= 8).then(|| {
            mean(
                &indices
                    .iter()
                    .map(|index| (sample_mean(&records[*index], 8) - full_means[*index]).abs())
                    .collect::<Vec<_>>(),
            )
        });
        (
            name.to_owned(),
            json!({
                "states": indices.len(),
                "mean_standard_error": mean(&errors),
                "r8_mean_absolute_drift": r8_drift,
            }),
        )
    })
    .collect::<serde_json::Map<_, _>>();

    let within_variances = standard_deviations
        .iter()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    let state_mean_stddev = sample_stddev(&full_means);
    let collection_seconds = manifest.collection_milliseconds as f64 / 1000.0;
    let continuations_per_second = if collection_seconds > 0.0 {
        manifest.total_continuations as f64 / collection_seconds
    } else {
        0.0
    };
    let projected_seconds = if continuations_per_second > 0.0 {
        (256 * 80 * 8) as f64 / continuations_per_second
    } else {
        f64::INFINITY
    };
    let r1_log_loss = prefixes["1"]["within_round_pairwise"]["log_loss"]
        .as_f64()
        .unwrap_or(f64::INFINITY);
    let r8 = prefixes.get("8");
    let substantive = manifest.completed_games == 2 && full_count == 16;
    let gates = substantive.then(|| {
        let r8_mae = r8
            .and_then(|value| value["mean_absolute_drift"].as_f64())
            .unwrap_or(f64::INFINITY);
        let r8_accuracy = r8
            .and_then(|value| value["within_round_pairwise"]["accuracy"].as_f64())
            .unwrap_or(0.0);
        let r8_log_loss = r8
            .and_then(|value| value["within_round_pairwise"]["log_loss"].as_f64())
            .unwrap_or(f64::INFINITY);
        BTreeMap::from([
            ("integrity".to_owned(), true),
            (
                "mean_r16_standard_error_at_most_1_50".to_owned(),
                mean(&standard_errors) <= 1.50,
            ),
            ("r8_mae_at_most_1_25".to_owned(), r8_mae <= 1.25),
            (
                "r8_pairwise_accuracy_at_least_0_70".to_owned(),
                r8_accuracy >= 0.70,
            ),
            (
                "r8_pairwise_log_loss_below_r1".to_owned(),
                r8_log_loss < r1_log_loss,
            ),
            (
                "state_mean_stddev_at_least_2".to_owned(),
                state_mean_stddev >= 2.0,
            ),
            (
                "projected_r8_corpus_at_most_24_hours".to_owned(),
                projected_seconds <= 24.0 * 3600.0,
            ),
        ])
    });
    let passed = gates
        .as_ref()
        .map(|values| values.values().all(|value| *value));
    Ok(json!({
        "schema_version": 1,
        "dataset": dataset.canonicalize()?.display().to_string(),
        "dataset_manifest_blake3": checksum_file(&dataset.join("dataset.json"))?,
        "dataset_id": manifest.dataset_id,
        "split": manifest.split.id(),
        "games": manifest.completed_games,
        "states": records.len(),
        "samples_per_state": full_count,
        "total_continuations": manifest.total_continuations,
        "collection": {
            "seconds": collection_seconds,
            "continuations_per_second": continuations_per_second,
            "projected_256_game_r8_seconds": projected_seconds,
            "projected_256_game_r8_hours": projected_seconds / 3600.0,
        },
        "return_distribution": {
            "reference_samples": full_count,
            "mean_reference_total": mean(&full_means),
            "state_mean_standard_deviation": state_mean_stddev,
            "mean_within_state_standard_deviation": mean(&standard_deviations),
            "mean_within_state_variance": mean(&within_variances),
            "mean_standard_error": mean(&standard_errors),
            "p90_standard_error": percentile(&standard_errors, 0.90),
        },
        "factual_single_trajectory": {
            "mean_absolute_error_to_reference": mean(&factual_errors),
            "p90_absolute_error_to_reference": percentile(&factual_errors, 0.90),
            "correlation_with_reference": pearson(&factual_totals, &full_means),
        },
        "prefixes": prefixes,
        "phase_bins": phase_bins,
        "substantive": substantive,
        "gates": gates,
        "passed": passed,
    }))
}

pub fn write_json_atomic(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

fn counterfactual_sample_seed(
    split: DatasetSplit,
    game_index: u64,
    turn: u16,
    sample_index: usize,
) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-counterfactual-value-v1");
    hasher.update(split.id().as_bytes());
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    hasher.update(&(sample_index as u64).to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn total(components: &[u16; 11]) -> u16 {
    components.iter().sum()
}

fn sample_mean(record: &CounterfactualValueRecord, count: usize) -> f64 {
    record.sample_finals[..count]
        .iter()
        .map(|components| f64::from(total(components)))
        .sum::<f64>()
        / count as f64
}

fn sample_standard_deviation(record: &CounterfactualValueRecord, count: usize) -> f64 {
    if count < 2 {
        return 0.0;
    }
    let mean = sample_mean(record, count);
    let variance = record.sample_finals[..count]
        .iter()
        .map(|components| {
            let difference = f64::from(total(components)) - mean;
            difference * difference
        })
        .sum::<f64>()
        / (count - 1) as f64;
    variance.sqrt()
}

fn pairwise_metrics(
    records: &[CounterfactualValueRecord],
    prefix_count: usize,
    full_count: usize,
) -> Value {
    let mut rounds = BTreeMap::<(u64, u8), Vec<&CounterfactualValueRecord>>::new();
    for record in records {
        rounds
            .entry((record.position.game_index, record.position.turn / 4))
            .or_default()
            .push(record);
    }
    let mut correct = 0usize;
    let mut ordered = 0usize;
    let mut loss = 0.0;
    let mut pairs = 0usize;
    for round in rounds.values() {
        for left in 0..round.len() {
            for right in left + 1..round.len() {
                let target_difference =
                    sample_mean(round[left], full_count) - sample_mean(round[right], full_count);
                let predicted_difference = sample_mean(round[left], prefix_count)
                    - sample_mean(round[right], prefix_count);
                let target_probability = sigmoid(target_difference / 2.0);
                let predicted_probability =
                    sigmoid(predicted_difference / 2.0).clamp(1e-12, 1.0 - 1e-12);
                loss -= target_probability * predicted_probability.ln()
                    + (1.0 - target_probability) * (1.0 - predicted_probability).ln();
                pairs += 1;
                if target_difference != 0.0 {
                    ordered += 1;
                    correct +=
                        usize::from(target_difference.signum() == predicted_difference.signum());
                }
            }
        }
    }
    json!({
        "pairs": pairs,
        "ordered_pairs": ordered,
        "accuracy": if ordered > 0 { correct as f64 / ordered as f64 } else { 0.0 },
        "log_loss": if pairs > 0 { loss / pairs as f64 } else { 0.0 },
    })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn sample_stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    (values
        .iter()
        .map(|value| {
            let difference = value - center;
            difference * difference
        })
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
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return 0.0;
    }
    let left_mean = mean(left);
    let right_mean = mean(right);
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
    if denominator > 0.0 {
        covariance / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_seed_is_deterministic_and_domain_separated() {
        let left = counterfactual_sample_seed(DatasetSplit::Validation, 7, 11, 3);
        let right = counterfactual_sample_seed(DatasetSplit::Validation, 7, 11, 3);
        let other = counterfactual_sample_seed(DatasetSplit::Validation, 7, 11, 4);

        assert_eq!(left, right);
        assert_ne!(left, other);
    }

    #[test]
    fn percentile_interpolates_without_mutating_input() {
        let values = vec![4.0, 1.0, 3.0, 2.0];

        assert_eq!(percentile(&values, 0.5), 2.5);
        assert_eq!(values, vec![4.0, 1.0, 3.0, 2.0]);
    }
}
