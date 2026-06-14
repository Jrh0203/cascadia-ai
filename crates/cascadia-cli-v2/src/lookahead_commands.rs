use std::time::Instant;

use cascadia_eval::{summarize_match_results, summarize_paired_match_results};
use cascadia_game::{GameConfig, GameSeed};
use cascadia_search::{
    BearCandidateLookaheadConfig, BearCandidateLookaheadStrategy,
    BearHabitatCandidateLookaheadConfig, BearHabitatCandidateLookaheadStrategy,
    DeterminizedLookaheadConfig, DeterminizedLookaheadStrategy, HabitatCandidateLookaheadConfig,
    HabitatCandidateLookaheadStrategy, NatureWipeLookaheadConfig, NatureWipeLookaheadStrategy,
    PatternBlueprintLookaheadConfig, PatternBlueprintLookaheadStrategy,
};
use cascadia_sim::{MatchConfig, PatternAwareConfig, StrategyKind, play_match};

use crate::candidate_recall::run_lookahead_recall;
use crate::cli::Command;
use crate::report::{ReportContext, write_report};

pub fn run(
    command: Command,
    report_context: &ReportContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::LookaheadBenchmark {
            games,
            first_seed,
            candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("lookahead-benchmark requires at least one game".into());
            }
            let strategy = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: candidates,
                determinizations,
                greedy_plies,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let matches = (0..games)
                .map(|index| {
                    strategy.play_match(game_config, GameSeed::from_u64(first_seed + index as u64))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_match_results(
                strategy.strategy_id(),
                games,
                first_seed,
                &matches,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::LookaheadCompare {
            games,
            first_seed,
            baseline,
            candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("lookahead-compare requires at least one game".into());
            }
            let baseline: StrategyKind = baseline.into();
            let strategy = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: candidates,
                determinizations,
                greedy_plies,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result =
                        play_match(&MatchConfig::symmetric(game_config, seed, baseline))?;
                    let treatment_result = strategy.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.id(),
                strategy.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::LookaheadAblate {
            games,
            first_seed,
            baseline_candidates,
            baseline_determinizations,
            baseline_greedy_plies,
            treatment_candidates,
            treatment_determinizations,
            treatment_greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("lookahead-ablate requires at least one game".into());
            }
            let baseline = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: baseline_candidates,
                determinizations: baseline_determinizations,
                greedy_plies: baseline_greedy_plies,
            })?;
            let treatment = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: treatment_candidates,
                determinizations: treatment_determinizations,
                greedy_plies: treatment_greedy_plies,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::LookaheadRecall {
            games,
            first_seed,
            retained_candidates,
            expanded_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            let report = run_lookahead_recall(
                games,
                first_seed,
                retained_candidates,
                expanded_candidates,
                determinizations,
                greedy_plies,
            )?;
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::NatureWipeCompare {
            games,
            first_seed,
            candidates,
            determinizations,
            greedy_plies,
            prelude_candidates,
            prelude_determinizations,
            prelude_greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("nature-wipe-compare requires at least one game".into());
            }
            let action_config = DeterminizedLookaheadConfig {
                candidate_limit: candidates,
                determinizations,
                greedy_plies,
            };
            let baseline = DeterminizedLookaheadStrategy::new(action_config)?;
            let treatment = NatureWipeLookaheadStrategy::new(NatureWipeLookaheadConfig {
                action_search: action_config,
                prelude_candidate_limit: prelude_candidates,
                prelude_determinizations,
                prelude_greedy_plies,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let treatment_paid_wipes = pairs
                .iter()
                .flat_map(|(_, _, result)| &result.replay.turns)
                .map(|action| action.wildlife_wipes.len())
                .sum::<usize>();
            let treatment_wildlife_slots_replaced = pairs
                .iter()
                .flat_map(|(_, _, result)| &result.replay.turns)
                .flat_map(|action| &action.wildlife_wipes)
                .map(|wipe| wipe.slots.len())
                .sum::<usize>();
            let mut json_value = serde_json::to_value(&report)?;
            let object = json_value
                .as_object_mut()
                .expect("comparison report serializes as an object");
            object.insert(
                "treatment_paid_wipes".to_owned(),
                treatment_paid_wipes.into(),
            );
            object.insert(
                "treatment_wildlife_slots_replaced".to_owned(),
                treatment_wildlife_slots_replaced.into(),
            );
            let json = serde_json::to_string_pretty(&report_context.enrich_value(json_value)?)?;
            let markdown = format!(
                "{}\n## Nature Token Decisions\n\n\
                 - Paid wipes: {}\n\
                 - Wildlife slots replaced: {}\n",
                report.to_markdown(),
                treatment_paid_wipes,
                treatment_wildlife_slots_replaced,
            );
            if let Some(path) = output {
                write_report(&path, &json, &markdown)?;
            }
            println!("{json}");
        }
        Command::BearCandidateCompare {
            games,
            first_seed,
            baseline_candidates,
            candidates,
            bear_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("bear-candidate-compare requires at least one game".into());
            }
            let baseline = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: baseline_candidates.unwrap_or(candidates),
                determinizations,
                greedy_plies,
            })?;
            let treatment = BearCandidateLookaheadStrategy::new(BearCandidateLookaheadConfig {
                immediate_candidate_limit: candidates,
                bear_candidate_limit: bear_candidates,
                determinizations,
                greedy_plies,
            })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::HabitatCandidateCompare {
            games,
            first_seed,
            baseline_candidates,
            candidates,
            habitat_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("habitat-candidate-compare requires at least one game".into());
            }
            let baseline = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
                candidate_limit: baseline_candidates.unwrap_or(candidates),
                determinizations,
                greedy_plies,
            })?;
            let treatment =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::BearHabitatCandidateCompare {
            games,
            first_seed,
            candidates,
            habitat_candidates,
            bear_candidates,
            determinizations,
            greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("bear-habitat-candidate-compare requires at least one game".into());
            }
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let treatment =
                BearHabitatCandidateLookaheadStrategy::new(BearHabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    bear_candidate_limit: bear_candidates,
                    determinizations,
                    greedy_plies,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::HabitatCandidateAblate {
            games,
            first_seed,
            baseline_candidates,
            baseline_habitat_candidates,
            baseline_determinizations,
            baseline_greedy_plies,
            treatment_candidates,
            treatment_habitat_candidates,
            treatment_determinizations,
            treatment_greedy_plies,
            output,
        } => {
            if games == 0 {
                return Err("habitat-candidate-ablate requires at least one game".into());
            }
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: baseline_candidates,
                    habitat_candidate_limit: baseline_habitat_candidates,
                    determinizations: baseline_determinizations,
                    greedy_plies: baseline_greedy_plies,
                })?;
            let treatment =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: treatment_candidates,
                    habitat_candidate_limit: treatment_habitat_candidates,
                    determinizations: treatment_determinizations,
                    greedy_plies: treatment_greedy_plies,
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::PatternBlueprintCompare {
            games,
            first_seed,
            candidates,
            habitat_candidates,
            determinizations,
            rollout_plies,
            policy_candidates,
            policy_habitat_candidates,
            policy_bear_candidates,
            policy_market_draws,
            output,
        } => {
            if games == 0 {
                return Err("pattern-blueprint-compare requires at least one game".into());
            }
            let baseline =
                HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    greedy_plies: rollout_plies,
                })?;
            let treatment =
                PatternBlueprintLookaheadStrategy::new(PatternBlueprintLookaheadConfig {
                    immediate_candidate_limit: candidates,
                    habitat_candidate_limit: habitat_candidates,
                    determinizations,
                    rollout_plies,
                    blueprint: PatternAwareConfig {
                        immediate_candidate_limit: policy_candidates,
                        habitat_candidate_limit: policy_habitat_candidates,
                        bear_candidate_limit: policy_bear_candidates,
                        future_market_draws: policy_market_draws,
                    },
                })?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let started = Instant::now();
            let pairs = (0..games)
                .map(|index| {
                    let numeric_seed = first_seed + index as u64;
                    let seed = GameSeed::from_u64(numeric_seed);
                    let baseline_result = baseline.play_match(game_config, seed)?;
                    let treatment_result = treatment.play_match(game_config, seed)?;
                    Ok::<_, Box<dyn std::error::Error>>((
                        numeric_seed,
                        baseline_result,
                        treatment_result,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let report = summarize_paired_match_results(
                baseline.strategy_id(),
                treatment.strategy_id(),
                first_seed,
                &pairs,
                elapsed_seconds,
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        _ => unreachable!("lookahead dispatcher received a non-lookahead command"),
    }
    Ok(())
}
