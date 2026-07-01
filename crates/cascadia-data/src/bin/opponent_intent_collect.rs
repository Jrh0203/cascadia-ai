use std::{collections::BTreeSet, env, error::Error, fs, path::PathBuf};

use cascadia_data::{
    DatasetSplit, OpponentIntentCohort, OpponentIntentDatasetConfig, OpponentIntentDatasetManifest,
    collect_opponent_intent_dataset, validate_opponent_intent_dataset,
};
use cascadia_sim::StrategyKind;

const HELP: &str = concat!(
    "Usage:\n",
    "  opponent_intent_collect collect --output PATH --split SPLIT \\\n",
    "    --first-game-index N --games N --shard-games N --cohort-id ID \\\n",
    "    --policy-pool IDS [--required-policy ID] [--resume]\n",
    "  opponent_intent_collect validate --dataset PATH\n\n",
    "SPLIT is train, validation, test, or final. IDS is a comma-separated list\n",
    "drawn from random, greedy, pattern-aware, pattern-commitment,\n",
    "pattern-competition, and pattern-portfolio."
);

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Collect {
        output: PathBuf,
        split: DatasetSplit,
        first_game_index: u64,
        games: usize,
        shard_games: usize,
        cohort_id: String,
        policy_pool: Vec<StrategyKind>,
        required_policy: Option<StrategyKind>,
        resume: bool,
    },
    Validate {
        dataset: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_args(env::args().skip(1))? {
        Command::Collect {
            output,
            split,
            first_game_index,
            games,
            shard_games,
            cohort_id,
            policy_pool,
            required_policy,
            resume,
        } => {
            let manifest = collect_opponent_intent_dataset(&OpponentIntentDatasetConfig {
                output,
                split,
                first_game_index,
                games,
                shard_games,
                cohort: OpponentIntentCohort {
                    cohort_id,
                    policy_pool,
                    required_policy,
                },
                resume,
            })?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Validate { dataset } => {
            let manifest: OpponentIntentDatasetManifest =
                serde_json::from_slice(&fs::read(dataset.join("dataset.json"))?)?;
            validate_opponent_intent_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
    }
    Ok(())
}

fn parse_args(arguments: impl IntoIterator<Item = String>) -> Result<Command, Box<dyn Error>> {
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Err(HELP.into());
    };
    if command == "validate" {
        let mut dataset = None;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--dataset" => dataset = Some(PathBuf::from(next(&mut arguments, "--dataset")?)),
                "--help" | "-h" => {
                    println!("{HELP}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}\n\n{HELP}").into()),
            }
        }
        return Ok(Command::Validate {
            dataset: dataset.ok_or("--dataset is required")?,
        });
    }
    if command != "collect" {
        return Err(format!("unknown command {command}\n\n{HELP}").into());
    }

    let mut output = None;
    let mut split = None;
    let mut first_game_index = None;
    let mut games = None;
    let mut shard_games = None;
    let mut cohort_id = None;
    let mut policy_pool = None;
    let mut required_policy = None;
    let mut resume = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--output" => output = Some(PathBuf::from(next(&mut arguments, "--output")?)),
            "--split" => split = Some(parse_split(&next(&mut arguments, "--split")?)?),
            "--first-game-index" => {
                first_game_index = Some(next(&mut arguments, "--first-game-index")?.parse()?)
            }
            "--games" => games = Some(next(&mut arguments, "--games")?.parse()?),
            "--shard-games" => shard_games = Some(next(&mut arguments, "--shard-games")?.parse()?),
            "--cohort-id" => cohort_id = Some(next(&mut arguments, "--cohort-id")?),
            "--policy-pool" => {
                policy_pool = Some(parse_policy_pool(&next(&mut arguments, "--policy-pool")?)?)
            }
            "--required-policy" => {
                required_policy = Some(parse_policy(&next(&mut arguments, "--required-policy")?)?)
            }
            "--resume" => resume = true,
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n\n{HELP}").into()),
        }
    }
    Ok(Command::Collect {
        output: output.ok_or("--output is required")?,
        split: split.ok_or("--split is required")?,
        first_game_index: first_game_index.ok_or("--first-game-index is required")?,
        games: games.ok_or("--games is required")?,
        shard_games: shard_games.ok_or("--shard-games is required")?,
        cohort_id: cohort_id.ok_or("--cohort-id is required")?,
        policy_pool: policy_pool.ok_or("--policy-pool is required")?,
        required_policy,
        resume,
    })
}

fn next(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn parse_split(value: &str) -> Result<DatasetSplit, Box<dyn Error>> {
    match value {
        "train" => Ok(DatasetSplit::Train),
        "validation" => Ok(DatasetSplit::Validation),
        "test" => Ok(DatasetSplit::Test),
        "final" => Ok(DatasetSplit::Final),
        _ => Err(format!("unknown split {value}").into()),
    }
}

fn parse_policy_pool(value: &str) -> Result<Vec<StrategyKind>, Box<dyn Error>> {
    let policies = value
        .split(',')
        .map(parse_policy)
        .collect::<Result<Vec<_>, _>>()?;
    let unique = policies
        .iter()
        .map(|policy| policy.id())
        .collect::<BTreeSet<_>>();
    if policies.is_empty() || unique.len() != policies.len() {
        return Err("policy pool must contain unique policies".into());
    }
    Ok(policies)
}

fn parse_policy(value: &str) -> Result<StrategyKind, Box<dyn Error>> {
    match value {
        "random" | "random-v1" => Ok(StrategyKind::Random),
        "greedy" | "greedy-v1" => Ok(StrategyKind::Greedy),
        "pattern-aware" | "pattern-aware-v1" | "pattern-aware-v1-k8-h6-b8-m4" => {
            Ok(StrategyKind::PatternAware)
        }
        "pattern-commitment"
        | "pattern-commitment-v1"
        | "pattern-commitment-v2-k8-h6-b8-m4-t2-phase-capped" => {
            Ok(StrategyKind::PatternCommitment)
        }
        "pattern-competition"
        | "pattern-competition-v1"
        | "pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation" => {
            Ok(StrategyKind::PatternCompetition)
        }
        "pattern-portfolio"
        | "pattern-portfolio-v1"
        | "pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium" => {
            Ok(StrategyKind::PatternPortfolio)
        }
        _ => Err(format!("unknown policy {value}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_collection_and_validation_commands() {
        let collect = parse_args(
            [
                "collect",
                "--output",
                "out",
                "--split",
                "validation",
                "--first-game-index",
                "100",
                "--games",
                "8",
                "--shard-games",
                "2",
                "--cohort-id",
                "heldout",
                "--policy-pool",
                "greedy,pattern-competition",
                "--required-policy",
                "pattern-competition",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert!(matches!(
            collect,
            Command::Collect {
                split: DatasetSplit::Validation,
                required_policy: Some(StrategyKind::PatternCompetition),
                ..
            }
        ));
        assert_eq!(
            parse_args(
                ["validate", "--dataset", "out"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .unwrap(),
            Command::Validate {
                dataset: PathBuf::from("out")
            }
        );
    }

    #[test]
    fn parser_rejects_duplicate_policy_pool() {
        let result = parse_policy_pool("greedy,greedy");
        assert!(result.is_err());
    }
}
