//! CPU driver for the WI-2 selfish tomography optimizers.
//!
//! Consumes a directory of sealed terminal trajectory-ledger JSON files,
//! runs the T0 repacking and chronology-preserving replay optimizers for
//! every seat, and emits one deterministic
//! `cascadiav3.rival_tomography_summary.v1` JSON document.  Every witness in
//! the summary is a feasible LOWER bound (`witness_semantics =
//! "lower_bound_only"`); nothing here is an upper bound or a policy claim.
//!
//! The summary is printed to stdout; `--out` additionally publishes it
//! durably without ever replacing an existing artifact.  CPU-only by
//! construction: no device, no network, no policy model.

use std::{path::PathBuf, process::ExitCode};

use cascadia_rival::{HarnessConfig, run_directory};

const DEFAULT_REPACK_ITERATIONS: u32 = 4_000;
const DEFAULT_BEAM_WIDTH: u32 = 4;
const DEFAULT_CANDIDATE_CAP: u32 = 16;

fn usage() -> &'static str {
    "usage: rival-tomography <ledger-directory> --seed <u64> \
     [--repack-iterations <u32>] [--beam-width <u32>] [--candidate-cap <u32>] \
     [--out <summary.json>]"
}

struct Arguments {
    directory: PathBuf,
    config: HarnessConfig,
    out: Option<PathBuf>,
}

fn parse_arguments(mut arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let directory = arguments.next().ok_or_else(|| usage().to_owned())?;
    if directory.starts_with("--") {
        return Err(usage().to_owned());
    }
    let mut seed: Option<u64> = None;
    let mut repack_iterations = DEFAULT_REPACK_ITERATIONS;
    let mut beam_width = DEFAULT_BEAM_WIDTH;
    let mut candidate_cap = DEFAULT_CANDIDATE_CAP;
    let mut out: Option<PathBuf> = None;
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} requires a value; {}", usage()))
        };
        match flag.as_str() {
            "--seed" => seed = Some(parse_canonical_u64(&flag, &value()?)?),
            "--repack-iterations" => {
                repack_iterations = parse_canonical_u64(&flag, &value()?)?
                    .try_into()
                    .map_err(|_| format!("{flag} is out of range"))?;
            }
            "--beam-width" => {
                beam_width = parse_canonical_u64(&flag, &value()?)?
                    .try_into()
                    .map_err(|_| format!("{flag} is out of range"))?;
            }
            "--candidate-cap" => {
                candidate_cap = parse_canonical_u64(&flag, &value()?)?
                    .try_into()
                    .map_err(|_| format!("{flag} is out of range"))?;
            }
            "--out" => {
                if out.is_some() {
                    return Err(format!("--out was supplied twice; {}", usage()));
                }
                out = Some(PathBuf::from(value()?));
            }
            _ => return Err(format!("unknown argument {flag:?}; {}", usage())),
        }
    }
    let seed = seed.ok_or_else(|| format!("--seed is required; {}", usage()))?;
    Ok(Arguments {
        directory: PathBuf::from(directory),
        config: HarnessConfig {
            seed,
            repack_iterations,
            beam_width,
            candidate_cap,
        },
        out,
    })
}

fn parse_canonical_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed: u64 = value
        .parse()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))?;
    if parsed.to_string() != value {
        return Err(format!("{flag} value must be canonical unsigned decimal"));
    }
    Ok(parsed)
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments(std::env::args().skip(1))?;
    let summary = run_directory(&arguments.directory, &arguments.config)
        .map_err(|error| format!("tomography harness failed: {error}"))?;
    let bytes = summary
        .canonical_json_bytes()
        .map_err(|error| format!("could not serialize validated summary: {error}"))?;
    let rendered = String::from_utf8(bytes)
        .map_err(|error| format!("canonical summary is not UTF-8: {error}"))?;
    println!("{rendered}");
    if let Some(destination) = &arguments.out {
        summary
            .write_json_immutable(destination)
            .map_err(|error| format!("could not publish summary to {destination:?}: {error}"))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Result<Arguments, String> {
        parse_arguments(values.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn seed_is_mandatory_and_defaults_are_applied() {
        assert!(arguments(&["ledgers"]).is_err());
        let parsed = arguments(&["ledgers", "--seed", "7"]).unwrap();
        assert_eq!(parsed.directory, PathBuf::from("ledgers"));
        assert_eq!(
            parsed.config,
            HarnessConfig {
                seed: 7,
                repack_iterations: DEFAULT_REPACK_ITERATIONS,
                beam_width: DEFAULT_BEAM_WIDTH,
                candidate_cap: DEFAULT_CANDIDATE_CAP,
            }
        );
        assert!(parsed.out.is_none());
    }

    #[test]
    fn arguments_are_strict_and_canonical() {
        assert!(arguments(&["--seed", "7"]).is_err());
        assert!(arguments(&["ledgers", "--seed", "07"]).is_err());
        assert!(arguments(&["ledgers", "--seed", "7", "--beam-width"]).is_err());
        assert!(arguments(&["ledgers", "--seed", "7", "--frobnicate", "1"]).is_err());
        let parsed = arguments(&[
            "ledgers",
            "--seed",
            "7",
            "--repack-iterations",
            "100",
            "--beam-width",
            "2",
            "--candidate-cap",
            "5",
            "--out",
            "summary.json",
        ])
        .unwrap();
        assert_eq!(parsed.config.repack_iterations, 100);
        assert_eq!(parsed.config.beam_width, 2);
        assert_eq!(parsed.config.candidate_cap, 5);
        assert_eq!(parsed.out, Some(PathBuf::from("summary.json")));
    }
}
