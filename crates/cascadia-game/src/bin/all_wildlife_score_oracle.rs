//! Batch production-score oracle for arbitrary pure-wildlife card combinations.

#[allow(dead_code)]
mod wildlife_solver_support;

use std::io::{self, Read};

use cascadia_game::{HexCoord, ScoringCards, ScoringVariant, Wildlife};
use serde::{Deserialize, Serialize};
use wildlife_solver_support::{
    COUNT_CAP, Layout, SPECIES_COUNT, TOKEN_COUNT, Token, production_score, wildlife_name,
};

#[derive(Deserialize)]
struct TokenRow {
    q: i8,
    r: i8,
    wildlife: String,
}

#[derive(Deserialize)]
struct Request {
    ruleset: String,
    tokens: Vec<TokenRow>,
}

#[derive(Serialize)]
struct Response {
    ruleset: String,
    score_breakdown: [u16; SPECIES_COUNT],
    score: u16,
}

fn variant(value: u8) -> Result<ScoringVariant, String> {
    match value {
        b'A' => Ok(ScoringVariant::A),
        b'B' => Ok(ScoringVariant::B),
        b'C' => Ok(ScoringVariant::C),
        b'D' => Ok(ScoringVariant::D),
        _ => Err(format!("invalid card {}", char::from(value))),
    }
}

fn cards(ruleset: &str) -> Result<ScoringCards, String> {
    let normalized = ruleset.to_ascii_uppercase();
    let values = normalized.as_bytes();
    if values.len() != SPECIES_COUNT {
        return Err(format!("ruleset {ruleset:?} must have five cards"));
    }
    Ok(ScoringCards {
        bear: variant(values[0])?,
        elk: variant(values[1])?,
        salmon: variant(values[2])?,
        hawk: variant(values[3])?,
        fox: variant(values[4])?,
    })
}

fn wildlife(value: &str) -> Result<Wildlife, String> {
    Wildlife::ALL
        .into_iter()
        .find(|candidate| wildlife_name(*candidate) == value)
        .ok_or_else(|| format!("unknown wildlife {value:?}"))
}

fn score(request: Request) -> Result<Response, String> {
    if request.tokens.len() != TOKEN_COUNT {
        return Err(format!(
            "{} has {} tokens, expected {TOKEN_COUNT}",
            request.ruleset,
            request.tokens.len()
        ));
    }
    let layout = Layout {
        tokens: request
            .tokens
            .into_iter()
            .map(|row| {
                Ok(Token {
                    coord: HexCoord::new(row.q, row.r),
                    wildlife: wildlife(&row.wildlife)?,
                })
            })
            .collect::<Result<_, String>>()?,
    };
    if !layout.is_connected() {
        return Err(format!("{} board is disconnected", request.ruleset));
    }
    let counts = layout.counts();
    if counts.into_iter().any(|count| count > COUNT_CAP) {
        return Err(format!(
            "{} exceeds the per-species cap: {counts:?}",
            request.ruleset
        ));
    }
    let score_breakdown = production_score(&layout, cards(&request.ruleset)?);
    Ok(Response {
        ruleset: request.ruleset.to_ascii_uppercase(),
        score: score_breakdown.into_iter().sum(),
        score_breakdown,
    })
}

fn run() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("failed reading stdin: {error}"))?;
    let requests: Vec<Request> =
        serde_json::from_str(&input).map_err(|error| format!("invalid request JSON: {error}"))?;
    let responses = requests
        .into_iter()
        .map(score)
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&responses)
            .map_err(|error| format!("failed serializing response: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_ruleset() {
        for bear in b"ABCD" {
            for elk in b"ABCD" {
                for salmon in b"ABCD" {
                    for hawk in b"ABCD" {
                        for fox in b"ABCD" {
                            let ruleset =
                                String::from_utf8(vec![*bear, *elk, *salmon, *hawk, *fox]).unwrap();
                            cards(&ruleset).unwrap();
                        }
                    }
                }
            }
        }
    }
}
