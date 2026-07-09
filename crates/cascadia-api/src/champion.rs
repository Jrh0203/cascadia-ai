//! Champion-strength suggestions from the v3 stack: a persistent
//! `cascadiav3-real-root-exporter --gumbel-suggest-server` child process
//! (CascadiaFormer checkpoint + Gumbel search) spoken to over JSONL on
//! stdin/stdout. The child command comes from `CASCADIA_CHAMPION_CMD`
//! (spawned via `sh -c`); when unset, the champion strength is
//! unavailable. The child is spawned lazily on first use, held behind a
//! mutex (one suggestion at a time), and respawned after any I/O error.

use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Mutex, OnceLock},
};

use cascadia_game::{GameState, TurnAction};
use serde::Deserialize;
use serde_json::{Value, json};

pub fn champion_command() -> Option<String> {
    std::env::var("CASCADIA_CHAMPION_CMD")
        .ok()
        .filter(|command| !command.trim().is_empty())
}

pub fn champion_available() -> bool {
    champion_command().is_some()
}

#[derive(Debug, Deserialize)]
pub struct ChampionSuggestion {
    #[serde(default)]
    pub game_over: bool,
    #[serde(default)]
    pub chosen_index: usize,
    #[serde(default)]
    pub actions: Vec<TurnAction>,
    #[serde(default)]
    pub completed_q: Vec<f64>,
    #[serde(default)]
    pub improved_policy: Vec<f64>,
    #[serde(default)]
    pub root_value: f64,
}

struct ChampionClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Drop for ChampionClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ChampionClient {
    fn spawn(command: &str) -> Result<Self, String> {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("spawning champion suggest server: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "champion suggest server has no stdin".to_owned())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "champion suggest server has no stdout".to_owned())?;
        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
        };
        // Block until the model bridge reports ready (first line).
        let ready = client.read_line()?;
        if ready.get("type").and_then(Value::as_str) != Some("suggest_ready") {
            return Err(format!("unexpected champion handshake: {ready}"));
        }
        Ok(client)
    }

    fn read_line(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .stdout
                .read_line(&mut line)
                .map_err(|error| format!("reading from champion suggest server: {error}"))?;
            if read == 0 {
                return Err("champion suggest server closed its stdout".to_owned());
            }
            if line.trim().is_empty() {
                continue;
            }
            return serde_json::from_str(&line)
                .map_err(|error| format!("bad champion response json: {error}"));
        }
    }

    fn suggest(
        &mut self,
        state: &GameState,
        overrides: Option<(u64, u64)>,
    ) -> Result<ChampionSuggestion, String> {
        self.next_id += 1;
        let request = match overrides {
            Some((n_simulations, determinizations)) => json!({
                "id": self.next_id,
                "game": state,
                "n_simulations": n_simulations,
                "determinizations": determinizations,
            }),
            None => json!({"id": self.next_id, "game": state}),
        };
        let mut payload = serde_json::to_string(&request)
            .map_err(|error| format!("serializing champion request: {error}"))?;
        payload.push('\n');
        self.stdin
            .write_all(payload.as_bytes())
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("writing to champion suggest server: {error}"))?;
        let response = self.read_line()?;
        match response.get("type").and_then(Value::as_str) {
            Some("suggest_response") => serde_json::from_value(response)
                .map_err(|error| format!("bad champion suggestion: {error}")),
            Some("suggest_error") => Err(response
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown champion error")
                .to_owned()),
            other => Err(format!("unexpected champion response type: {other:?}")),
        }
    }
}

fn client_slot() -> &'static Mutex<Option<ChampionClient>> {
    static SLOT: OnceLock<Mutex<Option<ChampionClient>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// One champion suggestion. Spawns the suggest server on first use; drops
/// and respawns it on the next call after any transport error. `deep`
/// switches to the full-strength search shape (n1024/d16) via per-request
/// overrides — same server, same loaded model.
pub fn suggest(state: &GameState, deep: bool) -> Result<ChampionSuggestion, String> {
    let command = champion_command()
        .ok_or_else(|| "CASCADIA_CHAMPION_CMD is not configured".to_owned())?;
    let mut slot = client_slot()
        .lock()
        .map_err(|_| "champion client mutex poisoned".to_owned())?;
    if slot.is_none() {
        *slot = Some(ChampionClient::spawn(&command)?);
    }
    let client = slot.as_mut().expect("client just ensured");
    let overrides = deep.then_some((1024_u64, 16_u64));
    match client.suggest(state, overrides) {
        Ok(suggestion) => Ok(suggestion),
        Err(error) => {
            // Kill the wedged child; the next request gets a fresh one.
            *slot = None;
            Err(error)
        }
    }
}
