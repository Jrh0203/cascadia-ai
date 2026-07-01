use std::collections::BTreeMap;

use blake3::Hasher;
use cascadia_game::{DraftChoice, GameSeed, GameState, MarketPrelude, TurnAction};
use cascadia_sim::{PatternAwareConfig, rank_pattern_actions};

use crate::SearchError;

const ACTION_HASH_DOMAIN: &[u8] = b"cascadia-v2-full-legal-action-v1";
const DETERMINIZATION_DOMAIN: &[u8] =
    b"cascadia-v2-offline-public-belief-post-root-determinization-v1";
const OPPONENT_UNIFORM_DOMAIN: &[u8] = b"cascadia-v2-offline-public-belief-opponent-uniform-v1";
const TRACE_DOMAIN: &[u8] = b"cascadia-v2-offline-public-belief-trace-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HorizonPatternPolicyConfig {
    pub blueprint: PatternAwareConfig,
    pub temperature_milli: u16,
}

impl HorizonPatternPolicyConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        if self.temperature_milli == 0 {
            return Err(SearchError::InvalidConfig(
                "public-belief pattern temperature must be positive",
            ));
        }
        Ok(self)
    }

    fn temperature(self) -> f64 {
        f64::from(self.temperature_milli) / 1_000.0
    }
}

impl Default for HorizonPatternPolicyConfig {
    fn default() -> Self {
        Self {
            blueprint: PatternAwareConfig::default(),
            temperature_milli: 1_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicBeliefTrajectory {
    pub state: GameState,
    pub focal_player: usize,
    pub opponent_action_hashes: Vec<[u8; 32]>,
    pub trace_hash: [u8; 32],
    pub public_leaf_hash: [u8; 32],
    pub opponent_decisions: usize,
    pub opponent_options: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialHalvingSchedule {
    pub additional_samples: Vec<usize>,
    pub retained_roots: Vec<usize>,
}

impl SequentialHalvingSchedule {
    pub fn validate(&self, root_count: usize) -> Result<(), SearchError> {
        if root_count == 0
            || self.additional_samples.is_empty()
            || self.additional_samples.len() != self.retained_roots.len()
            || self.additional_samples.contains(&0)
        {
            return Err(SearchError::InvalidConfig(
                "sequential halving requires roots and matched positive stages",
            ));
        }
        let mut active = root_count;
        for &retained in &self.retained_roots {
            if retained == 0 || retained > active {
                return Err(SearchError::InvalidConfig(
                    "sequential halving retained-root schedule is invalid",
                ));
            }
            active = retained;
        }
        if active != 1 {
            return Err(SearchError::InvalidConfig(
                "sequential halving must finish with one root",
            ));
        }
        Ok(())
    }

    pub fn total_evaluations(&self, root_count: usize) -> Result<usize, SearchError> {
        self.validate(root_count)?;
        let mut active = root_count;
        let mut total = 0usize;
        for (&samples, &retained) in self.additional_samples.iter().zip(&self.retained_roots) {
            total = total
                .checked_add(
                    active
                        .checked_mul(samples)
                        .ok_or(SearchError::InvalidConfig(
                            "sequential halving evaluation count overflowed",
                        ))?,
                )
                .ok_or(SearchError::InvalidConfig(
                    "sequential halving evaluation count overflowed",
                ))?;
            active = retained;
        }
        Ok(total)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalvingWork {
    pub root_index: usize,
    pub sample_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HalvingRootStatistics {
    pub samples: usize,
    pub mean: f64,
    pub standard_deviation: f64,
    pub eliminated_stage: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequentialHalvingResult {
    pub selected_root: usize,
    pub total_evaluations: usize,
    pub roots: Vec<HalvingRootStatistics>,
}

#[derive(Debug, Clone, Default)]
struct RunningMoments {
    samples: usize,
    sum: f64,
    square_sum: f64,
}

impl RunningMoments {
    fn add(&mut self, value: f64) {
        self.samples += 1;
        self.sum += value;
        self.square_sum += value * value;
    }

    fn mean(&self) -> f64 {
        self.sum / self.samples as f64
    }

    fn standard_deviation(&self) -> f64 {
        if self.samples < 2 {
            return 0.0;
        }
        let mean = self.mean();
        ((self.square_sum - self.samples as f64 * mean * mean) / (self.samples - 1) as f64)
            .max(0.0)
            .sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct SequentialHalving {
    schedule: SequentialHalvingSchedule,
    action_hashes: Vec<[u8; 32]>,
    moments: Vec<RunningMoments>,
    eliminated_stage: Vec<Option<usize>>,
    active: Vec<usize>,
    stage: usize,
    sample_start: usize,
}

impl SequentialHalving {
    pub fn new(
        action_hashes: Vec<[u8; 32]>,
        schedule: SequentialHalvingSchedule,
    ) -> Result<Self, SearchError> {
        schedule.validate(action_hashes.len())?;
        let root_count = action_hashes.len();
        Ok(Self {
            schedule,
            action_hashes,
            moments: vec![RunningMoments::default(); root_count],
            eliminated_stage: vec![None; root_count],
            active: (0..root_count).collect(),
            stage: 0,
            sample_start: 0,
        })
    }

    pub fn is_complete(&self) -> bool {
        self.stage == self.schedule.additional_samples.len()
    }

    pub fn stage(&self) -> usize {
        self.stage
    }

    pub fn work(&self) -> Result<Vec<HalvingWork>, SearchError> {
        if self.is_complete() {
            return Err(SearchError::InvalidConfig(
                "sequential halving has no work after completion",
            ));
        }
        let samples = self.schedule.additional_samples[self.stage];
        Ok(self
            .active
            .iter()
            .flat_map(|&root_index| {
                (self.sample_start..self.sample_start + samples).map(move |sample_index| {
                    HalvingWork {
                        root_index,
                        sample_index,
                    }
                })
            })
            .collect())
    }

    pub fn complete_stage(&mut self, values: &[f64]) -> Result<(), SearchError> {
        let work = self.work()?;
        if values.len() != work.len() {
            return Err(SearchError::PredictionCount {
                expected: work.len(),
                actual: values.len(),
            });
        }
        for (index, (&value, item)) in values.iter().zip(&work).enumerate() {
            if !value.is_finite() {
                return Err(SearchError::NonFinitePrediction { index });
            }
            self.moments[item.root_index].add(value);
        }
        self.sample_start += self.schedule.additional_samples[self.stage];
        self.active.sort_by(|&left, &right| {
            self.moments[right]
                .mean()
                .total_cmp(&self.moments[left].mean())
                .then_with(|| self.action_hashes[left].cmp(&self.action_hashes[right]))
        });
        let retained = self.schedule.retained_roots[self.stage];
        for &root in &self.active[retained..] {
            self.eliminated_stage[root] = Some(self.stage + 1);
        }
        self.active.truncate(retained);
        self.stage += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<SequentialHalvingResult, SearchError> {
        if !self.is_complete() || self.active.len() != 1 {
            return Err(SearchError::InvalidConfig(
                "sequential halving cannot finish before one root remains",
            ));
        }
        let total_evaluations = self.moments.iter().map(|moments| moments.samples).sum();
        let expected = self.schedule.total_evaluations(self.action_hashes.len())?;
        if total_evaluations != expected {
            return Err(SearchError::InvalidConfig(
                "sequential halving evaluation accounting drifted",
            ));
        }
        Ok(SequentialHalvingResult {
            selected_root: self.active[0],
            total_evaluations,
            roots: self
                .moments
                .into_iter()
                .zip(self.eliminated_stage)
                .map(|(moments, eliminated_stage)| HalvingRootStatistics {
                    samples: moments.samples,
                    mean: moments.mean(),
                    standard_deviation: moments.standard_deviation(),
                    eliminated_stage,
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DraftKey {
    kind: u8,
    tile_slot: u8,
    wildlife_slot: u8,
}

impl DraftKey {
    fn from_choice(choice: DraftChoice) -> Self {
        match choice {
            DraftChoice::Paired { slot } => Self {
                kind: 0,
                tile_slot: slot.index() as u8,
                wildlife_slot: slot.index() as u8,
            },
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => Self {
                kind: 1,
                tile_slot: tile_slot.index() as u8,
                wildlife_slot: wildlife_slot.index() as u8,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct DraftOption {
    action: TurnAction,
    action_hash: [u8; 32],
    heuristic_value: f64,
}

pub fn canonical_complete_action_hash(action: &TurnAction) -> Result<[u8; 32], SearchError> {
    let mut hasher = Hasher::new();
    hasher.update(ACTION_HASH_DOMAIN);
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

pub fn public_belief_determinization_seed(
    group_id: u64,
    action_hash: &[u8; 32],
    sample_index: usize,
) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(DETERMINIZATION_DOMAIN);
    hasher.update(&group_id.to_le_bytes());
    hasher.update(action_hash);
    hasher.update(&(sample_index as u64).to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

pub fn simulate_pattern_prior_horizon(
    game: &GameState,
    group_id: u64,
    root_action: &TurnAction,
    root_action_hash: &[u8; 32],
    sample_index: usize,
    opponent_turns: usize,
    config: HorizonPatternPolicyConfig,
) -> Result<PublicBeliefTrajectory, SearchError> {
    if canonical_complete_action_hash(root_action)? != *root_action_hash {
        return Err(SearchError::InvalidConfig(
            "public-belief root action hash does not match the action",
        ));
    }
    let focal_player = game.current_player();
    let root_afterstate = game.transition(root_action)?;
    simulate_pattern_prior_post_root_horizon(
        root_afterstate,
        focal_player,
        group_id,
        root_action_hash,
        sample_index,
        opponent_turns,
        config,
    )
}

pub fn simulate_pattern_prior_post_root_horizon(
    mut root_afterstate: GameState,
    focal_player: usize,
    group_id: u64,
    root_action_hash: &[u8; 32],
    sample_index: usize,
    opponent_turns: usize,
    config: HorizonPatternPolicyConfig,
) -> Result<PublicBeliefTrajectory, SearchError> {
    let config = config.validate()?;
    let maximum_opponents = usize::from(root_afterstate.config().player_count).saturating_sub(1);
    if opponent_turns > maximum_opponents || focal_player >= root_afterstate.boards().len() {
        return Err(SearchError::InvalidConfig(
            "public-belief horizon exceeds the player rotation",
        ));
    }
    if opponent_turns > 0 {
        root_afterstate.redeterminize_hidden(public_belief_determinization_seed(
            group_id,
            root_action_hash,
            sample_index,
        ));
    }
    let mut state = root_afterstate;
    let mut trace_hasher = Hasher::new();
    trace_hasher.update(TRACE_DOMAIN);
    trace_hasher.update(root_action_hash);
    let mut opponent_action_hashes = Vec::with_capacity(opponent_turns);
    let mut opponent_options = 0usize;

    for opponent_offset in 0..opponent_turns {
        if state.is_game_over() {
            break;
        }
        let (action, option_count) = select_pattern_prior_action(
            &state,
            opponent_uniform(group_id, root_action_hash, sample_index, opponent_offset),
            config,
        )?;
        let action_hash = canonical_complete_action_hash(&action)?;
        trace_hasher.update(&action_hash);
        opponent_action_hashes.push(action_hash);
        opponent_options += option_count;
        state.apply(&action)?;
    }
    Ok(PublicBeliefTrajectory {
        public_leaf_hash: *state.public_state().canonical_hash().as_bytes(),
        state,
        focal_player,
        opponent_decisions: opponent_action_hashes.len(),
        opponent_action_hashes,
        trace_hash: *trace_hasher.finalize().as_bytes(),
        opponent_options,
    })
}

fn select_pattern_prior_action(
    state: &GameState,
    uniform: f64,
    config: HorizonPatternPolicyConfig,
) -> Result<(TurnAction, usize), SearchError> {
    let prelude = MarketPrelude {
        replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let candidates = rank_pattern_actions(state, &prelude, config.blueprint)?;
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let mut by_draft = BTreeMap::<DraftKey, DraftOption>::new();
    for candidate in candidates {
        let key = DraftKey::from_choice(candidate.action.draft);
        let action_hash = canonical_complete_action_hash(&candidate.action)?;
        let option = DraftOption {
            action: candidate.action,
            action_hash,
            heuristic_value: candidate.heuristic_value,
        };
        match by_draft.get(&key) {
            None => {
                by_draft.insert(key, option);
            }
            Some(previous)
                if option.heuristic_value > previous.heuristic_value
                    || (option.heuristic_value == previous.heuristic_value
                        && option.action_hash < previous.action_hash) =>
            {
                by_draft.insert(key, option);
            }
            Some(_) => {}
        }
    }
    let options = by_draft.into_values().collect::<Vec<_>>();
    let best = options
        .iter()
        .map(|option| option.heuristic_value)
        .max_by(f64::total_cmp)
        .ok_or(SearchError::NoLegalActions)?;
    let weights = options
        .iter()
        .map(|option| {
            ((option.heuristic_value - best) / config.temperature())
                .clamp(-40.0, 0.0)
                .exp()
        })
        .collect::<Vec<_>>();
    let selected = weighted_index(&weights, uniform)?;
    Ok((options[selected].action.clone(), options.len()))
}

fn weighted_index(weights: &[f64], uniform: f64) -> Result<usize, SearchError> {
    let total = weights.iter().sum::<f64>();
    if weights.is_empty()
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || !total.is_finite()
        || total <= 0.0
        || !(0.0..1.0).contains(&uniform)
    {
        return Err(SearchError::InvalidConfig(
            "weighted public-belief selection received invalid inputs",
        ));
    }
    let target = uniform * total;
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if target < cumulative {
            return Ok(index);
        }
    }
    Ok(weights.len() - 1)
}

fn opponent_uniform(
    group_id: u64,
    action_hash: &[u8; 32],
    sample_index: usize,
    opponent_offset: usize,
) -> f64 {
    let mut hasher = Hasher::new();
    hasher.update(OPPONENT_UNIFORM_DOMAIN);
    hasher.update(&group_id.to_le_bytes());
    hasher.update(action_hash);
    hasher.update(&(sample_index as u64).to_le_bytes());
    hasher.update(&(opponent_offset as u64).to_le_bytes());
    let bytes = hasher.finalize();
    let numerator = u64::from_le_bytes(bytes.as_bytes()[..8].try_into().expect("eight bytes"));
    (numerator as f64) / ((u64::MAX as f64) + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::{GameConfig, MarketSlot, Rotation};

    fn standard_schedule() -> SequentialHalvingSchedule {
        SequentialHalvingSchedule {
            additional_samples: vec![4, 4, 8, 16],
            retained_roots: vec![32, 16, 8, 1],
        }
    }

    fn first_pattern_root(game: &GameState) -> TurnAction {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        rank_pattern_actions(game, &prelude, PatternAwareConfig::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .action
    }

    #[test]
    fn standard_halving_budget_is_exact() {
        assert_eq!(standard_schedule().total_evaluations(64).unwrap(), 640);
    }

    #[test]
    fn halving_uses_hashes_to_break_equal_values() {
        let hashes = vec![[2; 32], [0; 32], [1; 32], [3; 32]];
        let schedule = SequentialHalvingSchedule {
            additional_samples: vec![1, 1],
            retained_roots: vec![2, 1],
        };
        let mut halving = SequentialHalving::new(hashes, schedule).unwrap();
        halving.complete_stage(&[7.0; 4]).unwrap();
        assert_eq!(
            halving.work().unwrap(),
            vec![
                HalvingWork {
                    root_index: 1,
                    sample_index: 1,
                },
                HalvingWork {
                    root_index: 2,
                    sample_index: 1,
                },
            ]
        );
        halving.complete_stage(&[7.0; 2]).unwrap();
        let result = halving.finish().unwrap();
        assert_eq!(result.selected_root, 1);
        assert_eq!(result.total_evaluations, 6);
    }

    #[test]
    fn horizons_share_exact_opponent_prefixes() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(41),
        )
        .unwrap();
        let root = first_pattern_root(&game);
        let hash = canonical_complete_action_hash(&root).unwrap();
        let config = HorizonPatternPolicyConfig::default();
        let h1 = simulate_pattern_prior_horizon(&game, 17, &root, &hash, 3, 1, config).unwrap();
        let h2 = simulate_pattern_prior_horizon(&game, 17, &root, &hash, 3, 2, config).unwrap();
        let h3 = simulate_pattern_prior_horizon(&game, 17, &root, &hash, 3, 3, config).unwrap();
        assert_eq!(h1.opponent_action_hashes, h2.opponent_action_hashes[..1]);
        assert_eq!(h1.opponent_action_hashes, h3.opponent_action_hashes[..1]);
        assert_eq!(h2.opponent_action_hashes, h3.opponent_action_hashes[..2]);
    }

    #[test]
    fn registered_post_root_determinization_erases_prior_hidden_order() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(91),
        )
        .unwrap();
        let root = first_pattern_root(&game);
        let hash = canonical_complete_action_hash(&root).unwrap();
        let afterstate = game.transition(&root).unwrap();
        let mut perturbed = afterstate.clone();
        perturbed.redeterminize_hidden(GameSeed::from_u64(999));
        let left = simulate_pattern_prior_post_root_horizon(
            afterstate,
            0,
            23,
            &hash,
            5,
            3,
            HorizonPatternPolicyConfig::default(),
        )
        .unwrap();
        let right = simulate_pattern_prior_post_root_horizon(
            perturbed,
            0,
            23,
            &hash,
            5,
            3,
            HorizonPatternPolicyConfig::default(),
        )
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn complete_root_is_applied_before_future_redeterminization() {
        let mut witness = None;
        'seeds: for seed in 0..4_096 {
            let game = GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(seed),
            )
            .unwrap();
            if game.market().three_of_a_kind().is_none() {
                continue;
            }
            let prelude = MarketPrelude {
                replace_three_of_a_kind: true,
                wildlife_wipes: Vec::new(),
            };
            for action in game.legal_turn_actions(&prelude).unwrap() {
                if action.wildlife.is_none() {
                    continue;
                }
                for redetermination in 0..32 {
                    let mut perturbed = game.clone();
                    perturbed.redeterminize_hidden(GameSeed::from_u64(redetermination));
                    if perturbed.transition(&action).is_err() {
                        witness = Some((game, action));
                        break 'seeds;
                    }
                }
            }
        }
        let (game, action) =
            witness.expect("a staged action should depend on its observed prelude");
        let hash = canonical_complete_action_hash(&action).unwrap();
        let trajectory = simulate_pattern_prior_horizon(
            &game,
            31,
            &action,
            &hash,
            0,
            1,
            HorizonPatternPolicyConfig::default(),
        )
        .unwrap();
        assert_eq!(
            trajectory.state.completed_turns(),
            game.completed_turns() + 2
        );
    }

    #[test]
    fn weighted_selection_is_stable_at_boundaries() {
        assert_eq!(weighted_index(&[1.0, 1.0], 0.0).unwrap(), 0);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.49).unwrap(), 0);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.50).unwrap(), 1);
        assert_eq!(weighted_index(&[1.0, 1.0], 0.99).unwrap(), 1);
    }

    #[test]
    fn canonical_action_hash_matches_registered_json_domain() {
        let action = TurnAction::paired(
            MarketSlot::TWO,
            cascadia_game::HexCoord::new(0, 0),
            Rotation::ZERO,
        );
        let mut hasher = Hasher::new();
        hasher.update(ACTION_HASH_DOMAIN);
        hasher.update(&serde_json::to_vec(&action).unwrap());
        assert_eq!(
            canonical_complete_action_hash(&action).unwrap(),
            *hasher.finalize().as_bytes()
        );
    }
}
