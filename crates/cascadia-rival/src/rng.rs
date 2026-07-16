use std::fmt;

use blake3::Hasher;
use cascadia_game::GameSeed;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::{PublicRootId, SeatIndex, Sha256Digest};

pub const RNG_CONTRACT_ID: &str = "cascadiav3.rival_rng_domains.v1";

/// Root entropy for a Rival CPU run. This value is never passed to a policy;
/// the factory yields domain-specific capabilities instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RivalSeed([u8; 32]);

impl RivalSeed {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn from_u64(value: u64) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"cascadiav3.rival_seed.v1");
        hasher.update(&value.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }
}

impl fmt::Debug for RivalSeed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RivalSeed(<redacted>)")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationBranch {
    Incumbent,
    Challenger(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    Low,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CouplingGroupId([u8; 32]);

impl CouplingGroupId {
    pub fn from_manifest_key(key: &str) -> Self {
        Self(*blake3::hash(key.as_bytes()).as_bytes())
    }
}

/// A physical-randomness capability that may intentionally be shared by
/// registered high/low and incumbent/challenger branches.
#[derive(Clone)]
pub struct OuterPhysicalRng(ChaCha12Rng);

/// An opaque branch key. It contains no outer physical coupling coordinate.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BranchKey([u8; 32]);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRootSeed([u8; 32]);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedeterminationSeed([u8; 32]);

impl RedeterminationSeed {
    pub(crate) fn game_seed(self) -> GameSeed {
        GameSeed(self.0)
    }

    pub(crate) fn for_world_sample(self, sample_index: u32) -> Self {
        if sample_index == 0 {
            return self;
        }
        let mut hasher = Hasher::new();
        hasher.update(RNG_CONTRACT_ID.as_bytes());
        update_part(&mut hasher, b"redetermination-world-sample");
        update_part(&mut hasher, &self.0);
        update_part(&mut hasher, &sample_index.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }
}

#[derive(Clone)]
pub struct SearchRng(ChaCha12Rng);

#[derive(Clone)]
pub struct PolicyRng(ChaCha12Rng);

#[derive(Clone)]
pub struct TieBreakRng(ChaCha12Rng);

/// Complete identity coordinate for one policy-local random stream. Keeping
/// the coordinate typed and indivisible prevents call sites from silently
/// omitting the panel, seat, fidelity, or sample dimension.
#[derive(Clone, Copy)]
pub struct InnerRngCoordinate<'a> {
    root: &'a PublicRootId,
    panel_id: &'a Sha256Digest,
    branch: EvaluationBranch,
    fidelity: Fidelity,
    acting_seat: SeatIndex,
    replicate: u32,
    sample_index: u32,
}

impl<'a> InnerRngCoordinate<'a> {
    pub const fn new(
        root: &'a PublicRootId,
        panel_id: &'a Sha256Digest,
        branch: EvaluationBranch,
        fidelity: Fidelity,
        acting_seat: SeatIndex,
        replicate: u32,
        sample_index: u32,
    ) -> Self {
        Self {
            root,
            panel_id,
            branch,
            fidelity,
            acting_seat,
            replicate,
            sample_index,
        }
    }
}

macro_rules! impl_rng_capability {
    ($type:ident) => {
        impl RngCore for $type {
            fn next_u32(&mut self) -> u32 {
                self.0.next_u32()
            }

            fn next_u64(&mut self) -> u64 {
                self.0.next_u64()
            }

            fn fill_bytes(&mut self, destination: &mut [u8]) {
                self.0.fill_bytes(destination);
            }

            fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), rand::Error> {
                self.0.try_fill_bytes(destination)
            }
        }

        impl fmt::Debug for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(concat!(stringify!($type), "(<opaque>)"))
            }
        }
    };
}

impl_rng_capability!(OuterPhysicalRng);
impl_rng_capability!(SearchRng);
impl_rng_capability!(PolicyRng);
impl_rng_capability!(TieBreakRng);

/// Domain-separating capability factory.
///
/// Inner methods intentionally do not accept [`CouplingGroupId`] or an outer
/// event index. This makes accidental scenario-key reuse in policy/search RNG
/// construction difficult to express.
#[derive(Debug, Clone)]
pub struct RngFactory {
    master: RivalSeed,
}

impl RngFactory {
    pub const fn new(master: RivalSeed) -> Self {
        Self { master }
    }

    /// Hash-bound identity for evidence records.  The master bytes remain
    /// private, while any seed or domain-contract change produces a new ID.
    pub fn identity_sha256(&self) -> Sha256Digest {
        let mut bytes = Vec::with_capacity(RNG_CONTRACT_ID.len() + self.master.0.len());
        bytes.extend_from_slice(RNG_CONTRACT_ID.as_bytes());
        bytes.extend_from_slice(&self.master.0);
        Sha256Digest::of_bytes(&bytes)
    }

    pub fn source_root(&self, source_game: &[u8], root_ordinal: u32) -> SourceRootSeed {
        SourceRootSeed(self.derive(b"source-root", &[source_game, &root_ordinal.to_le_bytes()]))
    }

    pub fn outer_physical(
        &self,
        coupling_group: CouplingGroupId,
        chance_event_index: u64,
    ) -> OuterPhysicalRng {
        rng(self.derive(
            b"outer-physical",
            &[&coupling_group.0, &chance_event_index.to_le_bytes()],
        ))
        .into()
    }

    pub fn branch(&self, coordinates: InnerRngCoordinate<'_>) -> BranchKey {
        BranchKey(self.inner_key(b"branch", coordinates))
    }

    pub fn redetermination(&self, coordinates: InnerRngCoordinate<'_>) -> RedeterminationSeed {
        RedeterminationSeed(self.inner_key(b"redetermination", coordinates))
    }

    pub fn search(&self, coordinates: InnerRngCoordinate<'_>) -> SearchRng {
        SearchRng(rng(self.inner_key(b"search", coordinates)))
    }

    pub fn policy(&self, coordinates: InnerRngCoordinate<'_>) -> PolicyRng {
        PolicyRng(rng(self.inner_key(b"policy", coordinates)))
    }

    pub fn tie_break(&self, coordinates: InnerRngCoordinate<'_>) -> TieBreakRng {
        TieBreakRng(rng(self.inner_key(b"tie-break", coordinates)))
    }

    fn inner_key(&self, domain: &[u8], coordinates: InnerRngCoordinate<'_>) -> [u8; 32] {
        let (branch_tag, challenger) = match coordinates.branch {
            EvaluationBranch::Incumbent => (0u8, 0u16),
            EvaluationBranch::Challenger(index) => (1u8, index),
        };
        let fidelity_tag = match coordinates.fidelity {
            Fidelity::Low => 0u8,
            Fidelity::High => 1u8,
        };
        self.derive(
            domain,
            &[
                coordinates.root.as_str().as_bytes(),
                coordinates.panel_id.as_str().as_bytes(),
                &[branch_tag],
                &challenger.to_le_bytes(),
                &[fidelity_tag],
                &[coordinates.acting_seat.get()],
                &coordinates.replicate.to_le_bytes(),
                &coordinates.sample_index.to_le_bytes(),
            ],
        )
    }

    fn derive(&self, domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(RNG_CONTRACT_ID.as_bytes());
        hasher.update(&self.master.0);
        update_part(&mut hasher, domain);
        for part in parts {
            update_part(&mut hasher, part);
        }
        *hasher.finalize().as_bytes()
    }
}

impl From<ChaCha12Rng> for OuterPhysicalRng {
    fn from(value: ChaCha12Rng) -> Self {
        Self(value)
    }
}

fn rng(seed: [u8; 32]) -> ChaCha12Rng {
    ChaCha12Rng::from_seed(seed)
}

fn update_part(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use crate::{PolicyMemoryBank, PrivateSimState, PublicRootId, RootKind, SeatLocalMemory};

    use super::*;

    fn root() -> PublicRootId {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(17),
        )
        .unwrap();
        let private = PrivateSimState::new(game).unwrap();
        let memory = SeatLocalMemory::empty();
        let observation = private
            .public_observation(SeatIndex::new(0).unwrap(), memory)
            .unwrap();
        PublicRootId::new(&observation, RootKind::DraftPolicyRoot)
    }

    fn panel(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    #[test]
    fn all_inner_domains_and_coordinates_separate() {
        let root = root();
        let factory = RngFactory::new(RivalSeed::from_u64(9));
        let seat = SeatIndex::new(0).unwrap();
        let panel = panel("coordinate-panel");
        let base = factory.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        assert_ne!(
            base.0,
            factory
                .redetermination(InnerRngCoordinate::new(
                    &root,
                    &panel,
                    EvaluationBranch::Challenger(0),
                    Fidelity::High,
                    seat,
                    0,
                    0,
                ))
                .0
        );
        assert_ne!(
            base.0,
            factory
                .redetermination(InnerRngCoordinate::new(
                    &root,
                    &Sha256Digest::of_bytes(b"different-coordinate-panel"),
                    EvaluationBranch::Incumbent,
                    Fidelity::High,
                    seat,
                    0,
                    0,
                ))
                .0
        );
        assert_ne!(
            base.0,
            factory
                .redetermination(InnerRngCoordinate::new(
                    &root,
                    &panel,
                    EvaluationBranch::Incumbent,
                    Fidelity::Low,
                    seat,
                    0,
                    0,
                ))
                .0
        );
        assert_ne!(
            base.0,
            factory
                .redetermination(InnerRngCoordinate::new(
                    &root,
                    &panel,
                    EvaluationBranch::Incumbent,
                    Fidelity::High,
                    seat,
                    0,
                    1,
                ))
                .0
        );

        let mut policy = factory.policy(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        let mut search = factory.search(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        assert_ne!(policy.next_u64(), search.next_u64());
    }

    #[test]
    fn outer_coupling_choice_cannot_change_inner_policy_stream() {
        let root = root();
        let factory = RngFactory::new(RivalSeed::from_u64(10));
        let seat = SeatIndex::new(0).unwrap();
        let panel = panel("outer-independence-panel");
        let mut outer_a = factory.outer_physical(CouplingGroupId::from_manifest_key("a"), 3);
        let mut outer_b = factory.outer_physical(CouplingGroupId::from_manifest_key("b"), 3);
        assert_ne!(outer_a.next_u64(), outer_b.next_u64());

        let mut policy_before = factory.policy(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            7,
            2,
        ));
        let _ = factory.outer_physical(CouplingGroupId::from_manifest_key("different"), 99);
        let mut policy_after = factory.policy(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            7,
            2,
        ));
        assert_eq!(policy_before.next_u64(), policy_after.next_u64());
    }

    #[test]
    fn seat_memory_changes_root_and_therefore_inner_stream() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(17),
        )
        .unwrap();
        let private = PrivateSimState::new(game).unwrap();
        let mut memories = PolicyMemoryBank::new(4).unwrap();
        let seat = SeatIndex::new(0).unwrap();
        let root_a = PublicRootId::new(
            &private
                .public_observation(seat, memories.get(seat).unwrap().clone())
                .unwrap(),
            RootKind::DraftPolicyRoot,
        );
        memories
            .replace(seat, SeatLocalMemory::new(vec![1, 2, 3]))
            .unwrap();
        let root_b = PublicRootId::new(
            &private
                .public_observation(seat, memories.get(seat).unwrap().clone())
                .unwrap(),
            RootKind::DraftPolicyRoot,
        );
        assert_ne!(root_a, root_b);
    }

    #[test]
    fn domain_derivation_has_locked_cpu_golden_values() {
        let root = root();
        let factory = RngFactory::new(RivalSeed::from_u64(0x5eed));
        let seat = SeatIndex::new(2).unwrap();
        let panel = panel("golden-panel");
        let redetermination = factory.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Challenger(7),
            Fidelity::Low,
            seat,
            11,
            13,
        ));
        assert_eq!(
            crate::digest::encode_lower_hex(&redetermination.0),
            "fbd9468e689e6a5cff3e1bf5b33089b585c45ed7c157ff7016d6f2594a265f38"
        );
        let mut policy = factory.policy(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Challenger(7),
            Fidelity::Low,
            seat,
            11,
            13,
        ));
        assert_eq!(policy.next_u64(), 15_930_090_085_717_842_915);
    }
}
