use crate::{BoundedFeatureView, BoundedToken, BoundedTokenKind, R4Error, Result};

use crate::BoundedArm;

pub const BOUNDED_PARENT_CACHE_SCHEMA_VERSION: u16 = 1;
pub const BOUNDED_PARENT_CACHE_SCHEMA: &str = "r4-bounded-parent-mlx-cache-v1";
pub const BOUNDED_PARENT_EXPERIMENT_ID: &str = "r4-bounded-quotient-mlx-comparison-v1";
pub const BOUNDED_PARENT_PROTOCOL_ID: &str = "r4-bounded-parent-mlx-matched-comparison-v1";
pub const BOUNDED_PARENT_ADR_ID: &str = "0156";
pub const UNIVERSAL_PARENT_CLASS_COUNT: usize = 9;
pub const UNIVERSAL_PARENT_VALUE_WIDTH: usize = 144;
pub const BOUNDED_PARENT_ARMS: [BoundedArm; 3] = [
    BoundedArm::SeatMarginal,
    BoundedArm::Directional,
    BoundedArm::Affordance,
];

pub fn bounded_token_universal_class(kind: BoundedTokenKind) -> Result<u8> {
    match kind {
        BoundedTokenKind::NearCell => Ok(5),
        BoundedTokenKind::HabitatComponent => Ok(6),
        BoundedTokenKind::WildlifeComponent => Ok(7),
        BoundedTokenKind::WildlifeSummary => Ok(8),
        BoundedTokenKind::FrontierSummary => Ok(9),
        BoundedTokenKind::ExactWildlifeBucket | BoundedTokenKind::ExactFrontierBucket => {
            Err(R4Error::InvalidBoundedView(
                "ADR 0156 excludes Q4 exact-bucket token classes".to_owned(),
            ))
        }
    }
}

pub fn bounded_parent_token_owner(view: &BoundedFeatureView, token: &BoundedToken) -> Result<u8> {
    let owner = match token.kind {
        BoundedTokenKind::NearCell => i16::from(view.global.current_relative_seat),
        BoundedTokenKind::HabitatComponent
        | BoundedTokenKind::WildlifeComponent
        | BoundedTokenKind::WildlifeSummary
        | BoundedTokenKind::FrontierSummary => token.values.first().copied().ok_or_else(|| {
            R4Error::InvalidBoundedView("bounded parent token has no seat field".to_owned())
        })?,
        BoundedTokenKind::ExactWildlifeBucket | BoundedTokenKind::ExactFrontierBucket => {
            return Err(R4Error::InvalidBoundedView(
                "ADR 0156 excludes Q4 exact-bucket tokens".to_owned(),
            ));
        }
    };
    let owner = u8::try_from(owner).map_err(|_| {
        R4Error::InvalidBoundedView(format!("bounded parent token seat {owner} is invalid"))
    })?;
    if owner >= view.global.player_count {
        return Err(R4Error::InvalidBoundedView(format!(
            "bounded parent token seat {owner} is outside {} active players",
            view.global.player_count
        )));
    }
    if token.values.len() > UNIVERSAL_PARENT_VALUE_WIDTH {
        return Err(R4Error::InvalidBoundedView(format!(
            "bounded parent token width {} exceeds {}",
            token.values.len(),
            UNIVERSAL_PARENT_VALUE_WIDTH
        )));
    }
    bounded_token_universal_class(token.kind)?;
    Ok(owner)
}
