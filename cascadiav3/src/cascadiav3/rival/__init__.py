"""CPU-only contracts for the Cascadia Rival research program.

Package initialization deliberately imports no device library.  The pre-GPU
surface is limited to artifact validation, fixed-panel statistical inference,
symbolic planning, and an optional CPU-tested training-data collator whose
trainer integration remains held.  Canonical rules-aware bounds are produced
by the Rust ``cascadia-rival`` crate and only verified here.
"""

from .schema import (
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
    RIVAL_BOUND_CERTIFICATE_SCHEMA_ID,
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
    RIVAL_GPU_PERMIT_SCHEMA_ID,
    RIVAL_POLICY_IDENTITY_SCHEMA_ID,
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
    RIVAL_POWER_ENVELOPE_SCHEMA_ID,
    RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
    RIVAL_ROOT_MANIFEST_SCHEMA_ID,
    RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID,
    RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
    RIVAL_TRAINING_VIEW_SCHEMA_ID,
    RivalSchemaError,
)

__all__ = [
    "RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID",
    "RIVAL_BOUND_CERTIFICATE_SCHEMA_ID",
    "RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID",
    "RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID",
    "RIVAL_GPU_PERMIT_SCHEMA_ID",
    "RIVAL_POLICY_IDENTITY_SCHEMA_ID",
    "RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID",
    "RIVAL_POWER_ENVELOPE_SCHEMA_ID",
    "RIVAL_PREFERENCE_SHARD_SCHEMA_ID",
    "RIVAL_ROOT_MANIFEST_SCHEMA_ID",
    "RIVAL_TERMINAL_PAIR_LEDGER_SCHEMA_ID",
    "RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID",
    "RIVAL_TRAINING_VIEW_SCHEMA_ID",
    "RivalSchemaError",
]
