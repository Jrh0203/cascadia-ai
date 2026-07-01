# Conditional Tile Generalization Forensics V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-generalization-forensics-v1`

## Classifications

- Exact observable aliasing: `observable_label_aliasing_not_material`
- Input distribution shift: `input_covariate_shift_not_material`
- Normalized margin specialization: `late_fit_margin_specialization`

## Key Measurements

- Train positive mass under contradictory exact fingerprints:
  `0.0000%`
- Contradictory exact cross-split overlap:
  `0.0000%`
- Tile-query width Jensen-Shannon divergence:
  `0.024838`
- Largest block fraction above absolute SMD 0.50:
  `4.5455%`
- Largest validation outside-support cell fraction:
  `0.0072%`
- Train median margin improvement:
  `+1.7033`
- Validation median margin improvement:
  `-1.1260`
- Train-validation gap expansion:
  `+2.8293`

## Decision

If ADR 0120 fails, the frozen successor is
`structural_regularization`.

Combined scientific BLAKE3:
`a1ee7130e3d086d79d6a55f6474f0ef4d73a80090f6c7eeecad0aecc52f9da09`.
