# O1 High-Regret Draft-Ranking Integration v1 Label-Coverage Amendment

Experiment: `o1-high-regret-draft-ranking-integration-v1`

Date: 2026-06-17

Status: frozen mechanical amendment before any arm-level validation result was
computed or inspected

## Discovery

The strict exact-R2 top-64 validation cohort was built exactly as
preregistered. During the first end-to-end evaluator smoke, the metric layer
failed closed because five of the 240 validation decisions contain no
R4800-labeled action inside the frozen top-64 cohort.

Mechanical mask census:

- validation decisions: 240;
- decisions with at least one retained R4800 label: 235;
- decisions with zero retained R4800 labels: 5;
- zero-label cohort rows: 32, 114, 115, 122, and 210;
- every validation decision retains all 64 R1200 labels.

This census reads only frozen target-availability masks. It does not read or
compare any arm prediction, regret, recall, score, or treatment effect.

## Frozen Resolution

1. Every arm must still score every one of the 15,360 fixed validation
   candidates exactly once.
2. Mean top-1 retained R4800 regret and retained R4800-winner recall use the
   235 decisions where that endpoint is defined.
3. The five zero-R4800-label decisions are reported explicitly as
   `r4800_scorable=false`; they are not assigned an invented zero, fallback,
   or worst-case R4800 value.
4. R1200 pairwise ordering accuracy continues to use all 240 decisions.
5. Phase and condition slices report both total groups and R4800-scorable
   groups.
6. Paired bootstrap and high-regret analyses use the common scorable group set
   established by the frozen Z0 control.
7. All original effect-size and non-regression thresholds remain unchanged.

This amendment repairs an undefined endpoint without changing cohort
membership, model inputs, labels, optimization, treatment selection, or the
sealed-test boundary.
