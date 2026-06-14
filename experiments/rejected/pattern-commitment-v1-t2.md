# Pattern Commitment V1 T2

Status: rejected on 2026-06-10.

## Result

The exact two-placement opportunity model passed its runtime smoke, then
scored 91.250 against promoted pattern-aware at 91.925 over ten paired games.

- paired delta: -0.675
- 95% CI: [-1.925, 0.575]
- record: 5-0-5
- Bear: +0.100
- aggregate non-Bear wildlife: +0.375
- habitat: -0.825
- Nature Tokens: -0.325
- runtime: 0.955 seconds per game

## Mechanism Audit

The treatment increased wildlife by 0.475 but paid more in habitat and tokens.
After the result, code audit found that every non-final action received a
two-turn opportunity value, including positions where the acting seat had only
one future turn. That assigns value to setup which cannot be completed before
game end.

## Conclusion

The registered implementation is rejected and no confirmation was run. The
phase bug does not justify reinterpreting its result. A corrected policy must
cap the opportunity horizon by the acting seat's actual turns remaining and
run under a new experiment ID.
