# Habitat Candidate Union V1 H6

Status: confirmed as a research teacher on 2026-06-10; not promoted.

## Design

The treatment unions exact-immediate K8 with six distinct draft-and-tile
placements ranked by matching habitat edges, current habitat score, and
immediate base score. Candidate generation changes recall only. The existing
fair R4/D4 evaluator remains the sole rollout value.

## Pilot

Seeds 22100-22109:

- K8+H6 mean: 91.500
- paired delta: +1.375
- habitat delta: +1.200
- total wildlife delta: +0.200
- record: 8 wins, 0 ties, 2 losses
- runtime: 6.045 seconds per game

## K8 Confirmation

Seeds 22200-22249:

- K8 mean: 90.670
- K8+H6 mean: 91.760
- paired delta: +1.090
- 95% CI: [+0.558, +1.622]
- habitat delta: +0.725
- total wildlife delta: +0.240
- record: 36 wins, 2 ties, 12 losses
- treatment runtime: 4.812 seconds per game

H6 cleared every preregistered K8 confirmation gate.

## K16 Control

Seeds 22300-22349:

- K16 mean: 91.005
- K8+H6 mean: 91.520
- paired delta: +0.515
- 95% CI: [-0.140, +1.170]
- habitat delta: 0.000
- total wildlife delta: +0.485
- record: 29 wins, 0 ties, 21 losses
- K16 runtime: 7.335 seconds per game
- H6 runtime: 6.236 seconds per game

H6 was faster and directionally stronger than generic K16, but failed the
preregistered superiority and habitat gates. It is retained as a confirmed
research teacher. The promoted interactive product strategy remains K8.
