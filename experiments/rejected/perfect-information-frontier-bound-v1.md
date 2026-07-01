# Perfect-Information Frontier Bound V1

## Question

Does exact knowledge of future draws make one-step policy improvement over the
existing K8+H6+B8 frontier target-level when the continuation is the frozen
pattern-aware policy?

## Protocol

Each of ten numeric seeds formed a four-seat block. One all-pattern-aware game
provided four baseline scores. Four treatment games rotated one focal
perfect-information oracle through seats 0-3 while the other three seats
remained pattern-aware.

At each focal decision, every K8+H6+B8 candidate preserved the true hidden
stack and bag, then finished the complete game under a common deterministic
pattern-aware continuation. The candidate with the highest exact focal final
base score was selected.

An earlier symmetric smoke was invalidated before the pilot because real
future opponents used the oracle while rollouts assumed pattern-aware. Its
artifact is retained and explicitly excluded from evidence.

## Result

| Metric | Pattern-aware | Focal oracle | Delta |
|---|---:|---:|---:|
| Mean base score | 91.375 | 93.150 | +1.775 |
| Habitat | 28.525 | 28.900 | +0.375 |
| Wildlife | 59.050 | 60.725 | +1.675 |
| Nature Tokens | 3.800 | 3.525 | -0.275 |

Paired 95% CI: `[+0.299,+3.251]`; record 8-0-2 over ten seed blocks. Forty
focal seat scores were measured. The treatment P90 decision latency was
274.609 ms and four seat rotations took 12.457 seconds per seed block.

Wildlife deltas were Bear +11.325, Elk -4.975, Salmon -0.050, Hawk -4.050, and
Fox -0.575.

## Conclusion

Rejected as a target-level mechanism. Perfect future information produced a
real positive policy-improvement signal, but only 93.150 mean, below the
preregistered 97-point diagnostic boundary and 6.850 points below the target.
The result is not a global mathematical upper bound; it shows that one exact
greedy improvement step over the current frontier still inherits severe
single-species allocation error from the frozen continuation.
