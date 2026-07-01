# R2-MAP dashboard runtime refresh v1

Date: 2026-06-18 local / 2026-06-19 UTC

## Result

The production dashboard and every relevant API remained available. The R2-MAP
projection returned from `stale` to `fresh` and advanced across multiple
10-second canonical publication intervals.

This is an infrastructure status repair only. It did not execute Cascadia
project code, start research, open a protected seed, run a container, alter a
Docker daemon, or certify D0. D0 and W0 remain RED.

## Cause

John1's authenticated read-only fetch launch agent was healthy and continued to
replace the disposable projection every ten seconds. John2's canonical
publisher was absent. Its prior registered controller run had exited after
`control/dashboard-inputs/host-receipts.json` was temporarily malformed, so the
fetcher correctly mirrored an unchanged canonical payload until its
30-second age threshold expired.

## Repair path

The repair used the existing pinned John1-to-John2 remote-storage transport and
registered John2 controller-run path:

- remote worker SHA-256:
  `ef749dffd3e68a0921b7387af57f1db5a375dbac3cab4fb2092b92270508b3b4`;
- receipt-bound runtime evidence:
  `control/dashboard-inputs/runtime-profile-evidence-20260618-v1.json`, SHA-256
  `005d05ef07f41342132251309b6066b7b41dd6acf7e0edd6635f56a462029a51`,
  compare-and-swap predecessor
  `3a1d10a1bc1523a42b83ce2002ab1f18dd80d572d35637cc145ee7e1852e649d`,
  put receipt SHA-256
  `20e042c321eed2d468197c3a0d66a247f3e9ac95cb67c9579fb2eb0ac829f986`;
- mutable dashboard host input:
  `control/dashboard-inputs/host-receipts.json`, SHA-256
  `508f0e8cca4ca4ba9de681b6035d7c17f415add7f9af2e65bf3b532e9ff93562`,
  compare-and-swap predecessor
  `b41508005db20c5721cdcdfe71e4eb44bbc4efe752028565c3908becf646d889`,
  put receipt SHA-256
  `e4f087ba38b25e1fe747a0df068f725477e6b5923ab694116d80e80a11fa1dd5`;
- frozen controller source:
  `source/controller-freeze-e0069ff580ac0349-v1`, manifest SHA-256
  `ba44a7705f35ed1ae31fe7b0366eb90d64d748a9b01656162124e9a94f7f65d0`;
- one-shot validation run:
  `dashboard-publisher-runtime-state-smoke-v2`, run receipt SHA-256
  `c9979845ad7f9c7baacae08684615b51957a4ba248ef99eb036b493ade3db112`;
- live publisher run:
  `dashboard-publisher-runtime-state-v1`, request ID
  `req-run-controller-7329e4b5c1ecb09821f542af7efad333`, publishing every ten
  seconds with a 30-second freshness threshold; and
- the existing John1 fetch launch agent remains the sole read-only serving
  projection writer.

The version-1 dashboard schema has no Docker-specific fields. Existing
`hosts.<host>.detail` fields therefore carry only bounded current summaries.
The complete observed state and exact runtime receipt identities remain in the
separate receipt-bound runtime-evidence document rather than being added as invented
projection fields.

## Truth represented

- John1: macOS 26.2/25C56, sealed system volume broken, selected runtime absent,
  blocked on OS update/reboot.
- John2: Docker live under `cascadia-r2`, 10 CPUs and 14 GiB, buildx retained,
  runtime receipt file SHA-256
  `09a4327bec341f4cd57c89ba0ef1d38ca5ecad3f38f7731e62714ee9e11105d8`.
- John3: Docker live under `cascadia-r2`, 10 CPUs and 14 GiB, buildx absent,
  runtime receipt file SHA-256
  `41fd6d4b8615098ea019dad0b837a42f08e2f9105144423a5788d5e92ac665b7`.
- Lima hostagent TCP and UDP port 53 are closed on both live hosts and
  unreachable over Tailscale after disabling the host resolver in the shared
  runtime override.
- D0 is non-certified and RED; project execution remains unauthorized.

## Verification

Three API samples spanning two complete publication intervals were all `fresh`.
Canonical update time advanced monotonically through `1781834194189`,
`1781834204207`, and `1781834214222`; observed ages were 4.083, 6.067, and
8.064 seconds against the 30-second threshold. Every sample carried the revised
John2 and John3 receipt hashes plus `lima_tcp_udp53=closed`, while John1 remained
blocked and every host detail retained `d0=RED non-certified`.

The following all returned HTTP 200:

- `/cluster`
- `/api/v1/cluster`
- `/api/v1/cluster/queue`
- `/api/v1/cluster/r2-map`
- `/api/v1/cluster/history?range=1d`
