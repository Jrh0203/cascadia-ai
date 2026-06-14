# Local Web Product

The v2 web application is a standalone React/TypeScript client backed by the
canonical `cascadia-api` Rust application service. It does not reproduce game
rules in JavaScript.

## Run

```bash
make web-dev
```

Open `http://127.0.0.1:5187`. The cluster operations view is available at
`http://127.0.0.1:5187/cluster`.

For a production build and one local server:

```bash
make web-build
cargo run --release -p cascadia-api
```

Open `http://127.0.0.1:8787`.

The always-on john1 dashboard uses the release server at port 5187 through
`tools/com.johnherrick.cascadia.dashboard.plist`; see
[`CLUSTER_DASHBOARD.md`](CLUSTER_DASHBOARD.md) for build, bootstrap, and
restart commands.

## Product Surface

- 1-4 player setup with human and local AI seats
- deterministic numeric seeds
- configurable A-D wildlife cards and habitat bonuses
- paired and independent market drafts
- free three-of-a-kind replacement and paid wildlife refreshes
- legal habitat guidance, six rotations, and legal wildlife targets
- exact canonical scoring for every board
- undo, redo, automatic local persistence, JSON save/load, and replay history
- instant, interactive, and research AI strength tiers
- promoted pattern-aware and explicitly experimental terminal suggestions
- ranked candidate analysis with exact score and future opportunity value
- responsive desktop and mobile workspaces
- live john1/john2/john3 utilization, health, readiness, and workload telemetry
- persistent one-day and seven-day CPU and memory history

The local save document contains a schema version, the human-readable numeric
seed, and the canonical replay. Every API request reconstructs and validates
the game from that replay before applying work.

## Architecture

`cascadia-api` exposes versioned JSON endpoints. Game requests remain
stateless; cluster telemetry has an explicit local seven-day journal:

- `GET /api/v1/health`
- `GET /api/v1/cluster`
- `GET /api/v1/cluster/history?range=1d|7d`
- `GET /api/v1/capabilities`
- `POST /api/v1/games/new`
- `POST /api/v1/games/view`
- `POST /api/v1/games/turn-options`
- `POST /api/v1/games/placement-options`
- `POST /api/v1/games/apply`
- `POST /api/v1/games/undo`
- `POST /api/v1/games/suggest`

Market preparation and draft placement are staged through engine APIs, while a
complete turn is still committed as one canonical `TurnAction`. The browser
never receives hidden bag order.

The cluster endpoint runs a fixed, read-only macOS probe locally and through
the `john2` and `john3` SSH aliases. Hostnames and commands are not supplied by
the browser. Nodes are sampled in parallel with bounded SSH connection and
keepalive settings, and an unreachable worker is returned as an offline node
instead of failing the whole response. A john1 background task records one
sample every 30 seconds under `artifacts/cluster/`, including while the
dashboard is closed. History responses are bounded and server-downsampled for
the 1D and 7D charts. See
[`CLUSTER_DASHBOARD.md`](CLUSTER_DASHBOARD.md) for setup and telemetry details.

The `instant` tier uses exact immediate-score greedy play. The `interactive`
tier uses promoted `pattern-aware-v1-k8-h6-b8-m4` through the same Rust entry
point used by CLI benchmarks and tests. It evaluates a bounded immediate,
habitat, and Bear frontier using exact score plus the expected best legal
one-token marginal from public unplaced wildlife supply. It never receives
hidden stack or bag order.

The `research` tier exposes
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
Before the final-five cutoff it is exactly the interactive policy. During the
last five personal turns, it evaluates the original K8+H6+B8 frontier over
eight shared public-information determinizations and only replaces the exact
pattern-aware anchor when a challenger's one-sided 90% paired lower bound is
positive. That policy was previously promoted, but fresh requalification after
fixing canonical hidden-state redetermination found a +0.520 total-score gain
paired with a -0.375 non-Bear wildlife regression. It therefore remains
available for explicit research use and is not presented as the strongest
product policy.

## Verification

```bash
make web-test
```

The gate runs ESLint, Vitest, and Playwright against the real Rust API in
desktop Chrome and a mobile Chrome viewport. Browser evidence is generated at:

- `docs/v2/reports/web-desktop-play.png`
- `docs/v2/reports/web-desktop-analysis.png`
- `docs/v2/reports/web-mobile-market.png`
- `docs/v2/reports/web-cluster-dashboard.png`
- `docs/v2/reports/web-cluster-dashboard-mobile.png`

The dependency lockfile is audited with:

```bash
npm --prefix apps/web audit --audit-level=moderate
```
