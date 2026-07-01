# R2-MAP Gate 7 readmission reconciliation audit v1

Date: 2026-06-18

Status: **red; W7 is not authorized**

Authority: `docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md`, ADR 0193,
ADR 0195, and the resumed headless objective

## Verdict

The John2 storage and dashboard cutover is healthy, but it is not a scientific
or controller readmission. The canonical campaign is still revision 0 in
`contracts-ready`. Its state and two-entry decision chain are valid. Its queue,
ledger, controller history, controller schemas, controller dashboard inputs,
and W0 v1/v1.1 registration objects are absent. No consolidated current-source
W0-W6 evidence packet exists.

The absent terminal and stop objects are correct for an incomplete campaign.
No terminal sentinel may be created, no protected or final seed material may
be opened, and no W7 packet or phase transition may be installed from this
state.

The dashboard's `legal_next_transitions=["bootstrap-generating"]` describes
only the state-machine edge. It is not authorization to cross the external
W0-W6/Gate 7 barrier.

## Canonical control-plane inventory

The sole authoritative root is
`john2:/Users/john2/cascadia-bench/r2-map-v1`. All observations below used the
receipt-producing remote-storage boundary; no SSD path was read or written.

| Object | Observation | Identity or absence receipt |
|---|---|---|
| `control/campaign-state.json` | present, 752 bytes, mode `0600`; revision 0, `contracts-ready` | file SHA-256 `82bf6193895082392b237b45ed77ecf2a6b23b8349d5a21594e6c5bd701ed8f6`; logical state SHA-256 `f408d914186899efad68643a1c78d71a0e48f3a7994e4ce596378a50423e1a93`; open receipt `control/receipts/req-c729bb2c413a4d16a87bbcddabd4e7b5.json`, SHA-256 `16180af50aa69ebd5e5439c92bcb2005af40baa518472739c0713e3da0a726b9` |
| `control/decision-log.jsonl` | present, 2,684 bytes, mode `0600`; two-entry chain valid | file SHA-256 `444ee21d4c8cb6e1987724398069db087d5dabd2522e8d304c6c5e0ad6065678`; genesis `20145ee311f38e38878ae6e8652d8d6cc9a7b4ba696bdbe7df55406899ef5691`; head `698d465d0b3214133f09d23c59d305a209f615f2cc8ffdb46132b9e21ade9f9d`; open receipt `control/receipts/req-07232aec94a94da192e480dbc27d8a21.json`, SHA-256 `28952a0c6f23a2149bfde3973464a586316e7fc976c9434292dddc57765e0d79` |
| `control/research-queue-v1.json` | absent | `control/receipts/req-c53466512f234609be198659a6b07cc1.json`, SHA-256 `ab43e7e2eeb10d3cef6892d7adb0ca1cc23380c7365770c812ddd08949700ada` |
| `control/research-experiments-v1.json` | absent | `control/receipts/req-21c648248a2c403a9051ace89fa4a5fd.json`, SHA-256 `ff59b35ed60adba512cd49fa3d4f20084b4e0e58f324f4071d46fd9660c5e634` |
| `control/work-packets` | absent; therefore no phase packet exists | `control/receipts/req-acf0bd0a81174e998800eefff533fadf.json`, SHA-256 `d11068898063a6b7c75a43954e9e4f1dccfdd16bcd31f801ac848d95534ff263` |
| `control/incoming-receipts` | absent; therefore no incoming work receipt exists | `control/receipts/req-0ecd6af0d8e54054afdbb92bc0b03594.json`, SHA-256 `196ab08f1905264d3174a3e1340572b4ef75207b3847e0f2d6afa262f3048d8d` |
| `control/controller-history.jsonl` | absent | `control/receipts/req-bede76d5fca74d25b8b87a4a52c0025c.json`, SHA-256 `4b4c369f3398169844056b1f517c626e947bfeb59563bff97587e4e80b6608b2` |
| `control/controller-stop.json` | correctly absent | `control/receipts/req-505d94ef2a4f492e893c0bdfd53319b8.json`, SHA-256 `1ce4185bb520d60f652004981068731747dda69df0bd9a2e862196659e8fa748` |
| `control/headless-terminal.json` | correctly absent | `control/receipts/req-d7fcba4b36f6491cabca6268286c3460.json`, SHA-256 `a81b6407ce56ac63054269738f90aa7ba20d8d77fa5434895b13444da06e32bf` |
| `control/headless-STOP` | correctly absent | `control/receipts/req-6a8fb78fc21f403f881616eb85dc9533.json`, SHA-256 `91ea36b0faac2bd8052aa81581c07a03a5ccdf4f993dde102e34ae5a4297ee53` |
| `control/contracts/r2-map-work-packet-v2.schema.json` | absent | `control/receipts/req-1ae39800be4a476ba97fb8889e325ef5.json`, SHA-256 `99eaa52640750bb621fa7377666ae61302290ae75af86c03c655067fec9a06cb` |
| `control/contracts/r2-map-work-receipt-v2.schema.json` | absent | `control/receipts/req-ed4937e0e211475f996af8c338e7aad1.json`, SHA-256 `8e4166808dc1154d94f56528474927280fcc7f0fa0a7c66700096d2f08d488bb` |
| `control/dashboard-inputs/host-receipts.json` | present as the stable cutover input, 416 bytes, mode `0600` | SHA-256 `a12545f57acabc4df3f6e2987ed15b07a96464a79c40da7b7722c13021393614`; open receipt `control/receipts/req-c4a94765b1014f279108d62a6f9c659b.json`, SHA-256 `94c7b87e398191ca02b10f9f09a83cce5403593fa6859815676b5c1b67bc1ed7` |
| `control/dashboard-inputs/controller.json` | absent | `control/receipts/req-662098a175b54f0391342e01566a347f.json`, SHA-256 `421214119b21b0d72712e4910d3b6f5fdadd58a821e1d75a6cb277dd927954ff` |
| `control/dashboard-inputs/training-progress.json` | absent | `control/receipts/req-2dc95446a5024d52a6f9d899839c21e1.json`, SHA-256 `ab58d66522c87b9b7b2a348348148173b74b285eefd5b9dc4a00ceb7bf3f0aa9` |
| `control/dashboard-inputs/benchmark-aggregate.json` | absent | `control/receipts/req-89a3e9a8960a40d69f440455ac543839.json`, SHA-256 `808392f9e1fcd062c608f84df1485257dd083e3ea8e4b17b6e510b4283b5b246` |

The current dashboard host-receipts file is valid cutover evidence. It is not a
substitute for the four controller projections that `initialize_controller()`
and `reconcile()` must create.

## W0 preregistration inventory

The frozen v1 repository manifest remains byte-exact at SHA-256
`12555a92ab337eca8d299210e19f5c4bb52298822e82f688ad967ceeaed1f7ec`.
It is an immutable stale predecessor and the Rust initializer is required to
reject it after the sequential-public-market v1.1 repair.

The current repository does not contain
`docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json`. John2 also
lacks all four registered objects:

| Object | Absence receipt |
|---|---|
| `control/w0-preregistration/reference-panel-manifest-v1.json` | `control/receipts/req-e733037eaddb4b6abe111ce77ab3057d.json`, SHA-256 `147d38ef451453652d388ce55b21b7f70a81e3d23167ebeab1f604ae2ebf2aa3` |
| `control/w0-preregistration/reference-panel-manifest-v1.1.json` | `control/receipts/req-9e2509ccbe2f4a63b7f8c899ed62eab3.json`, SHA-256 `c63982bd6b0503f91d9d58de5c158327a1119e3b7cb4f75cb0615833454165e5` |
| `control/w0-preregistration/registration.json` | `control/receipts/req-dc8ae632813844e091629bdf989c0868.json`, SHA-256 `f2dd25b437590969bc7ff9c12ececa360e764b915430ec086689e71cc0cdafdd` |
| `control/w0-preregistration/registration-v1.1.json` | `control/receipts/req-3a1093c8bfd7420ab554c5e4e43b0e7a.json`, SHA-256 `429f031a1748227caed047d90511318bc39b07f2ae9ba0ba20da333432de1cd5` |

`docs/v2/CLI_REFERENCE.md` still shows the superseded v1 manifest and
registration in its open-panel initializer example. That command must not be
used for readmission; the documentation and frozen source must name v1.1.

## Current source and component evidence

There is no single current-source transaction that binds W0-W6. Existing
transactions are scoped component evidence:

- Dashboard/controller publisher:
  `source/controller-freeze-e0069ff580ac0349-v1`, manifest SHA-256
  `ba44a7705f35ed1ae31fe7b0366eb90d64d748a9b01656162124e9a94f7f65d0`.
  It contains seven runtime files, not the complete controller/storage test
  closure.
- Remote storage platform:
  `source/remote-storage-tests-b61b0293d37c958a-v1`, manifest SHA-256
  `05716a9f576ff901fbf11770f2d5c38a3b1f36990d8cfd67d61d446d43d3fe6b`.
  All five objects reopened with the registered hashes and modes. Run receipt
  `control/receipts/req-9e812a85bb01484492dbc603ad37edef.json`, receipt
  SHA-256 `71e09114c6a7152bf9acbe10760e92becdaa35bd47eac200538571354c16664d`,
  binds exit 0 and stdout SHA-256
  `f3789767709e86430673c9991341c89da4ce02f58944fe553ff64e33a3466c48`:
  44 tests passed. Its stderr contains only uv/Python package installation
  below John2's canonical root.
- Remote training and full-model in-memory resume:
  `source/remote-training-tests-3309d4147ac8e512-v3`, manifest SHA-256
  `db8241cd15a99b0141b120ef2d08927cd407fabd311d21a0120894932280fe39`,
  commit receipt `control/receipts/req-5bcafd4df62c4fbb8f98792f046f19e6.json`,
  receipt SHA-256
  `2ff16fcfc594871a5abdc780abe85377b4d57d88c332104be825b1836e3e9bbf`.
  Run receipt `control/receipts/req-99841a993e19420082b7f43fa3f51d5f.json`,
  receipt SHA-256
  `15cc48a70c90020e046feaa244b5ad89f96365b2a2e0da1a560a8d424f100b39`,
  and stdout SHA-256
  `32f2fb66bad832739a7be5c36d2a647854f27bd81d8329198097368d105e6c6d`
  prove 10 tests passed, including actual-MLX model/optimizer checkpoint
  transaction, verification-bound pointer, exact filesystem-free resume,
  tamper rejection, deterministic validation selection, Ruff, and the
  production CLI help boundary. Stderr is empty. Cleanup receipt
  `control/receipts/req-435f33ea274b45a5ad23a0fd0ea49800.json`, receipt
  SHA-256 `119c57b8c6d4224da3f898235b270e2b47c456bcb38a674ee15e8fa0b75d3834`,
  reports the run build/cache/tmp inventory absent.

The remote-training v3 evidence remains immutable supporting evidence. A later
review found and repaired O(checkpoint-count) John1 bundle retention by using
incremental deterministic best selection with at most the current and best
serialized bundles. The repaired `tools/r2_map_john1_train.py` SHA-256 is
`39a6a0a472437a9aedb6af303e065aa083724755361f5d8b59a9ebc10f94e963`.

John2 then froze the complete current Gate 2 closure at
`source/gate2-full-suite-fc5abcc97110807e-v3`, transaction-manifest SHA-256
`2c59cf094f7c57aa3cc2a4b47d1f1e40a523242a2ff271c84ad5532c271e1687`.
Its 4,874,240-byte source tar has SHA-256
`68000759eee8bfdfd7f6699ee48d314a8b4c1f0cb1efbd78009475a0754cb14d`;
all 231 manifest members independently matched exact path, hash, size, and
mode, with no links, special files, or path escapes. The run passed 165/165
tests, Ruff, and the trainer/compact/controller production help gates with
empty stderr and complete cleanup. The durable proof is
`control/storage-cutover/gate-2-full-suite-validation-v3.json`, SHA-256
`86e5ebcad64fefe75aaed4eae28ffd11f84cd53e14fd70212a9b7dcdab4d652d`.

### Gate 2 adjudication

Migration Gate 2 is green. John3 independently reopened the three-object
source transaction, validated every one of the 231 tar members, checked the
current trainer hash, recomputed the proof and receipt document hashes,
reopened all 16 original source/run/read/cleanup/publication receipts, and
verified the 165/165 stdout, empty stderr, workspace/basetemp removal, and
zero-entry build/cache inventory. The independent verification is
`control/storage-cutover/gate-2-full-suite-independent-verification-v1.json`,
SHA-256
`27961af2234051e6568c05468cffac5298dded4393797eff031d460bb79974a4`,
publication receipt
`control/receipts/req-2a41c2304bf94c5999fa37cdd5d04007.json`, receipt
SHA-256 `96857bf841984ecc5655216c192f03dc3d13132583bffc2cf91df2942620a6de`.
This closes the migration infrastructure gate only. It is not a W2 model/data
pass, a W3 scientific training/promotion pass, aggregate W0-W6 readmission, or
W7 authorization.

## W0-W6 reconciliation

| Work package | Current classification | Exact readmission evidence still required |
|---|---|---|
| W0 | red | One complete source freeze; byte-identical repository/John2 v1.1 manifest; digest-bound import of the immutable v1 predecessor; v1.1 registration; exact Python regeneration; Rust source rehash; explicit v1 rejection; all schemas/panels/power identities; protected-seed fields remain unopened. |
| W1 | red | Immutable gate packet for heterogeneous seats, exactly one newest seat, deterministic historical pool, independent RNG domains, compact replay, focal-only extraction, Pinecone accounting, and sequential-public-market v1.1 behavior; exact tests, logs, storage receipts, and independent verification. |
| W2 | red | Complete model/dataset gate packet for lazy replay-to-R2 materialization, losses, imitation, auxiliaries, public-only and D6 validation, resource ceilings, exact source closure, and independent verification. The remote-training v3 packet is only supporting infrastructure. |
| W3 | red | Complete trainer/checkpoint/verifier/promoter gate packet, exact failure injection and recovery, standalone verification, all-in-memory John1 boundary, resource/swap evidence, storage receipts, and independent verification. The v3 resume component is green but insufficient. |
| W4 | red | Complete exhaustive serving/direct-gameplay packet for sequential free replacement and paid wipes, grouped protocol, exact action mapping, model registry, maximum-width panel, no hidden refill observation, no pruning, exact logs/receipts, and independent verification. The prior Gatekeeper incident is historical evidence only. |
| W5 | red | Complete focal benchmark, order-independent aggregator, score/P10/P90 and animal/habitat/Pinecone distributions, ledger payload, dashboard integration, missing/duplicate/drift rejection, exact receipts, and independent verification. Dashboard transport health alone is not W5. |
| W6 | red | Gate 2's controller/storage suite is green. W6 still requires current controller schemas; isolated John2 dry run covering every transition/recovery shape; storage-owner initialization of empty queue and ledger; revision-0 reconciliation; controller dashboard inputs; zero phase packets/tasks; one pending ledger projection; no stop; exact report/receipts and independent verification. |

Legacy SSD W0-W6 hashes remain provenance only. They cannot satisfy a row
without an explicit migration manifest binding legacy path, byte count,
digest, John2 destination, and verified publication receipt.

## Exact readmission order

1. Freeze one complete W0-W6 source/test/documentation transaction on John2.
   Component freezes may remain supporting evidence, but the aggregate must
   bind one current implementation and include every runtime resource,
   fixture, test, runner, lockfile, and schema generator.
2. Complete the W0 v1.1 append-only repair. Preserve/import the v1 manifest and
   registration by their frozen hashes, publish the byte-identical v1.1
   repository/John2 manifest and registration, run Python exact regeneration
   and Rust live-source checks, and prove v1 launch rejection. Do not open a
   protected domain.
3. Produce one immutable evidence packet each for W1-W5. Every packet must bind
   the aggregate source manifest, exact commands, stdout/stderr hashes, test
   result, relevant artifact/schema hashes, RSS and swap observations,
   public-only/D6/sequential-market/no-pruning invariants, publication receipt,
   and independent verifier identity.
4. Retain the independently green Gate 2 full-suite packet by exact identity as
   infrastructure evidence. Do not reinterpret it as a W2, W3, or aggregate
   W0-W6 scientific pass.
5. Run W6's isolated synthetic dry run beneath John2 and publish its report.
   Then perform the explicit storage-owner, idempotent canonical controller
   initialization at revision 0. It must create the two controller schemas, an
   empty queue, a ledger with exactly the one pending R2-MAP projection, and
   the four controller dashboard inputs without creating a history record,
   stop, W7 packet, or phase transition.
6. Reconcile and re-read the canonical state, decision chain, schemas, queue,
   ledger, packet/receipt projection, dashboard inputs, controller history
   absence, stop absence, and terminal absence. The state and decision head
   must remain the identities in this audit.
7. Publish one Gate 7 aggregate manifest listing every W0-W6 packet and storage
   receipt, followed by independent John2 and John3 verification. It must say
   explicitly that protected/final seeds remain unopened and W7 has not run.
8. Only after root accepts that aggregate may a separately recorded bounded
   authorization install the bootstrap transition and W7 work packets.

## Dashboard single-writer health

The dashboard path is healthy and phase-stable:

- exactly one local supervisor group: PID/PGID `71365`, command
  `.venv/bin/python tools/r2_map_remote_storage.py controller-run --specification -`;
- exactly one strict SSH child: PID `71366`, PGID `71365`, fixed John2 worker;
- decoded run ID `dashboard-publisher-contracts-ready-v2`;
- source `source/controller-freeze-e0069ff580ac0349-v1`, manifest SHA-256
  `ba44a7705f35ed1ae31fe7b0366eb90d64d748a9b01656162124e9a94f7f65d0`;
- canonical publisher argv is `publish-dashboard-status --watch
  --interval-seconds 10 --stale-after-seconds 30` with the exact John2
  host-receipts path;
- zero old `contracts-ready-v1`, legacy SSD, v1 projection, or headless writer
  match;
- the local fetcher launch agent is running as PID `71672` and the production
  API as PID `71677`;
- both `http://127.0.0.1:5187/api/v1/cluster/r2-map` and
  `http://100.110.109.6:5187/api/v1/cluster/r2-map` remained `fresh`, reported
  `contracts-ready`, carried the exact John1 cleanup receipt identity, and
  advanced across publisher intervals; and
- publisher monitor session `31422` remains live and must not be interrupted.

This health result validates the cutover and single-writer boundary only. The
publisher currently consumes the stable cutover host-receipts file because
the W6 controller projections do not exist. After revision-0 controller
initialization, the publisher input must be reconciled to the controller-owned
projection without creating a second writer.

## Prohibited actions while red

- Do not run W7, bootstrap generation, training, a candidate gate, or a
  protected/final seed workflow.
- Do not create `control/headless-terminal.json`, `control/controller-stop.json`,
  or `control/headless-STOP` for ordinary incompleteness.
- Do not treat dashboard freshness, a legal state-machine edge, component
  tests, or legacy SSD hashes as aggregate authorization.
- Do not write, migrate, timestamp, lock, or use the legacy SSD as fallback.
- Do not use John4.
