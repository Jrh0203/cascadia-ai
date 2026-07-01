# R2-MAP John2 storage migration report v1

Date: 2026-06-18

Status: canonical storage, dashboard cutover, and migration Gate 2 complete;
Gate 7 campaign readmission remains incomplete and W7 remains blocked

Authority: ADR 0195 and
`docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md`

## Result

The active storage directive is now:

```text
john2:/Users/john2/cascadia-bench/r2-map-v1
```

John2 owns the only canonical writable campaign filesystem. No active plan or
operator command directs new R2-MAP output to the John1 SSD. Existing
`/Volumes/John_1/cascadia-cluster/r2-map-v1` objects are immutable legacy
evidence and may be imported only by a digest-bound migration manifest.

This report does not authorize W7. Canonical revision 0, storage-supersession
genesis/sequence 1, the continuing dashboard publisher, the v2 serving
projection, and the production API are now live and receipt-bound on the John2
topology. The complete controller/storage/compact-dataset/trainer suite is now
independently green under that topology. Every W0-W6 scientific identity and
the canonical controller projections still require readmission before W7.

The live API now reports `condition=fresh`, `phase=contracts-ready`, and
`source_path=artifacts/cluster/r2-map-dashboard-serving-projection-v2.json` on
both `127.0.0.1:5187` and `100.110.109.6:5187`. John1 detail exposes the exact
verified cleanup identity:

```text
runtime-stage:cleanup-verified run=live-smoke-v1 receipt_sha256=983c5a59599d969b49473379d6d3416843e64e783e2d545ea07b9e537c9a1e0d
```

## Active directives migrated

| Surface | New contract |
|---|---|
| Authoritative plan | John2 canonical root; no fallback; John1 persists only source, <=64-KiB dashboard mirror, and one <=64-MiB phase runtime |
| Architecture plan | Storage/resource section superseded by ADR 0195 |
| ADRs 0193/0194 | Storage clauses explicitly superseded; scientific and recovery clauses retained |
| ADR 0195 | Single John2 filesystem owner, immutable boundary transfers, no SSHFS/tree sync |
| Remote-storage contract | Frozen John2 host/volume/root identity, 100-GiB free floor, 80-GiB campaign/40-GiB run ceilings, framed receipts |
| CLI reference | Canonical controller, bundle, panel, aggregate, and gate paths use John2 |
| Dataset bridge | Canonical indexes/windows/runs use John2; John1 consumes token-verified windows and checkpoints entirely in memory |
| Experience format | John1 shard bytes stream from memory; only the signed/hash-verified executable+manifest runtime stages locally |
| APFS lifecycle | Retired before sparsebundle creation; retained as historical design evidence |
| Headless prompts | John2 endpoint and remote terminal sentinel are mandatory; SSD/APFS path forbidden |
| Headless supervisor | Turn JSONL/stderr stream through anonymous pipes into receipt-verified remote-worker objects; owner, lease, stop, and terminal state are remote; no local temp or SSD writes |
| Dashboard canonical status | John2 `control/dashboard-status.json` |
| Dashboard serving mirror | John1 `artifacts/cluster/r2-map-dashboard-serving-projection-v2.json`, read-only and at most 64 KiB |
| Dashboard API | Production default is the v2 mirror; canonical host, path, BLAKE3, and timestamp are verified |
| John1 runtime visibility | Existing host detail carries full executable/cleanup receipt identities and blocks the generation barrier while cleanup is pending |

## Reference classification

The repository-wide exact-path audit classified every remaining
`/Volumes/John_1/cascadia-cluster/r2-map-v1` reference as follows.

### Must change before W7

No active code path or operator directive selects the SSD. The remaining
`ssd_path` JSON key in `r2_map_commands.rs` is a frozen registration-schema
compatibility name; its value is validated as the canonical John2 path. The
remaining literal old-root test is an intentional rejection case.

The John1 owner has migrated `r2_map_contracts.py`, dataset/training path
guards, controller defaults, reference-panel root imports, and active tool
help/errors to the John2 contract. The canonical publisher now writes only
John2 `control/dashboard-status.json`; it no longer creates a v1 repository
projection. Legacy registration field names such as
`ssd_manifest` remain schema-compatibility identifiers; they do not select a
path or authorize SSD I/O.

### Intentional active negative guards

These references prohibit use of the retired root and must remain:

- authoritative plan, ADR 0195, architecture plan, data bridge, experience
  format, John2 remote-storage contract, CLI reference, dashboard guide, and
  both headless prompts;
- `python/cascadia_mlx/r2_map_contracts.py` legacy-root rejection constant;
- `python/tests/test_r2_map_contracts.py` proof that new SSD writes fail; and
- `tools/test_r2_map_dashboard_fetch.py` proof that the SSH read command cannot
  name the SSD.

### Historical evidence or compatibility

These references preserve prior facts and must not be rewritten as if the old
artifacts had lived elsewhere:

- `docs/v2/reports/r2-map-w0-gate-power-and-reference-panels-v1-preregistration.md`;
- `docs/v2/reports/r2-map-w4-external-macho-gatekeeper-incident-v1.md`;
- `docs/v2/APFS_WORKSPACE_LIFECYCLE.md`, now prominently marked retired;
- `docs/TECH_DEBT.md`, now carrying the ADR 0195 disposition;
- the v1 serving-projection compatibility decoder and its exact legacy path in
  `crates/cascadia-api/src/cluster_r2_map.rs`; and
- unrelated O2 preregistration/result/plist evidence under
  `/Volumes/John_1/cascadia-cluster/john1`, which is outside the R2-MAP campaign
  and outside this migration.

## Dashboard transport evidence

`tools/r2_map_dashboard_fetch.py`:

- invokes only `/usr/bin/ssh` with `BatchMode=yes`,
  `StrictHostKeyChecking=yes`, `ClearAllForwardings=yes`, no TTY, the fixed
  `john2` alias, and one fixed canonical file;
- expands the effective SSH configuration before every read and rejects any
  drift from user `john2`, Tailscale address `100.100.43.38`, key-based-only
  authentication, strict host-key checking, or the registered identity file;
- caps the canonical read before parsing;
- validates schema/campaign identity, exact host set, no John4, finite JSON, and
  update timestamp before replacing the mirror;
- binds exact source bytes with BLAKE3, canonical host/path, canonical update
  time, and fetch time;
- takes an advisory lock on the serving directory without leaving a lock file,
  writes a sibling temporary, fsyncs file and directory, installs mode `0444`,
  and atomically renames;
- preserves the prior valid mirror on transport or validation failure; and
- never reads, enumerates, or writes `/Volumes/John_1`.

The companion launch agent writes stdout/stderr to `/dev/null`, so the
projection is the only local campaign file it creates. The API retains v1
decode compatibility solely for immutable historical projections; production
defaults to v2.

The first live fetch failed closed before creating a projection because macOS
provides `test` at `/bin/test`, not `/usr/bin/test`. The fixed command is covered
by a regression assertion and the production fetch then succeeded. This was a
transport portability defect, not an SSD fallback or partial publication.

Verified locally:

```text
dashboard publisher/fetch tests: 23 passed
live fetcher regression suite after /bin/test repair: 13 passed
R2-MAP campaign panel tests: 3 passed
Rust API dashboard reader: 9 passed
Rust CLI storage boundary: 4 passed
Rust API and CLI no-deps clippy -D warnings: clean
ruff: clean
zsh -n tools/r2_map_headless_resume.sh: clean
plutil -lint tools/com.johnherrick.cascadia.r2-map-dashboard-fetch.plist: OK
```

A later fail-closed audit found that zsh does not propagate process-substitution
failures through the Codex foreground status or a later bare `wait`. The
headless runner now delegates each turn to a Python anonymous-pipe multiplexer
that supervises both John2 sinks, hashes the copied streams, validates both
object/storage receipt identities, and kills Codex on an early sink exit. The
exact three-file receipt, pump, bounded-capture, and supervisor-wiring suite is
15/15 green; the redundant existing dashboard-wiring selector is independently
green. Ruff and `zsh -n` are clean. No campaign file or local temporary is used
by that path.

John2 independently retained the exact three-file source at
`source/headless-integrity-03fde62de30f87e1-v1`, source identity
`03fde62de30f87e12b93ad7d0b697f29161d8cfea3420a957040e345e895da2d`,
transaction-manifest SHA-256
`d8bab5c6573c0b3b9f6b96c4af2a3d9cdefc962fa06668969ab9eb26a144e1ba`,
and source-commit receipt
`control/receipts/req-b6b11949e5f44395bc3699e1bef4eedb.json`, SHA-256
`a924122d20fb456ffe13be4008e40ac71b20bc3d172c2b3380d03d3094185207`.
The canonical run receipt is
`control/receipts/req-e8a77a4f885f45fda2ef55d5cd04f2ed.json`, SHA-256
`605ad06ca179df8417f7dc5a40aec7b76c432c62bafa15ac7f58d759812967df`;
stdout is 1,348 bytes with SHA-256
`3b38e76ae9021cfa6b2162c6cee39fe875a452640f9de7fcb0d645a0548f84e4`,
and stderr is empty. Cleanup committed under
`control/receipts/req-056a43db932d45c08d1357a2dc4e4989.json`, SHA-256
`b562b64ad97b74463ce016188a302c4eca67eef5611e25679ba31bb6ed10184e`.
The durable proof is
`control/storage-cutover/headless-integrity-validation-v1.json`, SHA-256
`eda06c1e00424c1d892870331d9fc16302cf29e5424f1f5f94f674fea26d39e8`;
its publication, reopen, and read receipt SHA-256 values are
`50e9602eb2bcb41d1dfdde653c14d086522fd9f8e2d555669c4233c2445e6044`,
`6d16508f40e47f66a5b50dfbec369d6f696ee21897bab17ad17a683fc6aca2b5`,
and
`8609ed6cc75cd1aa5b917a0340b17e7bacdda6bdc2db63dcbeb1edce6b0821ad`.

The frozen publisher produced four distinct canonical objects across 40,069 ms
with the cleanup detail above. John2 independently reproduced the gate across
50,090 ms. Three post-cutover local and Tailscale API samples across more than
two publisher/fetch intervals stayed fresh, advanced canonical timestamps and
BLAKE3 identities, and remained bound to the v2 source path. Both `/cluster`
URLs returned HTTP 200 and served the R2-MAP panel bundle; its render suite was
3/3 green. Interactive visual capture was unavailable in the headless run
because neither the in-app browser nor the Codex Chrome Extension was present.

Durable evidence:

- complete cutover proof:
  `control/dashboard-cutovers/john2-v2-cutover-v2.json`, SHA-256
  `456d5e0e0430bcaae7319f4b744191254ff791fdf4d77489237262f3d37dffbc`,
  publication receipt
  `control/receipts/req-183af243441f4af3b8e23f6be6b42747.json`, receipt SHA-256
  `65cced55fc6bd267a6a973369bee41330213c8ac362dd8e6dd90fcddfa194f17`;
- complete locator verification (18/18 content or receipt identities):
  `control/dashboard-cutovers/john2-v2-cutover-v2-verification.json`, SHA-256
  `36fd23e6faeb2d6df57584136202f54eb4ee106fbb17bef70dea7b698b36cce4`,
  publication receipt
  `control/receipts/req-d489225af4c24bc7bb34461c1fbf72d7.json`, receipt SHA-256
  `4f8cd3350d836954eaa6bed69f3acf6f0cbd60ea48cdc4dae364ff688fcfec1f`;
- independent publisher proof:
  `control/dashboard-publisher-verifications/contracts-ready-v2-john2-owner.json`,
  SHA-256
  `44ed006fb648d3c9d93a0cf2a803f6a7ff84e5acd478acb93e326452160e0b35`;
- release API bundle:
  `bundles/dashboard-api-710308b95b8f701c-v1/cascadia-api`, SHA-256
  `a40531d61489fa5daff795c14275490f05cecfbbb10ce7118aeb7575a014ad22`;
  and
- legacy writer retirement:
  `control/storage-cutover/legacy-ssd-process-retirement.json`, SHA-256
  `eee0e14aff8dfdd39374fe71849be0997ea69912c526aef3c1ea1c3f77bd267c`.

The immutable v2 proof supersedes v1 only because v1 named the wrong filename
for the legacy-retirement publication receipt; its receipt digest was already
correct. V2 names the actual
`control/receipts/req-0bb9fa09f1014fc38a9a61ec4b74747c.json` object. The
verification report reopened every referenced object, compared all content
SHA-256 values, and recomputed every receipt document SHA-256.

Six exact legacy SSD process groups (16 processes, including three blocked
direct write operations) were retired with `TERM`; no `KILL` was required.
The post-retirement process audit found no legacy-root, v1-projection, or old
headless process. The only loaded campaign dashboard jobs are the v2 fetcher
and production API.

## Required cutover gates

1. **Green.** John2 froze the signed arm64 runtime bundle identity, and John1
   proved the bounded `/private/tmp` stager and exact cleanup receipt without
   retaining local output.
2. **Green.** The complete current-source
   controller/storage/compact-dataset/trainer suite passed 165/165 tests, Ruff,
   three production help gates, exact full-model in-memory MLX
   checkpoint/resume, cleanup, and independent John3 receipt adjudication.
3. **Green, superseded.** No bulk legacy-tree copy is authoritative. Verified
   genesis-anchor continuity now consists of canonical revision 0, the genesis
   decision, storage-supersession sequence 1, and their cutover receipts.
   Legacy files remain immutable provenance and may be imported only when a
   later work packet binds an exact source path, byte count, and digest.
4. **Green.** John2 publishes a valid canonical
   `control/dashboard-status.json` every ten seconds from a frozen controller
   and stable compact input paths.
5. **Green.** The v2 fetcher creates one 2,844-byte mode-`0444` projection;
   Rust API tests, clippy, live endpoint responses, and stale/invalid failure
   modes pass.
6. **Green.** Old v1/SSD jobs were retired; the process, path, and launch-agent
   audit shows only the v2 fetcher and production API remain.
7. **Pending.** Controller state, queue, ledger, decision chain, terminal path,
   and every W0-W6 artifact identity must be reconciled to John2 before W7
   authorization.

### Gate 2 retained component evidence

John2 independently froze and validated the production John1 remote-training
caller plus the full-model filesystem-free checkpoint/resume path:

- source target `source/remote-training-tests-3309d4147ac8e512-v3`, manifest
  SHA-256
  `db8241cd15a99b0141b120ef2d08927cd407fabd311d21a0120894932280fe39`,
  commit receipt
  `control/receipts/req-5bcafd4df62c4fbb8f98792f046f19e6.json`, receipt
  SHA-256
  `2ff16fcfc594871a5abdc780abe85377b4d57d88c332104be825b1836e3e9bbf`;
- canonical run receipt
  `control/receipts/req-99841a993e19420082b7f43fa3f51d5f.json`, SHA-256
  `15cc48a70c90020e046feaa244b5ad89f96365b2a2e0da1a560a8d424f100b39`;
- stdout `reports/remote-training-validation-3309d4147ac8e512-v3/stdout.log`,
  SHA-256
  `32f2fb66bad832739a7be5c36d2a647854f27bd81d8329198097368d105e6c6d`,
  with open/read receipt SHA-256 values
  `d3ec6a7c7a204e47be18ca0152a2040a56cd2f66b5a64b5538dd0e14f8991790`
  and
  `6133702f5c68340df5c3dee82dc33faf10a5929c425d8140caddc7ff63299938`;
- empty stderr SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  and exact run-tree cleanup receipt
  `control/receipts/req-435f33ea274b45a5ad23a0fd0ea49800.json`, SHA-256
  `119c57b8c6d4224da3f898235b270e2b47c456bcb38a674ee15e8fa0b75d3834`;
- durable proof
  `control/storage-cutover/gate-2-remote-training-validation-v3.json`, SHA-256
  `33f561e5f023a88bf0601ea4670182517845f4cb371d547d8e029e79b2c79290`,
  publication receipt
  `control/receipts/req-1ac8b13a81b54123b97b5679c7271d74.json`, receipt
  SHA-256
  `f3833653465971f690f9e511b5c2554cbbb2ddb60f1d6a868b8f04ac85f0ba95`,
  and independent reopen/read receipt SHA-256 values
  `ea6b126909e67186af2b386b8b6688f8c502aef9603dc749cf0f6476d330396c`
  and
  `c7fc2010ee79abce7877466b65c57baa3e7c5fac6f040297862933dd657c96a5`.

The 10/10 run included actual MLX model and optimizer state, an immutable
remote transaction abstraction, verification-bound pointer publication, exact
resume, and tamper rejection. Ruff, the production CLI help path, stderr, and
post-run cleanup were green. This closes only those Gate 2 components; it does
not by itself close the complete controller/storage/compact-dataset/trainer
suite, W2, W3, Gate 2 as a whole, or Gate 7.

John2 then froze the current production tool and ran the full Gate 2 suite.
The retained source is `source/gate2-full-suite-fc5abcc97110807e-v3`, source
identity
`fc5abcc97110807e5a1989cae7440b204c52cec9fc4fc707a7fbfded51e9ff28`,
transaction-manifest SHA-256
`2c59cf094f7c57aa3cc2a4b47d1f1e40a523242a2ff271c84ad5532c271e1687`,
and source-commit receipt
`control/receipts/req-d63e8e4dad05493eb0e5c043275c27be.json`, SHA-256
`9d36f510797b6333eb33e4bd948f72bfbe60c88d63f56bbfa991cf474fc99cb3`.
The retained source manifest and tar are:

- `source/gate2-full-suite-fc5abcc97110807e-v3/source-manifest.json`, object
  SHA-256
  `e96a9591f58b94204d659103cf3b167007e993c036526447cc89642ec285fa4f`,
  internal document SHA-256
  `0e4ab7963a324610fae8cdf6c38ce5689b486b37aab2b57896c8679be13e7117`,
  open receipt
  `control/receipts/req-f126109e09e042d9959138eb708bb222.json`, SHA-256
  `9c09e041121accd769dee1f2f5af4743cd8d0c680b13789d552c25abad82453f`
  and read receipt
  `control/receipts/req-832a6ec91a054f66910c7d3c13d17128.json`, SHA-256
  `b224355c0613c06f1262f4ff3c190b4eafddd1d224d1849cabb26a420f61f90f`;
- `source/gate2-full-suite-fc5abcc97110807e-v3/source.tar`, 4,874,240 bytes,
  SHA-256
  `68000759eee8bfdfd7f6699ee48d314a8b4c1f0cb1efbd78009475a0754cb14d`,
  open receipt
  `control/receipts/req-43f40599627e450d854d2e07df7e270b.json`, SHA-256
  `d802cf08816f304c5694f099f0c74dec4429c3f008c46c2942f1b804c9bf75f0`
  and read receipt
  `control/receipts/req-e7b0ac6867754959986dd701029bee3d.json`, SHA-256
  `5cea8c33390568fc4cf687d837820518d73d620a01aaadea2c645954c64abf66`.

Run `gate2-full-suite-fc5abcc97110807e-v3` passed 165/165 tests across
contracts, controller, storage, dashboard status, compact dataset, checkpoint,
training resources, and remote training. Ruff and trainer/compact/controller
help gates passed; stderr was empty. The run receipt is
`control/receipts/req-462beb707062468d9a64858b50ad3e09.json`, SHA-256
`5d3339b2346e5b490a22d158dfb67889541423f17a745b3b093a57ef9fb45bdc`.
Stdout is
`reports/gate2-full-suite-fc5abcc97110807e-v3/stdout.log`, 11,172 bytes,
SHA-256
`6ae5f76c0dda133c0237a4825b035ab7466e30362ab247f784d297cbfc6842ad`;
its open receipt is
`control/receipts/req-f3f048cee0c646bab8f6964ef65d566c.json`, SHA-256
`462026b18bbfd2572541b5618478710f224c22ba821444b8a3818b69e40b0e7a`
and read receipt is
`control/receipts/req-96377b725fa54348b1f356bfd5f49826.json`, SHA-256
`2cc0e9d18015a5bc2ed82c791a2df434f1ee67ecf9de76300bdd6885a917e9e6`.
Stderr is the corresponding zero-byte `stderr.log`, SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
its open receipt is
`control/receipts/req-9c10f716b5b04015bce4b0ef86a4e4ff.json`, SHA-256
`d49d15b5d1a2d913f13bcca09d592aecbeb03cc2f5b56a49ab392e4a24d85ded`
and read receipt is
`control/receipts/req-ff8c541cc00e4108a72dce30bbd69be7.json`, SHA-256
`9a032270222b794176768c98343d91218c8c6f58a732ceca262c11db8573d6de`.
Cleanup prepare receipt
`control/receipts/req-6162cd2062114717a0007471e4d472b2.json` has SHA-256
`4b836f56a2fbb59da289f5a024d8ebab4b44bdf29e1d3b191a29f3911f1c27d6`
and commit receipt
`control/receipts/req-f555e903038f4501b9d3ed4483ecca7c.json` has SHA-256
`a667bf34ebe281c68bb1c71604e2292455777ff9c38ad167b5903be0c58122e0`;
the workspace, basetemp, and registered `tmp`/`build`/`cache` trees were absent
afterward.

The durable proof is
`control/storage-cutover/gate-2-full-suite-validation-v3.json`, SHA-256
`86e5ebcad64fefe75aaed4eae28ffd11f84cd53e14fd70212a9b7dcdab4d652d`.
Its publication receipt is
`control/receipts/req-4c2e2bbbb7a440278fe8a0888b503448.json`, SHA-256
`c5f97dc43fb353425b3ef80829320abf56a87f1f84cd0c96f11ea0f90dd7bcd5`,
its reopen receipt is
`control/receipts/req-79a7bee9f52d4a00b635a5e18fdffb2b.json`, SHA-256
`8c9a354b8a501b45367770b40befcb1e68eb74f3f181a2e90fdefdb38d0d1cee`,
and its read receipt is
`control/receipts/req-829ee89630164512bce6fa7c1182b06c.json`, SHA-256
`9342a5e3c4a8bc1ca40624a319c45fd5e8a2a9bf6a01628af7805e54e4b6c7bc`.
The proof asserts no John1 durable run files, no legacy-SSD I/O, John2-only
bulk/evidence storage, no John4, and no W7 launch. John3 then reopened the
transaction, all 231 source-tar members, run output, all 16 original receipts,
cleanup, proof publication, and proof read chain. The independent verification
is
`control/storage-cutover/gate-2-full-suite-independent-verification-v1.json`,
SHA-256
`27961af2234051e6568c05468cffac5298dded4393797eff031d460bb79974a4`,
internal document SHA-256
`a7c9f28ef14730d804bfc3437ac3e1f18007107ddc758c9a05a1eb95f476c3bc`.
Its publication receipt is
`control/receipts/req-2a41c2304bf94c5999fa37cdd5d04007.json`, SHA-256
`96857bf841984ecc5655216c192f03dc3d13132583bffc2cf91df2942620a6de`;
its independent reopen/read receipts are
`control/receipts/req-9f8e2599fdf242669fa002a57dab6a58.json`, SHA-256
`6e8cd390b3fac9bec8b065d6eb88e8e6b400cb18d94c9b51cbd7d9e9f23e46d0`,
and `control/receipts/req-1e3aa6291a59499fa7c47a950a2a127c.json`,
SHA-256
`afcf7f5fbc2853c34dabecba53d349f78da30a355b9957c224126795d32a81d0`.
Migration Gate 2 is therefore green. Gate 7 remains independently pending.

Gate 7 remains the hard barrier. Dashboard freshness and Gate 2 infrastructure
closure do not authorize W7, make legacy W0-W6 hashes current, or permit SSD
I/O.
