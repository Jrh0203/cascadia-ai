# Cascadia V2 Clean Checkout Rehearsal

Verdict: **PASS**

## Source

- Commit: `8610d1cb218b5fac7fb18b266d0e183a0d74f454`
- Branch source: `codex/cascadia-v2`
- Checkout: fresh Git clone from a bundle, detached at the exact commit
- Host: `john3`
- Clean path: `/Users/john3/cascadia-clean-8610d1c`

## Host Runtime

| Tool | Version |
|---|---|
| macOS | 26.4 |
| uv | 0.11.21 |
| CPython | 3.12.13 |
| MLX | 0.31.2 |
| Node.js | 26.3.0 |
| Rust | 1.94.1 |

## Commands

```bash
make bootstrap
make setup
make check
git status --short --branch
```

## Results

- Rust format and strict no-dependency Clippy passed.
- Ruff format and cache-independent lint passed.
- ESLint passed.
- 223 Rust unit, integration, property, fixture, and doc tests passed.
- 125 Python tests passed.
- 7 Vitest tests passed.
- Generated CLI reference freshness passed.
- All 11 versioned performance gates passed.
- Playwright passed 5 applicable desktop/mobile flows; 3 project-inapplicable
  cases were explicitly skipped.
- The final Git status was exactly `## HEAD (no branch)` with no modified or
  untracked source files.

## Defects Found And Permanently Fixed

The fresh node exposed five hidden host assumptions:

1. Host prerequisites were undocumented and not provisioned.
2. Apple GNU Make 3.81 resolved simple recipes against the pre-export SSH
   `PATH`.
3. npm and Cargo subcommands needed their runtime tool directories preserved.
4. Browser tests depended on a separately installed system Chrome.
5. Ruff's local cache concealed one stale import-order defect, and Playwright
   overwrote tracked report screenshots during ordinary tests.

The repository now has an idempotent macOS bootstrap, explicit resolved tool
paths, a pinned Playwright Chromium install, cache-independent Python lint, and
separate ordinary-test versus visual-report screenshot destinations.
