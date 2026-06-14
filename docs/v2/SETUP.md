# Cascadia V2 Setup

## Supported Host

The complete development, web, and MLX workflow is supported on Apple Silicon
macOS. Neural training and Apple-GPU inference require this platform.

Install the Xcode Command Line Tools and Homebrew once:

```bash
xcode-select --install
```

Homebrew installation instructions are maintained at
[brew.sh](https://brew.sh).

## Bootstrap

From a clean checkout:

```bash
make bootstrap
make setup
make mlx-device
make check
```

`make bootstrap` installs `uv`, Node.js, and `rustup` through Homebrew, then
installs the pinned Rust 1.94.1 toolchain with `rustfmt` and Clippy. It is
idempotent and does not modify shell startup files. The Makefile includes both
Apple Silicon and Intel Homebrew paths, including the keg-only `rustup`
proxies, so the same commands work from Terminal and noninteractive SSH.

`make setup` creates the locked uv environment from `uv.lock`, installs the web
dependencies from `package-lock.json`, and installs Playwright's pinned
Chromium runtime. Browser tests do not depend on a separately installed Chrome.

## Daily Commands

```bash
make format
make lint
make test
make benchmark-smoke
make web-dev
```

Use `make performance-check` to verify the checked-in product and MLX evidence
against the versioned latency and throughput contract.

## Clean Checkout Acceptance

A clean checkout is accepted only after:

```bash
make bootstrap
make setup
make check
```

The rehearsal report records the source revision, host, tool versions, and
results under `docs/v2/reports/`.
