#!/bin/sh
set -eu

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "usage: r2_map_remote_rust_verify.sh SOURCE_ARCHIVE [CARGO_HOME RUSTUP_HOME]" >&2
  exit 64
fi

source_archive=$1
case "$source_archive" in
  /Users/john2/cascadia-bench/r2-map-v1/source/*/cascadia-rust-source.tar) ;;
  *)
    echo "source archive is outside the immutable John2 source namespace" >&2
    exit 64
    ;;
esac

: "${CARGO_HOME:?CARGO_HOME is required}"
: "${RUSTUP_HOME:?RUSTUP_HOME is required}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR is required}"
: "${TMPDIR:?TMPDIR is required}"

source_root=$(dirname "$CARGO_TARGET_DIR")/source
/bin/mkdir -m 700 "$source_root"
/usr/bin/tar -xf "$source_archive" -C "$source_root"

if [ "$#" -eq 3 ]; then
  cargo_bin=$2/bin/cargo
  rustc_bin=$2/bin/rustc
  case "$2:$3" in
    /Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-*:/Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-*) ;;
    *)
      echo "borrowed toolchain is outside a frozen John2 run cache" >&2
      exit 64
      ;;
  esac
  [ -x "$cargo_bin" ] && [ -x "$rustc_bin" ]
  export RUSTUP_HOME=$3
  export RUSTUP_NO_UPDATE_CHECK=1
else
  cargo_bin=$CARGO_HOME/bin/cargo
  rustc_bin=$CARGO_HOME/bin/rustc
fi

if [ ! -x "$cargo_bin" ]; then
  installer=$TMPDIR/rustup-init
  /usr/bin/curl --proto '=https' --tlsv1.2 -fsS \
    https://static.rust-lang.org/rustup/dist/aarch64-apple-darwin/rustup-init \
    -o "$installer"
  /bin/chmod 500 "$installer"
  "$installer" -y --profile minimal --default-toolchain stable
fi

if [ "$#" -eq 1 ]; then
  "$CARGO_HOME/bin/rustup" component add clippy
fi

export PATH=$(dirname "$cargo_bin"):$PATH
export RUSTC=$rustc_bin

cd "$source_root"
"$rustc_bin" --version
"$cargo_bin" --version
"$cargo_bin" test -p cascadia-api cluster_r2_map --lib
"$cargo_bin" test -p cascadia-cli-v2 --bin cascadia-v2 r2_map_commands::tests
"$cargo_bin" clippy -p cascadia-api -p cascadia-cli-v2 --all-targets --no-deps -- -D warnings
