#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "bootstrap_macos.sh supports macOS only" >&2
  exit 1
fi

if [[ -x /opt/homebrew/bin/brew ]]; then
  brew=/opt/homebrew/bin/brew
elif [[ -x /usr/local/bin/brew ]]; then
  brew=/usr/local/bin/brew
else
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

install_formula() {
  local formula=$1
  if "$brew" list --versions "$formula" >/dev/null 2>&1; then
    return
  fi
  if ! "$brew" install "$formula"; then
    "$brew" list --versions "$formula" >/dev/null 2>&1 || return 1
  fi
}

install_formula uv
install_formula node
install_formula rustup

rustup_bin="$("$brew" --prefix rustup)/bin"
uv_bin="$("$brew" --prefix uv)/bin/uv"
node_bin="$("$brew" --prefix node)/bin/node"
"$rustup_bin/rustup" toolchain install 1.94.1 --profile default \
  --component rustfmt --component clippy
"$rustup_bin/rustup" default 1.94.1

echo "uv: $("$uv_bin" --version)"
echo "node: $("$node_bin" --version)"
echo "rustc: $("$rustup_bin/rustc" --version)"
