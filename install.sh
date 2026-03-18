#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/Drlucaslu/louter.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BUILD_DIR="$(mktemp -d)"

info()  { printf "\033[1;34m==>\033[0m \033[1m%s\033[0m\n" "$*"; }
error() { printf "\033[1;31merror:\033[0m %s\n" "$*" >&2; exit 1; }

# ── Check dependencies ──
command -v git   >/dev/null || error "git is required. Install it first."
command -v cargo >/dev/null || error "Rust is required. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
command -v node  >/dev/null || error "Node.js is required. Install via: https://nodejs.org"
command -v npm   >/dev/null || error "npm is required (comes with Node.js)."

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

# ── Clone & Build ──
info "Cloning louter..."
git clone --depth 1 "$REPO" "$BUILD_DIR/louter"
cd "$BUILD_DIR/louter"

info "Building frontend..."
cd web && npm install --silent && npm run build && cd ..

info "Building louter (release)..."
cargo build --release --quiet

# ── Install ──
mkdir -p "$INSTALL_DIR"
cp target/release/louter "$INSTALL_DIR/louter"

# Ensure INSTALL_DIR is in PATH
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    SHELL_RC=""
    case "${SHELL:-}" in
        */zsh)  SHELL_RC="$HOME/.zshrc" ;;
        */bash) SHELL_RC="$HOME/.bashrc" ;;
    esac
    if [ -n "$SHELL_RC" ]; then
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
        info "Added $INSTALL_DIR to PATH in $SHELL_RC (restart shell or run: source $SHELL_RC)"
    else
        info "Add $INSTALL_DIR to your PATH manually."
    fi
fi

info "Installed louter to $INSTALL_DIR/louter"
echo ""
echo "  Start:   louter"
echo "  Web UI:  http://localhost:6188"
echo ""
