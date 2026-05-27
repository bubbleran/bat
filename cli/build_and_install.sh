#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/.local/bin"
BINARY_NAME="bat"
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$SCRIPT_DIR"
DIST_BINARY="$CLI_DIR/dist/$BINARY_NAME"
INSTALL_PATH="$INSTALL_DIR/$BINARY_NAME"

echo "==> Syncing dependencies (dev + packaging groups)..."
(cd "$CLI_DIR" && uv sync --group dev --group packaging)

echo "==> Building with PyInstaller..."
(cd "$CLI_DIR" && uv run pyinstaller --clean --noconfirm bat_cli.spec)

if [ ! -f "$DIST_BINARY" ]; then
    echo "ERROR: expected binary not found at $DIST_BINARY" >&2
    exit 1
fi

echo "==> Build complete: $DIST_BINARY"
echo ""
read -r -p "Install $BINARY_NAME to $INSTALL_PATH? [Y/n] " reply
reply="${reply:-Y}"

case "$reply" in
    [Yy]|[Yy][Ee][Ss])
        mkdir -p "$INSTALL_DIR"
        if [ -w "$INSTALL_DIR" ]; then
            mv "$DIST_BINARY" "$INSTALL_PATH"
            chmod 755 "$INSTALL_PATH"
        else
            sudo mv "$DIST_BINARY" "$INSTALL_PATH"
            sudo chmod 755 "$INSTALL_PATH"
        fi
        echo "==> Done. $BINARY_NAME installed at $INSTALL_PATH"
        case ":$PATH:" in
            *":$INSTALL_DIR:"*) ;;
            *) echo "NOTE: $INSTALL_DIR is not in your PATH. Add it with:"
               echo "      export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
        esac
        ;;
    *)
        echo "==> Skipped install. Binary left at: $DIST_BINARY"
        ;;
esac
