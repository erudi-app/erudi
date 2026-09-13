#!/usr/bin/env bash
# build-mac-silicon.sh
#
# Full build pipeline for Erudi on macOS Apple Silicon (arm64).
#
# Steps:
#   1. Verify prerequisites (Python venv, Node/npm)
#   2. Install PyInstaller into the backend venv (if missing)
#   3. Run PyInstaller with backend-mac-silicon.spec  → backend/dist/backend/
#   4. Copy backend bundle into frontend/             → frontend/backend/
#   5. Run `npm run dist:mac` in frontend/            → electron-builder DMG + zip
#
# Usage (from repo root):
#   bash scripts/build/build-mac-silicon.sh
#
# Optional — sign & notarize the DMG:
#   source .env.notarize   # exports APPLE_ID, APPLE_APP_SPECIFIC_PASSWORD,
#                          #         APPLE_TEAM_ID, APPLE_SIGNING_IDENTITY (see BUILD.md)
#   bash scripts/build/build-mac-silicon.sh
#
# Output:
#   frontend/dist/*.dmg

set -euo pipefail

# ── Helpers ────────────────────────────────────────────────────────────────────
step() { echo -e "\n\033[36m[build]\033[0m   $*"; }
ok()   { echo -e "\033[32m[ok]\033[0m      $*"; }
warn() { echo -e "\033[33m[warning]\033[0m $*"; }
fail() { echo -e "\033[31m[error]\033[0m   $*"; exit 1; }

# ── Path resolution ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_ROOT="$REPO_ROOT/backend"
FRONTEND_ROOT="$REPO_ROOT/frontend"
VENV_PYTHON="$BACKEND_ROOT/venv/bin/python"
VENV_PIP="$BACKEND_ROOT/venv/bin/pip"
BACKEND_SPEC="$BACKEND_ROOT/backend-mac-silicon.spec"
BACKEND_DIST="$BACKEND_ROOT/dist/backend"
FRONTEND_BACKEND="$FRONTEND_ROOT/backend"

step "Erudi macOS Silicon build"
echo "  Repo root : $REPO_ROOT"
echo "  Backend   : $BACKEND_ROOT"
echo "  Frontend  : $FRONTEND_ROOT"

# ── Prerequisites ──────────────────────────────────────────────────────────────
step "Checking prerequisites..."

[ -f "$VENV_PYTHON" ] || \
    fail "Backend venv not found at $VENV_PYTHON.\nRun: bash scripts/dev/backend/setup-mac-silicon.sh"
ok "Backend venv found"

[ -f "$BACKEND_SPEC" ] || \
    fail "backend-mac-silicon.spec not found at $BACKEND_SPEC."
ok "backend-mac-silicon.spec found"

command -v npm >/dev/null 2>&1 || \
    fail "npm not found in PATH. Install Node.js 20."
ok "npm found: $(command -v npm)"

# ── PyInstaller ────────────────────────────────────────────────────────────────
step "Checking PyInstaller..."
if ! "$VENV_PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    step "PyInstaller not installed. Installing..."
    "$VENV_PIP" install pyinstaller || fail "Failed to install PyInstaller."
fi
PYINSTALLER_VERSION=$("$VENV_PYTHON" -m PyInstaller --version 2>/dev/null)
ok "PyInstaller $PYINSTALLER_VERSION"

# ── Build backend with PyInstaller ─────────────────────────────────────────────
step "Building backend with PyInstaller (this takes 5-15 minutes)..."
cd "$BACKEND_ROOT"
"$VENV_PYTHON" -m PyInstaller backend-mac-silicon.spec --noconfirm || fail "PyInstaller build failed."
cd "$REPO_ROOT"

[ -f "$BACKEND_DIST/backend" ] || \
    fail "backend executable not found after PyInstaller build. Check output above."
ok "PyInstaller build complete: $BACKEND_DIST/backend"

# Ensure the binary is executable
chmod +x "$BACKEND_DIST/backend"
ok "Executable permission set"

# ── Copy backend bundle into frontend ─────────────────────────────────────────
step "Copying backend bundle to frontend..."
if [ -d "$FRONTEND_BACKEND" ]; then
    step "Removing old frontend/backend/..."
    rm -rf "$FRONTEND_BACKEND"
fi
cp -r "$BACKEND_DIST" "$FRONTEND_BACKEND"
ok "Backend copied to $FRONTEND_BACKEND"

# ── Install frontend dependencies if needed ────────────────────────────────────
step "Checking frontend node_modules..."
if [ ! -d "$FRONTEND_ROOT/node_modules" ]; then
    step "Installing frontend dependencies..."
    cd "$FRONTEND_ROOT"
    npm install || fail "npm install failed."
    cd "$REPO_ROOT"
fi
ok "node_modules present"

# ── Clean old out/ to prevent stale resources from previous builds ─────────────
OUT_DIR="$FRONTEND_ROOT/out"
if [ -d "$OUT_DIR" ]; then
    step "Removing stale out/ directory..."
    rm -rf "$OUT_DIR"
    ok "Cleaned out/"
fi

# ── Build Electron app + DMG ───────────────────────────────────────────────────
step "Building Electron app and DMG (electron-builder)..."
cd "$FRONTEND_ROOT"
npm run dist:mac || fail "npm run dist:mac failed."
cd "$REPO_ROOT"

# ── Report output ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
ok "Build complete!"
echo ""
echo "Installer output:"
if [ -d "$FRONTEND_ROOT/dist" ]; then
    find "$FRONTEND_ROOT/dist" -name "*.dmg" | while read -r f; do
        echo "  $f"
    done
else
    warn "dist directory not found. Check build output above."
fi
echo "============================================================"
echo ""
echo "To install:"
echo "  Open the DMG and drag Erudi to Applications."
if [ -z "${APPLE_SIGNING_IDENTITY:-}" ]; then
    echo ""
    warn "App is unsigned. On first launch, macOS may block it."
    echo "  → System Settings > Privacy & Security > Allow"
    echo "  To sign: source .env.notarize && bash scripts/build/build-mac-silicon.sh"
fi
echo ""
