#!/usr/bin/env bash
# Build Voicebox.app (PyInstaller) + Voicebox.dmg (create-dmg).
# Requires: venv already populated (run ./setup.sh first) and create-dmg
# installed via Homebrew (`brew install create-dmg`).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=./venv/bin/python
PYINSTALLER=./venv/bin/pyinstaller

if [[ ! -x "$PYINSTALLER" ]]; then
  echo "ERROR: PyInstaller not in venv. Run: ./venv/bin/pip install pyinstaller" >&2
  exit 1
fi

if ! command -v create-dmg >/dev/null 2>&1; then
  echo "ERROR: create-dmg not found. Run: brew install create-dmg" >&2
  exit 1
fi

echo "==> Cleaning previous build"
rm -rf build dist Voicebox.app Voicebox.dmg

echo "==> Building Voicebox.app via PyInstaller"
"$PYINSTALLER" --clean --noconfirm Voicebox.spec

if [[ ! -d dist/Voicebox.app ]]; then
  echo "ERROR: build failed — dist/Voicebox.app not produced" >&2
  exit 1
fi

APP_SIZE=$(du -sh dist/Voicebox.app | awk '{print $1}')
echo "==> Built: dist/Voicebox.app ($APP_SIZE)"

# Optional: ad-hoc code-sign the bundle so Gatekeeper at least sees a
# consistent signature (does NOT replace Developer ID notarization but
# stops some crashes on M-series due to corrupt unsigned binaries).
echo "==> Ad-hoc signing"
codesign --force --deep --sign - dist/Voicebox.app || \
  echo "WARN: ad-hoc sign failed; continuing anyway"

echo "==> Building Voicebox.dmg via create-dmg"
create-dmg \
  --volname "Voicebox" \
  --volicon "assets/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 128 \
  --icon "Voicebox.app" 175 200 \
  --app-drop-link 425 200 \
  --no-internet-enable \
  "dist/Voicebox.dmg" \
  "dist/Voicebox.app/"

DMG_SIZE=$(du -sh dist/Voicebox.dmg | awk '{print $1}')
echo ""
echo "==> Done."
echo "    app: dist/Voicebox.app ($APP_SIZE)"
echo "    dmg: dist/Voicebox.dmg ($DMG_SIZE)"
echo ""
echo "First-launch instructions for users (because the bundle is unsigned):"
echo "  1. Open Finder, navigate to the .dmg, drag Voicebox.app to Applications."
echo "  2. In Applications, right-click Voicebox → Open → Open. macOS asks once."
echo "  3. Subsequent launches: normal double-click."
echo ""
