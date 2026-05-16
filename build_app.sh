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
# Close any Finder window pointing at dist/ to stop it recreating .DS_Store
# mid-clean.
osascript -e 'tell application "Finder" to close (every window whose target as string contains "dist")' 2>/dev/null || true
# Loop until truly empty — Finder may race-write .DS_Store back in.
for path in build dist; do
  for _ in 1 2 3 4 5; do
    [[ -e "$path" ]] || break
    find "$path" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$path" 2>/dev/null || true
    sleep 0.2
  done
done
rm -rf Voicebox.app Voicebox.dmg 2>/dev/null || true

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
DMG_OK=0
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
  "dist/Voicebox.app/" && DMG_OK=1

if [[ "$DMG_OK" != "1" ]]; then
  echo "WARN: create-dmg failed (likely macOS Automation permission missing for Finder)."
  echo "      Falling back to a plain hdiutil DMG without window decorations."
  rm -f dist/Voicebox.dmg
  STAGE=$(mktemp -d)
  cp -R dist/Voicebox.app "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "Voicebox" -srcfolder "$STAGE" \
    -ov -format UDZO "dist/Voicebox.dmg"
  rm -rf "$STAGE"
fi

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
