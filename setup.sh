#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> Detecting platform"
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "WARN: arch is $ARCH; Apple Silicon (arm64) is the supported target. Continuing."
fi

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew not installed. Install from https://brew.sh and re-run." >&2
  exit 1
fi

echo "==> Checking python@3.10 / cmake / portaudio"
MISSING=()
brew list python@3.10 >/dev/null 2>&1 || MISSING+=("python@3.10")
brew list cmake       >/dev/null 2>&1 || MISSING+=("cmake")
brew list portaudio   >/dev/null 2>&1 || MISSING+=("portaudio")
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "ERROR: missing brew packages: ${MISSING[*]}" >&2
  echo "Run: brew install ${MISSING[*]}" >&2
  exit 1
fi

PY=/opt/homebrew/bin/python3.10
if [[ ! -x "$PY" ]]; then
  PY="$(brew --prefix python@3.10)/bin/python3.10"
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: python3.10 not found via Homebrew" >&2
  exit 1
fi
echo "    python: $PY"

echo "==> Checking BlackHole 2ch"
if ! brew list --cask blackhole-2ch >/dev/null 2>&1; then
  cat <<'EOF'
BlackHole 2ch is NOT installed. Install with:

    brew install blackhole-2ch

Then create a Multi-Output Device:
  1. Open Audio MIDI Setup (Cmd+Space -> "Audio MIDI Setup")
  2. Click "+" -> "Create Multi-Output Device"
  3. Check both "BlackHole 2ch" AND your normal output (e.g. MacBook Speakers)
  4. Right-click the new device -> "Use This Device For Sound Output"

Re-run ./setup.sh after the install.
EOF
  exit 1
fi

echo "==> Creating venv (./venv)"
if [[ ! -d venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY" ./venv
  else
    "$PY" -m venv ./venv
  fi
fi

PIP=./venv/bin/pip
if command -v uv >/dev/null 2>&1; then
  echo "==> Installing requirements (uv)"
  uv pip install --python ./venv/bin/python -r requirements.txt
else
  echo "==> Installing requirements (pip)"
  "$PIP" install --upgrade pip wheel
  "$PIP" install -r requirements.txt
fi

echo "==> Attempting optional rvc-python install"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python ./venv/bin/python rvc-python || \
    echo "WARN: rvc-python install failed; the vendored loader will be used."
else
  "$PIP" install rvc-python || \
    echo "WARN: rvc-python install failed; the vendored loader will be used."
fi

echo "==> Downloading base models"
HF_BASE="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
for f in hubert_base.pt rmvpe.pt; do
  TARGET="models/base/$f"
  if [[ -s "$TARGET" ]]; then
    echo "    skip $TARGET (already present)"
    continue
  fi
  echo "    downloading $f"
  curl -L --fail -o "$TARGET" "$HF_BASE/$f"
done

mkdir -p models/rvc

cat <<'EOF'

==> Setup complete.

Next steps:
  1. Drop your target RVC voice into:    models/rvc/<voice_name>/<voice_name>.pth
                                        models/rvc/<voice_name>/<voice_name>.index
  2. Grant microphone permission: System Settings -> Privacy & Security -> Microphone
     -> allow the terminal/IDE the first time the script asks.
  3. List CoreAudio devices:
         ./venv/bin/python realtime_voice.py --list-devices
  4. Run the pipeline:
         ./venv/bin/python realtime_voice.py \
             --rvc-model <voice_name> \
             --output-device <BlackHole index>
  5. Configure Discord/Zoom/OBS to use "BlackHole 2ch" as the microphone input.

EOF
