# Real-Time Voice Pipeline (macOS)

Microphone → DeepFilterNet (denoise) → RVC (voice change) → BlackHole 2ch.
Any app on the system can then use "BlackHole 2ch" as a microphone source.

## One-command setup

```bash
./setup.sh
```

This installs the venv, Python deps, downloads HuBERT + rmvpe base models,
and prints the manual steps for BlackHole and microphone permission.

If `python3.10`, `cmake`, `portaudio`, or `Homebrew` are missing, the script
exits with instructions.

## Models

Base models are downloaded to `models/base/`:
- `hubert_base.pt`
- `rmvpe.pt`

Target voice models live under `models/rvc/<voice_name>/`. Each voice needs
a `.pth` (the generator) and ideally a `.index` (faiss retrieval). Two
common sources:

- Hugging Face: <https://huggingface.co/lj1995/VoiceConversionWebUI>
- Community RVC models: <https://www.weights.gg/>

## BlackHole + Multi-Output Device

1. `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (Cmd+Space → "Audio MIDI Setup").
3. Click `+` → **Create Multi-Output Device**.
4. In the new device, check **both** "BlackHole 2ch" and your normal output
   (e.g. "MacBook Pro Speakers").
5. Right-click the Multi-Output Device → **Use This Device For Sound Output**.

Now BlackHole receives whatever the script emits, while you still hear it
through the speakers.

## Use with Discord / Zoom / OBS

- **Discord:** Settings → Voice & Video → Input Device → **BlackHole 2ch**.
- **Zoom:** Settings → Audio → Microphone → **BlackHole 2ch**.
- **OBS:** Sources → Audio Input Capture → **BlackHole 2ch**.

In each app, keep the **output** as your speakers/headphones (or the
Multi-Output Device), not BlackHole.

## Run

List devices:

```bash
./venv/bin/python realtime_voice.py --list-devices
```

Full pipeline (replace `my_voice` with the subdir under `models/rvc/`):

```bash
./venv/bin/python realtime_voice.py \
    --rvc-model my_voice \
    --pitch-shift 0
```

Useful flags:

- `--input-device N` — pick mic by index from `--list-devices`.
- `--output-device N` — pick output (auto-detects BlackHole if not set).
- `--device mps|cpu|auto` — torch backend (default `auto`).
- `--denoise on|off` — toggle DeepFilterNet (default `on`).
- `--bypass` — straight passthrough mic → BlackHole, no ML.
- `--window-ms 256` / `--crossfade-ms 64` — chunking tradeoff.
- `--pitch-shift N` — semitones (positive raises pitch).

Stop with Ctrl+C.

## Expected latency

| Hardware       | End-to-end |
|----------------|-----------|
| M1 / M2 (MPS)  | ~330 ms   |
| M3 / M4 (MPS)  | ~300 ms   |
| Intel / CPU    | ~380–450 ms |

These are dominated by the 256 ms accumulation window. Lower `--window-ms`
trades quality for latency.

## Microphone permission

On first run, macOS prompts for microphone access in
**System Settings → Privacy & Security → Microphone**. The terminal or IDE
running the script must be allow-listed there.

## Troubleshooting

**`No module named 'fairseq'`**
Vendored RVC path needs fairseq. Easiest fix: `./venv/bin/pip install rvc-python`.
If that also fails, building fairseq from source is possible but rvc-python
is the recommended route.

**High latency**
Raise `--window-ms` (e.g. 384), disable denoise (`--denoise off`), or check
Activity Monitor — a CPU-bound process competing for cores will starve the
audio callback.

**No audio in Discord/Zoom**
Verify BlackHole is the **input** in the target app and that the script is
running with `--output-device` pointing at BlackHole. `--list-devices` shows
the indices.

**Mic not detected by the script**
System Settings → Privacy & Security → Microphone → allow your terminal/IDE.
After granting, fully quit and re-launch the terminal so the entitlement is
picked up.

**Audio crackles / dropouts**
Raise `--blocksize` (e.g. 960) and/or increase `in_capacity` / `out_capacity`
in `audio_io.py` (the defaults give 1 s of slack). Underrun counts in the
stats line tell you which buffer is starving.

**`ModuleNotFoundError: No module named 'pkg_resources'`**
`pyworld` (used by rvc-python) still imports `pkg_resources`, which was removed
in `setuptools>=81`. Pin: `./venv/bin/pip install 'setuptools<81'`.

**`sounddevice` import error or "PortAudio not found"**
```bash
./venv/bin/pip install --force-reinstall --no-binary :all: sounddevice
# or
brew reinstall portaudio
```

**arm64 wheel missing for a dep**
Try `./venv/bin/pip install --no-binary <pkg> <pkg>`. As a last resort,
fall back to running the venv under Rosetta.

## File layout

```
.
├── setup.sh                # one-command bootstrap
├── requirements.txt
├── realtime_voice.py       # entry point
├── audio_io.py             # ring buffer + sounddevice I/O
├── denoise.py              # DeepFilterNet wrapper
├── rvc.py                  # RVC loader (rvc-python primary)
├── rvc_vendored.py         # fairseq-based fallback
├── pipeline.py             # windowing + crossfade chain
├── tests/                  # ring buffer + pipeline unit tests
├── docs/superpowers/specs/ # design spec
├── docs/superpowers/plans/ # implementation plan
└── models/
    ├── base/               # hubert_base.pt, rmvpe.pt
    └── rvc/                # <voice>/<voice>.pth + .index
```
