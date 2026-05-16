# Voicebox

Real-time voice pipeline for macOS:

```
Microphone → DeepFilterNet (denoise) → RVC (voice change) → BlackHole 2ch
```

Any app that lets you pick a microphone — Discord, Zoom, OBS, Google Meet, QuickTime — will receive the processed voice when its mic input is set to **BlackHole 2ch**.

Built with PyQt6 on top of `deepfilternet`, `rvc-python`, `fairseq`, `torch` (MPS on Apple Silicon, CPU fallback for unsupported ops). Apple Silicon native; Intel macOS works on CPU.

---

## What it does

- **Low-latency mic capture** at 48 kHz mono via CoreAudio (`sounddevice`).
- **Real-time denoise** with DeepFilterNet3.
- **Real-time voice conversion** with RVC v2:
  - HuBERT or ContentVec content encoder (ContentVec recommended for non-English speech).
  - rmvpe F0 extraction with configurable pitch shift, filter radius, protect, and index rate.
  - Streaming sliding-window inference with equal-power crossfade.
- **Routing** to BlackHole 2ch (virtual audio loopback), so any app can pick the processed voice up as a mic.
- **Native macOS UX**: frameless window with custom title bar, menu-bar tray icon (live indicator), close-to-tray, dock icon, mic permission prompt, Audio MIDI Setup walkthrough.

## Installation (end users)

Download the `.dmg` from the [Releases](#) page (or build it yourself, see below):

1. Open `Voicebox.dmg`, drag **Voicebox.app** to `/Applications`.
2. **First launch:** Right-click Voicebox in `/Applications` → **Open** → **Open** (the bundle is unsigned; macOS Gatekeeper asks once).
3. Open the app, hit **Setup**, click **Download** next to *Base models* (~350 MB).
4. (Optional, recommended for non-English voices) Click **Download ContentVec (multilingual)** under *Voice models* → *Encoder*.
5. Install **BlackHole 2ch** if you haven't already: [Existential Audio · BlackHole](https://existential.audio/blackhole/) or `brew install blackhole-2ch`.
6. In Audio MIDI Setup, create a **Multi-Output Device** (BlackHole + your speakers) so you can hear yourself while routing audio to BlackHole. Voicebox shows a step-by-step guide.
7. Drop a `.pth` + `.index` voice model into the **Voice models** drop zone, or paste a Hugging Face / direct URL.
8. Open **Voice Pipeline**, pick input/output devices and RVC model, hit **Start pipeline**.
9. In your target app (Discord/Zoom/OBS), set the **microphone input** to *BlackHole 2ch*. Keep the output (speakers/headphones) on your normal device.

Find more voice models on [weights.gg](https://www.weights.gg/) or [Hugging Face](https://huggingface.co/models?other=rvc).

## Pipeline knobs (Voice tuning)

- **Pitch shift** (−24..+24 semitones)
- **Protect** (0.00..0.50) — RVC's voiceless-consonant protection. Lower = more original consonants preserved.
- **Filter radius** (0..7) — F0 contour smoothing. Higher = smoother but less expressive.
- **Index rate** (0.00..1.00) — faiss retrieval mix. Only takes effect when the loaded model ships with an `.index`.
- **Denoise** toggle, **Bypass model** toggle.

All knobs update **live** without restarting the engine.

## Latency budget

| Hardware       | End-to-end |
|----------------|-----------|
| M1 / M2 (MPS)  | ~350–400 ms |
| M3 / M4 (MPS)  | ~300–370 ms |
| Intel / CPU    | ~400–500 ms |

Most of the latency is in the 384 ms accumulation window. Lower `--window-ms` (or change in `engine.py`) trades quality for latency.

## CLI

Voicebox also ships a CLI for headless / scripted use:

```bash
./venv/bin/python realtime_voice.py \
    --rvc-model <voice_name> \
    --pitch-shift 0 \
    --output-device <BlackHole device index>
```

Run `--list-devices` to see all CoreAudio devices.

## Build from source

Requires macOS 13+, Apple Silicon recommended.

```bash
brew install python@3.10 cmake portaudio create-dmg
git clone https://github.com/<your-fork>/voicebox.git
cd voicebox
./setup.sh         # creates ./venv, installs deps, downloads HuBERT + rmvpe
./venv/bin/python ui.py
```

To produce a `.app` + `.dmg`:

```bash
./build_app.sh
# → dist/Voicebox.app (~1.6 GB) + dist/Voicebox.dmg (~1.1 GB)
```

The bundle uses ad-hoc signing only. For Apple Developer ID + notarization, edit `Voicebox.spec` and the codesign step in `build_app.sh`.

## Architecture

```
ui.py
 └─ ui_window.MainWindow
     ├─ ui_setup.SetupScreen      ← system checks, models, downloads, log
     ├─ ui_pipeline.PipelineScreen ← routing, voice tuning, transport, telemetry
     └─ tray icon (close-to-tray)

engine.Engine
 ├─ audio_io.AudioIO  ← sounddevice streams + lock-protected ring buffers
 ├─ denoise.Denoiser  ← DeepFilterNet3 wrapper
 ├─ rvc.RVC           ← rvc-python primary, vendored fairseq fallback
 └─ pipeline.Pipeline ← sliding-window + equal-power crossfade

system_checks.py   ← homebrew / python / BlackHole / Multi-Output / base models / mic permission
models_manager.py  ← scan / drop / remove RVC voice folders
app_paths.py       ← user-data dir (works in source and frozen .app)
```

Specs and plans live in `docs/superpowers/`.

## Troubleshooting

- **`No module named 'fairseq'`** — `./venv/bin/pip install rvc-python`.
- **`aten::_fft_r2c` not implemented for MPS** — `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in `ui.py` and `realtime_voice.py`).
- **High latency / underruns** — raise `--blocksize` (e.g. 960), disable Denoise, check Activity Monitor.
- **No audio in Discord/Zoom** — verify BlackHole 2ch is the **input** in the target app, and Voicebox's output device is BlackHole.
- **Voice models list squashed** — fixed in v0.4.1: list scrolls inside a fixed-height area.
- **macOS prompts for mic on every Start** — only happens once per process. If it repeats, the bundle identifier may have changed; rebuild the .app.
- **App "frozen" right after Start** — first RVC model load takes 10–25 s on cold start; the transport button shows `⏳ Loading models…`. Subsequent starts are fast (Engine reuses loaded models).
- **`sounddevice` import error / PortAudio not found** — `./venv/bin/pip install --force-reinstall --no-binary :all: sounddevice` or `brew reinstall portaudio`.

## Privacy & permissions

- Voicebox processes audio entirely **locally** — nothing is uploaded.
- It requests **microphone access** the first time you start the pipeline.
- Models, base encoders, and settings live in:
  - From source: the repo dir under `models/` and `~/.config/microphone/`.
  - From the `.app`: `~/Library/Application Support/Voicebox/`.

## Acknowledgements

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — Rikorose et al.
- [Retrieval-based-Voice-Conversion-WebUI / RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
- [rvc-python](https://pypi.org/project/rvc-python/) — Python wrapper for RVC inference.
- [ContentVec](https://github.com/auspicious3000/contentvec) — multilingual content encoder.
- [BlackHole](https://existential.audio/blackhole/) — Existential Audio's virtual audio loopback driver.
- [qtawesome](https://github.com/spyder-ide/qtawesome) — Font Awesome / Material Design icons for Qt.

## License

MIT — see [LICENSE](LICENSE).
