# PyInstaller spec for Voicebox.app
# Build: ./venv/bin/pyinstaller --clean --noconfirm Voicebox.spec
# Output: dist/Voicebox.app

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

block_cipher = None

# ── data: bundled assets + ML library resources ───────────────────────
datas = []

# rmvpe.pt — hybrid bundling: ship inside the .app to save user the
# download. HuBERT/ContentVec stays out (downloaded on first run).
import os
if os.path.isfile("models/base/rmvpe.pt"):
    datas.append(("models/base/rmvpe.pt", "models/base"))

# icon (also referenced from BUNDLE below for the Finder icon)
datas.append(("assets/icon.icns", "assets"))

# Third-party packages that ship resources (configs, weights metadata, fonts)
for pkg in (
    "qtawesome",
    "df",                # DeepFilterNet ships model configs
    "deepfilternet",
    "rvc_python",
    "fairseq",
    "librosa",
    "pyworld",
    "torchcrepe",
    "scipy",
    "soundfile",
    "resampy",
):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ── hidden imports — modules loaded dynamically that PyInstaller misses ─
hiddenimports = []
for pkg in (
    "df",
    "df.enhance",
    "df.model",
    "deepfilternet",
    "rvc_python",
    "rvc_python.infer",
    "rvc_python.modules.vc.modules",
    "rvc_python.modules.vc.pipeline",
    "rvc_python.modules.vc.utils",
    "rvc_python.lib.infer_pack.models",
    "rvc_python.lib.rmvpe",
    "rvc_python.lib.audio",
    "rvc_python.configs.config",
    "fairseq",
    "fairseq.tasks",
    "fairseq.checkpoint_utils",
    "fairseq.models.hubert.hubert",
    "fairseq.dataclass",
    "librosa",
    "librosa.filters",
    "pyworld",
    "torchcrepe",
    "praat_parselmouth",
    "parselmouth",
    "faiss",
    "scipy.signal",
    "scipy.special",
    "scipy.io.wavfile",
    "soundfile",
    "resampy",
    "sounddevice",
    "requests",
):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# ── native libraries — pulled in by torch / faiss / portaudio ──────────
binaries = []
for pkg in ("torch", "torchaudio", "faiss", "sounddevice"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass


a = Analysis(
    ["ui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Trim things we don't need to keep .app size sane.
        "tkinter",
        "PyQt5",
        "PySide2",
        "PySide6",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "test",
        "tests",
    ],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Voicebox",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Voicebox",
)

app = BUNDLE(
    coll,
    name="Voicebox.app",
    icon="assets/icon.icns",
    bundle_identifier="app.voicebox",
    version="0.4.1",
    info_plist={
        "CFBundleName": "Voicebox",
        "CFBundleDisplayName": "Voicebox",
        "CFBundleIdentifier": "app.voicebox",
        "CFBundleShortVersionString": "0.4.1",
        "CFBundleVersion": "0.4.1",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "NSMicrophoneUsageDescription": "Voicebox processes your voice in real time and routes it to BlackHole for use in Discord, Zoom, OBS, and other apps.",
        "NSPrincipalClass": "NSApplication",
        # Important: Voicebox is a regular UI app, not a background daemon.
        "LSUIElement": False,
    },
)
