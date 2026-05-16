"""Manage voice model directories under ./models/rvc/.

A voice model lives in `models/rvc/<name>/` and is expected to contain
`<name>.pth` (the generator) plus an optional `<name>.index` (faiss
retrieval). Many community models also include `.npy`, `.json`, or
config files — those are ignored."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoiceModel:
    name: str
    path: Path
    has_pth: bool
    has_index: bool
    size_mb: float

    @property
    def full(self) -> bool:
        return self.has_pth and self.has_index

    @property
    def files_label(self) -> str:
        if self.has_pth and self.has_index:
            return ".pth + .index"
        if self.has_pth:
            return ".pth only"
        return "incomplete"

    @property
    def size_label(self) -> str:
        return f"{self.size_mb:.1f} MB"


def list_voice_models(rvc_dir: Path = Path("models/rvc")) -> list[VoiceModel]:
    if not rvc_dir.is_dir():
        return []
    result: list[VoiceModel] = []
    for sub in sorted(p for p in rvc_dir.iterdir() if p.is_dir()):
        pths = list(sub.glob("*.pth"))
        idxs = list(sub.glob("*.index"))
        if not pths and not idxs:
            continue
        size = sum(p.stat().st_size for p in sub.iterdir() if p.is_file())
        result.append(
            VoiceModel(
                name=sub.name,
                path=sub,
                has_pth=bool(pths),
                has_index=bool(idxs),
                size_mb=size / (1024 * 1024),
            )
        )
    return result


def remove_model(name: str, rvc_dir: Path = Path("models/rvc")) -> None:
    target = rvc_dir / name
    if not target.is_dir():
        raise FileNotFoundError(target)
    shutil.rmtree(target)


def open_models_folder(rvc_dir: Path = Path("models/rvc")) -> None:
    rvc_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["open", str(rvc_dir.resolve())], check=False)


def _basename_no_ext(path: Path) -> str:
    return path.stem


def accept_drop(paths: list[Path], rvc_dir: Path = Path("models/rvc")) -> list[Path]:
    """Move dropped `.pth` and `.index` files into per-voice subdirs.

    Rules:
    - `<name>.pth` lands in `<rvc_dir>/<name>/<name>.pth`.
    - `<name>.index` lands in the matching `<rvc_dir>/<name>/` if it exists,
      otherwise alongside in a fresh `<rvc_dir>/<name>/` keyed by the
      file's basename.
    - Anything else is ignored and not returned.

    Returns the list of destination paths that were created/updated."""
    rvc_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []

    pth_paths = [p for p in paths if p.suffix.lower() == ".pth"]
    index_paths = [p for p in paths if p.suffix.lower() == ".index"]

    for pth in pth_paths:
        name = _basename_no_ext(pth)
        target_dir = rvc_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / pth.name
        shutil.move(str(pth), dest)
        moved.append(dest)

    for idx in index_paths:
        name = _basename_no_ext(idx)
        # try exact name match first
        target_dir = rvc_dir / name
        if not target_dir.is_dir():
            # fall back to fuzzy: if any existing voice dir name appears as a
            # substring of the index filename, use that one.
            candidates = [d for d in rvc_dir.iterdir() if d.is_dir() and d.name in name]
            if candidates:
                target_dir = max(candidates, key=lambda d: len(d.name))
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / idx.name
        shutil.move(str(idx), dest)
        moved.append(dest)

    return moved
