"""RVC inference. Tries `rvc-python` first; if it cannot be imported, falls
back to a thin fairseq-based loader. Returns audio plus its native sample
rate (resampling done by the caller in `pipeline.py`)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np


class RVCBackendError(RuntimeError):
    pass


class RVC:
    def __init__(
        self,
        model_dir: Path,
        base_dir: Path,
        device: str = "cpu",
        pitch_shift_semitones: int = 0,
        pitch_method: str = "rmvpe",
        index_rate: float = 0.5,
    ):
        self.model_dir = Path(model_dir)
        self.base_dir = Path(base_dir)
        self.device = device
        self.pitch_shift_semitones = int(pitch_shift_semitones)
        self.pitch_method = pitch_method
        self.index_rate = float(index_rate)

        self._validate_paths()
        self._impl = self._try_rvc_python() or self._build_vendored()
        self._warmup()

    def _validate_paths(self) -> None:
        for f in ("hubert_base.pt", "rmvpe.pt"):
            p = self.base_dir / f
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing base model: {p}. Download from "
                    f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{f}"
                )
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"RVC model dir not found: {self.model_dir}")
        pth = list(self.model_dir.glob("*.pth"))
        if not pth:
            raise FileNotFoundError(
                f"No .pth in {self.model_dir}. Expected <name>.pth (+ optional .index)."
            )
        self._pth_path = pth[0]
        idx = list(self.model_dir.glob("*.index"))
        self._index_path = idx[0] if idx else None
        if self._index_path is None:
            warnings.warn(
                f"No .index in {self.model_dir}; running without faiss retrieval."
            )

    def _try_rvc_python(self):
        try:
            from rvc_python.infer import RVCInference  # type: ignore
            from rvc_python.modules.vc.utils import load_hubert  # type: ignore
        except Exception:
            warnings.warn(
                "rvc-python not importable; using vendored fairseq-based loader."
            )
            return None
        os.environ.setdefault("HUBERT_BASE_PATH", str(self.base_dir / "hubert_base.pt"))
        os.environ.setdefault("RMVPE_PATH", str(self.base_dir / "rmvpe.pt"))
        impl = RVCInference(device=self.device)
        impl.load_model(str(self._pth_path))
        # Preload HuBERT now so the first inference call doesn't stall.
        try:
            if impl.vc.hubert_model is None:
                impl.vc.hubert_model = load_hubert(impl.config, impl.lib_dir)
        except Exception as exc:
            warnings.warn(f"HuBERT preload failed: {exc!r}; will retry on first inference.")
        return impl

    def _build_vendored(self):
        try:
            import fairseq  # type: ignore  # noqa: F401
        except Exception as exc:
            raise RVCBackendError(
                "Both rvc-python and fairseq are unavailable. Install one of them. "
                "Recommended: `pip install rvc-python`."
            ) from exc

        from rvc_vendored import VendoredRVC

        return VendoredRVC(
            pth_path=self._pth_path,
            index_path=self._index_path,
            hubert_path=self.base_dir / "hubert_base.pt",
            rmvpe_path=self.base_dir / "rmvpe.pt",
            device=self.device,
            index_rate=self.index_rate,
        )

    def _warmup(self) -> None:
        try:
            silence = np.zeros(16000 // 2, dtype=np.float32)
            for _ in range(2):
                self._infer_raw(silence)
        except Exception as exc:
            warnings.warn(f"RVC warmup failed: {exc!r}")

    def _infer_raw(self, x_16k: np.ndarray):
        # For the rvc-python backend, call the numpy-in pipeline directly to
        # avoid the file-based `infer_file` round-trip. The vendored backend
        # exposes a compatible `infer()` method that already takes numpy.
        impl = self._impl
        if hasattr(impl, "vc") and hasattr(impl.vc, "pipeline"):
            import torch

            if impl.vc.hubert_model is None:
                from rvc_python.modules.vc.utils import load_hubert  # type: ignore

                impl.vc.hubert_model = load_hubert(impl.config, impl.lib_dir)

            audio = x_16k.astype(np.float32, copy=False)
            # Normalize peaks the same way rvc-python's vc_single does.
            peak = float(np.abs(audio).max()) / 0.95
            if peak > 1.0:
                audio = audio / peak

            times = [0.0, 0.0, 0.0]
            sid = torch.tensor(0)
            file_index = str(self._index_path) if self._index_path is not None else ""

            audio_opt = impl.vc.pipeline.pipeline(
                impl.vc.hubert_model,
                impl.vc.net_g,
                sid,
                audio,
                "voicebox_rt.wav",
                times,
                self.pitch_shift_semitones,
                self.pitch_method,
                file_index,
                self.index_rate,
                impl.vc.if_f0,
                3,            # filter_radius
                impl.vc.tgt_sr,
                0,            # resample_sr
                0.25,         # rms_mix_rate
                impl.vc.version,
                0.33,         # protect
                None,         # f0_file
            )
            return np.asarray(audio_opt, dtype=np.float32), int(impl.vc.tgt_sr)
        # Vendored fallback path
        return impl.infer(
            x_16k,
            f0_up_key=self.pitch_shift_semitones,
            f0_method=self.pitch_method,
            index_rate=self.index_rate,
        )

    def infer(self, x_16k: np.ndarray) -> tuple[np.ndarray, int]:
        """Returns `(audio, sample_rate)` where audio is float32, mono."""
        if x_16k.dtype != np.float32:
            x_16k = x_16k.astype(np.float32, copy=False)
        return self._infer_raw(x_16k)
