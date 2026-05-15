"""Minimal vendored RVC inference path. Only loaded when rvc-python is
unavailable; relies on fairseq for HuBERT and on the rmvpe checkpoint for F0.

This implementation is intentionally minimal: it loads the checkpoints,
runs HuBERT feature extraction (CPU if MPS rejects an op), runs the rmvpe
pitch predictor, and feeds both to the generator from the .pth file. faiss
retrieval is applied to HuBERT features when an .index is provided."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch


class VendoredRVC:
    OUTPUT_SR = 40000

    def __init__(
        self,
        pth_path: Path,
        index_path: Path | None,
        hubert_path: Path,
        rmvpe_path: Path,
        device: str = "cpu",
        index_rate: float = 0.5,
    ):
        self.pth_path = Path(pth_path)
        self.index_path = Path(index_path) if index_path else None
        self.hubert_path = Path(hubert_path)
        self.rmvpe_path = Path(rmvpe_path)
        self.device = device
        self.index_rate = float(index_rate)

        self._torch_device = torch.device(device if device != "auto" else "cpu")
        self._hubert = self._load_hubert()
        self._f0 = self._load_rmvpe()
        self._net_g, self._sr = self._load_generator()
        self._index = self._load_index() if self.index_path else None

    def _load_hubert(self):
        from fairseq import checkpoint_utils  # type: ignore

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [str(self.hubert_path)], suffix=""
        )
        model = models[0]
        try:
            model = model.to(self._torch_device).eval()
        except Exception:
            warnings.warn("HuBERT could not move to selected device; using CPU.")
            self._torch_device = torch.device("cpu")
            model = model.to(self._torch_device).eval()
        return model

    def _load_rmvpe(self):
        try:
            from rvc_python.lib.rmvpe import RMVPE  # type: ignore
        except Exception:
            from rmvpe import RMVPE  # type: ignore
        return RMVPE(str(self.rmvpe_path), is_half=False, device=str(self._torch_device))

    def _load_generator(self):
        cp = torch.load(str(self.pth_path), map_location="cpu")
        try:
            from rvc_python.lib.infer_pack.models import SynthesizerTrnMs256NSFsid  # type: ignore
        except Exception:
            from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid  # type: ignore
        net_g = SynthesizerTrnMs256NSFsid(*cp["config"], is_half=False)
        net_g.load_state_dict(cp["weight"], strict=False)
        net_g.eval().to(self._torch_device)
        sr = cp.get("sr") or self.OUTPUT_SR
        return net_g, int(sr)

    def _load_index(self):
        import faiss  # type: ignore

        try:
            return faiss.read_index(str(self.index_path))
        except Exception as exc:
            warnings.warn(f"faiss could not load {self.index_path}: {exc!r}; continuing without retrieval.")
            return None

    def infer(self, x_16k: np.ndarray, f0_up_key: int = 0, f0_method: str = "rmvpe", index_rate: float = 0.5):
        with torch.no_grad():
            wav = torch.from_numpy(x_16k).float().to(self._torch_device).unsqueeze(0)
            feats = self._hubert.extract_features(source=wav, padding_mask=None, output_layer=12)[0]

            if self._index is not None and index_rate > 0:
                npy = feats[0].cpu().numpy().astype(np.float32)
                _, ids = self._index.search(npy, 1)
                retrieved = np.stack([self._index.reconstruct(int(i)) for i in ids[:, 0]])
                blended = retrieved * index_rate + npy * (1 - index_rate)
                feats = torch.from_numpy(blended).to(self._torch_device).unsqueeze(0)

            f0 = self._f0.infer_from_audio(x_16k, thred=0.03)
            f0 = f0 * (2 ** (f0_up_key / 12.0))
            f0_t = torch.from_numpy(f0).to(self._torch_device).float().unsqueeze(0)
            f0_coarse = torch.clamp(
                (1127 * torch.log(1 + f0_t / 700)).round(), 1, 255
            ).long()

            audio = self._net_g.infer(
                feats,
                torch.LongTensor([feats.shape[1]]).to(self._torch_device),
                f0_coarse,
                f0_t,
                sid=torch.LongTensor([0]).to(self._torch_device),
            )[0][0, 0]
            return audio.cpu().numpy().astype(np.float32), self._sr
