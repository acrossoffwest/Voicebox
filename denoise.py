"""DeepFilterNet3 wrapper. Always operates at 48 kHz, mono, float32."""

from __future__ import annotations

import numpy as np


class Denoiser:
    SR = 48000

    def __init__(self) -> None:
        try:
            from df.enhance import enhance, init_df  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DeepFilterNet failed to import. Install with `pip install deepfilternet` "
                "or check the DeepFilterNet README."
            ) from exc

        self._enhance = enhance
        self._model, self._df_state, _ = init_df()

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """`x` is mono float32 at 48 kHz. Returns same shape and rate."""
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        y = self._enhance(self._model, self._df_state, x)
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()
        return np.asarray(y, dtype=np.float32)
