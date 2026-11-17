"""
Entrada de dados: vídeo (.mp4/.avi/.mov), pasta de frames ou sequência .tif/.png
exportada de câmera rápida (Photron/Phantom). Calibração espacial e ROI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from glob import glob
from typing import Iterator

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg")


@dataclass
class Calib:
    """Calibração do ensaio. `mm_per_px` vem de um alvo de escala na janela."""
    fps: float = 240.0
    mm_per_px: float = 1.0
    roi: tuple[int, int, int, int] | None = None      # (x, y, w, h)
    t0_frame: int = 0                                  # frame da ignição
    t1_frame: int | None = None                        # frame do fim da queima
    nome: str = "ensaio"
    notas: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Calib":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def px_to_mm(self, v):
        return np.asarray(v, np.float64) * self.mm_per_px

    def px_s_to_mm_s(self, v):
        return np.asarray(v, np.float64) * self.mm_per_px


class VideoSource:
    """Acesso aleatório a frames, seja de vídeo comprimido ou pasta de imagens."""

    def __init__(self, path: str, calib: Calib | None = None):
        self.path = path
        self.calib = calib or Calib()
        self._cap = None
        self._files: list[str] = []

        if os.path.isdir(path):
            self._files = sorted(
                p for p in glob(os.path.join(path, "*"))
                if p.lower().endswith(IMG_EXT)
            )
            if not self._files:
                sub = os.path.join(path, "frames")
                if os.path.isdir(sub):
                    self._files = sorted(
                        p for p in glob(os.path.join(sub, "*"))
                        if p.lower().endswith(IMG_EXT)
                    )
            if not self._files:
                raise FileNotFoundError(f"nenhuma imagem em {path}")
            self.n = len(self._files)
            h, w = cv2.imread(self._files[0]).shape[:2]
        elif path.lower().endswith(VID_EXT):
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise IOError(f"não abriu {path}")
            self.n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if calib is None or calib.fps <= 0:
                self.calib.fps = self._cap.get(cv2.CAP_PROP_FPS) or 240.0
        else:
            raise ValueError(f"formato não suportado: {path}")

        self.full_shape = (h, w)
        x0, y0, ww, hh = self.calib.roi or (0, 0, w, h)
        self.roi = (int(x0), int(y0), int(ww), int(hh))
        self.shape = (self.roi[3], self.roi[2])

    # ------------------------------------------------------------------ #
    def _raw(self, i: int) -> np.ndarray:
        if self._files:
            img = cv2.imread(self._files[i], cv2.IMREAD_UNCHANGED)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self._cap.read()
            if not ok:
                raise IOError(f"falha lendo frame {i}")
        if img is None:
            raise IOError(f"frame {i} vazio")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype == np.uint16:                       # 12/16 bit da câmera
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img[:, :, :3]

    def frame(self, i: int) -> np.ndarray:
        """Frame BGR uint8 já recortado na ROI."""
        x, y, w, h = self.roi
        return np.ascontiguousarray(self._raw(i)[y:y + h, x:x + w])

    def frames(self, idx=None) -> Iterator[tuple[int, np.ndarray]]:
        for i in (range(self.n) if idx is None else idx):
            yield i, self.frame(i)

    def stack(self, idx) -> np.ndarray:
        """(T, H, W, 3) uint8 — cuidado com memória em sequências longas."""
        return np.stack([self.frame(i) for i in idx])

    def t(self, i: int) -> float:
        """Tempo [s] relativo à ignição."""
        return (i - self.calib.t0_frame) / self.calib.fps

    @property
    def burn_range(self) -> range:
        return range(self.calib.t0_frame, self.calib.t1_frame or self.n)

    def close(self):
        if self._cap is not None:
            self._cap.release()

    def __len__(self):
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# --------------------------------------------------------------------------- #
def sample_indices(n: int, k: int, start: int = 0, stop: int | None = None) -> list[int]:
    """k índices uniformemente espalhados em [start, stop) — para frames-âncora."""
    stop = n if stop is None else stop
    if k >= stop - start:
        return list(range(start, stop))
    return list(np.unique(np.linspace(start, stop - 1, k).round().astype(int)))


def detect_burn_window(src: VideoSource, probe: int = 200,
                       q: float = 0.995) -> tuple[int, int]:
    """
    Detecta ignição e apagamento pelo sinal de luminância do topo do quadro
    (onde só há chama). Retorna (t0, t1) em índices de frame.
    """
    idx = sample_indices(src.n, min(probe, src.n))
    sig = []
    for i in idx:
        g = cv2.cvtColor(src.frame(i), cv2.COLOR_BGR2GRAY)
        sig.append(np.quantile(g[: max(4, g.shape[0] // 3)], q))
    sig = np.asarray(sig, np.float32)
    lo, hi = sig.min(), sig.max()
    if hi - lo < 8:                                    # sem transição clara
        return 0, src.n
    on = sig > lo + 0.35 * (hi - lo)
    if not on.any():
        return 0, src.n
    a, b = int(np.argmax(on)), int(len(on) - np.argmax(on[::-1]))
    return int(idx[a]), int(idx[min(b, len(idx) - 1)]) + 1
