"""
Dataset e aumento de dados.

O aumento é escolhido pela física do problema, não por reflexo:
  - espelhamento horizontal: OK (troca montante/jusante, mas o modelo não deve
    depender disso para reconhecer *material*);
  - espelhamento vertical: PROIBIDO — inverteria "chama em cima, cera embaixo",
    que é a única pista geométrica confiável;
  - variações de ganho/gama/temperatura de cor: essenciais, porque entre ensaios
    mudam exposição, densidade de fuligem na janela e transmitância do vidro —
    é a principal fonte de queda de desempenho de um ensaio para o outro.
"""

from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import N_CLASSES


# --------------------------------------------------------------------------- #
def aug_photometric(img: np.ndarray, rng) -> np.ndarray:
    """img BGR uint8 -> BGR uint8 com ganho, gama, balanço de cor e fuligem."""
    x = img.astype(np.float32) / 255.0
    x *= rng.uniform(0.70, 1.35)                                  # exposição
    x = np.clip(x, 0, 1) ** rng.uniform(0.75, 1.35)               # gama
    x *= rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)  # cor
    if rng.random() < 0.5:                                        # fuligem/vinheta
        H, W = x.shape[:2]
        n = cv2.GaussianBlur(rng.random((H // 8 + 1, W // 8 + 1)).astype(np.float32),
                             (0, 0), 2.0)
        n = cv2.resize(n, (W, H))
        x *= (1.0 - rng.uniform(0.0, 0.40) * n)[..., None]
    if rng.random() < 0.5:                                        # ruído do sensor
        x += rng.normal(0, rng.uniform(0.004, 0.030), x.shape).astype(np.float32)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def aug_geometric(img, mask, conf, rng, out_hw):
    """Recorte aleatório + espelhamento horizontal + pequena rotação."""
    H, W = img.shape[:2]
    oh, ow = out_hw
    sc = rng.uniform(0.80, 1.0)
    ch, cw = min(H, int(H * sc)), min(W, int(W * sc))
    y0 = rng.integers(0, H - ch + 1); x0 = rng.integers(0, W - cw + 1)
    img = img[y0:y0 + ch, x0:x0 + cw]
    mask = mask[y0:y0 + ch, x0:x0 + cw]
    conf = conf[y0:y0 + ch, x0:x0 + cw]

    if rng.random() < 0.5:
        img, mask, conf = img[:, ::-1], mask[:, ::-1], conf[:, ::-1]
    if rng.random() < 0.3:                                        # ±3° de nível
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (cw, ch), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT)
        conf = cv2.warpAffine(conf, M, (cw, ch), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    img = cv2.resize(np.ascontiguousarray(img), (ow, oh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(np.ascontiguousarray(mask), (ow, oh), interpolation=cv2.INTER_NEAREST)
    conf = cv2.resize(np.ascontiguousarray(conf), (ow, oh), interpolation=cv2.INTER_LINEAR)
    return img, mask, conf


# --------------------------------------------------------------------------- #
class SlabDataset(Dataset):
    """
    Espera o layout produzido por `chamacera bootstrap` / `chamacera review`:

        root/
          img/    00042.png       frame BGR
          mask/   00042.png       uint8 {0,1,2}
          conf/   00042.png       uint8 0..255 (opcional; 255 se ausente)

    Rótulos revisados à mão devem ir para `root/mask` sobrescrevendo o
    pseudo-rótulo, com `conf` = 255 — assim eles dominam o gradiente
    naturalmente, sem precisar de peso extra ou de um segundo laço de treino.
    """

    def __init__(self, root: str, size=(256, 512), train: bool = True,
                 seed: int = 0, ids: list[str] | None = None):
        self.root, self.size, self.train = root, tuple(size), train
        self.rng = np.random.default_rng(seed)
        self.ids = ids or sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob(os.path.join(root, "img", "*"))
        )
        if not self.ids:
            raise FileNotFoundError(f"sem imagens em {root}/img")

    def __len__(self):
        return len(self.ids)

    def _read(self, i):
        k = self.ids[i]
        img = cv2.imread(os.path.join(self.root, "img", f"{k}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.root, "mask", f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        cp = os.path.join(self.root, "conf", f"{k}.png")
        conf = (cv2.imread(cp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                if os.path.exists(cp) else np.ones(mask.shape, np.float32))
        return img, mask.astype(np.uint8), conf

    def __getitem__(self, i):
        img, mask, conf = self._read(i)
        if self.train:
            img = aug_photometric(img, self.rng)
            img, mask, conf = aug_geometric(img, mask, conf, self.rng, self.size)
        else:
            oh, ow = self.size
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            conf = cv2.resize(conf, (ow, oh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return (x, torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(np.ascontiguousarray(conf)).float())


def class_weights(root: str, ids=None, cap: float = 8.0) -> torch.Tensor:
    """Pesos ∝ 1/sqrt(freq) — chama e cera não podem ser afogadas pelo fundo."""
    ids = ids or sorted(os.path.splitext(os.path.basename(p))[0]
                        for p in glob(os.path.join(root, "mask", "*")))
    cnt = np.zeros(N_CLASSES, np.float64)
    for k in ids:
        m = cv2.imread(os.path.join(root, "mask", f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            cnt += np.bincount(m.ravel(), minlength=N_CLASSES)[:N_CLASSES]
    w = 1.0 / np.sqrt(np.maximum(cnt, 1.0))
    w = np.clip(w / w.mean(), 1.0 / cap, cap)
    return torch.tensor(w, dtype=torch.float32)
