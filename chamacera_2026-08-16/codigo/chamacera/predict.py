"""Inferência do aluno com quantificação de incerteza (MC-Dropout)."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from .models import UNet


def load_model(ckpt: str, device: str | None = None, cfg: dict | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg is None:
        cp = os.path.join(os.path.dirname(ckpt), "config.json")
        cfg = json.load(open(cp)) if os.path.exists(cp) else {}
    net = UNet(3, base=cfg.get("base", 24), depth=cfg.get("depth", 4),
               p_drop=cfg.get("p_drop", 0.20)).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net, device, cfg


def _pad_to(x, m=16):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"), h, w


@torch.no_grad()
def predict_frames(net, device, frames, T: int = 12, size=None, batch: int = 2):
    """
    frames: iterável de imagens BGR uint8.
    Retorna (masks, ent, epi) — máscara argmax, entropia total e incerteza
    epistêmica (BALD), todas no tamanho original de entrada.
    """
    masks, ents, epis = [], [], []
    buf, shapes = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        x, h, w = _pad_to(x)
        p, ent, epi = net.predict_mc(x, T=T) if T > 1 else _det(net, x)
        pm = p.argmax(1).cpu().numpy()[:, :h, :w]
        en = ent.cpu().numpy()[:, :h, :w]
        ep = epi.cpu().numpy()[:, :h, :w]
        for k, (H0, W0) in enumerate(shapes):
            masks.append(cv2.resize(pm[k].astype(np.uint8), (W0, H0),
                                    interpolation=cv2.INTER_NEAREST))
            ents.append(cv2.resize(en[k], (W0, H0)))
            epis.append(cv2.resize(ep[k], (W0, H0)))
        buf.clear(); shapes.clear()

    for img in frames:
        H0, W0 = img.shape[:2]
        im = cv2.resize(img, (size[1], size[0])) if size else img
        buf.append(torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
        shapes.append((H0, W0))
        if len(buf) >= batch:
            flush()
    flush()
    return np.stack(masks), np.stack(ents), np.stack(epis)


@torch.no_grad()
def _det(net, x):
    p = torch.softmax(net(x), 1)
    ent = -(p * (p + 1e-9).log()).sum(1)
    return p, ent, torch.zeros_like(ent)
