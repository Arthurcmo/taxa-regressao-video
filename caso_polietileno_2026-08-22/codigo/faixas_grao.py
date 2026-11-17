"""Faixas do grão: centro de cada coluna em mm e taxa de regressão numa faixa e num trecho."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

import experimento_p2_v2 as P2
import recompensa_lei as R
from experimento_s_sessao import DT, MARGEM_LINHAS
from taxas_sessao import MM

SAIDA = R.OUT / "faixas"
X_MM = (20 * (np.arange(93) + 0.5) - 10) * MM          # centro de cada coluna, mm desde o início do grão


def linhas() -> dict:
    cl = np.load(P2.E1 / "classico.npz")
    z = np.load(R.OUT / "runs/nenhum_d4/saida.npz")
    return {"2b": (z["baixo"].astype(float), z["cima"].astype(float)),
            "classico": (cl["y_inf"].astype(float), cl["y_sup"].astype(float))}


def taxa_faixa(Yb, Yc, cols, k0, k1):
    """(r, G, encostou) da faixa no trecho. Mediana móvel de 7 amostras no tempo por coluna."""
    ks = np.arange(k0, k1 + 1)
    yb = median_filter(Yb[ks][:, cols], size=(7, 1), mode="nearest")
    yc = median_filter(Yc[ks][:, cols], size=(7, 1), mode="nearest")
    H = (yb - yc).mean(1) * MM
    incl = np.polyfit((ks - k0) * DT, H, 1)[0]
    r = incl / 2
    G = R.G_ox(R.mdot(k0, k1), H.mean())
    encostou = bool((yc.min() <= MARGEM_LINHAS + 3) or (yb.max() >= 593 - MARGEM_LINHAS - 3))
    return float(r), float(G), encostou


def faixa_cols(x0, x1):
    return np.flatnonzero((X_MM >= x0) & (X_MM < x1))
