"""Janelas iguais no tempo para medir a taxa de regressão e o fluxo de oxidante.

Em cada janela, a taxa sai dos extremos da altura média da porta, r = (H_b - H_a) / (2 t),
média das duas faces, e o fluxo é G_ox = ṁ_ox / (L H_médio).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import relogio_e_corte as C
import serie_porta as RL
from taxas_sessao import L_MM

C.mdot = lru_cache(None)(C.mdot)
L_M = L_MM * 1e-3          # largura da porta, m


def janelas_k(H, tm, t_ini, t_fim, n, x_m=0.065):
    """n janelas iguais entre t_ini e t_fim (s desde a ignição), taxa pelos extremos."""
    ks = np.linspace(float(tm.k(tm.ign + t_ini)), float(tm.k(tm.ign + t_fim)), n + 1)
    out = []
    for a, b in zip(ks[:-1], ks[1:]):
        Ha, Hb = RL.H_em(H, a) * 1e-3, RL.H_em(H, b) * 1e-3
        md = C.mdot(tm, a, b)
        Hs = np.array([RL.H_em(H, k) for k in np.linspace(a, b, 41)]) * 1e-3
        r = (Hb - Ha) / (2 * (b - a) * C.CAM * tm.s)            # m/s
        w = dict(t=[float(tm.desde_ign(tm.banco(a))), float(tm.desde_ign(tm.banco(b)))],
                 Ha=Ha, Hb=Hb, Hm=0.5 * (Ha + Hb), md=md, r=r, x=x_m,
                 Ginst=float(np.mean(md / (L_M * Hs))))
        if np.isfinite(r) and np.isfinite(md):
            out.append(w)
    return out
