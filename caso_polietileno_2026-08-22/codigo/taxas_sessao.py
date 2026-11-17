"""Funções comuns sobre as marcações e as linhas de cada método: escala, largura da
porta, matriz das marcações manuais e inclinação de uma reta no tempo.
"""
from __future__ import annotations

import numpy as np

import ensaio as E
import experimento_p2_v2 as P2
from experimento_s_sessao import DT, OUT, marcas, alvo_por_coluna

MM = E.MM_POR_PX
L_MM = E.LARGURA_PORTA_MM
GRAO = np.arange(0, 93)
MEIO = np.arange(18, 91)


def matriz_autor(face: str) -> np.ndarray:
    Y = np.full((963, 93), np.nan)
    for k, p in marcas(face).items():
        y, w = alvo_por_coluna(p)
        Y[k, w > 0] = y[w > 0]
    return Y


def metodos() -> dict[str, tuple[str, np.ndarray, np.ndarray | None]]:
    cl = np.load(P2.E1 / "classico.npz")
    out = {"autor": ("Marcações do autor", matriz_autor("baixo"), matriz_autor("cima")),
           "classico": ("Detector clássico", cl["y_inf"], cl["y_sup"])}
    r4 = np.mean([np.load(P2.RUNS / f"R4_s{s}/saida.npz")["viterbi"] for s in range(3)], axis=0)
    out["R4"] = ("R4 (rótulos híbridos)", r4, None)
    leg = np.load(P2.RAIZ / "resultados/rotulos_humanos_face_inferior_v1.npz")
    yleg = np.full((963, 93), np.nan)
    yleg[leg["k"]] = leg["y"][:, 1]
    out["rotulos_1g"] = ("Rótulos de primeira geração", yleg, None)
    out["R1"] = ("R1 (rótulos de primeira geração)", np.load(P2.RUNS / "R1_s0/saida.npz")["bruto"], None)
    for v, rot in (("U", "U (ImageNet, 11 quadros)"), ("F", "F (ajuste fino com 11 quadros)")):
        p = RAIZ_E3 / f"runs/{v}_d0/saida.npz"
        if p.exists():
            out[v] = (rot, np.load(p)["viterbi"], None)
    for v in ("S1", "S2"):
        pb, pc = OUT / f"runs/{v}_baixo_s0/saida.npz", OUT / f"runs/{v}_cima_s0/saida.npz"
        if pb.exists() and pc.exists():
            out[v] = (f"{v} (sessão)", np.load(pb)["viterbi"], np.load(pc)["viterbi"])
    return out


RAIZ_E3 = P2.RAIZ / "experimento_p3_usuario_2026-10-11"


def inclinacao(Y: np.ndarray, cols: np.ndarray, k0: int, k1: int) -> float:
    """Média, nas colunas, da inclinação (px/s) de uma reta sobre os quadros com
    valor entre k0 e k1; só colunas com 3 ou mais quadros cobrindo 60% da janela."""
    ks = np.arange(k0, k1 + 1)
    incl = []
    for j in cols:
        y = Y[ks, j]
        ok = np.isfinite(y)
        if ok.sum() < 3 or (ks[ok].max() - ks[ok].min()) < 0.6 * (k1 - k0):
            continue
        t = (ks[ok] - k0) * DT
        incl.append(np.polyfit(t, y[ok], 1)[0])
    return float(np.mean(incl)) if incl else float("nan")
