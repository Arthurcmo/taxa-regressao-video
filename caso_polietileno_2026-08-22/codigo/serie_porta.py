"""Altura média da porta no tempo, numa faixa do grão, a partir da linha 2b."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

import faixas_grao as FG
import recompensa_lei as R
import registro_vazao as V
from experimento_s_sessao import DT, K_IGN
from taxas_sessao import MM


def k_de(t, relogio):
    """Amostra de um instante t do relógio do registro, com o fator de tempo `relogio`."""
    return K_IGN + (np.asarray(t, float) - V.T_IGN_PLAN) / (DT * relogio)


def serie_H(x0, x1):
    """Altura média da porta (mm) entre x0 e x1 mm, linha 2b semente 0, mediana de 9 amostras."""
    z = np.load(R.OUT / "runs/nenhum_d4/saida.npz")
    cols = FG.faixa_cols(x0, x1)
    return (median_filter(z["baixo"].astype(float), size=(9, 1))
            - median_filter(z["cima"].astype(float), size=(9, 1)))[:, cols].mean(1) * MM


def H_em(H, k):
    """Altura H interpolada na amostra (fracionária) k."""
    return float(np.interp(k, np.arange(len(H)), H))
