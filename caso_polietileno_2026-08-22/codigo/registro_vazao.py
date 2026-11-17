"""Vazão de oxidante no registro do banco, alinhada às amostras do vídeo.

O registro e o vídeo se encontram na ignição: a amostra da ignição corresponde ao
instante T_IGN_PLAN do relógio do registro, e daí em diante cada amostra vale DT
segundos. Os arquivos e valores vêm de configuracao_pe.json (ensaio.py).
"""
from __future__ import annotations

import numpy as np

import ensaio as E
from experimento_s_sessao import DT, K_IGN

RHO = E.DENSIDADE_OXIDANTE
T_IGN_PLAN = E.T_IGNICAO_REGISTRO


def t_planilha(k):
    """Relógio do registro na amostra k."""
    return T_IGN_PLAN + (np.asarray(k, float) - K_IGN) * DT


def k_da_planilha(t):
    """Amostra (fracionária) de um instante do relógio do registro."""
    return K_IGN + (np.asarray(t, float) - T_IGN_PLAN) / DT


def vazao(ka: int, kb: int) -> float:
    """Vazão média do registro entre as amostras ka e kb, na unidade do registro."""
    d = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    a, b = t_planilha(ka), t_planilha(kb)
    return float(d[(d[:, 0] >= a) & (d[:, 0] <= b), 1].mean())


vazao_media = vazao
