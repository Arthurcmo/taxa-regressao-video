"""Galeria da queima inteira para o apêndice de imagens do TG (15/11/2026).

A 2b (semente 0, azul) e a SM1 v4 (semente 2, magenta) sobre o quadro inteiro da região
de interesse, com as marcações manuais em círculos verdes, em oito quadros marcados nas
duas faces, de 60 a 240 s após a ignição (tempo real). Duas figuras de quatro quadros:

  fig_galeria_queima_1.jpg   60, 84, 110 e 125 s
  fig_galeria_queima_2.jpg   145, 170, 204 e 240 s

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/figuras_galeria_queima.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import ensaio as E
import avaliar_linhas as AL
import figuras_sementes as FS
import figuras_tg2 as F
from experimento_s_sessao import XC

PARTES = {1: (225, 296, 375, 420), 2: (480, 555, 655, 765)}
LINHAS = (("2b, semente 0", "nenhum_d4", F.AZUL), ("SM1 v4, semente 2", "sm1v4_s2", F.MAGENTA))


def nota(t):
    if t < E.JANELA_ANALISE_S[0]:
        return "antes da janela de análise"
    if t <= E.JANELA_ANALISE_S[1]:
        return "dentro da janela de análise"
    return "depois da janela de análise"


def figura(parte, ks, L):
    fig, axs = plt.subplots(len(ks), 1, figsize=(F.LARG, 8.9))
    for i, (ax, k) in enumerate(zip(axs, ks)):
        im = F.roi(k)
        ax.imshow(im, extent=(0, im.shape[1], im.shape[0], 0))
        for _, (Yb, Yc), cor in L:
            F.linha(ax, XC, Yb[k], cor, lw=1.1)
            F.linha(ax, XC, Yc[k], cor, lw=1.1)
        for f in ("baixo", "cima"):
            p = FS.pontos(f, k)
            if len(p):
                F.cruzes(ax, p, s=7)
        ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        t = FS.t(k)
        letra = "abcdefgh"[i + 4 * (parte - 1)]
        ax.set_title(f"({letra}) {F.br(t, 0)} s após a ignição, {nota(t)}", loc="left")
    for nome, _, cor in LINHAS:
        axs[0].plot([], [], color=cor, lw=1.4, label=nome)
    axs[0].scatter([], [], marker="o", facecolors="none", edgecolors=F.VERDE, label="marcações manuais")
    axs[0].legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none",
                  labelcolor="white", fontsize=6.5, ncol=3)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(F.SAIDA / f"fig_galeria_queima_{parte}.jpg", pil_kwargs={"quality": 90}, dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    L = [(nome, AL.run(run), cor) for nome, run, cor in LINHAS]
    for parte, ks in PARTES.items():
        figura(parte, ks, L)
        print("parte", parte, [round(FS.t(k), 1) for k in ks], flush=True)


if __name__ == "__main__":
    main()
