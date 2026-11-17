"""Figuras de resultado no tempo real: fator de tempo, corte do aquecimento e recompensa de forma.

Mesmo padrão de figuras_tg2.py (Times, rótulos em português, linhas frias com contorno
sobre o vídeo, marcações manuais como círculos vazados). Lê os resultados gravados por
relogio_e_corte.py e pelo treino com recompensa de forma.

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/figuras_tg2_resultados.py [nome ...]
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

import figuras_tg2 as F
import recompensa_lei as R
import relogio_e_corte as C
import ensaio as E

EXP = R.OUT
S = E.FATOR_TEMPO_REAL
CAM = E.CAM
MM = F.ESC
AZUL, VERDE_G, CINZA, MAGENTA_G = "#2f6bff", "#00a86b", "0.35", "#b000c8"


def t_real(k):
    return (np.asarray(k, float) - E.AMOSTRA_IGNICAO) * CAM * S


def k_de_t(t):
    return int(round(E.AMOSTRA_IGNICAO + t / (CAM * S)))


# ----------------------------------------------------------------------------- relógio
def fig_relogio():
    B = C.banco()
    ch = C.chama_eventos()
    rb = C.chama_queima()
    aj = C.ajustar(C.eventos_video(ch), C.eventos_banco(B))
    tb, Fv = B["t_planilha_s"], B["vazao_mL_s"] * C.RHO
    tcam = np.arange(len(rb)) * CAM
    fig, axs = plt.subplots(2, 1, figsize=(F.LARG, 4.6), sharex=True)
    casos = ((1.0, aj["banco"]["volta 1"] - aj["video"]["volta 1"], "(a) tempo marcado pela câmera, sem fator"),
             (aj["s"], aj["T0"], f"(b) tempo marcado pela câmera multiplicado por {F.br(aj['s'], 4)}"))
    for ax, (s, T0, tit) in zip(axs, casos):
        ax.plot(T0 + s * tcam, uniform_filter1d(rb, 3), color=AZUL, lw=0.7, label="cor da chama no vídeo")
        a2 = ax.twinx()
        a2.step(tb, Fv, where="post", color="black", lw=0.9, label="vazão de oxidante no registro do banco")
        a2.set_ylim(0, 7); a2.set_ylabel("vazão (g/s)")
        a2.spines["top"].set_visible(False)
        for n in aj["banco"]:
            ax.axvline(aj["banco"][n], color=VERDE_G, lw=1.0, ls=(0, (4, 2)))
        ax.set_ylim(-20, 240); ax.set_ylabel("vermelho menos azul")
        ax.set_xlim(0, 305)
        ax.set_title(tit, loc="left")
        F.virgula(ax, a2)
    axs[1].set_xlabel("tempo do registro do banco (s)")
    from matplotlib.lines import Line2D
    alcas = [Line2D([], [], color=AZUL, lw=1.0, label="cor da chama no vídeo"),
             Line2D([], [], color="black", lw=1.0, label="vazão de oxidante no registro do banco"),
             Line2D([], [], color=VERDE_G, lw=1.0, ls=(0, (4, 2)), label="eventos de vazão")]
    fig.legend(handles=alcas, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F.salvar(fig, "fig_relogio")
    return aj


# ----------------------------------------------------------------------------- corte
def fig_corte():
    import serie_porta as RL
    B = C.banco()
    d = json.loads((EXP / "relogio_e_corte/relogio_e_corte.json").read_text())
    tm = C.Tempo(d["fator"]["s"], d["fator"]["T0"])
    t = tm.desde_ign(B["t_planilha_s"])
    cri, inter, corte = d["criterios"], d["interrupcao"], d["corte"]
    fig, axs = plt.subplots(3, 1, figsize=(F.LARG, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 1, 1]))
    axs[0].step(t, B["vazao_mL_s"] * C.RHO, where="post", color="black", lw=0.9)
    axs[0].set_ylabel("vazão (g/s)"); axs[0].set_ylim(0, 3.2)
    axs[0].set_title("(a) vazão de oxidante no registro do banco", loc="left")
    T2 = uniform_filter1d(B["T_leito2_C"], 11)
    axs[1].plot(t, T2, color=MAGENTA_G, lw=1.2, label="leito catalítico")
    axs[1].plot(t, uniform_filter1d(B["T_entrada_camara_C"], 11), color=AZUL, lw=1.0, label="entrada da câmara")
    axs[1].set_ylim(250, 720); axs[1].set_ylabel("temperatura (°C)")
    axs[1].legend(loc="lower right", frameon=False)
    axs[1].set_title("(b) temperaturas do leito e da entrada da câmara", loc="left")
    k = np.arange(963)
    tk = tm.desde_ign(tm.banco(k))
    for (x0, x1), cor, rot in (((10, 120), AZUL, "grão todo"), ((60, 120), VERDE_G, "segunda metade do grão")):
        axs[2].plot(tk, RL.serie_H(x0, x1), color=cor, lw=1.0, label=rot)
    axs[2].set_ylim(6, 30); axs[2].set_ylabel("altura da porta (mm)")
    axs[2].legend(loc="upper left", frameon=False)
    axs[2].set_title("(c) altura média da porta pela linha 2b", loc="left")
    for a in axs:
        a.axvspan(*inter, color="0.85", zorder=0)
        a.axvline(corte, color="black", lw=1.4)
        a.axvline(E.JANELA_ANALISE_S[1], color="black", lw=1.0, ls=(0, (2, 2)))
        a.grid(alpha=0.25)
        F.virgula(a)
    axs[0].text(corte + 2, 2.7, f"início das janelas, {F.br(corte, 1)} s", fontsize=7.5)
    axs[0].text(E.JANELA_ANALISE_S[1] - 2, 0.35, f"fim, {F.br(E.JANELA_ANALISE_S[1], 1)} s", fontsize=7.5, ha="right")
    axs[2].set_xlim(0, 265)
    axs[2].set_xlabel("tempo desde a ignição (s)")
    fig.tight_layout(h_pad=0.6)
    F.salvar(fig, "fig_corte")


# ----------------------------------------------------------------------------- recompensa nos quadros
def fig_recompensa_quadros():
    forma = R.OUT / "runs/oc_forma_w5_s0/saida.npz"
    zf = np.load(forma)
    mb, mc = F._marcas_sessao("baixo"), F._marcas_sessao("cima")
    ks = (324, 450, 585, 765)
    xa, xb = int(35 / MM + 10), int(85 / MM + 10)
    fig, axs = plt.subplots(2, 2, figsize=(F.LARG, 4.4))
    for ax, k in zip(axs.ravel(), ks):
        im = F.roi(k)
        ax.imshow(im[:, xa:xb], extent=(xa, xb, im.shape[0], 0))
        sel = (F.XS >= xa) & (F.XS <= xb)
        for f in ("baixo", "cima"):
            F.linha(ax, F.XS[sel], zf[f][k][sel], F.CIANO, lw=1.2)
        for m in (mb, mc):
            if k in m:
                p = m[k]; p = p[(p[:, 0] >= xa) & (p[:, 0] <= xb)]
                F.cruzes(ax, p, s=10)
        ax.set_xlim(xa, xb); ax.set_ylim(im.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{t_real(k):.0f} s após a ignição", loc="left")
    axs[0, 0].plot([], [], color=F.CIANO, lw=1.4, label="recompensa de forma")
    axs[0, 0].scatter([], [], marker="o", facecolors="none", edgecolors=F.VERDE, label="marcações manuais")
    axs[0, 0].legend(loc="lower right", frameon=True, facecolor="#222222", edgecolor="none", labelcolor="white", fontsize=6.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    F.FOTOS.add("fig_recompensa_quadros")
    F.salvar(fig, "fig_recompensa_quadros")


TODAS = dict(relogio=fig_relogio, corte=fig_corte, recompensa=fig_recompensa_quadros)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(TODAS):
        TODAS[n]()
