"""Figuras de comparação conjunta do Capítulo 4 (11/11/2026).

Lê comparacao_geral.json (codigo/comparacao_geral.py) e grava em figuras_tg2/:
  fig_comparacao_posicao.png   erro contra as marcações, todas as linhas, nos blocos de teste
                               e na janela de análise (40 a 120 mm)
  fig_comparacao_taxa.png      taxa do ensaio inteiro e expoente (meio e degraus)
  fig_comparacao_altura.png    altura média da porta (40 a 120 mm) no tempo e diferença
                               para as marcações do autor
  fig_comparacao_quadros.jpg   quadro sem linha, clássico, S1, 2b e SM1 v4 sobre os mesmos
                               recortes, em 93 s (B1) e 150 s

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/figuras_comparacao_cap4.py
"""
from __future__ import annotations

import ensaio as E
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import conferencia_sem_marcacao as CS
import figuras_sementes as FS
import figuras_tg2 as F
import recompensa_lei as R
from experimento_s_sessao import XC
from taxas_sessao import MM

D = json.loads((R.OUT / "comparacao_geral" / "comparacao_geral.json").read_text())
L = D["linhas"]
ORDEM = ["clássico", "S1", "2b", "forma", "SM1 v4", "SMF v4"]
COR = {"marcações do autor": "#2ca02c", "clássico": "#9b59ff", "S1": "#00a5c8", "2b": "#1f4fd6",
       "forma": "#11806a", "SM1 v4": "#d61fd6", "SMF v4": "#2b2d6e"}
GRUPO = {"clássico": "sem rede", "S1": "com marcações", "2b": "com marcações", "forma": "com marcações",
         "SM1 v4": "sem marcação", "SMF v4": "sem marcação"}
VIU_TESTE = {"2b", "forma"}                       # viram as marcações dos blocos de teste
VIU_JANELA = {"S1", "2b", "forma"}                # treinaram com quadros da janela


def sementes(nome):
    return [v for v in L[nome].values()]


def stat(nome, f):
    x = np.array([f(v) for v in sementes(nome)], float)
    return x.mean(), x.min(), x.max()


def eixo_log(ax):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "30"])
    ax.minorticks_off(); ax.set_xlim(0.9, 35)


def fig_posicao():
    fg, axs = plt.subplots(1, 2, figsize=(F.LARG, 3.6), sharey=True)
    y = np.arange(len(ORDEM))[::-1]
    for ax, (tit, chave, viu) in zip(axs, (("(a) blocos de teste B1 e B2", "teste", VIU_TESTE),
                                          ("(b) janela de análise, 40 a 120 mm", "janela", VIU_JANELA))):
        ax.axvspan(1.0, 2.0, color="0.85", lw=0, zorder=0)
        for yi, nome in zip(y, ORDEM):
            for face, mk, dy in (("baixo", "o", 0.12), ("cima", "s", -0.12)):
                m, a, b = stat(nome, lambda v: v["erros"][f"{chave} {face}"][0])
                cheio = nome not in viu
                ax.plot([a, b], [yi + dy] * 2, color=COR[nome], lw=1.2, zorder=2)
                ax.plot(m, yi + dy, mk, ms=5.5, color=COR[nome], mfc=COR[nome] if cheio else "white",
                        mec=COR[nome], mew=1.3, zorder=3)
        eixo_log(ax)
        ax.set_title(tit, loc="left", fontsize=8.5)
        ax.set_xlabel("erro médio contra as marcações (px)", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=7.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels(ORDEM, fontsize=8)
    for yi, nome in zip(y, ORDEM):
        axs[0].get_yticklabels()[list(y).index(yi)].set_color(COR[nome])
    h = [plt.Line2D([], [], marker="o", ls="none", color="0.3", label="face inferior"),
         plt.Line2D([], [], marker="s", ls="none", color="0.3", label="face superior"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", label="vazado: a linha viu essas marcações"),
         plt.Rectangle((0, 0), 1, 1, fc="0.85", label="dispersão das marcações")]
    fg.tight_layout(rect=(0, 0.1, 1, 1))
    fg.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fg.savefig(F.SAIDA / "fig_comparacao_posicao.png", dpi=300); plt.close(fg)


def fig_taxa():
    nomes = ["marcações do autor"] + ORDEM
    fg, axs = plt.subplots(1, 2, figsize=(F.LARG, 3.9), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(nomes))[::-1]
    a = axs[0]
    for yi, nome in zip(y, nomes):
        c = COR[nome]
        m, lo, hi = stat(nome, lambda v: v["taxa"]["grão todo"]["ensaio"]["r"])
        a.plot([lo, hi], [yi] * 2, color=c, lw=1.1)
        a.plot(m, yi, "o", ms=5, color=c, mec="black" if nome != "marcações do autor" else c, mew=0.4,
               mfc=c if nome != "marcações do autor" else "white")
    a.set_xlabel("taxa de regressão (mm/s), tempo real", fontsize=8)
    a.set_title("(a) grão todo, ensaio inteiro", loc="left", fontsize=8.5)
    a.xaxis.set_major_formatter(F.VIRG); a.grid(axis="x", alpha=0.25); a.tick_params(labelsize=7.5)
    a.set_yticks(y); a.set_yticklabels(nomes, fontsize=8)
    b = axs[1]
    for yi, nome in zip(y, nomes):
        c = COR[nome]
        m, lo, hi = stat(nome, lambda v: v["taxa"]["meio"]["n"])
        b.plot([lo, hi], [yi + 0.12] * 2, color=c, lw=1.1)
        b.plot(m, yi + 0.12, "D", ms=4.5, color=c, mec="black" if nome != "marcações do autor" else c, mew=0.4,
               mfc=c if nome != "marcações do autor" else "white")
        nd = [x for v in sementes(nome) for x in v["taxa"]["degraus"]["n"]]
        if nd:
            b.plot([min(nd), max(nd)], [yi - 0.12] * 2, color=c, lw=1.1)
            b.plot(np.median(nd), yi - 0.12, "v", ms=5, color=c, mec="black" if nome != "marcações do autor" else c,
                   mew=0.4, mfc=c if nome != "marcações do autor" else "white")
    b.set_xlim(-0.6, 1.3)
    b.set_xlabel("expoente n", fontsize=8)
    b.set_title("(b) expoente da lei", loc="left", fontsize=8.5)
    b.xaxis.set_major_formatter(F.VIRG); b.grid(axis="x", alpha=0.25); b.tick_params(labelsize=7.5)
    for lab, nome in zip(a.get_yticklabels(), nomes):
        lab.set_color(COR[nome])
    h = [plt.Line2D([], [], marker="o", ls="none", color="0.3", label="ensaio inteiro"),
         plt.Line2D([], [], marker="D", ls="none", color="0.3", label="meio do grão, 10 janelas"),
         plt.Line2D([], [], marker="v", ls="none", color="0.3", label="degraus de vazão")]
    fg.tight_layout(rect=(0, 0.16, 1, 1))
    fg.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.8)
    fg.savefig(F.SAIDA / "fig_comparacao_taxa.png", dpi=300); plt.close(fg)


def fig_altura():
    tt = np.array([FS.t(k) for k in range(963)])
    Hm = np.array(L["marcações do autor"]["0"]["taxa"]["altura_40_120"], float)
    A = R.autor("baixo")
    km = [k for k in range(963) if np.isfinite(A[k]).sum() > 10 and np.isfinite(Hm[k])]
    fg, axs = plt.subplots(2, 1, figsize=(F.LARG, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    a, b = axs
    for ax in axs:
        ax.axvspan(*E.JANELA_ANALISE_S, color="0.93", lw=0, zorder=0)
    for nome in ORDEM:
        Hs = np.mean([np.array(v["taxa"]["altura_40_120"], float) for v in sementes(nome)], 0)
        a.plot(tt, Hs, color=COR[nome], lw=1.0, label=nome)
        b.plot(tt[km], Hs[km] - Hm[km], "-", color=COR[nome], lw=1.0, marker=".", ms=3)
    a.plot(tt[km], Hm[km], "o", ms=3.5, mfc="white", mec=COR["marcações do autor"], mew=1.1, label="marcações do autor", zorder=5)
    a.set_ylabel("altura da porta,\n40 a 120 mm (mm)", fontsize=8)
    a.set_xlim(40, 200); a.set_ylim(5, 30)
    a.legend(frameon=False, fontsize=6.6, ncol=4, loc="upper left")
    b.axhline(0, color="black", lw=0.8)
    b.set_ylim(-1.2, 1.2)
    b.set_ylabel("diferença para as\nmarcações (mm)", fontsize=8)
    b.set_xlabel(F.ROT_TEMPO, fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25)
        F.virgula(ax)
    a.set_title("(a) altura média da porta no tempo; faixa cinza: janela de análise", loc="left", fontsize=8.5)
    b.set_title("(b) diferença para as marcações do autor, nos quadros marcados", loc="left", fontsize=8.5)
    fg.tight_layout()
    fg.savefig(F.SAIDA / "fig_comparacao_altura.png", dpi=300); plt.close(fg)


def fig_quadros():
    import experimento_p2_v2 as P2
    cl = np.load(P2.E1 / "classico.npz")
    LIN = [("clássico", (cl["y_inf"].astype(float), cl["y_sup"].astype(float)), (255, 89, 155)),
           ("S1", tuple(np.load(R.S1_RUNS / f"S1_{f}_s0/saida.npz")["viterbi"].astype(float) for f in ("baixo", "cima")), (255, 229, 0)),
           ("2b", AL.run("nenhum_d4"), (214, 79, 31)),
           ("SM1 v4", AL.run("sm1v4_s2"), (255, 92, 255))]
    ref = AL.run("nenhum_d4")
    linhas = [(k, x0, face, idx) for k in (324, 495) for x0 in (50, 95) for face, idx in (("cima", 1), ("baixo", 0))]
    fg, axs = plt.subplots(len(linhas), 5, figsize=(F.LARG, 6.6),
                           gridspec_kw=dict(hspace=0.5, wspace=0.03, left=0.155, right=0.998, top=0.955, bottom=0.005))
    for i, (k, x0mm, face, idx) in enumerate(linhas):
        im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR)
        pts = FS.pontos(face, k)
        x0, x1 = CS.x_px(x0mm), CS.x_px(x0mm) + int(20 / MM)
        dentro = [yy for xx, yy in pts if x0 <= xx <= x1]
        cols = (XC >= x0) & (XC <= x1)
        centro = float(np.median(dentro)) if dentro else float(np.nanmedian(ref[idx][k, cols]))
        for j, item in enumerate([None] + LIN):
            Y, cor = (None, None) if item is None else (item[1][idx][k], item[2])
            c, info = CS.recorte(im, Y, pts, x0mm, centro, cor, face)
            ax = axs[i, j]
            ax.imshow(c); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if info:
                ax.set_title(info.split(" px")[0].replace("erro ", "") + " px", fontsize=5.6, pad=1.2, loc="left")
        axs[i, 0].set_ylabel(f"{F.br(FS.t(k), 0)} s, {x0mm} a {x0mm + 20} mm\nface {'superior' if face == 'cima' else 'inferior'}",
                             fontsize=5.8, rotation=0, ha="right", va="center")
    for j, c in enumerate(("quadro sem linha", "clássico", "S1", "2b", "SM1 v4")):
        axs[0, j].annotate(c, (0.5, 1.5), xycoords="axes fraction", ha="center", va="bottom", fontsize=7.2)
    fg.savefig(F.SAIDA / "fig_comparacao_quadros.jpg", dpi=300, pil_kwargs={"quality": 92}); plt.close(fg)


if __name__ == "__main__":
    fig_posicao()
    fig_taxa()
    fig_altura()
    fig_quadros()
    print("ok")
