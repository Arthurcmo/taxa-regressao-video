"""Figuras e números da linha sem marcação humana para o TG (09/11/2026).

  fig_sem_marcacao_versoes.jpg  quadro sem nada desenhado, SM1 v1 e SM1 v4 (semente 2, a de
                                menor erro médio), em 93 s (B1) e 180 s; recortes de 50 a 70 mm
                                e de 95 a 115 mm, nas duas faces; marcações do autor em verde.
  fig_quimograma_fuligem.jpg    quimogramas da face inferior: (a) 52 mm, fuligem parada marcada
                                pelo mapa; (b) 66 mm, mancha que desce mais rápido do que a
                                superfície; (c) 20 mm, fuligem da entrada que desce mais devagar.
  sem_marcacao_tg.json          leis por seção da SM1 v4 semente 2.

Saída das figuras: Template_Thesis_ITA-main/figuras_tg2/.

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/figuras_tg2_sem_marcacao.py
"""
from __future__ import annotations

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

SEM = 2
V1, V4 = f"sm1_s{SEM}", f"sm1v4_s{SEM}"
CIANO, VIOLETA = (255, 229, 0), (255, 123, 194)          # BGR
SECS_TG = {"grão sem a entrada": (40, 120), "meio": (40, 80), "final": (80, 120)}


def fig_versoes():
    L = {"v1": AL.run(V1), "v4": AL.run(V4)}
    ref = AL.run("nenhum_d4")
    linhas = []
    for k in (324, 585):
        for x0mm in (50, 95):
            for face, idx in (("cima", 1), ("baixo", 0)):
                linhas.append((k, x0mm, face, idx))
    fg, axs = plt.subplots(len(linhas), 3, figsize=(F.LARG, 8.6),
                           gridspec_kw=dict(hspace=0.42, wspace=0.03, left=0.2, right=0.995, top=0.955, bottom=0.005))
    for i, (k, x0mm, face, idx) in enumerate(linhas):
        im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR)
        pts = FS.pontos(face, k)
        x0, x1 = CS.x_px(x0mm), CS.x_px(x0mm) + int(20 / MM)
        dentro = [y for x, y in pts if x0 <= x <= x1]
        cols = (XC >= x0) & (XC <= x1)
        centro = float(np.median(dentro)) if dentro else float(np.nanmedian(ref[idx][k, cols]))
        for j, (v, cor) in enumerate(((None, None), ("v1", CIANO), ("v4", VIOLETA))):
            Y = None if v is None else L[v][idx][k]
            c, info = CS.recorte(im, Y, pts, x0mm, centro, cor, face)
            ax = axs[i, j]
            ax.imshow(c); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if info:
                ax.set_title(info.replace("p/ a chama", "para a chama"), fontsize=5.6, pad=1.5, loc="left")
        axs[i, 0].set_ylabel(f"{F.br(FS.t(k), 0)} s, {x0mm} a {x0mm + 20} mm\nface {'superior' if face == 'cima' else 'inferior'}",
                             fontsize=6.2, rotation=0, ha="right", va="center")
    for j, c in enumerate(("quadro sem linha", "SM1 v1 (ciano)", "SM1 v4 (violeta)")):
        axs[0, j].annotate(c, (0.5, 1.45), xycoords="axes fraction", ha="center", va="bottom", fontsize=7.5)
    fg.savefig(F.SAIDA / "fig_sem_marcacao_versoes.jpg", dpi=300, pil_kwargs={"quality": 92}); plt.close(fg)


def fig_quimograma():
    C = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    PAR = np.load(R.OUT / "fuligem_estatica" / "parado.npy", mmap_mode="r")
    Y1, Y4 = AL.run(V1)[0], AL.run(V4)[0]
    A = R.autor("baixo")
    paineis = (("(a) 52 mm: fuligem parada", 52, 300, 640, 70),
               ("(b) 66 mm: mancha que desce", 66, 440, 640, 60),
               ("(c) 20 mm: fuligem da entrada", 20, 150, 640, 0))
    fg, axs = plt.subplots(1, 3, figsize=(F.LARG, 3.3))
    for ax, (tit, xmm, k0, k1, meia) in zip(axs, paineis):
        ks = np.arange(k0, k1)
        t = np.array([FS.t(k) for k in ks])
        j = int(np.argmin(np.abs(XC - (xmm / MM + 10))))
        img = np.transpose(C[ks, :, j, ::-1].astype(np.float32) / 255.0, (1, 0, 2))
        ax.imshow(np.clip(img * 1.2, 0, 1), aspect="auto", extent=(t[0], t[-1], img.shape[0], 0))
        pm = np.ma.masked_less(PAR[ks, :, j].T.astype(float) / 255, 0.5)
        ax.imshow(pm, aspect="auto", extent=(t[0], t[-1], img.shape[0], 0), cmap="cool", alpha=0.55, vmin=0.5, vmax=1)
        ax.plot(t, Y1[ks, j], color="#00e5ff", lw=0.9, ls=(0, (4, 2)))
        ax.plot(t, Y4[ks, j], color="#c27bff", lw=0.9, ls=(0, (4, 2)))
        m = np.isfinite(A[ks, j])
        ax.plot(t[m], A[ks, j][m], "o", ms=3, mfc="none", mec="#6aff00", mew=1.0)
        if meia:
            yc = np.nanmedian(Y4[ks, j]); ax.set_ylim(yc + meia * 1.6, yc - meia * 1.6)
        else:
            ax.set_ylim(593, 330)
        ax.set_title(tit, fontsize=7.5, loc="left")
        ax.set_xlabel("tempo após a ignição (s)", fontsize=7)
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel("posição vertical na imagem (px)", fontsize=7)
    h = [plt.Line2D([], [], color="#00e5ff", ls=(0, (4, 2)), label="SM1 v1"),
         plt.Line2D([], [], color="#c27bff", ls=(0, (4, 2)), label="SM1 v4"),
         plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#6aff00", label="marcações do autor"),
         plt.Rectangle((0, 0), 1, 1, fc="#d080ff", alpha=0.6, label="fuligem parada (mapa)")]
    fg.tight_layout(rect=(0, 0.08, 1, 1))
    fg.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fg.savefig(F.SAIDA / "fig_quimograma_fuligem.jpg", dpi=300, pil_kwargs={"quality": 92}); plt.close(fg)


def numeros():
    Yb, Yc = AL.run(V4)
    res = {}
    for sec, (x0, x1) in SECS_TG.items():
        r, G, _ = CS.pontos(AL.serie_H(Yb, Yc, x0, x1))
        res[sec] = dict(r=r.tolist(), G=G.tolist(), **CS.ajuste(G, r))
    (R.OUT / "sem_marcacao" / "versoes_v4_2026-11-09" / "sem_marcacao_tg.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for s, d in res.items():
        print(s, round(d["a"], 4), round(d["n"], 2), round(d["r2"], 2), round(d["razao"], 2))


if __name__ == "__main__":
    fig_versoes()
    fig_quimograma()
    numeros()
