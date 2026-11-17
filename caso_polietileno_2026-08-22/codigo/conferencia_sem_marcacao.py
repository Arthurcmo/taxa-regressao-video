"""Conferência da linha sem marcação humana contra as marcações do autor (09/11/2026).

SM1: rede da ImageNet treinada só com o detector clássico. SMF: SM1 + recompensa de forma.
Nenhuma das duas viu marcação. Aqui elas são comparadas com as marcações do autor:

  zoom_<t>s.jpg   um quadro por figura (93 s = B1, 135 s, 180 s, 230 s = B2); recortes de
                  20 mm em volta de cada face, na entrada (15 a 35 mm), no meio (50 a 70 mm)
                  e no fim (95 a 115 mm). Colunas: o quadro sem nada desenhado, a SM1 e a
                  SMF da semente 0 (tracejadas, para a superfície ficar visível), com as
                  marcações do autor em círculos verdes vazados. Acima de cada recorte, o
                  erro médio contra as marcações e o deslocamento médio para o lado da chama.
  erros.md        erro por face e por trecho do grão, nas três sementes, na janela de
                  análise e nos blocos B1 e B2.
  taxa_lei.png    taxa contra G_ox em 10 janelas iguais na janela de análise, no tempo real,
                  no grão todo, na entrada, no meio e no final; pontos e lei ajustada
                  (equação no gráfico) para as marcações, a SM1 e a SMF (média das três
                  sementes).
  taxa_lei.md/.json  os pontos e as equações.

Saída: experimento_recompensa_lei_2026-11-06/sem_marcacao/conferencia_2026-11-09/

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/conferencia_sem_marcacao.py
"""
from __future__ import annotations

import ensaio as E
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import face_cima as FC
import faixas_grao as FG
import figuras_sementes as FS
import figuras_tg2 as F
import janelas_taxa as Q
import recompensa_lei as R
import lei_potencia as RG
import resultados_tg2 as T
from experimento_s_sessao import XC, divisao, marcas
from taxas_sessao import MM

OUT = R.OUT / "sem_marcacao" / "conferencia_2026-11-09"
S = E.FATOR_TEMPO_REAL         # tempo real = S × tempo marcado pela câmera
L_M = 0.020
SEMENTES = (0, 1, 2)
KS = (324, 450, 585, 735)
SECS_ZOOM = (("entrada", 15), ("meio", 50), ("fim", 95))
ESC, MEIA = 2, 62
SECS = {"grão todo": (10, 120), "entrada": (10, 40), "meio": (40, 80), "final": (80, 120)}
N_JAN = 10
CIANO_BGR, MAGENTA_BGR, VERDE_BGR = (255, 229, 0), (255, 92, 255), (0, 255, 106)
SENT = {"baixo": -1, "cima": +1}   # sinal que leva o erro para o lado da chama


def linhas() -> dict:
    return {"SM1": [AL.run(f"sm1_s{s}") for s in SEMENTES],
            "SMF": [AL.run(f"smf_s{s}") for s in SEMENTES],
            "2b": [AL.run("nenhum_d4" + ("" if s == 0 else f"_s{s}")) for s in SEMENTES]}


# ----------------------------------------------------------------------------- recortes
def x_px(xmm):
    return int(xmm / MM + 10)


def recorte(im, Y, pts, x0mm, centro, cor, face):
    x0 = x_px(x0mm); x1 = x0 + int(20 / MM)
    ya, yb = int(max(0, centro - MEIA)), int(min(im.shape[0], centro + MEIA))
    c = cv2.resize(im[ya:yb, x0:x1], None, fx=ESC, fy=ESC, interpolation=cv2.INTER_CUBIC)
    info = ""
    if Y is not None:
        FC.tracejado(c, np.stack([(XC - x0) * ESC, (Y - ya) * ESC], 1), cor, 2, 10, 7)
        sel = [(x, y) for x, y in pts if x0 <= x <= x1]
        for x, y in sel:
            if ya <= y <= yb:
                p = (int((x - x0) * ESC), int((y - ya) * ESC))
                cv2.circle(c, p, 5, (0, 0, 0), 2, cv2.LINE_AA); cv2.circle(c, p, 5, VERDE_BGR, 1, cv2.LINE_AA)
        if sel:
            P = np.array(sel)
            e = np.interp(P[:, 0], XC, Y) - P[:, 1]
            info = f"erro {F.br(np.abs(e).mean(), 1)} px, {F.br(np.mean(SENT[face] * e), 1)} px p/ a chama"
        else:
            info = "sem marcações neste trecho"
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB), info


def fig_zoom(L, k):
    Lb = {"SM1": L["SM1"][0], "SMF": L["SMF"][0]}
    ref = L["2b"][0]
    im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR)
    fig, axs = plt.subplots(6, 3, figsize=(11.5, 9.6),
                            gridspec_kw=dict(hspace=0.32, wspace=0.03, left=0.115, right=0.995, top=0.875, bottom=0.01))
    for i, (rot, x0mm) in enumerate(SECS_ZOOM):
        for f, (face, idx) in enumerate((("cima", 1), ("baixo", 0))):
            pts = FS.pontos(face, k)
            x0, x1 = x_px(x0mm), x_px(x0mm) + int(20 / MM)
            dentro = [y for x, y in pts if x0 <= x <= x1]
            cols = (XC >= x0) & (XC <= x1)
            centro = float(np.median(dentro)) if dentro else float(np.nanmedian(ref[idx][k, cols]))
            linha_ax = axs[2 * i + f]
            for j, (nome, cor) in enumerate(((None, None), ("SM1", CIANO_BGR), ("SMF", MAGENTA_BGR))):
                Y = None if nome is None else Lb[nome][idx][k]
                c, info = recorte(im, Y, pts, x0mm, centro, cor, face)
                ax = linha_ax[j]
                ax.imshow(c); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if info:
                    ax.set_title(info, fontsize=7, pad=2, loc="left")
            linha_ax[0].set_ylabel(f"{rot}, {x0mm} a {x0mm + 20} mm\nface {'superior' if face == 'cima' else 'inferior'}",
                                   fontsize=7.5, rotation=0, ha="right", va="center")
    cab = ("quadro, sem nada desenhado", "SM1, semente 0 (ciano tracejado)", "SMF, semente 0 (magenta tracejado)")
    for j, c in enumerate(cab):
        axs[0, j].annotate(c, (0.5, 1.32), xycoords="axes fraction", ha="center", va="bottom", fontsize=9, weight="bold")
    blk = " (bloco de teste B1)" if k == 324 else (" (bloco de teste B2)" if k == 735 else "")
    fig.suptitle(f"{F.br(FS.t(k), 0)} s após a ignição{blk}. Círculos verdes: marcações do autor. "
                 f"Recortes de 20 mm por {F.br(2 * MEIA * MM, 1)} mm, ampliados duas vezes.",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=10)
    nome = OUT / f"zoom_{int(round(FS.t(k))):03d}s.jpg"
    fig.savefig(nome, dpi=190, pil_kwargs={"quality": 92}); plt.close(fig)
    return nome


# ----------------------------------------------------------------------------- erros
def erros(L) -> dict:
    res = {}
    div = divisao()
    for nome in ("SM1", "SMF", "2b"):
        for s, (Yb, Yc) in zip(SEMENTES, L[nome]):
            d = {"baixo": Yb, "cima": Yc}
            r = {}
            for face in ("baixo", "cima"):
                A = R.autor(face)
                ks = [k for k in range(963) if FS.CORTE <= FS.t(k) <= FS.FIM and np.isfinite(A[k]).sum() > 10]
                for sec, (x0, x1) in (("entrada", (10, 40)), ("meio", (40, 80)), ("final", (80, 120)), ("grão todo", (10, 120))):
                    cols = FG.faixa_cols(x0, x1)
                    e = np.concatenate([(d[face][k, cols] - A[k, cols])[np.isfinite(A[k, cols])] for k in ks])
                    r[f"{face} {sec}"] = dict(erro=float(np.abs(e).mean()), chama=float(np.mean(SENT[face] * e)),
                                              ate4=float(np.mean(np.abs(e) <= 4)))
                for bloco in ("B1", "B2"):
                    mk = marcas(face); e = []
                    for k in div[bloco]:
                        if k in mk:
                            p = mk[k]
                            sel = (p[:, 0] >= FS.CENTRAIS[0]) & (p[:, 0] <= FS.CENTRAIS[1])
                            e.append(np.interp(p[sel, 0], XC, d[face][k]) - p[sel, 1])
                    e = np.concatenate(e)
                    r[f"{face} {bloco}"] = dict(erro=float(np.abs(e).mean()), chama=float(np.mean(SENT[face] * e)),
                                                ate4=float(np.mean(np.abs(e) <= 4)))
            res[f"{nome} s{s}"] = r
    return res


def tabela_erros(res) -> str:
    chaves = [f"{f} {s}" for f in ("baixo", "cima") for s in ("entrada", "meio", "final", "grão todo", "B1", "B2")]
    lin = ["# Erro contra as marcações do autor (px)", "",
           "Janela da janela de análise (alvo por coluna, quadros marcados) e blocos de teste B1 e B2 (pontos marcados, "
           "colunas centrais). Cada célula: erro médio / deslocamento médio para o lado da chama / % dos pontos a até 4 px.", "",
           "| linha | " + " | ".join(c.replace("baixo", "inferior").replace("cima", "superior") for c in chaves) + " |",
           "|---" * (len(chaves) + 1) + "|"]
    for nome, r in res.items():
        cel = [f"{F.br(r[c]['erro'], 1)} / {F.br(r[c]['chama'], 1)} / {F.br(100 * r[c]['ate4'], 0)} %" for c in chaves]
        lin.append(f"| {nome} | " + " | ".join(cel) + " |")
    return "\n".join(lin) + "\n"


# ----------------------------------------------------------------------------- taxa
def pontos(H):
    P = Q.janelas_k(H, T._tm, T._corte, T.FIM, N_JAN)
    r = np.array([w["r"] * 1e3 for w in P])
    G = np.array([w["md"] / (L_M * w["Hm"]) for w in P])
    t = [w["t"] for w in P]
    return r, G, t


def ajuste(G, r):
    ok = (r > 0) & np.isfinite(r) & np.isfinite(G)
    n, b = np.polyfit(np.log(G[ok]), np.log(r[ok]), 1)
    rr = np.corrcoef(np.log(G[ok]), np.log(r[ok]))[0, 1]
    return dict(a=float(np.exp(b)), n=float(n), r2=float(rr ** 2))


def taxa(L) -> dict:
    M = AL.manual()
    res = {}
    for sec, (x0, x1) in SECS.items():
        d = {}
        r, G, t = pontos(AL.serie_H(*M, x0, x1))
        d["marcações do autor"] = dict(r=r.tolist(), G=G.tolist(), **ajuste(G, r))
        d["t"] = t
        for nome in ("SM1", "SMF", "2b"):
            RR, GG = [], []
            for s, (Yb, Yc) in zip(SEMENTES, L[nome]):
                r, G, _ = pontos(AL.serie_H(Yb, Yc, x0, x1))
                RR.append(r); GG.append(G)
                d[f"{nome} s{s}"] = dict(r=r.tolist(), G=G.tolist(), **ajuste(G, r))
            r, G = np.mean(RR, 0), np.mean(GG, 0)
            d[nome] = dict(r=r.tolist(), G=G.tolist(), **ajuste(G, r))
        res[sec] = d
    return res


def eq(d):
    return f"ṙ = {F.br(d['a'], 4)} G$^{{{F.br(d['n'], 2)}}}$ (R² {F.br(d['r2'], 2)})"


def fig_taxa(res):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True, sharey=True)
    estilos = (("marcações do autor", "marcações do autor", F.VERDE, "o", "white", (0, (4, 2))),
               ("SM1", "SM1, sem marcação, média de 3 sementes", "#00b8d4", "^", None, "-"),
               ("SMF", "SMF, sem marcação, média de 3 sementes", "#d61fd6", "D", None, "-"))
    for ax, (sec, (x0, x1)) in zip(axs.flat, SECS.items()):
        d = res[sec]
        ax.set_xscale("log"); ax.set_yscale("log")
        txt = []
        for chave, rot, cor, mk, mfc, ls in estilos:
            x = d[chave]
            G, r = np.array(x["G"]), np.array(x["r"])
            ax.plot(G, r, mk, ms=6, color=cor, mfc=mfc or cor, mec=cor if mfc else "black", mew=1.2 if mfc else 0.5, zorder=5)
            g = np.geomspace(G.min(), G.max(), 30)
            ax.plot(g, x["a"] * g ** x["n"], color=cor, lw=1.7, ls=ls, zorder=4)
            nm = rot.split(",")[0]
            txt.append(f"{nm}: {eq(x)}")
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=7.3, va="top",
                bbox=dict(fc="white", ec="0.7", alpha=0.92, pad=2.5))
        ax.set_xlim(2.8, 8.0); ax.set_ylim(0.02, 0.6)
        ax.set_xticks([3, 4, 5, 6, 7, 8]); ax.set_xticklabels(["3", "4", "5", "6", "7", "8"])
        yt = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
        ax.set_yticks(yt); ax.set_yticklabels([F.br(v, 2) for v in yt])
        ax.minorticks_off(); ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"{sec[0].upper() + sec[1:]}, {x0} a {x1} mm", loc="left", fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel(r"$\bar G_{ox}$ (kg/(m$^2$ s))")
    for ax in axs[:, 0]:
        ax.set_ylabel("taxa de regressão (mm/s), tempo real")
    h = [plt.Line2D([], [], marker=mk, ms=6, color=cor, mfc=mfc or cor, mec=cor if mfc else "black", lw=1.7, ls=ls, label=rot)
         for _, rot, cor, mk, mfc, ls in estilos]
    fig.suptitle("Taxa contra fluxo de oxidante: 10 janelas iguais na janela de análise, tempo real",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8.2)
    fig.savefig(OUT / "taxa_lei.png", dpi=180); plt.close(fig)


def tabela_taxa(res) -> str:
    lin = ["# Taxa contra G_ox, 10 janelas iguais na janela de análise", "",
           "Taxa em mm/s no tempo real; G_ox em kg/(m² s).", ""]
    for sec in SECS:
        d = res[sec]
        lin += [f"## {sec}", "", "| linha | equação | R² |", "|---|---|---|"]
        for nome in ("marcações do autor", "SM1", "SM1 s0", "SM1 s1", "SM1 s2", "SMF", "SMF s0", "SMF s1", "SMF s2", "2b"):
            x = d[nome]
            rot = nome + (" (média das 3 sementes)" if nome in ("SM1", "SMF", "2b") else "")
            lin.append(f"| {rot} | ṙ = {F.br(x['a'], 4)} G^{F.br(x['n'], 2)} | {F.br(x['r2'], 2)} |")
        lin += ["", "| janela (s) | marcações G / ṙ | SM1 G / ṙ | SMF G / ṙ | 2b G / ṙ |", "|---|---|---|---|---|"]
        for i, (t0, t1) in enumerate(d["t"]):
            cel = [f"{F.br(d[n]['G'][i], 2)} / {F.br(d[n]['r'][i], 4)}" for n in ("marcações do autor", "SM1", "SMF", "2b")]
            lin.append(f"| {F.br(t0, 1)} a {F.br(t1, 1)} | " + " | ".join(cel) + " |")
        lin.append("")
    return "\n".join(lin) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = linhas()
    for k in KS:
        print("figura", fig_zoom(L, k))
    ER = erros(L)
    (OUT / "erros.md").write_text(tabela_erros(ER))
    (OUT / "erros.json").write_text(json.dumps(ER, indent=1, ensure_ascii=False))
    X = taxa(L)
    (OUT / "taxa_lei.json").write_text(json.dumps(X, indent=1, ensure_ascii=False))
    (OUT / "taxa_lei.md").write_text(tabela_taxa(X))
    fig_taxa(X)
    print("ok", OUT)


if __name__ == "__main__":
    main()
