"""Altura da porta e taxa por seção, e os degraus de vazão como teste do expoente (08/11/2026).

Por seção do grão, compara as marcações manuais com a melhor rede e procura mudanças no
tempo que ajudem a medir o expoente.

Seções: entrada 10–40, meio 40–80, final 80–120, 1ª metade 10–60, 2ª metade 60–120 e
grão de 10 a 120 mm. Cada seção vai até a face de baixo encostar na borda da imagem (mediana
das colunas a menos de 14 px da borda), no máximo até o fim da vazão.
Janelas que cruzam uma interrupção de vazão ficam de fora.

Degraus: em cada degrau de vazão a porta quase não muda, e o G muda de uma vez. A taxa
antes e depois dá o expoente sem depender da geometria: n = ln(ṙ2/ṙ1) / ln(G2/G1).
Janelas de antes e depois escolhidas na tabela de vazão, longe das interrupções (a camada aquecida do sólido
leva da ordem de α/ṙ² ≈ 30 s para se ajustar, então o "depois" tende a subestimar a resposta).

Linhas: marcações do autor e a melhor rede (2bv2_s1).
Saída: experimento_recompensa_lei_2026-11-06/secoes_e_degraus/

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/secoes_e_degraus.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import faixas_grao as FG
import figuras_tg2 as F
import recompensa_lei as R
import ensaio as E
import lei_potencia as RG
import relogio_e_corte as C
import resultados_tg2 as T
from experimento_s_sessao import DT, K_IGN
from taxas_sessao import MM

OUT = R.OUT / "secoes_e_degraus"
SECOES = {"entrada 10–40": (10, 40), "meio 40–80": (40, 80), "final 80–120": (80, 120),
          "1ª metade 10–60": (10, 60), "2ª metade 60–120": (60, 120), "grão 10–120": (10, 120)}
BASE = "2bv2_s1"


def t(k):
    return (np.asarray(k, float) - K_IGN) * DT


def vazao():
    tm, _, _ = AL.contexto()
    d = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    return d[:, 0] - tm.ign, d[:, 1]


def interrupcoes(tq, q):
    """Trechos com vazão < 0,3 (unidade da planilha), alargados 1 s."""
    baixo = (q < 0.3) & (tq > 30) & (tq < 286)
    out, ini = [], None
    for ti, b in zip(tq, baixo):
        if b and ini is None:
            ini = ti
        if not b and ini is not None:
            out.append((ini - 1.0, ti + 1.0)); ini = None
    return out


def degraus(tq, q):
    """Instantes em que a média de 5 s muda mais de 25 % de um lado para o outro."""
    ts = np.arange(30, 285, 0.5)
    m = lambda a, b: np.mean(q[(tq >= a) & (tq < b)])
    r = np.array([m(x, x + 5) / max(m(x - 5, x), 1e-3) for x in ts])
    out = []
    for x, v in zip(ts, r):
        if (v > 1.25 or v < 0.8) and m(x - 5, x) > 0.5 and m(x, x + 5) > 0.5 and all(abs(x - o) > 8 for o in out):
            out.append(float(x))
    return out


def borda(Yb, Hh=593):
    """Tempo (s) em que a face de baixo encosta na borda, por seção (mediana das colunas)."""
    res = {}
    for sec, (x0, x1) in SECOES.items():
        cols = FG.faixa_cols(x0, x1)
        y = np.nanpercentile(Yb[:, cols], 90, axis=1)          # 10 % das colunas já perto da borda
        ks = np.flatnonzero((y > Hh - 20) & (np.arange(963) > 400))
        res[sec] = float(t(ks[0])) if len(ks) else 289.0
    return res


def janelas_livres(H, t_ini, t_fim, dur, passo, inter):
    tm, _, _ = AL.contexto()
    P = []
    for a in np.arange(t_ini, t_fim - dur + 1e-6, passo):
        if any(a < i1 and a + dur > i0 for i0, i1 in inter):
            continue
        P += [p for p in C.janelas(H, tm, float(tm.k(tm.ign + a)), float(tm.k(tm.ign + a + dur)), 1)
              if np.isfinite(p["r"]) and np.isfinite(p["G"])]
    return P


def taxa_trecho(H, a, b):
    tm, _, _ = AL.contexto()
    P = C.janelas(H, tm, float(tm.k(tm.ign + a)), float(tm.k(tm.ign + b)), 1)
    return P[0] if P and np.isfinite(P[0]["r"]) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tq, q = vazao()
    inter = interrupcoes(tq, q)
    # degraus limpos (tabela de vazão em s desde a ignição): (nome, antes, depois)
    # o degrau de 63 s fica de fora: cai no aquecimento do leito (a taxa ainda sobe sozinha)
    deg = [(195.0, (172.0, 186.0), (197.0, 211.0)),
           (214.0, (200.0, 212.0), (217.0, 237.0)), (240.5, (225.0, 239.0), (243.0, 260.0))]
    print("interrupções (s):", [(round(a, 1), round(b, 1)) for a, b in inter])
    print("degraus de vazão (s):", deg)
    L = {"marcações do autor": AL.manual(), "melhor rede (2b nova, semente 1)": AL.run(BASE)}
    fim = borda(L["melhor rede (2b nova, semente 1)"][0])
    print("face de baixo encosta na borda (s):", {k: round(v, 1) for k, v in fim.items()})
    res = dict(interrupcoes=inter, degraus=deg, fim_secao=fim, secoes={})
    for sec, (x0, x1) in SECOES.items():
        fs = min(fim[sec], 286.0)
        d = {}
        for nome, (Yb, Yc) in L.items():
            H = AL.serie_H(Yb, Yc, x0, x1)
            tempo = janelas_livres(H, float(t(78)), fs, 20.0, 10.0, inter)
            jan = AL.janelas(H, T._corte, 20.0)          # as 5 janelas do TG, da janela de análise
            aj = T.ajuste(jan) if len(jan) >= 3 else None
            dg = []
            for x, (a1, b1), (a2, b2) in deg:
                if b2 > fs:
                    continue
                p1, p2 = taxa_trecho(H, a1, b1), taxa_trecho(H, a2, b2)
                if p1 and p2 and p1["r"] > 0 and p2["r"] > 0:
                    dg.append(dict(t=x, r1=p1["r"], G1=p1["G"], r2=p2["r"], G2=p2["G"],
                                   n=float(np.log(p2["r"] / p1["r"]) / np.log(p2["G"] / p1["G"]))))
            d[nome] = dict(H=H.tolist(), tempo=tempo, janelas=jan, ajuste=aj, degraus=dg)
        res["secoes"][sec] = d
    (OUT / "secoes_e_degraus.json").write_text(json.dumps(
        {k: (v if k != "secoes" else {s: {n: {kk: vv for kk, vv in x.items() if kk != "H"} for n, x in dd.items()} for s, dd in v.items()})
         for k, v in res.items()}, ensure_ascii=False, indent=1, default=float))

    print("\nseção            | fim  | linha        | janelas desde o início da janela de análise: a G^n R² razão (n jan.) | degraus: n = Δln ṙ / Δln G")
    for sec, d in res["secoes"].items():
        for nome, x in d.items():
            a = x["ajuste"]
            aj = f"{a['a']:.4f} G^{a['n']:.2f} R² {a['r2']:.2f} {a['razao']:.2f}× ({len(x['janelas'])})" if a else "—"
            dg = "; ".join(f"{g['t']:.0f} s: G {g['G1']:.1f}→{g['G2']:.1f}, ṙ {g['r1']:.3f}→{g['r2']:.3f}, n {g['n']:.2f}" for g in x["degraus"])
            print(f"{sec:16s} | {res['fim_secao'][sec]:5.0f} | {nome[:12]:12s} | {aj} | {dg}")

    # ------------------------------------------------------------------ figuras
    secs = list(SECOES)
    cores = {"marcações do autor": F.VERDE, "melhor rede (2b nova, semente 1)": "#0a8fc4"}
    tt = t(np.arange(963))
    fig, axs = plt.subplots(2, 3, figsize=(F.LARG * 1.8, 6.4), sharex=True)
    for ax, sec in zip(axs.ravel(), secs):
        for nome, x in res["secoes"][sec].items():
            ax.plot(tt, x["H"], color=cores[nome], lw=1.4, label=nome)
        ax.axvline(fim[sec], color="0.4", ls="--", lw=0.8)
        for a, b in inter:
            ax.axvspan(a, b, color="0.85", zorder=0)
        ax.set_title(sec, loc="left"); ax.grid(alpha=0.25); ax.set_xlim(0, 290); F.virgula(ax)
    for ax in axs[-1]:
        ax.set_xlabel(F.ROT_TEMPO)
    for ax in axs[:, 0]:
        ax.set_ylabel("altura da porta (mm)")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "altura_por_secao.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(F.LARG * 1.8, 6.6), sharex=True, sharey=True)
    for ax, sec in zip(axs.ravel(), secs):
        ax2 = ax.twinx(); ax2.plot(tq, q * E.DENSIDADE_OXIDANTE, color="0.75", lw=0.8); ax2.set_ylim(0, 14); ax2.set_yticks([])
        for nome, x in res["secoes"][sec].items():
            P = x["tempo"]
            ax.plot([0.5 * (p["t"][0] + p["t"][1]) for p in P], [p["r"] for p in P], "o-", color=cores[nome], ms=3, lw=1.2, label=nome)
        ax.axvline(fim[sec], color="0.4", ls="--", lw=0.8)
        for x_, _, _ in deg:
            ax.axvline(x_, color="#c27bff", lw=0.8, ls=":")
        ax.set_title(sec, loc="left"); ax.grid(alpha=0.25); ax.set_xlim(0, 290); ax.set_ylim(-0.02, 0.2); F.virgula(ax)
    for ax in axs[-1]:
        ax.set_xlabel(F.ROT_TEMPO)
    for ax in axs[:, 0]:
        ax.set_ylabel("taxa de regressão (mm/s)")
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "taxa_tempo_por_secao.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(F.LARG * 1.8, 6.6), sharey=True)
    Gg = np.geomspace(2, 16, 40)
    for ax, sec in zip(axs.ravel(), secs):
        for nome, x in res["secoes"][sec].items():
            J = x["janelas"]
            if not J:
                continue
            G = np.array([p["G"] for p in J]); r = np.array([p["r"] for p in J])
            ok = r > 0
            ax.plot(G[ok], r[ok], "o", color=cores[nome], mec="black", mew=0.4, ms=4.5)
            a = x["ajuste"]
            if a:
                Gl = np.geomspace(G.min(), G.max(), 20)
                ax.plot(Gl, a["a"] * Gl ** a["n"], color=cores[nome], lw=1.2,
                        label=f"{nome.split(' (')[0]}: {F.br(a['a'], 4)} G$^{{{F.br(a['n'], 2)}}}$, R² {F.br(a['r2'], 2)}")
            for g in x["degraus"]:
                ax.annotate("", xy=(g["G2"], g["r2"]), xytext=(g["G1"], g["r1"]),
                            arrowprops=dict(arrowstyle="->", color=cores[nome], lw=1.0))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, which="both"); ax.set_ylim(0.015, 0.25)
        ax.set_title(sec, loc="left"); ax.legend(frameon=False, fontsize=5.6, loc="lower right")
        ax.set_xlabel(r"$G_{ox}$ (kg/(m$^2$ s))")
    for ax in axs[:, 0]:
        ax.set_ylabel("taxa de regressão (mm/s)")
    fig.tight_layout(); fig.savefig(OUT / "taxa_gox_por_secao.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("figuras em", OUT)


if __name__ == "__main__":
    main()
