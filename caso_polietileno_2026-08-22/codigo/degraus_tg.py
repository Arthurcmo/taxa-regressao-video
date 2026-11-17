"""Degraus de vazão para o TG e figuras separadas por seção (08/11/2026).

Linhas: marcações do autor e a 2b das três sementes (as linhas do texto).
Degraus limpos (s desde a ignição, tempo real) e janelas de antes e depois, longe das
interrupções: 195 s (172–186 / 197–211), 214 s (200–212 / 217–237), 240,5 s
(225–239 / 243–260). Cada seção entra no degrau só se a face de baixo ainda não
encostou na borda até o fim da janela de depois.

Figuras:
  fig_degraus (TG)          (a) vazão e altura da porta na seção final; (b) taxa contra G_ox
                            em cada degrau, antes e depois ligados por uma seta;
  janelas_por_secao.png     só as 5 janelas do TG e as retas, por seção;
  degraus_por_secao.png     só os degraus, por seção.

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/degraus_tg.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import figuras_tg2 as F
import recompensa_lei as R
import lei_potencia as RG
import registro_vazao as V
import resultados_tg2 as T
import secoes_e_degraus as SD

OUT = R.OUT / "secoes_e_degraus"
DEG = [(195.0, (172.0, 186.0), (197.0, 211.0)), (214.0, (200.0, 212.0), (217.0, 237.0)),
       (240.5, (225.0, 239.0), (243.0, 260.0))]
SECS = {"meio 40–80": (40, 80), "2ª metade 60–120": (60, 120), "final 80–120": (80, 120)}


def linhas():
    L = {"marcações do autor": AL.manual()}
    for s in (0, 1, 2):
        L[f"2b, semente {s}"] = AL.run("nenhum_d4" + ("" if s == 0 else f"_s{s}"))
    return L


def main():
    L = linhas()
    fim = SD.borda(AL.run("nenhum_d4")[0])
    res = {}
    for sec, (x0, x1) in SECS.items():
        for nome, (Yb, Yc) in L.items():
            H = AL.serie_H(Yb, Yc, x0, x1)
            for x, (a1, b1), (a2, b2) in DEG:
                if b2 > fim[sec]:
                    continue
                p1, p2 = SD.taxa_trecho(H, a1, b1), SD.taxa_trecho(H, a2, b2)
                if p1 and p2 and p1["r"] > 0 and p2["r"] > 0:
                    res.setdefault(sec, {}).setdefault(nome, []).append(dict(
                        t=x, G1=p1["G"], r1=p1["r"], G2=p2["G"], r2=p2["r"],
                        n=float(np.log(p2["r"] / p1["r"]) / np.log(p2["G"] / p1["G"]))))
    (OUT / "degraus_tg.json").write_text(json.dumps(dict(borda=fim, degraus=res), ensure_ascii=False, indent=1))
    todos = []
    print("borda (s):", {k: round(v) for k, v in fim.items() if k in SECS})
    for sec, d in res.items():
        for nome, g in d.items():
            print(f"{sec:17s} {nome:20s} " + " | ".join(f"{x['t']:.0f} s: G {x['G1']:.1f}→{x['G2']:.1f}, ṙ {x['r1']:.3f}→{x['r2']:.3f}, n {x['n']:.2f}" for x in g))
            if sec != "meio 40–80":
                todos += [x["n"] for x in g]
    print(f"\nfim do grão (2ª metade e final), todas as linhas: n mediana {np.median(todos):.2f}, faixa {min(todos):.2f} a {max(todos):.2f}, {len(todos)} casos")

    # ---------------------------------------------------------------- figura do TG
    tq, q = SD.vazao()
    tt = SD.t(np.arange(963))
    fig, axs = plt.subplots(1, 2, figsize=(F.LARG, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = axs[0]
    for nome, cor, lw in (("marcações do autor", F.VERDE, 1.4), ("2b, semente 0", "#0a8fc4", 1.2)):
        Yb, Yc = L[nome]
        a.plot(tt, AL.serie_H(Yb, Yc, 80, 120), color=cor, lw=lw, label=f"altura, {nome}")
    a.set_xlim(160, 270); a.set_ylim(15, 36)
    a.set_xlabel(F.ROT_TEMPO); a.set_ylabel("altura da porta, 80 a 120 mm (mm)")
    b = a.twinx()
    b.plot(tq, q * V.RHO, color="0.55", lw=0.9, label="vazão de oxidante")
    b.set_ylabel("vazão de oxidante (g/s)"); b.set_ylim(0, 8)
    for x, (a1, b1), (a2, b2) in DEG:
        a.axvspan(a1, b1, color="#2f6bff", alpha=0.10, lw=0); a.axvspan(a2, b2, color="#ff5cff", alpha=0.10, lw=0)
    a.set_title("(a) seção final", loc="left")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper left")
    F.virgula(a); F.virgula(b)
    c = axs[1]
    estilo = {"marcações do autor": (F.VERDE, "o"), "2b, semente 0": ("#0a8fc4", "s"), "2b, semente 1": ("#2f6bff", "^"),
              "2b, semente 2": ("#5b3a8c", "v")}
    pos = {195.0: 0, 214.0: 1, 240.5: 2}
    desl = dict(zip(estilo, (-0.21, -0.07, 0.07, 0.21)))
    for sec, cheio in (("final 80–120", True), ("2ª metade 60–120", False)):
        for nome, g in res.get(sec, {}).items():
            cor, mk = estilo[nome]
            for x in g:
                c.plot(pos[x["t"]] + desl[nome] + (0 if cheio else 0.03), x["n"], mk, ms=5, mec=cor if not cheio else "black",
                       mfc=cor if cheio else "white", mew=0.9 if not cheio else 0.4)
    c.axhline(float(np.median(todos)), color="0.45", lw=0.9, ls=(0, (4, 2)), label=f"mediana, {F.br(float(np.median(todos)), 2)}")
    for nome, (cor, mk) in estilo.items():
        c.plot([], [], mk, color=cor, mec="black", mew=0.4, label=nome)
    c.plot([], [], "o", mfc="white", mec="0.3", label="vazio: 2ª metade; cheio: final")
    c.set_xticks([0, 1, 2]); c.set_xticklabels(["195 s\n" + r"$G\times1{,}7$", "214 s\n" + r"$G\times0{,}65$", "240,5 s\n" + r"$G\times2{,}5$"])
    c.set_xlim(-0.5, 2.5); c.set_ylim(0, 1.2); c.grid(alpha=0.25, axis="y")
    c.set_ylabel(r"expoente, $\ln(\dot r_2/\dot r_1)\,/\,\ln(G_2/G_1)$")
    c.set_title("(b) expoente medido em cada degrau", loc="left")
    c.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=1)
    c.yaxis.set_major_formatter(F.VIRG)
    fig.tight_layout()
    fig.savefig(F.SAIDA / "fig_degraus.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------------- figuras separadas para o autor
    res_all = json.loads((OUT / "secoes_e_degraus.json").read_text())["secoes"]
    secs6 = list(SD.SECOES)
    Gg = np.geomspace(2.5, 9, 30)
    fig, axs = plt.subplots(2, 3, figsize=(F.LARG * 1.8, 6.2), sharey=True)
    for ax, sec in zip(axs.ravel(), secs6):
        for nome, cor in (("marcações do autor", F.VERDE), ("melhor rede (2b nova, semente 1)", "#0a8fc4")):
            J = res_all[sec][nome]["janelas"]; a_ = res_all[sec][nome]["ajuste"]
            G = np.array([p["G"] for p in J]); r = np.array([p["r"] for p in J])
            ax.plot(G, r, "o", color=cor, mec="black", mew=0.4, ms=5)
            Gl = np.geomspace(G.min(), G.max(), 20)
            ax.plot(Gl, a_["a"] * Gl ** a_["n"], color=cor, lw=1.3,
                    label=f"{nome.split(' (')[0]}: n {F.br(a_['n'], 2)}, {F.br(a_['razao'], 2)}× a lei")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, which="both"); ax.set_ylim(0.025, 0.12)
        ax.set_title(f"{sec}: 5 janelas da janela de análise", loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.3, loc="lower right"); ax.set_xlabel(r"$G_{ox}$ (kg/(m$^2$ s))")
    for ax in axs[:, 0]:
        ax.set_ylabel("taxa de regressão (mm/s)")
    fig.tight_layout(); fig.savefig(OUT / "janelas_por_secao.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(F.LARG * 1.8, 3.3), sharey=True)
    for ax, sec in zip(axs, SECS):
        for nome, g in res.get(sec, {}).items():
            cor, mk = estilo[nome]
            for x in g:
                ax.annotate("", xy=(x["G2"], x["r2"]), xytext=(x["G1"], x["r1"]), arrowprops=dict(arrowstyle="->", color=cor, lw=1.1))
                ax.plot([x["G1"]], [x["r1"]], mk, color=cor, ms=4, mec="black", mew=0.3)
                Gp = np.array([x["G1"], x["G2"]])
                ax.annotate(f"{x['t']:.0f} s", (x["G2"], x["r2"]), fontsize=6, xytext=(3, 0), textcoords="offset points")
        for nome, (cor, mk) in estilo.items():
            ax.plot([], [], mk, color=cor, mec="black", mew=0.3, label=nome)
        ax.plot([], [], color="0.45", lw=0.7, ls=(0, (3, 2)), label="se n fosse 0,70")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, which="both")
        ax.set_title(f"{sec}: degraus de vazão", loc="left", fontsize=8.5); ax.set_xlabel(r"$G_{ox}$ (kg/(m$^2$ s))")
        ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    axs[0].set_ylabel("taxa de regressão (mm/s)")
    fig.tight_layout(); fig.savefig(OUT / "degraus_por_secao.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("figuras gravadas")


if __name__ == "__main__":
    main()
