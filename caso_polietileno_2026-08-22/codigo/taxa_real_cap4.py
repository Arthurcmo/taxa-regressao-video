"""Taxa de regressão no tempo real e lei de regressão do ensaio.

A taxa entra no tempo real (tempo marcado pela câmera vezes o fator de tempo). A medida
começa em 45 s depois da ignição, quando começam as marcações e o recuo monótono da
superfície.

Intervalos, no grão todo (10 a 120 mm), pelos extremos da altura média da porta:
  ensaio            de 45 s ao fim da janela de análise, como um ponto;
  janela de análise como um ponto e em 5, 10, 20 e 50 janelas iguais.
Lei final de cada método: n = mediana dos expoentes nos degraus de vazão (degraus_tg.py)
e a pela taxa da janela de análise.
Linhas das figuras: marcações manuais, 2b e SM1 v4 (média das sementes 0, 1 e 2).

Saída: experimento_recompensa_lei_2026-11-06/comparacao_geral/taxa_real_cap4.json e, em figuras_tg2/:
  fig_taxa_ensaio.png    taxa contra fluxo, intervalos e reta ajustada
  fig_taxa_tamanho.png   as mesmas linhas com intervalos de tamanhos diferentes
  fig_lei_final.png      intervalos, reta ajustada e lei final de cada método

    BASE_TEMPO=real PYTHONPATH=codigo python -W ignore codigo/taxa_real_cap4.py [--figuras]
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import comparacao_geral as CG
import conferencia_sem_marcacao as CS
import ensaio as E
import figuras_tg2 as F
import janelas_taxa as Q
import recompensa_lei as R
import lei_potencia as RG
import resultados_tg2 as T

OUT = R.OUT / "comparacao_geral"
S = T._tm.s
CAM = T.C.CAM
T0 = 45.0
N_JAN = (5, 10, 20, 50)
PAINEIS = (5, 10, 15, 50)
N_MARC = (5, 10)                       # as marcações têm um quadro a cada 5 a 12 s
VERDE, AZUL = "#2ca02c", "#1f4fd6"
def k_de(t):
    return E.AMOSTRA_IGNICAO + t / (CAM * S)


def virg(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def ponto(H, t0, t1):
    p = T.um_ponto(H, k_de(t0), k_de(t1))
    return dict(r=float(p["r_real"]), G=float(p["G"]))


def janelas(H, n):
    P = Q.janelas_k(H, T._tm, T._corte, T.FIM, n)
    r = np.array([w["r"] * 1e3 for w in P]); G = np.array([w["md"] / (Q.L_M * w["Hm"]) for w in P])
    return G, r


def ajuste(G, r):
    aj = RG.ajuste_proprio(G, r)
    ok = r > 0
    disp = float(np.std(np.log(r[ok]) - np.log(aj["a"] * G[ok] ** aj["n"]), ddof=2)) if ok.sum() > 3 else np.nan
    return dict(a=float(aj["a"]), n=float(aj["n"]), r2=float(aj["r2"]),
                rmin=float(r.min()), rmax=float(r.max()), disp=disp)


def medir(Yb, Yc, ns=N_JAN):
    H = CG.serie(Yb, Yc, 10, 120)
    out = dict(ensaio=ponto(H, T0, T.FIM), analise=ponto(H, T._corte, T.FIM), janelas={})
    for n in ns:
        G, r = janelas(H, n)
        out["janelas"][str(n)] = dict(G=G.tolist(), r=r.tolist(), **ajuste(G, r))
    rm, Gm, _ = CS.pontos(AL.serie_H(Yb, Yc, 40, 80))           # taxa no tempo real
    am = CS.ajuste(Gm, rm)
    out["meio"] = dict(a=float(am["a"]), n=float(am["n"]), r2=float(am["r2"]))
    return out


def media(sem):
    """Média das sementes: pontos únicos e janelas (mesmos instantes em todas as sementes)."""
    m = {}
    for c in ("ensaio", "analise"):
        r = np.mean([s[c]["r"] for s in sem]); G = np.mean([s[c]["G"] for s in sem])
        m[c] = dict(r=float(r), G=float(G),
                    rmin=float(min(s[c]["r"] for s in sem)), rmax=float(max(s[c]["r"] for s in sem)))
    m["janelas"] = {}
    for n in sem[0]["janelas"]:
        G = np.mean([s["janelas"][n]["G"] for s in sem], axis=0)
        r = np.mean([s["janelas"][n]["r"] for s in sem], axis=0)
        m["janelas"][n] = dict(G=G.tolist(), r=r.tolist(), rmed=float(np.mean(r)), **ajuste(G, r))
    m["meio"] = dict(a=float(np.mean([s["meio"]["a"] for s in sem])), n=float(np.mean([s["meio"]["n"] for s in sem])))
    return m


PRINCIPAIS = {"2b": N_JAN + (15,), "SM1 v4": N_JAN + (15,)}


def calcular():
    res = {"marcações manuais": media([medir(*AL.manual(), ns=N_MARC)])}
    for nome, sem in CG.familias().items():
        ns = tuple(sorted(set(PRINCIPAIS.get(nome, (10,)))))
        res[nome] = media([medir(*d["taxa"], ns=ns) for d in sem.values()])
        print(nome, "ok", flush=True)
    return res


# ----------------------------------------------------------------------------- figuras
def eixos(ax, xlim=(2.8, 6.0), ylim=(0.025, 0.10)):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([3, 4, 5, 6]); ax.set_xticklabels(["3", "4", "5", "6"])
    yt = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    ax.set_yticks([y for y in yt if ylim[0] <= y <= ylim[1]])
    ax.set_yticklabels([f"{y:g}".replace(".", ",") for y in yt if ylim[0] <= y <= ylim[1]])
    ax.minorticks_off(); ax.grid(alpha=0.25)


MAGENTA = "#d61fd6"
PRINC = (("marcações manuais", VERDE, "o", False), ("2b", AZUL, "s", True), ("SM1 v4", MAGENTA, "^", True))
ROT = {"marcações manuais": "marcações manuais", "2b": "2b, com marcação", "SM1 v4": "SM1 v4, sem marcação"}


def fig_ensaio(res):
    fig, ax = plt.subplots(figsize=(F.LARG, 4.0))
    gg = np.geomspace(2.8, 6.2, 50)
    for nome, cor, mk, cheio in PRINC:
        J = res[nome]["janelas"]["10"]
        G, r = np.array(J["G"]), np.array(J["r"])
        g2 = np.geomspace(G.min(), G.max(), 20)
        ax.plot(G, r, mk, ms=5.5, color=cor, mfc=cor if cheio else "white", mec=cor, mew=1.2, ls="none")
        ax.plot(g2, J["a"] * g2 ** J["n"], color=cor, lw=1.1)
        ax.plot([], [], marker=mk, ms=5.5, color=cor, mfc=cor if cheio else "white", mec=cor, mew=1.2, lw=1.1,
                label=ROT[nome])
    eixos(ax, xlim=(3.0, 6.1), ylim=(0.04, 0.09))
    ax.set_xlabel(r"fluxo mássico de oxidante $\bar G_{ox}$ (kg/(m$^2$ s)), escala log.")
    ax.set_ylabel(r"taxa de regressão $\bar{\dot r}$ (mm/s), escala log.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    F.salvar(fig, "fig_taxa_ensaio")


def fig_tamanho(res):
    fig, axs = plt.subplots(2, 2, figsize=(F.LARG, 5.4), sharex=True, sharey=True)
    gg = np.geomspace(2.8, 6.0, 50)
    for ax, n in zip(axs.flat, PAINEIS):
        dur = (T.FIM - T._corte) / n
        J = res["2b"]["janelas"][str(n)]
        ax.plot(J["G"], J["r"], "s", ms=4.5 if n < 50 else 3, color=AZUL, mec=AZUL)
        m = res["marcações manuais"]["janelas"].get(str(n))
        if m:
            ax.plot(m["G"], m["r"], "o", ms=5, mfc="white", mec=VERDE, mew=1.3)
        eixos(ax, xlim=(3.0, 6.1), ylim=(0.03, 0.095))
        ax.set_title(f"{n} intervalos de {virg(dur, 1)} s", loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    for ax in axs[1]:
        ax.set_xlabel(r"$\bar G_{ox}$ (kg/(m$^2$ s)), escala log.", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\bar{\dot r}$ (mm/s), escala log.", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=VERDE, mew=1.3, label="marcações manuais"),
         plt.Line2D([], [], marker="s", ls="none", color=AZUL, label="2b")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    F.salvar(fig, "fig_taxa_tamanho")


# ----------------------------------------------------------------------------- lei final
DEGRAUS = R.OUT / "secoes_e_degraus/degraus_tg.json"
SECOES_DEGRAU = ("2ª metade 60–120", "final 80–120")
BASE_DEGRAU = {"marcações manuais": "marcações do autor", "2b": "2b"}


def lei_final(res):
    """Lei de regressão deste ensaio, por método: n = mediana dos expoentes dos degraus de vazão
    (resposta ao fluxo com a porta quase fixa) e a pela taxa da janela de análise
    (uma janela só).
    A SM1 v4 não alcança os degraus e usa a mediana de todos eles."""
    d = json.loads(DEGRAUS.read_text())["degraus"]
    por = {}
    for s in SECOES_DEGRAU:
        for k, v in d[s].items():
            por.setdefault(k.split(",")[0], []).extend(x["n"] for x in v)
    todos = [x for v in por.values() for x in v]
    out = {}
    for nome, _, _, _ in PRINC:
        ns = por.get(BASE_DEGRAU.get(nome, ""), todos)
        n = float(np.median(ns)); e = res[nome]["analise"]
        a = e["r"] / e["G"] ** n
        out[nome] = dict(a=float(a), n=n, n_min=float(min(ns)), n_max=float(max(ns)), n_degraus=len(ns),
                         degraus_proprios=nome in BASE_DEGRAU, r=e["r"], G=e["G"],
                         r_45=float(a * 4.5 ** n))
    return out


def fig_lei_final(res):
    lit = dict(lei_final=lei_final(res))
    (OUT / "lei_final_cap4.json").write_text(json.dumps(lit, ensure_ascii=False, indent=1))
    fig, axs = plt.subplots(1, 3, figsize=(F.LARG, 3.3), sharey=True)
    for ax, (nome, cor, mk, cheio), let in zip(axs, PRINC, "abc"):
        J = res[nome]["janelas"]["10"]; G, r = np.array(J["G"]), np.array(J["r"])
        ax.plot(G, r, mk, ms=4, color=cor, mfc=cor if cheio else "white", mec=cor, mew=1.0, zorder=4)
        g2 = np.geomspace(G.min(), G.max(), 20)
        ax.plot(g2, J["a"] * g2 ** J["n"], color=cor, lw=1.3, zorder=4)
        q = lit["lei_final"][nome]
        g3 = np.geomspace(3.0, 8.0, 40)
        ax.plot(g3, q["a"] * g3 ** q["n"], color=cor, lw=1.2, ls="--", zorder=3)
        ax.plot([q["G"]], [q["r"]], "*", ms=11, color=cor, mec="black", mew=0.7, zorder=6)
        eixos(ax, xlim=(2.9, 8.3), ylim=(0.03, 0.11))
        ax.set_xticks([3, 4, 5, 6, 8]); ax.set_xticklabels(["3", "4", "5", "6", "8"])
        ax.set_title(f"({let}) {ROT[nome]}", loc="left", fontsize=8)
        vm = lambda x, d: virg(x, d).replace(",", "{,}")
        txt = (f"intervalos: $\\dot r = {vm(J['a'], 4)}\\,G^{{{vm(J['n'], 2)}}}$\n"
               f"lei final: $\\dot r = {vm(q['a'], 4)}\\,G^{{{vm(q['n'], 2)}}}$")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=6.6, color="0.1")
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel(r"$\bar G_{ox}$ (kg/(m$^2$ s)), escala log.", fontsize=7.5)
    axs[0].set_ylabel(r"$\bar{\dot r}$ (mm/s), escala log.", fontsize=8)
    h = [plt.Line2D([], [], color="0.4", lw=1.3, marker="s", ms=3.5, label="dez intervalos e reta ajustada a eles"),
         plt.Line2D([], [], color="0.4", lw=1.0, ls="--", marker="*", ms=8, mec="black", mew=0.6,
                    label="média da janela de análise e lei final (n dos degraus)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    F.salvar(fig, "fig_lei_final")
    print(json.dumps(lit, ensure_ascii=False, indent=1))


def main():
    if "--figuras" in sys.argv:
        res = json.loads((OUT / "taxa_real_cap4.json").read_text())["linhas"]
    else:
        res = calcular()
        (OUT / "taxa_real_cap4.json").write_text(json.dumps(dict(linhas=res), ensure_ascii=False, indent=0))
    for nome, d in res.items():
        e, a_ = d["ensaio"], d["analise"]
        print(f"{nome:18s} 45 s: {e['r']:.4f} ({e['rmin']:.4f}–{e['rmax']:.4f}) G {e['G']:.2f} | "
              f"análise: {a_['r']:.4f} G {a_['G']:.2f} | meio {d['meio']['a']:.4f} G^{d['meio']['n']:.2f}")
        for n, J in sorted(d["janelas"].items(), key=lambda x: int(x[0])):
            print(f"   {n:>2} janelas: r̄ {J['rmed']:.4f} {J['a']:.4f} G^{J['n']:.2f} R2 {J['r2']:.2f} "
                  f"r {J['rmin']:.4f}–{J['rmax']:.4f} disp {J['disp'] * 100:.0f}%")
    fig_ensaio(res)
    fig_tamanho(res)
    fig_lei_final(res)


if __name__ == "__main__":
    main()
