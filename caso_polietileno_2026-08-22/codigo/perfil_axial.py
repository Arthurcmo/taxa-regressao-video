"""Perfil da taxa ao longo do grão, recirculação na entrada e o todo contra o local.

Faixas de 5 mm de 10 a 120 mm. Em cada faixa, a altura da porta é a média das colunas
(AL.serie_H) e a taxa sai dos extremos da altura em dois intervalos, no tempo real desde a
ignição:
  ensaio             de 45 s ao fim da janela de análise;
  janela de análise  do início ao fim da janela de análise da configuração.
O fluxo local é G_ox = ṁ / (L H̄), com H̄ a média dos extremos.
Linhas: marcações manuais e a 2b (média das sementes 0, 1 e 2).
Saídas: fig_perfil_axial.png e fig_mapa_taxa.png; números em perfil_axial.json.

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/perfil_axial.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import ensaio as E
import faixas_grao as FG
import figuras_tg2 as F
import relogio_e_corte as C
import resultados_tg2 as T

OUT = AL.R.OUT / "perfil_axial"
S = E.FATOR_TEMPO_REAL
L_M = E.LARGURA_PORTA_MM * 1e-3
BINS = [(x, x + 5) for x in range(10, 120, 5)]
XM = np.array([0.5 * (a + b) for a, b in BINS])
CAM = E.CAM


def k_de(t):
    return E.AMOSTRA_IGNICAO + t / (CAM * S)


INTERV = {"ensaio desde 45 s": (k_de(45.0), k_de(T.FIM)),
          "janela de análise": (k_de(T._corte), k_de(T.FIM))}


def linhas():
    L = {"marcações do autor": [AL.manual()]}
    L["2b"] = [AL.run("nenhum_d4" + ("" if s == 0 else f"_s{s}")) for s in (0, 1, 2)]
    return L


def H_em(H, k):
    return float(np.interp(k, np.arange(len(H)), H))


def extremos(H, k0, k1):
    tm = T._tm
    Ha, Hb = H_em(H, k0), H_em(H, k1)
    dt = (k1 - k0) * CAM * S
    md = C.mdot(tm, k0, k1)                                   # kg/s
    r = (Hb - Ha) / (2 * dt)                                  # mm/s
    G = md / (L_M * 0.5 * (Ha + Hb) * 1e-3)
    return r, G, Ha, Hb


def perfil(Ls, x0, x1, k0, k1):
    rs, Gs = [], []
    for Yb, Yc in Ls:
        H = AL.serie_H(Yb, Yc, x0, x1)
        k0_ = k0 if np.isfinite(H_em(H, k0)) else k0 + 0.1
        r, G, _, _ = extremos(H, k0_, k1)
        rs.append(r); Gs.append(G)
    return float(np.nanmean(rs)), float(np.nanmean(Gs)), float(np.nanstd(rs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = linhas()
    res = {}
    for nome, Ls in L.items():
        res[nome] = {}
        for iv, (k0, k1) in INTERV.items():
            r = [perfil(Ls, a, b, k0, k1) for a, b in BINS]
            todo = perfil(Ls, 10, 120, k0, k1)
            res[nome][iv] = dict(r=[x[0] for x in r], G=[x[1] for x in r], dp=[x[2] for x in r],
                                 todo=dict(r=todo[0], G=todo[1]))
    # alturas em instantes escolhidos (2b, média das sementes)
    tsnap = (45.0, 70.0, float(T._corte), 130.0, 160.0, float(T.FIM))
    Hs = {}
    for t in tsnap:
        k = k_de(t)
        Hs[t] = [float(np.nanmean([H_em(AL.serie_H(Yb, Yc, a, b), k) for Yb, Yc in L["2b"]])) for a, b in BINS]
    (OUT / "perfil_axial.json").write_text(json.dumps(dict(x=XM.tolist(), res=res, H=Hs), ensure_ascii=False, indent=1))

    print("faixa (mm) | janela de análise: ṙ (mm/s), G | ensaio desde 45 s: ṙ")
    for i, x in enumerate(XM):
        d = res["2b"]
        print(f"{x:5.1f} | {d['janela de análise']['r'][i]:.4f} {d['janela de análise']['G'][i]:.2f} | "
              f"{d['ensaio desde 45 s']['r'][i]:.4f}")
    for nome in L:
        for iv in INTERV:
            t_ = res[nome][iv]["todo"]
            print(f"{nome:20s} {iv:22s} grão todo: {t_['r']:.4f} mm/s em G {t_['G']:.2f}")

    # --------------------------------------------------------------------- figura do perfil
    fig, axs = plt.subplots(2, 1, figsize=(F.LARG, 5.4), sharex=True)
    a = axs[0]
    cores = ["#b9d7f0", "#7fb3e0", "#3f86c9", "#1f5fa8", "#123f78", "#081f40"]
    for (t, H), cor in zip(Hs.items(), cores):
        a.plot(XM, H, "o-", color=cor, ms=3, lw=1.2, label=f"{F.br(t, 0)} s")
    a.set_ylabel("altura da porta (mm)"); a.legend(frameon=False, fontsize=7, ncol=3, loc="upper right", title="tempo desde a ignição", title_fontsize=7)
    a.set_title("(a) forma da porta ao longo do grão", loc="left")
    b = axs[1]
    est = {"janela de análise": ("#0a8fc4", "o"), "ensaio desde 45 s": ("#5b3a8c", "s")}
    for iv, (cor, mk) in est.items():
        d = res["2b"][iv]
        b.errorbar(XM, d["r"], yerr=d["dp"], color=cor, marker=mk, ms=3.5, lw=1.2, capsize=1.5, label=f"{iv}, 2b")
        dm = res["marcações do autor"][iv]
        b.plot(XM, dm["r"], mk, mfc="white", mec=F.VERDE, ms=4, mew=1.0)
        b.axhline(d["todo"]["r"], color=cor, lw=0.8, ls=(0, (4, 2)))
    b.plot([], [], "o", mfc="white", mec=F.VERDE, label="marcações do autor")
    b.plot([], [], color="0.3", lw=0.8, ls=(0, (4, 2)), label="média do grão todo")
    b.set_ylabel("taxa de regressão (mm/s)"); b.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper right")
    b.set_title("(b) taxa local pelo método dos extremos", loc="left")
    for ax in axs:
        ax.grid(alpha=0.25); F.virgula(ax)
    axs[1].set_xlim(8, 122); axs[1].set_xlabel("posição ao longo do grão (mm)")
    fig.tight_layout()
    fig.savefig(F.SAIDA / "fig_perfil_axial.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # --------------------------------------------------------------------- mapa da taxa local
    ts = np.arange(45.0, T.FIM - 20 + 1e-6, 2.5)
    M = np.full((len(BINS), len(ts)), np.nan)
    Hcache = {(a, b): [AL.serie_H(Yb, Yc, a, b) for Yb, Yc in L["2b"]] for a, b in BINS}
    for j, t0 in enumerate(ts):
        k0, k1 = k_de(t0), k_de(t0 + 20)
        for i, (a, b) in enumerate(BINS):
            M[i, j] = np.nanmean([(H_em(H, k1) - H_em(H, k0)) / (2 * 20.0) for H in Hcache[(a, b)]])
    fig, ax = plt.subplots(figsize=(F.LARG, 3.3))
    im = ax.pcolormesh(ts + 10, XM, M, cmap="viridis", vmin=0, vmax=0.12, shading="nearest")
    ax.axvspan(87.8 - 10, float(T._corte) + 10, color="white", alpha=0.88, lw=0)        # janelas que cruzam a 1ª interrupção
    ax.text(90.3, 118, "janelas que cruzam\na 1ª interrupção", ha="center", va="bottom", fontsize=6.5, color="0.15")
    ax.axvline(T._corte, color="white", lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel("centro da janela de 20 s, tempo desde a ignição (s)"); ax.set_ylabel("posição ao longo do grão (mm)")
    cb = fig.colorbar(im, ax=ax, pad=0.01, format=F.VIRG); cb.set_label("taxa local (mm/s)")
    ax.invert_yaxis()
    F.virgula(ax)
    fig.tight_layout(); fig.savefig(F.SAIDA / "fig_mapa_taxa.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("figuras gravadas")


if __name__ == "__main__":
    main()
