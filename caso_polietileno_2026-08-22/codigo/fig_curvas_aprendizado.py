"""Curvas de aprendizado das redes de desenvolvimento R1, R3 e R4 (08/11/2026).

Lê os históricos gravados por experimento_p2_v2.py (perda de treino e erro mediano
nos três quadros de validação, a cada 50 passos) e desenha a figura para o TG2.
Mesmo estilo de figuras_tg2.py (Times, eixos limpos, 300 dpi).

    .venv/bin/python codigo/fig_curvas_aprendizado.py [saida.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import textos_tg  # noqa: F401  termos do TG nos textos das figuras (11/11/2026)

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "experimento_p2_v2_2026-10-10/runs"
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "figuras/fig_curvas_aprendizado.png"

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})

AZUL, MAGENTA, VERDE = "#2f6bff", "#b000c8", "#00a86b"
FAMILIAS = (
    ("R1", ["R1_s0"], AZUL, (0, (4, 2)), "R1, rede própria, rótulos de primeira geração"),
    ("R3", ["R3_s0", "R3_s1", "R3_s2"], MAGENTA, "-", "R3, rede própria, rótulos híbridos"),
    ("R4", ["R4_s0", "R4_s1", "R4_s2"], VERDE, "-", "R4, ResNet18 da ImageNet, rótulos híbridos"),
)


def ler(run):
    h = json.loads((RUNS / run / "hist.json").read_text())
    b = json.loads((RUNS / run / "best.json").read_text())
    return ([d["passo"] for d in h], [d["perda"] for d in h], [d["val_med_px"] for d in h], b["passo"], b["val_med_px"])


fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
for nome, runs, cor, ls, rotulo in FAMILIAS:
    for i, run in enumerate(runs):
        p, perda, val, pb, vb = ler(run)
        kw = dict(color=cor, lw=1.4, ls=ls, label=rotulo if i == 0 else None)
        axs[0].plot(p, perda, **kw)
        axs[1].plot(p, val, **kw)
        axs[1].plot([pb], [vb], "o", ms=5, mfc="white", mec=cor, mew=1.3, zorder=5)
    desloc = {"R1": 7, "R3": -6, "R4": 0}[nome]
    axs[1].annotate(nome, (p[-1], val[-1]), xytext=(5, desloc), textcoords="offset points", va="center",
                    fontsize=8, color="0.15")

for ax, ylab, let, ticks in ((axs[0], "perda de treinamento", "(a)", [3, 10, 30, 100]),
                             (axs[1], "erro de validação (px)", "(b)", [1, 3, 10, 30, 100])):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0, 1300)
    ax.set_xlabel("passo")
    ax.set_ylabel(ylab)
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.set_title(let, loc="left", fontsize=9)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=2.2, bbox_to_anchor=(0.5, -0.02),
           columnspacing=1.2)
fig.tight_layout(w_pad=1.5, rect=(0, 0.1, 1, 1))
SAIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SAIDA)
print(SAIDA)
