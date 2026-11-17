"""Figuras do ensaio, geradas a partir dos resultados salvos.

Nada aqui recalcula método nenhum: só lê o que os experimentos já gravaram e
desenha no padrão do texto (fonte Times, rótulos em português, sem datas de
rodada nas legendas). Linhas sobre o vídeo usam a paleta fria do projeto, com
contorno escuro, porque vermelho, laranja e amarelo somem na chama.

    python codigo/figuras_tg2.py                 # todas
    python codigo/figuras_tg2.py folha_teste     # só uma

Saída em ../figuras_tg2/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import textos_tg  # noqa: F401  termos do TG nos textos das figuras (11/11/2026)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ.parent / "figuras_tg2"
E1 = RAIZ / "experimento_prioridades12_2026-10-09"
E2 = RAIZ / "experimento_p2_v2_2026-10-10"
E3 = RAIZ / "experimento_p3_usuario_2026-10-11"
E4 = RAIZ / "experimento_s_sessao_2026-10-28"
PERFIS = RAIZ / "resultados/revisao_humana/perfis"
# Perfis do autor usados como referência. O de 18/10 (19 pontos) foi deixado de
# fora: em k=179 ele fica cerca de 44 px abaixo do perfil vigente de 21/10, no
# mesmo quadro, e o próprio autor seguiu marcando no de 21/10.
PERFIS_AUTOR = ("marcacoes-manuais-b6f9e36d", "teste1-15bf532d", "sessao-2026-10-21-41227435")

# Base de tempo: a do marcador, o tempo de câmera por amostra. As taxas lidas dos
# resultados salvos são multiplicadas por FATOR_TEMPO, que no tempo do marcador vale 1.
import ensaio as E
from experimento_s_sessao import DT, BASE_TEMPO
ROT_TEMPO = "tempo desde a ignição (s)" if BASE_TEMPO == "real" else "tempo marcado pela câmera, desde a ignição (s)"
FATOR_TEMPO = E.CAM / DT              # 1 no tempo do marcador
K_IGN = E.AMOSTRA_IGNICAO   # amostra da ignição
K_MOLDURA = 643             # depois disso a face de baixo encosta na moldura
K_FACE_CIMA_ANDA = 269      # antes disso a face de cima quase não anda
ESC = E.MM_POR_PX           # mm/px
XS = 20 * (np.arange(93) + 0.5)   # centro das colunas na ROI
LARG = 6.3                  # polegadas, largura útil da página

# Paleta fria sobre o vídeo (RGB 0..1)
CIANO = "#00e5ff"
AZUL = "#2f6bff"
VIOLETA = "#c27bff"
VERDE = "#6aff00"
MAGENTA = "#ff5cff"
BRANCO = "#ffffff"
CONTORNO = [pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]
CONTORNO_TXT = [pe.withStroke(linewidth=2.5, foreground="black")]

from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

VIRG = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def virgula(*axs):
    for ax in axs:
        ax.xaxis.set_major_formatter(VIRG)
        ax.yaxis.set_major_formatter(VIRG)


def br(v, casas):
    return f"{v:.{casas}f}".replace(".", ",")


plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def t_de_k(k):
    return (np.asarray(k) - K_IGN) * DT


FOTOS = {"fig_ensaio_quadros", "fig_transiente", "fig_fuligem", "fig_quimograma", "fig_folha_visual",
         "fig_folha_teste"}


def salvar(fig, nome):
    """Figuras com quadros do vídeo saem em JPEG (o PDF fica dez vezes menor);
    gráficos saem em PNG."""
    SAIDA.mkdir(parents=True, exist_ok=True)
    if nome in FOTOS:
        fig.savefig(SAIDA / f"{nome}.jpg", pil_kwargs={"quality": 90})
    else:
        fig.savefig(SAIDA / f"{nome}.png")
    plt.close(fig)
    print("ok", nome)


_meta = None


def meta():
    global _meta
    if _meta is None:
        _meta = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())
    return _meta


def quadro(k: int) -> np.ndarray:
    """Quadro inteiro do vídeo (RGB) correspondente à amostra k."""
    m = meta()
    cap = cv2.VideoCapture(m["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(m["quadro_arquivo"][k]))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"quadro {k} ilegível")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def roi(k: int) -> np.ndarray:
    y0, y1, x0, x1 = meta()["roi"]
    return quadro(k)[y0:y1, x0:x1]


def marcas_autor(face: str = "baixo") -> dict[int, np.ndarray]:
    """Pontos clicados pelo autor (exclui perfis com pontos de outra origem).
    Devolve {k: array (n, 2) de (x_roi, y_roi)}."""
    pts: dict[int, list] = {}
    for nome in PERFIS_AUTOR:
        g = PERFIS / nome / "guia.csv"
        if not g.exists():
            continue
        for r in csv.DictReader(g.open()):
            if r["tipo"] == "ancora" and r["face"] == face:
                pts.setdefault(int(r["k"]), []).append((float(r["x_px"]) - 30, float(r["y_roi_px"])))
    return {k: np.unique(np.array(v), axis=0) for k, v in pts.items()}


def linha(ax, x, y, cor, lw=1.6, **kw):
    ax.plot(x, y, color=cor, lw=lw, path_effects=CONTORNO, solid_capstyle="round", **kw)


def cruzes(ax, p, cor=VERDE, s=22):
    """Marcações do autor como círculos vazados (regra do texto: sem marcador X)."""
    ax.scatter(p[:, 0], p[:, 1], marker="o", s=s * 0.8, facecolors="none", edgecolors="black", linewidths=2.2, zorder=5)
    ax.scatter(p[:, 0], p[:, 1], marker="o", s=s * 0.8, facecolors="none", edgecolors=cor, linewidths=0.9, zorder=6)


def rotulo(ax, x, y, s, **kw):
    ax.text(x, y, s, color="white", fontsize=8, path_effects=CONTORNO_TXT, **kw)


# ---------------------------------------------------------------------------
def fig_ensaio():
    """Quadro antes da ignição e quadro durante a queima, com a nomenclatura."""
    y0, y1, x0, x1 = meta()["roi"]
    c = np.load(E1 / "classico.npz")
    fig, axs = plt.subplots(2, 1, figsize=(LARG, 4.4))
    for ax, k, titulo in ((axs[0], 20, "(a) antes da ignição"),
                          (axs[1], 300, f"(b) durante a queima, {t_de_k(300):.0f} s após a ignição")):
        im = roi(k)
        ax.imshow(im)
        ax.set_title(titulo, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    a = axs[0]
    rotulo(a, 900, 110, "grão superior", ha="center")
    rotulo(a, 900, 300, "porta de combustão", ha="center", va="center")
    rotulo(a, 900, 500, "grão inferior", ha="center")
    a.annotate("", xy=(330, 300), xytext=(40, 300),
               arrowprops=dict(arrowstyle="-|>", color=CIANO, lw=1.6))
    rotulo(a, 45, 270, "oxidante", ha="left")
    a.annotate("", xy=(XS[-1] + 10, 575), xytext=(XS[0] - 10, 575),
               arrowprops=dict(arrowstyle="<->", color=CIANO, lw=1.0))
    rotulo(a, 0.5 * (XS[0] + XS[-1]), 560, "comprimento do grão", ha="center")
    b = axs[1]
    j = 50
    ys, yi = c["y_sup"][300, j], c["y_inf"][300, j]
    b.annotate("face superior", xy=(XS[j], ys), xytext=(XS[j] + 230, ys - 120),
               color="white", fontsize=8, path_effects=CONTORNO_TXT,
               arrowprops=dict(arrowstyle="-|>", color=CIANO, lw=1.2))
    b.annotate("face inferior", xy=(XS[j], yi), xytext=(XS[j] + 230, yi + 110),
               color="white", fontsize=8, path_effects=CONTORNO_TXT,
               arrowprops=dict(arrowstyle="-|>", color=CIANO, lw=1.2))
    rotulo(b, 1500, 300, "chama", ha="center")
    rotulo(b, 60, 60, "cabeça do grão", ha="left")
    rotulo(b, 1830, 60, "saída", ha="right")
    fig.tight_layout(h_pad=0.6)
    salvar(fig, "fig_ensaio_quadros")


def fig_transiente():
    """A faixa escura sobe acima da parede vista antes da ignição."""
    c = np.load(E1 / "classico.npz")
    mk = marcas_autor("baixo")
    p0 = mk[0]
    ordem = np.argsort(p0[:, 0])
    parede = np.interp(XS, p0[ordem, 0], p0[ordem, 1])
    ks = (K_IGN + 15, K_IGN + 75, E.AMOSTRA_INICIO_MEDIDA)
    fig, axs = plt.subplots(len(ks), 1, figsize=(LARG, 3.9))
    for ax, k in zip(axs, ks):
        im = roi(k)[260:460]
        ax.imshow(im, extent=(0, im.shape[1], 460, 260))
        linha(ax, XS, parede, BRANCO, lw=1.2, ls=(0, (4, 3)))
        linha(ax, XS, c["y_inf"][k], VIOLETA, lw=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{t_de_k(k):.0f} s após a ignição", loc="left")
    axs[0].plot([], [], color=BRANCO, ls=(0, (4, 3)), lw=1.2, label="parede inferior antes da ignição")
    axs[0].plot([], [], color=VIOLETA, lw=1.4, label="faixa escura seguida pelo detector")
    leg = axs[0].legend(loc="lower right", frameon=True, facecolor="#222222", edgecolor="none",
                        labelcolor="white", fontsize=7)
    fig.tight_layout(h_pad=0.4)
    salvar(fig, "fig_transiente")


def fig_fuligem():
    """Convenção da superfície: marcações do autor e linha do clássico, com a
    cabeça do grão ampliada."""
    c = np.load(E1 / "classico.npz")
    k = 300
    p = marcas_autor("baixo")[k]
    im = roi(k)
    ya, yb = 300, 470
    fig = plt.figure(figsize=(LARG, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=(1, 1, 1.55))
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axs[:2]:
        ax.imshow(im[ya:yb], extent=(0, im.shape[1], yb, ya))
    xa, xb, za, zb = 0, 760, 330, 450
    axs[2].imshow(im[za:zb, xa:xb], extent=(xa, xb, zb, za))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axs[0].set_title(f"(a) face inferior, {t_de_k(k):.0f} s após a ignição, sem marcações", loc="left")
    axs[1].set_title("(b) marcações do autor e linha do detector clássico", loc="left")
    axs[2].set_title("(c) cabeça do grão ampliada", loc="left")
    for ax in axs[1:]:
        linha(ax, XS, c["y_inf"][k], VIOLETA, lw=1.4)
        cruzes(ax, p, s=12 if ax is axs[1] else 26)
    axs[2].set_xlim(xa, xb); axs[2].set_ylim(zb, za)
    axs[1].add_patch(plt.Rectangle((xa, za), xb - xa, zb - za, fill=False, ec="white", lw=0.8, ls=(0, (3, 2))))
    fig.tight_layout(h_pad=0.5)
    salvar(fig, "fig_fuligem")


def fig_quimograma():
    c = np.load(E1 / "classico.npz")
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    cols = (26, 50, 80)
    N = cubo.shape[0]
    t = t_de_k(np.arange(N))
    fig, axs = plt.subplots(len(cols), 1, figsize=(LARG, 6.6), sharex=True)
    for ax, j in zip(axs, cols):
        img = np.asarray(cubo[:, :, j, ::-1]).transpose(1, 0, 2)
        ax.imshow(img, aspect="auto", extent=(t[0], t[-1], img.shape[0], 0))
        linha(ax, t, c["y_sup"][:, j], AZUL, lw=1.1)
        linha(ax, t, c["y_inf"][:, j], CIANO, lw=1.1)
        for tt in (t_de_k(180), t_de_k(480)):
            ax.axvline(tt, color="white", lw=0.8, ls=(0, (3, 3)))
        ax.set_ylabel("y (px)")
        virgula(ax)
        rotulo(ax, t[0] + 2, 40, f"coluna em x = {XS[j]:.0f} px", ha="left", va="top")
        ax.set_ylim(593, 0)
    axs[-1].set_xlabel(ROT_TEMPO)
    axs[0].plot([], [], color=AZUL, lw=1.4, label="face superior")
    axs[0].plot([], [], color=CIANO, lw=1.4, label="face inferior")
    axs[0].legend(loc="lower right", frameon=True, facecolor="#222222", edgecolor="none",
                  labelcolor="white", fontsize=7)
    fig.tight_layout(h_pad=0.5)
    salvar(fig, "fig_quimograma")


def _csv_comparacao():
    linhas = list(csv.DictReader((E2 / "comparacao.csv").open()))
    d = {}
    for r in linhas:
        d.setdefault(r["rodada"], r)
    return d


def fig_ablacao():
    d = _csv_comparacao()

    def agrega(prefixo, dec):
        chaves = [k for k in d if k.startswith(prefixo) and k.endswith("/" + dec)]
        mae = np.array([float(d[k]["mae_usuario_px"]) for k in chaves])
        tx = np.array([float(d[k]["taxa_mm_s"]) for k in chaves])
        return mae.mean(), mae.std() if len(mae) > 1 else 0.0, tx.mean(), tx.std() if len(tx) > 1 else 0.0

    grupos = [
        ("C", "clássico", [("classico", None)]),
        ("CA", "clássico ancorado", [("semi_automatico", None)]),
        ("R0", "R0", [("R0_codex", "bruto"), ("R0_codex", "viterbi"), ("R0_codex", "viterbi_pos")]),
        ("R1", "R1", [("R1_s0", "bruto"), ("R1_s0", "viterbi"), ("R1_s0", "viterbi_pos")]),
        ("R2", "R2", [("R2_s0", "bruto"), ("R2_s0", "viterbi"), ("R2_s0", "viterbi_pos")]),
        ("R3", "R3", [("R3_", "bruto"), ("R3_", "viterbi"), ("R3_", "viterbi_pos")]),
        ("R4", "R4", [("R4_", "bruto"), ("R4_", "viterbi"), ("R4_", "viterbi_pos")]),
    ]
    cores = {None: "#555555", "bruto": "#7fa7d9", "viterbi": "#1f4e9c", "viterbi_pos": "#4cb5ae"}
    nomes = {"bruto": "quadro a quadro", "viterbi": "decodificação temporal, evidência = distribuição",
             "viterbi_pos": "decodificação temporal, evidência = posição"}
    fig, axs = plt.subplots(1, 2, figsize=(LARG, 3.6), sharey=True)
    ypos, rot = [], []
    y = 0.0
    lim = (45.0, None)
    for cod, nome, itens in grupos:
        y0 = y
        for pref, dec in itens:
            if dec is None:
                r = d[pref]
                m, sm, tx, st = float(r["mae_usuario_px"]), 0, float(r["taxa_mm_s"]), 0
            else:
                m, sm, tx, st = agrega(pref, dec)
            tx, st = tx * FATOR_TEMPO, st * FATOR_TEMPO
            for ax, val, sd, lmax in ((axs[0], m, sm, 45.0), (axs[1], tx, st, None)):
                v = min(val, lmax) if lmax else val
                ax.barh(y, v, height=0.8, color=cores[dec], xerr=sd if sd else None,
                        error_kw=dict(lw=0.8, capsize=2))
                if lmax and val > lmax:
                    ax.text(lmax - 0.5, y, br(val, 0), ha="right", va="center", color="white", fontsize=7)
                else:
                    fim = v + (sd if sd else 0)
                    ax.text(fim + (0.8 if lmax else 0.0015), y, br(val, 1) if lmax else br(val, 3),
                            va="center", fontsize=7)
            y += 1
        ypos.append((y0 + y - 1) / 2)
        rot.append(nome)
        y += 0.6
    axs[0].set_yticks(ypos, rot)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 45)
    axs[0].set_xlabel("erro médio contra as marcações do autor (px)")
    tref = 0.08790672 * FATOR_TEMPO
    axs[1].axvline(tref, color="black", lw=0.8, ls=(0, (3, 2)))
    axs[1].text(tref, -1.4, "referência de desenvolvimento", fontsize=7, ha="center", va="bottom")
    axs[1].set_xlim(0, 0.13)
    axs[1].set_xlabel("taxa da face inferior (mm/s)")
    virgula(axs[0], axs[1])
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_yticks(ypos, rot)
    alcas = [Patch(color=cores[None], label="sem rede")] + [Patch(color=cores[d], label=nomes[d])
                                                           for d in ("bruto", "viterbi", "viterbi_pos")]
    fig.legend(handles=alcas, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "fig_ablacao")


def fig_series_dev():
    ref = np.load(E1 / "referencia.npz")
    ks, cols, yref = ref["k"], ref["cols"], ref["y"]
    c = np.load(E1 / "classico.npz")
    series = [
        ("referência de desenvolvimento", yref.mean(1), "black", "o", 2.0),
        ("clássico", c["y_inf"][ks][:, cols].mean(1), "#555555", "s", 1.2),
        ("R1, rótulos atrasados", np.load(E2 / "runs/R1_s0/saida.npz")["bruto"][ks][:, cols].mean(1), "#d08c2f", "v", 1.2),
        ("R2, rótulos corrigidos, 16 quadros", np.load(E2 / "runs/R2_s0/saida.npz")["bruto"][ks][:, cols].mean(1), "#3aa17e", "^", 1.2),
        ("R4 com decodificação temporal", np.load(E2 / "runs/R4_s0/saida.npz")["viterbi"][ks][:, cols].mean(1), "#1f4e9c", "D", 1.4),
    ]
    tt = t_de_k(ks)
    fig, ax = plt.subplots(figsize=(LARG, 3.0))
    for nome, s, cor, mk, lw in series:
        ax.plot(tt, s * ESC, color=cor, marker=mk, ms=3.5, lw=lw, label=nome)
    ax.set_xlabel(ROT_TEMPO)
    ax.set_ylabel("posição média da face inferior (mm)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    virgula(ax)
    salvar(fig, "fig_series_dev")


def fig_folha_visual():
    c = np.load(E1 / "classico.npz")
    r4 = np.load(E2 / "runs/R4_s0/saida.npz")["viterbi"]
    pF = E3 / "runs/F_d0/saida.npz"
    fF = np.load(pF)["viterbi"] if pF.exists() else None
    mk = marcas_autor("baixo")
    ks = (180, 300, 480, 720)
    fig, axs = plt.subplots(len(ks), 1, figsize=(LARG, 7.4))
    for ax, k in zip(axs, ks):
        im = roi(k)
        ya, yb = 250, 593
        ax.imshow(im[ya:yb], extent=(0, im.shape[1], yb, ya))
        linha(ax, XS, c["y_inf"][k], VIOLETA, lw=1.2)
        linha(ax, XS, r4[k], AZUL, lw=1.4)
        if fF is not None:
            linha(ax, XS, fF[k], CIANO, lw=1.4)
        perto = [v for kk, v in mk.items() if abs(kk - k) <= 3]
        if perto:
            cruzes(ax, np.vstack(perto))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{t_de_k(k):.0f} s após a ignição", loc="left")
    axs[0].plot([], [], color=VIOLETA, lw=1.4, label="clássico")
    axs[0].plot([], [], color=AZUL, lw=1.4, label="R4")
    axs[0].plot([], [], color=CIANO, lw=1.4, label="F, ajuste fino com marcações do autor")
    axs[0].scatter([], [], marker="o", facecolors="none", edgecolors=VERDE, label="marcações do autor")
    axs[0].legend(loc="lower right", frameon=True, facecolor="#222222", edgecolor="none",
                  labelcolor="white", fontsize=7, ncol=2)
    fig.tight_layout(h_pad=0.4)
    salvar(fig, "fig_folha_visual")


def _marcas_sessao(face: str) -> dict[int, np.ndarray]:
    sys.path.insert(0, str(RAIZ / "codigo"))
    from experimento_s_sessao import marcas
    return marcas(face)


def fig_folha_teste():
    """Quatro quadros dos blocos de teste, com as duas faces: marcações do autor
    (só da sessão) e as linhas do clássico, da R4 (só face inferior), da S1 e da S2."""
    c = np.load(E1 / "classico.npz")
    r4 = np.load(E2 / "runs/R4_s0/saida.npz")["viterbi"]
    s = {v: {f: np.load(E4 / f"runs/{v}_{f}_s0/saida.npz")["viterbi"] for f in ("baixo", "cima")}
         for v in ("S1", "S2")}
    mb, mc = _marcas_sessao("baixo"), _marcas_sessao("cima")
    ks = (324, 360, 707, 750)
    fig, axs = plt.subplots(len(ks), 1, figsize=(LARG, 8.8))
    for ax, k in zip(axs, ks):
        im = roi(k)
        ya, yb = 40, 593
        ax.imshow(im[ya:yb], extent=(0, im.shape[1], yb, ya))
        for yc in (c["y_inf"], c["y_sup"]):
            linha(ax, XS, yc[k], VIOLETA, lw=1.0)
        linha(ax, XS, r4[k], AZUL, lw=1.3)
        for f in ("baixo", "cima"):
            linha(ax, XS, s["S2"][f][k], MAGENTA, lw=1.3)
            linha(ax, XS, s["S1"][f][k], CIANO, lw=1.3)
        for m in (mb, mc):
            if k in m:
                cruzes(ax, m[k], s=7)
        ax.set_xlim(0, im.shape[1]); ax.set_ylim(yb, ya)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{'B1' if k < 500 else 'B2'}, {t_de_k(k):.0f} s após a ignição", loc="left")
    axs[0].plot([], [], color=VIOLETA, lw=1.4, label="clássico")
    axs[0].plot([], [], color=AZUL, lw=1.4, label="R4, só face inferior")
    axs[0].plot([], [], color=CIANO, lw=1.4, label="S1, ImageNet e sessão")
    axs[0].plot([], [], color=MAGENTA, lw=1.4, label="S2, ajuste fino e sessão")
    axs[0].scatter([], [], marker="o", facecolors="none", edgecolors=VERDE, label="marcações do autor")
    axs[0].legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none",
                  labelcolor="white", fontsize=6.5, ncol=3)
    fig.tight_layout(h_pad=0.4)
    salvar(fig, "fig_folha_teste")


TODAS = dict(ensaio=fig_ensaio, transiente=fig_transiente, fuligem=fig_fuligem,
             quimograma=fig_quimograma, ablacao=fig_ablacao, series=fig_series_dev,
             folha=fig_folha_visual, folha_teste=fig_folha_teste)

if __name__ == "__main__":
    pedidas = sys.argv[1:] or list(TODAS)
    for n in pedidas:
        TODAS[n]()
