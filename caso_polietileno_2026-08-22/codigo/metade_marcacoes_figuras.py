"""Figuras e relatório do experimento de metade das marcações (16/11/2026).

Lê experimento_metade_marcacoes_2026-11-16/resultado.json (metade_marcacoes.py avaliar).
  erro_por_fracao.png       erro nos quadros NÃO vistos contra a fração marcada
  quadros_nao_vistos_*.jpg  linhas sobre quadros que a rede não viu (validação visual)
  RELATORIO.md

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/metade_marcacoes_figuras.py
"""
from __future__ import annotations

import ensaio as ENS
import json

import matplotlib.pyplot as plt
import numpy as np

import experimento_s_sessao as E
import figuras_sementes as FS
import figuras_tg2 as F
import metade_marcacoes as M

OUT = M.OUTM
D = json.loads((OUT / "resultado.json").read_text())
DIV = D["divisao"]
SAIDA = "bruto"            # a saída que entra na taxa, como na 2b e na S1 (taxa)


def media_grupo(g, f, rg, chave="media", saida=SAIDA, parte="teste"):
    v = [D["grupos"][g][str(s)][f"{f}|{saida}|{rg}|{parte}"][chave] for s in M.SEMENTES]
    return float(np.mean(v)), float(min(v)), float(max(v))


def media_ref(nome, g, f, rg, chave="media"):
    d = D["referencias"][nome]
    v = [d[k][chave] for k in d if k.startswith(f"{g}|") and k.endswith(f"|{f}|{rg}")]
    return float(np.mean(v)), float(min(v)), float(max(v))


def tabela() -> str:
    L = []
    for rg in ("grão todo", "40 a 120 mm"):
        L.append(f"\n### {rg}, quadros que a rede NÃO viu (média das 3 sementes; faixa entre sementes)\n")
        L.append("| linha | quadros marcados usados (baixo / cima) | face | erro médio (px) | mediana (px) | a até 4 px | viés (px, + = chama) |")
        L.append("|---|---|---|---|---|---|---|")
        for f in E.FACES:
            for g, rot in (("Q", "um quarto"), ("A", "metade A"), ("B", "metade B")):
                n = f"{len(DIV[g]['baixo']['treino'])} / {len(DIV[g]['cima']['treino'])}"
                m, a, b = media_grupo(g, f, rg)
                md = media_grupo(g, f, rg, "mediana")[0]
                c4 = media_grupo(g, f, rg, "ate4")[0]
                vi = media_grupo(g, f, rg, "vies")[0]
                L.append(f"| rede, {rot} | {n} | {f} | {m:.2f} ({a:.2f}–{b:.2f}) | {md:.2f} | {c4*100:.0f} % | {vi:+.2f} |")
            for nome in ("clássico (sem treino)", "SM1 v4 (nenhuma marcação)"):
                m, a, b = media_ref(nome, "A", f, rg)
                md = media_ref(nome, "A", f, rg, "mediana")[0]
                c4 = media_ref(nome, "A", f, rg, "ate4")[0]
                vi = media_ref(nome, "A", f, rg, "vies")[0]
                L.append(f"| {nome} | 0 | {f} | {m:.2f} | {md:.2f} | {c4*100:.0f} % | {vi:+.2f} |")
    return "\n".join(L)


def tabela_treino() -> str:
    L = ["| grupo | face | erro nos quadros vistos (px) | erro nos NÃO vistos (px) |", "|---|---|---|---|"]
    for g in ("Q", "A", "B"):
        for f in E.FACES:
            L.append(f"| {g} | {f} | {media_grupo(g, f, 'grão todo', parte='treino')[0]:.2f} | "
                     f"{media_grupo(g, f, 'grão todo')[0]:.2f} |")
    return "\n".join(L)


def tabela_taxa() -> str:
    import taxa_real_cap4 as TR
    import avaliar_linhas as AL
    man = D["marcacoes_taxa"]["analise"]["r"]
    L = [f"Marcações manuais (todas): {man:.4f} mm/s na janela de análise.\n",
         "| grupo | quadro a quadro, sementes 0 / 1 / 2 (mm/s) | média | razão | decodificada, sementes 0 / 1 / 2 | média | razão |",
         "|---|---|---|---|---|---|---|"]
    for g in ("Q", "A", "B"):
        v = [D["grupos"][g][str(s)]["taxa"]["analise"]["r"] for s in M.SEMENTES]
        w = []
        for s in M.SEMENTES:
            Z = {f: np.load(OUT / f"runs/{g}_{f}_s{s}/saida.npz")["viterbi"].astype(float) for f in E.FACES}
            w.append(TR.medir(Z["baixo"], Z["cima"], ns=(10,))["analise"]["r"])
        L.append(f"| {g} | {' / '.join(f'{x:.4f}' for x in v)} | {np.mean(v):.4f} | {np.mean(v)/man:.3f} | "
                 f"{' / '.join(f'{x:.4f}' for x in w)} | {np.mean(w):.4f} | {np.mean(w)/man:.3f} |")
    sm = [TR.medir(*AL.run(f"sm1v4_s{s}"), ns=(10,))["analise"]["r"] for s in M.SEMENTES]
    L.append(f"| SM1 v4 (nenhuma marcação) | {' / '.join(f'{x:.4f}' for x in sm)} | {np.mean(sm):.4f} | {np.mean(sm)/man:.3f} | – | – | – |")
    return "\n".join(L)


_J0, _J1 = ENS.JANELA_ANALISE_S
PERIODOS = {"começo, antes da janela de análise": (0, _J0), "janela de análise": (_J0, _J1),
            "fim, depois da janela de análise": (_J1, 999)}
REGS = {"grão todo": None, "40 a 120 mm": (40, 120)}


def _erros(Y, A, ks, cols):
    e = [(Y[k, cols] - A[k, cols])[np.isfinite(A[k, cols]) & np.isfinite(Y[k, cols])] for k in ks]
    e = np.concatenate(e) if e else np.zeros(0)
    return e


def por_periodo() -> tuple[str, dict]:
    """Erro nos quadros não vistos, por trecho da queima. As metades A e B juntas testam
    cada quadro marcado uma vez (cada um pela rede que não o viu)."""
    import faixas_grao as FG
    import avaliar_linhas as AL
    import experimento_p2_v2 as P2
    A = {f: M.marcas_col(f) for f in E.FACES}
    cl = np.load(P2.E1 / "classico.npz")
    clas = dict(baixo=cl["y_inf"].astype(float), cima=cl["y_sup"].astype(float))
    sm1 = [dict(zip(E.FACES, AL.run(f"sm1v4_s{s}"))) for s in M.SEMENTES]
    tab, out = [], {}
    for rg, faixa in REGS.items():
        cols = np.arange(93) if faixa is None else FG.faixa_cols(*faixa)
        for per, (t0, t1) in PERIODOS.items():
            tab.append(f"\n#### {rg}, {per}\n")
            tab.append("| linha | face | erro médio, quadro a quadro (px) | mediana (px) | a até 4 px | erro médio, decodificada (px) |")
            tab.append("|---|---|---|---|---|---|")
            for f in E.FACES:
                linhas = {}
                for nome, grupos in (("rede com metade dos quadros (A e B)", ("A", "B")), ("rede com 1/4 dos quadros", ("Q",))):
                    eb, ev = [], []
                    for g in grupos:
                        ks = [k for k in DIV[g][f]["teste"] if t0 <= FS.t(k) < t1]
                        for s in M.SEMENTES:
                            z = np.load(OUT / f"runs/{g}_{f}_s{s}/saida.npz")
                            eb.append(_erros(z["bruto"].astype(float), A[f], ks, cols))
                            ev.append(_erros(z["viterbi"].astype(float), A[f], ks, cols))
                    linhas[nome] = (np.concatenate(eb), np.concatenate(ev))
                ks_all = [k for k in sorted(set(DIV["A"][f]["teste"]) | set(DIV["B"][f]["teste"])) if t0 <= FS.t(k) < t1]
                linhas["SM1 v4 (nenhuma marcação)"] = (np.concatenate([_erros(d[f], A[f], ks_all, cols) for d in sm1]), None)
                linhas["clássico (sem treino)"] = (_erros(clas[f], A[f], ks_all, cols), None)
                for nome, (eb, ev) in linhas.items():
                    if eb.size == 0:
                        continue
                    out[(rg, per, f, nome)] = float(np.abs(eb).mean())
                    dv = f"{np.abs(ev).mean():.2f}" if ev is not None and ev.size else "–"
                    tab.append(f"| {nome} | {f} | {np.abs(eb).mean():.2f} | {np.median(np.abs(eb)):.2f} | "
                               f"{np.mean(np.abs(eb) <= 4) * 100:.0f} % | {dv} |")
    return "\n".join(tab), out


def fig_erro(out):
    """Erro nos quadros não vistos, na janela de análise, de 40 a 120 mm."""
    per, rg = "janela de análise", "40 a 120 mm"
    fig, axs = plt.subplots(1, 2, figsize=(F.LARG, 3.0), sharey=True)
    for ax, f in zip(axs, E.FACES):
        ax.axhspan(1.0, 2.0, color="0.85", lw=0, label="piso da marcação manual, 1 a 2 px")
        pts = [("clássico (sem treino)", 0, "#9b59ff", "D"), ("SM1 v4 (nenhuma marcação)", 0, F.MAGENTA, "^"),
               ("rede com 1/4 dos quadros", len(DIV["Q"][f]["treino"]), "#2ca02c", "o"),
               ("rede com metade dos quadros (A e B)", len(DIV["A"][f]["treino"]), F.AZUL, "s")]
        for nome, x, cor, mk in pts:
            ax.plot([x], [out[(rg, per, f, nome)]], mk, color=cor, mec="black", mew=0.5, ms=7, label=nome, ls="none")
        ax.set_title(f"face {'inferior' if f == 'baixo' else 'superior'}", loc="left", fontsize=9)
        ax.set_xlabel("quadros marcados usados no treino", fontsize=8)
        ax.set_xlim(-2, 22); ax.grid(alpha=0.25); ax.tick_params(labelsize=7.5)
    axs[0].set_yscale("log"); axs[0].set_yticks([1, 2, 3, 5, 10]); axs[0].yaxis.set_major_formatter(F.VIRG)
    axs[0].set_ylabel("erro médio nos quadros\nnão vistos (px)", fontsize=8)
    for ax in axs:
        ax.xaxis.set_major_formatter(F.VIRG)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.6)
    fig.suptitle("Janela de análise, de 40 a 120 mm", fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(OUT / "erro_por_fracao.png", dpi=200); plt.close(fig)


def fig_quadros():
    """Quadros que a rede da metade A NÃO viu, nas duas faces, com a linha da metade A
    (semente 0, azul tracejado) e a do quarto Q (semente 0, verde-escuro tracejado)."""
    te = sorted(set(DIV["A"]["baixo"]["teste"]) & set(DIV["A"]["cima"]["teste"]))
    ks = [te[i] for i in np.linspace(0, len(te) - 1, min(8, len(te))).round().astype(int)]
    za = {f: np.load(OUT / f"runs/A_{f}_s0/saida.npz")[SAIDA].astype(float) for f in E.FACES}
    zq = {f: np.load(OUT / f"runs/Q_{f}_s0/saida.npz")[SAIDA].astype(float) for f in E.FACES}
    for parte, grupo in enumerate((ks[:4], ks[4:]), 1):
        if not grupo:
            continue
        fig, axs = plt.subplots(len(grupo), 1, figsize=(F.LARG, 2.2 * len(grupo)))
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, grupo):
            im = F.roi(k)
            ax.imshow(im, extent=(0, im.shape[1], im.shape[0], 0))
            for f in E.FACES:
                F.linha(ax, E.XC, zq[f][k], "#00a000", lw=1.0, ls=(0, (5, 2.5)))
                F.linha(ax, E.XC, za[f][k], F.AZUL, lw=1.0, ls=(0, (5, 2.5)))
                p = E.marcas(f).get(k)
                if p is not None and len(p):
                    F.cruzes(ax, p, s=7)
            ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            qv = "também não visto pelo 1/4" if k not in DIV["Q"]["baixo"]["treino"] else "visto pelo 1/4"
            ax.set_title(f"{F.br(FS.t(k), 0)} s após a ignição: quadro NÃO visto pela metade A ({qv})", loc="left", fontsize=8)
        axs[0].plot([], [], color=F.AZUL, ls=(0, (5, 2.5)), label="rede com metade dos quadros (A)")
        axs[0].plot([], [], color="#00a000", ls=(0, (5, 2.5)), label="rede com 1/4 dos quadros")
        axs[0].scatter([], [], marker="o", facecolors="none", edgecolors=F.VERDE, label="marcações manuais (não usadas)")
        axs[0].legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none", labelcolor="white",
                      fontsize=6.3, ncol=3)
        fig.tight_layout(h_pad=0.4)
        fig.savefig(OUT / f"quadros_nao_vistos_{parte}.jpg", dpi=220, pil_kwargs={"quality": 90}, bbox_inches="tight")
        plt.close(fig)
    return ks


def main():
    tab_per, out = por_periodo()
    fig_erro(out)
    ks = fig_quadros()
    rel = ["# Rede treinada com parte das marcações (16/11/2026)\n",
           "Mesma receita da S1 (ResNet18 da ImageNet, 1200 passos, uma rede por face, 3 sementes), "
           "mas cada rede vê só parte dos quadros marcados (k ≥ 179, todos os da sessão de 21/10), "
           "e o erro é medido nos quadros que ela NÃO viu, espalhados pela queima inteira. "
           "Saída quadro a quadro (a que entra na taxa).\n",
           "- Metade A: quadros marcados de índice par; teste nos ímpares. Metade B: o contrário.",
           "- Um quarto (Q): índices 0, 4, 8...; teste nos outros 3/4.",
           "- Referências nos mesmos quadros de teste da metade A: clássico e SM1 v4 (nenhuma marcação).",
           "- Piso da referência (TG, Cap. 3): 1 a 2 px de erro médio; mediana da dispersão das marcações 1,0 a 1,1 px. A S1 nos blocos de teste: 1,6 e 2,1 px.\n",
           "## Por trecho da queima (o resultado que vale)\n", tab_per,
           "\n## Queima inteira (inclui o começo e o fim, onde o erro é grande)\n",
           tabela(), "\n## Vistos × não vistos (grão todo)\n", tabela_treino(),
           "\n## Taxa na janela de análise (grão todo)\n", tabela_taxa(),
           f"\n## Conferência visual\n\nQuadros não vistos pela metade A: {', '.join(f'{FS.t(k):.0f} s' for k in ks)}. "
           "Ver quadros_nao_vistos_1.jpg e _2.jpg.\n"]
    (OUT / "RELATORIO.md").write_text("\n".join(rel))
    print("\n".join(rel))


if __name__ == "__main__":
    main()
