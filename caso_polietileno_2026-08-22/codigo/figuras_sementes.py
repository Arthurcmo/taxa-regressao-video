"""Conferência de cada semente contra as marcações do autor (08/11/2026).

Para cada semente (0, 1 e 2), as três linhas aceitas da mesma semente: S1 (ciano),
2b (azul) e forma (magenta), com as marcações do autor em círculos verdes.

  figura   4 quadros no padrão do TG: B1 (93 s, teste da S1), 135 s, 180 s (fim da
           análise) e B2 (230 s, teste da S1), tempo real desde a ignição;
  folha    todos os quadros marcados da janela da janela de análise e os de B2, para
           conferência a olho;
  erros    erro contra as marcações por semente:
             teste    B1 e B2, nos pontos clicados, colunas centrais (como a Tab. 4.1);
                      só a S1 nunca viu esses quadros;
             janela   quadros marcados da janela de análise, alvo por coluna (como o
                      critério de aceitação): erro médio e % a até 4 px.
           mais a taxa de cada semente lida de resultados_tg2.json.

Saída: experimento_recompensa_lei_2026-11-06/sementes/

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/figuras_sementes.py
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import figuras_tg2 as F
import recompensa_lei as R
from experimento_s_sessao import DT, K_IGN, XC, divisao, marcas

OUT = R.OUT / "sementes"
RUNS = {0: ("nenhum_d4", "oc_forma_w5_s0"), 1: ("nenhum_d4_s1", "p3_forma_w5_s1"), 2: ("nenhum_d4_s2", "p3_forma_w5_s2")}
CORTE, FIM = 92.77834766221454, 187.57834766221353
CENTRAIS = (370.0, 1830.0)
KS_FIG = (324, 450, 585, 735)
ROTULO = {324: "bloco de teste B1", 450: "meio da janela de análise", 585: "fim da janela de análise",
          735: "bloco de teste B2, fora da janela de análise"}
NOMES = ("S1", "2b", "forma")
CORES = {"S1": F.CIANO, "2b": F.AZUL, "forma": F.MAGENTA}


def t(k):
    return (k - K_IGN) * DT


def linhas(s: int) -> dict[str, dict[str, np.ndarray]]:
    b2, fo = RUNS[s]
    L = {"S1": {f: np.load(R.S1_RUNS / f"S1_{f}_s{s}/saida.npz")["bruto"].astype(float) for f in ("baixo", "cima")}}
    for nome, run in (("2b", b2), ("forma", fo)):
        z = np.load(R.OUT / f"runs/{run}/saida.npz")
        L[nome] = {f: z[f].astype(float) for f in ("baixo", "cima")}
    return L


def pontos(face: str, k: int) -> np.ndarray:
    """Pontos clicados no quadro k: sessão de 28/10 e camada B de 05/11."""
    p = [m[k] for m in (marcas(face), R.marcas_b(face)) if k in m]
    return np.vstack(p) if p else np.zeros((0, 2))


# ----------------------------------------------------------------------------- erros
def erros(L) -> dict:
    div = divisao()
    res = {}
    for nome, d in L.items():
        r = {}
        for bloco in ("B1", "B2"):
            for f in ("baixo", "cima"):
                mk = marcas(f)
                e = []
                for k in div[bloco]:
                    if k in mk and np.isfinite(d[f][k]).all():
                        p = mk[k]
                        sel = (p[:, 0] >= CENTRAIS[0]) & (p[:, 0] <= CENTRAIS[1])
                        e.append(np.interp(p[sel, 0], XC, d[f][k]) - p[sel, 1])
                e = np.concatenate(e)
                r[f"{bloco} {f}"] = dict(mae=float(np.abs(e).mean()), p95=float(np.percentile(np.abs(e), 95)),
                                         vies=float(e.mean()), ate4=float(np.mean(np.abs(e) <= 4)), n=int(e.size))
        for f in ("baixo", "cima"):
            A = R.autor(f)
            ks = [k for k in range(K_IGN, 963) if CORTE <= t(k) <= FIM and np.isfinite(A[k]).sum() > 10]
            e = np.concatenate([(d[f][k] - A[k])[np.isfinite(A[k]) & np.isfinite(d[f][k])] for k in ks])
            r[f"janela {f}"] = dict(mae=float(np.abs(e).mean()), p95=float(np.percentile(np.abs(e), 95)),
                                    vies=float(e.mean()), ate4=float(np.mean(np.abs(e) <= 4)), n=int(e.size),
                                    quadros=len(ks))
        res[nome] = r
    return res


# ----------------------------------------------------------------------------- figura do TG
def figura(s: int, L) -> None:
    fig, axs = plt.subplots(len(KS_FIG), 1, figsize=(F.LARG, 8.9))
    for i, (ax, k) in enumerate(zip(axs, KS_FIG)):
        im = F.roi(k)
        ya, yb = 0, im.shape[0]
        ax.imshow(im[ya:yb], extent=(0, im.shape[1], yb, ya))
        for nome in NOMES:
            for f in ("baixo", "cima"):
                F.linha(ax, XC, L[nome][f][k], CORES[nome], lw=1.1)
        for f in ("baixo", "cima"):
            p = pontos(f, k)
            if len(p):
                F.cruzes(ax, p, s=7)
        ax.set_xlim(0, im.shape[1]); ax.set_ylim(yb, ya)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'abcd'[i]}) {F.br(t(k), 0)} s após a ignição, {ROTULO[k]}", loc="left")
    for nome, rot in (("S1", "S1"), ("2b", "2b"), ("forma", "forma")):
        axs[0].plot([], [], color=CORES[nome], lw=1.4, label=f"{rot}, semente {s}")
    axs[0].scatter([], [], marker="o", facecolors="none", edgecolors=F.VERDE, label="marcações do autor")
    axs[0].legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none",
                  labelcolor="white", fontsize=6.5, ncol=4)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(OUT / f"fig_semente_{s}.jpg", pil_kwargs={"quality": 90}, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- folha de conferência
BGR = {"S1": (255, 229, 0), "2b": (255, 107, 47), "forma": (255, 92, 255)}
VERDE = (0, 255, 106)


def folha(s: int, L) -> None:
    div = divisao()
    A = {f: R.autor(f) for f in ("baixo", "cima")}
    ks = [k for k in range(K_IGN, 963) if CORTE <= t(k) <= FIM
          and any(np.isfinite(A[f][k]).sum() > 10 for f in A)] + div["B2"]
    ims = []
    for k in ks:
        im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR).copy()
        for nome in NOMES:
            for f in ("baixo", "cima"):
                p = np.stack([XC, L[nome][f][k]], 1)
                p = p[np.isfinite(p[:, 1])].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(im, [p], False, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.polylines(im, [p], False, BGR[nome], 2, cv2.LINE_AA)
        for f in ("baixo", "cima"):
            for x, y in pontos(f, k):
                cv2.circle(im, (int(x), int(y)), 6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(im, (int(x), int(y)), 6, VERDE, 1, cv2.LINE_AA)
        im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        bloco = " (teste B1)" if k in div["B1"] else (" (teste B2)" if k in div["B2"] else "")
        barra = np.zeros((28, im.shape[1], 3), np.uint8)
        txt = f"semente {s} | amostra {k + 1} | {t(k):.1f} s desde a ignicao{bloco}"
        cv2.putText(barra, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ims.append(np.vstack([barra, im]))
    leg = np.zeros((34, ims[0].shape[1] * 2 + 6, 3), np.uint8)
    x = 8
    for nome in NOMES:
        cv2.line(leg, (x, 17), (x + 40, 17), BGR[nome], 3)
        cv2.putText(leg, nome, (x + 48, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 160
    cv2.circle(leg, (x + 10, 17), 6, VERDE, 1, cv2.LINE_AA)
    cv2.putText(leg, "marcacoes do autor", (x + 26, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if len(ims) % 2:
        ims.append(np.zeros_like(ims[0]))
    esp = np.full((ims[0].shape[0], 6, 3), 255, np.uint8)
    linhas_ = [np.hstack([ims[i], esp, ims[i + 1]]) for i in range(0, len(ims), 2)]
    sep = np.full((6, linhas_[0].shape[1], 3), 255, np.uint8)
    out = np.vstack([leg] + sum([[l_, sep] for l_ in linhas_], [])[:-1])
    cv2.imwrite(str(OUT / f"folha_semente_{s}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------- tabela
def taxa(s: int) -> dict:
    d = json.loads((R.OUT / "resultados_tg2/resultados_tg2.json").read_text())["linhas"]
    out = {}
    for nome in NOMES:
        l = d[f"{nome} s{s}"]
        g, m = l["grão todo"], l["meio 40–80"]
        out[nome] = dict(ensaio_r=l["ensaio"]["r_real"], ensaio_q=l["ensaio"]["q_real"], ensaio_qcam=l["ensaio"]["q_cam"],
                         jan_a=g["janelas"]["a"], jan_n=g["janelas"]["n"], jan_r2=g["janelas"]["r2"],
                         jan_q=g["janelas"]["razao"], multi_q=g["multiverso"]["razao"], multi_r=g["multiverso"]["pearson"],
                         meio_a=m["janelas"]["a"], meio_n=m["janelas"]["n"], meio_r2=m["janelas"]["r2"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pedidas = set(sys.argv[1:]) or {"figura", "folha", "erros"}
    res = {}
    for s in (0, 1, 2):
        L = linhas(s)
        if "figura" in pedidas:
            figura(s, L); print("figura", s, flush=True)
        if "folha" in pedidas:
            folha(s, L); print("folha", s, flush=True)
        if "erros" in pedidas:
            res[s] = dict(erros=erros(L), taxa=taxa(s))
    if res:
        (OUT / "sementes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("\nteste (pontos clicados, colunas centrais): erro médio (P95) px | % a até 4 px")
        for s, r in res.items():
            for nome in NOMES:
                e = r["erros"][nome]
                print(f"s{s} {nome:6s}" + "".join(f" {c}: {e[c]['mae']:.1f} ({e[c]['p95']:.1f}) {e[c]['ate4'] * 100:.0f}% |"
                                                    for c in ("B1 baixo", "B1 cima", "B2 baixo", "B2 cima")))
        print("\njanela janela de análise (alvo por coluna): erro médio, viés e % a até 4 px")
        for s, r in res.items():
            for nome in NOMES:
                e = r["erros"][nome]
                print(f"s{s} {nome:6s}" + "".join(f" {c}: {e[c]['mae']:.1f} px, viés {e[c]['vies']:+.1f}, {e[c]['ate4'] * 100:.0f}% "
                                                    f"({e[c]['quadros']} q) |" for c in ("janela baixo", "janela cima")))
        print("\ntaxa por semente (tempo real)")
        for s, r in res.items():
            for nome in NOMES:
                q = r["taxa"][nome]
                print(f"s{s} {nome:6s} ensaio {q['ensaio_r']:.4f} mm/s {q['ensaio_q']:.2f}× ({q['ensaio_qcam']:.2f}× câmera) | "
                      f"janelas grão {q['jan_a']:.4f} G^{q['jan_n']:.2f} R² {q['jan_r2']:.2f} {q['jan_q']:.2f}× | "
                      f"54 escolhas {q['multi_q']:.2f}× r {q['multi_r']:.2f} | meio {q['meio_a']:.4f} G^{q['meio_n']:.2f} R² {q['meio_r2']:.2f}")


if __name__ == "__main__":
    main()
