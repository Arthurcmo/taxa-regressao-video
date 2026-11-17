"""Todas as linhas na mesma régua.

Linhas (famílias, cada uma com as suas sementes):
  marcações do autor        referência (só para a taxa e a altura)
  clássico                  detector de vale de três passadas
  S1                        ImageNet + sessão de marcação; posição pela saída decodificada,
                            taxa pela saída quadro a quadro (como no texto)
  2b, forma                 ajustes finos com todas as marcações
  SM1 v4, SMF v4            sem nenhuma marcação

Para cada linha e semente:
  posição   erro absoluto médio e % a até 4 px contra as marcações do autor
            - blocos de teste B1 e B2, pontos marcados, colunas centrais;
            - janela de análise, 40 a 120 mm, alvo por coluna;
  taxa      no tempo real:
            - ensaio inteiro (amostra 79 ao fim da janela de análise), no grão todo e de
              40 a 120 mm;
            - correlação e expoente, mediana das 54 escolhas de janela;
            - lei com 10 janelas no meio (40 a 80 mm);
            - expoente nos degraus de vazão (final e 2ª metade), mediana;
  altura    altura média da porta de 40 a 120 mm, quadro a quadro.

Saída: experimento_recompensa_lei_2026-11-06/comparacao_geral/comparacao_geral.json

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/comparacao_geral.py
"""
from __future__ import annotations

import json
import ensaio as E

import numpy as np

import avaliar_linhas as AL
import conferencia_sem_marcacao as CS
import experimento_p2_v2 as P2
import faixas_grao as FG
import figuras_sementes as FS
import recompensa_lei as R
import resultados_tg2 as T
import secoes_e_degraus as SD
from experimento_s_sessao import XC, divisao, marcas

OUT = R.OUT / "comparacao_geral"
S = E.FATOR_TEMPO_REAL
SENT = {"baixo": -1, "cima": +1}
DEG = [(195.0, (172.0, 186.0), (197.0, 211.0)), (214.0, (200.0, 212.0), (217.0, 237.0)),
       (240.5, (225.0, 239.0), (243.0, 260.0))]
SEC_DEG = {"2ª metade 60–120": (60, 120), "final 80–120": (80, 120)}
SEC_TAXA = {"grão todo": (10, 120), "40 a 120 mm": (40, 120)}


def s1(s, chave):
    return tuple(np.load(R.S1_RUNS / f"S1_{f}_s{s}/saida.npz")[chave].astype(float) for f in ("baixo", "cima"))


def familias():
    cl = np.load(P2.E1 / "classico.npz")
    F = {"clássico": {0: dict(pos=(cl["y_inf"].astype(float), cl["y_sup"].astype(float)))}}
    F["clássico"][0]["taxa"] = F["clássico"][0]["pos"]
    F["S1"] = {s: dict(pos=s1(s, "viterbi"), taxa=s1(s, "bruto")) for s in (0, 1, 2)}
    runs = {"2b": ("nenhum_d4", "nenhum_d4_s1", "nenhum_d4_s2"),
            "forma": ("oc_forma_w5_s0", "p3_forma_w5_s1", "p3_forma_w5_s2"),
            "SM1 v4": ("sm1v4_s0", "sm1v4_s1", "sm1v4_s2"),
            "SMF v4": ("smfv4_s0", "smfv4_s1", "smfv4_s2")}
    for nome, rs in runs.items():
        F[nome] = {}
        for s, r in enumerate(rs):
            Y = AL.run(r)
            F[nome][s] = dict(pos=Y, taxa=Y)
    return F


def erros(Yb, Yc):
    d = {"baixo": Yb, "cima": Yc}
    div = divisao()
    out = {}
    for face in ("baixo", "cima"):
        A = R.autor(face)
        ks = [k for k in range(963) if FS.CORTE <= FS.t(k) <= FS.FIM and np.isfinite(A[k]).sum() > 10]
        cols = FG.faixa_cols(40, 120)
        e = np.concatenate([(d[face][k, cols] - A[k, cols])[np.isfinite(A[k, cols]) & np.isfinite(d[face][k, cols])] for k in ks])
        out[f"janela {face}"] = [float(np.abs(e).mean()), float(np.mean(SENT[face] * e)), float(np.mean(np.abs(e) <= 4))]
        mk = marcas(face)
        tudo = []
        for bloco in ("B1", "B2"):
            eb = []
            for k in div[bloco]:
                if k in mk and np.isfinite(d[face][k]).all():
                    p = mk[k]; sel = (p[:, 0] >= FS.CENTRAIS[0]) & (p[:, 0] <= FS.CENTRAIS[1])
                    eb.append(np.interp(p[sel, 0], XC, d[face][k]) - p[sel, 1])
            eb = np.concatenate(eb)
            tudo.append(eb)
            out[f"{bloco} {face}"] = [float(np.abs(eb).mean()), float(np.mean(SENT[face] * eb)), float(np.mean(np.abs(eb) <= 4))]
        e = np.concatenate(tudo)
        out[f"teste {face}"] = [float(np.abs(e).mean()), float(np.mean(SENT[face] * e)), float(np.mean(np.abs(e) <= 4))]
    return out


def serie(Yb, Yc, x0, x1):
    """Altura média da seção; nas marcações, o primeiro valor é repetido uma amostra antes
    (o ensaio começa exatamente no primeiro quadro marcado, amostra 79)."""
    H = AL.serie_H(Yb, Yc, x0, x1)
    i = np.flatnonzero(np.isfinite(H))
    if len(i) and i[0] > 0:
        H[i[0] - 1] = H[i[0]]
    return H


def taxas(Yb, Yc, borda):
    out = {}
    for sec, (x0, x1) in SEC_TAXA.items():
        H = serie(Yb, Yc, x0, x1)
        d = {}
        try:
            p = T.um_ponto(H, 78, E.AMOSTRA_IGNICAO + T.FIM / (T.C.CAM * T._tm.s))
            d["ensaio"] = dict(r=p["r_real"], G=p["G"])
        except Exception:
            d["ensaio"] = None
        var = [AL.metricas(AL.janelas(H, T._corte + t0, dur)) for t0, dur in AL.ESCOLHAS]
        d["janelas"] = dict(pearson=float(np.nanmedian([v["pearson"] for v in var])),
                            n=float(np.nanmedian([v["n"] for v in var])))
        out[sec] = d
    r, G, _ = CS.pontos(AL.serie_H(Yb, Yc, 40, 80))
    out["meio"] = dict(r=r.tolist(), G=G.tolist(), **CS.ajuste(G, r))
    nd = []
    for sec, (x0, x1) in SEC_DEG.items():
        H = AL.serie_H(Yb, Yc, x0, x1)
        for x, (a1, b1), (a2, b2) in DEG:
            if b2 > borda[sec]:
                continue
            p1, p2 = SD.taxa_trecho(H, a1, b1), SD.taxa_trecho(H, a2, b2)
            if p1 and p2 and p1["r"] > 0 and p2["r"] > 0:
                nd.append(float(np.log(p2["r"] / p1["r"]) / np.log(p2["G"] / p1["G"])))
    out["degraus"] = dict(n=nd, mediana=float(np.median(nd)) if nd else None)
    out["altura_40_120"] = AL.serie_H(Yb, Yc, 40, 120).tolist()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    borda = SD.borda(AL.run("nenhum_d4")[0])
    res = {"marcações do autor": {0: dict(taxa=taxas(*AL.manual(), borda))}}
    for nome, sem in familias().items():
        res[nome] = {}
        for s, d in sem.items():
            res[nome][s] = dict(erros=erros(*d["pos"]), taxa=taxas(*d["taxa"], borda))
            t = res[nome][s]["taxa"]; e = res[nome][s]["erros"]
            print(f"{nome:8s} s{s} teste inf {e['teste baixo'][0]:.1f} sup {e['teste cima'][0]:.1f} | janela inf "
                  f"{e['janela baixo'][0]:.1f} sup {e['janela cima'][0]:.1f} | "
                  f"ensaio {t['grão todo']['ensaio'] and round(t['grão todo']['ensaio']['r'], 4)} mm/s janelas n "
                  f"{t['grão todo']['janelas']['n']:.2f} | meio n {t['meio']['n']:.2f} | "
                  f"degraus {t['degraus']['mediana']}", flush=True)
    m = res["marcações do autor"][0]["taxa"]
    print("marcações: ensaio", round(m["grão todo"]["ensaio"]["r"], 4), "mm/s, janelas n", round(m["grão todo"]["janelas"]["n"], 2),
          "meio n", round(m["meio"]["n"], 2), "degraus", m["degraus"]["mediana"])
    (OUT / "comparacao_geral.json").write_text(json.dumps(dict(borda=borda, linhas=res), ensure_ascii=False))
    print("ok", OUT)


if __name__ == "__main__":
    main()
