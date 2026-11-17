"""Números de resultado no tempo real, para a janela de análise da configuração.

Base no tempo real (tempo marcado pela câmera vezes o fator medido em relogio_e_corte.py).
Calcula, para cada linha (marcações manuais, clássico, S1, 2b e forma):
  janelas    janelas de 20 s do início ao fim da janela de análise: reta a·G^n e R²;
  multiverso 54 escolhas de início (0 a 6,8 s depois do início da janela) e duração (15, 20
             e 25 s), mediana da correlação log G × log ṙ e de n, por seção do grão;
  ensaio     a queima inteira como um ponto só, da amostra 78 ao fim da janela.
Saída: experimento_recompensa_lei_2026-11-06/resultados_tg2/resultados_tg2.json

    PYTHONPATH=codigo python -W ignore codigo/resultados_tg2.py
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

import avaliar_linhas as AL
import janelas_taxa as Q
import recompensa_lei as R
import lei_potencia as RG
import ensaio as E
import relogio_e_corte as C

C.mdot = lru_cache(None)(C.mdot)
OUT = R.OUT / "resultados_tg2"
FIM = E.JANELA_ANALISE_S[1]         # fim da janela de análise, s desde a ignição
SECOES = {"grão todo": (10, 120), "entrada 10–40": (10, 40), "meio 40–80": (40, 80), "final 80–120": (80, 120)}
_tm, _corte, _ = AL.contexto()
AL.contexto = lambda: (_tm, _corte, _tm.ign + FIM)


def linhas() -> dict:
    import experimento_p2_v2 as P2
    cl = np.load(P2.E1 / "classico.npz")
    L = {"marcações do autor": AL.manual(),
         "clássico": (cl["y_inf"].astype(float), cl["y_sup"].astype(float))}
    for s in (0, 1, 2):
        L[f"S1 s{s}"] = tuple(np.load(R.S1_RUNS / f"S1_{f}_s{s}/saida.npz")["bruto"].astype(float)
                              for f in ("baixo", "cima"))
    for nome, rot in (("nenhum_d4", "2b s0"), ("nenhum_d4_s1", "2b s1"), ("nenhum_d4_s2", "2b s2"),
                      ("oc_forma_w5_s0", "forma s0"), ("p3_forma_w5_s1", "forma s1"), ("p3_forma_w5_s2", "forma s2")):
        L[rot] = AL.run(nome)
    return L


def ajuste(P):
    G = np.array([p["G"] for p in P]); r = np.array([p["r"] for p in P])
    aj = RG.ajuste_proprio(G, r)
    return dict(G=G.tolist(), r=r.tolist(), t=[p["t"] for p in P], a=float(aj["a"]), n=float(aj["n"]),
                r2=float(aj["r2"]))


def um_ponto(H, k0, k1):
    """Janela única entre as amostras k0 e k1, nos dois tempos."""
    t0 = (k0 - E.AMOSTRA_IGNICAO) * C.CAM * _tm.s
    t1 = (k1 - E.AMOSTRA_IGNICAO) * C.CAM * _tm.s
    w = Q.janelas_k(H, _tm, t0, t1, 1)[0]
    r = w["r"] * 1e3
    G = w["md"] / (Q.L_M * w["Hm"])
    return dict(t=[t0, t1], r_real=r, r_cam=r * _tm.s, G=G, md=w["md"] * 1e3, Hi=w["Ha"] * 1e3, Hf=w["Hb"] * 1e3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = linhas()
    res = dict(fim=FIM, corte=_corte, s=_tm.s, linhas={})
    for nome, (Yb, Yc) in L.items():
        d = {}
        for sec, (x0, x1) in SECOES.items():
            H = AL.serie_H(Yb, Yc, x0, x1)
            base = AL.janelas(H, _corte, 20.0)
            var = [AL.metricas(AL.janelas(H, _corte + t0, dur)) for t0, dur in AL.ESCOLHAS]
            d[sec] = dict(janelas=ajuste(base),
                          multiverso=dict(pearson=float(np.nanmedian([v["pearson"] for v in var])),
                                          n=float(np.nanmedian([v["n"] for v in var])),
                                          n_min=float(np.nanmin([v["n"] for v in var])),
                                          n_max=float(np.nanmax([v["n"] for v in var]))))
            if sec == "grão todo":
                try:
                    d["ensaio"] = um_ponto(H, 78, E.AMOSTRA_IGNICAO + FIM / (C.CAM * _tm.s))
                except Exception as e:
                    d["ensaio"] = dict(erro=str(e))
        res["linhas"][nome] = d
        g = d["grão todo"]; m = d["meio 40–80"]
        print(f"{nome:20s} grão: janelas {g['janelas']['a']:.4f} G^{g['janelas']['n']:.2f} R² {g['janelas']['r2']:.2f} "
              f"| multiverso n {g['multiverso']['n']:.2f} "
              f"r {g['multiverso']['pearson']:.2f} | meio n {m['multiverso']['n']:.2f} "
              f"r {m['multiverso']['pearson']:.2f}", flush=True)
        for k in ("ensaio",):
            e = d.get(k, {})
            if "r_real" in e:
                print(f"{'':20s} {k}: {e['r_real']:.4f} real / {e['r_cam']:.4f} câmera em G {e['G']:.2f}, "
                      f"ṁ {e['md']:.2f} g/s, H {e['Hi']:.2f}→{e['Hf']:.2f}")
    (OUT / "resultados_tg2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
