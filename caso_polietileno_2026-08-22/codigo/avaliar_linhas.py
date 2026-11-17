"""Avaliação única de qualquer linha da superfície.

Base: tempo da câmera vezes o fator de tempo medido, janelas da janela de análise até o fim
da análise, taxa pelos extremos da altura média da porta e G_ox = ṁ_ox / (L H médio).
Ver codigo/relogio_e_corte.py.

Por seção do grão: correlação log G × log taxa e expoente n, na mediana de 54 escolhas
de janela (início de 0 a 6,8 s depois do começo da janela de análise, janelas de 15, 20
e 25 s).
Travas, medidas na linha inteira:
  marcas   % dos pontos marcados a até 4 px (cada face);
  recuo    % de (coluna, passo de 5 s) em que a face volta para dentro da porta > 3 px;
  borda    contraste vertical na linha / maior contraste a ±15 px (1 = na borda mais forte);
  ref      distância mediana à linha 2b (px) fora das marcas.

    PYTHONPATH=codigo python codigo/avaliar_linhas.py
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.ndimage import median_filter

import ensaio as E
import faixas_grao as FG
import recompensa_lei as R
import lei_potencia as RG
import relogio_e_corte as C
from taxas_sessao import MM

SAIDA = R.OUT / "melhor_marcacao"
SECOES = {"grão todo": (10, 120), "início 10–40": (10, 40), "meio 40–80": (40, 80), "final 80–120": (80, 120),
          "1ª metade": (10, 60), "2ª metade": (60, 120)}
OCULTA = "meio 40–80"
K0, K1 = 78, 766                      # trecho das travas: chama no grão todo até o salto de vazão
GRAO = R.GRAO


@lru_cache(1)
def contexto():
    """Tempo (fator e deslocamento de relogio_e_corte.py), início da janela de análise
    (configuração) e fim das janelas no relógio do registro (salto de vazão)."""
    d = json.loads((C.SAIDA / "relogio_e_corte.json").read_text())
    tm = C.Tempo(d["fator"]["s"], d["fator"]["T0"])
    return tm, float(E.JANELA_ANALISE_S[0]), float(d["fim_banco"])


# ----------------------------------------------------------------------------- linhas
def manual():
    """Marcações do autor, interpoladas no tempo coluna a coluna (só entre quadros marcados)."""
    out = []
    for f in ("baixo", "cima"):
        A = R.autor(f)
        Y = np.full_like(A, np.nan)
        for j in range(93):
            ok = np.isfinite(A[:, j])
            if ok.sum() >= 2:
                ks = np.flatnonzero(ok)
                kk = np.arange(ks[0], ks[-1] + 1)
                Y[kk, j] = np.interp(kk, ks, A[ks, j])
        out.append(Y)
    return tuple(out)


def run(nome):
    z = np.load(R.OUT / f"runs/{nome}/saida.npz")
    return z["baixo"].astype(float), z["cima"].astype(float)


def candidatos_base() -> dict:
    import experimento_p2_v2 as P2
    cl = np.load(P2.E1 / "classico.npz")
    L = {"manual (suas marcações)": manual(),
         "visão clássica pura": (cl["y_inf"].astype(float), cl["y_sup"].astype(float))}
    for s in (0, 1, 2):
        L[f"S1 s{s} (marcações de 21/10)"] = tuple(np.load(R.S1_RUNS / f"S1_{f}_s{s}/saida.npz")["bruto"].astype(float)
                                                  for f in ("baixo", "cima"))
    for nome, rot in (("nenhum_d4", "2b s0"), ("nenhum_d4_s1", "2b s1"), ("nenhum_d4_s2", "2b s2")):
        if (R.OUT / f"runs/{nome}/saida.npz").exists():
            L[rot] = run(nome)
    return L


# ----------------------------------------------------------------------------- recompensa
def serie_H(Yb, Yc, x0, x1):
    """Altura média da porta (mm) na seção, quadro a quadro. Linhas das redes: filtro de
    mediana de 9 amostras no tempo (como em serie_porta.serie_H). Linha manual (com
    NaN): só as colunas com as duas faces, sem filtro."""
    cols = FG.faixa_cols(x0, x1)
    if np.isnan(Yb).any() or np.isnan(Yc).any():
        h = (Yb - Yc)[:, cols]
        n = np.isfinite(h).sum(1)
        H = np.where(n >= max(3, 0.3 * len(cols)), np.nanmean(np.where(np.isfinite(h), h, np.nan), axis=1), np.nan)
        return H * MM
    return (median_filter(Yb, size=(9, 1)) - median_filter(Yc, size=(9, 1)))[:, cols].mean(1) * MM


def janelas(H, t0, dur):
    tm, _, fim = contexto()
    P = C.janelas_desde(H, tm, t0, fim, dur)
    return [p for p in P if np.isfinite(p["r"]) and np.isfinite(p["G"])]


def metricas(P):
    if len(P) < 3:
        return dict(n_jan=len(P), pearson=np.nan, n=np.nan, r2=np.nan)
    G = np.array([p["G"] for p in P]); r = np.array([p["r"] for p in P])
    ok = r > 0
    lr = np.log10(np.where(ok, r, np.nan)); lg = np.log10(G)
    if ok.sum() >= 3:
        pe = float(np.corrcoef(lg[ok], lr[ok])[0, 1]); n, b = np.polyfit(lg[ok], lr[ok], 1)
    else:
        pe, n = np.nan, np.nan
    return dict(n_jan=len(P), pearson=pe, n=float(n), r2=pe ** 2 if np.isfinite(pe) else np.nan)


ESCOLHAS = [(t0, dur) for t0 in np.arange(0, 7.01, 0.4) for dur in (15.0, 20.0, 25.0)]


def recompensa(Yb, Yc) -> dict:
    """Correlação e expoente por seção do grão, na janela de análise."""
    _, corte, _ = contexto()
    out = {}
    for ns, (x0, x1) in SECOES.items():
        H = serie_H(Yb, Yc, x0, x1)
        base = metricas(janelas(H, corte, 20.0))
        var = [metricas(janelas(H, corte + t0, dur)) for t0, dur in ESCOLHAS]
        med = {k: float(np.nanmedian([v[k] for v in var])) for k in ("pearson", "n")}
        out[ns] = dict(base=base, mediana=med, pontos=janelas(H, corte, 20.0))
    return out


# ----------------------------------------------------------------------------- travas
@lru_cache(1)
def _grad():
    cubo = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    ks = np.arange(K0, K1 + 1, 10)
    g = np.stack([np.abs(np.diff(np.asarray(cubo[k], np.float32).mean(2), axis=0)) for k in ks])   # (n, 592, 93)
    return ks, median_filter(g, size=(1, 3, 1))


def travas(Yb, Yc, ref=None) -> dict:
    Ab, Ac = R.autor("baixo"), R.autor("cima")
    res = {}
    for f, Y, A in (("baixo", Yb, Ab), ("cima", Yc, Ac)):
        ks = [k for k in range(K0, K1 + 1) if np.isfinite(A[k]).sum() > 10]
        e = np.concatenate([np.abs(Y[k] - A[k])[np.isfinite(A[k]) & np.isfinite(Y[k])] for k in ks])
        res[f"marcas_{f}"] = float(np.mean(e <= 4.0)) if e.size else np.nan
        res[f"erro_mediano_{f}"] = float(np.median(e)) if e.size else np.nan
    # recuo: a face anda para dentro da porta mais de 3 px em 15 amostras (~5 s)
    rec = []
    for Y, s in ((Yb, 1), (Yc, -1)):
        Yf = Y[K0:K1 + 1][:, GRAO] if np.isnan(Y).any() else median_filter(Y[K0:K1 + 1][:, GRAO], size=(9, 1))
        d = s * (Yf[15:] - Yf[:-15])
        rec.append(np.nanmean(d[::15] < -3.0))
    res["recuo"] = float(np.mean(rec))
    # borda
    ks, g = _grad()
    rel = []
    for Y in (Yb, Yc):
        for i, k in enumerate(ks):
            for j in GRAO:
                y = Y[k, j]
                if not np.isfinite(y) or y < 16 or y > g.shape[1] - 17:
                    continue
                yi = int(round(y))
                rel.append(g[i, yi - 1:yi + 2, j].max() / max(g[i, yi - 15:yi + 16, j].max(), 1e-6))
    res["borda"] = float(np.mean(rel)) if rel else np.nan
    if ref is not None:
        d = np.abs(np.concatenate([(Yb - ref[0])[K0:K1 + 1][:, GRAO].ravel(), (Yc - ref[1])[K0:K1 + 1][:, GRAO].ravel()]))
        res["dist_ref_px"] = float(np.nanmedian(d)); res["dist_ref_p95"] = float(np.nanpercentile(d, 95))
    return res


def passa(tr, rc, tr_ref) -> tuple[bool, list[str]]:
    """Travas duras: nenhuma pode falhar."""
    falhas = []
    for f in ("baixo", "cima"):
        if not tr[f"marcas_{f}"] >= 0.90:
            falhas.append(f"marcas {f} {tr[f'marcas_{f}'] * 100:.0f} % < 90 %")
    if tr["recuo"] > max(0.05, 2 * tr_ref["recuo"]):
        falhas.append(f"recuo {tr['recuo'] * 100:.1f} %")
    if np.isfinite(tr["borda"]) and tr["borda"] < 0.9 * tr_ref["borda"]:
        falhas.append(f"borda {tr['borda']:.2f} < 0,9× a 2b")
    if tr.get("dist_ref_p95", 0) > 12:
        falhas.append(f"95 % da distância à 2b {tr['dist_ref_p95']:.0f} px > 12")
    g = rc["grão todo"]["mediana"]
    if not g["pearson"] >= 0.5:
        falhas.append(f"grão todo quebrou (r {g['pearson']:.2f})")
    return (not falhas), falhas


def avaliar(L: dict, ref_nome="2b s0") -> dict:
    ref = L[ref_nome]
    tr_ref = travas(*ref, ref=ref)
    out = {}
    for nome, (Yb, Yc) in L.items():
        rc = recompensa(Yb, Yc)
        tr = travas(Yb, Yc, ref=None if nome.startswith("manual") else ref)
        ok, falhas = passa(tr, rc, tr_ref) if not nome.startswith("manual") else (True, [])
        m = rc[OCULTA]["mediana"]
        out[nome] = dict(recompensa=rc, travas=tr, passa=ok, falhas=falhas,
                         pontuacao=dict(pearson=m["pearson"], n=m["n"]))
        print(f"{nome:40s} meio: r {m['pearson']:+.2f} n {m['n']:+.2f} | "
              f"marcas {tr['marcas_baixo'] * 100:3.0f}/{tr['marcas_cima'] * 100:3.0f} % recuo {tr['recuo'] * 100:4.1f} % "
              f"borda {tr['borda']:.2f} ref {tr.get('dist_ref_px', np.nan):.1f}/{tr.get('dist_ref_p95', np.nan):.0f} px | "
              f"grão r {rc['grão todo']['mediana']['pearson']:+.2f} n {rc['grão todo']['mediana']['n']:+.2f} | "
              f"{'PASSA' if ok else 'FALHA: ' + '; '.join(falhas)}", flush=True)
    return out


def gravar(res, arq="avaliacao_base.json"):
    SAIDA.mkdir(parents=True, exist_ok=True)

    def limpo(o):
        if isinstance(o, dict):
            return {k: limpo(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpo(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (SAIDA / arq).write_text(json.dumps(limpo(res), indent=1, ensure_ascii=False))


ROTULOS = {"oc_": "recompensa oculta"}


def rotulo_run(nome):
    if not nome.startswith("oc_"):
        return nome
    cfg = json.loads((R.OUT / f"runs/{nome}/config.json").read_text())
    if cfg["recompensa"] == "nada":
        return f"{nome} (sem recompensa, d {cfg['d']:g})"
    marc = "com marcas" if cfg.get("marcas", True) else "SEM marcas"
    return f"{nome} ({cfg['recompensa']} w {cfg['w']:g}, {marc}, d {cfg['d']:g}, D {cfg['D']:g})"


def tabela(res: dict, arq="tabela.md"):
    """Ranking: primeiro quem passa nas travas, depois a maior fração de pontos marcados a até 4 px."""
    ordem = sorted(res, key=lambda n: (not res[n]["passa"],
                                       -(res[n]["travas"]["marcas_baixo"] + res[n]["travas"]["marcas_cima"])))
    linhas = ["| linha | passa | meio 40–80: r / n | grão todo: r / n | 2ª metade: r / n | "
              "final: r / n | marcas b/c | recuo | dist. 2b (p95) |", "|" + "---|" * 9]
    for n in ordem:
        d = res[n]; rc = d["recompensa"]; tr = d["travas"]
        o, g, m2, fi = rc[OCULTA]["mediana"], rc["grão todo"]["mediana"], rc["2ª metade"]["mediana"], rc["final 80–120"]["mediana"]
        linhas.append(f"| {n} | {'sim' if d['passa'] else 'não: ' + '; '.join(d['falhas'])} | "
                      f"{o['pearson']:.2f} / {o['n']:.2f} | {g['pearson']:.2f} / {g['n']:.2f} | "
                      f"{m2['pearson']:.2f} / {m2['n']:.2f} | {fi['pearson']:.2f} / {fi['n']:.2f} | {tr['marcas_baixo'] * 100:.0f}/{tr['marcas_cima'] * 100:.0f} % | "
                      f"{tr['recuo'] * 100:.1f} % | {tr.get('dist_ref_p95', float('nan')):.0f} px |")
    (SAIDA / arq).write_text("\n".join(linhas) + "\n")
    return ordem


if __name__ == "__main__":
    import sys
    L = candidatos_base()
    for nome in sys.argv[1:]:
        if (R.OUT / f"runs/{nome}/saida.npz").exists():
            L[rotulo_run(nome)] = run(nome)
    res = avaliar(L)
    gravar(res, "avaliacao.json")
    tabela(res)
