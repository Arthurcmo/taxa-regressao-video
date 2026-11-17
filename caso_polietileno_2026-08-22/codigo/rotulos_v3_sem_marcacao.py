"""Rótulos v3 da linha sem marcação humana (09/11/2026).

A SM1 aprendeu com o detector clássico de três passadas. Três defeitos dela vêm desses
rótulos, e não da rede:
  1. a curva da porta de entrada é cortada, porque a 2ª passada prende cada coluna a
     um polinômio de grau 4 ao longo do grão, que não faz curva fechada;
  2. a linha fica de 3 a 10 px do lado da chama, porque o rótulo é o FUNDO do vale
     escuro, e a definição escrita da superfície (Seção 3.3) é o limite do sólido, a
     borda do vale do lado do grão;
  3. na entrada da face inferior, ela fica na fuligem (cerca de 32 px).

Rótulos v3, todos a partir da imagem, sem nenhuma marcação:
  a. passada no tempo, coluna a coluna, com a irreversibilidade (igual à 1ª passada);
  b. limpeza local ao longo do grão: corredor em volta da mediana de 5 colunas, sem
     polinômio, para não apagar a curva;
  c. borda do lado do grão: a posição vai do fundo do vale até onde a intensidade sobe
     metade do caminho até o ombro do lado do grão (bordas_vale do detector);
  d. aprovação sem marcação: peso pela força do vale (como na SM1) e peso zero onde a
     coluna discorda das vizinhas em mais de 4 px ou onde o quadro está estourado.
Variante opcional --fuligem: na entrada da face inferior (10 a 40 mm), entre os vales
fortes da coluna, o escore favorece o mais fundo (lado do grão), pela regra escrita de
que a superfície fica abaixo da faixa escura.

As marcações do autor só entram em `avaliar`, como teste.

    PYTHONPATH=codigo .venv/bin/python -W ignore codigo/rotulos_v3_sem_marcacao.py gerar [--fuligem]
    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/rotulos_v3_sem_marcacao.py avaliar
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import detectar_interfaces as D
import experimento_p2_v2 as P2
import recompensa_lei as R

OUT = R.OUT / "sem_marcacao_v3"


def usar_saida(nome: str) -> None:
    global OUT
    OUT = R.OUT / nome
MO = {"y_sup": -1, "y_inf": +1}


def x_mm():
    import faixas_grao as FG
    return FG.X_MM


def gerar(fuligem: bool, base: str = "tempo", frac: float = 0.5, frac_inf: float | None = None,
          estatico: float = 0.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cl = np.load(P2.E1 / "classico.npz")
    C = np.load(P2.RAIZ / "_tmp/cubo_bgr.npy")
    G = C.astype(np.float32).mean(3)
    flare, _, _ = D.quadros_com_flare(C)
    del C
    N, H, nb = G.shape
    yc0 = D.centro_inicial(G)
    peso_q = np.where(flare, 0.2, 1.0)[:, None]
    xm = x_mm()
    res = {k: np.zeros((N, nb)) for k in ("y_sup", "y_inf", "c_sup", "c_inf", "v_sup", "v_inf")}
    Sc, I_, proibs = {}, {}, {}
    for j in range(nb):
        I = gaussian_filter1d(G[:, :, j], 2.0, axis=1)
        I_[j] = I
        Sc[j] = D.escore(I) * peso_q
        if estatico > 0:
            # o que fica escuro no mesmo pixel por muito tempo (fuligem, mancha na janela) não é a
            # superfície que recua: peso pela diferença para a média de ±30 s no mesmo pixel
            from scipy.ndimage import uniform_filter1d
            Mt = uniform_filter1d(I, 181, axis=0, mode="nearest")
            wst = np.clip((Mt - I) / estatico, 0, 1)
            Sc[j] = Sc[j] * gaussian_filter1d(wst, 2.0, axis=1)
        c0 = int(round(yc0[j]))
        for lado, sy in ((-1, "y_sup"), (+1, "y_inf")):
            proib = np.zeros(H, bool)
            if lado < 0:
                proib[c0:] = True; proib[:14] = True
            else:
                proib[:c0] = True; proib[H - 14:] = True
            proibs[(j, sy)] = proib
            S = Sc[j]
            if fuligem and sy == "y_inf" and 10 <= xm[j] < 40:
                # favorece o vale mais fundo: rampa de 0 a 0,6 ao longo de 120 px abaixo do centro
                ramp = np.clip((np.arange(H) - c0) / 120.0, 0, 1) * 0.6
                S = S + ramp[None, :] * (S > 0.35)
            res[sy][:, j] = D.viterbi(S, 2, 0.5, proib, MO[sy])
    for sy, sc, sv in (("y_sup", "c_sup", "v_sup"), ("y_inf", "c_inf", "v_inf")):
        ref = median_filter(res[sy], size=(1, 5), mode="nearest")
        for j in range(nb):
            S2 = D.corredor_no_escore(Sc[j], ref[:, j], 12.0, 2.5)
            if base == "classico" and estatico <= 0:
                y = cl[sy][:, j].astype(float)
            else:
                y = D.viterbi(S2, 2, 0.5, proibs[(j, sy)], MO[sy])
                y = D.refinar(S2, y, mono=MO[sy])
            res[sc][:, j] = Sc[j][np.arange(N), np.clip(np.round(y).astype(int), 0, H - 1)]
            lo, hi = D.bordas_vale(I_[j], y, frac=(frac_inf if (frac_inf is not None and sy == "y_inf") else frac))
            borda = hi if sy == "y_inf" else lo          # lado do grão
            borda = np.where(np.abs(borda - y) <= 15, borda, y)
            m = median_filter(borda - y, size=9, mode="nearest")   # deslocamento lento no tempo
            yb = y + m
            yb = np.maximum.accumulate(yb) if MO[sy] > 0 else -np.maximum.accumulate(-yb)
            res[sv][:, j] = y                               # fundo do vale, para comparação
            res[sy][:, j] = yb
    for sy, sc, face in (("y_sup", "c_sup", "cima"), ("y_inf", "c_inf", "baixo")):
        y = res[sy]; c = res[sc]
        w = np.clip(c / 0.6, 0, 1); w[c < 0.3] = 0.0
        dev = np.abs(y - median_filter(y, size=(1, 5), mode="nearest"))
        w[dev > 4] = 0.0
        w[flare] = 0.0
        np.savez_compressed(OUT / f"alvos_{face}.npz", y=y.astype(np.float32), peso=w.astype(np.float32),
                            fundo=res["v_sup" if face == "cima" else "v_inf"].astype(np.float32))
    (OUT / "rotulos_v3.json").write_text(json.dumps(dict(fuligem=fuligem, base=base, estatico=estatico, frac_sup=frac, frac_inf=frac if frac_inf is None else frac_inf, quadros_estourados=int(flare.sum())), indent=1))
    print("gravado em", OUT)


def avaliar() -> None:
    import figuras_sementes as FS
    import faixas_grao as FG
    from experimento_s_sessao import XC, divisao, marcas
    SENT = {"baixo": -1, "cima": +1}
    cl = np.load(P2.E1 / "classico.npz")
    linhas = {"clássico (rótulo da SM1)": {"baixo": cl["y_inf"], "cima": cl["y_sup"]},
              "v3, fundo do vale": {f: np.load(OUT / f"alvos_{f}.npz")["fundo"] for f in ("baixo", "cima")},
              "v3, borda do lado do grão": {f: np.load(OUT / f"alvos_{f}.npz")["y"] for f in ("baixo", "cima")}}
    div = divisao()
    lin = ["| rótulo | " + " | ".join(f"{f} {s}" for f in ("inferior", "superior") for s in ("entrada", "meio", "final", "B1", "B2")) + " |",
           "|---" * 11 + "|"]
    for nome, d in linhas.items():
        cel = []
        for face in ("baixo", "cima"):
            A = R.autor(face)
            ks = [k for k in range(963) if FS.CORTE <= FS.t(k) <= FS.FIM and np.isfinite(A[k]).sum() > 10]
            for x0, x1 in ((10, 40), (40, 80), (80, 120)):
                cols = FG.faixa_cols(x0, x1)
                e = np.concatenate([(d[face][k, cols] - A[k, cols])[np.isfinite(A[k, cols])] for k in ks])
                cel.append(f"{np.abs(e).mean():.1f} / {np.mean(SENT[face] * e):+.1f} / {100 * np.mean(np.abs(e) <= 4):.0f} %")
            for bloco in ("B1", "B2"):
                mk = marcas(face); e = []
                for k in div[bloco]:
                    if k in mk:
                        p = mk[k]; sel = (p[:, 0] >= FS.CENTRAIS[0]) & (p[:, 0] <= FS.CENTRAIS[1])
                        e.append(np.interp(p[sel, 0], XC, d[face][k]) - p[sel, 1])
                e = np.concatenate(e)
                cel.append(f"{np.abs(e).mean():.1f} / {np.mean(SENT[face] * e):+.1f} / {100 * np.mean(np.abs(e) <= 4):.0f} %")
        lin.append(f"| {nome} | " + " | ".join(cel) + " |")
    txt = ("Erro dos rótulos contra as marcações do autor (px): erro médio / deslocamento para a chama / "
           "% a até 4 px. Entrada, meio e final: quadros marcados da janela de análise.\n\n" + "\n".join(lin) + "\n")
    (OUT / "avaliacao_rotulos.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["gerar", "avaliar"])
    ap.add_argument("--fuligem", action="store_true")
    ap.add_argument("--base", choices=["tempo", "classico"], default="tempo")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--frac-inf", type=float, default=None)
    ap.add_argument("--estatico", type=float, default=0.0)
    ap.add_argument("--saida", default="sem_marcacao_v3")
    a = ap.parse_args()
    usar_saida(a.saida)
    gerar(a.fuligem, a.base, a.frac, a.frac_inf, a.estatico) if a.etapa == "gerar" else avaliar()
