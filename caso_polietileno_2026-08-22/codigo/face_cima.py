"""Face superior: aproximar a curva da chama de 0,5 em 0,5 px até 3 px (08/11/2026).

Pedido do autor: a face de baixo parece boa e a de cima parece um pouco fora. Na melhor
curva (2bv2_s1), só a face superior anda δ px para o lado da chama (y + δ). Para cada δ:
  borda    gradiente vertical médio do brilho na posição da linha (quadros da janela
           de análise, colunas do grão), relativo ao de δ = 0;
  marcas   distância média às marcações da face superior;
  taxa     ensaio inteiro e as 5 janelas do TG, grão e seções.
GIFs: a melhor curva tracejada e as marcações leves em todos os quadros marcados (inteiro
e faixa da face superior), e um GIF que percorre δ = 0 a 3 px em recortes ampliados 3×.
Saída: experimento_recompensa_lei_2026-11-06/face_cima/

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/face_cima.py
"""
from __future__ import annotations

import ensaio as E
import json

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

import avaliar_linhas as AL
import figuras_sementes as FS
import figuras_tg2 as F
import recompensa_lei as R
import resultados_tg2 as T
from experimento_s_sessao import XC
from taxas_sessao import MM

OUT = R.OUT / "face_cima"
BASE = "2bv2_s1"
DELTAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SECS = {"grão 10–120": (10, 120), "entrada 10–40": (10, 40), "meio 40–80": (40, 80), "2ª metade 60–120": (60, 120)}
CIANO, VERDE, MAGENTA = (255, 229, 0), (0, 255, 106), (255, 92, 255)


def tracejado(im, pts, cor, esp=1, liga=7, desliga=6):
    """Polilinha tracejada (segmentos de `liga` px separados por `desliga` px)."""
    P = np.asarray(pts, float)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 2:
        return
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = 0.0
    while s < seg[-1]:
        a, b = s, min(s + liga, seg[-1])
        q = np.stack([np.interp(np.linspace(a, b, 6), seg, P[:, 0]), np.interp(np.linspace(a, b, 6), seg, P[:, 1])], 1)
        cv2.polylines(im, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, cor, esp, cv2.LINE_AA)
        s += liga + desliga


def marcas_leves(im, pts, escala=1.0, raio=2):
    cam = im.copy()
    for x, y in pts:
        cv2.circle(cam, (int(round(x * escala)), int(round(y * escala))), raio, VERDE, -1, cv2.LINE_AA)
    cv2.addWeighted(cam, 0.55, im, 0.45, 0, dst=im)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Yb, Yc0 = AL.run(BASE)
    cubo = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    ks = [k for k in range(323, 607, 4)]
    grad = {}
    for k in ks:
        I = gaussian_filter1d(np.asarray(cubo[k], np.float32).mean(2), 1.0, axis=0)
        grad[k] = np.abs(np.gradient(I, axis=0))
    A = R.autor("cima")
    res = {}
    for d in DELTAS:
        Yc = Yc0 + d
        g = [np.interp(Yc[k, j], np.arange(grad[k].shape[0]), grad[k][:, j]) for k in ks for j in R.GRAO]
        km = [k for k in range(323, 607) if np.isfinite(A[k]).sum() > 10]
        e = np.concatenate([(Yc[k] - A[k])[np.isfinite(A[k])] for k in km])
        r = dict(borda=float(np.mean(g)), marcas_mae=float(np.abs(e).mean()), marcas_vies=float(e.mean()),
                 marcas_ate4=float(np.mean(np.abs(e) <= 4)))
        for sec, (x0, x1) in SECS.items():
            H = AL.serie_H(Yb, Yc, x0, x1)
            r[sec] = T.ajuste(AL.janelas(H, T._corte, 20.0))
            if sec == "grão 10–120":
                e_ = T.um_ponto(H, 78, E.AMOSTRA_IGNICAO + T.FIM / (T.C.CAM * T._tm.s))
                r["ensaio"] = dict(r=e_["r_real"], G=e_["G"], q=e_["q_real"])
        res[d] = r
    b0 = res[0.0]["borda"]
    print("δ (px) | borda/δ0 | marcas: erro, viés (+ = para a chama), ≤4 px | ensaio ṙ, G, razão | janelas grão | meio | 2ª metade")
    for d, r in res.items():
        print(f"{d:4.1f} | {r['borda'] / b0:5.3f} | {r['marcas_mae']:.2f} px, {r['marcas_vies']:+.2f}, {r['marcas_ate4'] * 100:.0f}% | "
              f"{r['ensaio']['r']:.4f}, {r['ensaio']['G']:.2f}, {r['ensaio']['q']:.2f}× | "
              + " | ".join(f"{r[s]['a']:.4f} G^{r[s]['n']:.2f} {r[s]['razao']:.2f}×" for s in ("grão 10–120", "meio 40–80", "2ª metade 60–120")))
    (OUT / "face_cima.json").write_text(json.dumps({str(k): v for k, v in res.items()}, ensure_ascii=False, indent=1, default=float))

    # ---------------------------------------------------------------- GIF da melhor curva, quadros marcados
    marc = sorted({k for f in ("baixo", "cima") for k in range(963) if np.isfinite(R.autor(f)[k]).sum() > 10 and 78 <= k <= 766})
    inteiro, faixa = [], []
    for k in marc:
        im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR).copy()
        base = im.copy()
        for Y in (Yb, Yc0):
            tracejado(im, np.stack([XC[1:], Y[k][1:]], 1), CIANO, 2)
        marcas_leves(im, np.vstack([FS.pontos(f, k) for f in ("baixo", "cima")]), 1.0, 3)
        a = cv2.resize(im, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        bar = np.zeros((26, a.shape[1], 3), np.uint8)
        cv2.putText(bar, f"amostra {k + 1} | {FS.t(k):.1f} s | tracejado: melhor curva (2b nova s1) | pontos: suas marcacoes",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        inteiro.append(Image.fromarray(cv2.cvtColor(np.vstack([bar, a]), cv2.COLOR_BGR2RGB)))
        # faixa da face superior, resolução cheia
        yc = Yc0[k][1:]
        y0, y1 = int(max(0, np.nanmin(yc) - 35)), int(min(im.shape[0], np.nanmax(yc) + 35))
        y0, y1 = int(max(0, (y0 + y1) / 2 - 70)), int(min(im.shape[0], (y0 + y1) / 2 + 70))
        im2 = base.copy()
        tracejado(im2, np.stack([XC[1:], Yc0[k][1:]], 1), CIANO, 1)
        marcas_leves(im2, FS.pontos("cima", k), 1.0, 2)
        s_ = im2[y0:y1]
        s_ = cv2.copyMakeBorder(s_, 0, 140 - s_.shape[0], 0, 0, cv2.BORDER_CONSTANT) if s_.shape[0] < 140 else s_[:140]
        bar = np.zeros((24, s_.shape[1], 3), np.uint8)
        cv2.putText(bar, f"face superior | amostra {k + 1} | {FS.t(k):.1f} s | resolucao cheia", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        faixa.append(Image.fromarray(cv2.cvtColor(np.vstack([bar, s_]), cv2.COLOR_BGR2RGB)))
    inteiro[0].save(OUT / "melhor_curva_marcacoes.gif", save_all=True, append_images=inteiro[1:], duration=900, loop=0)
    faixa[0].save(OUT / "face_superior_marcacoes.gif", save_all=True, append_images=faixa[1:], duration=900, loop=0)

    # ---------------------------------------------------------------- GIF que percorre δ
    x0, x1 = int(50 / MM), int(70 / MM)
    quadros = []
    for d in DELTAS:
        tiles = []
        for k in (375, 450, 525, 600):
            full = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR)
            yc = float(np.nanmedian(Yc0[k, x0 // 20:x1 // 20]))
            ya, yb = int(max(0, yc - 25)), int(min(full.shape[0], yc + 25))
            im = cv2.resize(full[ya:yb, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            tracejado(im, np.stack([(XC - x0) * 3, (Yc0[k] + d - ya) * 3], 1), CIANO, 1, 10, 8)
            pts = FS.pontos("cima", k)
            pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)] if len(pts) else pts
            marcas_leves(im, np.c_[(pts[:, 0] - x0) * 3, (pts[:, 1] - ya) * 3] if len(pts) else [], 1.0, 4)
            bar = np.zeros((24, im.shape[1], 3), np.uint8)
            cv2.putText(bar, f"{FS.t(k):.0f} s | face superior + {d:.1f} px para a chama", (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, im]))
        sep = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
        g_ = np.vstack([np.hstack([tiles[0], sep, tiles[1]]), np.full((6, tiles[0].shape[1] * 2 + 6, 3), 255, np.uint8),
                        np.hstack([tiles[2], sep, tiles[3]])])
        quadros.append(Image.fromarray(cv2.cvtColor(g_, cv2.COLOR_BGR2RGB)))
    quadros[0].save(OUT / "face_superior_deslocamentos.gif", save_all=True, append_images=quadros[1:], duration=1100, loop=0)
    print("GIFs em", OUT, f"({len(marc)} quadros marcados)")


if __name__ == "__main__":
    main()
