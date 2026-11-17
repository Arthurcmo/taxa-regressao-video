"""
Le o contador de quadros gravado na imagem pela camera.

Esse contador e a unica base de tempo confiavel do arquivo: o conteiner foi
recodificado (WhatsApp) e traz 29,97 fps com um quadro repetido a cada cinco,
que e a assinatura de uma conversao de 24 para 30 quadros por segundo. O
contador da camera nao sofre disso -- ele anda de 1 em 1 a cada quadro
REALMENTE capturado, e repete quando o quadro do arquivo e copia.

Reconhecimento sem OCR externo: os digitos sao brancos, grandes e em posicao
fixa. Os dez moldes saem do proprio video, porque nos primeiros quadros o
contador vale 0, 1, 2, ... 9 com um digito so, na ordem. Depois disso e
casamento de molde por soma de quadrados.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, cv2

import ensaio as E

RAIZ = Path(__file__).resolve().parent.parent
FAIXA_CONTADOR = (908, 976, 1560, 1920)      # y0, y1, x0, x1
FAIXA_TEMPO = (984, 1052, 1560, 1920)
LARG_DIGITO = 37                             # largura de um digito, em px
ALT_DIGITO_MIN = 48                          # digito ~58 px, dois pontos ~41 px
TAM = (20, 28)                               # molde normalizado (largura, altura)


def glifos(quadro: np.ndarray, faixa) -> list[np.ndarray]:
    """Recorta a faixa, binariza e devolve os digitos separados, da esq. p/ dir."""
    y0, y1, x0, x1 = faixa
    g = cv2.cvtColor(quadro[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    b = (g > 170).astype(np.uint8)
    col = b.sum(0)
    saida, i = [], 0
    W = len(col)
    while i < W:
        if col[i] == 0:
            i += 1; continue
        j = i
        while j < W and col[j] > 0:
            j += 1
        larg = j - i
        lin = np.flatnonzero(b[:, i:j].sum(1) > 0)
        alt = (lin[-1] - lin[0] + 1) if lin.size else 0
        # os dois pontos do relogio tem a mesma largura de um "1" mas sao bem
        # mais baixos, entao quem separa e a ALTURA, nao a largura
        if alt >= ALT_DIGITO_MIN:
            # digitos vizinhos que se encostam viram um bloco largo: fatia
            n_dig = max(1, int(round(larg / LARG_DIGITO))) if larg >= 1.6 * LARG_DIGITO else 1
            for k in range(n_dig):
                a2 = i + int(round(k * larg / n_dig))
                b2 = i + int(round((k + 1) * larg / n_dig))
                rec = b[:, a2:b2]
                l2 = np.flatnonzero(rec.sum(1) > 0)
                if l2.size and (l2[-1] - l2[0]) >= 10:
                    rec = rec[l2[0]:l2[-1] + 1]
                    saida.append(cv2.resize(rec.astype(np.float32), TAM,
                                            interpolation=cv2.INTER_AREA))
        i = j
    return saida


def treinar_moldes(video: str, n: int = 60) -> np.ndarray:
    """
    Moldes dos dez digitos, tirados do proprio video: enquanto o contador tem
    um digito so, os valores distintos que aparecem sao 0, 1, 2, ... 9 nessa
    ordem. Nao ha nada para adivinhar.
    """
    cap = cv2.VideoCapture(video)
    vistos: list[np.ndarray] = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok: break
        gs = glifos(f, FAIXA_CONTADOR)
        if len(gs) != 1:
            break                              # passou de 9, acabou o treino
        g = gs[0]
        if not vistos or float(np.abs(g - vistos[-1]).sum()) > 8.0:
            vistos.append(g)
        if len(vistos) == 10:
            break
    cap.release()
    if len(vistos) != 10:
        raise RuntimeError(f"consegui {len(vistos)} moldes, esperava 10")
    return np.stack(vistos)


def ler(gs: list[np.ndarray], moldes: np.ndarray) -> tuple[int, float]:
    """Casa cada glifo com o molde mais proximo. Devolve (numero, pior erro)."""
    val, pior = 0, 0.0
    for g in gs:
        e = np.abs(moldes - g[None]).sum((1, 2))
        d = int(np.argmin(e))
        val = val * 10 + d
        pior = max(pior, float(e[d]))
    return val, pior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(E.VIDEO))
    ap.add_argument("--out", default=str(RAIZ / "resultados/quadros_camera.json"))
    a = ap.parse_args()

    moldes = treinar_moldes(a.video)
    cap = cv2.VideoCapture(a.video)
    n_arq = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    contador, tc, erro = [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        c, e1 = ler(glifos(f, FAIXA_CONTADOR), moldes)
        t, e2 = ler(glifos(f, FAIXA_TEMPO), moldes)
        contador.append(c); tc.append(t); erro.append(max(e1, e2))
        i += 1
    cap.release()

    c = np.array(contador); tcv = np.array(tc); er = np.array(erro)
    # o relogio e MMSSFF a 30 quadros por segundo -> conferencia independente
    mm, ss, ff = tcv // 10000, (tcv // 100) % 100, tcv % 100
    prev = mm * 1800 + ss * 30 + ff
    bate = int((prev == c).sum())

    d = np.diff(c)
    unico = np.concatenate([[True], d == 1])
    print(f"quadros no arquivo         : {i}")
    print(f"contador da camera         : {c[0]} a {c[-1]}  ({c[-1]-c[0]+1} quadros)")
    print(f"relogio gravado bate com o contador em {bate}/{i} quadros "
          f"({100*bate/i:.2f} %)")
    print(f"pior erro de casamento     : {er.max():.1f} (tipico {np.median(er):.1f})")
    print(f"saltos != 0 e != 1         : {int(((d != 0) & (d != 1)).sum())}")
    print(f"quadros unicos             : {int(unico.sum())}")
    print(f"quadros repetidos          : {int((d == 0).sum())} "
          f"({100*(d == 0).mean():.1f} % -- 20,0 % e a assinatura de 24->30)")

    json.dump(dict(n_arquivo=i, contador=c.tolist(), relogio=tcv.tolist(),
                   unico=unico.tolist(), bate_relogio=bate,
                   c0=int(c[0]), c1=int(c[-1])),
              open(a.out, "w"))
    print("gravado em", a.out)


if __name__ == "__main__":
    main()
