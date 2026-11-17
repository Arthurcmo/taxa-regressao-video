"""Recorta o tempo gravado pela câmera em cada amostra, para o marcador mostrar.

A ROI do cubo não inclui o canto com o tempo. Este recorte guarda, por amostra, a
faixa do tempo do quadro original, em cinza e na metade do tamanho.

    .venv/bin/python codigo/extrair_carimbo.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import ensaio as E

from ler_contador import FAIXA_TEMPO

RAIZ = Path(__file__).resolve().parent.parent
VIDEO = E.VIDEO
SAIDA = RAIZ / "_tmp/carimbo_camera.npy"
Q_POR_AMOSTRA = 10


def main() -> None:
    y0, y1, x0, x1 = FAIXA_TEMPO
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saida, q = [], 0
    while cap.grab():
        if q % Q_POR_AMOSTRA == 0:
            ok, im = cap.retrieve()
            if not ok:
                raise RuntimeError(f"quadro {q} ilegível")
            g = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            saida.append(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        q += 1
    cap.release()
    np.save(SAIDA, np.stack(saida))
    print(f"{q} quadros de {n}; {len(saida)} recortes em {SAIDA}")


if __name__ == "__main__":
    main()
