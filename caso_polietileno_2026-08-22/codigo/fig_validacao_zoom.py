"""Validação sobre os quadros marcados, ampliada (09/11/2026).

Faixas de 20 mm de largura em volta de cada face, na entrada (15 a 35 mm), no meio (50 a
70 mm) e no fim do grão (95 a 115 mm), em três instantes: 93 s (bloco de teste B1),
135 s e 180 s. Linha 2b (semente 0) tracejada, para deixar a superfície visível; marcações
do autor em círculos verdes vazados. Saída: figuras_tg2/fig_validacao_zoom.jpg.

    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python -W ignore codigo/fig_validacao_zoom.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

import avaliar_linhas as AL
import face_cima as FC
import figuras_sementes as FS
import figuras_tg2 as F
from experimento_s_sessao import XC
from taxas_sessao import MM

KS = (324, 450, 585)
SECS = (("entrada, 15 a 35 mm", 15), ("meio, 50 a 70 mm", 50), ("fim, 95 a 115 mm", 95))
ESC, MEIA = 2, 28


def faixa(im, Y, pts, k, x0mm, face):
    x0 = int(x0mm / MM + 10); x1 = x0 + int(20 / MM)
    cols = (XC >= x0) & (XC <= x1)
    yc = float(np.nanmedian(Y[k, cols]))
    ya, yb = int(max(0, yc - MEIA)), int(min(im.shape[0], yc + MEIA))
    c = cv2.resize(im[ya:yb, x0:x1], None, fx=ESC, fy=ESC, interpolation=cv2.INTER_CUBIC)
    FC.tracejado(c, np.stack([(XC - x0) * ESC, (Y[k] - ya) * ESC], 1), (255, 229, 0), 1, 10, 7)
    for x, y in pts:
        if x0 <= x <= x1 and ya <= y <= yb:
            p = (int((x - x0) * ESC), int((y - ya) * ESC))
            cv2.circle(c, p, 5, (0, 0, 0), 2, cv2.LINE_AA); cv2.circle(c, p, 5, (0, 255, 106), 1, cv2.LINE_AA)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def main():
    Yb, Yc = AL.run("nenhum_d4")
    fig, axs = plt.subplots(6, 3, figsize=(F.LARG, 3.55), gridspec_kw=dict(hspace=0.12, wspace=0.04, left=0.13, right=0.995, top=0.94, bottom=0.01))
    for j, k in enumerate(KS):
        im = cv2.cvtColor(F.roi(k), cv2.COLOR_RGB2BGR)
        for i, (rot, x0) in enumerate(SECS):
            for f, (Y, nome) in enumerate(((Yc, "cima"), (Yb, "baixo"))):
                ax = axs[2 * i + f, j]
                ax.imshow(faixa(im, Y, FS.pontos(nome, k), k, x0, nome))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(f"{rot.split(',')[0]}, {'superior' if nome == 'cima' else 'inferior'}", fontsize=6.8,
                                  rotation=0, ha="right", va="center")
        axs[0, j].set_title(f"{F.br(FS.t(k), 0)} s{' (teste B1)' if k == 324 else ''}", fontsize=8.5)
    fig.savefig(F.SAIDA / "fig_validacao_zoom.jpg", dpi=300, pil_kwargs={"quality": 92})
    print("ok")


if __name__ == "__main__":
    main()
