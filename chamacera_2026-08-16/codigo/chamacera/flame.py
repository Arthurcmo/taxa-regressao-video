"""
Métricas de chama derivadas da máscara — a metade "chama" do trabalho.

Todas são grandezas que aparecem na literatura de slab burner e que a segmentação
torna mensuráveis frame a frame, em vez de por inspeção visual:

  h_f(x)      altura da chama acima da superfície  -> espessura da camada limite
  standoff    distância superfície->base da chama  -> bloqueio por sopro (blowing)
  A_f(t)      área luminosa                        -> proxy de liberação de calor
  f_flicker   frequência dominante de A_f(t)       -> assinatura do modo instável
  x_ig        posição de ancoragem da chama        -> comprimento de ignição
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CHAMA


@dataclass
class FlameMetrics:
    h_mm: np.ndarray            # (N, W) altura da chama por coluna [mm]
    standoff_mm: np.ndarray     # (N, W) folga superfície->chama [mm]
    area_mm2: np.ndarray        # (N,)   área luminosa [mm²]
    h_bar_mm: np.ndarray        # (N,)   altura média
    x_ig_mm: np.ndarray         # (N,)   ancoragem axial da chama [mm]
    flicker_hz: float           # frequência dominante
    flicker_power: float        # fração da potência no pico (0..1)


def flame_metrics(masks: np.ndarray, surface_px: np.ndarray, fps: float,
                  mm_per_px: float, min_cols: int = 3) -> FlameMetrics:
    """
    masks: (N, H, W) uint8 ; surface_px: (N, W) linha da superfície.
    Alturas são medidas *a partir da superfície instantânea*, não do fundo do
    quadro — caso contrário o recuo do grão contamina a altura da chama.
    """
    N, H, W = masks.shape
    h = np.zeros((N, W)); so = np.full((N, W), np.nan); xig = np.zeros(N)
    area = np.zeros(N)
    yy = np.arange(H)[:, None]

    for n in range(N):
        fl = masks[n] == CHAMA
        area[n] = fl.sum() * mm_per_px ** 2
        tem = fl.any(0)
        ytop = np.where(tem, np.argmax(fl, 0), np.nan)
        ybot = np.where(tem, H - 1 - np.argmax(fl[::-1], 0), np.nan)
        surf = surface_px[n]
        h[n] = np.where(tem, np.clip(surf - ytop, 0, None), 0.0) * mm_per_px
        so[n] = np.where(tem, np.clip(surf - ybot, 0, None), np.nan) * mm_per_px
        cols = np.flatnonzero(tem)
        xig[n] = cols[0] * mm_per_px if cols.size >= min_cols else np.nan

    # ---- flicker: pico do espectro da série de área, sem componente DC ---- #
    f_hz, f_pow = 0.0, 0.0
    if N >= 16 and np.isfinite(area).all() and area.std() > 0:
        a = (area - area.mean()) * np.hanning(N)
        P = np.abs(np.fft.rfft(a)) ** 2
        fr = np.fft.rfftfreq(N, 1.0 / fps)
        if len(P) > 2:
            k = 1 + int(np.argmax(P[1:]))
            f_hz, f_pow = float(fr[k]), float(P[k] / max(P[1:].sum(), 1e-12))

    return FlameMetrics(h, so, area, np.nanmean(h, 1), xig, f_hz, f_pow)
