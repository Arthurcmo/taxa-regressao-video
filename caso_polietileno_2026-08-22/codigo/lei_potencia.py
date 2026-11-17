"""Ajuste da lei de regressão r = a G^n a uma nuvem de pontos (G, taxa)."""
from __future__ import annotations

import numpy as np


def ajuste_proprio(g, r):
    """Ajusta r = a G^n por regressão linear em log-log. Devolve dict com a, n, R^2."""
    g = np.asarray(g, float)
    r = np.asarray(r, float)
    ok = np.isfinite(g) & np.isfinite(r) & (g > 0) & (r > 0)
    g, r = g[ok], r[ok]
    if g.size < 3:
        return dict(a=np.nan, n=np.nan, r2=np.nan, n_pontos=int(g.size),
                    faixa=(np.nan, np.nan))
    x, y = np.log10(g), np.log10(r)
    n, b = np.polyfit(x, y, 1)
    yhat = n * x + b
    r2 = 1.0 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return dict(a=float(10 ** b), n=float(n), r2=float(r2),
                n_pontos=int(g.size), faixa=(float(g.min()), float(g.max())))
