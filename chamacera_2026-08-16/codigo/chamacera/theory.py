"""Priors físicos fracos para a taxa de regressão.

Uma lei de combustão não observa a interface no vídeo. Ela só informa qual
*incremento temporal* é plausível. Por isso este módulo nunca transforma a lei
em rótulo e nunca corrige uma borda isoladamente: ele fornece um custo suave
para a decodificação temporal e uma métrica de discordância para seleção ativa.

A forma ``r = a G_ox**n`` é a correlação empírica mais comum para combustíveis
híbridos. Os coeficientes devem vir de ensaios independentes do vídeo usado na
validação. Para combustível liquefazente, ``a`` e ``n`` continuam sendo
parâmetros efetivos: a teoria de camada líquida/arraste não deve ser usada como
se fosse uma verdade de pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawPrior:
    """Prior ``r = a G_ox**n`` em unidades SI.

    ``sigma_rel`` é a incerteza relativa de modelo (por exemplo, 0.30 para
    30 %). Ela deve incorporar dispersão entre ensaios e incerteza de vazão,
    não apenas o erro numérico do ajuste. Mantê-la explícita impede que uma
    correlação aproximada domine uma observação visual boa.
    """

    a: float
    n: float
    sigma_rel: float = 0.30

    def __post_init__(self):
        if not np.isfinite(self.a) or self.a <= 0:
            raise ValueError("a deve ser positivo e finito")
        if not np.isfinite(self.n):
            raise ValueError("n deve ser finito")
        if not np.isfinite(self.sigma_rel) or self.sigma_rel <= 0:
            raise ValueError("sigma_rel deve ser positivo e finito")

    def rate_mm_s(self, G_ox: np.ndarray | float) -> np.ndarray:
        """Taxa prevista [mm/s], para ``G_ox`` em kg/(m² s)."""
        G = np.asarray(G_ox, dtype=float)
        if not np.isfinite(G).all() or np.any(G <= 0):
            raise ValueError("G_ox deve ser positivo e finito")
        return self.a * G ** self.n

    def sigma_mm_s(self, G_ox: np.ndarray | float) -> np.ndarray:
        return self.sigma_rel * self.rate_mm_s(G_ox)

    def steps_grid(self, G_ox: np.ndarray | float, dt_s: float,
                   mm_per_grid_px: float) -> tuple[np.ndarray, np.ndarray]:
        """Converte a taxa em alvo e desvio-padrão por quadro na grade da rede."""
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s deve ser positivo e finito")
        if not np.isfinite(mm_per_grid_px) or mm_per_grid_px <= 0:
            raise ValueError("mm_per_grid_px deve ser positivo e finito")
        f = dt_s / mm_per_grid_px
        return self.rate_mm_s(G_ox) * f, self.sigma_mm_s(G_ox) * f


def rate_disagreement(observed_mm_s, predicted_mm_s, sigma_model_mm_s):
    """Resíduo normalizado da taxa; valores altos são candidatos à revisão.

    Não é uma métrica de acurácia nem um teste de validade da lei: mede apenas
    incompatibilidade entre duas fontes independentes de evidência.
    """
    obs, pred, sig = np.broadcast_arrays(np.asarray(observed_mm_s, float),
                                         np.asarray(predicted_mm_s, float),
                                         np.asarray(sigma_model_mm_s, float))
    if not (np.isfinite(obs).all() and np.isfinite(pred).all() and
            np.isfinite(sig).all()) or np.any(sig <= 0):
        raise ValueError("taxas e sigma devem ser finitos; sigma deve ser positivo")
    return np.abs(obs - pred) / sig
