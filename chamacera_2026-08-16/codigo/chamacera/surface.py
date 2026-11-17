"""
Da máscara à física: perfil de superfície y_s(x,t) e taxa de regressão r-dot.

Esta é a etapa em que erros de segmentação viram erro de engenharia, então ela
é tratada como um problema de estimação, não de pós-processamento:

  - a interface é extraída com precisão sub-pixel (a regressão típica desloca a
    superfície por frações de pixel entre frames de câmera rápida);
  - a série y_s(x,·) é projetada no cone das funções não-decrescentes (PAVA),
    impondo a irreversibilidade física da queima antes de derivar;
  - a derivada usa Savitzky-Golay, que estima dy/dt por ajuste polinomial local
    em vez de diferença finita — diferença finita amplifica o ruído de
    segmentação por fps (~240x), e é o motivo usual de r-dot(t) sair "ruidoso".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from . import CERA
from .bootstrap import isotonic_increasing


# --------------------------------------------------------------------------- #
def surface_from_masks(masks: np.ndarray, subpixel: bool = True) -> np.ndarray:
    """
    (N, H, W) uint8 -> (N, W) float64 com a linha da superfície por coluna.
    Sub-pixel por interpolação linear da fração de cobertura no pixel de borda.
    Colunas sem cera recebem NaN (ex.: grão consumido localmente).
    """
    N, H, W = masks.shape
    out = np.full((N, W), np.nan)
    for n in range(N):
        cera = masks[n] == CERA
        tem = cera.any(0)
        first = np.argmax(cera, 0).astype(np.float64)
        if subpixel:
            frac = cera.sum(0) / max(H, 1)
            first = first - 0.5 * (frac > 0)          # borda no meio do pixel
        out[n, tem] = first[tem]
    return out


def fill_nan_columns(S: np.ndarray) -> np.ndarray:
    """Interpola buracos em x e propaga bordas; NaN quebra qualquer derivada."""
    S = S.copy()
    for n in range(S.shape[0]):
        row = S[n]
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            row[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), row[ok])
        elif ok.any():
            row[~ok] = row[ok][0]
    return S


def hampel_mask(x: np.ndarray, k: int = 9, nsig: float = 3.0) -> np.ndarray:
    """
    Máscara booleana de outliers pelo filtro de Hampel: ponto a mais de `nsig`
    MADs da mediana móvel de janela `k`. Usa MAD, não desvio-padrão, porque o
    próprio outlier inflaria o desvio-padrão e se esconderia.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < k or k < 3:
        return np.zeros(n, bool)
    h = k // 2
    idx = np.clip(np.arange(n)[:, None] + np.arange(-h, h + 1)[None, :], 0, n - 1)
    jan = x[idx]
    med = np.nanmedian(jan, 1)
    mad = np.nanmedian(np.abs(jan - med[:, None]), 1) * 1.4826
    esc = np.maximum(mad, 1e-6)
    return np.abs(x - med) > nsig * esc


def frame_trust(S: np.ndarray, fps: float, mm_per_px: float = 1.0,
                rdot_max_mm_s: float | None = None,
                folga: float = 4.0) -> np.ndarray:
    """
    Marca quais frames são confiáveis, pelo salto da mediana da superfície
    entre quadros vizinhos.

    O teste é físico e não precisa de rótulo: entre dois quadros a superfície
    recua no máximo `rdot_max · dt`. No caso_testemari isso é ~2 px; um salto
    de 100 px é impossível e denuncia oclusão total pela chama. Um frame
    reprovado aqui **não deve entrar em nenhum ajuste** — nem depois de
    suavizado, porque não há informação a recuperar, só ruído a espalhar.

    `rdot_max_mm_s=None` auto-calibra em 5× a mediana do |Δ| observado.
    """
    S = np.asarray(S, float)
    med = np.nanmedian(S, 1)
    d = np.abs(np.diff(med))
    if rdot_max_mm_s is None:
        lim = folga * max(np.nanmedian(d), 1e-3)
    else:
        lim = folga * rdot_max_mm_s / (fps * mm_per_px)
    ruim = np.zeros(len(S), bool)
    ruim[1:] |= d > lim
    ruim[:-1] |= d > lim
    return ~ruim


def robust_monotone(S: np.ndarray, valid: np.ndarray | None = None,
                    med_k: int = 15, hampel_k: int = 9, nsig: float = 3.0,
                    max_descartado: float = 0.5) -> np.ndarray:
    """
    Impõe recessão monotônica de forma **robusta**: descarta → interpola → PAVA.

    A ordem não é detalhe. A projeção isotônica (PAVA) é de mínimos quadrados,
    portanto **não é robusta a outliers**: um único frame em que a chama oclui
    totalmente o grão põe a interface no lugar errado, e o PAVA *espalha* esse
    erro por todo o bloco vizinho ao promediá-lo. Medido no caso_testemari:
    aplicar PAVA direto sobre a série crua enviesava a superfície em −62 px
    (mediana) na fase 38–45 s, **em colunas cuja estimativa crua estava correta**.

    Etapas, por coluna:
      1. `valid=False` (oclusão detectada naquela coluna) → NaN;
      2. Hampel em t remove os outliers grosseiros restantes → NaN;
      3. interpolação linear em t sobre os NaN;
      4. **filtro de mediana** de janela `med_k` — este é o passo que segura os
         outliers *em sequência*, que o Hampel deixa passar porque a própria
         mediana móvel dele fica contaminada;
      5. PAVA sobre a série já limpa.

    `max_descartado` é um freio de segurança: se mais que essa fração da coluna
    caiu, não há sinal para reconstruir e a coluna sai como NaN em vez de virar
    uma interpolação inventada de ponta a ponta.

    ATENÇÃO — limite conhecido: nenhuma etapa aqui recupera um trecho em que a
    chama oclui o grão *de forma sustentada*. Use `frame_trust()` para reportar
    a janela válida e **descarte o resto do ajuste**, em vez de confiar no
    valor suavizado. Medido no caso_testemari: acima de ~33 s a oclusão passa
    de 10 % dos frames e a superfície final não é mais defensável.
    """
    from scipy.ndimage import median_filter

    S = np.array(S, float, copy=True)
    N, W = S.shape
    if valid is None:
        valid = np.ones_like(S, bool)

    for x in range(W):
        y = S[:, x].copy()
        y[~valid[:, x]] = np.nan
        ruim = hampel_mask(np.where(np.isnan(y), np.nanmedian(y), y), hampel_k, nsig)
        y[ruim] = np.nan

        ok = np.isfinite(y)
        if ok.sum() < 2 or (1.0 - ok.mean()) > max_descartado:
            S[:, x] = np.nan
            continue
        y = np.interp(np.arange(N), np.flatnonzero(ok), y[ok])
        if med_k >= 3 and N > med_k:
            y = median_filter(y, size=med_k | 1, mode="nearest")
        S[:, x] = isotonic_increasing(y)
    return S


def _odd_win(win: int, n: int) -> int:
    """Janela ímpar, >= 5 e <= n. Devolve 0 quando não cabe (sinal curto demais)."""
    w = min(int(win) | 1, (int(n) - 1) | 1)
    return w if w >= 5 else 0


def smooth_surface(S: np.ndarray, win_t: int = 21, win_x: int = 15,
                   poly: int = 2, monotonic: bool = True) -> np.ndarray:
    """
    Suaviza y_s(x,t) em x (ondas de superfície) e em t (ruído de rotulagem),
    depois impõe monotonicidade temporal. A ordem importa: monotonizar por
    último garante que a saída satisfaz a restrição física exatamente.
    """
    S = fill_nan_columns(np.asarray(S, np.float64))
    N, W = S.shape
    wx = _odd_win(win_x, W)
    if wx:
        S = savgol_filter(S, wx, min(poly, wx - 1), axis=1)
    wt = _odd_win(win_t, N)
    if wt:
        S = savgol_filter(S, wt, min(poly, wt - 1), axis=0)
    if monotonic and N > 2:
        for x in range(W):
            S[:, x] = isotonic_increasing(S[:, x])
    return S


# --------------------------------------------------------------------------- #
@dataclass
class RegressionResult:
    t: np.ndarray               # (N,)   tempo [s]
    x_mm: np.ndarray            # (W,)   coordenada axial [mm]
    y_mm: np.ndarray            # (N, W) recuo da superfície [mm], 0 no início
    rdot: np.ndarray            # (N, W) taxa de regressão local [mm/s]
    rdot_x: np.ndarray          # (W,)   r-dot médio no tempo, por posição
    rdot_t: np.ndarray          # (N,)   r-dot médio no espaço, por instante
    rdot_bar: float             # média espaço-temporal [mm/s]
    rdot_std: float             # dispersão espacial de rdot_x [mm/s]
    total_mm: float             # recuo total [mm]


def regression_rate(S: np.ndarray, fps: float, mm_per_px: float,
                    win_t: int = 21, win_x: int = 15, poly: int = 2,
                    trim: tuple[float, float] = (0.05, 0.05),
                    smooth: bool = True) -> RegressionResult:
    """
    S: (N, W) superfície em px (y cresce para baixo, logo dy/dt > 0 na queima).
    `trim` descarta frações do início/fim do domínio axial, onde o bordo de
    ataque e a saída sofrem efeitos de borda e não representam o grão.
    """
    S = smooth_surface(S, win_t, win_x, poly) if smooth else fill_nan_columns(S)
    N, W = S.shape
    t = np.arange(N) / fps

    a, b = int(trim[0] * W), W - int(trim[1] * W)
    a, b = max(0, a), max(a + 1, min(W, b))
    Sx = S[:, a:b]

    y_mm = (Sx - Sx[0][None, :]) * mm_per_px
    dt = 1.0 / fps
    wt = _odd_win(win_t, N)
    if wt:
        rdot = savgol_filter(Sx, wt, min(poly, wt - 1),
                             deriv=1, delta=dt, axis=0) * mm_per_px
    elif N >= 2:
        rdot = np.gradient(Sx, dt, axis=0) * mm_per_px
    else:
        rdot = np.zeros_like(Sx)
    rdot = np.clip(rdot, 0, None)                    # regressão não é negativa

    return RegressionResult(
        t=t, x_mm=np.arange(a, b) * mm_per_px, y_mm=y_mm, rdot=rdot,
        rdot_x=rdot.mean(0), rdot_t=rdot.mean(1),
        rdot_bar=float(rdot.mean()), rdot_std=float(rdot.mean(0).std()),
        total_mm=float(y_mm[-1].mean()),
    )


def rdot_by_linear_fit(S: np.ndarray, fps: float, mm_per_px: float,
                       trim: tuple[float, float] = (0.05, 0.05)):
    """
    Estimativa alternativa e mais conservadora: ajuste linear de y_s(t) por
    coluna (r-dot constante no ensaio). Reporte ambas — a diferença entre a
    versão instantânea e a do ajuste é uma medida honesta da incerteza do
    método, e evita a crítica clássica de "derivada de dado ruidoso".
    """
    S = fill_nan_columns(np.asarray(S, np.float64))
    N, W = S.shape
    a, b = int(trim[0] * W), W - int(trim[1] * W)
    Sx = S[:, max(0, a):min(W, max(a + 1, b))]
    t = np.arange(N) / fps

    tc = t - t.mean()
    den = float((tc ** 2).sum())
    slope = (tc[:, None] * (Sx - Sx.mean(0)[None, :])).sum(0) / den   # px/s
    pred = Sx.mean(0)[None, :] + slope[None, :] * tc[:, None]
    resid = Sx - pred
    dof = max(N - 2, 1)
    se = np.sqrt((resid ** 2).sum(0) / dof / den)                     # px/s

    return dict(
        rdot_x=slope * mm_per_px,
        rdot_bar=float(slope.mean() * mm_per_px),
        se_x=se * mm_per_px,
        se_bar=float(se.mean() * mm_per_px / np.sqrt(max(len(slope), 1))),
        r2=float(1 - (resid ** 2).sum() / max(((Sx - Sx.mean(0)) ** 2).sum(), 1e-12)),
    )


# --------------------------------------------------------------------------- #
def fit_power_law(G_ox: np.ndarray, rdot: np.ndarray):
    """
    Ajusta a lei clássica r = a·G_ox^n (log-log). Precisa de >= 2 ensaios com
    fluxos de oxidante diferentes; devolve a, n, R² e o erro-padrão de n.
    """
    G = np.asarray(G_ox, float); r = np.asarray(rdot, float)
    ok = (G > 0) & (r > 0)
    if ok.sum() < 2:
        raise ValueError("são necessários >= 2 pontos (G_ox, r-dot) positivos")
    lg, lr = np.log(G[ok]), np.log(r[ok])
    n, la = np.polyfit(lg, lr, 1)
    pred = n * lg + la
    ss_res = float(((lr - pred) ** 2).sum())
    ss_tot = float(((lr - lr.mean()) ** 2).sum())
    dof = max(ok.sum() - 2, 1)
    se_n = float(np.sqrt(ss_res / dof / max(((lg - lg.mean()) ** 2).sum(), 1e-12)))
    return dict(a=float(np.exp(la)), n=float(n), se_n=se_n,
                r2=float(1 - ss_res / max(ss_tot, 1e-12)), n_pts=int(ok.sum()))
