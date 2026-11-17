"""
Rotulagem-semente sem supervisão ("zero-label bootstrap").

Produz máscaras 3-classes e um mapa de confiança a partir do vídeo cru, sem
nenhum frame rotulado à mão. Essas máscaras são o *pseudo-rótulo* inicial que
alimenta o professor (SAM) e o aluno (U-Net).

Restrição estrutural que torna o problema fácil: numa vista lateral de slab
burner, cada coluna x tem uma única interface sólido→gás. Logo, em vez de
segmentar 2D livremente, resolve-se um problema 1D — achar y_s(x) — por
programação dinâmica (Viterbi) sobre um custo casado. É a mesma família de
métodos usada para segmentar camadas de retina em OCT, e é muito mais robusta
a fuligem e a oclusão pela chama do que limiarização (Otsu) ou Canny.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import CERA, CHAMA, FUNDO
from .features import TemporalStats, flameness, temporal_stats, waxness

EPS = 1e-6


# --------------------------------------------------------------------------- #
# 1) interface sólido→gás por programação dinâmica
# --------------------------------------------------------------------------- #
def _matched_cost(W: np.ndarray, K: int) -> np.ndarray:
    """
    Custo de colocar a interface na linha y da coluna x.
    Resposta casada = média da cerosidade nas K linhas abaixo menos a das K
    linhas acima. Interface boa ⇒ resposta alta ⇒ custo baixo.
    """
    H, _ = W.shape
    cs = np.vstack([np.zeros((1, W.shape[1]), np.float32), np.cumsum(W, 0)])

    def band(a, b):                                   # média de W[a:b] por coluna
        a = np.clip(a, 0, H); b = np.clip(b, 0, H)
        n = np.maximum(b - a, 1)
        return (cs[b] - cs[a]) / n[:, None]

    y = np.arange(H)
    abaixo = band(y + 1, y + 1 + K)
    acima = band(y - K, y)
    return -(abaixo - acima).astype(np.float32)


def _viterbi(cost: np.ndarray, lam: float) -> np.ndarray:
    """
    Caminho de custo mínimo com penalidade L1 |Δy| entre colunas vizinhas.
    A min-convolução com |·| é feita em O(H) por coluna (duas varreduras),
    então o DP inteiro é O(H·W) — roda em ms mesmo em 4 Mpx.
    """
    H, Wd = cost.shape
    acc = cost[:, 0].copy()
    back = np.zeros((H, Wd), np.int32)
    idx = np.arange(H, dtype=np.int32)

    for x in range(1, Wd):
        m, am = acc.copy(), idx.copy()
        for y in range(1, H):                          # varredura para baixo
            if m[y - 1] + lam < m[y]:
                m[y], am[y] = m[y - 1] + lam, am[y - 1]
        for y in range(H - 2, -1, -1):                 # varredura para cima
            if m[y + 1] + lam < m[y]:
                m[y], am[y] = m[y + 1] + lam, am[y + 1]
        back[:, x] = am
        acc = m + cost[:, x]

    path = np.zeros(Wd, np.int32)
    path[-1] = int(np.argmin(acc))
    for x in range(Wd - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return path


def find_surface(W: np.ndarray, K: int = 9, lam: float = 0.06,
                 y_min: int = 0, y_max: int | None = None,
                 downsample: int = 1) -> np.ndarray:
    """
    y_s(x): linha da superfície de combustível, uma por coluna (float, px).
    `lam` controla a rigidez: alto = superfície lisa, baixo = segue as ondas.
    """
    H, Wd = W.shape
    y_max = H if y_max is None else y_max
    band = W[y_min:y_max]
    if downsample > 1:
        band = cv2.resize(band, (Wd // downsample, band.shape[0]),
                          interpolation=cv2.INTER_AREA)
    cost = _matched_cost(band, K)
    path = _viterbi(cost, lam).astype(np.float32)
    if downsample > 1:
        path = np.interp(np.linspace(0, len(path) - 1, Wd),
                         np.arange(len(path)), path)
    return path + y_min


def refine_subpixel(W: np.ndarray, ys: np.ndarray, K: int = 4) -> np.ndarray:
    """
    Refina y_s(x) para resolução sub-pixel pelo centroide do gradiente vertical
    da cerosidade numa vizinhança ±K. Importante: a taxa de regressão típica
    (~1 mm/s) desloca a interface por *frações* de pixel entre frames de câmera
    rápida, então sub-pixel não é luxo — é o que torna r-dot(t) mensurável.
    """
    H, Wd = W.shape
    gy = np.abs(np.gradient(W, axis=0))
    out = ys.astype(np.float64).copy()
    for x in range(Wd):
        c = int(round(ys[x]))
        a, b = max(0, c - K), min(H, c + K + 1)
        w = gy[a:b, x]
        s = w.sum()
        if s > EPS:
            out[x] = a + float((np.arange(b - a) * w).sum() / s)
    return out


# --------------------------------------------------------------------------- #
# 2) máscara de chama por histerese
# --------------------------------------------------------------------------- #
def hysteresis(score: np.ndarray, t_lo: float, t_hi: float,
               valid: np.ndarray | None = None) -> np.ndarray:
    """
    Duplo limiar com propagação por conectividade (mesma lógica do Canny).
    Captura a ponta fria/tênue da chama sem incorporar ruído de fundo, que é o
    modo de falha típico do Otsu simples em imagens de combustão.
    """
    strong = score >= t_hi
    weak = score >= t_lo
    if valid is not None:
        strong &= valid
        weak &= valid
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return strong
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def _otsu(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 16:
        return 0.5
    u8 = np.clip(v * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


# --------------------------------------------------------------------------- #
@dataclass
class BootstrapOut:
    mask: np.ndarray        # (H, W) uint8 — 0 fundo, 1 chama, 2 cera
    conf: np.ndarray        # (H, W) float32 [0,1] — confiança do pseudo-rótulo
    surface: np.ndarray     # (W,) float64 — y_s(x) sub-pixel
    waxness: np.ndarray
    flameness: np.ndarray


def bootstrap_frame(ts: TemporalStats, K: int = 9, lam: float = 0.15,
                    band: tuple[float, float] = (0.15, 0.98),
                    flame_level: float = 0.50,
                    flame_abs_min: float = 0.06,
                    min_area_frac: float = 2e-4,
                    perfil: str = "fuligem",
                    downsample: int = 1) -> BootstrapOut:
    """
    Rotula um instante do ensaio a partir das estatísticas temporais locais.

    `band` restringe a busca da interface a uma faixa vertical da ROI (fração da
    altura), evitando que a DP fuja para o topo do quadro em frames com chama
    muito extensa.

    `flame_level` fixa a *convenção* de borda da chama: a máscara é a isolinha
    a essa fração da emissão de pico, medida acima do nível de fundo. 0.50 é a
    meia-altura (FWHM), padrão em imageamento de chamas e o valor que melhor
    recupera a verdade-terreno no ensaio sintético. Não existe borda de chama
    "objetiva" — por isso ela é um parâmetro explícito, e por isso a etapa de
    revisão humana (`active`) é o que de fato ancora a convenção do trabalho.
    """
    Wmap = waxness(ts, perfil=perfil).astype(np.float32)
    Fmap = flameness(ts, perfil=perfil).astype(np.float32)
    H, Wd = Wmap.shape

    ys = find_surface(Wmap, K=K, lam=lam, downsample=downsample,
                      y_min=int(band[0] * H), y_max=int(band[1] * H))
    ys = refine_subpixel(Wmap, ys)

    yy = np.arange(H)[:, None]
    abaixo = yy >= ys[None, :]

    # ---- cera: abaixo da interface e opaca; maior componente conexa ---- #
    lo_cera = max(0.02, 0.45 * _otsu(Wmap[abaixo]))
    cera = abaixo & (Wmap > lo_cera)
    cera = cv2.morphologyEx(cera.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cera.astype(np.uint8), 8)
    if n > 1:
        cera = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))

    # ---- chama: isolinha de emissão normalizada, com histerese ---- #
    acima = ~abaixo
    E = ts.med_E.astype(np.float32)
    bg = float(np.quantile(E[acima], 0.25)) if acima.any() else 0.0
    pk = float(np.quantile(E[acima], 0.995)) if acima.any() else 1.0
    En = np.clip((E - bg) / max(pk - bg, EPS), 0, 2)
    t_hi, t_lo = flame_level, 0.60 * flame_level
    # A isolinha é RELATIVA ao pico do frame, então num frame sem chama alguma
    # (antes da ignição, depois do burnout) ela normalizaria ruído e inventaria
    # uma chama. O piso absoluto de contraste é o que impede isso.
    if pk - bg < flame_abs_min:
        chama = np.zeros_like(acima)
    else:
        chama = hysteresis(En, t_lo, t_hi, valid=acima)
    chama = cv2.morphologyEx(chama.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(chama.astype(np.uint8), 8)
    if n > 1:                                          # remove faíscas isoladas
        small = stats[:, cv2.CC_STAT_AREA] < min_area_frac * H * Wd
        small[0] = False
        chama &= ~small[lab]

    mask = np.full((H, Wd), FUNDO, np.uint8)
    mask[cera] = CERA
    mask[chama] = CHAMA

    # ---- confiança: margem à decisão, atenuada perto da interface ---- #
    margem = np.where(mask == CERA, Wmap,
                      np.where(mask == CHAMA, (En - t_lo) / max(t_hi - t_lo, EPS),
                               1.0 - np.maximum(Wmap, En)))
    dist = np.clip(np.abs(yy - ys[None, :]) / max(K, 1), 0, 1)
    conf = np.clip(margem, 0, 1).astype(np.float32) * (0.35 + 0.65 * dist)

    return BootstrapOut(mask, conf.astype(np.float32), ys, Wmap, Fmap)


# --------------------------------------------------------------------------- #
# 3) sequência: janela temporal deslizante + coerência física
# --------------------------------------------------------------------------- #
def isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """
    PAVA: projeção isotônica (não-decrescente). Aplicada em t sobre y_s(x,·),
    impõe a restrição física de que a superfície só pode recuar — combustível
    não "volta a crescer". Filtra grande parte do ruído de segmentação.
    """
    y = np.asarray(y, np.float64).copy()
    n = len(y)
    val = np.empty(n); wgt = np.empty(n); idx = np.empty(n, int)
    k = 0
    for i in range(n):
        val[k], wgt[k], idx[k] = y[i], 1.0, i
        while k > 0 and val[k - 1] > val[k]:
            w = wgt[k - 1] + wgt[k]
            val[k - 1] = (wgt[k - 1] * val[k - 1] + wgt[k] * val[k]) / w
            wgt[k - 1] = w
            k -= 1
        k += 1
    out = np.empty(n)
    j = n
    for b in range(k - 1, -1, -1):
        a = idx[b]
        out[a:j] = val[b]
        j = a
    return out


def bootstrap_sequence(src, indices, win: int = 9, stride_win: int = 1,
                       enforce_monotonic: bool = True, **kw):
    """
    Rotula uma lista de frames. Para cada frame alvo usa uma janela temporal
    centrada de `win` frames para as estatísticas — larga o suficiente para o
    flicker aparecer, curta o suficiente para a superfície ser ~estática.

    Retorna (masks, confs, surfaces) com shapes (N,H,W), (N,H,W), (N,W).
    """
    idx = list(indices)
    half = win // 2
    masks, confs, surfs = [], [], []
    for i in idx:
        lo = max(0, i - half * stride_win)
        hi = min(len(src), i + half * stride_win + 1)
        wnd = list(range(lo, hi, stride_win)) or [i]
        ts = temporal_stats(src.stack(wnd))
        out = bootstrap_frame(ts, **kw)
        masks.append(out.mask); confs.append(out.conf); surfs.append(out.surface)

    S = np.stack(surfs)                                # (N, W)
    if enforce_monotonic and len(idx) > 2:
        for x in range(S.shape[1]):
            S[:, x] = isotonic_increasing(S[:, x])
        for n in range(len(idx)):                      # reprojeta as máscaras
            yy = np.arange(masks[n].shape[0])[:, None]
            ab = yy >= S[n][None, :]
            m = masks[n]
            m[(m == CERA) & ~ab] = FUNDO
            m[(m == CHAMA) & ab] = FUNDO
    return np.stack(masks), np.stack(confs), S
