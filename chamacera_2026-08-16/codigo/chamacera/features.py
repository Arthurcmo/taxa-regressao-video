"""
Banco de atributos físicos por pixel.

Ideia central: chama e cera são separáveis **sem rótulo algum** por três
evidências independentes, todas baratas de calcular:

  1. Radiométrica  — a chama *emite* (auto-luminosa); a cera apenas *reflete*.
  2. Cromática     — radiação de fuligem é laranja-amarela (R > G > B); a cera
                     é acromática/creme. Índice (R-B)/(R+B) separa bem.
  3. Temporal      — a chama pisca em dezenas de Hz; a superfície de cera recua
                     ~mm/s, ou seja, é quase estática na janela de alguns ms.
                     O desvio-padrão temporal por pixel é o discriminador mais
                     forte e é imune a fuligem e vinhetagem (que são estáticas).

Generaliza a subtração de imagem final ("TLIS") de Surina et al. (2022) para uma
estatística temporal local, que continua válida mesmo com o transiente do ensaio.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-6


def luminance(bgr: np.ndarray) -> np.ndarray:
    """Luminância ITU-R BT.601 em [0,1] (mesma usada por Surina et al.)."""
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    return (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255.0


def flame_chroma(bgr: np.ndarray) -> np.ndarray:
    """
    Índice cromático de chama em [0,1]: (R-B)/(R+B).
    ~0.6-0.9 para radiação contínua de fuligem, ~0.0-0.1 para cera/fundo.
    Robusto a ganho da câmera porque é uma razão.
    """
    b = bgr[..., 0].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    return np.clip((r - b) / (r + b + EPS), 0.0, 1.0)


def emission(bgr: np.ndarray) -> np.ndarray:
    """
    Evidência de emissão: luminância ponderada pela cromaticidade, com um termo
    extra para o núcleo saturado (onde R≈G≈B e a cromaticidade colapsa).
    """
    L = luminance(bgr)
    C = flame_chroma(bgr)
    core = np.clip((L - 0.80) / 0.20, 0, 1)            # núcleo estourado
    return np.clip(L * (0.35 + 0.65 * C) + 0.55 * core, 0, 1.5)


def local_cv(bgr: np.ndarray, win: int = 9, piso: float = 0.10) -> np.ndarray:
    """
    Coeficiente de variação local da luminância: desvio-padrão numa janela
    win×win dividido pela média local.

    É o descritor de *textura invariante à iluminação*. Importa quando a chama
    ilumina o combustível (câmara de quartzo, chama colada à superfície): ali a
    cera herda a cor E o flicker da chama, e tanto a cromaticidade quanto o
    desvio-padrão temporal deixam de separar. O que continua separando é a
    estrutura espacial — a barra sólida é **lisa e sem textura**, a chama é
    **filamentosa**. Dividir pela média local remove o ganho imposto pela
    iluminação da própria chama.

    `piso` limita por baixo o denominador. Sem ele, região escura (barra antes
    da ignição, fundo) dividiria ruído de sensor por uma média quase nula e
    explodiria em "textura" — o modo de falha que enche a barra apagada de
    buracos. Abaixo do piso estamos no ruído do sensor e o desvio absoluto já é
    a medida certa.
    """
    g = luminance(bgr)
    k = (win, win)
    m = cv2.blur(g, k)
    v = np.maximum(cv2.blur(g * g, k) - m * m, 0.0)
    return np.sqrt(v) / np.maximum(m, piso)


# --------------------------------------------------------------------------- #
@dataclass
class TemporalStats:
    """Estatísticas temporais por pixel numa janela de frames."""
    med_L: np.ndarray      # mediana da luminância            (H, W)
    std_L: np.ndarray      # desvio-padrão robusto (MAD)      (H, W)
    med_C: np.ndarray      # mediana da cromaticidade         (H, W)
    med_E: np.ndarray      # mediana da emissão               (H, W)
    med_T: np.ndarray      # mediana da textura local (CV)    (H, W)


def temporal_stats(frames: np.ndarray, blur: float = 0.5,
                   win_tex: int = 9) -> TemporalStats:
    """
    frames: (T, H, W, 3) uint8 BGR. Usa MAD em vez de desvio-padrão amostral
    para não ser dominado por um único frame com faísca/partícula incandescente.
    """
    L = np.stack([luminance(f) for f in frames])
    C = np.stack([flame_chroma(f) for f in frames])
    E = np.stack([emission(f) for f in frames])
    T = np.stack([local_cv(f, win_tex) for f in frames])

    med_L = np.median(L, 0)
    mad = np.median(np.abs(L - med_L[None]), 0) * 1.4826      # ≈ sigma gaussiano
    med_T = np.median(T, 0)
    if blur > 0:
        k = dict(ksize=(0, 0), sigmaX=blur)
        med_L = cv2.GaussianBlur(med_L, **k)
        mad = cv2.GaussianBlur(mad, **k)
        med_T = cv2.GaussianBlur(med_T, **k)
    return TemporalStats(med_L, mad, np.median(C, 0), np.median(E, 0), med_T)


# --------------------------------------------------------------------------- #
# Perfis de ensaio. A escolha não é cosmética: define QUAIS evidências são
# confiáveis, e usar o perfil errado zera a segmentação (ver README).
#
#   "fuligem" — chama laranja de fuligem sobre grão que ela NÃO ilumina muito
#               (slab burner clássico). Cor e flicker temporal separam bem.
#   "quartzo" — chama fina, colada à superfície, que ILUMINA o combustível
#               (câmara de quartzo). A cera herda a cor e o flicker da chama,
#               então cromaticidade e desvio temporal deixam de separar; o que
#               separa é a TEXTURA (barra lisa × chama filamentosa).
PERFIS = {
    "fuligem": dict(w_chroma=1.0, w_tempo=1.0, w_textura=0.0, tex_ref=0.30),
    "quartzo": dict(w_chroma=0.0, w_tempo=0.45, w_textura=1.0, tex_ref=0.11),
}


def _mix(peso: float, evidencia: np.ndarray) -> np.ndarray:
    """Interpola entre "evidência ignorada" (1.0) e a evidência, pelo peso."""
    return 1.0 - peso + peso * evidencia


def opacity(med_L: np.ndarray, q: tuple[float, float] = (0.15, 0.90),
            span_min: float = 0.04) -> np.ndarray:
    """
    Evidência "isto é corpo opaco, não fundo", em [0,1].

    Normalizada pelos percentis robustos da *própria* imagem, nunca por
    constantes absolutas: o nível de cinza do grão depende de exposição,
    abertura, ganho e de quanto a chama o ilumina — e muda ordens de grandeza
    entre o frame antes da ignição e o frame em queima plena. Escala fixa
    funciona num ensaio e zera no seguinte.

    `span_min` protege o caso degenerado (quadro uniforme): sem faixa dinâmica
    real, amplificar ruído produziria "objeto" onde só há fundo.
    """
    lo, hi = np.quantile(med_L, q[0]), np.quantile(med_L, q[1])
    if hi - lo < span_min:
        return np.clip((med_L - lo) / span_min, 0, 1)
    return np.clip((med_L - lo) / (hi - lo), 0, 1)


def waxness(ts: TemporalStats, std_ref: float | None = None,
            chroma_ref: float = 0.35, perfil: str = "fuligem",
            **over) -> np.ndarray:
    """
    Mapa contínuo [0,1] de "isto é cera sólida" — produto de evidências
    independentes, com pesos definidos pelo perfil do ensaio.
    `std_ref` é auto-calibrado pelo percentil 75 do MAD se não informado.
    """
    p = {**PERFIS[perfil], **over}
    if std_ref is None:
        std_ref = max(float(np.quantile(ts.std_L, 0.75)), 1e-3)

    opaco = opacity(ts.med_L)
    estatico = _mix(p["w_tempo"], 1.0 - np.clip(ts.std_L / std_ref, 0, 1))
    acromatico = _mix(p["w_chroma"], 1.0 - np.clip(ts.med_C / chroma_ref, 0, 1))
    liso = _mix(p["w_textura"], 1.0 - np.clip(ts.med_T / p["tex_ref"], 0, 1))
    return opaco * estatico * acromatico * liso


def flameness(ts: TemporalStats, std_ref: float | None = None,
              perfil: str = "fuligem", **over) -> np.ndarray:
    """Mapa contínuo [0,1] de "isto é chama" (emissivo, flutuante, filamentoso)."""
    p = {**PERFIS[perfil], **over}
    if std_ref is None:
        std_ref = max(float(np.quantile(ts.std_L, 0.75)), 1e-3)

    emissivo = np.clip(ts.med_E / 0.45, 0, 1)
    flutua = np.clip(ts.std_L / std_ref, 0, 1)
    filamento = np.clip(ts.med_T / p["tex_ref"], 0, 1)
    wt, wx = p["w_tempo"], p["w_textura"]
    num = 0.65 * emissivo + 0.35 * (wt * flutua + wx * filamento) / max(wt + wx, EPS)
    return np.clip(num, 0, 1)


def feature_stack(bgr: np.ndarray, ts: TemporalStats) -> np.ndarray:
    """
    (H, W, 7) float32 — entrada opcional do aluno quando se quer um modelo
    leve e explicável em vez de RGB puro.
    Canais: L, C, E, med_L, std_L, waxness, y_normalizado.
    """
    H, W = bgr.shape[:2]
    yy = np.tile(np.linspace(0, 1, H, dtype=np.float32)[:, None], (1, W))
    return np.stack(
        [luminance(bgr), flame_chroma(bgr), emission(bgr).astype(np.float32),
         ts.med_L, ts.std_L / (ts.std_L.max() + EPS), waxness(ts), yy], -1
    ).astype(np.float32)
