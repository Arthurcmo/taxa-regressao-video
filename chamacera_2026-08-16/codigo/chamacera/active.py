"""
O laço que automatiza e melhora o trabalho.

Rotular vídeo de câmera rápida à mão é inviável: 5 s a 240 fps são 1200 frames,
e frames vizinhos são quase idênticos — rotular todos é caro *e* redundante.
A saída é escolher os poucos frames que de fato ensinam algo.

Quatro sinais de aquisição, complementares por construção:

  1. incerteza epistêmica (BALD)  — onde o modelo não sabe *e* mais dados
                                    resolveriam. Ignora ruído do sensor.
  2. inconsistência temporal      — a máscara "pula" entre frames vizinhos;
                                    como a física é contínua, salto = erro.
  3. violação física              — a superfície subiu (combustível "voltou").
                                    Impossível: é erro de rotulagem, garantido.
  4. discordância com o professor — aluno vs. SAM/bootstrap.

Os sinais 2 e 3 são *auto-supervisionados*: apontam erro sem nenhum rótulo de
referência. É o que permite o laço rodar sozinho entre as revisões humanas.

Depois de pontuar, a seleção é k-center greedy no espaço de descritores, para
não gastar as revisões humanas em 20 frames visualmente idênticos.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import CERA, CHAMA, PALETTE_BGR
from .surface import fill_nan_columns, surface_from_masks
from .theory import rate_disagreement


# --------------------------------------------------------------------------- #
# sinais de aquisição
# --------------------------------------------------------------------------- #
def score_uncertainty(epi: np.ndarray, masks: np.ndarray | None = None,
                      band_px: int = 12) -> np.ndarray:
    """
    Média da incerteza epistêmica. Se as máscaras forem dadas, pondera pela
    vizinhança da interface — é ali que o erro vira erro de r-dot; incerteza no
    meio do fundo escuro não interessa.
    """
    if masks is None:
        return epi.reshape(len(epi), -1).mean(1)
    out = np.zeros(len(epi))
    for n in range(len(epi)):
        cera = (masks[n] == CERA).astype(np.uint8)
        if cera.sum() == 0:
            out[n] = float(epi[n].mean()); continue
        borda = cv2.dilate(cera, np.ones((3, 3), np.uint8)) - \
            cv2.erode(cera, np.ones((3, 3), np.uint8))
        w = cv2.GaussianBlur(borda.astype(np.float32), (0, 0), band_px / 2.0)
        w = w / max(w.max(), 1e-6) + 0.15
        out[n] = float((epi[n] * w).sum() / w.sum())
    return out


def score_temporal(masks: np.ndarray) -> np.ndarray:
    """1 - IoU entre frames consecutivos (média das classes chama e cera)."""
    N = len(masks)
    out = np.zeros(N)
    for n in range(N):
        viz = [m for m in (n - 1, n + 1) if 0 <= m < N]
        if not viz:
            continue
        vals = []
        for m in viz:
            for k in (CHAMA, CERA):
                a, b = masks[n] == k, masks[m] == k
                u = (a | b).sum()
                if u:
                    vals.append(1.0 - (a & b).sum() / u)
        out[n] = float(np.mean(vals)) if vals else 0.0
    return out


def score_physics(surfaces: np.ndarray, tol_px: float | None = None,
                  k_sigma: float = 3.0) -> np.ndarray:
    """
    Fração de colunas em que a superfície *subiu* além da tolerância entre
    frames vizinhos. Violação estrita da irreversibilidade da queima.

    A tolerância é auto-calibrada por padrão, e isso importa: entre dois frames
    consecutivos a superfície recua tipicamente *menos de um pixel* (a 240 fps e
    1 mm/s, ~0.02 px). Uma tolerância fixa em 0.5 px marcaria quase todo frame
    como violação — o sinal satura e perde poder de discriminação. Usa-se então
    k·MAD do próprio incremento como escala de ruído, o que torna o detector
    invariante à taxa de aquisição e à calibração espacial.
    """
    S = fill_nan_columns(np.asarray(surfaces, float))
    N, W = S.shape
    out = np.zeros(N)
    if N < 2:
        return out
    d = np.diff(S, axis=0)                              # >0 = recuo correto
    if tol_px is None:
        mad = float(np.median(np.abs(d - np.median(d)))) * 1.4826
        tol_px = max(k_sigma * mad, 1e-3)
    viol = (d < -tol_px).mean(1)
    out[1:] += viol
    out[:-1] += viol
    return out / 2.0


def score_theory(surfaces: np.ndarray, fps: float, mm_per_px: float,
                 predicted_mm_s, sigma_model_mm_s) -> np.ndarray:
    """Discordância imagem-teoria na taxa, para priorizar revisão humana.

    A teoria entra como uma quinta fonte de *seleção*, nunca como rótulo. O
    score de cada frame k compara o recuo mediano entre k-1 e k com a lei
    prevista para aquele instante/posição. Para não amplificar ruído de vídeo
    rápido, a diferença é estimada na mediana das colunas.
    """
    S = fill_nan_columns(np.asarray(surfaces, float))
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(mm_per_px) or mm_per_px <= 0:
        raise ValueError("fps e mm_per_px devem ser positivos e finitos")
    obs = np.zeros(len(S), float)
    if len(S) > 1:
        obs[1:] = np.median(np.diff(S, axis=0), axis=1) * fps * mm_per_px
        obs[0] = obs[1]
    return rate_disagreement(obs, predicted_mm_s, sigma_model_mm_s)


def score_disagreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - IoU médio entre dois conjuntos de máscaras (aluno vs. professor)."""
    out = np.zeros(len(a))
    for n in range(len(a)):
        vals = []
        for k in (CHAMA, CERA):
            p, q = a[n] == k, b[n] == k
            u = (p | q).sum()
            if u:
                vals.append(1.0 - (p & q).sum() / u)
        out[n] = float(np.mean(vals)) if vals else 0.0
    return out


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    s = v.std()
    return np.zeros_like(v) if s < 1e-12 else (v - v.mean()) / s


def acquisition(epi=None, masks=None, surfaces=None, teacher_masks=None,
                fps=None, mm_per_px=None, theory_rate=None, theory_sigma=None,
                w=(1.0, 0.7, 1.2, 0.5, 0.8)) -> dict:
    """Combina os sinais em z-score. Peso maior na violação física: é o único
    sinal com garantia de que há erro de fato."""
    n = len(masks) if masks is not None else len(epi)
    parts = {}
    parts["incerteza"] = score_uncertainty(epi, masks) if epi is not None else np.zeros(n)
    parts["temporal"] = score_temporal(masks) if masks is not None else np.zeros(n)
    parts["fisica"] = score_physics(surfaces) if surfaces is not None else np.zeros(n)
    parts["discordancia"] = (score_disagreement(masks, teacher_masks)
                             if (masks is not None and teacher_masks is not None)
                             else np.zeros(n))
    usar_teoria = all(v is not None for v in (surfaces, fps, mm_per_px, theory_rate, theory_sigma))
    parts["teoria"] = (score_theory(surfaces, fps, mm_per_px, theory_rate, theory_sigma)
                        if usar_teoria else np.zeros(n))
    total = (w[0] * _z(parts["incerteza"]) + w[1] * _z(parts["temporal"]) +
             w[2] * _z(parts["fisica"]) + w[3] * _z(parts["discordancia"]) +
             w[4] * _z(parts["teoria"]))
    parts["total"] = total
    return parts


# --------------------------------------------------------------------------- #
# seleção diversa
# --------------------------------------------------------------------------- #
def descriptors(frames, dim: int = 16) -> np.ndarray:
    """Descritor barato: imagem cinza reduzida + histograma, normalizados."""
    D = []
    for img in frames:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (dim, dim), interpolation=cv2.INTER_AREA).ravel() / 255.0
        h = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel()
        h = h / max(h.sum(), 1)
        D.append(np.concatenate([small, h * 8]))
    D = np.asarray(D, np.float32)
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def select_diverse(scores: np.ndarray, desc: np.ndarray, k: int,
                   pool_frac: float = 0.35) -> list[int]:
    """
    k-center greedy restrito ao topo do ranking: primeiro filtra os `pool_frac`
    frames mais informativos, depois escolhe k que estejam o mais espalhados
    possível nesse conjunto. Evita gastar o orçamento de revisão em frames
    quase idênticos — que é o que a seleção puramente por score faz.
    """
    n = len(scores)
    k = min(k, n)
    pool = np.argsort(-np.asarray(scores))[:max(k, int(pool_frac * n))]
    if len(pool) <= k:
        return sorted(pool.tolist())
    P = desc[pool]
    sp = np.asarray(scores)[pool]
    chosen = [int(np.argmax(sp))]
    d = np.linalg.norm(P - P[chosen[0]], axis=1)
    d[chosen[0]] = -np.inf
    while len(chosen) < k:
        # empate em distância (descritores idênticos) é desempatado pelo score,
        # senão o argmax volta sempre ao mesmo índice e a seleção repete frames
        best = float(np.max(d))
        cand = np.flatnonzero(d >= best - 1e-9)
        i = int(cand[np.argmax(sp[cand])])
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
        d[chosen] = -np.inf
    return sorted(pool[chosen].tolist())


# --------------------------------------------------------------------------- #
# exportação para revisão humana
# --------------------------------------------------------------------------- #
def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
            surface: np.ndarray | None = None) -> np.ndarray:
    pal = np.array(PALETTE_BGR, np.uint8)
    col = pal[np.clip(mask, 0, len(pal) - 1)]
    out = cv2.addWeighted(img, 1 - alpha, col, alpha, 0)
    borda = cv2.Canny((mask * 100).astype(np.uint8), 50, 150)
    out[borda > 0] = (255, 255, 255)
    if surface is not None:
        for x, y in enumerate(surface):
            if np.isfinite(y) and 0 <= int(y) < out.shape[0]:
                out[int(round(y)), x] = (0, 0, 255)
    return out


@dataclass
class ReviewItem:
    id: str
    frame: int
    score: float
    incerteza: float
    temporal: float
    fisica: float
    discordancia: float
    teoria: float
    motivo: str


def export_review(out_dir: str, frames, masks, indices, parts: dict,
                  chosen: list[int], surfaces=None) -> list[ReviewItem]:
    """
    Escreve o pacote de revisão. Mantém `mask/` editável: o revisor corrige o
    PNG (qualquer editor com pincel, ou Label Studio / CVAT via tasks.json) e
    roda `chamacera apply-review`, que reinjeta a correção com confiança máxima.
    """
    for sub in ("img", "mask", "overlay"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    itens = []
    for j in chosen:
        fid = f"{indices[j]:06d}"
        cv2.imwrite(os.path.join(out_dir, "img", f"{fid}.png"), frames[j])
        cv2.imwrite(os.path.join(out_dir, "mask", f"{fid}.png"), masks[j])
        cv2.imwrite(os.path.join(out_dir, "overlay", f"{fid}.png"),
                    overlay(frames[j], masks[j],
                            surface=None if surfaces is None else surfaces[j]))
        motivos = {"incerteza": parts["incerteza"][j], "temporal": parts["temporal"][j],
                   "fisica": parts["fisica"][j], "discordancia": parts["discordancia"][j],
                   "teoria": parts["teoria"][j]}
        itens.append(ReviewItem(
            fid, int(indices[j]), float(parts["total"][j]),
            *[float(motivos[k]) for k in
              ("incerteza", "temporal", "fisica", "discordancia", "teoria")],
            max(motivos, key=motivos.get)))

    with open(os.path.join(out_dir, "review.json"), "w") as f:
        json.dump([asdict(i) for i in itens], f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "tasks.json"), "w") as f:      # Label Studio
        json.dump([{"data": {"image": f"img/{i.id}.png"},
                    "meta": {"frame": i.frame, "motivo": i.motivo}} for i in itens],
                  f, indent=2, ensure_ascii=False)
    return itens


def apply_review(review_dir: str, dataset_dir: str, conf_value: int = 255) -> int:
    """Move as máscaras revisadas para o dataset de treino com confiança 1.0."""
    for sub in ("img", "mask", "conf"):
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)
    n = 0
    mdir = os.path.join(review_dir, "mask")
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".png"):
            continue
        m = cv2.imread(os.path.join(mdir, fn), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(review_dir, "img", fn), cv2.IMREAD_COLOR)
        if m is None or img is None:
            continue
        cv2.imwrite(os.path.join(dataset_dir, "img", fn), img)
        cv2.imwrite(os.path.join(dataset_dir, "mask", fn), m)
        cv2.imwrite(os.path.join(dataset_dir, "conf", fn),
                    np.full(m.shape, conf_value, np.uint8))
        n += 1
    return n
