"""Regras únicas de aceitação, geometria inicial e exportação da medição de PE."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def mascara_chama(quente, frac=.05):
    q = np.asarray(quente, float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q < 0) or not 0 < frac < 1:
        raise ValueError("sinal de chama ou fração inválidos")
    mask = np.zeros(len(q), bool)
    hits = np.flatnonzero(q > frac * q.max())
    if len(hits):
        mask[hits[0]:hits[-1] + 1] = True
    return mask


def knn_distancias(X, R, k=5, ids=None, ref_ids=None):
    """k-ésimo vizinho; exclui apenas a MESMA identidade, não vetores iguais."""
    X, R = np.asarray(X, float), np.asarray(R, float)
    if X.ndim != 2 or R.ndim != 2 or X.shape[1] != R.shape[1] or len(R) <= k or k < 1:
        raise ValueError("referência insuficiente ou incompatível para kNN")
    if not np.isfinite(X).all() or not np.isfinite(R).all():
        raise ValueError("descritores não finitos")
    mu, sd = R.mean(0), R.std(0) + 1e-6
    A, B = (X - mu) / sd, (R - mu) / sd
    d2 = ((A[:, None] - B[None]) ** 2).sum(2)
    if ids is not None and ref_ids is not None:
        d2[np.asarray(ids)[:, None] == np.asarray(ref_ids)[None, :]] = np.inf
    return np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])


def particao_temporal(ids, k=5, embargo=3):
    """Calibração em dois blocos (20%); referência separada por embargo em frames.

    A divisão é determinística e não consulta erros de contorno. Os blocos 2 e
    7 de dez cobrem fases diferentes sem misturar quadros vizinhos nos conjuntos.
    Não há garantia conformal: observações do mesmo vídeo são correlacionadas.
    """
    ids = np.unique(np.asarray(ids, int))
    if len(ids) < 5 * (k + 1):
        raise ValueError("poucos quadros para separar referência e calibração")
    blocos = np.array_split(ids, 10)
    calib = np.concatenate([blocos[2], blocos[7]])
    ref = np.array([i for i in ids if np.min(np.abs(calib - i)) > embargo])
    if len(ref) <= k or len(calib) < 2:
        raise ValueError("partição temporal insuficiente após embargo")
    return ref, calib


def calibrar_validade(D, candidatos, quente, k=5, percentil=95, embargo=3, frac_chama=.05):
    D = np.asarray(D)
    if not 0 < percentil < 100:
        raise ValueError("percentil deve estar entre 0 e 100")
    n, surfaces, _ = D.shape
    dist, aprovado = np.empty((n, surfaces)), np.zeros((n, surfaces), bool)
    descricoes = []
    flame = mascara_chama(quente, frac_chama)
    for s in range(surfaces):
        ref, calib = particao_temporal(candidatos[s], k, embargo)
        d = knn_distancias(D[:, s], D[ref, s], k, np.arange(n), ref)
        lim = float(np.percentile(d[calib], percentil))
        dist[:, s], aprovado[:, s] = d, (d <= lim) & flame
        descricoes.append(dict(ref_ids=ref.tolist(), calibracao_ids=calib.tolist(),
                               limiar=lim, k=k, percentil=percentil, embargo_frames=embargo))
    return dist, aprovado, dict(faces=descricoes, frac_chama=frac_chama,
                               criterio="kNN calibrado em blocos separados; heurístico, não certificado")


def inicial_pre_ignicao(C, quente, frac=.05):
    """Estima bordas de uma faixa escura estável SOMENTE antes da primeira chama.

    É um pseudo-rótulo visual com incerteza, não uma posição física exata.
    Recusa baixo contraste, bandas na borda da busca e variação temporal excessiva.
    """
    import cv2
    flame = mascara_chama(quente, frac)
    ks = np.arange(np.flatnonzero(flame)[0]) if flame.any() else np.array([], dtype=int)
    h, w = C.shape[1:3]
    if len(ks) < 3:
        return np.full((2, w), np.nan), np.full((2, w), np.nan), np.zeros((2, w), bool), ks
    medidas = np.full((len(ks), 2, w), np.nan)
    a, b = int(.2 * h), int(.8 * h)
    for j, k in enumerate(ks):
        gray = np.asarray(C[k], float).mean(2)
        for x in range(w):
            perfil = cv2.GaussianBlur(gray[a:b, x].astype(np.float32)[:, None], (1, 5), 0)[:, 0]
            if np.percentile(perfil, 90) - np.percentile(perfil, 10) < 15:
                continue
            _, mask = cv2.threshold(np.clip(perfil, 0, 255).astype(np.uint8), 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            d = np.diff(np.r_[False, mask.ravel() > 0, False].astype(int))
            pares = [(l, r) for l, r in zip(np.where(d == 1)[0], np.where(d == -1)[0])
                     if l > 0 and r < b - a and .02 * h <= r - l <= .4 * h]
            if not pares:
                continue
            l, r = max(pares, key=lambda z: z[1] - z[0])
            medidas[j, :, x] = [a + l - .5, a + r - .5]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmedian(medidas, axis=0)
        sigma = 1.4826 * np.nanmedian(abs(medidas - y), axis=0) + .5
    ok = (np.isfinite(medidas).mean(0) >= .8) & np.isfinite(y) & (sigma <= 2)
    return y, sigma, ok, ks


def taxas_locais(y, t, aceito, janela_s=4, minimo=7):
    """Inclinação local e SE descritivo; nunca cruza uma lacuna de aceitação.

    SE vem dos resíduos da reta, não é incerteza total metrológica/calibrada.
    Exige janela temporal completa centrada no ponto, totalmente aceita.
    """
    y, t, ok = np.asarray(y, float), np.asarray(t, float), np.asarray(aceito, bool)
    if y.shape != t.shape or ok.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("série/tempo incompatíveis")
    if janela_s <= 0 or minimo < 3:
        raise ValueError("janela/minimo inválidos")
    rate, se = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in np.flatnonzero(ok):
        if t[i] - janela_s / 2 < t[0] or t[i] + janela_s / 2 > t[-1]:
            continue
        l = max(0, np.searchsorted(t, t[i] - janela_s / 2, side="right") - 1)
        r = min(len(t), np.searchsorted(t, t[i] + janela_s / 2, side="left") + 1)
        if r - l < minimo or not ok[l:r].all() or not np.isfinite(y[l:r]).all():
            continue
        x = t[l:r] - t[l:r].mean()
        v = y[l:r] - y[l:r].mean()
        slope = float(x @ v / (x @ x))
        rate[i] = slope
        se[i] = np.sqrt(np.sum((v - slope * x) ** 2) / (len(x) - 2) / (x @ x))
    return rate, se


def exportar_serie(out, Y, gate, t, qcam, col, sigma=None, janela_s=4, sufixo="_mono",
                   flame=None, mm_por_px=None, fps_confirmado=False):
    """Os campos y_sup/y_inf/a_px agora representam APENAS posições aceitas."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    y = np.nanmedian(Y[:, :, col], axis=2)
    gate = np.asarray(gate, bool)
    rows, taxas = [], []
    for s in range(2):
        taxas.append(taxas_locais((-1, 1)[s] * y[:, s], t, gate[:, s], janela_s))
    for k in range(len(t)):
        row = dict(quadro_camera=int(qcam[k]), t_s=float(t[k]))
        for s, nome in enumerate(("sup", "inf")):
            ok = bool(gate[k, s])
            row[f"y_{nome}_estimado"] = float(y[k, s])
            row[f"y_{nome}"] = float(y[k, s]) if ok else "NaN"
            row[f"portao_{nome}"] = int(ok)
            row[f"estado_{nome}"] = "aceito" if ok else "reconstruido_sem_medida"
            row[f"motivo_{nome}"] = ("" if ok else "fora_janela_chama" if flame is not None and not flame[k]
                                      else "fora_referencia_visual")
            row[f"sigma_modelo_{nome}_px"] = (float(np.nanmedian(sigma[k, s, col]))
                                               if ok and sigma is not None else "NaN")
            for key, values in (("rdot", taxas[s][0]), ("se_ajuste", taxas[s][1])):
                row[f"{key}_{nome}_px_s"] = float(values[k]) if np.isfinite(values[k]) else "NaN"
            if mm_por_px is not None and fps_confirmado:
                value = taxas[s][0][k] * mm_por_px
                row[f"rdot_{nome}_mm_s"] = float(value) if np.isfinite(value) else "NaN"
        row["a_px_estimado"] = float((y[k, 1] - y[k, 0]) / 2)
        row["a_px"] = row["a_px_estimado"] if gate[k].all() else "NaN"
        rows.append(row)
    with (out / f"serie{sufixo}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows
