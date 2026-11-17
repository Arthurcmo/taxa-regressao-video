"""Métricas de segmentação e de superfície, para a seção de validação do TG."""

from __future__ import annotations

import cv2
import numpy as np

from . import CLASSES, N_CLASSES


def confusion(pred: np.ndarray, gt: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    m = (gt >= 0) & (gt < k)
    return np.bincount(k * gt[m].astype(int) + pred[m].astype(int),
                       minlength=k * k).reshape(k, k)


def from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    out = {"acc": float(tp.sum() / max(cm.sum(), 1)),
           "miou": float(np.nanmean(np.where(cm.sum(1) > 0, iou, np.nan)))}
    for c in range(len(iou)):
        out[f"iou_{CLASSES[c]}"] = float(iou[c])
        out[f"dice_{CLASSES[c]}"] = float(dice[c])
    return out


def seg_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    return from_confusion(confusion(np.asarray(pred), np.asarray(gt)))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, cls: int,
                tol_px: int = 2) -> float:
    """
    BF-score: F1 entre os contornos de `cls`, com tolerância de `tol_px`.
    Mais sensível que IoU ao que importa aqui — a *posição da interface* —
    porque IoU é dominado pelo interior maciço do grão.
    """
    def edges(m):
        m = (m == cls).astype(np.uint8)
        if m.sum() == 0:
            return None
        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        return (m - er).astype(bool)

    ep, eg = edges(np.asarray(pred)), edges(np.asarray(gt))
    if ep is None or eg is None:
        return float(ep is None and eg is None)
    dt_g = cv2.distanceTransform((~eg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((dt_g[ep] <= tol_px).mean())
    rec = float((dt_p[eg] <= tol_px).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def surface_error(pred_px: np.ndarray, gt_px: np.ndarray) -> dict:
    """
    Erro da linha de superfície. Reporta o viés separado do erro de forma: um
    viés constante some na derivada temporal (r-dot) e portanto é inofensivo
    para a grandeza de interesse; o RMS destendenciado, não.
    """
    e = np.asarray(pred_px, float) - np.asarray(gt_px, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"bias_px": np.nan, "mae_px": np.nan, "rms_detrend_px": np.nan}
    bias = float(np.median(e))
    return {"bias_px": bias, "mae_px": float(np.abs(e).mean()),
            "rms_detrend_px": float(np.sqrt(((e - bias) ** 2).mean())),
            "p95_abs_px": float(np.quantile(np.abs(e - bias), 0.95))}
