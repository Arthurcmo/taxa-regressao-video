"""
Professor opcional: SAM 2 / SAM 3 para refinar e propagar máscaras no vídeo.

Papel no pipeline: o bootstrap acerta *onde* estão chama e cera, mas sua borda
vem de um limiar; SAM tem um decodificador de máscara treinado em bordas e
costuma entregar contorno melhor, além de propagar a máscara por memória ao
longo do vídeo a partir de pouquíssimos prompts.

Duas ressalvas honestas, que devem constar da metodologia do TG:

  1. SAM foi treinado em imagens naturais. Para o grão de parafina (objeto
     opaco, bem delimitado) a transferência costuma ser boa; para a chama
     (objeto semitransparente, sem borda "objetiva") ela é incerta. Por isso
     este módulo é opcional e o pipeline mede o ganho em vez de supô-lo:
     `chamacera compare` reporta IoU bootstrap-vs-SAM contra os frames revisados.
  2. SAM 3 exige aceitar a licença e baixar `sam3.pt` (~3.45 GB) manualmente do
     Hugging Face; ~2.9 s/imagem em GPU. Inviável para 1200 frames, viável — e
     é assim que se usa aqui — como professor em algumas dezenas de frames-âncora,
     destilado depois no aluno U-Net (~ms/frame).

Instalação:
    SAM 2:  pip install "git+https://github.com/facebookresearch/sam2.git"
    SAM 3:  pip install -U ultralytics   +   sam3.pt no diretório de trabalho
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from . import CERA, CHAMA, FUNDO


# --------------------------------------------------------------------------- #
# geração automática de prompts a partir do bootstrap
# --------------------------------------------------------------------------- #
def sample_prompts(mask: np.ndarray, conf: np.ndarray, cls: int,
                   n_pos: int = 12, n_neg: int = 8,
                   erode: int = 5, seed: int = 0):
    """
    Amostra pontos positivos no *núcleo* da classe (após erosão, ponderando por
    confiança) e negativos no núcleo das outras classes. Erodir é essencial:
    prompts na borda do pseudo-rótulo herdariam justamente o erro que se quer
    corrigir.

    Retorna (points [K,2] em (x,y), labels [K] com 1=positivo, 0=negativo).
    """
    rng = np.random.default_rng(seed)
    ker = np.ones((erode, erode), np.uint8)

    def draw(m, k):
        core = cv2.erode(m.astype(np.uint8), ker, iterations=1).astype(bool)
        core = core if core.sum() >= k else m.astype(bool)
        ys, xs = np.nonzero(core)
        if len(xs) == 0:
            return np.zeros((0, 2), np.float32)
        w = np.clip(conf[ys, xs], 1e-3, None).astype(np.float64)
        sel = rng.choice(len(xs), size=min(k, len(xs)), replace=False, p=w / w.sum())
        return np.stack([xs[sel], ys[sel]], 1).astype(np.float32)

    pos = draw(mask == cls, n_pos)
    neg = draw((mask != cls) & (mask != FUNDO) if (mask != cls).any() else mask == FUNDO,
               n_neg // 2)
    neg2 = draw(mask == FUNDO, n_neg - len(neg))
    neg = np.concatenate([neg, neg2]) if len(neg2) else neg

    pts = np.concatenate([pos, neg]) if len(neg) else pos
    lab = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)])
    return pts, lab


def bbox_from_mask(mask: np.ndarray, cls: int, pad: int = 4):
    ys, xs = np.nonzero(mask == cls)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    return np.array([max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                     min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1)], np.float32)


# --------------------------------------------------------------------------- #
class SamTeacher:
    """
    Fachada única sobre os backends. `backend='auto'` tenta SAM2 e depois SAM3;
    se nenhum estiver instalado, `available` fica False e o pipeline segue só
    com o bootstrap (sem quebrar).
    """

    def __init__(self, backend: str = "auto", ckpt: str | None = None,
                 device: str | None = None, model_cfg: str | None = None):
        self.backend, self.available, self.info = None, False, ""
        self._pred = None
        self.device = device
        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.info = "PyTorch ausente"
            return

        order = ["sam2", "sam3"] if backend == "auto" else [backend]
        for b in order:
            try:
                getattr(self, f"_init_{b}")(ckpt, model_cfg)
                self.backend, self.available = b, True
                return
            except Exception as e:                       # backend indisponível
                self.info += f"[{b}] {type(e).__name__}: {e}\n"

    # ------------------------------------------------------------------ #
    def _init_sam2(self, ckpt, model_cfg):
        from sam2.build_sam import build_sam2_video_predictor   # noqa: F401
        self._ckpt = ckpt or "checkpoints/sam2.1_hiera_large.pt"
        self._cfg = model_cfg or "configs/sam2.1/sam2.1_hiera_l.yaml"
        if not os.path.exists(self._ckpt):
            raise FileNotFoundError(f"checkpoint SAM2 não encontrado: {self._ckpt}")
        self._build = build_sam2_video_predictor

    def _init_sam3(self, ckpt, model_cfg):
        from ultralytics.models.sam import SAM3VideoSemanticPredictor  # noqa: F401
        self._ckpt = ckpt or "sam3.pt"
        if not os.path.exists(self._ckpt):
            raise FileNotFoundError(
                "sam3.pt não encontrado — aceite a licença e baixe do Hugging Face")
        self._Sam3Video = SAM3VideoSemanticPredictor

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dump_jpegs(src, indices, tmp):
        for k, i in enumerate(indices):
            cv2.imwrite(os.path.join(tmp, f"{k:06d}.jpg"), src.frame(i),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        return tmp

    def propagate_sam2(self, src, indices, anchor_prompts: dict,
                       obj_ids=(CHAMA, CERA)) -> np.ndarray:
        """
        anchor_prompts: {idx_local: {classe: (points, labels)}}
        Retorna (N, H, W) uint8 com o rótulo propagado. Cera é escrita primeiro
        e a chama por cima — a chama é o objeto oclusor na vista lateral.
        """
        if not self.available or self.backend != "sam2":
            raise RuntimeError("backend SAM2 indisponível")
        import torch

        pred = self._build(self._cfg, self._ckpt, device=self.device)
        H, W = src.shape
        out = np.full((len(indices), H, W), FUNDO, np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            self._dump_jpegs(src, indices, tmp)
            with torch.inference_mode():
                state = pred.init_state(video_path=tmp)
                for f_local, per_class in anchor_prompts.items():
                    for cls, (pts, lab) in per_class.items():
                        if len(pts) == 0:
                            continue
                        pred.add_new_points_or_box(
                            inference_state=state, frame_idx=int(f_local),
                            obj_id=int(cls), points=pts, labels=lab)
                for f_idx, ids, logits in pred.propagate_in_video(state):
                    order = sorted(range(len(ids)), key=lambda k: ids[k] == CHAMA)
                    for k in order:
                        m = (logits[k] > 0).cpu().numpy().reshape(H, W)
                        out[f_idx][m] = int(ids[k])
        return out

    def propagate_sam3_text(self, video_path: str,
                            textos=("flame", "solid wax fuel slab"),
                            classes=(CHAMA, CERA), conf: float = 0.25,
                            imgsz: int = 640) -> np.ndarray:
        """
        Caminho de vocabulário aberto do SAM 3: prompt textual, sem clique
        nenhum. Útil como verificação independente do bootstrap — se ambos
        concordam sem compartilhar hipóteses, a confiança no rótulo sobe.
        """
        if not self.available or self.backend != "sam3":
            raise RuntimeError("backend SAM3 indisponível")
        p = self._Sam3Video(overrides=dict(conf=conf, task="segment", mode="predict",
                                           imgsz=imgsz, model=self._ckpt, save=False))
        masks = []
        for r in p(source=video_path, text=list(textos), stream=True):
            H, W = r.orig_shape
            m = np.full((H, W), FUNDO, np.uint8)
            if r.masks is not None:
                cl = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
                data = r.masks.data.cpu().numpy()
                for k in np.argsort([1 if c == 0 else 0 for c in cl]):
                    mk = cv2.resize(data[k].astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    m[mk] = classes[min(int(cl[k]), len(classes) - 1)]
            masks.append(m)
        return np.stack(masks) if masks else np.zeros((0, 1, 1), np.uint8)


# --------------------------------------------------------------------------- #
def anchors_from_bootstrap(masks, confs, n_anchors: int = 4, seed: int = 0) -> dict:
    """
    Escolhe frames-âncora bem espaçados no tempo e gera prompts automáticos
    neles. Resultado: o professor roda com zero cliques humanos — o bootstrap
    faz o papel do anotador na primeira volta.
    """
    N = len(masks)
    idx = np.unique(np.linspace(0, N - 1, min(n_anchors, N)).round().astype(int))
    out = {}
    for f in idx:
        per = {}
        for cls in (CHAMA, CERA):
            if (masks[f] == cls).sum() > 0:
                per[cls] = sample_prompts(masks[f], confs[f], cls, seed=seed + int(f))
        if per:
            out[int(f)] = per
    return out
