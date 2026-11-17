"""
Dados para o aluno de fronteira: alvo por coluna e canais de evidência física.

Duas diferenças em relação a `dataset.SlabDataset`, e as duas são deliberadas.

**1. O alvo é uma curva, não uma máscara.** Para cada quadro, `alvo[i, x]` é a
linha da superfície na coluna x, em pixel da ROI e com parte fracionária — o
professor já entrega subpixel, e arredondar para máscara jogava isso fora antes
mesmo do treino começar. `peso[i, x]` é a confiança do professor naquela coluna;
onde ele não sabe, o peso é ~0 e a coluna não ensina erro nenhum.

**2. A rede recebe as evidências físicas, não só RGB.** O aluno de máscara era
`UNet(3, ...)`: via um quadro RGB isolado e tinha de redescobrir sozinho, a
partir dele, o que o professor sabia olhando uma janela temporal e três
estatísticas. `features.feature_stack` já existia no pacote para isso e nunca
foi ligado a lugar nenhum. Aqui ele entra:

    canal 0-2  B, G, R
    canal 3    textura local (coeficiente de variação) — invariante à
               iluminação, é o que separa barra lisa de chama filamentosa
               quando a chama ilumina o próprio combustível
    canal 4    desvio temporal (MAD) — a chama pisca, a superfície não
    canal 5    emissão — emite × reflete

Isto muda o estatuto dos `PERFIS` ("fuligem" × "quartzo") de
`features.py`. Hoje eles são pesos escolhidos à mão por ensaio, e usar o perfil
errado zera a segmentação. Passando as evidências cruas como canais, a
combinação vira **peso aprendido**, e é o treino que decide quanto vale cada uma
naquele ensaio. É a diferença entre uma receita que se re-sintoniza a cada
ensaio e um medidor que transfere.

Textura e emissão são recalculadas do quadro **já aumentado**, não lidas do
disco: as duas são funções do quadro, e congelá-las ensinaria a rede que os
canais podem discordar entre si. O MAD temporal vem do disco porque é uma
estatística de janela, e é normalizado por um percentil global guardado em
`state.json` para que treino e inferência usem a mesma escala.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import aug_photometric
from .features import emission, local_cv

TEX_REF = 0.30          # escala do canal de textura (o mesmo tex_ref do perfil)


# --------------------------------------------------------------------------- #
def montar_canais(bgr: np.ndarray, mad: np.ndarray | None,
                  n_ch: int = 6) -> np.ndarray:
    """(H, W, n_ch) float32 em [0,1]. n_ch=3 devolve só BGR (ablação)."""
    x = bgr.astype(np.float32) / 255.0
    if n_ch <= 3:
        return x
    tex = np.clip(local_cv(bgr, 9) / TEX_REF, 0, 1).astype(np.float32)
    emi = np.clip(emission(bgr), 0, 1).astype(np.float32)
    m = (mad.astype(np.float32) / 255.0 if mad is not None
         else np.zeros(bgr.shape[:2], np.float32))
    return np.concatenate([x, tex[..., None], m[..., None], emi[..., None]], -1)


# --------------------------------------------------------------------------- #
def aug_geom_superficie(bgr, mad, y, w, rng, out_hw, escala=(0.80, 1.0)):
    """
    Recorte + espelhamento horizontal, com o alvo transformado junto.

    Sem rotação, ao contrário de `dataset.aug_geometric`: girar a imagem gira a
    superfície, e uma curva y(x) girada não é mais uma função de x sem
    reamostrar. O ganho não paga a chance de introduzir viés no próprio alvo.

    Colunas cuja superfície cai fora do recorte saem com peso zero — o recorte
    não pode inventar interface onde ela não está mais visível.
    """
    H, W = bgr.shape[:2]
    oh, ow = out_hw
    sc = rng.uniform(*escala)
    ch, cw = max(8, int(H * sc)), max(8, int(W * sc))
    y0 = int(rng.integers(0, H - ch + 1))
    x0 = int(rng.integers(0, W - cw + 1))

    bgr = bgr[y0:y0 + ch, x0:x0 + cw]
    mad = mad[y0:y0 + ch, x0:x0 + cw] if mad is not None else None
    y = y[:, x0:x0 + cw] - y0                       # (S, cw)
    w = w[:, x0:x0 + cw].copy()
    w[~np.isfinite(y) | (y < 0) | (y > ch - 1)] = 0.0

    if rng.random() < 0.5:
        bgr = bgr[:, ::-1]
        mad = mad[:, ::-1] if mad is not None else None
        y, w = y[:, ::-1], w[:, ::-1]

    bgr = cv2.resize(np.ascontiguousarray(bgr), (ow, oh), interpolation=cv2.INTER_LINEAR)
    if mad is not None:
        mad = cv2.resize(np.ascontiguousarray(mad), (ow, oh), interpolation=cv2.INTER_LINEAR)

    # o alvo vai para a grade da rede: escala vertical e reamostragem em x
    y = np.ascontiguousarray(y) * (oh / ch)
    w = np.ascontiguousarray(w)
    xs, xp = np.linspace(0, cw - 1, ow), np.arange(cw)
    y = np.stack([np.interp(xs, xp, np.nan_to_num(r, nan=0.0)) for r in y]).astype(np.float32)
    w = np.stack([np.interp(xs, xp, r) for r in w]).astype(np.float32)
    return bgr, mad, y, w


def _para_grade(y, w, ch, cw, oh, ow):
    """Versão sem aumento: só escala e reamostra (usada em validação/inferência)."""
    y = np.nan_to_num(np.asarray(y, np.float32), nan=0.0) * (oh / ch)
    w = np.asarray(w, np.float32)
    xs, xp = np.linspace(0, cw - 1, ow), np.arange(cw)
    return (np.stack([np.interp(xs, xp, r) for r in y]).astype(np.float32),
            np.stack([np.interp(xs, xp, r) for r in w]).astype(np.float32))


# --------------------------------------------------------------------------- #
class SurfDataset(Dataset):
    """
    Layout esperado (o mesmo de `chamacera bootstrap`, mais dois arrays):

        run/dataset/img/000042.png     quadro BGR na ROI
        run/dataset/mad/000042.png     desvio temporal já normalizado (opcional)
        run/dataset/mask/000042.png    máscara 3 classes (só p/ cabeça auxiliar)
        run/alvo.npy    (N_total, S, W)  superfícies por coluna, em px da ROI
        run/peso.npy    (N_total, S, W)  confiança por coluna, em [0,1]

    `S` é o número de interfaces por coluna: 1 na barra de parafina vista de
    lado, 2 na porta do polietileno (parede de cima e parede de baixo). Arrays
    2-D `(N, W)` são aceitos e promovidos a `S = 1`, para não quebrar os `run`
    que já existem.

    `alvo`/`peso` são indexados pelo **número original do quadro**, que é o
    próprio nome do arquivo. Assim o mesmo par de arrays serve para treino,
    validação e para o ensaio inteiro na inferência, sem tabela de conversão.
    """

    def __init__(self, root: str, alvo: np.ndarray, peso: np.ndarray,
                 ids: list[str], size=(256, 640), train: bool = True,
                 n_ch: int = 6, seed: int = 0, com_mascara: bool = True,
                 blocos: dict | None = None):
        """
        `blocos` liga cada id a um índice de bloco temporal. Quando ele existe, a
        transformação geométrica passa a ser sorteada por BLOCO e não por quadro,
        de modo que todos os quadros de um lote contíguo saem no mesmo recorte e
        na mesma escala. Sem isso o alvo de dois quadros vizinhos vive em
        sistemas de coordenadas diferentes e comparar y(k+1) com y(k) — que é o
        que `surfnet.temporal_loss` faz — não significa nada. O chamador deve
        atualizar `self.epoca` a cada época para que o sorteio continue mudando.
        """
        self.root, self.size, self.train = root, tuple(size), train
        self.blocos, self.epoca, self.semente = blocos, 0, seed
        alvo = alvo[:, None] if alvo.ndim == 2 else alvo
        peso = peso[:, None] if peso.ndim == 2 else peso
        self.alvo, self.peso, self.ids = alvo, peso, list(ids)
        self.n_surf = alvo.shape[1]
        self.n_ch, self.com_mascara = n_ch, com_mascara
        self.rng = np.random.default_rng(seed)
        if not self.ids:
            raise ValueError("lista de ids vazia")

    def __len__(self):
        return len(self.ids)

    def _ler(self, k):
        bgr = cv2.imread(os.path.join(self.root, "img", f"{k}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"{self.root}/img/{k}.png")
        p_mad = os.path.join(self.root, "mad", f"{k}.png")
        mad = cv2.imread(p_mad, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_mad) else None
        msk = None
        if self.com_mascara:
            p = os.path.join(self.root, "mask", f"{k}.png")
            msk = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
        return bgr, mad, msk

    def __getitem__(self, i):
        k = self.ids[i]
        j = int(k)
        bgr, mad, msk = self._ler(k)
        H, W = bgr.shape[:2]
        oh, ow = self.size
        y, w = self.alvo[j].copy(), self.peso[j].copy()

        if self.train:
            bgr = aug_photometric(bgr, self.rng)
            rng_g = (self.rng if self.blocos is None else np.random.default_rng(
                (self.semente, self.epoca, self.blocos.get(k, 0))))
            bgr, mad, y, w = aug_geom_superficie(bgr, mad, y, w, rng_g, self.size)
            if msk is not None:
                msk = cv2.resize(msk, (ow, oh), interpolation=cv2.INTER_NEAREST)
        else:
            bgr = cv2.resize(bgr, (ow, oh), interpolation=cv2.INTER_LINEAR)
            if mad is not None:
                mad = cv2.resize(mad, (ow, oh), interpolation=cv2.INTER_LINEAR)
            if msk is not None:
                msk = cv2.resize(msk, (ow, oh), interpolation=cv2.INTER_NEAREST)
            y, w = _para_grade(y, w, H, W, oh, ow)

        x = montar_canais(bgr, mad, self.n_ch)
        out = [torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float(),
               torch.from_numpy(np.ascontiguousarray(y)).float(),   # (S, W)
               torch.from_numpy(np.ascontiguousarray(w)).float()]
        out.append(torch.from_numpy(msk.astype(np.int64)) if msk is not None
                   else torch.zeros(self.size, dtype=torch.long))
        return tuple(out)


# --------------------------------------------------------------------------- #
def erro_px(y_grade: np.ndarray, y_ref: np.ndarray, altura_roi: int,
            altura_grade: int) -> np.ndarray:
    """Converte erro da grade da rede de volta para pixel da ROI."""
    return (np.asarray(y_grade) - np.asarray(y_ref)) * (altura_roi / altura_grade)
