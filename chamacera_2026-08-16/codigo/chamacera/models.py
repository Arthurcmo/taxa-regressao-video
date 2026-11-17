"""
Modelo aluno: U-Net com Monte-Carlo Dropout.

Por que U-Net e não um modelo maior: com ~10² frames rotulados (o realista num
TG), um modelo de 1-2 M parâmetros com forte aumento de dados generaliza melhor
e treina em minutos numa GPU modesta. É também a arquitetura já validada nesta
aplicação (Surina et al., Acta Astronautica 2022; Rüttgers et al., 2024), o que
facilita comparar resultados.

Por que MC-Dropout: mantendo o dropout *ligado na inferência*, T passagens dão
uma distribuição preditiva. A entropia dessa média é o sinal que alimenta a
seleção ativa de frames — é o que fecha o laço de melhoria automática.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import N_CLASSES


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, p_drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = N_CLASSES,
                 base: int = 24, depth: int = 4, p_drop: float = 0.20):
        super().__init__()
        self.in_ch, self.n_classes, self.depth = in_ch, n_classes, depth
        chs = [base * 2 ** i for i in range(depth + 1)]

        self.down = nn.ModuleList()
        c = in_ch
        for i in range(depth):
            self.down.append(DoubleConv(c, chs[i], p_drop if i >= depth - 2 else 0.0))
            c = chs[i]
        self.bottleneck = DoubleConv(c, chs[depth], p_drop)

        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconv.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.up.append(DoubleConv(2 * chs[i], chs[i],
                                      p_drop if i >= depth - 2 else 0.0))
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for k, (uc, blk) in enumerate(zip(self.upconv, self.up)):
            x = uc(x)
            s = skips[-1 - k]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = blk(torch.cat([s, x], 1))
        return self.head(x)

    # -------------------------------------------------------------- #
    def enable_mc_dropout(self):
        """Deixa só o dropout em modo treino — BatchNorm continua em eval."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def predict_mc(self, x: torch.Tensor, T: int = 12):
        """
        Retorna (prob_media, entropia, incerteza_epistemica).
          entropia            = H[E_t p]        — incerteza total
          incerteza_epistemica= H[E p] - E H[p] — informação mútua (BALD),
                                a parcela que *mais dados resolvem*. É ela que
                                deve guiar a seleção ativa; ruído do sensor
                                (aleatório) inflaria a entropia total à toa.
        """
        self.enable_mc_dropout()
        probs = torch.stack([F.softmax(self(x), 1) for _ in range(T)])   # (T,B,C,H,W)
        p = probs.mean(0)
        ent = -(p * (p + 1e-9).log()).sum(1)
        exp_ent = -(probs * (probs + 1e-9).log()).sum(2).mean(0)
        return p, ent, (ent - exp_ent).clamp_min(0)


# --------------------------------------------------------------------------- #
def dice_loss(logits, target, weight=None, eps=1.0):
    """Dice suave multi-classe — compensa o desbalanceamento fundo/chama/cera."""
    p = F.softmax(logits, 1)
    t = F.one_hot(target.long(), p.shape[1]).permute(0, 3, 1, 2).float()
    if weight is not None:
        w = weight.unsqueeze(1)
        p, t = p * w, t * w
    dims = (0, 2, 3)
    num = 2 * (p * t).sum(dims) + eps
    den = p.sum(dims) + t.sum(dims) + eps
    return 1.0 - (num / den).mean()


def confidence_weighted_ce(logits, target, conf, class_w=None, floor=0.10):
    """
    CE ponderada pela confiança do pseudo-rótulo. Pixels perto da interface —
    justamente os que o bootstrap erra — pesam menos, então o aluno não aprende
    a copiar o erro sistemático do professor. `floor` evita que pixels difíceis
    sumam por completo do gradiente.
    """
    ce = F.cross_entropy(logits, target.long(), weight=class_w, reduction="none")
    w = conf.clamp_min(floor)
    return (ce * w).sum() / w.sum().clamp_min(1.0)
