"""
Região de validade sem rótulo: a rede reconhece quando está extrapolando.

POR QUE ISTO EXISTE, E POR QUE A SIGMA NÃO BASTA

A largura da distribuição por coluna (`surfnet.soft_argmax_col`) mede a
**ambiguidade da imagem**: quantas posições de y explicam este pixel. Ela sobe
com oclusão, e no caso da parafina isso funcionou — a σ epistêmica valeu 96 px
exatamente no quadro ruim e menos de 3 px em todos os outros.

No polietileno ela falhou por completo: o erro da parede de cima foi de 5,7 para
57,1 px e a σ ficou parada em 6 px. A diferença é o tipo de falha. Ali a imagem
não era ambígua — era **diferente** de tudo que a rede viu, e a rede respondeu com
convicção apoiada num atributo que naquele regime significa outra coisa.

Extrapolação não aparece na saída. Aparece nos **atributos**. Este módulo mede a
distância entre os atributos ao longo da interface prevista e os atributos que a
rede viu no treino.

POR SUPERFÍCIE, NÃO POR QUADRO

Um descritor do quadro inteiro não serve quando há mais de uma interface: no
mesmo quadro do polietileno a parede de baixo está bem medida e a de cima não.
Os atributos são amostrados **na linha prevista**, coluna a coluna, e reduzidos
por média e desvio ao longo de x. O descritor responde "a vizinhança desta linha
se parece com a vizinhança das linhas que eu aprendi?".

DUAS DISTÂNCIAS

`kNN` é o padrão deste módulo. Ele não supõe forma nenhuma para a distribuição de
treino, e aqui isso importa: a aparência da interface muda muito ao longo da
queima, então o treino não é um blob único e a gaussiana da Mahalanobis descreve
mal o conjunto. A Mahalanobis fica disponível para comparação, com encolhimento
na covariância — sem ele, com ~10² quadros para dimensão 48, a matriz fica mal
condicionada e a distância vira ruído amplificado.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .surfnet import soft_argmax_col


# --------------------------------------------------------------------------- #
@torch.no_grad()
def descritor_lote(net, x: torch.Tensor, janela: int | None = 15) -> np.ndarray:
    """
    x (B, C, H, W) -> (B, S, 2*base) — média e desvio dos atributos do decodificador
    lidos exatamente sobre cada interface prevista.
    """
    net.eval()
    f = F.relu(net.trunk(x))                       # (B, base, H, W)
    y, _ = soft_argmax_col(net.head_surf(f), janela=janela)      # (B, S, W)
    B, base, H, W = f.shape
    yy = y.clamp(0, H - 1).round().long()
    saida = []
    for b in range(B):
        linhas = []
        for s in range(y.shape[1]):
            idx = yy[b, s].view(1, 1, W).expand(base, 1, W)
            am = torch.gather(f[b], 1, idx).squeeze(1)           # (base, W)
            linhas.append(torch.cat([am.mean(1), am.std(1)]))
        saida.append(torch.stack(linhas))
    return torch.stack(saida).cpu().numpy()


# --------------------------------------------------------------------------- #
class DetectorValidade:
    """
    Ajustado nos descritores dos quadros que **tiveram professor** e usado em
    todos os outros. O limiar sai do percentil da própria referência, não de uma
    constante escolhida à mão — é o que permite levá-lo para um ensaio novo sem
    ninguém sintonizar nada.
    """

    def __init__(self, k: int = 5, percentil: float = 95.0):
        self.k, self.percentil = k, percentil
        self.mu = self.sd = self.ref = None
        self.limiar = None

    def ajustar(self, X: np.ndarray) -> "DetectorValidade":
        """X (n, d) — descritores da faixa em que a superfície teve professor."""
        if len(X) < self.k + 2:
            raise ValueError(f"referência com {len(X)} quadros é pequena demais "
                             f"para k = {self.k}")
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ref = (X - self.mu) / self.sd
        self.limiar = float(np.percentile(self.distancia(X), self.percentil))
        return self

    def distancia(self, X: np.ndarray) -> np.ndarray:
        A = (np.atleast_2d(X) - self.mu) / self.sd
        d2 = ((A[:, None, :] - self.ref[None, :, :]) ** 2).sum(-1)
        k = min(self.k, d2.shape[1] - 1)
        return np.sqrt(np.sort(d2, 1)[:, k])

    def valido(self, X: np.ndarray) -> np.ndarray:
        return self.distancia(X) <= self.limiar


# --------------------------------------------------------------------------- #
def mahalanobis(X_ref: np.ndarray, X: np.ndarray, alfa: float | None = None):
    """Distância de Mahalanobis com encolhimento para a diagonal."""
    mu = X_ref.mean(0)
    Z = X_ref - mu
    S = Z.T @ Z / max(len(X_ref) - 1, 1)
    d = S.shape[0]
    if alfa is None:
        alfa = float(np.clip(d / max(len(X_ref), 1), 0.05, 0.9))
    S = (1 - alfa) * S + alfa * np.trace(S) / d * np.eye(d)
    P = np.linalg.pinv(S)
    Zt = np.atleast_2d(X) - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Zt, P, Zt), 0)), alfa
