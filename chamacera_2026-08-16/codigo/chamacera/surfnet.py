"""
Aluno de *fronteira*: a rede devolve a superfície, não uma máscara.

Motivação — o que estava errado no aluno de máscara
---------------------------------------------------
O aluno anterior é um segmentador de três classes, e a superfície sai depois,
como borda superior da máscara de cera. Isso desalinha o treino da medida em
três pontos, todos verificáveis nos números deste caso:

1. **A perda olha para o lugar errado.** Entropia cruzada e Dice somam sobre a
   área das regiões. O interior da barra e o fundo dominam o gradiente, e a
   fronteira — que é a medida inteira — entra com peso proporcional ao seu
   perímetro, ou seja, quase nada. Daí IoU de cera 0,955 conviver com 11 px de
   erro na interface: as duas grandezas quase não se falam.

2. **A saída é mais grossa que o sinal.** Com ROI de 715 px de altura reduzida
   para 256, cada célula da rede vale 2,79 px de ROI. O recuo entre quadros
   amostrados é 1,95 px. Um `argmax` sobre a máscara nunca resolve abaixo de uma
   célula, logo o passo físico fica *abaixo* da resolução da própria saída.

3. **A seleção de modelo premia a coisa errada.** `best.pt` é escolhido por
   mIoU, isto é, o checkpoint que melhor colore região, não o que melhor localiza
   fronteira.

O que este módulo faz
---------------------
Troca a cabeça e a perda, mantendo o mesmo corpo U-Net já validado, para que a
comparação seja uma ablação limpa: muda a formulação, não a capacidade.

A cabeça devolve, para cada coluna x, uma **distribuição sobre as linhas y**
(softmax na vertical), e a posição da superfície é a esperança dessa
distribuição, `ŷ(x) = Σ_y y·p(y|x)` — o *soft-argmax*. Três consequências:

- **subpixel de graça.** A esperança é contínua: mesmo numa grade de 256 linhas,
  ŷ assume valores entre células, e a precisão deixa de ser limitada pelo passo
  da grade. É o que resolve o ponto 2 acima.
- **a estrutura do problema entra na arquitetura.** "Cada coluna tem uma única
  interface sólido→gás" deixa de ser um pós-processamento (o Viterbi do
  professor) e passa a ser a forma da saída. A rede não consegue devolver duas
  superfícies numa coluna nem nenhuma.
- **a incerteza sai em pixels.** O desvio-padrão da mesma distribuição,
  `σ(x) = sqrt(Σ_y p(y|x)·(y−ŷ)²)`, é uma barra de erro na *unidade da medida*,
  num único passe adiante. O MC-Dropout continua disponível e passa a medir
  outra coisa (dispersão entre passes, epistêmica), com as duas separadas em vez
  de fundidas numa entropia em nats.

Precedente: é a formulação de He et al., "Fully Convolutional Boundary
Regression for Retina OCT Segmentation" (MICCAI 2019) e "Structured layer
surface segmentation for retina OCT using fully convolutional regression
networks" (Medical Image Analysis 68, 2021) — mesma família do Viterbi/graph
search que o professor já usa para camadas de retina, agora dentro da rede e
diferenciável ponta a ponta.

`n_surf` generaliza para o caso do polietileno, em que a porta tem duas
interfaces por coluna (`n_surf=2`): a ordenação sai garantida pelo termo de
ordem, sem regra extra.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import N_CLASSES
from .models import UNet, dice_loss

EPS = 1e-9


# --------------------------------------------------------------------------- #
class SurfaceNet(nn.Module):
    """
    Corpo U-Net (o mesmo de `models.UNet`, com MC-Dropout) e duas cabeças:

      - `surf`: (B, n_surf, H, W) — logits que viram distribuição por coluna;
      - `seg` : (B, 3, H, W)      — segmentação auxiliar, opcional.

    A cabeça de segmentação é mantida porque é barata (uma convolução 1×1) e
    porque as duas tarefas se ajudam: a de região dá gradiente denso no começo
    do treino, quando a de fronteira ainda não tem onde se apoiar. É o mesmo
    arranjo de duas saídas de He et al. (2019). Pode ser desligada para isolar
    o efeito da fronteira.
    """

    def __init__(self, in_ch: int = 3, n_surf: int = 1, n_classes: int = N_CLASSES,
                 base: int = 24, depth: int = 4, p_drop: float = 0.20,
                 seg_head: bool = True):
        super().__init__()
        self.in_ch, self.n_surf, self.n_classes = in_ch, n_surf, n_classes
        # o corpo devolve `base` canais de atributo; a "cabeça" da U-Net vira
        # a última camada de refino, e as cabeças de tarefa vêm depois dela.
        self.trunk = UNet(in_ch, n_classes=base, base=base, depth=depth,
                          p_drop=p_drop)
        self.head_surf = nn.Conv2d(base, n_surf, 1)
        self.head_seg = nn.Conv2d(base, n_classes, 1) if seg_head else None

    def forward(self, x):
        f = F.relu(self.trunk(x), inplace=True)
        seg = self.head_seg(f) if self.head_seg is not None else None
        return self.head_surf(f), seg

    # -- inferência ---------------------------------------------------------- #
    def enable_mc_dropout(self):
        """Só o dropout volta a treino; BatchNorm continua em eval."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def predict_surface(self, x: torch.Tensor, T: int = 1,
                        janela: int | None = None):
        """
        Devolve (y, sigma_aleat, sigma_epist, seg_prob), tudo em unidades da
        grade da rede:

          y            (B, S, W)  posição da superfície, subpixel
          sigma_aleat  (B, S, W)  largura da distribuição por coluna — quão
                                  borrada está a fronteira *neste* passe
          sigma_epist  (B, S, W)  desvio entre os T passes de MC-Dropout — o
                                  quanto o modelo discorda de si mesmo, que é a
                                  parcela que mais dados resolvem
          seg_prob     (B, C, H, W) ou None

        Separar as duas incertezas importa para a medida: a primeira cresce
        quando a chama borra a interface (limite de informação no pixel), a
        segunda cresce em regime que a rede não viu (limite de treino). Só a
        segunda melhora rotulando mais quadros — e é ela que deve guiar a
        seleção ativa.
        """
        if T <= 1:
            self.eval()
            logits, seg = self(x)
            y, sd = soft_argmax_col(logits, janela=janela)
            p = F.softmax(seg, 1) if seg is not None else None
            return y, sd, torch.zeros_like(y), p

        self.enable_mc_dropout()
        ys, sds, segs = [], [], []
        for _ in range(T):
            logits, seg = self(x)
            y, sd = soft_argmax_col(logits, janela=janela)
            ys.append(y); sds.append(sd)
            if seg is not None:
                segs.append(F.softmax(seg, 1))
        Y = torch.stack(ys)
        return (Y.mean(0), torch.stack(sds).mean(0), Y.std(0, unbiased=False),
                torch.stack(segs).mean(0) if segs else None)


# --------------------------------------------------------------------------- #
# soft-argmax e alvo estruturado
# --------------------------------------------------------------------------- #
def soft_argmax_col(logits: torch.Tensor, tau: float = 1.0,
                    janela: int | None = None):
    """
    logits (B, S, H, W) -> (y, sigma), ambos (B, S, W).

    A softmax é sobre a dimensão H **dentro de cada coluna**: é aí que a
    restrição "uma interface por coluna" entra na arquitetura. `tau` < 1 aperta
    a distribuição; deixamos 1,0 porque a largura é usada como incerteza e
    apertar por fora a descalibraria.

    `janela` — a esperança é calculada só dentro de ±janela células da **moda**
    da coluna, e não sobre a coluna inteira.

    Isto não é detalhe de implementação, é o conserto do modo de falha medido em
    `ANALISE_FRONTEIRA_2026-08-30.md`. A esperança global assume que a
    distribuição é unimodal. No regime ocluído ela não é: aparecem dois picos, um
    na camada luminosa colada à superfície e outro na frente de chama acima, e a
    média cai **no vale entre os dois**, que não é superfície nenhuma. Foi o que
    puxou a previsão para 374 px onde a leitura manual dava 460. Ancorar na moda
    e integrar só a vizinhança mantém o subpixel (a esperança continua contínua)
    e devolve a robustez do argmax à multimodalidade.

    Com `janela=None` o comportamento é o da esperança global, mantido para
    reproduzir a ablação.
    """
    p = F.softmax(logits / tau, dim=2)
    H = logits.shape[2]
    ys = torch.arange(H, device=logits.device, dtype=p.dtype).view(1, 1, H, 1)
    if janela is not None and janela > 0:
        moda = p.argmax(dim=2, keepdim=True).to(p.dtype)
        p = p * ((ys - moda).abs() <= janela).to(p.dtype)
        p = p / p.sum(2, keepdim=True).clamp_min(EPS)
    y = (p * ys).sum(2)
    var = (p * (ys - y.unsqueeze(2)) ** 2).sum(2)
    return y, var.clamp_min(0).sqrt()


def gaussian_column_target(y: torch.Tensor, H: int, sigma: float = 3.0):
    """
    Alvo (B, S, H, W): gaussiana centrada na superfície verdadeira, normalizada
    por coluna.

    Supervisionar a *distribuição* e não só a esperança é o que impede a solução
    degenerada em que a rede acerta a média com uma distribuição bimodal — dois
    picos, um na chama e outro na barra, cuja média cai no meio e não é
    superfície nenhuma. Com o alvo gaussiano, a largura da saída passa a
    significar incerteza, que é o que permite usá-la como barra de erro.
    """
    ys = torch.arange(H, device=y.device, dtype=y.dtype).view(1, 1, H, 1)
    q = torch.exp(-0.5 * ((ys - y.unsqueeze(2)) / sigma) ** 2)
    return q / q.sum(2, keepdim=True).clamp_min(EPS)


# --------------------------------------------------------------------------- #
# perdas
# --------------------------------------------------------------------------- #
def surface_loss(logits, y_true, w, sigma: float = 3.0,
                 w_pos: float = 1.0, w_dist: float = 1.0, beta: float = 2.0):
    """
    Perda de fronteira, ponderada por coluna.

      L_pos   Huber(ŷ, y)  — erro na unidade da medida. `beta` em células da
                             grade: abaixo dele é quadrático (afinação fina),
                             acima é linear (não deixa uma coluna ocluída
                             dominar o gradiente).
      L_dist  −Σ q log p    — alinha a *forma* da distribuição ao alvo gaussiano.

    `w` (B, S, W) é a confiança do pseudo-rótulo por coluna. Colunas em que o
    professor não sabe entram com peso ~0 e não ensinam erro — é o análogo, na
    fronteira, do `confidence_weighted_ce` do aluno de máscara.
    """
    H = logits.shape[2]
    y_hat, _ = soft_argmax_col(logits)
    pos = F.smooth_l1_loss(y_hat, y_true, beta=beta, reduction="none")

    logp = F.log_softmax(logits, dim=2)
    q = gaussian_column_target(y_true, H, sigma)
    dist = -(q * logp).sum(2)

    den = w.sum().clamp_min(1.0)
    return (w_pos * (w * pos).sum() + w_dist * (w * dist).sum()) / den


def order_loss(y_hat: torch.Tensor, gap: float = 1.0):
    """
    Penaliza superfícies fora de ordem quando `n_surf > 1` (caso da porta do
    polietileno: a interface de cima tem de ficar acima da de baixo em toda
    coluna). Sem isto as duas cabeças podem trocar de papel no meio do grão e a
    série temporal fica com um degrau que não é física.
    """
    if y_hat.shape[1] < 2:
        return y_hat.sum() * 0.0
    d = y_hat[:, 1:] - y_hat[:, :-1]
    return F.relu(gap - d).mean()


def temporal_loss(y_hat: torch.Tensor, passo_max: float,
                  sinal: torch.Tensor | None = None,
                  dk: torch.Tensor | None = None,
                  pares: bool = False):
    """
    Penaliza recuo incompatível com a física entre quadros vizinhos de um lote
    ordenado no tempo. `passo_max` em células da grade **por quadro de
    amostragem**.

    `sinal` (S,) diz para que lado cada superfície pode andar: na porta do
    polietileno a parede de cima só sobe e a de baixo só desce, então uma regra
    única "y cresce" penalizaria a face de cima por fazer exatamente o certo. Ele
    sai do próprio alvo, pela inclinação da tendência de cada superfície, e não
    de uma constante escrita à mão.

    `dk` (B−1,) é o intervalo em quadros entre elementos vizinhos do lote. Sem
    ele um lote que atravessa um buraco na amostragem — os 47 quadros entre a
    primeira chama e o início da janela do professor, por exemplo — seria punido
    por um passo grande que é legítimo. O limite superior escala com o intervalo;
    o limite inferior, o de não voltar, vale para qualquer intervalo.

    `pares` diz que o lote é uma lista de PARES vizinhos — (a0, a1, b0, b1, ...)
    — em vez de um bloco contíguo único. É o arranjo que se deve usar: um lote
    inteiro contíguo faz a BatchNorm ver quatro quadros quase idênticos, as
    estatísticas de lote degeneram e o treino não sai do lugar (medido: 54 px de
    erro de validação contra 1,3 px do mesmo treino sem o termo). Com pares
    vindos de instantes diferentes a BatchNorm continua vendo variedade e o
    termo temporal continua tendo vizinhos de verdade para comparar.

    Só faz sentido com quadros vizinhos e com a MESMA transformação geométrica
    dentro de cada par; fora disso o chamador deve deixar o peso em zero.
    Aqui a monotonicidade entra como *pressão*, nunca como projeção dura — a
    projeção isotônica ingênua foi justamente o que enviesou a série em −61,6 px
    no regime ocluído (ver LEIA-ME). A garantia dura vem depois, na
    decodificação de `monotono.py`.
    """
    if y_hat.shape[0] < 2:
        return y_hat.sum() * 0.0
    d = (y_hat[1::2] - y_hat[0::2]) if pares else (y_hat[1:] - y_hat[:-1])
    if sinal is not None:
        d = d * sinal.view(1, -1, 1).to(d.dtype)
    lim = (passo_max if dk is None
           else passo_max * dk.view(-1, 1, 1).clamp(min=1).to(d.dtype))
    return (F.relu(-d).mean() + F.relu(d - lim).mean())


def seg_loss(logits, target, conf, class_w=None, w_dice: float = 0.5):
    """Perda auxiliar de região — a mesma do aluno de máscara, para comparar."""
    ce = F.cross_entropy(logits, target.long(), weight=class_w, reduction="none")
    w = conf.clamp_min(0.10)
    return (ce * w).sum() / w.sum().clamp_min(1.0) + w_dice * dice_loss(logits, target, conf)
