"""Decodificação de contornos com sinal, passo máximo e barreiras explícitos.

A monotonicidade é uma hipótese para este enquadramento da combustão; movimento
da câmera, deformação ou oclusão podem invalidar sua interpretação física.
O caminho usa distribuições de imagem ponderadas. Peso zero retira a evidência
daquele quadro e produz reconstrução, sem certificar a posição do combustível.
Depois do refino subpixel e espacial, o chamador deve aplicar
`limitar_trajetoria` e verificar novamente os limites na unidade exportada.
"""

from __future__ import annotations

import numpy as np

NEG = -1e30


# --------------------------------------------------------------------------- #
def viterbi_monotono(L, sinal=+1, dmax=4, mu=0.05, w=None, progresso=None,
                      step_target=None, step_sigma=None):
    """
    L      (N, H, W) log-verossimilhança da linha y na coluna x do quadro k.
    sinal  +1 se y só pode crescer no tempo, −1 se só pode decrescer.
    dmax   passo máximo por quadro, em linhas da grade. Escalar, ou vetor (N,)
           quando o limite depende do quadro — é assim que "sem chama não há
           regressão" entra: fora da janela de combustão o passo permitido é
           zero e o trecho sai constante, sem ninguém estimar nada ali.
    mu     custo quadrático do passo (0 deixa a DP indiferente entre segurar e
           pular quando não há evidência; > 0 faz o trecho sem evidência sair
           como rampa, que é o que se quer).
    w      (N,) ou (N, W) confiança do quadro. Onde vale 0 a imagem não opina.
    step_target, step_sigma
           prior físico opcional do incremento positivo por quadro, na grade da
           rede. Ambos têm forma (N,), (N,W), ou escalar; o primeiro elemento é
           ignorado. O custo adicional é ``.5*((d-target)/sigma)**2``. É uma
           restrição *fraca*: a imagem ainda pode vencê-la. Use somente quando
           a escala e G_ox forem medidos; uma lei de taxa não é gabarito visual.

    Devolve idx (N, W) int32 com a linha escolhida.
    """
    L = np.asarray(L, np.float32)
    if L.ndim != 3 or min(L.shape) == 0 or not np.isfinite(L).all():
        raise ValueError("verossimilhança vazia, não finita ou com forma inválida")
    if sinal not in (-1, 1) or not np.isfinite(mu) or mu < 0:
        raise ValueError("sinal deve ser ±1 e mu não negativo")
    if sinal < 0:
        L = L[:, ::-1, :]
    N, H, W = L.shape

    if w is None:
        w = np.ones((N, W), np.float32)
    w = np.asarray(w, np.float32)
    if w.ndim == 1:
        w = np.repeat(w[:, None], W, 1)
    if w.shape != (N, W) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("pesos inválidos")

    def _passo(v, nome):
        if v is None:
            return None
        z = np.asarray(v, np.float32)
        if z.ndim == 0:
            z = np.full((N, W), float(z), np.float32)
        elif z.ndim == 1 and z.shape == (N,):
            z = np.repeat(z[:, None], W, 1)
        elif z.shape != (N, W):
            raise ValueError(f"{nome} deve ser escalar, (N,) ou (N,W)")
        if not np.isfinite(z).all() or np.any(z <= 0 if nome == "step_sigma" else z < 0):
            raise ValueError(f"{nome} inválido")
        return z

    alvo = _passo(step_target, "step_target")
    sigma = _passo(step_sigma, "step_sigma")
    if (alvo is None) != (sigma is None):
        raise ValueError("step_target e step_sigma devem ser informados juntos")

    raw_dk = np.broadcast_to(np.asarray(dmax), (N,))
    if not np.isfinite(raw_dk).all() or np.any(raw_dk < 0) or np.any(raw_dk != np.floor(raw_dk)):
        raise ValueError("dmax da DP deve ser inteiro não negativo")

    dk = (np.full(N, int(dmax), np.int32) if np.ndim(dmax) == 0
          else np.asarray(dmax, np.int32))
    dtop = int(dk.max())
    if dtop > 255:                       # o retrocesso e guardado em uint8
        raise ValueError(f"dmax = {dtop} nao cabe no ponteiro de retrocesso")
    pen = (mu * np.arange(dtop + 1, dtype=np.float32) ** 2)[:, None, None]
    bp = np.zeros((N, H, W), np.uint8)
    # Barreiras de geometria são duras, inclusive quando o peso da imagem é 0.
    D = np.where(L[0] <= NEG / 2, NEG, w[0][None, :] * L[0]).astype(np.float32)
    cand = np.empty((dtop + 1, H, W), np.float32)

    for k in range(1, N):
        cand[0] = D
        for d in range(1, dtop + 1):
            if d > dk[k]:
                cand[d] = NEG
                continue
            cand[d, :d] = NEG
            cand[d, d:] = D[:-d]
        cand -= pen
        if alvo is not None:
            # A física penaliza o PASSO, e não a coordenada absoluta. Assim ela
            # ajuda a atravessar oclusões sem puxar toda a trajetória para uma
            # borda teórica inexistente na imagem.
            ds = np.arange(dtop + 1, dtype=np.float32)[:, None, None]
            cand -= 0.5 * ((ds - alvo[k][None]) / sigma[k][None]) ** 2
        bp[k] = cand.argmax(0).astype(np.uint8)
        D = np.take_along_axis(cand, bp[k][None].astype(np.intp), 0)[0]
        D = np.where(L[k] <= NEG / 2, NEG, D + w[k][None, :] * L[k])
        if progresso is not None and k % 100 == 0:
            progresso(k, N)

    if np.any(D.max(0) <= NEG / 2):
        raise ValueError("não existe caminho compatível com as barreiras e o passo máximo")
    idx = np.zeros((N, W), np.int32)
    idx[N - 1] = D.argmax(0)
    xs = np.arange(W)
    for k in range(N - 1, 0, -1):
        idx[k - 1] = idx[k] - bp[k][idx[k], xs]

    if sinal < 0:
        idx = (H - 1) - idx
    return idx


# --------------------------------------------------------------------------- #
def refinar_subpixel(P, idx, sinal=+1, raio=4, corte=1.5, mediana=5,
                     w=None, dmax=None, inferior=0, superior=None):
    """
    Centroide da distribuição em ±`raio` linhas em torno do índice decodificado,
    com a correção cortada em ±`corte` e passada por mediana curta no tempo.

    Os três limites vêm do erro documentado no detector clássico: o refino sem
    limite pulava para o lobo vizinho e deslocava até 12 px entre quadros, seis
    vezes o que a própria DP permitia, e era ele — não o detector — a fonte de
    todos os picos da série. Cortar sem a mediana faz a correção oscilar entre os
    extremos do corte e o serrilhado volta.

    Com `dmax`, reconcilia sinal, passo e limites geométricos. Sem esse argumento
    mantém o comportamento legado, que garante somente monotonicidade. `w=0`
    impede a distribuição recusada de alterar o refino; interpola correções das
    observações aceitas. Isso não dá uma garantia de erro máximo do contorno.
    """
    P = np.asarray(P, np.float32)
    N, H, W = P.shape
    off = np.arange(-raio, raio + 1)
    lin = np.clip(idx[:, None, :] + off[None, :, None], 0, H - 1)

    p = np.take_along_axis(P, lin, 1)
    yy = lin.astype(np.float32)
    s = p.sum(1)
    corr = np.where(s > 1e-9, (p * yy).sum(1) / np.maximum(s, 1e-9) - idx, 0.0)
    corr = np.clip(corr, -corte, corte)
    if w is not None:
        pesos = np.asarray(w)
        if pesos.ndim == 1:
            pesos = pesos[:, None]
        ok = np.broadcast_to(pesos, (N, W)) > 0
        # Nenhuma distribuição recusada participa do refino; nas lacunas apenas
        # interpolamos correções observadas. A posição continua sendo a da DP.
        for x in range(W):
            ks = np.flatnonzero(ok[:, x])
            corr[:, x] = np.interp(np.arange(N), ks, corr[ks, x]) if len(ks) else 0

    if mediana and mediana > 1:
        r = mediana // 2
        pad = np.pad(corr, ((r, r), (0, 0)), mode="edge")
        corr = np.median(np.stack([pad[i:i + N] for i in range(2 * r + 1)]), 0)

    y = idx.astype(np.float32) + corr
    if dmax is not None:
        return limitar_trajetoria(y, sinal, dmax, inferior,
                                  H - 1 if superior is None else superior)
    return (np.maximum.accumulate(y, 0) if sinal > 0
            else -np.maximum.accumulate(-y, 0))


def limitar_trajetoria(y, sinal, dmax, inferior=0, superior=np.inf):
    """Reconciliação factível em O(NW), após TODO refino espacial/subpixel.

    Primeiro calcula intervalos que ainda permitem completar o caminho, de trás
    para frente. Depois escolhe a posição mais próxima da proposta em cada
    intervalo factível. Não é uma projeção global de mínimos quadrados: prioriza
    continuidade causal, sem prometer optimalidade estatística.
    Retorna float64 para preservar tolerâncias subpixel ao converter a ROI.
    """
    y = np.asarray(y, np.float64)
    if y.ndim != 2 or min(y.shape) == 0 or not np.isfinite(y).all() or sinal not in (-1, 1):
        raise ValueError("trajetória inválida")
    n, _ = y.shape
    dk = np.broadcast_to(np.asarray(dmax, np.float64), (n,))
    if not np.isfinite(dk).all() or np.any(dk < 0):
        raise ValueError("passo máximo deve ser finito e não negativo")
    lo = np.broadcast_to(np.asarray(inferior, float), y.shape).copy()
    hi = np.broadcast_to(np.asarray(superior, float), y.shape).copy()
    if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi):
        raise ValueError("limites de geometria incompatíveis")
    if sinal < 0:
        lo, hi = -hi, -lo
    for k in range(n - 2, -1, -1):
        lo[k] = np.maximum(lo[k], lo[k + 1] - dk[k + 1])
        hi[k] = np.minimum(hi[k], hi[k + 1])
    if np.any(lo > hi + 1e-10):
        raise ValueError("geometria sem trajetória factível")
    z = sinal * y.copy()
    z[0] = np.clip(z[0], lo[0], hi[0])
    for k in range(1, n):
        z[k] = np.clip(z[k], np.maximum(lo[k], z[k - 1]),
                       np.minimum(hi[k], z[k - 1] + dk[k]))
    return sinal * z


# --------------------------------------------------------------------------- #
def coerencia_x(y, grau=4, raio=8, iters=3):
    """
    Corrige coluna presa no vale errado, sem tocar na forma axial real.

    Ajuste polinomial robusto em x por quadro (IRLS com peso de Tukey), e depois
    a coluna é puxada para o ajuste só naquilo que exceder `raio`. É a segunda
    das três passadas que o detector clássico usa; a suavização local sozinha não
    resolve um BLOCO de colunas vizinhas erradas, porque a mediana mora dentro do
    bloco, e o ajuste global resolve porque o bloco é minoria.

    Este ajuste não preserva restrições temporais. O chamador deve reconciliar
    sinal, passo máximo, limites da ROI e separação das faces após esta etapa.
    """
    y = np.asarray(y, np.float32).copy()
    N, W = y.shape
    xs = np.linspace(-1, 1, W)
    V = np.vander(xs, grau + 1)
    for k in range(N):
        pk = np.ones(W, np.float32)
        for _ in range(iters):
            c, *_ = np.linalg.lstsq(V * pk[:, None], y[k] * pk, rcond=None)
            r = y[k] - V @ c
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            pk = (1 - np.clip(r / (4.685 * s), -1, 1) ** 2) ** 2
        aj = V @ c
        d = np.clip(y[k] - aj, -raio, raio)
        y[k] = aj + d
    return y
