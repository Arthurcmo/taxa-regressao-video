"""
Deteccao das duas interfaces solido->gas (o "traco preto") em cada coluna.

Geometria deste ensaio: porta horizontal no meio de um grao de polietileno
translucido, filmado de lado. A chama fica DENTRO da porta; o combustivel
esta acima e abaixo dela. Ha portanto DUAS superficies, que regridem em
sentidos opostos, e a grandeza medida e a meia-altura da porta

    a(t) = ( y_inf(t) - y_sup(t) ) / 2 ,   r-dot = da/dt

O traco preto e a camada de pirolise/fuligem na superficie do combustivel:
escura por DENTRO (contra a chama) e por FORA (contra o polietileno que a
chama ilumina por transparencia). Isso da o detector certo -- um filtro de
VALE, nao de borda:

    D(y) = min( <I acima> , <I abaixo> ) - <I no nucleo>

O min() e o que importa. Um detector de borda (claro de um lado, escuro do
outro) engancha na fronteira da chama saturada, que nao e superficie nenhuma;
o vale so responde onde o pixel e mais escuro que os DOIS lados, que e a
definicao do traco. E invariante ao brilho absoluto, entao sobrevive a chama
passar de apagada (t=0) a saturada (t=60 s) e de volta a fraca (t=280 s).

A trajetoria sai por programacao dinamica (Viterbi) no kimograma inteiro, nao
por limiar quadro a quadro: cada coluna tem UMA interface de cada lado, entao
o problema 2D vira 1D e a solucao otima e exata. E o mesmo argumento do
caso_testemari.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter

RAIZ = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
def filtro_vale(I: np.ndarray, w1: int, w2: int) -> np.ndarray:
    """
    I: (N, H) perfis de uma coluna ao longo do tempo. Resposta (N, H) ao
    "vale escuro de meia-largura ~w1 com material mais claro dos dois lados".
    """
    nuc = uniform_filter1d(I, 2 * w1 + 1, axis=1, mode="nearest")
    ac = uniform_filter1d(I, max(w2 - w1, 3), axis=1, mode="nearest")
    d = (w1 + w2) // 2
    cima = np.roll(ac, d, axis=1);  cima[:, :d] = ac[:, :d]
    baixo = np.roll(ac, -d, axis=1); baixo[:, -d:] = ac[:, -d:]
    return np.minimum(cima, baixo) - nuc


def escore(I: np.ndarray, escalas=((6, 30), (12, 50))) -> np.ndarray:
    """Vale multiescala, normalizado por quadro (robusto a percentil 99)."""
    acc = None
    for w1, w2 in escalas:
        S = np.clip(filtro_vale(I, w1, w2), 0, None)
        S = S / np.maximum(np.percentile(S, 99, axis=1, keepdims=True), 1e-6)
        acc = S if acc is None else np.maximum(acc, S)
    return np.clip(acc, 0, 1.5)


def viterbi(score: np.ndarray, dmax: int, lam: float,
            proibido: np.ndarray, mono: int = 0) -> np.ndarray:
    """
    Caminho y(t) de custo minimo sobre score (N, H): maximiza a soma do score
    com penalidade lam*|dy| e salto de no maximo dmax px por passo.
    `proibido` (H,) booleano marca posicoes vetadas (lado errado da porta,
    bordas do bloco).

    `mono` impoe a IRREVERSIBILIDADE da queima dentro da otimizacao: +1 exige
    y nao-decrescente (superficie de baixo, que desce), -1 exige nao-crescente
    (a de cima, que sobe), 0 deixa livre. Isso e melhor do que projetar a serie
    depois (PAVA): a projecao e de minimos quadrados e ESPALHA um quadro ruim
    para tras, enquanto aqui o custo e global -- para passar por um ponto ruim
    o caminho teria de pagar em todos os quadros seguintes, entao a DP prefere
    sacrificar aquele quadro. No caso_testemari a projecao ingenua enviesou a
    superficie em -62 px; a restricao dentro da DP nao tem como fazer isso.

    O laco no tempo e sequencial por natureza, entao o custo aqui e dominado
    por CHAMADA de numpy, nao por tamanho de vetor -- medido: 53 ms com H=600
    e 21 ms com H=25, para o mesmo N. Por isso cada passo usa fatias (uma
    operacao por deslocamento) em vez de mascara booleana com indexacao
    fantasia (quatro), e reaproveita os mesmos buffers.
    """
    N, H = score.shape
    bruto = np.asarray(dmax)
    if bruto.ndim == 0:
        if int(bruto) != bruto or bruto < 0:
            raise ValueError("dmax deve ser inteiro não negativo")
        dk = np.full(N, int(bruto), np.int32)
    else:
        if bruto.shape != (N,) or not np.isfinite(bruto).all() or \
                np.any(bruto < 0) or np.any(bruto != np.floor(bruto)):
            raise ValueError("dmax deve ser escalar ou vetor (N,) inteiro não negativo")
        dk = bruto.astype(np.int32)
    dtop = int(dk.max(initial=0))
    custo = -score.astype(np.float64) + np.where(proibido, 1e3, 0.0)[None, :]
    D = np.empty((N, H)); volta = np.empty((N, H), np.int32)
    D[0] = custo[0]
    if mono > 0:
        desloc = np.arange(0, dtop + 1)
    elif mono < 0:
        desloc = np.arange(-dtop, 1)
    else:
        desloc = np.arange(-dtop, dtop + 1)
    pen = lam * np.abs(desloc).astype(np.float64)
    K = len(desloc)
    cand = np.empty((K, H))
    idx = np.arange(H)
    for t in range(1, N):
        ant = D[t - 1]
        for k, d in enumerate(desloc):
            if abs(d) > dk[t]:
                cand[k] = np.inf
                continue
            # cand[k, y] = D[t-1][y - d] + pen[k], com as bordas em +inf
            a, b = max(0, d), H + min(0, d)
            cand[k, a:b] = ant[a - d:b - d] + pen[k]
            if d > 0:
                cand[k, :d] = np.inf
            elif d < 0:
                cand[k, d:] = np.inf
        arg = cand.argmin(0)
        np.add(cand.min(0), custo[t], out=D[t])
        np.subtract(idx, desloc[arg], out=volta[t])
    y = np.empty(N, np.int32)
    y[-1] = int(np.argmin(D[-1]))
    for t in range(N - 1, 0, -1):
        y[t - 1] = volta[t, y[t]]
    return y


def refinar(score: np.ndarray, y: np.ndarray, R: int = 4,
            frac: float = 0.35, limite: float = 1.5, suav: int = 5,
            mono: int = 0) -> np.ndarray:
    """
    Refina o caminho do Viterbi para subpixel.

    O Viterbi devolve inteiros e, com penalidade de salto, uma rampa lenta sai
    em degraus -- o que nao muda a media mas envenena a derivada instantanea.
    Aqui a posicao vira o CENTROIDE do score numa vizinhanca +-R do caminho,
    com corte em `frac` do pico local para nao arrastar cauda.

    `limite` NAO e detalhe, e a diferenca entre refinar e estragar. Com a
    versao anterior (R=12, sem limite) o centroide pulava para um lobo vizinho
    do score e deslocava a estimativa em ate 12 px de um quadro para o outro --
    seis vezes o que o proprio Viterbi permite. Medido em uma coluna tipica: o
    caminho do Viterbi tinha ZERO saltos acima de 3 px, e depois do refino
    tinha 70, o maior de 14,7 px. Todos os "picos" da trajetoria vinham daqui,
    de uma etapa que deveria mexer em fracoes de pixel.

    Correcao maior que `limite` nao e refinamento, e uma decisao diferente
    sobre onde esta a interface -- e essa decisao e do Viterbi, que tem a
    restricao de continuidade para tomá-la. Entao ela e cortada.

    `suav` passa uma mediana curta na CORRECAO (nao no caminho): a correcao
    subpixel e uma quantidade lenta, e sem isso ela oscila entre +limite e
    -limite em quadros vizinhos e devolve o serrilhado pela porta dos fundos.
    """
    from scipy.ndimage import median_filter

    N, H = score.shape
    base = np.clip(np.asarray(y, float), 0, H - 1).round().astype(int)
    del_ = np.zeros(N)
    for t in range(N):
        c = base[t]
        a, b = max(0, c - R), min(H, c + R + 1)
        w = score[t, a:b].astype(np.float64)
        w = np.clip(w - frac * w.max(), 0, None)
        idx = np.arange(a, b)
        cen = (w * idx).sum() / w.sum() if w.sum() > 1e-9 else c
        del_[t] = cen - c
    if suav >= 3 and N > suav:
        del_ = median_filter(del_, size=suav | 1, mode="nearest")
    out = base + np.clip(del_, -limite, limite)
    if mono > 0:                      # o refino nao pode desfazer a fisica
        out = np.maximum.accumulate(out)
    elif mono < 0:
        out = -np.maximum.accumulate(-out)
    return out


def quadros_com_flare(C: np.ndarray, nsig: float = 4.0, jan: int = 31):
    """
    Quadros em que a chama estourou de luz, achados sem rotulo nenhum.

    Um flare nao escurece nem clareia o quadro inteiro de forma suave: ele
    poe uma labareda saturada por cima da cena por alguns quadros. Entao o
    sinal e o DESVIO em relacao a mediana movel, em duas grandezas que sobem
    juntas num estouro e nao sobem juntas numa mudanca legitima de regime:
    o brilho medio e a fracao de pixels saturados.

    Devolve (mascara booleana, brilho, fracao saturada). Nao serve so para
    marcar: nesses quadros o traco preto some sob a luz, e insistir em medir
    ali e inventar numero.
    """
    from scipy.ndimage import median_filter

    V = C.max(3).astype(np.float32)
    sinais = (V.mean((1, 2)), (V > 250).mean((1, 2)))
    ruim = np.zeros(len(V), bool)
    for x in sinais:
        r = x - median_filter(x, size=jan | 1, mode="nearest")
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        ruim |= r > nsig * max(mad, 1e-9)
    return ruim, sinais[0], sinais[1]


def suavizar_x(Y: np.ndarray, jan: int = 11, nsig: float = 2.5) -> np.ndarray:
    """
    Regularizacao ESPACIAL: em cada quadro, tira os outliers ao longo de x e
    devolve uma superficie suave.

    Isto conserta a falha mais grave que este detector teve. Cada coluna roda
    o seu proprio Viterbi no tempo, entao existe restricao de continuidade em
    t e NENHUMA em x -- a linha desenhada num quadro era a justaposicao de
    dezenas de estimativas independentes. Medido no PRIMEIRO quadro, onde a
    superficie e uma reta sem ambiguidade nenhuma: colunas vizinhas davam
    341, 296, 304, 428, 312 px, com salto de 124 px entre vizinhas. Uma coluna
    que engancha no vale errado ficava errada o ensaio inteiro, sozinha, sem
    nada para puxa-la de volta.

    Hampel em x (a mediana movel resiste ao proprio outlier) -> descarta ->
    interpola -> Savitzky-Golay. A superficie de um grao e suave em x na
    escala destas colunas; degrau de dezenas de pixels entre vizinhas nao e
    fisica, e erro de rotulagem.
    """
    from scipy.signal import savgol_filter

    Y = np.array(Y, float, copy=True)
    N, W = Y.shape
    j = min(int(jan) | 1, (W - 1) | 1)
    if j < 5 or W < 5:
        return Y
    h = j // 2
    idx = np.clip(np.arange(W)[:, None] + np.arange(-h, h + 1)[None, :], 0, W - 1)
    for t in range(N):
        y = Y[t].copy()
        jan_ = y[idx]
        med = np.median(jan_, 1)
        mad = np.median(np.abs(jan_ - med[:, None]), 1) * 1.4826
        ruim = np.abs(y - med) > nsig * np.maximum(mad, 1.0)
        if ruim.all():
            continue
        if ruim.any():
            y[ruim] = np.interp(np.flatnonzero(ruim), np.flatnonzero(~ruim),
                                y[~ruim])
        Y[t] = savgol_filter(y, j, 2)
    return Y


def colunas_confiaveis(Y: np.ndarray, nsig: float = 3.0) -> np.ndarray:
    """
    Colunas cujo recuo total e compativel com o das demais.

    Serve para escolher quem entra no ajuste rigido em x. As colunas da
    cavidade do ignitor e da saida veem outra coisa que nao a superficie e,
    se entrarem, entortam o polinomio para todas as outras. O criterio nao
    precisa de rotulo: recuo total muito fora da mediana denuncia a coluna.
    """
    tot = Y[-1] - Y[0]
    med = np.median(tot)
    mad = np.median(np.abs(tot - med)) * 1.4826
    ok = np.abs(tot - med) <= nsig * max(mad, 1.0)
    return ok if ok.sum() >= max(8, 0.3 * len(ok)) else np.ones(len(ok), bool)


def ajuste_robusto_x(Y: np.ndarray, grau: int = 4, iters: int = 4,
                     nsig: float = 2.0, usar: np.ndarray | None = None) -> np.ndarray:
    """
    Superficie axial rigida: polinomio de grau baixo em x, por quadro, com
    reponderacao iterativa que joga fora as colunas discordantes.

    A mediana movel de `suavizar_x` resolve outlier ISOLADO, mas nao resolve
    um BLOCO de colunas vizinhas todas enganchadas no mesmo vale errado -- a
    mediana da janela mora dentro do bloco e segue o erro. O ajuste global
    resolve, porque um bloco de dez colunas erradas ainda e minoria contra as
    outras cinquenta e sete, e a reponderacao o descarta.

    Grau 4 e escolhido para caber o perfil axial real (a cabeca regride mais
    que a saida, de forma suave) sem espaco para acompanhar erro de rotulagem.
    """
    Y = np.asarray(Y, float)
    N, W = Y.shape
    if W < grau + 2:
        return Y.copy()
    x = np.linspace(-1, 1, W)
    V = np.vander(x, grau + 1)
    base = np.ones(W) if usar is None else np.asarray(usar, float)
    out = np.empty_like(Y)
    for t in range(N):
        y = Y[t]
        w = base.copy()
        for _ in range(iters):
            try:
                c, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
            except np.linalg.LinAlgError:
                break
            r = y - V @ c
            rr = r[base > 0]
            mad = np.median(np.abs(rr - np.median(rr))) * 1.4826
            w = base * (np.abs(r) <= nsig * max(mad, 1.0))
            if w.sum() < grau + 2:
                w = base.copy(); break
        out[t] = V @ c
    return out


def corredor_no_escore(S: np.ndarray, ref: np.ndarray, raio: float,
                       peso: float) -> np.ndarray:
    """
    Prende o escore a um corredor de raio `raio` ao redor de `ref(t)`, com
    custo `peso` proporcional ao afastamento. Fora do corredor, proibido.
    """
    yy = np.arange(S.shape[1])
    dist = np.abs(yy[None, :] - np.asarray(ref, float)[:, None])
    S = S - peso * np.minimum(dist / max(raio, 1e-6), 1.0)
    S[dist > raio] = -1e3
    return S


def bordas_vale(I: np.ndarray, y: np.ndarray, frac: float = 0.5,
                busca: int = 70) -> tuple[np.ndarray, np.ndarray]:
    """
    Bordas do traco escuro em cada quadro: onde a intensidade sobe metade do
    caminho entre o fundo do vale e o OMBRO DAQUELE LADO.

    Cada lado tem o seu proprio ombro, e essa e a parte que importa. O traco
    tem chama saturada de um lado e polietileno de brilho variavel do outro;
    um alvo unico calculado no min/max da janela inteira ia longe demais para
    o lado escuro e dava espessuras de 80 px onde o traco tem 30. Com ombro
    por lado a medida deixa de depender de quao clara esta a chama.
    """
    N, H = I.shape
    lo = np.zeros(N); hi = np.zeros(N)
    for t in range(N):
        c = int(np.clip(round(y[t]), 1, H - 2))
        fundo = I[t, max(0, c - 2):c + 3].min()
        for sinal, saida in ((-1, "lo"), (+1, "hi")):
            a = c
            fim = max(0, c - busca) if sinal < 0 else min(H - 1, c + busca)
            ombro = fundo
            k = c
            while k != fim:
                k += sinal
                ombro = max(ombro, I[t, k])
            alvo = fundo + frac * max(ombro - fundo, 1e-6)
            k = c
            while k != fim and I[t, k] <= alvo:
                k += sinal
            if saida == "lo": lo[t] = k
            else: hi[t] = k
    return lo, hi


def largura_vale(I: np.ndarray, y: np.ndarray, frac: float = 0.5,
                 busca: int = 70) -> np.ndarray:
    """Espessura do traco: e a incerteza fisica da medida, porque o traco de
    pirolise nao e uma linha, e uma camada com espessura."""
    lo, hi = bordas_vale(I, y, frac, busca)
    return hi - lo


# --------------------------------------------------------------------------- #
def centro_inicial(G: np.ndarray, n0: int = 12) -> np.ndarray:
    """
    Centro da porta em t=0, por coluna: centroide da regiao escura no quadro
    inicial. Antes da ignicao a porta e a unica coisa escura no grao claro,
    entao esta e a unica etapa do metodo com um limiar -- e ela nao mede
    nada, so define de que lado do centro cada interface pode estar.
    """
    N, H, nb = G.shape
    ini = gaussian_filter1d(G[:n0].mean(0), 4, axis=0)          # (H, nb)
    miolo = slice(int(0.15 * H), int(0.85 * H))
    idx = np.arange(H)[miolo]
    yc = np.zeros(nb)
    for j in range(nb):
        p = ini[miolo, j]
        w = np.clip(np.percentile(p, 20) - p, 0, None)
        yc[j] = (w * idx).sum() / max(w.sum(), 1e-9)
    return median_filter(yc, 7, mode="nearest")


def detectar(G: np.ndarray, dmax: int = 2, lam: float = 0.5, margem: int = 14,
             escalas=((6, 30), (12, 50)), jan_x: int = 11,
             raio: float = 40.0, peso: float = 1.5,
             raio2: float = 12.0, peso2: float = 2.5, grau: int = 4,
             flare: np.ndarray | None = None, mono: bool = True):
    """
    G: (N, H, nb) luminancia -> dict com y_sup, y_inf (N, nb) em px da ROI.

    `mono` impoe que a face de cima so sobe e a de baixo so desce, dentro do
    Viterbi. `flare` (mascara de quadros estourados de luz) reduz o peso do
    escore naqueles quadros, para o caminho atravessa-los pela continuidade em
    vez de perseguir o brilho.

    TRES PASSADAS, e nenhuma e enfeite. A primeira roda um Viterbi por coluna,
    com continuidade no tempo e nenhuma em x. A segunda prende cada coluna a
    um corredor largo ao redor de um ajuste polinomial ROBUSTO em x, que
    desmancha ate blocos inteiros de colunas erradas. A terceira aperta o
    corredor ao redor de uma suavizacao local, que ja pode ser local porque os
    blocos ja sairam, e devolve a estrutura axial de verdade.

    Sem isso, a linha desenhada num quadro era a justaposicao de dezenas de
    estimativas independentes: no primeiro quadro, onde a superficie e uma reta
    sem ambiguidade, colunas vizinhas davam 341, 296, 304, 428 px.
    """
    N, H, nb = G.shape
    yc0 = centro_inicial(G)
    # a queima e irreversivel: a face de cima so sobe (y cai), a de baixo so
    # desce (y cresce). Nenhuma das duas pode devolver combustivel.
    mo = {"y_sup": -1, "y_inf": +1}
    saida = {k: np.zeros((N, nb)) for k in
             ("y_sup", "y_inf", "c_sup", "c_inf", "e_sup", "e_inf")}
    if flare is not None and flare.any():
        peso_q = np.where(flare, 0.2, 1.0)[:, None]     # quadro estourado pesa pouco
    else:
        peso_q = 1.0
    Sc = {}
    proibs = {}
    for j in range(nb):
        I = gaussian_filter1d(G[:, :, j], 2.0, axis=1)
        Sc[j] = escore(I, escalas) * peso_q
        c0 = int(round(yc0[j]))
        for lado, sy in ((-1, "y_sup"), (+1, "y_inf")):
            proib = np.zeros(H, bool)
            if lado < 0:
                proib[c0:] = True; proib[:margem] = True
            else:
                proib[:c0] = True; proib[H - margem:] = True
            proibs[(j, sy)] = proib
            saida[sy][:, j] = viterbi(Sc[j], dmax, lam, proib,
                                      mo[sy] if mono else 0)

    for sy, sc, se in (("y_sup", "c_sup", "e_sup"), ("y_inf", "c_inf", "e_inf")):
        for etapa, (ref_f, r_, p_) in enumerate((
                (lambda Y: ajuste_robusto_x(Y, grau, usar=colunas_confiaveis(Y)),
                 raio, peso),
                (lambda Y: suavizar_x(Y, jan_x), raio2, peso2))):
            ref = ref_f(saida[sy])
            for j in range(nb):
                S2 = corredor_no_escore(Sc[j], ref[:, j], r_, p_)
                m_ = mo[sy] if mono else 0
                y = viterbi(S2, dmax, lam, proibs[(j, sy)], m_)
                saida[sy][:, j] = refinar(S2, y, mono=m_) if etapa else y
                if etapa:
                    saida[sc][:, j] = Sc[j][np.arange(N), y]   # confianca da IMAGEM
                    I = gaussian_filter1d(G[:, :, j], 2.0, axis=1)
                    saida[se][:, j] = largura_vale(I, y)
    saida["yc0"] = yc0
    return saida


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubo", default=str(RAIZ / "_tmp/cubo_bgr.npy"))
    ap.add_argument("--meta", default=str(RAIZ / "_tmp/cubo_meta.json"))
    ap.add_argument("--dmax", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=str(RAIZ / "resultados"))
    a = ap.parse_args()

    C = np.load(a.cubo); meta = json.loads(Path(a.meta).read_text())
    G = C.astype(np.float32).mean(3)
    flare, brilho, sat = quadros_com_flare(C)
    print(f"quadros com estouro de luz: {int(flare.sum())} de {len(flare)} "
          f"({100*flare.mean():.1f} %)")
    r = detectar(G, dmax=a.dmax, lam=a.lam, flare=flare)
    r["flare"] = flare; r["brilho"] = brilho; r["saturado"] = sat
    out = Path(a.out); out.mkdir(exist_ok=True)
    np.savez(out / "interfaces.npz", **r, roi=meta["roi"], bin_x=meta["bin_x"],
             dq=meta["dq"], fps_camera=meta["fps_camera"],
             quadro_camera=np.asarray(meta["quadro_camera"]),
             quadro_arquivo=np.asarray(meta["quadro_arquivo"]))
    a_px = (r["y_inf"] - r["y_sup"]) / 2.0
    print(f"meia-altura da porta: t=0 {np.nanmedian(a_px[0]):.1f} px  "
          f"t=fim {np.nanmedian(a_px[-1]):.1f} px")
    print(f"espessura mediana do traco: sup {np.median(r['e_sup']):.0f} px  "
          f"inf {np.median(r['e_inf']):.0f} px")
    print(f"gravado em {out/'interfaces.npz'}")


if __name__ == "__main__":
    main()
