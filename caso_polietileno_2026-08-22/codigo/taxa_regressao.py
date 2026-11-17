"""
Da interface detectada a taxa de regressao.

DUAS MEDIDAS INDEPENDENTES, nao uma. A porta e horizontal e tem combustivel
dos dois lados, entao ha duas superficies e elas nao tem por que regredir
igual (a de baixo recebe o polietileno fundido que escorre da de cima). Cada
uma sai aqui com sua propria taxa, sua propria janela de validade, seu proprio
perfil axial e seu proprio fechamento contra a espessura de combustivel que
existia daquele lado (o web).
A meia-altura da porta a(t) = (y_inf - y_sup)/2 vem depois, como agregado.

EIXO DE TEMPO. A unidade natural e o QUADRO DE CAMERA, lido do contador que a
propria camera gravou na imagem (ver ler_contador.py). O conteiner nao serve:
ele tem 20 % de quadros repetidos por conversao de 24 para 30. Toda taxa sai
primeiro em px por quadro de camera, que nao depende de hipotese nenhuma, e so
depois e convertida para px/s pela taxa nominal da camera.

Limpeza (mesma receita do caso_testemari, e pelo mesmo motivo): descartar ->
interpolar -> mediana movel. A projecao isotonica fica FORA da serie principal
e entra so como verificacao: PAVA e minimos quadrados e devolve uma escada de
blocos constantes, o que nao muda a media mas destroi a derivada.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
from scipy.signal import savgol_filter                              # noqa: E402
from scipy.ndimage import median_filter                             # noqa: E402


# --------------------------------------------------------------------------- #
def hampel_mask(x, k: int = 9, nsig: float = 3.0):
    """
    Outliers pelo filtro de Hampel: ponto a mais de `nsig` MADs da mediana
    movel de janela `k`. Usa MAD, e nao desvio-padrao, porque o proprio
    outlier inflaria o desvio-padrao e se esconderia atras dele.
    """
    x = np.asarray(x, float); n = len(x)
    if n < k or k < 3:
        return np.zeros(n, bool)
    h = k // 2
    idx = np.clip(np.arange(n)[:, None] + np.arange(-h, h + 1)[None, :], 0, n - 1)
    jan = x[idx]
    med = np.nanmedian(jan, 1)
    mad = np.nanmedian(np.abs(jan - med[:, None]), 1) * 1.4826
    return np.abs(x - med) > nsig * np.maximum(mad, 1e-6)


def pava(y):
    """
    Projecao isotonica crescente (pool adjacent violators). Aqui ela NAO entra
    na serie principal: serve so como verificacao, porque e projecao de minimos
    quadrados e devolve uma escada de blocos constantes -- nao muda a media,
    mas envenena a derivada.
    """
    y = np.asarray(y, float).copy()
    val = []; peso = []
    for v in y:
        val.append(v); peso.append(1.0)
        while len(val) > 1 and val[-2] > val[-1]:
            v2 = val.pop(); p2 = peso.pop()
            v1 = val.pop(); p1 = peso.pop()
            val.append((v1 * p1 + v2 * p2) / (p1 + p2)); peso.append(p1 + p2)
    out = np.empty_like(y); i = 0
    for v, p in zip(val, peso):
        out[i:i + int(p)] = v; i += int(p)
    return out


# --------------------------------------------------------------------------- #
def janela_queima(C: np.ndarray, lim: float = 0.25):
    """
    Ignicao e apagamento pela fracao de pixels quentes (R alto, B baixo =
    laranja de chama). Nao usa brilho absoluto porque o grao e claro antes da
    ignicao e escuro depois.
    """
    R = C[..., 2].astype(np.float32); B = C[..., 0].astype(np.float32)
    quente = ((R > 170) & (B < 0.65 * R)).mean((1, 2))
    quente = np.convolve(quente, np.ones(9) / 9, "same")
    aceso = quente > lim
    i0 = int(np.argmax(aceso))
    i1 = len(quente) - int(np.argmax(aceso[::-1])) - 1
    return i0, i1, quente


def geometria_inicial(C: np.ndarray, i0: int, yc0: np.ndarray, nb: int):
    """
    Geometria do grao ANTES da ignicao, por coluna: paredes da porta e bordas
    externas do polietileno. E o unico instante do ensaio em que a porta e
    escura e o grao e claro, entao ali um limiar global e legitimo. Ele nao
    mede taxa nenhuma: fixa a condicao inicial e o web disponivel de cada
    lado, que depois viram o fechamento independente.
    """
    n = max(int(i0 * 0.6), 3)
    ini = C[:n].astype(np.float32).mean((0, 3))              # (H, nb)
    H = ini.shape[0]
    lim = 0.5 * (np.percentile(ini, 5) + np.percentile(ini, 95))
    ps = np.full(nb, np.nan); pi = np.full(nb, np.nan)
    gs = np.full(nb, np.nan); gi = np.full(nb, np.nan)
    for j in range(nb):
        p = ini[:, j]
        c = int(np.clip(round(yc0[j]), 1, H - 2))
        if p[c] >= lim:
            continue
        k = c
        while k > 0 and p[k] < lim: k -= 1
        ps[j] = k + 0.5
        k = c
        while k < H - 1 and p[k] < lim: k += 1
        pi[j] = k - 0.5
        k = int(ps[j])
        while k > 0 and p[k] >= lim: k -= 1
        gs[j] = k + 0.5
        if gs[j] >= ps[j] - 1: gs[j] = np.nan
        k = int(pi[j]) + 1
        while k < H - 1 and p[k] >= lim: k += 1
        gi[j] = k - 0.5
        if gi[j] <= pi[j] + 1: gi[j] = np.nan
    return ps, pi, gs, gi


def limpar(S: np.ndarray, med_k: int = 9, hampel_k: int = 9,
           nsig: float = 3.0) -> np.ndarray:
    """Hampel -> interpolacao -> mediana movel. Sem projecao isotonica."""
    S = np.array(S, float, copy=True)
    N, W = S.shape
    for x in range(W):
        y = S[:, x].copy()
        y[hampel_mask(y, hampel_k, nsig)] = np.nan
        ok = np.isfinite(y)
        if ok.sum() < 2:
            S[:, x] = np.nan
            continue
        y = np.interp(np.arange(N), np.flatnonzero(ok), y[ok])
        S[:, x] = median_filter(y, size=med_k | 1, mode="nearest")
    return S


def ajuste_linear(x, y):
    """Ajuste y = b + m x. Devolve (m, erro-padrao de m, R²)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    xc = x - x.mean(); den = float((xc ** 2).sum())
    m = float((xc * (y - y.mean())).sum() / den)
    res = y - (y.mean() + m * xc)
    se = float(np.sqrt((res ** 2).sum() / max(len(x) - 2, 1) / den))
    r2 = float(1 - (res ** 2).sum() / max(float(((y - y.mean()) ** 2).sum()), 1e-12))
    return m, se, r2


def janela_confiavel(conf: np.ndarray, i0: int, i1: int, k0: int, k1: int,
                     jan: int = 61, frac: float = 0.5, ponte: int = 45):
    """
    Maior trecho em que a evidencia existe. O score do vale no caminho
    escolhido e uma confianca sem rotulo: o Viterbi SEMPRE devolve um caminho,
    mas se o vale sumiu o score cai a zero e o caminho vira chute. O limiar e
    auto calibrado em um quarto da mediana da propria confianca dentro da
    queima, entao nao e um numero escolhido a mao.

    Buracos curtos (ate `ponte` amostras) sao fechados antes de procurar o
    trecho: uma labareda que apaga o traco por alguns segundos nao invalida a
    medida, porque a continuidade do Viterbi atravessa o buraco. O que
    invalida e o colapso SUSTENTADO da evidencia.
    """
    from scipy.ndimage import binary_closing
    lim = 0.25 * float(np.median(conf[i0:i1]))
    ok = np.convolve(conf > lim, np.ones(jan) / jan, "same") > frac
    ok = binary_closing(ok, np.ones(ponte, bool))
    ok[:k0] = False; ok[k1 + 1:] = False
    melhor, ini, a, b = 0, None, k0, k1
    for k in range(len(ok) + 1):
        if k < len(ok) and ok[k]:
            ini = k if ini is None else ini
        elif ini is not None:
            if k - ini > melhor:
                melhor, a, b = k - ini, ini, k - 1
            ini = None
    return a, b, lim


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-min", type=float, default=400.0, help="px, corta a cabeca")
    ap.add_argument("--x-max", type=float, default=1850.0, help="px, corta a saida")
    ap.add_argument("--folga-ign", type=float, default=8.0,
                    help="s apos a ignicao descartados (transiente)")
    ap.add_argument("--folga-fim", type=float, default=5.0)
    ap.add_argument("--mm-per-px", type=float, default=1.0)
    ap.add_argument("--fps-camera", type=float, default=None)
    ap.add_argument("--fator-tempo-real", type=float, default=None,
                    help="segundos reais por segundo do vídeo (3 para vídeo em 3x)")
    ap.add_argument("--fps-alternativo", type=float, default=24.0,
                    help="a outra hipotese de taxa da camera, so para relatorio")
    ap.add_argument("--interfaces", default=str(RAIZ / "resultados/interfaces.npz"),
                    help="trocar por interfaces_revisado.npz apos o revisor")
    ap.add_argument("--sufixo", default="", help="sufixo dos arquivos de saida")
    ap.add_argument("--out", default=str(RAIZ / "resultados"))
    a = ap.parse_args()

    C = np.load(RAIZ / "_tmp/cubo_bgr.npy")
    meta = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())
    r = np.load(a.interfaces)
    dq = int(r["dq"])
    fps_cam = float(a.fps_camera or r["fps_camera"])
    try:
        cfg = json.loads((RAIZ / "configuracao_pe.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    fator_tempo = float(
        a.fator_tempo_real if a.fator_tempo_real is not None else
        (r["fator_tempo_real"] if "fator_tempo_real" in r else
         cfg.get("fator_tempo_real", 1.0)))
    if not np.isfinite(fator_tempo) or fator_tempo <= 0:
        raise ValueError("fator-tempo-real deve ser positivo")
    q = np.asarray(r["quadro_camera"], float)
    dt = dq / fps_cam * fator_tempo
    Y0, Y1, X0, X1 = meta["roi"]; bx = meta["bin_x"]
    ys, yi = r["y_sup"].copy(), r["y_inf"].copy()
    N, nb = ys.shape
    t_video = q / fps_cam
    t = t_video * fator_tempo
    xc = X0 + bx * np.arange(nb) + bx / 2
    mm = a.mm_per_px

    # ---- janelas ---------------------------------------------------------- #
    i0, i1, quente = janela_queima(C)
    k0 = int(np.searchsorted(t, t[i0] + a.folga_ign))
    k1 = int(np.searchsorted(t, t[i1] - a.folga_fim))
    col = (xc >= a.x_min) & (xc <= a.x_max)

    # ---- geometria inicial ------------------------------------------------ #
    ps0, pi0, gs0, gi0 = geometria_inicial(C, i0, r["yc0"], nb)
    origem_estado_inicial = {"sup": "imagem_pre_ignicao",
                             "inf": "imagem_pre_ignicao"}
    # Se a revisão humana marcou o estado pré-ignição, ela substitui a borda
    # automática também na metrologia final. Antes, os cliques apareciam na
    # trajetória revisada, mas porta inicial, web e fechamento ainda eram
    # calculados silenciosamente com a segmentação automática original.
    for nome, chave_h, Y, destino in (
            ("sup", "humano_sup", ys, "ps0"),
            ("inf", "humano_inf", yi, "pi0")):
        if chave_h not in r:
            continue
        humano = np.asarray(r[chave_h], float)
        kh = np.flatnonzero(np.nanmax(humano[:max(i0, 1)], axis=1) > 0)
        if not len(kh):
            continue
        inicial = np.nanmedian(Y[kh], axis=0)
        if destino == "ps0":
            ps0 = inicial
        else:
            pi0 = inicial
        origem_estado_inicial[nome] = "marcacao_humana_pre_ignicao"
    h0 = np.nanmedian((pi0 - ps0)[col])
    web_sup = np.nanmedian((ps0 - gs0)[col])
    web_inf = np.nanmedian((gi0 - pi0)[col])

    # ---- limpeza ---------------------------------------------------------- #
    ys_l, yi_l = limpar(ys), limpar(yi)
    A = (yi_l - ys_l) / 2.0
    # verificacao: a mesma limpeza, com monotonicidade imposta por cima
    A_mono = np.stack([np.apply_along_axis(pava, 0, limpar(z, med_k=15))
                       for z in (yi, -ys)]).sum(0) / 2.0
    rec_sup = ps0[None, :] - ys_l                # recuo da superficie de cima
    rec_inf = yi_l - pi0[None, :]                # recuo da superficie de baixo

    # ---- colunas boas ----------------------------------------------------- #
    tot = A[k1] - A[k0]
    bom = col & np.isfinite(tot) & (tot > 0)
    tt = t[k0:k1 + 1]
    por_col = np.array([ajuste_linear(tt, A[k0:k1 + 1, j] * mm) for j in range(nb)])
    rdot_x, se_x, r2_x = por_col[:, 0], por_col[:, 1], por_col[:, 2]
    # o criterio olha a QUALIDADE do ajuste de cada coluna, nao a concordancia
    # com a mediana: a variacao axial de r-dot e fisica e nao pode ser filtrada
    bom &= r2_x > 0.85
    print(f"colunas usadas: {bom.sum()}/{nb}  "
          f"(x de {xc[bom].min():.0f} a {xc[bom].max():.0f} px)")

    # ---- as duas superficies, separadas ----------------------------------- #
    conf = dict(sup=np.nanmedian(r["c_sup"][:, bom], 1),
                inf=np.nanmedian(r["c_inf"][:, bom], 1))
    sup = {}
    for nome, rec, web in (("sup", rec_sup, web_sup), ("inf", rec_inf, web_inf)):
        med = np.nanmedian(rec[:, bom], 1)
        ka, kb, lim = janela_confiavel(conf[nome], i0, i1, k0, k1)
        m_t, se_t, r2_t = ajuste_linear(t[k0:k1 + 1], med[k0:k1 + 1] * mm)
        m_c, se_c, r2_c = ajuste_linear(t[ka:kb + 1], med[ka:kb + 1] * mm)
        pc = np.array([ajuste_linear(t[ka:kb + 1], rec[ka:kb + 1, j] * mm)
                       for j in range(nb)])
        sup[nome] = dict(
            med=med, k_conf=(ka, kb), lim_conf=lim,
            rdot=m_t, se=se_t, r2=r2_t,
            rdot_conf=m_c, se_conf=se_c, r2_conf=r2_c,
            rdot_x=pc[:, 0], r2_x=pc[:, 2],
            web=float(web),
            recuo=float(med[k1]),                  # recuo no fim da queima
            recuo_conf=float(med[kb]),             # recuo no fim da janela boa
            fechamento=float(100 * med[k1] / web),
        )

    # ---- agregado a(t) ---------------------------------------------------- #
    a_med = np.nanmedian(A[:, bom], 1)
    a_mad = np.nanmedian(np.abs(A[:, bom] - a_med[:, None]), 1)
    a_mono = np.nanmedian(A_mono[:, bom], 1)
    m_a, se_a, r2_a = ajuste_linear(tt, a_med[k0:k1 + 1] * mm)
    m_mono, _, r2_mono = ajuste_linear(tt, a_mono[k0:k1 + 1] * mm)
    kc0 = max(sup["sup"]["k_conf"][0], sup["inf"]["k_conf"][0])
    kc1 = min(sup["sup"]["k_conf"][1], sup["inf"]["k_conf"][1])
    m_c, se_c, r2_c = ajuste_linear(t[kc0:kc1 + 1], a_med[kc0:kc1 + 1] * mm)

    w = min(int(41 / dt) | 1, (N - 1) | 1)
    rdot_t = savgol_filter(a_med, w, 2, deriv=1, delta=dt) * mm

    fases = []
    for f0 in range(int(t[k0] // 20 * 20) + 20, int(t[k1]), 40):
        f1 = min(f0 + 40, t[k1])
        i, jj = int(np.searchsorted(t, f0)), int(np.searchsorted(t, f1))
        if jj - i < 10: continue
        fases.append((f0, f1,
                      *ajuste_linear(t[i:jj], a_med[i:jj] * mm),
                      ajuste_linear(t[i:jj], sup["sup"]["med"][i:jj] * mm)[0],
                      ajuste_linear(t[i:jj], sup["inf"]["med"][i:jj] * mm)[0]))

    linhas_j = []
    for frac in (0.4, 0.6, 0.8, 1.0):
        kf = int(k0 + frac * (k1 - k0))
        linhas_j.append((t[kf], *ajuste_linear(t[k0:kf + 1], a_med[k0:kf + 1] * mm)))

    # ---- saida ------------------------------------------------------------ #
    out = Path(a.out); out.mkdir(exist_ok=True)
    np.savez(out / f"serie{a.sufixo}.npz", t=t, q=q, dq=dq, fps_camera=fps_cam, dt=dt,
             A=A, ys=ys_l, yi=yi_l, bom=bom, xc=xc, k0=k0, k1=k1,
             a_med=a_med, a_mad=a_mad, a_mono=a_mono, rdot_t=rdot_t,
             rec_sup=sup["sup"]["med"], rec_inf=sup["inf"]["med"],
             rec_sup_x=rec_sup, rec_inf_x=rec_inf,
             rdot_x=rdot_x, r2_x=r2_x, se_x=se_x,
             rdot_x_sup=sup["sup"]["rdot_x"], rdot_x_inf=sup["inf"]["rdot_x"],
             conf_s=conf["sup"], conf_i=conf["inf"],
             lim_conf=[sup["sup"]["lim_conf"], sup["inf"]["lim_conf"]],
             k_conf_sup=sup["sup"]["k_conf"], k_conf_inf=sup["inf"]["k_conf"],
             kc0=kc0, kc1=kc1, quente=quente,
             ps0=ps0, pi0=pi0, gs0=gs0, gi0=gi0, mm_per_px=mm,
             t_video=t_video, fator_tempo_real=fator_tempo)

    with open(out / f"serie_temporal{a.sufixo}.csv", "w") as f:
        f.write("quadro_camera,t_s,y_sup_px,y_inf_px,recuo_sup_px,recuo_inf_px,"
                "a_px,a_mad_px,rdot_inst_px_s,conf_sup,conf_inf,frac_quente\n")
        for k in range(N):
            f.write(f"{q[k]:.0f},{t[k]:.3f},{np.nanmedian(ys_l[k][bom]):.2f},"
                    f"{np.nanmedian(yi_l[k][bom]):.2f},{sup['sup']['med'][k]:.2f},"
                    f"{sup['inf']['med'][k]:.2f},{a_med[k]:.2f},{a_mad[k]:.2f},"
                    f"{rdot_t[k]:.4f},{conf['sup'][k]:.3f},{conf['inf'][k]:.3f},"
                    f"{quente[k]:.3f}\n")
    with open(out / f"rdot_por_coluna{a.sufixo}.csv", "w") as f:
        f.write("x_px,rdot_a_px_s,rdot_sup_px_s,rdot_inf_px_s,se_px_s,r2,usada\n")
        for j in range(nb):
            f.write(f"{xc[j]:.0f},{rdot_x[j]:.4f},{sup['sup']['rdot_x'][j]:.4f},"
                    f"{sup['inf']['rdot_x'][j]:.4f},{se_x[j]:.4f},{r2_x[j]:.4f},"
                    f"{int(bom[j])}\n")
    with open(out / f"rdot_por_fase{a.sufixo}.csv", "w") as f:
        f.write("t0_s,t1_s,rdot_a_px_s,se,r2,rdot_sup_px_s,rdot_inf_px_s\n")
        for f0, f1, m_, se_, r2_, ms_, mi_ in fases:
            f.write(f"{f0},{f1:.0f},{m_:.4f},{se_:.4f},{r2_:.4f},{ms_:.4f},{mi_:.4f}\n")
    with open(out / f"rdot_por_janela{a.sufixo}.csv", "w") as f:
        f.write("t_fim_s,rdot_px_s,se_px_s,r2\n")
        for tf, m_, se_, r2_ in linhas_j:
            f.write(f"{tf:.1f},{m_:.4f},{se_:.4f},{r2_:.4f}\n")

    def em_quadro(v):        # px/s real -> px por quadro de camera
        return v * fator_tempo / fps_cam

    res = dict(
        base_de_tempo=dict(
            fps_camera_usada=fps_cam, fps_alternativa=a.fps_alternativo,
            fator_tempo_real=fator_tempo,
            quadros_arquivo=meta["n_quadros_arquivo"],
            quadros_unicos=meta["n_unicos"],
            quadros_repetidos=meta["n_quadros_arquivo"] - meta["n_unicos"],
            amostragem_quadros=dq,
            nota=("tempo real = tempo do vídeo * fator_tempo_real; taxas em "
                  "px/s usam o tempo real. Para a outra hipótese de FPS, "
                  "multiplique por fps_alternativa/fps_camera_usada")),
        t_ignicao_s=float(t[i0]), t_apagamento_s=float(t[i1]),
        duracao_queima_s=float(t[i1] - t[i0]),
        quadro_ignicao=int(q[i0]), quadro_apagamento=int(q[i1]),
        janela_ajuste_s=[float(t[k0]), float(t[k1])],
        porta_inicial_px=float(h0), meia_porta_inicial_px=float(h0 / 2),
        origem_estado_inicial=origem_estado_inicial,
        superficie_de_cima=dict(
            web_px=sup["sup"]["web"],
            recuo_total_px=sup["sup"]["recuo"],
            recuo_na_janela_confiavel_px=sup["sup"]["recuo_conf"],
            fechamento_pct=sup["sup"]["fechamento"],
            janela_confiavel_s=[float(t[sup["sup"]["k_conf"][0]]), float(t[sup["sup"]["k_conf"][1]])],
            rdot_px_s=sup["sup"]["rdot"], r2=sup["sup"]["r2"],
            rdot_confiavel_px_s=sup["sup"]["rdot_conf"], se=sup["sup"]["se_conf"],
            r2_confiavel=sup["sup"]["r2_conf"],
            rdot_px_por_quadro=em_quadro(sup["sup"]["rdot_conf"])),
        superficie_de_baixo=dict(
            web_px=sup["inf"]["web"],
            recuo_total_px=sup["inf"]["recuo"],
            recuo_na_janela_confiavel_px=sup["inf"]["recuo_conf"],
            fechamento_pct=sup["inf"]["fechamento"],
            janela_confiavel_s=[float(t[sup["inf"]["k_conf"][0]]), float(t[sup["inf"]["k_conf"][1]])],
            rdot_px_s=sup["inf"]["rdot"], r2=sup["inf"]["r2"],
            rdot_confiavel_px_s=sup["inf"]["rdot_conf"], se=sup["inf"]["se_conf"],
            r2_confiavel=sup["inf"]["r2_conf"],
            rdot_px_por_quadro=em_quadro(sup["inf"]["rdot_conf"])),
        meia_altura=dict(
            rdot_px_s=m_a, se_px_s=se_a, r2=r2_a,
            rdot_px_por_quadro=em_quadro(m_a),
            janela_confiavel_s=[float(t[kc0]), float(t[kc1])],
            rdot_confiavel_px_s=m_c, se_confiavel=se_c, r2_confiavel=r2_c,
            rdot_isotonica_px_s=m_mono, r2_isotonica=r2_mono,
            meia_porta_final_px=float(a_med[k1]),
            mediana_colunas_px_s=float(np.nanmedian(rdot_x[bom])),
            dispersao_colunas_mad_px_s=float(np.nanmedian(np.abs(rdot_x[bom] - np.nanmedian(rdot_x[bom]))))),
        n_colunas=int(bom.sum()), mm_per_px=mm,
        fases_40s=[dict(t0=f0, t1=float(f1), a=float(m_), r2=float(r2_),
                        sup=float(ms_), inf=float(mi_))
                   for f0, f1, m_, se_, r2_, ms_, mi_ in fases],
        dependencia_janela=[dict(t_fim_s=float(x), rdot=float(y), se=float(z), r2=float(w_))
                            for x, y, z, w_ in linhas_j],
    )
    (out / f"resumo{a.sufixo}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # ---- relatorio -------------------------------------------------------- #
    print(f"\nBASE DE TEMPO: vídeo acelerado {fator_tempo:g}x; "
          f"tempo real = tempo do vídeo x {fator_tempo:g}. "
          f"{meta['n_quadros_arquivo']} quadros no arquivo, "
          f"{meta['n_unicos']} unicos ({meta['n_quadros_arquivo']-meta['n_unicos']} "
          f"repetidos descartados). Eixo = quadro de camera a {fps_cam:g} fps.")
    print(f"ignicao no quadro {q[i0]:.0f} ({t[i0]:.1f}s), apagamento no quadro "
          f"{q[i1]:.0f} ({t[i1]:.1f}s)  ->  queima de {t[i1]-t[i0]:.1f}s")
    print(f"porta inicial (pre-ignicao): {h0:.0f} px de altura total")
    print(f"\n{'':22s}{'de cima':>12s}{'de baixo':>12s}")
    print(f"{'web disponivel [px]':22s}{web_sup:12.0f}{web_inf:12.0f}")
    for rot, ch in (("recuo total [px]", "recuo"), ("fechamento [%]", "fechamento")):
        print(f"{rot:22s}{sup['sup'][ch]:12.1f}{sup['inf'][ch]:12.1f}")
    print(f"{'r-dot [px/s]':22s}{sup['sup']['rdot']:12.3f}{sup['inf']['rdot']:12.3f}")
    print(f"{'R2':22s}{sup['sup']['r2']:12.3f}{sup['inf']['r2']:12.3f}")
    print(f"{'r-dot [px/quadro]':22s}{em_quadro(sup['sup']['rdot']):12.4f}"
          f"{em_quadro(sup['inf']['rdot']):12.4f}")
    f_alt = a.fps_alternativo / fps_cam
    print(f"{f'r-dot [px/s] a {a.fps_alternativo:g}fps':22s}"
          f"{sup['sup']['rdot']*f_alt:12.3f}{sup['inf']['rdot']*f_alt:12.3f}")
    print(f"\n{'evidencia solida ate':22s}"
          f"{t[sup['sup']['k_conf'][1]]:9.0f} s{t[sup['inf']['k_conf'][1]]:9.0f} s")
    print(f"{'r-dot so ate ali':22s}{sup['sup']['rdot_conf']:12.3f}"
          f"{sup['inf']['rdot_conf']:12.3f}")
    print("   CUIDADO: a taxa nao e constante, entao ajustar so no trecho de"
          " evidencia solida\n   mede a fase, nao o ensaio. A linha de cima e"
          " a taxa do ensaio; esta e o quanto\n   dela esta apoiada em"
          " evidencia direta.")
    print(f"\nagregado, meia-altura da porta: {m_a:.3f} +/- {se_a:.3f} px/s  "
          f"R2={r2_a:.4f}   (isotonica {m_mono:.3f}, difere {100*abs(m_mono-m_a)/m_a:.1f} %)")
    print(f"por coluna: mediana {np.nanmedian(rdot_x[bom]):.3f} px/s  "
          f"MAD {np.nanmedian(np.abs(rdot_x[bom]-np.nanmedian(rdot_x[bom]))):.3f}")
    print("\ntaxa por fase de 40 s (a, cima, baixo) [px/s]:")
    for f0, f1, m_, se_, r2_, ms_, mi_ in fases:
        print(f"   {f0:3d}-{f1:3.0f}s -> {m_:6.3f}  {ms_:6.3f}  {mi_:6.3f}   R2={r2_:.3f}")


if __name__ == "__main__":
    main()
