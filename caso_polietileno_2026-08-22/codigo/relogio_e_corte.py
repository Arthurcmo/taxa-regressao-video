"""Fator do tempo da câmera e corte do aquecimento, medidos com dados que não usam a taxa.

1. Fator. Quatro eventos de vazão aparecem dos dois lados: no vídeo (cor da chama, R-B
   médio na região do grão, quadro a quadro, no tempo marcado pela câmera) e no registro
   do banco (vazão e pressão do leito, em segundos reais). Reta t_banco = T0 + s t_câmera
   por mínimos quadrados: s é o fator.
2. Corte. Critérios do sistema de oxidante, nenhum deles usa a linha da superfície: início
   da vazão constante, temperatura do leito catalítico e temperatura na entrada da câmara.
3. Taxa (linha 2b) com o fator medido e janelas de ~20 s que começam em cada critério e
   vão até o salto de vazão, mais a sensibilidade ao início, ao tamanho e ao fim das janelas.

As janelas de tempo em que cada evento é procurado (JANELA_VIDEO, LER_VIDEO) e os limites
de tempo usados nos critérios foram escolhidos olhando o vídeo e o registro do ensaio
original; revise-os para outro ensaio.

    PYTHONPATH=codigo python codigo/relogio_e_corte.py
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textos_tg  # noqa: F401  termos do TG nos textos das figuras (11/11/2026)
import numpy as np
from scipy.ndimage import uniform_filter1d

import recompensa_lei as R
import lei_potencia as RG
import ensaio as E
import serie_porta as RL
from experimento_s_sessao import K_IGN
from taxas_sessao import MM

SAIDA = R.OUT / "relogio_e_corte"
VIDEO = E.VIDEO
Y0, Y1, X0, X1 = json.loads((R.RAIZ / "_tmp/cubo_meta.json").read_text())["roi"]
CAM = E.CAM                       # tempo marcado pela câmera por amostra do cubo (s)
RHO = E.DENSIDADE_OXIDANTE        # g/mL do oxidante
FIM_ANTIGO = 241.0                # s do registro: fim usado em 07/11 (8,7 s antes do salto)
DUR = 20.0                        # s, tamanho das janelas
# Janelas do vídeo onde cada evento é procurado (escolhidas olhando só o vídeo, em s da câmera)
JANELA_VIDEO = {"volta 1": (84.5, 86.0), "volta 2": (164.5, 165.2), "salto": (202.5, 205.0), "fim": (239.5, 241.5)}
LER_VIDEO = {"volta 1": (83.0, 87.5), "volta 2": (163.5, 168.0), "salto": (201.0, 207.0), "fim": (238.0, 247.0)}
SECOES = {"2ª metade (60–120 mm)": (60, 120), "grão todo (10–120 mm)": (10, 120), "60–100 mm": (60, 100)}


# ----------------------------------------------------------------------------- dados
def banco() -> dict[str, np.ndarray]:
    d = np.genfromtxt(E.CANAIS_CSV, delimiter=",", names=True)
    return {n: np.asarray(d[n], float) for n in d.dtype.names}


def quadros_camera():
    q = json.loads((R.RAIZ / "resultados/quadros_camera.json").read_text())
    return np.array(q["contador"]), np.array(q["unico"], bool)


def chama_eventos() -> dict[str, np.ndarray]:
    """R-B e cinza médios na região do grão, em cada quadro único da câmera, perto dos eventos."""
    cache = SAIDA / "chama_eventos.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    cont, unico = quadros_camera()
    cap = cv2.VideoCapture(str(VIDEO))
    out = {}
    for nome, (a, b) in LER_VIDEO.items():
        f0, f1 = int(np.searchsorted(cont, a * 30)), int(np.searchsorted(cont, b * 30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        linhas = []
        for f in range(f0, f1):
            ok, fr = cap.read()
            if ok and unico[f]:
                r = fr[Y0:Y1, X0:X1].astype(np.float32)
                linhas.append((cont[f] / 30, float((r[..., 2] - r[..., 0]).mean()), float(r.mean())))
        out[nome] = np.array(linhas)
    cap.release()
    np.savez(cache, **out)
    return out


def chama_queima() -> np.ndarray:
    """R-B médio por amostra do cubo (a cada 8 quadros da câmera), a queima inteira."""
    cache = SAIDA / "chama_cubo.npy"
    if cache.exists():
        return np.load(cache)
    c = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    rb = np.array([float((np.asarray(c[k], np.float32)[..., 2] - np.asarray(c[k], np.float32)[..., 0]).mean())
                   for k in range(c.shape[0])])
    np.save(cache, rb)
    return rb


# ----------------------------------------------------------------------------- 1. fator
def eventos_video(ch) -> dict[str, float]:
    ev = {}
    for nome, (a, b) in JANELA_VIDEO.items():
        v = ch[nome]
        x = uniform_filter1d(v[:, 2], 9) if nome == "salto" else v[:, 1]   # salto: brilho suavizado
        dx = np.diff(x)
        m = (v[1:, 0] >= a) & (v[1:, 0] <= b)
        i = np.flatnonzero(m)[np.argmin(dx[m]) if nome == "fim" else np.argmax(dx[m])]
        ev[nome] = float(0.5 * (v[i, 0] + v[i + 1, 0]))
    return ev


def eventos_banco(B) -> dict[str, float]:
    t, F, P = B["t_planilha_s"], np.nan_to_num(B["vazao_mL_s"]), B["P_leito"]

    def volta(t0):            # primeira leitura com vazão depois do trecho em zero
        z = t[(t > t0) & (F == 0)][0]
        return float(t[(t > z) & (F > 0.2)][0])
    return {"volta 1": volta(95), "volta 2": volta(195),
            "salto": float(t[(t > 240) & (F > 2.5)][0]),
            "fim": float(t[(t > 290) & (P < 0.55)][0])}    # pressão do leito cai (o medidor segue até 298,7 s)


def ajustar(ev_v, ev_b) -> dict:
    nomes = list(ev_v)
    X = np.array([ev_v[n] for n in nomes]); Y = np.array([ev_b[n] for n in nomes])
    A = np.vstack([np.ones_like(X), X]).T
    c = np.linalg.lstsq(A, Y, rcond=None)[0]
    res = Y - A @ c
    cov = (res @ res) / (len(X) - 2) * np.linalg.inv(A.T @ A)
    loo = {n: float(np.polyfit(np.delete(X, i), np.delete(Y, i), 1)[0]) for i, n in enumerate(nomes)}
    pares = {f"{nomes[i]} → {nomes[j]}": float((Y[j] - Y[i]) / (X[j] - X[i]))
             for i in range(len(X)) for j in range(i + 1, len(X))}
    return dict(T0=float(c[0]), s=float(c[1]), erro_s=float(np.sqrt(cov[1, 1])), residuos=dict(zip(nomes, res.tolist())),
                sem_um=loo, pares=pares, video=ev_v, banco=ev_b,
                residuo_max_com_s_1=float(np.max(np.abs((Y - X) - np.mean(Y - X)))))


class Tempo:
    """Conversões com o fator e o deslocamento medidos."""

    def __init__(self, s, T0):
        self.s, self.T0 = s, T0
        self.ign = T0 + s * K_IGN * CAM               # registro do banco na ignição

    def banco(self, k):
        return self.T0 + self.s * CAM * np.asarray(k, float)

    def k(self, t_banco):
        return (np.asarray(t_banco, float) - self.T0) / (self.s * CAM)

    def desde_ign(self, t_banco):
        return np.asarray(t_banco, float) - self.ign


# ----------------------------------------------------------------------------- 2. corte
def criterios(B, tm: Tempo) -> dict[str, float]:
    t = B["t_planilha_s"]; F = np.nan_to_num(B["vazao_mL_s"])
    T2 = uniform_filter1d(B["T_leito2_C"], 11); Te = uniform_filter1d(B["T_entrada_camara_C"], 11)
    pat2 = float(np.median(T2[(t > 110) & (t < 195)])); pate = float(np.median(Te[(t > 110) & (t < 195)]))
    apos = t > tm.ign

    def primeiro(cond):
        return float(tm.desde_ign(t[apos & cond][0]))
    z = t[(t > 95) & (F == 0)][0]
    return {
        "vazão constante (volta completa depois da 1ª interrupção)": float(tm.desde_ign(t[(t > z) & (F >= 1.0)][0])),
        "leito a 95 % do patamar": primeiro(T2 >= 0.95 * pat2),
        "entrada da câmara a 90 % do patamar": primeiro(Te >= 0.90 * pate),
        "_patamar_leito": pat2, "_patamar_entrada": pate,
    }


# ----------------------------------------------------------------------------- 3. taxa
def mdot(tm: Tempo, k0, k1):
    d = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    a, b = tm.banco(k0), tm.banco(k1)
    return float(np.nanmean(d[(d[:, 0] >= a) & (d[:, 0] <= b), 1])) * RHO * 1e-3


def janelas(H, tm: Tempo, k0, k1, n):
    ks = np.linspace(k0, k1, n + 1)
    out = []
    for a, b in zip(ks[:-1], ks[1:]):
        Ha, Hb = RL.H_em(H, a), RL.H_em(H, b)
        out.append(dict(t=[float(tm.desde_ign(tm.banco(a))), float(tm.desde_ign(tm.banco(b)))],
                        r=(Hb - Ha) / (2 * (b - a) * CAM * tm.s), G=float(R.G_ox(mdot(tm, a, b), 0.5 * (Ha + Hb)))))
    return out


def resumo(P):
    G = np.array([p["G"] for p in P]); r = np.array([p["r"] for p in P])
    aj = RG.ajuste_proprio(G, r)
    return dict(G=G.tolist(), r=r.tolist(), t=[p["t"] for p in P], a=aj["a"], n=aj["n"], r2=aj["r2"])


def interrupcao(B, tm: Tempo):
    """1ª interrupção de vazão (vazão abaixo de 1 mL/s entre os dois patamares), em s desde a ignição."""
    t = B["t_planilha_s"]; F = np.nan_to_num(B["vazao_mL_s"])
    m = (t > 90) & (t < 105) & (F < 1.0)
    return float(tm.desde_ign(t[m][0])), float(tm.desde_ign(t[m][-1] + 0.1))


def fim_analise(aj) -> float:
    """Fim da análise: o salto de vazão para 4,7 mL/s (registro do banco)."""
    return aj["banco"]["salto"] - 0.1


def janelas_desde(H, tm: Tempo, t0, fim_banco, dur=DUR):
    k0, k1 = float(tm.k(tm.ign + t0)), float(tm.k(fim_banco))
    n = max(3, int(round((fim_banco - tm.ign - t0) / dur)))
    return janelas(H, tm, k0, k1, n)


def variantes(tm: Tempo, cri, fim, inter) -> dict:
    """Janelas de ~20 s que começam em cada critério e vão até o salto de vazão, mais a grade
    usada em 07/11 (11 janelas desde a amostra 79 até 241 s, sem as 4 primeiras), para comparar."""
    res = {}
    for nome_s, (x0, x1) in SECOES.items():
        H = RL.serie_H(x0, x1)
        v = {"grade de 07/11 (começa em 91,5 s, dentro da interrupção)":
             resumo(janelas(H, tm, 78, float(tm.k(FIM_ANTIGO)), 11)[4:])}
        for nc, tc in cri.items():
            if nc.startswith("_"):
                continue
            d = resumo(janelas_desde(H, tm, tc, fim))
            d["dentro_da_interrupcao"] = bool(inter[0] - 0.5 <= tc <= inter[1])
            v[nc] = d
        res[nome_s] = v
    return res


def sensibilidade(tm: Tempo, inicio, fim) -> dict:
    """Início de inicio a inicio+7 s (passo 0,4 s), janelas de 15, 20 e 25 s, fim no salto de vazão
    ou em 241 s: quanto o resultado depende dessas escolhas."""
    out = {}
    for nome_s, (x0, x1) in SECOES.items():
        H = RL.serie_H(x0, x1)
        linhas = []
        for f in (fim, FIM_ANTIGO):
            for dur in (15.0, 20.0, 25.0):
                for t0 in np.arange(inicio, inicio + 7.01, 0.4):
                    d = resumo(janelas_desde(H, tm, float(t0), f, dur))
                    linhas.append(dict(fim=f, dur=dur, t0=float(t0), n=d["n"], r2=d["r2"]))
        out[nome_s] = linhas
    return out


def faixa(vals):
    v = np.asarray(vals, float)
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))


# ----------------------------------------------------------------------------- figuras
def fig_fator(ch, rb, B, aj, tm: Tempo):
    fig = plt.figure(figsize=(17, 13.5))
    g = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.05, 1.0], hspace=0.42, wspace=0.55)
    tb, F = B["t_planilha_s"], B["vazao_mL_s"]
    tcam = np.arange(len(rb)) * CAM
    for lin, (s, T0, tit) in enumerate(((1.0, aj["banco"]["volta 1"] - aj["video"]["volta 1"],
                                         "tempo marcado pela câmera, sem fator (alinhado na 1ª volta da vazão)"),
                                        (aj["s"], aj["T0"], f"tempo marcado pela câmera × {aj['s']:.3f} (fator medido)"))):
        ax = fig.add_subplot(g[lin, :])
        ax.plot(T0 + s * tcam, rb, color="#2f6bff", lw=0.8, label="cor da chama no vídeo (R − B médio no grão)")
        ax2 = ax.twinx()
        ax2.plot(tb, F * RHO, color="k", lw=1.0, label="vazão de oxidante, registro do banco (g/s)")
        for n in aj["banco"]:
            ax.axvline(aj["banco"][n], color="#00c853", lw=1.4, ls="--")
            ax.text(aj["banco"][n], 228, n, color="#00a040", ha="center", fontsize=9, weight="bold")
        ax.set_xlim(0, 305); ax.set_ylim(-20, 240); ax2.set_ylim(0, 7)
        ax.set_ylabel("R − B"); ax2.set_ylabel("vazão (g/s)")
        desvio = max(abs(aj["banco"][n] - (T0 + s * aj["video"][n])) for n in aj["banco"])
        ax.set_title(tit + (f"   →  os eventos da chama ficam até {desvio:.0f} s fora das linhas verdes" if lin == 0 else
                            f"   →  os 4 eventos caem a ≤ {max(abs(x) for x in aj['residuos'].values()):.2f} s das linhas verdes"),
                     fontsize=10.5, loc="left")
        ax.grid(alpha=0.3)
        if lin == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            fig.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncols=2, frameon=False)
        if lin == 1:
            ax.set_xlabel("tempo do registro do banco (s)")
    for i, n in enumerate(aj["video"]):
        ax = fig.add_subplot(g[2, i])
        v = ch[n]
        tv = aj["T0"] + aj["s"] * v[:, 0]
        col = 2 if n == "salto" else 1          # o salto foi achado no brilho, os outros na cor
        ax.plot(tv, v[:, col], ".-", color="#2f6bff", ms=2.5, lw=0.8)
        ax.set_ylabel("brilho médio" if n == "salto" else "R − B", fontsize=8)
        ax2 = ax.twinx()
        m = (tb >= tv.min()) & (tb <= tv.max())
        ax2.step(tb[m], (B["P_leito"][m] if n == "fim" else F[m] * RHO), where="post", color="k", lw=1.2)
        ax2.set_ylabel("pressão do leito (bar)" if n == "fim" else "vazão (g/s)", fontsize=8)
        ax.axvline(aj["banco"][n], color="#00c853", lw=1.5, ls="--")
        ax.axvline(aj["T0"] + aj["s"] * aj["video"][n], color="#d500f9", lw=1.5, ls=":")
        ax.set_title(f"{n}: banco {aj['banco'][n]:.1f} s | vídeo {aj['video'][n]:.2f} s da câmera\n"
                     f"resíduo {aj['residuos'][n]:+.2f} s", fontsize=9, loc="left")
        ax.set_xlabel("tempo do registro do banco (s)", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Fator do tempo da câmera medido com 4 eventos de vazão: s = {aj['s']:.4f} ± {aj['erro_s']:.4f} "
                 f"(30 quadros por segundo no carimbo contra 23,976 gravados: 1,2513)", fontsize=12.5)
    fig.savefig(SAIDA / "1_fator_do_tempo.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def fig_quadros(aj):
    """Validação a olho: um quadro antes e um depois de cada evento, no vídeo."""
    cont, _ = quadros_camera()
    cap = cv2.VideoCapture(str(VIDEO))
    antes_depois = {"volta 1": (84.6, 85.3), "volta 2": (164.6, 165.1), "salto": (203.0, 205.5), "fim": (240.3, 240.8)}
    linhas = []
    for n, (a, b) in antes_depois.items():
        par = []
        for tc, rot in ((a, "antes"), (b, "depois")):
            f = int(np.searchsorted(cont, tc * 30))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
            im = cv2.resize(fr[Y0 - 10:Y1 + 10, X0:X1], None, fx=0.42, fy=0.42, interpolation=cv2.INTER_AREA)
            barra = np.zeros((40, im.shape[1], 3), np.uint8)
            tb = aj["T0"] + aj["s"] * cont[f] / 30
            cv2.putText(barra, f"{n}, {rot}: camera {cont[f] / 30:.2f} s -> banco {tb:.1f} s (evento no banco: {aj['banco'][n]:.1f} s)",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            par.append(np.vstack([barra, im]))
        linhas.append(np.hstack([par[0], np.full((par[0].shape[0], 8, 3), 255, np.uint8), par[1]]))
    cap.release()
    cv2.imwrite(str(SAIDA / "2_quadros_dos_eventos.jpg"), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 88])




def fig_corte(B, tm: Tempo, cri, inter, corte):
    t = tm.desde_ign(B["t_planilha_s"])
    fig, axs = plt.subplots(3, 1, figsize=(15, 12.5), sharex=True,
                            gridspec_kw=dict(hspace=0.13, height_ratios=[1, 1.2, 1.25]))
    ax = axs[0]
    ax.step(t, B["vazao_mL_s"] * RHO, where="post", color="k", lw=1)
    ax.set_ylabel("vazão (g/s)"); ax.set_ylim(0, 3.5)
    ax.set_title("1. vazão de oxidante: degraus 0,96 → 1,2 → 1,6 g/s, interrupção, e o 1º trecho de vazão constante começa em "
                 f"{cri['vazão constante (volta completa depois da 1ª interrupção)']:.1f} s", fontsize=10, loc="left")
    ax = axs[1]
    T2 = uniform_filter1d(B["T_leito2_C"], 11)
    ax.plot(t, T2, color="#e91e63", lw=1.6, label="termopar do leito catalítico (T_leito2)")
    ax.plot(t, uniform_filter1d(B["T_entrada_camara_C"], 11), color="#2f6bff", lw=1.3, label="entrada da câmara de combustão")
    ax.set_ylabel("temperatura (°C)"); ax.set_ylim(250, 720); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"2. leito catalítico e entrada da câmara; patamar medido no leito {cri['_patamar_leito']:.0f} °C",
                 fontsize=10, loc="left")
    ax = axs[2]
    k = np.arange(963)
    tk = tm.desde_ign(tm.banco(k))
    for (x0, x1), c, rot in (((60, 120), "#2f6bff", "2ª metade (60–120 mm)"), ((10, 120), "#00acc1", "grão todo (10–120 mm)")):
        ax.plot(tk, RL.serie_H(x0, x1), color=c, lw=1.4, label=f"altura média da porta pela linha 2b, {rot}")
    ax.axvline(91.5, color="0.4", lw=1.6, ls="-.")
    ax.text(91.0, 22, "início da janela de 07/11 (91,5 s)", rotation=90, fontsize=8.5, color="0.3", ha="right", va="top")
    ax.set_ylim(6, 30); ax.set_ylabel("altura da porta (mm)"); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("3. o que a linha faz na interrupção: a porta parece encolher 1,2 mm (impossível); ali a linha não está na "
                 "superfície (quadros no arquivo 4)", fontsize=10, loc="left")
    ax.set_xlabel("tempo desde a ignição (s): tempo marcado pela câmera × fator medido = tempo do registro do banco")
    cores = ["#00c853", "#ff9100", "#e91e63", "#9c27b0", "#2f6bff"]
    nomes = [n for n in cri if not n.startswith("_")]
    for n, c in zip(nomes, cores):
        for a in axs:
            a.axvline(cri[n], color=c, lw=1.4, ls="--")
    for n, c, y in zip(nomes, cores, np.linspace(0.97, 0.62, len(nomes))):
        axs[0].text(0.40, y, f"{cri[n]:5.1f} s  {n}", transform=axs[0].transAxes, color=c, fontsize=9, weight="bold",
                    va="top", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
    for a in axs:
        a.axvspan(min(cri[n] for n in nomes), max(cri[n] for n in nomes), color="#fff59d", alpha=0.5, zorder=0)
        a.axvspan(*inter, color="#ff8a80", alpha=0.45, zorder=0)
        a.axvline(corte, color="k", lw=2.4)
        a.grid(alpha=0.3)
    axs[0].text(corte + 1, 0.12, f"corte: {corte:.1f} s (vazão constante, leito a 95 %)", fontsize=9.5, weight="bold")
    axs[0].text(np.mean(inter), 2.0, "interrupção", ha="center", va="center", rotation=90, fontsize=8.5, color="#c62828",
                weight="bold")
    ins = axs[2].inset_axes([0.55, 0.05, 0.43, 0.42])
    for (x0, x1), c in (((60, 120), "#2f6bff"), ((10, 120), "#00acc1")):
        ins.plot(tk, RL.serie_H(x0, x1), ".-", color=c, ms=3, lw=1.2)
    ins.axvspan(*inter, color="#ff8a80", alpha=0.45); ins.axvline(91.5, color="0.4", lw=1.4, ls="-.")
    ins.axvline(corte, color="k", lw=2)
    ins.set_xlim(78, 108); ins.set_ylim(8, 15); ins.grid(alpha=0.3); ins.tick_params(labelsize=8)
    ins.set_title("zoom: 78 a 108 s", fontsize=8.5, loc="left")
    axs[2].set_xlim(0, 265)
    fig.suptitle("Corte do aquecimento: os critérios do sistema de oxidante caem entre "
                 f"{min(cri[n] for n in nomes):.0f} e {max(cri[n] for n in nomes):.0f} s depois da ignição (faixa amarela); "
                 "vermelho = 1ª interrupção de vazão", fontsize=12, y=0.975)
    fig.subplots_adjust(top=0.95)
    fig.savefig(SAIDA / "3_criterio_do_corte.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_linha(tm: Tempo):
    """Validação a olho: a linha 2b antes, durante e depois da 1ª interrupção."""
    z = np.load(R.OUT / "runs/nenhum_d4/saida.npz")
    Yb, Yc = z["baixo"].astype(float), z["cima"].astype(float)
    cap = cv2.VideoCapture(str(VIDEO))
    xc = 20 * (np.arange(93) + 0.5)
    x60 = int(60 / MM + 40 - X0)
    ims = []
    for tt in (86.0, 88.5, 90.0, 91.5, round(E.JANELA_ANALISE_S[0], 1), 95.0):
        kk = int(round(float(tm.k(tm.ign + tt))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10 * kk); ok, f = cap.read()
        im = np.ascontiguousarray(f[Y0:Y1, X0:X1])
        for Y in (Yb, Yc):
            p = np.stack([xc, Y[kk]], 1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(im, [p], False, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(im, [p], False, (255, 229, 0), 2, cv2.LINE_AA)
        cv2.line(im, (x60, 0), (x60, im.shape[0]), (255, 0, 255), 2)
        im = cv2.resize(im, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        barra = np.zeros((34, im.shape[1], 3), np.uint8)
        cv2.putText(barra, f"{tt:.1f} s desde a ignicao (amostra {kk + 1})" + ("   VAZAO INTERROMPIDA" if 87.8 <= tt <= 92.7 else ""),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        ims.append(np.vstack([barra, im]))
    cap.release()
    esp = np.full((ims[0].shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(str(SAIDA / "4_linha_na_interrupcao.jpg"),
                np.vstack([np.hstack([ims[i], esp, ims[i + 1]]) for i in (0, 2, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def fig_resultado(var, sens):
    nomes_s = list(SECOES)
    fig, axs = plt.subplots(1, len(nomes_s), figsize=(20, 10), sharey=True)
    Gg = np.geomspace(2, 10, 50)
    cores = ["0.55", "#00c853", "#ff9100", "#e91e63", "#9c27b0", "#2f6bff"]
    for ax, ns in zip(axs, nomes_s):
        for (nv, d), c in zip(var[ns].items(), cores):
            ruim = nv.startswith("grade") or d.get("dentro_da_interrupcao", False)
            ax.plot(d["G"], d["r"], "x" if ruim else "o", color=c, ms=7, mec=c, zorder=5, alpha=0.8 if ruim else 1)
            gg = np.geomspace(min(d["G"]), max(d["G"]), 20)
            ax.plot(gg, d["a"] * gg ** d["n"], ":" if ruim else "--", color=c, lw=1.2 if ruim else 1.6,
                    label=("[descartada] " if ruim else "") + f"{nv.split(' (')[0][:34]}: n {d['n']:.2f}, R² {d['r2']:.2f}")
        S = [x for x in sens[ns] if x["fim"] != FIM_ANTIGO]
        n_, r2_ = faixa([x["n"] for x in S]), faixa([x["r2"] for x in S])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2.5, 9); ax.set_ylim(0.03, 0.13)
        ax.grid(alpha=0.3, which="both"); ax.set_xlabel("G_ox (kg/m²s)")
        ax.set_title(f"{ns}\nem {len(S)} escolhas de início e de tamanho de janela:\nn {n_[0]:.2f} ({n_[1]:.2f} a {n_[2]:.2f}), "
                     f"R² {r2_[0]:.2f} ({r2_[1]:.2f} a {r2_[2]:.2f})", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False)
    axs[0].set_ylabel("taxa de regressão (mm/s)")
    fig.suptitle("Taxa com o fator medido, janelas de ~20 s do corte até o salto de vazão (n = expoente); "
                 "x = começa dentro da interrupção, descartada", fontsize=11.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.27, wspace=0.06)
    fig.savefig(SAIDA / "5_resultado_por_criterio.png", dpi=110)
    plt.close(fig)


def fig_sensibilidade(sens, corte):
    nomes_s = list(SECOES)
    fig, axs = plt.subplots(2, len(nomes_s), figsize=(18, 8.5), sharex=True, sharey="row")
    for col, ns in enumerate(nomes_s):
        for f, ls in ((None, "-"), (FIM_ANTIGO, ":")):
            for dur, c in ((15.0, "#ff9100"), (20.0, "#2f6bff"), (25.0, "#00c853")):
                S = [x for x in sens[ns] if x["dur"] == dur and ((x["fim"] == FIM_ANTIGO) == (f == FIM_ANTIGO))]
                t0 = [x["t0"] for x in S]
                lab = f"janelas de {dur:.0f} s, " + ("até 241 s (como em 07/11)" if f else "até o salto de vazão")
                axs[0, col].plot(t0, [x["n"] for x in S], ls, color=c, lw=1.5, label=lab)
                axs[1, col].plot(t0, [x["r2"] for x in S], ls, color=c, lw=1.5)
        axs[0, col].set_title(ns, fontsize=11, loc="left")
        axs[1, col].set_xlabel("início da 1ª janela (s desde a ignição)")
        for a in axs[:, col]:
            a.grid(alpha=0.3)
    axs[0, 0].set_ylabel("expoente n da reta"); axs[1, 0].set_ylabel("R² da reta")
    axs[0, 0].set_ylim(-0.4, 1.3); axs[1, 0].set_ylim(0, 1)
    axs[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Quanto o resultado depende de escolhas que não têm critério físico (início dentro da faixa dos critérios, "
                 "tamanho e fim das janelas)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SAIDA / "6_sensibilidade.png", dpi=110)
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    B = banco()
    ch = chama_eventos()
    rb = chama_queima()
    aj = ajustar(eventos_video(ch), eventos_banco(B))
    tm = Tempo(aj["s"], aj["T0"])
    print(f"fator s = {aj['s']:.4f} ± {aj['erro_s']:.4f}; T0 = {aj['T0']:.2f} s; ignição no banco {tm.ign:.2f} s; "
          f"resíduo máximo com s = 1: {aj['residuo_max_com_s_1']:.1f} s")
    print("  resíduos (s):", {k: round(v, 2) for k, v in aj["residuos"].items()})
    print("  sem um evento:", {k: round(v, 4) for k, v in aj["sem_um"].items()})
    cri = criterios(B, tm)
    inter = interrupcao(B, tm)
    corte = cri["vazão constante (volta completa depois da 1ª interrupção)"]
    fim = fim_analise(aj)
    print(f"1ª interrupção de vazão: {inter[0]:.1f} a {inter[1]:.1f} s desde a ignição; fim da análise {fim:.1f} s do banco "
          f"({float(tm.desde_ign(fim)):.1f} s desde a ignição)")
    print("critérios (s desde a ignição):")
    for n, v in cri.items():
        print(f"  {v:7.1f}  {n}")
    var = variantes(tm, cri, fim, inter)
    sens = sensibilidade(tm, corte, fim)
    for ns, v in var.items():
        print(ns)
        for nv, d in v.items():
            print(f"  {nv:58s} {len(d['G']):2d} janelas desde {d['t'][0][0]:5.1f} s: {d['a']:.4f} G^{d['n']:.2f} "
                  f"R² {d['r2']:.2f}" + ("  [dentro da interrupção]" if d.get("dentro_da_interrupcao") else ""))
        for nome_f, sel in (("até o salto", lambda x: x["fim"] != FIM_ANTIGO), ("até 241 s", lambda x: x["fim"] == FIM_ANTIGO)):
            S = [x for x in sens[ns] if sel(x)]
            print(f"  sensibilidade {nome_f}: n %.2f (%.2f a %.2f), R² %.2f (%.2f a %.2f)"
                  % (faixa([x["n"] for x in S]) + faixa([x["r2"] for x in S])))
    (SAIDA / "relogio_e_corte.json").write_text(json.dumps(dict(fator=aj, criterios=cri, interrupcao=inter, corte=corte,
                                                               fim_banco=fim, variantes=var, sensibilidade=sens),
                                                          indent=1, ensure_ascii=False))
    fig_fator(ch, rb, B, aj, tm)
    fig_quadros(aj)
    fig_corte(B, tm, cri, inter, corte)
    fig_linha(tm)
    fig_resultado(var, sens)
    fig_sensibilidade(sens, corte)


if __name__ == "__main__":
    main()
