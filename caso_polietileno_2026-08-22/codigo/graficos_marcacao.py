"""Gráficos que acompanham a marcação.

Recebe os pontos do marcador e refaz as contas a cada alteração. A altura da porta
de cada quadro é a parede de baixo menos a de cima, coluna a coluna, e só contam os
quadros marcados nas duas faces. A taxa de um intervalo usa apenas as colunas
marcadas nos dois quadros dele, para a cobertura diferente entre quadros não virar
taxa falsa. Taxa negativa fica como está: o programa não corrige nada.

O tempo é o marcado pela câmera (amostra vezes o tempo de câmera por amostra), e o fluxo
de cada seção usa a altura dessa seção, com a vazão do registro no mesmo intervalo.
Os gráficos mostram só a marcação ativa (A ou B, tecla m no marcador).
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import ensaio as E

MM = E.MM_POR_PX               # mm/px
L_MM = E.LARGURA_PORTA_MM      # largura da porta
RHO = E.DENSIDADE_OXIDANTE     # g/mL do oxidante
DT = E.CAM                     # s por amostra, tempo marcado pela câmera
K_IGN, T_IGN_PLAN = E.AMOSTRA_IGNICAO, E.T_IGNICAO_REGISTRO   # registro e vídeo se encontram na ignição
X0_GRAO = 40.0                 # px do quadro no início do grão
XG = np.arange(40.0, 1891.0, 10.0)
XMM = (XG - X0_GRAO) * MM
VAO = 120.0                    # px sem ponto além do qual a parede não é interpolada
MIN_COMUM = 0.2                # fração mínima da seção marcada nos dois quadros
SECOES = {"grão inteiro": (0.0, 121.0), "primeira metade": (0.0, 60.5), "segunda metade": (60.5, 121.0)}
COR = {"grão inteiro": "#111111", "primeira metade": "#7b3fbf", "segunda metade": "#2a9d3f"}
CAMADAS = ("A", "B")
COR_GRAO = {"A": "#2a9d3f", "B": "#2f7fe6"}
DT_MIN_INTERVALO = 5.0         # s: intervalo mais curto que entra nos gráficos de taxa por intervalo
VIRG = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def t_camera(k):
    """Tempo marcado pela câmera na amostra k (s)."""
    return np.asarray(k, float) * DT


def carimbo(t: float) -> str:
    """Tempo marcado pela câmera como aparece no quadro (minutos:segundos:quadros)."""
    fps = int(round(E.FPS_CAMERA))
    v = int(round(t * fps))
    return f"{v // (60 * fps):02d}:{v // fps % 60:02d}:{v % fps:02d}"


def parede(pontos: list[tuple[float, float]]) -> np.ndarray:
    """Parede na grade do grão; NaN fora dos pontos e nos vãos grandes."""
    p = np.array(sorted(pontos), float)
    if len(p) < 2:
        return np.full(XG.shape, np.nan)
    y = np.interp(XG, p[:, 0], p[:, 1])
    i = np.clip(np.searchsorted(p[:, 0], XG), 1, len(p) - 1)
    y[~((XG >= p[0, 0]) & (XG <= p[-1, 0]) & (p[i, 0] - p[i - 1, 0] <= VAO))] = np.nan
    return y


def _vazao():
    d = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1] * RHO


def espacar(ks: list[int], dt_min: float) -> list[int]:
    """Quadros para os intervalos, com pelo menos dt_min s entre eles. Dois quadros
    muito próximos dariam uma taxa feita só de erro de clique; o último quadro sempre
    fica, no lugar do anterior se estiver perto demais."""
    out: list[int] = []
    for k in ks:
        if not out or (k - out[-1]) * DT >= dt_min:
            out.append(k)
        elif k == ks[-1] and len(out) > 1:
            out[-1] = k
    return out


def contas(ancoras: list[dict], secoes: dict = SECOES, vazao=None, faixa=None, dt_min: float = 0.0) -> dict:
    """Série de cada seção a partir dos pontos do marcador.

    Com `faixa`, só entram os quadros dentro dela. Com `dt_min`, os intervalos pulam
    quadros a menos de dt_min s do anterior (a taxa média desde o primeiro quadro usa
    todos).
    """
    t_reg, m_reg = vazao or _vazao()
    faces = {"baixo": {}, "cima": {}}
    for p in ancoras:
        if faixa and not faixa[0] <= int(p["k"]) <= faixa[1]:
            continue
        faces[p["face"]].setdefault(int(p["k"]), []).append((float(p["x"]), float(p["y"])))
    pares = sorted(set(faces["baixo"]) & set(faces["cima"]))
    H = {k: (parede(faces["baixo"][k]) - parede(faces["cima"][k])) * MM for k in pares}
    pares = [k for k in pares if np.isfinite(H[k]).any()]
    so_uma = sorted(set(faces["baixo"]) ^ set(faces["cima"]))

    def fluxo(k0, k1, altura):
        s = (t_reg >= T_IGN_PLAN + (k0 - K_IGN) * DT) & (t_reg <= T_IGN_PLAN + (k1 - K_IGN) * DT)
        m = float(m_reg[s].mean()) if s.any() else float(np.interp(T_IGN_PLAN + (0.5 * (k0 + k1) - K_IGN) * DT, t_reg, m_reg))
        return m * 1e-3 / (L_MM * 1e-3 * altura * 1e-3)

    def trecho(k0, k1, sel):
        ok = sel & np.isfinite(H[k0]) & np.isfinite(H[k1])
        if ok.sum() < MIN_COMUM * sel.sum():
            return None
        a, b = float(H[k0][ok].mean()), float(H[k1][ok].mean())
        r = (b - a) / (2 * (k1 - k0) * DT)
        G = fluxo(k0, k1, 0.5 * (a + b))
        return dict(k0=k0, k1=k1, t0=float(t_camera(k0)), t1=float(t_camera(k1)), r=r, G=G)

    out = dict(pares=pares, so_uma=so_uma, secoes={})
    for nome, (a, b) in secoes.items():
        sel = (XMM >= a) & (XMM < b)
        ks = [k for k in pares if np.isfinite(H[k][sel]).sum() >= MIN_COMUM * sel.sum()]
        ke = espacar(ks, dt_min)
        intervalos = [q for q in (trecho(i, j, sel) for i, j in zip(ke[:-1], ke[1:])) if q]
        media = [q for q in (trecho(ks[0], k, sel) for k in ks[1:]) if q] if ks else []
        out["secoes"][nome] = dict(k=ks, t=t_camera(ks).tolist(), H=[float(np.nanmean(H[k][sel])) for k in ks],
                                   intervalos=intervalos, media=media)
    return out


def br(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def titulo(c: dict, ativa: str) -> str:
    """Duas linhas: o que está marcado e a taxa média do grão inteiro."""
    l1 = f"Marcação {ativa}: {len(c['pares'])} quadros com as duas faces"
    if c["so_uma"]:
        l1 += f", {len(c['so_uma'])} com uma face só"
    m = c["secoes"]["grão inteiro"]["media"]
    if m:
        q = m[-1]
        l2 = (f"{carimbo(q['t0'])} a {carimbo(q['t1'])}: taxa média do grão inteiro {br(q['r'])} mm/s "
              f"em G {br(q['G'], 2)} kg/m²s")
    else:
        l2 = "Taxa: marque pelo menos dois quadros nas duas faces"
    return "\n".join((l1, l2))


class Painel:
    """Quatro gráficos empilhados dentro de uma janela Tk, só da marcação ativa."""

    def __init__(self, mestre, largura: int, altura: int, faixa: tuple[int, int] | None = None):
        self.faixa = faixa
        self.vazao = _vazao()
        self.fig = Figure(figsize=(largura / 100, altura / 100), dpi=100)
        self.ax = self.fig.subplots(4, 1)
        self.fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.055, hspace=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mestre)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.atualizar([])

    def _limites_tempo(self, c: dict):
        ts = [t for s in c["secoes"].values() for t in s["t"]]
        if self.faixa:
            ts += [float(t_camera(self.faixa[0])), float(t_camera(self.faixa[1]))]
        if not ts:
            return 0.0, 1.0
        return min(ts) - 2, max(ts) + 2

    def atualizar(self, ancoras: list[dict], ativa: str = "A") -> dict:
        c = contas([p for p in ancoras if p.get("camada", "A") == ativa], vazao=self.vazao, faixa=self.faixa,
                   dt_min=DT_MIN_INTERVALO)
        cor = COR_GRAO[ativa]
        aP, aI, aM, aG = self.ax
        for a in self.ax:
            a.clear()
            a.grid(alpha=0.25)
            a.tick_params(labelsize=7.5)
            a.yaxis.set_major_formatter(VIRG)
            a.xaxis.set_major_formatter(VIRG)
            for lado in ("top", "right"):
                a.spines[lado].set_visible(False)
        self.fig.suptitle(titulo(c, ativa), fontsize=7.4, y=0.997, linespacing=1.35)
        xl = self._limites_tempo(c)
        estilos = (("grão inteiro", "grão inteiro", cor, 2.0, 3.6),
                   ("primeira metade", "1ª metade", COR["primeira metade"], 0.9, 2.4),
                   ("segunda metade", "2ª metade", COR["segunda metade"], 0.9, 2.4))

        # 1. altura média da porta
        for nome, rot, cn, lw, ms in estilos:
            s = c["secoes"][nome]
            if s["k"]:
                aP.plot(s["t"], s["H"], "-o", ms=ms, lw=lw, color=cn, label=rot)
        aP.set_ylabel("altura (mm)", fontsize=7.5)
        aP.set_title("Altura média da porta", loc="left", fontsize=9)

        # 2. taxa em cada intervalo
        for nome, rot, cn, lw, _ in estilos:
            for i, q in enumerate(c["secoes"][nome]["intervalos"]):
                aI.hlines(q["r"], q["t0"], q["t1"], color=cn, lw=lw + 0.6, label=rot if i == 0 else None)
        aI.axhline(0, color="#555555", lw=0.7)
        aI.set_ylabel("mm/s", fontsize=7.5)
        aI.set_title("Taxa em cada intervalo", loc="left", fontsize=9)

        # 3. taxa média desde o primeiro quadro marcado
        for nome, rot, cn, lw, ms in estilos:
            m = c["secoes"][nome]["media"]
            if m:
                aM.plot([q["t1"] for q in m], [q["r"] for q in m], "-o", ms=ms, lw=lw, color=cn, label=rot)
        aM.axhline(0, color="#555555", lw=0.7)
        aM.set_ylabel("mm/s", fontsize=7.5)
        aM.set_title("Taxa média desde o primeiro quadro marcado", loc="left", fontsize=9)

        for a in (aP, aI, aM):
            a.set_xlim(*xl)
            a.set_xlabel("tempo marcado pela câmera (s)", fontsize=7.5, labelpad=1)
            y0, y1 = a.get_ylim()
            a.set_ylim(y0, y1 + 0.45 * (y1 - y0))   # espaço para a legenda
            if a.get_legend_handles_labels()[0]:
                a.legend(frameon=False, fontsize=6.6, ncols=3, loc="upper left")
        self._fluxo(aG, c, ativa)
        self.canvas.draw_idle()
        return c

    def _fluxo(self, a, c: dict, ativa: str) -> None:
        cor = COR_GRAO[ativa]
        pts = []
        q = c["secoes"]["grão inteiro"]["intervalos"]
        if q:
            g, r = [x["G"] for x in q], [x["r"] for x in q]
            a.plot(g, r, "-o", ms=3, lw=0.7, color=cor, alpha=0.75, label="cada intervalo")
            pts += list(zip(g, r))
            if len(q) <= 14:
                for n, (x, y) in enumerate(zip(g, r), 1):
                    a.annotate(str(n), (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6.2, color=cor)
        m = c["secoes"]["grão inteiro"]["media"]
        if m:
            j = m[-1]
            a.scatter(j["G"], j["r"], marker="s", s=46, color=cor, edgecolors="black", lw=0.6, zorder=7,
                      label="do primeiro ao último quadro")
            pts.append((j["G"], j["r"]))
        if pts:
            P = np.array(pts)
            g0, g1 = P[:, 0].min() - 0.3, P[:, 0].max() + 0.3
            r0, r1 = min(0.0, P[:, 1].min() - 0.01), P[:, 1].max() + 0.01
            a.set_xlim(g0, g1)
            a.set_ylim(r0, r1 + 0.45 * (r1 - r0))       # espaço para a legenda
            a.legend(frameon=False, fontsize=6.6, ncols=2, loc="upper left")
        a.axhline(0, color="#555555", lw=0.7)
        a.set_xlabel("fluxo mássico de oxidante (kg/m²s)", fontsize=7.5, labelpad=1)
        a.set_ylabel("mm/s", fontsize=7.5)
        a.set_title("Taxa por fluxo mássico", loc="left", fontsize=9)
