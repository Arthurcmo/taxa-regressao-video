"""
Nucleo do ajuste guiado: os dados, a semantica dos pontos e o recalculo.

Fica separado da interface de proposito. A semantica de "ponto marcado" e de
"quadro ignorado" ja mordeu uma vez por existir em duas copias que
divergiram, entao agora ela mora aqui e a interface, o revisor em HTML e o
`reajustar.py` todos chamam o mesmo codigo.

OS PONTOS MARCADOS SAO A BASE DO CALCULO, nao um retoque quadro a quadro.

A primeira versao errou justamente aqui: cada ponto prendia SO o quadro em que
foi clicado, e entre um ponto e outro o Viterbi continuava livre para seguir
qualquer vale. Marcar cinco pontos ao longo da queima nao mudava quase nada --
os picos e o comeco ruim sobreviviam intactos, porque nenhum ponto falava sobre
eles. Marcar mais nao resolvia: resolveria marcar tudo, que e o que ninguem quer
fazer.

Agora os pontos definem uma TRAJETORIA DE REFERENCIA para o ensaio inteiro:

  referencia(t, x) = forma(t, x) + deslocamento(t, x)

`forma` e a tendencia que a imagem ja dava, suavizada no tempo (mediana movel +
Savitzky-Golay) para nao carregar pico nenhum para dentro da referencia.
`deslocamento` vem dos seus pontos: em cada quadro marcado, a diferenca entre
onde voce clicou e a forma; interpolado em x entre os pontos daquele quadro e
em t entre os quadros marcados, com extrapolacao plana nas pontas. Um ponto so
desloca a linha inteira; varios pontos no mesmo quadro fazem a linha seguir o
perfil que voce marcou.

Sobre essa referencia o Viterbi anda dentro de um CORREDOR de raio `rigidez`, e
paga um custo proporcional ao quanto se afasta dela. Ou seja: onde a imagem tem
evidencia boa ela ainda manda no detalhe, e onde nao tem, quem manda e a sua
trajetoria. Com `rigidez` pequena o resultado converge para exatamente a curva
que voce marcou; com ela grande, volta a ser o automatico de antes.

No quadro exatamente marcado a trajetoria continua proibida de sair de uma
janela de `tol` px ao redor do ponto -- ali voce viu, entao ali nao se discute.
(Bonus mole nunca serviu para isso, e foi medido: um bonus de altura 12 sobre
um escore que vale no maximo 1,5 ainda perdia para o custo de sair do vale
errado por uma duzia de quadros, e o ponto era ignorado.)

QUADRO IGNORADO tem o escore zerado. Nao e apagar: o Viterbi continua passando
por ali, so que sem evidencia, e ai a unica coisa que decide e a penalidade de
salto -- ele mantem a linha e liga os vizinhos que tem evidencia por uma rampa
exata. Quadro ilegivel para de entrar no ajuste como chute.
"""
from __future__ import annotations
import json, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "codigo"))
from detectar_interfaces import (escore, viterbi, refinar, largura_vale,   # noqa: E402
                                 suavizar_x, ajuste_robusto_x, colunas_confiaveis,
                                 corredor_no_escore, quadros_com_flare)
from taxa_regressao import ajuste_linear                               # noqa: E402

FACES = ("cima", "baixo")
CH = {"cima": "y_sup", "baixo": "y_inf"}
BASE = {"cima": "sup", "baixo": "inf"}


@dataclass
class Par:
    """
    `obediencia` e um controle so, de 0 a 1, porque os dois numeros que ele
    resolve nunca fazem sentido separados: de nada adianta um corredor
    estreito com custo fraco, nem custo forte com corredor largo. Ele fecha o
    corredor (120 -> 6 px) e sobe o custo de sair da referencia (0,15 -> 5,0)
    ao mesmo tempo. Em 0 o resultado e o automatico de sempre; em 1 a
    trajetoria que voce marcou e a resposta, e a imagem so decide o subpixel.

    As duas curvas sao quadraticas de proposito. Com mapeamento linear o meio
    da escala nao fazia nada de util -- em 0,6 a imagem ainda ganhava e o
    deslocamento pedido de 10 px saia como 2 px, o que da a impressao (certa)
    de que marcar nao adianta.
    """
    dmax: int = 2
    lam: float = 0.5
    tol: float = 1.0          # meia-janela permitida no quadro marcado [px]
    obediencia: float = 1.0   # 0 = suas marcas sao sugestao, 1 = sao a resposta
    suav_t: int = 41          # janela da suavizacao da forma, em amostras
    jan_x: int = 11           # janela da regularizacao espacial, em colunas
    grau: int = 4             # grau do ajuste rigido em x
    # corredores espaciais das passadas 2 e 3: (raio [px], custo)
    esp2: tuple = (40.0, 1.5)
    esp3: tuple = (12.0, 2.5)
    mono: bool = True         # a queima e irreversivel: so recua, nunca volta
    peso_flare: float = 0.2   # peso do escore em quadro estourado de luz
    margem: int = 14


def corredor(obediencia: float) -> tuple[float, float]:
    """(raio do corredor [px], custo de sair dele) a partir de `obediencia`."""
    o = float(np.clip(obediencia, 0.0, 1.0))
    return 6.0 + 114.0 * (1 - o) ** 2, 0.15 + 5.0 * o ** 2


def suavizar_t(Y: np.ndarray, jan: int) -> np.ndarray:
    """
    Tendencia temporal de cada coluna: mediana movel (tira pico) seguida de
    Savitzky-Golay (tira degrau). E o que vira a FORMA da referencia -- se um
    pico entrasse aqui, ele seria promovido a verdade em vez de removido.
    """
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    n = Y.shape[0]
    j = min(int(jan) | 1, (n - 1) | 1)
    if j < 5:
        return Y.copy()
    Z = median_filter(Y, size=(j, 1), mode="nearest")
    return savgol_filter(Z, j, 2, axis=0)


def trechos(ks):
    """Agrupa indices contiguos em intervalos [i0, i1]."""
    ks = sorted(ks)
    if not ks:
        return []
    out, i0 = [], ks[0]
    for a, b in zip(ks, ks[1:] + [None]):
        if b is None or b != a + 1:
            out.append((i0, a)); i0 = b
    return out


def perfil_humano_absoluto(xc: np.ndarray, pontos) -> np.ndarray:
    """Perfil inicial absoluto, linear entre cliques e plano nas pontas.

    No estado inicial não faz sentido transportar os defeitos do detector como
    uma "forma" de base: dois ou mais pontos humanos já definem a própria
    geometria observada. Os centros dos bins são usados para que cada ponto
    permaneça exato na malha do solver.
    """
    por_j = {}
    for p in pontos:
        por_j.setdefault(int(p["j"]), []).append(float(p["y"]))
    if len(por_j) < 2:
        raise ValueError("o perfil inicial absoluto requer ao menos dois bins")
    js = np.array(sorted(por_j))
    ys = np.array([np.mean(por_j[j]) for j in js])
    return np.interp(np.asarray(xc, float), np.asarray(xc)[js], ys)


def limitar_passos(y: np.ndarray, sentido: int, dmax) -> np.ndarray:
    """Reconcilia a saída final com irreversibilidade e passo máximo.

    O Viterbi já respeita estes limites, mas o centroide subpixel e a correção
    espacial vêm depois dele. Aplicar o limite novamente garante que nenhuma
    etapa posterior reintroduza movimento antes da ignição ou salto de flare.
    """
    z = sentido * np.asarray(y, float).copy()
    dk = np.broadcast_to(np.asarray(dmax, float), z.shape)
    if sentido not in (-1, 1) or not np.isfinite(z).all() or \
            not np.isfinite(dk).all() or np.any(dk < 0):
        raise ValueError("trajetória ou limite de passo inválido")
    for k in range(1, len(z)):
        z[k] = np.clip(z[k], z[k - 1], z[k - 1] + dk[k])
    return sentido * z


def limitar_passos_ancorado(y: np.ndarray, sentido: int, dmax, fixos) -> np.ndarray:
    """Impõe a física preservando exatamente os valores humanos em ``fixos``.

    Preserva a trajetória calculada sempre que ela já satisfaz monotonicidade e
    passo máximo. Entre duas âncoras, cada amostra é apenas recortada ao menor
    intervalo que simultaneamente respeita o vizinho anterior e ainda permite
    alcançar a próxima âncora. Isso evita a implementação antiga, que substituía
    todo o trecho por uma rampa reta e criava morros somente nas colunas que
    continham cliques humanos.
    """
    z = sentido * np.asarray(y, float).copy()
    dk = np.broadcast_to(np.asarray(dmax, float), z.shape)
    hard = {int(k): sentido * float(v) for k, v in fixos.items()}
    ks = sorted(hard)
    if not ks:
        return limitar_passos(y, sentido, dmax)

    # Confere primeiro se o conjunto de âncoras admite alguma trajetória.
    for ka, kb in zip(ks, ks[1:]):
        mov = hard[kb] - hard[ka]
        cap = float(dk[ka + 1:kb + 1].sum())
        if mov < -1e-7 or mov > cap + 1e-7:
            raise ValueError("âncoras exatas incompatíveis com monotonicidade/passo máximo")

    for k in ks:
        z[k] = hard[k]

    # Antes da primeira âncora, projeta para trás sem mexer mais do que o
    # necessário. Normalmente a primeira âncora é k=0, mas a função permanece
    # correta para usos independentes.
    for k in range(ks[0] - 1, -1, -1):
        z[k] = np.clip(z[k], z[k + 1] - dk[k + 1], z[k + 1])

    # Dentro de cada intervalo, os envelopes inferior/superior codificam a
    # capacidade restante até a âncora da direita. O passe para frente mantém
    # a forma de ``y`` quando ela é física, em vez de fabricar uma reta.
    for ka, kb in zip(ks, ks[1:]):
        a, b = hard[ka], hard[kb]
        passos = dk[ka + 1:kb + 1]
        acumulada = np.r_[0.0, np.cumsum(passos)]
        restante = acumulada[-1] - acumulada
        inferior = np.maximum(a, b - restante)
        superior = np.minimum(b, a + acumulada)
        z[ka] = a
        for rel, k in enumerate(range(ka + 1, kb + 1), start=1):
            lo = max(float(inferior[rel]), float(z[k - 1]))
            hi = min(float(superior[rel]), float(z[k - 1] + dk[k]))
            if lo > hi + 1e-9:
                raise ValueError("projeção ancorada ficou sem intervalo físico")
            z[k] = np.clip(z[k], lo, hi)
        z[kb] = b

    # Depois da última âncora, mantém o resultado original dentro da física.
    for k in range(ks[-1] + 1, len(z)):
        z[k] = np.clip(z[k], z[k - 1], z[k - 1] + dk[k])
    for k, valor in hard.items():
        z[k] = valor
    delta = np.diff(z)
    if np.any(delta < -1e-7) or np.any(delta > dk[1:] + 1e-7):
        raise ValueError("âncora exata conflita com a restrição física")
    return sentido * z


class Nucleo:
    def __init__(self, interfaces: str | Path | None = None, par: Par | None = None):
        self.par = par or Par()
        self.meta = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())
        self.C = np.load(RAIZ / "_tmp/cubo_bgr.npy")
        self.d = np.load(RAIZ / "resultados/serie.npz")
        r0 = np.load(interfaces or RAIZ / "resultados/interfaces.npz")
        try:
            cfg = json.loads((RAIZ / "configuracao_pe.json").read_text())
        except (OSError, ValueError):
            cfg = {}
        self.fator_tempo_real = float(cfg.get("fator_tempo_real", 1.0))
        if not np.isfinite(self.fator_tempo_real) or self.fator_tempo_real <= 0:
            raise ValueError("fator_tempo_real deve ser positivo")
        self.G = self.C.astype(np.float32).mean(3)
        self.N, self.H, self.nb = self.G.shape
        self.Y0, self.Y1, self.X0, self.X1 = self.meta["roi"]
        self.q_arq = np.asarray(self.meta["quadro_arquivo"])
        self.q_cam = np.asarray(self.meta["quadro_camera"])
        # O arquivo recebido está acelerado: um segundo da linha do vídeo
        # representa `fator_tempo_real` segundos do ensaio. Mantemos os dois
        # eixos explícitos para não misturar o instante visual pedido pelo
        # usuário com a taxa física, que usa o tempo real.
        self.t_video = np.asarray(self.d["t"], float)
        self.t = self.t_video * self.fator_tempo_real
        self.dt = float(self.d["dt"]) * self.fator_tempo_real
        self.xc = self.d["xc"]; self.bom = self.d["bom"]
        self.yc0 = r0["yc0"]
        self.yc_med = float(np.median(self.yc0[self.bom]))
        self.k0, self.k1 = int(self.d["k0"]), int(self.d["k1"])
        self.ref = {"cima": self.d["ps0"], "baixo": self.d["pi0"]}
        self.Y_ini = {f: np.array(r0[CH[f]], float) for f in FACES}
        self.Y = {f: self.Y_ini[f].copy() for f in FACES}
        self.incerteza = {}
        for f in FACES:
            e = np.asarray(r0[f"e_{BASE[f]}"], float)
            c = np.asarray(r0[f"c_{BASE[f]}"], float)
            largura = np.nanmedian(e[:, self.bom], axis=1)
            confianca = np.nanmedian(c[:, self.bom], axis=1)
            def escala_robusta(v):
                a, b = np.nanpercentile(v, [5, 95])
                return np.clip((v - a) / max(b - a, 1e-9), 0, 1)
            self.incerteza[f] = escala_robusta(largura) + 1 - escala_robusta(confianca)
        self.cols = np.flatnonzero(self.bom)
        self._score: dict[int, np.ndarray] = {}
        # face de cima so sobe (y cai), de baixo so desce (y cresce)
        self.sentido = {"cima": -1, "baixo": +1}
        self.flare = (np.asarray(r0["flare"], bool) if "flare" in r0
                      else quadros_com_flare(self.C)[0])
        # ignicao e apagamento, para a interface poder mostrar o que e medida
        # e o que e so a linha existindo fora da queima
        self.i_ign, self.i_apag = self.k0, self.k1
        try:
            r = json.loads((RAIZ / "resultados/resumo.json").read_text())
            self.i_ign = int(np.searchsorted(self.q_cam, r["quadro_ignicao"]))
            self.i_apag = int(np.searchsorted(self.q_cam, r["quadro_apagamento"]))
        except Exception:
            pass
        # Início da medida (quadro). Antes dele a regra de não recuar não vale:
        # logo após a ignição a borda vista pela janela pode SUBIR (o PE incha ao
        # fundir e o líquido é empurrado para as bordas pela pressão da ignição).
        # None mantém o comportamento antigo (regra desde a ignição).
        self.k_livre = None

    def _livre(self, a, b) -> bool:
        """Par de pontos fora da regra física: algum está antes do início da medida."""
        return self.k_livre is not None and min(int(a["k"]), int(b["k"])) < self.k_livre

    def sugerir_quadros(self, alvos_video_s=(20, 60, 120, 180, 235),
                        raio_video_s=8.0):
        """Escolhe, perto de cada alvo visual, o quadro mais incerto por face.

        A largura do vale e a falta de confiança contribuem igualmente. Flares
        ficam fora: eles devem ser ignorados, não oferecidos para rotulagem.
        """
        saida = []
        for alvo in alvos_video_s:
            cand = np.flatnonzero((np.abs(self.t_video - alvo) <= raio_video_s)
                                  & ~self.flare)
            if not len(cand):
                cand = np.array([int(np.argmin(np.abs(self.t_video - alvo)))])
            for face in FACES:
                k = int(cand[np.nanargmax(self.incerteza[face][cand])])
                saida.append(dict(face=face, k=k,
                                  alvo_video_s=float(alvo),
                                  t_video_s=float(self.t_video[k]),
                                  t_real_s=float(self.t[k]),
                                  incerteza=float(self.incerteza[face][k])))
        return saida

    def cobertura_inicial(self, ancoras):
        """Cobertura axial das âncoras pré-ignição em cada face."""
        xmin, xmax = float(self.xc[self.cols].min()), float(self.xc[self.cols].max())
        out = {}
        for face in FACES:
            xs = [float(a["x"]) for a in ancoras
                  if a["face"] == face and a["k"] < self.i_ign]
            out[face] = dict(xmin=min(xs) if xs else None,
                             xmax=max(xs) if xs else None,
                             alvo_min=xmin, alvo_max=xmax,
                             completa=bool(xs and min(xs) <= xmin + 20 and
                                           max(xs) >= xmax - 20))
        return out

    # Dois pontos de quadros diferentes só se comparam se estiverem a menos de TOL_X
    # px no eixo do grão. A coluna de 20 px era grossa demais: na curva íngreme da
    # cavidade, pontos a 10 px de distância têm alturas legitimamente diferentes.
    TOL_X = 10.0

    def erros_ancoras(self, ancoras):
        """Explica restrições humanas incompatíveis com monotonicidade/dmax."""
        erros = []
        for face in FACES:
            s = self.sentido[face]
            por_coluna = {}
            for p in (a for a in ancoras if a["face"] == face):
                por_coluna.setdefault(int(p["j"]), []).append(p)
            for pontos in por_coluna.values():
                pontos.sort(key=lambda p: p["k"])
                k_livre = getattr(self, "k_livre", None)
                pontos = [p for p in pontos
                         if k_livre is None or int(p["k"]) >= k_livre]
                # pares de quadros DIFERENTES e próximos em x; pontos do mesmo
                # quadro nunca conflitam (são posições diferentes da mesma curva)
                pares = [(a, b) for n_, a in enumerate(pontos) for b in pontos[n_ + 1:]
                         if int(a["k"]) != int(b["k"])
                        and abs(float(a.get("x", a["j"])) -
                                float(b.get("x", b["j"]))) <= self.TOL_X]
                for a, b in pares:
                    movimento = s * (float(b["y"]) - float(a["y"]))
                    ini = max(int(a["k"]) + 1, self.i_ign)
                    fim = min(int(b["k"]), self.i_apag)
                    capacidade = max(0, fim - ini + 1) * self.par.dmax
                    if movimento < -1e-7:
                        erros.append(
                            f"{face}: o ponto de {self.t_video[b['k']]:.1f}s "
                            "voltaria a superfície para dentro do combustível")
                    elif movimento > capacidade + 1e-7:
                        erros.append(
                            f"{face}: os pontos em {self.t_video[a['k']]:.1f}s e "
                            f"{self.t_video[b['k']]:.1f}s exigem {movimento:.1f}px, "
                            f"mas o limite físico permite {capacidade:.1f}px")
        return erros

    def ajustar_ao_novo(self, ancoras, novo):
        """A marcação mais NOVA manda (pedido do usuário, 14/10/2026).

        Em vez de recusar um ponto que conflita com marcações antigas da mesma face
        e coluna (recuo negativo ou passo acima do limite físico), move cada ponto
        antigo o MÍNIMO necessário para ficar compatível com o novo, que fica fixo.
        Antes do novo, a face de baixo só pode estar acima dele (y menor) e no
        máximo `capacidade` px acima; depois, só abaixo. Devolve [(ponto, y_antigo)]
        para o desfazer.
        """
        s = self.sentido[novo["face"]]
        yn = float(novo["y"])
        movidos = []
        for p in ancoras:
            if (p is novo or p["face"] != novo["face"] or int(p["j"]) != int(novo["j"])
                    or int(p["k"]) == int(novo["k"]) or self._livre(p, novo)
                    or abs(float(p["x"]) - float(novo["x"])) > self.TOL_X):
                continue
            a, b = (p, novo) if p["k"] < novo["k"] else (novo, p)
            ini = max(int(a["k"]) + 1, self.i_ign)
            fim = min(int(b["k"]), self.i_apag)
            cap = max(0, fim - ini + 1) * self.par.dmax
            if p["k"] < novo["k"]:
                lo, hi = sorted((yn - s * cap, yn))
            else:
                lo, hi = sorted((yn, yn + s * cap))
            y = float(p["y"])
            y2 = min(max(y, lo), hi)
            if abs(y2 - y) > 1e-9:
                movidos.append((p, y))
                p["y"] = y2
        return movidos

    # ------------------------------------------------------------------ #
    def score(self, j: int) -> np.ndarray:
        if j not in self._score:
            I = gaussian_filter1d(self.G[:, :, j], 2.0, axis=1)
            S = escore(I)
            # quadro estourado de luz nao e evidencia: o traco some sob o
            # brilho, e insistir em medir ali e inventar numero
            S[self.flare] *= self.par.peso_flare
            self._score[j] = S
        return self._score[j]

    def referencia(self, face, ancoras):
        """
        Trajetoria de referencia (N, nb) construida a partir dos seus pontos.
        Sem ponto nenhum, e so a tendencia suavizada da imagem.
        """
        # a forma tem de ser lisa nos DOIS eixos: se ela carregar o serrilhado
        # em x da primeira passada, a referencia promove esse serrilhado a
        # verdade em vez de remove-lo
        # A forma só depende da detecção inicial e de dois parâmetros: era refeita
        # a cada clique (~0,2 s por face) e travava a janela (22/10/2026).
        chave = (face, self.par.suav_t, self.par.jan_x)
        cache = self.__dict__.setdefault("_forma_cache", {})
        if chave not in cache:
            cache[chave] = suavizar_x(suavizar_t(self.Y_ini[face], self.par.suav_t),
                                      self.par.jan_x)
        forma = cache[chave].copy()
        # Antes da ignição existe uma observação muito mais forte que qualquer
        # vale luminoso: a porta parada no quadro sem chama. O detector antigo
        # deixava até `dmax` por amostra desde t=0; 74 pequenos passos podiam
        # acumular dezenas de pixels antes de o motor sequer começar a regredir.
        # A geometria inicial ps0/pi0 vem do regime escuro pré-ignição.
        if self.i_ign > 0:
            forma[:self.i_ign] = self.ref[face][None, :]
        A = [a for a in ancoras if a["face"] == face]
        if not A:
            ref = forma
            if self.i_ign > 0:
                ref[:self.i_ign] = self.ref[face][None, :]
            if self.i_apag + 1 < self.N:
                ref[self.i_apag + 1:] = ref[self.i_apag]
            return ref
        ks = sorted({a["k"] for a in A})
        desl = np.zeros((len(ks), self.nb))
        for n, k in enumerate(ks):
            pts = sorted((a for a in A if a["k"] == k), key=lambda a: a["x"])
            if k < self.i_ign and len({int(a["j"]) for a in pts}) >= 2:
                # Estado inicial: os cliques definem a geometria absoluta.
                # Não preserve ondulações/artefatos da detecção automática.
                desl[n] = perfil_humano_absoluto(self.xc, pts) - forma[k]
            else:
                # O solver vive nos centros dos bins, não na coordenada
                # contínua do mouse. Interpolar pelo centro garante exatidão.
                xs = np.array([self.xc[a["j"]] for a in pts])
                off = np.array([a["y"] - forma[k, a["j"]] for a in pts])
                # um ponto desloca a linha toda; varios seguem o perfil.
                desl[n] = np.interp(self.xc, xs, off) if len(xs) > 1 else off[0]
        if len(ks) == 1:
            D = np.repeat(desl, self.N, axis=0)
        else:
            D = np.empty((self.N, self.nb))
            tt = np.arange(self.N)
            for j in range(self.nb):
                D[:, j] = np.interp(tt, ks, desl[:, j])
        # Uma correção feita durante a queima não pode ser extrapolada para
        # trás e mudar a porta antes da ignição. Sem âncora pré-ignição, o
        # deslocamento inicial é zero; depois ele interpola suavemente até o
        # primeiro clique em combustão. Com âncora pré-ignição, usa a correção
        # humana e ainda a mantém constante nesse trecho de regressão zero.
        if self.i_ign > 0:
            pre = [i for i, k in enumerate(ks) if k < self.i_ign]
            d0 = desl[pre[-1]] if pre else np.zeros(self.nb)
            for j in range(self.nb):
                futuros = [(k, desl[i, j]) for i, k in enumerate(ks)
                            if k >= self.i_ign]
                if futuros:
                    kf = np.array([self.i_ign - 1] + [q[0] for q in futuros])
                    vf = np.array([d0[j]] + [q[1] for q in futuros])
                    D[self.i_ign - 1:, j] = np.interp(
                        np.arange(self.i_ign - 1, self.N), kf, vf)
            D[:self.i_ign] = d0
        ref = forma + D
        if self.i_ign > 0:
            ref[:self.i_ign] = self.ref[face][None, :] + D[:self.i_ign]
        if self.i_apag + 1 < self.N:
            ref[self.i_apag + 1:] = ref[self.i_apag]
        if self.par.mono:
            ref = (np.maximum.accumulate(ref, axis=0) if self.sentido[face] > 0
                   else -np.maximum.accumulate(-ref, axis=0))
        return ref

    def recalcular(self, ancoras, ignora, cols=None, faces=FACES, progresso=None):
        """
        Devolve {face: (N, nb)} com a trajetoria refeita nas colunas pedidas.
        `progresso(feito, total)` e chamado a cada coluna, para a barra.
        """
        p = self.par
        conflitos = self.erros_ancoras(ancoras)
        if conflitos:
            raise ValueError("marcações humanas conflitantes:\n- " +
                             "\n- ".join(conflitos))
        cols = self.cols if cols is None else np.asarray(cols)
        # a coluna de um ponto marcado SEMPRE entra, senao o ponto nao age
        js = [x["j"] for x in ancoras]
        if js:
            cols = np.union1d(cols, js).astype(int)
        Y = {f: self.Y[f].copy() for f in FACES}
        yy = np.arange(self.H)
        total = len(cols) * len(faces) * 3; feito = 0
        for f in faces:
            lado = self.sentido[f]
            mo = lado if p.mono else 0
            ign = sorted(ignora[f])
            anc = [x for x in ancoras if x["face"] == f]
            faixas = trechos(ign)
            ref = self.referencia(f, ancoras)
            # Regressão só pode ocorrer com chama. O vetor de passo zera antes
            # da ignição e depois do apagamento; no primeiro quadro de queima a
            # superfície pode andar no máximo `dmax`, nunca saltar para o traço
            # brilhante que acabou de aparecer.
            dlim = np.zeros(self.N, np.int32)
            dlim[self.i_ign:self.i_apag + 1] = p.dmax
            raio, peso = corredor(p.obediencia)
            if not anc:                       # sem marcacao, nada a obedecer
                raio, peso = 250.0, 0.0
            # limites e escore de cada coluna, montados uma vez so
            band, prb, esc = {}, {}, {}
            for j in cols:
                c0 = int(round(self.yc0[j]))
                lo, hi = (0, min(self.H, c0 + 20)) if lado < 0 else \
                         (max(0, c0 - 20), self.H)
                pr = np.zeros(hi - lo, bool)
                if lado < 0:
                    pr[c0 - lo:] = True; pr[:p.margem] = True
                else:
                    pr[:c0 - lo] = True; pr[hi - lo - p.margem:] = True
                band[j] = (lo, hi); prb[j] = pr
                S = self.score(j).copy()
                if ign:
                    S[np.array(ign)] = 0.0
                # corredor da SUA trajetoria, valido no ensaio inteiro
                dist = np.abs(yy[None, :] - ref[:, j][:, None])
                S = S - peso * np.minimum(dist / raio, 1.0)
                S[dist > raio] = -1e3
                if self.i_ign > 0:
                    # Posição absoluta observada antes da chama. É uma âncora
                    # física, não uma previsão do detector já contaminado.
                    pre_fora = dist[:self.i_ign] > p.tol
                    S[:self.i_ign][pre_fora] = -1e3
                    S[:self.i_ign][~pre_fora] = 1.5
                for k in {x["k"] for x in anc}:
                    # no quadro marcado nao se discute. O alvo e a REFERENCIA
                    # naquela coluna, que na coluna do clique vale exatamente o
                    # ponto e nas outras segue a forma -- usar o y do clique em
                    # todas ignoraria a inclinacao axial da superficie
                    fora = np.abs(yy - ref[k, j]) > p.tol
                    S[k, fora] = -1e3
                    S[k, ~fora] = 1.5
                # guarda ja recortado na faixa da face: as passadas seguintes
                # so mexem nisso, e o corredor espacial custa metade
                esc[j] = S[:, lo:hi]

            # TRES PASSADAS. A 1a nao tem restricao nenhuma em x, e por isso
            # colunas vizinhas podem enganchar em vales diferentes. A 2a prende
            # cada coluna a um corredor largo ao redor de um ajuste polinomial
            # ROBUSTO em x, que desmancha ate blocos inteiros de colunas
            # erradas (a mediana local nao desmancha: ela mora dentro do bloco).
            # A 3a aperta o corredor ao redor de uma suavizacao local, que ja
            # pode ser local porque os blocos sairam, e devolve a estrutura
            # axial de verdade.
            Yf = np.empty((self.N, len(cols)))
            for etapa in range(3):
                if etapa == 0:
                    espacial = None
                elif etapa == 1:
                    espacial = ajuste_robusto_x(Yf, p.grau,
                                                usar=colunas_confiaveis(Yf))
                else:
                    espacial = suavizar_x(Yf, p.jan_x)
                r_, w_ = (0, 0) if espacial is None else \
                         (p.esp2 if etapa == 1 else p.esp3)
                for m, j in enumerate(cols):
                    lo, hi = band[j]
                    Sb = esc[j]
                    if espacial is not None:
                        Sb = corredor_no_escore(Sb, espacial[:, m] - lo, r_, w_)
                    y = viterbi(Sb, dlim, p.lam, prb[j], mo)
                    Yf[:, m] = (refinar(Sb, y, mono=mo) if etapa == 2 else y) + lo
                    feito += 1
                    if progresso:
                        progresso(feito, total)
            for m, j in enumerate(cols):
                yf = Yf[:, m]
                for i0, i1 in faixas:
                    e0, e1 = i0 - 1, i1 + 1
                    if e0 < 0 or e1 >= self.N:
                        continue
                    yf[i0:i1 + 1] = np.interp(np.arange(i0, i1 + 1), [e0, e1],
                                              [yf[e0], yf[e1]])
                if p.mono:                    # a rampa tambem obedece a fisica
                    yf = (np.maximum.accumulate(yf) if lado > 0
                          else -np.maximum.accumulate(-yf))
                # Refinamento subpixel e interpolação não podem reabrir o modo
                # de falha que a DP bloqueou: fora da chama a posição é fixa.
                if self.i_ign > 0:
                    yf[:self.i_ign] = ref[0, j]
                if self.i_apag + 1 < self.N:
                    yf[self.i_apag + 1:] = yf[self.i_apag]
                # Deve ser a ÚLTIMA operação geométrica: assim nem o pin da
                # posição inicial cria um salto na transição para a chama.
                yf = limitar_passos(yf, lado, dlim)
                aj = [a for a in anc if int(a["j"]) == int(j)]
                if aj:
                    fixos = {0: float(ref[0, j])}
                    fixos.update({int(a["k"]): float(a["y"]) for a in aj})
                    yf = limitar_passos_ancorado(yf, lado, dlim, fixos)
                Y[f][:, j] = yf
            # Conferência final: nenhuma etapa posterior à projeção ancorada
            # pode deslocar uma marca humana.
            for j in {int(a["j"]) for a in anc}:
                for a in (q for q in anc if int(q["j"]) == j):
                    if abs(Y[f][int(a["k"]), j] - float(a["y"])) > 1e-9:
                        raise RuntimeError("uma marca humana foi alterada internamente")
                delta = lado * np.diff(Y[f][:, j])
                ruim = np.flatnonzero((delta < -1e-7) |
                                      (delta > dlim[1:] + 1e-7))
                if len(ruim):
                    k = int(ruim[0] + 1)
                    raise ValueError(
                        f"ponto humano de {f} em vídeo t="
                        f"{self.t_video[k]:.1f}s conflita com a restrição "
                        "física de monotonicidade/passo máximo; ele não foi movido")
        return Y, cols

    # ------------------------------------------------------------------ #
    def taxa(self, face, cols=None, Y=None):
        cols = self.cols if cols is None else np.asarray(cols)
        Yf = (Y or self.Y)[face][:, cols]
        r = self.ref[face][cols][None, :]
        rec = (r - Yf) if face == "cima" else (Yf - r)
        med = np.nanmedian(rec, 1)
        m, se, r2 = ajuste_linear(self.t[self.k0:self.k1 + 1],
                                  med[self.k0:self.k1 + 1])
        return m, r2

    def salvar(self, ancoras, ignora, out=None):
        """Grava guia.csv e interfaces_revisado.npz. Devolve os caminhos."""
        import csv
        out = Path(out or RAIZ / "resultados")
        with open(out / "guia.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tipo", "k", "t_s", "face", "x_px", "y_roi_px",
                        "t_video_s"])
            for p in ancoras:
                w.writerow(["ancora", p["k"], f"{self.t[p['k']]:.1f}", p["face"],
                            f"{p['x']:.0f}", f"{p['y']:.1f}",
                            f"{self.t_video[p['k']]:.1f}"])
            for f in FACES:
                for k in sorted(ignora[f]):
                    w.writerow(["ignorar", k, f"{self.t[k]:.1f}", f, "", "",
                                f"{self.t_video[k]:.1f}"])
        saida = {}
        for f in FACES:
            saida[CH[f]] = self.Y[f]
            # A confiança da imagem e a autoridade humana são grandezas
            # diferentes. Antes desta marca explícita, o pipeline de DL lia a
            # trajetória revisada, mas continuava pesando-a com `c_sup/c_inf` e
            # com a janela do detector. Justamente onde a pessoa corrigia um
            # erro visual, esse peso costumava ser baixo ou zero e o treino
            # ignorava a correção.
            #
            # Um clique define o deslocamento da linha inteira naquele quadro
            # (ver `referencia`); logo o quadro marcado inteiro recebe peso
            # humano 1. Vários cliques no mesmo quadro refinam a forma axial,
            # mas não mudam a autoridade da anotação.
            humano = np.zeros_like(self.Y[f], dtype=np.float32)
            for k in {int(p["k"]) for p in ancoras if p["face"] == f}:
                humano[k, :] = 1.0
            saida[f"humano_{BASE[f]}"] = humano
            c = np.zeros_like(self.Y[f]); e = np.zeros_like(self.Y[f])
            for j in self.cols:
                S = self.score(j)                    # confianca da IMAGEM,
                yi = np.clip(self.Y[f][:, j].round().astype(int), 0, self.H - 1)
                c[:, j] = S[np.arange(self.N), yi]   # nao do que voce mandou
                I = gaussian_filter1d(self.G[:, :, j], 2.0, axis=1)
                e[:, j] = largura_vale(I, self.Y[f][:, j])
            saida[f"c_{BASE[f]}"] = c; saida[f"e_{BASE[f]}"] = e
        np.savez(out / "interfaces_revisado.npz", **saida, yc0=self.yc0,
                 flare=self.flare,
                 roi=self.meta["roi"], bin_x=self.meta["bin_x"],
                 dq=self.meta["dq"], fps_camera=self.meta["fps_camera"],
                 fator_tempo_real=self.fator_tempo_real,
                 quadro_camera=self.q_cam, quadro_arquivo=self.q_arq,
                 ignorados_cima=np.array(sorted(ignora["cima"])),
                 ignorados_baixo=np.array(sorted(ignora["baixo"])))
        return out / "guia.csv", out / "interfaces_revisado.npz"


# --------------------------------------------------------------------- #
def cache_quadros(escala: float = 0.8, x=(620, 1700), q: int = 74,
                  progresso=None):
    """
    Le o video UMA vez e guarda os quadros amostrados ja recortados e em JPEG.

    Sem isso nao ha video tocando: procurar quadro a quadro no arquivo custa
    dezenas de ms por quadro e a reproducao engasga. Em JPEG na memoria a
    decodificacao e de ~3 ms, e o cache em disco faz a segunda abertura ser
    quase instantanea.
    """
    import cv2
    meta = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())
    Y0, Y1, X0, X1 = meta["roi"]; XA, XB = x
    alvo = RAIZ / "_tmp"
    bin_, idx = alvo / "quadros.bin", alvo / "quadros.json"
    assin = dict(escala=escala, x=list(x), q=q, roi=meta["roi"],
                 n=len(meta["quadro_arquivo"]))
    if bin_.exists() and idx.exists():
        m = json.loads(idx.read_text())
        if m.get("assinatura") == assin:
            dados = bin_.read_bytes()
            return [dados[a:b] for a, b in m["cortes"]], m["larg"], m["alt"]

    unicos = np.asarray(meta["quadro_arquivo"])
    alvo_set = {int(v): n for n, v in enumerate(unicos)}
    W = int((XB - XA) * escala); Hh = int((Y1 - Y0) * escala)
    saida = [None] * len(unicos)
    cap = cv2.VideoCapture(meta["video"])
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i in alvo_set:
            ok, f = cap.retrieve()
            if ok:
                crop = cv2.resize(f[Y0:Y1, XA:XB], (W, Hh),
                                  interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])
                saida[alvo_set[i]] = buf.tobytes()
            if progresso:
                progresso(alvo_set[i] + 1, len(unicos))
        i += 1
    cap.release()
    vazio = cv2.imencode(".jpg", np.zeros((Hh, W, 3), np.uint8))[1].tobytes()
    saida = [s if s is not None else vazio for s in saida]
    cortes, pos = [], 0
    for s in saida:
        cortes.append([pos, pos + len(s)]); pos += len(s)
    bin_.write_bytes(b"".join(saida))
    idx.write_text(json.dumps(dict(assinatura=assin, cortes=cortes,
                                   larg=W, alt=Hh)))
    return saida, W, Hh
