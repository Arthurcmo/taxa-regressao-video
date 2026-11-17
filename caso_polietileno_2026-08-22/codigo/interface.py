"""
Interface do ajuste guiado: video tocando, voce escolhe o quadro, corrige e
manda recalcular.

Tres decisoes que vieram de a versao anterior travar:

 1. O recalculo NAO acontece sozinho a cada clique. Ele so roda no botao
    "Recalcular". Marcar pontos e navegar continuam instantaneos.
 2. Quando roda, roda em uma THREAD separada, com barra de progresso. A janela
    continua respondendo, da para cancelar pela janela e nada congela.
 3. Os quadros sao lidos do video UMA vez, recortados e guardados em JPEG na
    memoria (e em cache no disco). Procurar quadro a quadro no arquivo custa
    dezenas de ms e era o que fazia a reproducao engasgar; em JPEG a
    decodificacao e de ~3 ms e o video toca liso.

    python codigo/interface.py
"""
from __future__ import annotations
import argparse, csv, faulthandler, hashlib, json, queue, re, shutil, sys, threading, time, traceback, unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "codigo"))
from nucleo import Nucleo, Par, FACES, CH, trechos, cache_quadros        # noqa: E402
import ensaio as E                                                         # noqa: E402

# Cores frias: sobre a chama laranja/amarela, vermelho e amarelo somem (pedido do
# usuário em 11/10/2026). Face de baixo em ciano, de cima em azul, marcas em verde.
COR = {"cima": "#5b8cff", "baixo": "#00e5ff"}
LARG_GRAFICOS = 470             # largura da janela de gráficos, em pontos da tela
COR_CAMADA = {"A": "#00ff6a", "B": "#2f9bff"}   # duas marcações independentes do mesmo quadro
GUIA = "#c8f4ff"
# marcações de OUTROS quadros, bem apagadas: anterior em ciano escuro, próximo em violeta
# marcações dos outros quadros, por grupo: o tom é o grupo, claro antes e escuro depois
FANT_CAMADA = {"A": ("#7dffb2", "#1f9e58"), "B": ("#a8d6ff", "#2a5fe6")}
ZOOM_MAX = 8.0
RAIO_TROCA = 8.0              # px do vídeo: clique a esta distância de um ponto o substitui
# Cruz do mouse: 1 px, misturada à imagem com opacidade parcial (o vídeo aparece
# através dela) e vazada no centro, para o pixel do clique ficar livre.
CRUZ_COR = np.array([255, 229, 0], np.float32)     # ciano, em BGR
CRUZ_ALFA, CRUZ_VAO, CRUZ_CURTA = 0.45, 5, 40
BG, FG, BG2 = "#161616", "#e8e8e8", "#232323"


def id_perfil(nome: str) -> str:
    """Nome de pasta estável e seguro, mantendo o nome humano no metadado."""
    base = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-_").lower()
    base = base[:48] or "perfil"
    digest = hashlib.sha256(nome.strip().encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


class App:
    def __init__(self, raiz: tk.Tk, a):
        self.raiz, self.a = raiz, a
        self.n = Nucleo(a.interfaces, Par(dmax=a.dmax, lam=a.lam, tol=a.tol,
                                          obediencia=a.obediencia))
        if getattr(a, "grao_inteiro", False):
            # a linha e o recálculo cobrem o grão inteiro, da porta de entrada à
            # saída, não só as colunas 18 a 90 em que o detector antigo era confiável
            self.n.cols = np.flatnonzero((self.n.xc >= a.x[0]) & (self.n.xc <= a.x[1]))
        if getattr(a, "inicio_medida_s", None) is not None:
            t0 = float(self.n.t[self.n.i_ign]) + float(a.inicio_medida_s)
            self.n.k_livre = int(np.searchsorted(self.n.t, t0))
        # a imagem cabe na largura da tela; o detalhe vem do zoom (vídeo em cache
        # em resolução cheia), não de uma janela maior que o monitor
        livre = max(self.raiz.winfo_screenwidth() - 70 - (LARG_GRAFICOS if a.graficos else 0), 600)
        self.E = min(a.escala, livre / float(a.x[1] - a.x[0]))
        self.Ec = 1.0                 # o vídeo fica em resolução cheia para o zoom
        self.Z, self.vx, self.vy = 1.0, 0.0, 0.0
        self.mouse = (0, 0)
        self.fantasma = 1             # 0 desligado, 1 quadros vizinhos, 2 todos
        self.cruz = 1                 # 1 cruz longa, 2 curta, 0 cursor do sistema
        self.mouse_fora, self._cruz_pendente = True, False
        self._vis, self.id_img = None, None
        self.XA, self.XB = a.x
        self.i = 0
        self.k_min, self.k_min_s = None, 0.0
        self.faixa: tuple[int, int] | None = None   # amostras da faixa de tempo a marcar
        self.livre = False                          # sessão em que a superfície pode recuar
        self.camada = "A"                           # marcação que recebe os cliques
        self.relogio = json.loads((RAIZ / "resultados/quadros_camera.json").read_text())["relogio"]
        arq = RAIZ / "_tmp/carimbo_camera.npy"
        self.carimbos = np.load(arq, mmap_mode="r") if arq.exists() else None
        self.img_carimbo = None
        self.painel, self.jan_graficos, self._sig_graficos, self._graficos_pendente = None, None, None, False
        self.ancoras: list[dict] = []
        self.ignora = {f: set() for f in FACES}
        self.pilha: list = []
        self.ver = {f: tk.BooleanVar(value=True) for f in FACES}
        self.tocando = False
        self.ini_trecho: int | None = None
        self.fila: queue.Queue = queue.Queue()
        self.calculando = False
        self.prog = (0, 1)
        self.cols_atual = self.n.cols
        self.sujo = False
        self._mexendo = False
        self.rig = tk.DoubleVar(value=100 * a.obediencia)
        self.perfil = tk.StringVar(value="")
        self.perfis: dict[str, Path] = {}
        self.nome_para_salvar: str | None = None
        self.nome_sessao: str | None = None      # perfil em que SALVAR grava direto
        self.alterado = False                    # há marcações não salvas
        # assinatura do estado vazio: abrir sem marcações não é "não salvo"
        self._assinatura_salva = self._assinatura()
        self._autosalvar_pendente = False
        self.passo = tk.IntVar(value=a.passo)
        self.ver_ref = tk.BooleanVar(value=True)
        self.ver_flare = tk.BooleanVar(value=True)
        self.sugestoes = (self._ler_sessao(a.sessao) if getattr(a, "sessao", None)
                          else self.n.sugerir_quadros())
        if getattr(a, "sem_linhas", False):
            # marcação às cegas: nenhuma linha automática induz o clique
            for f in FACES:
                self.ver[f].set(False)
            self.ver_ref.set(False)
        self.sugestao_atual: dict | None = None
        self.ind_sugestao = 0
        self.ref = {f: None for f in FACES}
        self._refazer_ref()

        self._montar()
        self._carregar_quadros()
        self._pintar()
        if (self._modo_sessao() and not getattr(a, "abrir_perfil", None)
                and not getattr(a, "novo_perfil", None)):
            self.raiz.after(300, self._oferecer_recuperacao)

    # ---------------- erros: registrar em arquivo, salvar e seguir ----------------
    @property
    def arquivo_erros(self) -> Path:
        return self.pasta_perfis.parent / "erros_interface.log"

    def _erro_callback(self, tipo, valor, tb):
        """Erro dentro de um clique/tecla: grava o rastro, salva a reserva e segue."""
        try:
            with self.arquivo_erros.open("a", encoding="utf-8") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} (quadro {self.i}, "
                         f"{len(self.ancoras)} pontos)\n")
                fh.write("".join(traceback.format_exception(tipo, valor, tb)))
        except OSError:
            pass
        traceback.print_exception(tipo, valor, tb)
        try:
            self._autosalvar()
            self.lb2.config(text=f"erro interno registrado em {self.arquivo_erros.name} "
                                 "(marcações guardadas na cópia de segurança). Pode continuar.")
        except Exception:
            pass

    # ---------------- salvamento da sessão ----------------
    def _modo_sessao(self) -> bool:
        return bool(getattr(self.a, "sessao", None))

    @property
    def arquivo_reserva(self) -> Path:
        return self.pasta_perfis.parent / "reserva_sessao_marcacao.csv"

    def _assinatura(self, ancoras=None, ignora=None):
        ancoras = self.ancoras if ancoras is None else ancoras
        ignora = self.ignora if ignora is None else ignora
        return tuple(sorted((p["k"], p["face"], round(p["x"], 1), round(p["y"], 1), p.get("camada", "A"))
                            for p in ancoras)) + tuple(
            (f, tuple(sorted(ignora[f]))) for f in FACES)

    def _da_camada(self) -> list[dict]:
        """Pontos da marcação ativa; a outra não interfere em nenhuma conta."""
        return [p for p in self.ancoras if p.get("camada", "A") == self.camada]

    def alternar_camada(self):
        """Troca a marcação que recebe os cliques (A ou B) no mesmo quadro."""
        self.camada = "B" if self.camada == "A" else "A"
        self._ref_sujo = True
        self.b_camada.config(text=f"marcação {self.camada} (m)")
        self.lb2.config(text=f"marcando em {self.camada}: {len(self._da_camada())} pontos nesta marcação")
        self._pintar()

    def carregar_grupo_a(self, nome: str):
        """Traz as marcações de um perfil salvo para o grupo A; o perfil de origem não muda."""
        pasta = self.perfis.get(nome)
        if pasta is None:
            self.lb2.config(text=f'perfil "{nome}" não encontrado')
            return
        ancoras, _ = self._ler_guia(pasta / "guia.csv")
        self.ancoras = [p for p in self.ancoras if p.get("camada", "A") != "A"] + ancoras
        self.pilha.clear()
        self.sujo = True
        self._refazer_ref()
        self._pintar()
        self.lb2.config(text=f'{len(ancoras)} marcações de "{nome}" no grupo A. '
                             "SALVAR grava neste perfil; o de origem não muda.")

    def _escrever_guia(self, caminho: Path):
        """Mesmo formato do guia.csv do núcleo (lido por _ler_guia e pelo treino)."""
        caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp = caminho.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tipo", "k", "t_s", "face", "x_px", "y_roi_px", "t_video_s", "camada"])
            for p in self.ancoras:
                w.writerow(["ancora", p["k"], f"{self.n.t[p['k']]:.1f}", p["face"],
                            f"{p['x']:.1f}", f"{p['y']:.1f}", f"{self.n.t_video[p['k']]:.1f}",
                            p.get("camada", "A")])
            for f in FACES:
                for k in sorted(self.ignora[f]):
                    w.writerow(["ignorar", k, f"{self.n.t[k]:.1f}", f, "", "",
                                f"{self.n.t_video[k]:.1f}", ""])
        tmp.replace(caminho)        # troca atômica: um arquivo nunca fica pela metade

    def _verificar_alteracao(self):
        """Chamado a cada redesenho: marca 'não salvo' e agenda a reserva."""
        if not self._modo_sessao():
            return
        alterado = self._assinatura() != self._assinatura_salva
        if alterado != self.alterado:
            self.alterado = alterado
            self._titulo()
        if alterado and not self._autosalvar_pendente:
            self._autosalvar_pendente = True
            self.raiz.after(1500, self._autosalvar)

    def _guardar_reserva_antiga(self):
        """Antes da primeira cópia de segurança desta abertura, guarda a anterior se
        ela tinha marcações não salvas (queda, janela fechada sem salvar): a cópia
        nova nunca apaga trabalho de outra abertura."""
        if getattr(self, "_reserva_guardada", False):
            return
        self._reserva_guardada = True
        r, meta_p = self.arquivo_reserva, self.arquivo_reserva.with_suffix(".json")
        try:
            if not (r.exists() and meta_p.exists()):
                return
            if json.loads(meta_p.read_text(encoding="utf-8")).get("salvo"):
                return
            destino = r.parent / "reservas_antigas"
            destino.mkdir(exist_ok=True)
            marca = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(r.stat().st_mtime))
            shutil.copy2(r, destino / f"reserva_{marca}.csv")
            shutil.copy2(meta_p, destino / f"reserva_{marca}.json")
        except (OSError, ValueError):
            pass

    def _oferecer_recuperacao_perfil(self):
        """Ao abrir um perfil da sessão: se a cópia de segurança é deste perfil e tem
        marcações feitas depois do último SALVAR, oferece recuperá-las."""
        r, meta_p = self.arquivo_reserva, self.arquivo_reserva.with_suffix(".json")
        if not (self._modo_sessao() and self.nome_sessao and r.exists() and meta_p.exists()):
            return
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            ancoras, ignora = self._ler_guia(r)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        if (meta.get("salvo") or meta.get("sessao") != str(self.a.sessao)
                or meta.get("perfil") != self.nome_sessao or not ancoras
                or self._assinatura(ancoras, ignora) == self._assinatura()):
            return
        if messagebox.askyesno(
                "recuperar marcações",
                f'O perfil "{self.nome_sessao}" tem marcações feitas depois do último SALVAR '
                f"(cópia de segurança de {meta.get('hora', '?')}: {len(ancoras)} pontos, "
                f"contra {len(self.ancoras)} no perfil salvo).\n\nRecuperar?\n\n"
                "Se não, a cópia fica guardada em revisao_humana/reservas_antigas."):
            self._guardar_reserva_antiga()
            self.ancoras, self.ignora = ancoras, ignora
            self.pilha.clear(); self._refazer_ref(); self._pintar()
            self.lb2.config(text=f"{len(ancoras)} marcações recuperadas da cópia de segurança. "
                                 "Aperte SALVAR para gravá-las no perfil.")

    def _autosalvar(self):
        self._autosalvar_pendente = False
        self._guardar_reserva_antiga()
        try:
            self._escrever_guia(self.arquivo_reserva)
            self.arquivo_reserva.with_suffix(".json").write_text(json.dumps(dict(
                sessao=str(self.a.sessao), perfil=self.nome_sessao, pontos=len(self.ancoras),
                salvo=not self.alterado, hora=time.strftime("%d/%m %H:%M")), ensure_ascii=False))
        except OSError:
            pass

    def _titulo(self):
        base = "marcação da superfície — polietileno"
        if self.nome_sessao:
            base += f"  ·  perfil {self.nome_sessao}"
        self.raiz.title(base + ("   •  NÃO SALVO" if self.alterado else ""))

    def _oferecer_recuperacao(self):
        r, meta_p = self.arquivo_reserva, self.arquivo_reserva.with_suffix(".json")
        if not (r.exists() and meta_p.exists()):
            return
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            ancoras, ignora = self._ler_guia(r)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        if meta.get("sessao") != str(self.a.sessao) or not ancoras:
            return
        perfil = meta.get("perfil")
        if meta.get("salvo") and perfil in self.perfis:
            # tudo já estava salvo: só oferece continuar de onde parou
            if messagebox.askyesno(
                    "continuar sessão",
                    f'Continuar de onde parou?\n\nAbrir o perfil "{perfil}" '
                    f"({len(ancoras)} pontos, salvo em {meta.get('hora', '?')})."):
                self.perfil.set(perfil)
                self.abrir_perfil()
            return
        if messagebox.askyesno(
                "recuperar marcações",
                f"Encontrei {len(ancoras)} marcação(ões) desta sessão que não foram salvas "
                f"num perfil (cópia de segurança de {meta.get('hora', '?')}).\n\nRecuperar?"):
            self.ancoras, self.ignora = ancoras, ignora
            self.nome_sessao = meta.get("perfil") or None
            self.pilha.clear(); self._refazer_ref(); self._pintar()
            self.lb2.config(text=f"{len(ancoras)} marcações recuperadas da cópia de segurança. "
                                 "Aperte SALVAR para gravá-las no perfil.")

    def _salvar_sessao(self, pedir_nome=False) -> bool:
        """SALVAR da sessão: grava na hora, sem recalcular e sem caixas repetidas."""
        nome = self.nome_sessao
        if not nome or pedir_nome:
            padrao = nome or "sessao-" + Path(str(self.a.sessao)).parent.name.split("_")[-1]
            nome = simpledialog.askstring("salvar", "Nome do perfil desta sessão:",
                                          initialvalue=padrao, parent=self.raiz)
            if not nome or not nome.strip():
                return False
            nome = nome.strip()
            if (nome != self.nome_sessao and (self.pasta_perfis / id_perfil(nome)).exists()
                    and not messagebox.askyesno("substituir perfil",
                                                f'Já existe o perfil "{nome}". Substituir?')):
                return False
        pasta = self.pasta_perfis / id_perfil(nome)
        pasta.mkdir(parents=True, exist_ok=True)
        self._escrever_guia(pasta / "guia.csv")
        np.savez_compressed(pasta / "interfaces_revisado.npz",
                            **{CH[f]: self.n.Y[f] for f in FACES})
        (pasta / "perfil.json").write_text(json.dumps(dict(
            nome=nome, formato=3, tipo="sessao_marcacao", sessao=str(self.a.sessao),
            salvo_em=time.strftime("%Y-%m-%d %H:%M:%S"), pontos=len(self.ancoras),
            trajetoria_recalculada=False,
            nota="só os pontos (guia.csv) importam para o treino; a trajetória não foi recalculada"),
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.nome_sessao = nome
        self._assinatura_salva = self._assinatura()
        self.alterado = False
        self._autosalvar()
        self._atualizar_perfis(selecionar=nome)
        self._titulo()
        feitos = sum(any(p["face"] == q["face"] and p["k"] == q["k"] for p in self._da_camada())
                     for q in self.sugestoes)
        self.lb2.config(text=f'salvo em "{nome}" às {time.strftime("%H:%M")}  ·  '
                             f"{len(self.ancoras)} pontos  ·  pedidos feitos: {feitos} de {len(self.sugestoes)}")
        return True

    def _ler_sessao(self, caminho):
        """Lista de quadros da sessão de marcação (JSON com k, face e rótulo).

        Se a sessão define `k_minimo`, o programa abre nesse quadro e não aceita
        marcação antes dele (transiente da ignição, chama ainda não consolidada).
        Se define `faixa`, a tela mostra a faixa de tempo a marcar e diz se o
        quadro atual está dentro dela; a faixa só informa, não bloqueia clique.
        Se define `livre`, a regra de não recuar não vale em nenhum quadro: a
        superfície pode subir, e a taxa negativa fica como foi marcada.
        """
        dados = json.loads(Path(caminho).read_text())
        itens = dados["quadros"]
        if dados.get("k_minimo") is not None:
            self.k_min = int(dados["k_minimo"])
            self.k_min_s = float(dados.get("marcar_a_partir_s", 0.0))
            self.i = self.k_min
        if dados.get("faixa"):
            self.faixa = (int(dados["faixa"]["k_ini"]), int(dados["faixa"]["k_fim"]))
        if dados.get("livre"):
            self.livre = True
            self.n.k_livre = self.n.N
        if dados.get("camada_inicial") in ("A", "B"):
            self.camada = dados["camada_inicial"]
        return [dict(face=q["face"], k=int(q["k"]), rotulo=q.get("rotulo", ""),
                     alvo_video_s=float(self.n.t_video[int(q["k"])]),
                     t_video_s=float(self.n.t_video[int(q["k"])]),
                     t_real_s=float(self.n.t[int(q["k"])]), incerteza=float("nan"))
                for q in itens]

    def _carimbo(self, k: int) -> str:
        """Tempo marcado pela câmera na amostra k, como aparece no quadro."""
        v = int(self.relogio[int(self.n.q_arq[k])])
        return f"{v // 10000:02d}:{v // 100 % 100:02d}:{v % 100:02d}"

    def _pintar_tempo(self, i: int):
        """Tempo marcado pela câmera em todo quadro: o recorte original e a leitura."""
        meio = self.W / 2
        txt = f"{self._carimbo(i)}  =  {self.n.t_video[i]:.1f} s".replace(".", ",")
        self.cq.create_rectangle(meio - 195, 2, meio + 195, 39, fill="#000000", outline="")
        if self.carimbos is not None:
            self.img_carimbo = ImageTk.PhotoImage(Image.fromarray(np.asarray(self.carimbos[i])))
            self.cq.create_image(meio - 100, 21, image=self.img_carimbo)
        self.cq.create_text(meio + 95, 21, text=txt, fill="#ffffff", font=("Menlo", 15, "bold"))
        if self._modo_sessao():
            self.cq.create_rectangle(40, 4, 200, 32, fill="#000000", outline="")
            self.cq.create_text(48, 18, anchor="w", text=f"MARCAÇÃO {self.camada}",
                                fill=COR_CAMADA[self.camada], font=("", 16, "bold"))

    def _pintar_faixa(self, i: int):
        a, b = self.faixa
        dentro = a <= i <= b
        self.lb_faixa.config(
            foreground="#00ff6a" if dentro else "#ff4fd8",
            text=f"FAIXA A MARCAR: {self.n.t_video[a]:.1f} a {self.n.t_video[b]:.1f} s ".replace(".", ",") +
                 f"({self._carimbo(a)} a {self._carimbo(b)}), amostras {a + 1} a {b + 1}"
                 f"   ·   este quadro, {self._carimbo(i)}: {'DENTRO' if dentro else 'FORA'} DA FAIXA")
        if not dentro:
            self.cq.create_text(self.W - 12, 66, anchor="e", text="FORA DA FAIXA",
                                fill="#ff4fd8", font=("", 18, "bold"))

    # ------------------------------------------------------------ layout #
    def _montar(self):
        """Janela da sessão de marcação (refeita em 17/10/2026 a pedido do usuário).

        Tudo fica dentro de uma página com rolagem: se a tela for pequena, rola
        para cima e para baixo (e para os lados), e as barras só aparecem quando
        o conteúdo não cabe. A ordem segue o uso: pedido atual e SALVAR no topo,
        avisos logo abaixo, imagem, ferramentas de visualização, tempo, atalhos,
        e as ferramentas avançadas (recalcular, ignorar, trechos) recolhidas no fim.
        """
        R = self.raiz
        R.title("marcação da superfície — polietileno")
        R.configure(bg=BG)
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=BG2)
        st.configure("TButton", background=BG2, foreground=FG, padding=(8, 4))
        st.map("TButton", background=[("active", "#3a3a3a")])
        st.configure("Acao.TButton", background="#0a7a52", foreground="#fff",
                     font=("", 13, "bold"), padding=(12, 6))
        st.map("Acao.TButton", background=[("active", "#0d9c68"),
                                           ("disabled", "#3a3a3a")])
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Dica.TLabel", background=BG, foreground="#8f9aa3", font=("", 11))
        st.configure("Pedido.TLabel", background=BG2, foreground="#7fdcff",
                     font=("", 15, "bold"), padding=(10, 6))
        st.configure("Aviso.TLabel", background=BG, foreground="#9fe7ff", font=("", 12))
        st.configure("Secao.TFrame", background=BG)
        st.configure("Horizontal.TProgressbar", background="#0a7a52")
        st.configure("TCheckbutton", background=BG, foreground=FG)

        W = int((self.XB - self.XA) * self.E)
        HQ = int((self.n.Y1 - self.n.Y0) * self.E)
        self.W, self.HQ, self.HK = W, HQ, int(self.a.alt_kim)

        # ---------------- página com rolagem ----------------
        self.pagina = tk.Canvas(R, bg=BG, highlightthickness=0, yscrollincrement=6,
                                xscrollincrement=6)
        self.rv = ttk.Scrollbar(R, orient="vertical", command=self.pagina.yview)
        self.rh = ttk.Scrollbar(R, orient="horizontal", command=self.pagina.xview)
        self.pagina.configure(yscrollcommand=self.rv.set, xscrollcommand=self.rh.set)
        self.pagina.grid(row=0, column=0, sticky="nsew")
        R.grid_rowconfigure(0, weight=1); R.grid_columnconfigure(0, weight=1)
        r = ttk.Frame(self.pagina, style="Secao.TFrame")
        self.pagina.create_window((0, 0), window=r, anchor="nw")
        r.bind("<Configure>", lambda e: self._ajustar_rolagem())
        self.pagina.bind("<Configure>", lambda e: self._ajustar_rolagem())
        R.bind_all("<MouseWheel>", self._rolar_pagina)
        R.bind_all("<Shift-MouseWheel>", lambda e: self._rolar_pagina(e, "x"))

        # ---------------- 1. pedido atual, navegação e salvar ----------------
        compacto = self.a.graficos      # com os gráficos ao lado, as barras cabem na imagem
        top = ttk.Frame(r); top.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Button(top, text="PRÓXIMO PEDIDO ▶", style="Acao.TButton",
                   command=self.proxima_sugestao).pack(side="left")
        ttk.Button(top, text="◀ marcado  ( , )",
                   command=lambda: self.ir_marcado(-1)).pack(side="left", padx=(8, 2))
        ttk.Button(top, text="marcado ▶  ( . )",
                   command=lambda: self.ir_marcado(+1)).pack(side="left", padx=2)
        self.b_salvar = ttk.Button(top, text="SALVAR  (⌘S)", style="Acao.TButton",
                                   command=self.salvar)
        self.b_salvar.pack(side="right")
        self.b_abrir = ttk.Button(top, text="abrir", command=self.abrir_perfil)
        self.b_abrir.pack(side="right", padx=4)
        self.b_novo = ttk.Button(top, text="novo  (⌘N)", command=self.novo_perfil)
        self.b_novo.pack(side="right", padx=2)
        self.cb_perfil = ttk.Combobox(top, textvariable=self.perfil, width=10 if compacto else 22)
        self.cb_perfil.pack(side="right", padx=2)
        ttk.Label(top, text="perfil").pack(side="right", padx=(12, 4))
        self._atualizar_perfis()

        self.lb_sug = ttk.Label(
            r, style="Pedido.TLabel",
            text=(f"Sessão com {len(self.sugestoes)} pedidos. Aperte PRÓXIMO PEDIDO para começar."
                  if getattr(self.a, "sessao", None) else
                  "10 sugestões: maior incerteza perto de 20, 60, 120, 180 e 235 s do vídeo"))
        self.lb_sug.pack(fill="x", padx=10, pady=(2, 2))
        self.lb_faixa = ttk.Label(r, text="", font=("Menlo", 14, "bold"))
        if self.faixa:
            self.lb_faixa.pack(fill="x", padx=10, pady=(0, 2))

        avisos = ttk.Frame(r); avisos.pack(fill="x", padx=10, pady=(0, 4))
        self.lb2 = ttk.Label(avisos, text="", style="Aviso.TLabel", wraplength=max(400, W - 260))
        self.lb2.pack(side="left")
        self.lb_pos = ttk.Label(avisos, text="", font=("Menlo", 12), foreground="#c9d1d9")
        self.lb_pos.pack(side="right")

        # ---------------- 2. imagem ----------------
        self.cq = tk.Canvas(r, width=W, height=HQ, bg="#000",
                            highlightthickness=1, highlightbackground="#333",
                            cursor="crosshair")
        self.cq.pack(padx=10, pady=2)
        self.cq.bind("<Button-1>", self._clique_esq)
        self.cq.bind("<Button-2>", self._clique_dir)
        self.cq.bind("<Button-3>", self._clique_dir)
        self.cq.bind("<Motion>", self._mouse_move)
        self.cq.bind("<Leave>", self._mouse_sai)
        self._cursor()
        self.cq.bind("<MouseWheel>", lambda e: self._roda(e, "y"))
        self.cq.bind("<Shift-MouseWheel>", lambda e: self._roda(e, "x"))
        self.cq.bind("<Control-MouseWheel>", self._roda_zoom)
        self.cq.bind("<Command-MouseWheel>", self._roda_zoom)

        # ---------------- 3. visualização ----------------
        vis = ttk.Frame(r); vis.pack(fill="x", padx=10, pady=(4, 2))
        ttk.Label(vis, text="zoom").pack(side="left")
        ttk.Button(vis, text="−", width=2, command=lambda: self.zoom(1 / 1.5, W / 2, HQ / 2)).pack(side="left", padx=(4, 1))
        ttk.Button(vis, text="+", width=2, command=lambda: self.zoom(1.5, W / 2, HQ / 2)).pack(side="left", padx=1)
        ttk.Button(vis, text="1:1  (0)", command=self.zoom_zerar).pack(side="left", padx=(1, 14))
        self.b_fant = ttk.Button(vis, text="", command=self.alternar_fantasma)
        self.b_fant.pack(side="left", padx=2)
        self.b_cruz = ttk.Button(vis, text="", command=self.alternar_cruz)
        self.b_cruz.pack(side="left", padx=2)
        ttk.Button(vis, text="gráficos (g)", command=self.alternar_graficos).pack(side="left", padx=2)
        self.b_camada = ttk.Button(vis, text=f"marcação {self.camada} (m)", command=self.alternar_camada)
        if self._modo_sessao():
            self.b_camada.pack(side="left", padx=2)
        vis2 = vis
        if compacto:
            vis2 = ttk.Frame(r); vis2.pack(fill="x", padx=10, pady=(0, 2))
        for f in FACES:
            ttk.Checkbutton(vis2, text=f"linha auto {f}", variable=self.ver[f],
                            command=self._pintar).pack(side="left", padx=(10, 0))
        # status de largura fixa: texto comprido é cortado em vez de alargar a janela
        self.lb = ttk.Label(vis2, text="", font=("Menlo", 11), foreground="#8f9aa3",
                            width=38, anchor="e")
        self.lb.pack(side="right")
        self.pb = ttk.Progressbar(vis2, mode="determinate", length=140)

        # ---------------- 4. tempo ----------------
        tempo = ttk.Frame(r); tempo.pack(fill="x", padx=10, pady=(6, 0))
        for txt, cmd, larg in (("|◀", lambda: self.ir(0, abs_=True), 3),
                               ("◀◀", lambda: self.ir(-10 * self._pa()), 3),
                               ("◀", lambda: self.ir(-self._pa()), 2)):
            ttk.Button(tempo, text=txt, width=larg, command=cmd).pack(side="left", padx=1)
        self.b_play = ttk.Button(tempo, text="▶  tocar", width=9, command=self.tocar)
        self.b_play.pack(side="left", padx=4)
        for txt, cmd, larg in (("▶", lambda: self.ir(self._pa()), 2),
                               ("▶▶", lambda: self.ir(10 * self._pa()), 3),
                               ("▶|", lambda: self.ir(self.n.N - 1, abs_=True), 3)):
            ttk.Button(tempo, text=txt, width=larg, command=cmd).pack(side="left", padx=1)
        ttk.Label(tempo, text="   passo").pack(side="left")
        tk.Spinbox(tempo, from_=1, to=60, width=3, textvariable=self.passo,
                   command=self._estado, bg=BG2, fg=FG, buttonbackground=BG2,
                   highlightthickness=0, justify="center").pack(side="left", padx=3)
        ttk.Label(tempo, text="   velocidade").pack(side="left")
        self.vel = tk.IntVar(value=1)
        tk.Scale(tempo, from_=1, to=6, orient="horizontal", variable=self.vel,
                 length=100, bg=BG, fg=FG, highlightthickness=0, troughcolor=BG2,
                 showvalue=False).pack(side="left")
        if not compacto:
            ttk.Label(tempo, text="kimograma do ensaio: clique ou arraste para ir a um instante",
                      style="Dica.TLabel").pack(side="right")

        self.esc = tk.Scale(r, from_=0, to=self.n.N - 1, orient="horizontal",
                            command=self._arrastar, bg=BG, fg=FG, length=W,
                            highlightthickness=0, troughcolor=BG2, showvalue=False)
        self.esc.pack(padx=10)
        self.ck = tk.Canvas(r, width=W, height=self.HK, bg="#000",
                            highlightthickness=1, highlightbackground="#333",
                            cursor="sb_h_double_arrow")
        self.ck.pack(padx=10, pady=(0, 4))
        self.ck.bind("<Button-1>", self._clique_kim)
        self.ck.bind("<B1-Motion>", self._clique_kim)

        # ---------------- 5. atalhos ----------------
        ttk.Label(r, style="Dica.TLabel", wraplength=W, justify="left",
                  text="Atalhos:  + −  zoom no mouse   ·   0  zoom normal   ·   rolar sobre a imagem: "
                       "move a página (com zoom, move a imagem)   ·   c  cruz   ·   f  marcações de outros "
                       "quadros   ·   , .  quadro marcado anterior/seguinte   ·   ← →  quadros   ·   "
                       "espaço  tocar   ·   control+Z  desfaz   ·   botão direito  apaga ponto   ·   "
                       "⌘S  salvar").pack(fill="x", padx=10, pady=(2, 6))

        # ---------------- 6. ferramentas avançadas (recolhidas) ----------------
        self.av_aberto = not bool(getattr(self.a, "sessao", None))
        self.b_av = ttk.Button(r, text="", command=self._alternar_avancado)
        self.b_av.pack(anchor="w", padx=10, pady=(2, 4))
        self.av = ttk.Frame(r)
        ac = ttk.Frame(self.av); ac.pack(fill="x", pady=(2, 2))
        self.b_calc = ttk.Button(ac, text="RECALCULAR", style="Acao.TButton",
                                 command=lambda: self.recalcular(False))
        self.b_calc.pack(side="left")
        for txt, cmd in (("ignorar este quadro", self.ignorar),
                         ("ignorar os flares", self.ignorar_flares),
                         ("início do trecho", lambda: self.trecho(False)),
                         ("fim do trecho", lambda: self.trecho(True)),
                         ("limpar pontos do quadro", self.limpar_quadro),
                         ("desfazer", self.desfazer)):
            ttk.Button(ac, text=txt, command=cmd).pack(side="left", padx=2)
        ac2 = ttk.Frame(self.av); ac2.pack(fill="x", pady=(2, 6))
        ttk.Label(ac2, text="obedecer às marcações").pack(side="left")
        tk.Scale(ac2, from_=0, to=100, orient="horizontal", variable=self.rig,
                 length=130, bg=BG, fg=FG, highlightthickness=0,
                 troughcolor=BG2, resolution=5,
                 command=lambda v: self._mudou_rigidez()).pack(side="left")
        ttk.Checkbutton(ac2, text="ver referência", variable=self.ver_ref,
                        command=self._pintar).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(ac2, text="marcar flares", variable=self.ver_flare,
                        command=self._pintar).pack(side="left", padx=(8, 0))
        ttk.Button(ac2, text="completar estado inicial",
                   command=self.proxima_inicial).pack(side="left", padx=(12, 2))
        ttk.Button(ac2, text="voltar ao automático",
                   command=self.zerar).pack(side="left", padx=2)
        self._mostrar_avancado()
        self._rotular_botoes()

        # ---------------- teclado ----------------
        R.bind("<Left>", lambda e: self.ir(-self._pa()))
        R.bind("<Right>", lambda e: self.ir(self._pa()))
        R.bind("<Shift-Left>", lambda e: self.ir(-10 * self._pa()))
        R.bind("<Shift-Right>", lambda e: self.ir(10 * self._pa()))
        R.bind("<space>", lambda e: self.tocar())
        R.bind("<Return>", lambda e: self.recalcular(False))
        R.bind("i", lambda e: self.ignorar())
        R.bind("<bracketleft>", lambda e: self.trecho(False))
        R.bind("<bracketright>", lambda e: self.trecho(True))
        R.bind("<Control-z>", lambda e: self.desfazer())
        for tecla in ("<plus>", "<equal>", "<KP_Add>"):
            R.bind(tecla, lambda e: self.zoom(1.5))
        for tecla in ("<minus>", "<KP_Subtract>"):
            R.bind(tecla, lambda e: self.zoom(1 / 1.5))
        R.bind("<Key-0>", lambda e: self.zoom_zerar())
        R.bind("<comma>", lambda e: self.ir_marcado(-1))
        R.bind("<period>", lambda e: self.ir_marcado(+1))
        R.bind("f", lambda e: self.alternar_fantasma())
        R.bind("c", lambda e: self.alternar_cruz())
        R.bind("g", lambda e: self.alternar_graficos())
        if self._modo_sessao():
            R.bind("m", lambda e: self.alternar_camada())
        R.bind("<Control-s>", lambda e: self.salvar())
        R.bind("<Command-s>", lambda e: self.salvar())
        R.bind("<Command-n>", lambda e: self.novo_perfil())
        R.protocol("WM_DELETE_WINDOW", self.sair)
        self._tamanho_janela(r)

    # ---------------- página: tamanho, rolagem e seções ----------------
    def _tamanho_janela(self, conteudo):
        """Janela do tamanho do conteúdo, sem passar da tela; o resto rola."""
        R = self.raiz
        conteudo.update_idletasks()
        sw, sh = R.winfo_screenwidth(), R.winfo_screenheight()
        # Em algumas sessões do macOS/Tk a tela ainda reporta 0 durante a
        # inicialização. Sem o limite, a geometria vira algo como -10x-90 e
        # a janela morre antes de aparecer.
        sw = max(sw, 640)
        sh = max(sh, 480)
        w = max(600, min(conteudo.winfo_reqwidth() + 24, sw - 10))
        h = max(400, min(conteudo.winfo_reqheight() + 24, sh - 90))
        R.geometry(f"{w}x{h}+5+30")
        R.minsize(600, 400)

    def _ajustar_rolagem(self):
        """Atualiza a área rolável e mostra cada barra só quando precisa."""
        pg = self.pagina
        bb = pg.bbox("all")
        if not bb:
            return
        pg.configure(scrollregion=bb)
        precisa_v = bb[3] > pg.winfo_height() + 1
        precisa_h = bb[2] > pg.winfo_width() + 1
        if precisa_v and not self.rv.winfo_ismapped():
            self.rv.grid(row=0, column=1, sticky="ns")
        elif not precisa_v and self.rv.winfo_ismapped():
            self.rv.grid_remove(); pg.yview_moveto(0)
        if precisa_h and not self.rh.winfo_ismapped():
            self.rh.grid(row=1, column=0, sticky="ew")
        elif not precisa_h and self.rh.winfo_ismapped():
            self.rh.grid_remove(); pg.xview_moveto(0)

    def _rolar_pagina(self, e, eixo="y"):
        if eixo == "y" and self.rv.winfo_ismapped():
            self.pagina.yview_scroll(int(-e.delta), "units")
        elif eixo == "x" and self.rh.winfo_ismapped():
            self.pagina.xview_scroll(int(-e.delta), "units")

    def _alternar_avancado(self):
        self.av_aberto = not self.av_aberto
        self._mostrar_avancado()

    def _mostrar_avancado(self):
        if self.av_aberto:
            self.av.pack(fill="x", padx=10, after=self.b_av)
            self.b_av.config(text="▾ esconder ferramentas avançadas")
        else:
            self.av.pack_forget()
            self.b_av.config(text="▸ ferramentas avançadas (recalcular, ignorar, trechos, obedecer)")

    def _rotular_botoes(self):
        self.b_fant.config(text="marcações de outros quadros: " +
                           ("nenhuma", "anterior e seguinte", "todas")[self.fantasma] + "  (f)")
        self.b_cruz.config(text="mouse: " + ("cursor normal", "cruz longa", "cruz curta")[self.cruz]
                           + "  (c)")

    def _carregar_quadros(self):
        self.lb.config(text="lendo o vídeo uma vez (só na primeira abertura)…")
        self.raiz.update()
        self.quadros, self.Wq, self.Hq = cache_quadros(
            escala=self.Ec, x=(self.XA, self.XB))
        K = np.median(self.n.C[:, :, self.n.cols, :], axis=2)[..., ::-1]
        K = cv2.resize(K.transpose(1, 0, 2).astype(np.uint8), (self.W, self.HK),
                       interpolation=cv2.INTER_AREA)
        self.img_kim = ImageTk.PhotoImage(Image.fromarray(K))
        self.ck.create_image(0, 0, anchor="nw", image=self.img_kim)
        self.lb.config(text="")

    # ------------------------------------------------------- desenho #
    def _refazer_ref(self):
        """A trajetoria que as suas marcacoes definem. Desenhada tracejada.

        Na sessão ela só é calculada quando aparece na tela (linhas ligadas):
        refazê-la a cada clique custava ~0,4 s e enfileirava os cliques até a
        janela parecer travada (crash relatado em 21/10/2026).
        """
        self._ref_sujo = True
        if not self._modo_sessao() or (self.ver_ref.get() and any(v.get() for v in self.ver.values())):
            self._ref_atual()

    def _ref_atual(self):
        if getattr(self, "_ref_sujo", True) or getattr(self, "ref", None) is None:
            self.ref = {f: self.n.referencia(f, self._da_camada()) for f in FACES}
            self._ref_sujo = False
        return self.ref

    # Coordenadas: x do quadro e y da ROI <-> pixel da tela, com zoom Z e origem
    # da vista (vx, vy) em pixel nativo. Sem zoom, é o mapeamento antigo.
    def cx(self, x):  return (x - self.XA - self.vx) * self.E * self.Z
    def cy(self, y):  return (y - self.vy) * self.E * self.Z
    def px(self, cx): return self.XA + self.vx + cx / (self.E * self.Z)
    def py(self, cy): return self.vy + cy / (self.E * self.Z)

    def _limitar_vista(self):
        s = self.E * self.Z
        self.vx = float(np.clip(self.vx, 0, max(0.0, (self.XB - self.XA) - self.W / s)))
        self.vy = float(np.clip(self.vy, 0, max(0.0, (self.n.Y1 - self.n.Y0) - self.HQ / s)))

    def zoom(self, fator, ex=None, ey=None):
        """Aproxima ou afasta mantendo fixo o ponto sob o mouse."""
        ex, ey = (self.mouse if ex is None else (ex, ey))
        ux, uy = self.px(ex), self.py(ey)
        self.Z = float(np.clip(self.Z * fator, 1.0, ZOOM_MAX))
        s = self.E * self.Z
        self.vx, self.vy = ux - self.XA - ex / s, uy - ey / s
        self._limitar_vista(); self._pintar()

    def zoom_zerar(self):
        self.Z, self.vx, self.vy = 1.0, 0.0, 0.0
        self._pintar()

    def mover_vista(self, dx, dy):
        """Arrasta a vista em pixel da tela (só faz sentido com zoom)."""
        s = self.E * self.Z
        self.vx += dx / s; self.vy += dy / s
        self._limitar_vista(); self._pintar()

    def _roda(self, e, eixo):
        if self.Z <= 1.0:
            return None               # sem zoom, a roda rola a página
        d = -4.0 * e.delta
        self.mover_vista(d if eixo == "x" else 0.0, d if eixo == "y" else 0.0)
        return "break"

    def _roda_zoom(self, e):
        self.zoom(1.15 if e.delta > 0 else 1 / 1.15, e.x, e.y)
        return "break"

    def _imagem_visivel(self, bgr):
        """Recorta e amplia a parte visível do quadro (resolução cheia em cache)."""
        if self.Z <= 1.0 and self.vx == 0 and self.vy == 0:
            return cv2.resize(bgr, (self.W, self.HQ), interpolation=cv2.INTER_AREA)
        s = self.E * self.Z
        M = np.float32([[s / self.Ec, 0, -self.vx * s], [0, s / self.Ec, -self.vy * s]])
        return cv2.warpAffine(bgr, M, (self.W, self.HQ), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    def _foto(self, bgr):
        return ImageTk.PhotoImage(Image.frombytes(
            "RGB", (self.W, self.HQ), np.ascontiguousarray(bgr[..., ::-1]).tobytes()))

    def _com_cruz(self, vis):
        """Mistura a cruz do mouse na imagem (opacidade CRUZ_ALFA, vão no centro)."""
        if not self.cruz or self.mouse_fora:
            return vis
        x, y = int(self.mouse[0]), int(self.mouse[1])
        if not (0 <= x < self.W and 0 <= y < self.HQ):
            return vis
        out = vis.copy()
        L = max(self.W, self.HQ) if self.cruz == 1 else CRUZ_CURTA
        g = CRUZ_VAO

        def mistura(idx):
            out[idx] = (out[idx].astype(np.float32) * (1 - CRUZ_ALFA)
                        + CRUZ_COR * CRUZ_ALFA).astype(np.uint8)
        for a, b in ((max(0, x - L), max(0, x - g)), (min(self.W, x + g + 1), min(self.W, x + L + 1))):
            if b > a:
                mistura((y, slice(a, b)))
        for a, b in ((max(0, y - L), max(0, y - g)), (min(self.HQ, y + g + 1), min(self.HQ, y + L + 1))):
            if b > a:
                mistura((slice(a, b), x))
        return out

    def _mouse_move(self, e):
        self.mouse, self.mouse_fora = (e.x, e.y), False
        if self.cruz and not self._cruz_pendente:
            # junta vários movimentos num só redesenho: a janela não engasga
            self._cruz_pendente = True
            self.raiz.after(8, self._atualizar_cruz)

    def _mouse_sai(self, e):
        self.mouse_fora = True
        self._atualizar_cruz()

    def _atualizar_cruz(self):
        self._cruz_pendente = False
        if self._vis is None or self.id_img is None:
            return
        self.img_q = self._foto(self._com_cruz(self._vis))
        self.cq.itemconfig(self.id_img, image=self.img_q)

    def _cursor(self):
        try:
            self.cq.config(cursor="none" if self.cruz else "crosshair")
        except tk.TclError:
            self.cq.config(cursor="crosshair")

    def alternar_cruz(self):
        self.cruz = (self.cruz + 1) % 3
        self._cursor()
        self._rotular_botoes()
        self.lb2.config(text="mouse: " + ("cursor normal", "cruz longa", "cruz curta")[self.cruz])
        self._atualizar_cruz()

    def ir_marcado(self, sentido):
        """Pula para o quadro marcado anterior (-1) ou seguinte (+1)."""
        ks = sorted({p["k"] for p in self.ancoras})
        alvo = ([k for k in ks if k < self.i][-1:] if sentido < 0 else
                [k for k in ks if k > self.i][:1])
        if alvo:
            self.i = alvo[0]; self._pintar()
        else:
            self.lb2.config(text="não há quadro marcado nessa direção")

    def alternar_fantasma(self):
        self.fantasma = (self.fantasma + 1) % 3
        self._rotular_botoes()
        self.lb2.config(text=("marcações de outros quadros: " +
                              ("desligadas", "quadro marcado anterior e seguinte", "todas")[self.fantasma]))
        self._pintar()

    def _pintar(self):
        i = self.i
        b = self.quadros[i]
        self._vis = self._imagem_visivel(cv2.imdecode(np.frombuffer(b, np.uint8), 1))
        self.img_q = self._foto(self._com_cruz(self._vis))
        self.cq.delete("all")
        self.id_img = self.cq.create_image(0, 0, anchor="nw", image=self.img_q)
        passo_regua = 100 if self.Z < 2 else (50 if self.Z < 4 else 20)
        for y in range(0, self.n.H, passo_regua):
            self.cq.create_line(0, self.cy(y), self.W, self.cy(y),
                                fill=GUIA, width=1, dash=(1, 6))
            self.cq.create_text(4, self.cy(y) + 8, text=str(y), anchor="w",
                                fill=GUIA, font=("Menlo", 9))
        self.cq.create_line(0, self.cy(self.n.yc_med), self.W,
                            self.cy(self.n.yc_med), fill="#ffffff", dash=(7, 6))
        cols = self.cols_atual
        fora = not (self.n.i_ign <= i <= self.n.i_apag)
        for f in FACES:
            if not self.ver[f].get():
                continue
            if self.ver_ref.get() and self.ancoras:
                ref = self._ref_atual()
                pts = []
                for j in cols:
                    pts += [self.cx(self.n.xc[j]), self.cy(ref[f][i, j])]
                if len(pts) >= 4:
                    self.cq.create_line(*pts, fill="#00ff6a", width=1,
                                        dash=(6, 5))
            pts = []
            for j in cols:
                pts += [self.cx(self.n.xc[j]), self.cy(self.n.Y[f][i, j])]
            if len(pts) >= 4:
                # fora da queima a linha existe mas nao mede nada: fica apagada
                self.cq.create_line(*pts, fill=COR[f], width=1 if fora else 2,
                                    dash=(3, 4) if fora else ())
        if fora:
            self.cq.create_text(self.W - 12, 16, anchor="e",
                                text="fora da queima — a linha aqui não é medida",
                                fill="#bbb", font=("", 11))
        self._pintar_fantasmas(i)
        if self.Z > 1.0:
            self.cq.create_text(self.W - 12, self.HQ - 14, anchor="e",
                                text=f"zoom {self.Z:.1f}x  (0 volta ao normal)",
                                fill="#ffffff", font=("", 11, "bold"))
        for p in self.ancoras:
            if p["k"] != i:
                continue
            c = p.get("camada", "A")
            s, larg = (7, 3) if c == self.camada else (6, 2)
            x, y = self.cx(p["x"]), self.cy(p["y"])
            # so um "x" pequeno: linha cheia tapava os outros pontos
            self.cq.create_line(x - s, y - s, x + s, y + s, fill=COR_CAMADA[c], width=larg)
            self.cq.create_line(x - s, y + s, x + s, y - s, fill=COR_CAMADA[c], width=larg)
        if (self.sugestao_atual is not None and
                self.sugestao_atual["k"] == i):
            f = self.sugestao_atual["face"]
            if "x" in self.sugestao_atual:
                xx = self.cx(self.sugestao_atual["x"])
                self.cq.create_line(xx, 0, xx, self.HQ, fill=GUIA,
                                    width=2, dash=(6, 5))
            self.cq.create_text(
                12, self.HQ - 16, anchor="w",
                text=(f"ESTADO INICIAL: marque a face de {f} perto da linha amarela"
                      if "x" in self.sugestao_atual else
                      (f"marque a face de {f} no grão inteiro, ABAIXO da fuligem"
                       if self.sugestao_atual.get("rotulo") else
                       f"SUGESTÃO: marque a face de {f}")),
                fill=COR[f], font=("", 14, "bold"))
        if self.k_min is not None and i < self.k_min:
            self.cq.create_text(self.W // 2, self.HQ // 2,
                                text=f"antes de {self.k_min_s:g} s após a ignição: não marcar",
                                fill="#ffffff", font=("", 20, "bold"))
        elif (not self.livre and self.n.k_livre is not None and self.n.i_ign <= i < self.n.k_livre):
            self.cq.create_text(self.W - 12, 60, anchor="e",
                                text=f"transiente da ignição (até {self.a.inicio_medida_s:g} s): "
                                     "pontos livres, sem a regra de não recuar",
                                fill="#ffffff", font=("", 11, "bold"))
        if bool(self.n.flare[i]) and i not in self.ignora["cima"]:
            self.cq.create_text(self.W - 12, 46, anchor="e",
                                text="estouro de luz neste quadro — pesa pouco no ajuste",
                                fill="#ffffff", font=("", 11))
        if i in self.ignora["cima"]:
            self.cq.create_rectangle(0, 0, self.W, self.HQ, fill="#6a2cff",
                                     stipple="gray25", outline="")
            self.cq.create_text(12, 16, text="IGNORADO", anchor="w",
                                fill="#e6dcff", font=("", 15, "bold"))
        self._pintar_tempo(i)
        if self.faixa:
            self._pintar_faixa(i)
        self._pintar_kim()
        if not self._mexendo:
            self._mexendo = True; self.esc.set(i); self._mexendo = False
        self.lb_pos.config(
            text=f"{self.n.t[i] - self.n.t[self.n.i_ign]:+6.1f} s da ignição  ·  "
                 f"{self.n.t_video[i]:6.1f} s marcados pela câmera  ·  amostra {i + 1}/{self.n.N}")
        self._estado()

    def _pintar_kim(self):
        self.ck.delete("ov")
        sx = self.W / self.n.N; sy = self.HK / self.n.H
        if self.ver_flare.get():
            for a, b in trechos(np.flatnonzero(self.n.flare).tolist()):
                self.ck.create_rectangle(a * sx, 0, (b + 1) * sx, 8,
                                         fill="#ffffff", outline="", tags="ov")
        for a, b in trechos(self.ignora["cima"]):
            self.ck.create_rectangle(a * sx, 0, (b + 1) * sx, self.HK,
                                     fill="#6a2cff", stipple="gray50",
                                     outline="", tags="ov")
        for f in FACES:
            if not self.ver[f].get():
                continue
            m = np.nanmedian(self.n.Y[f][:, self.cols_atual], 1)
            pts = []
            for k in range(0, self.n.N, 2):
                pts += [k * sx, m[k] * sy]
            self.ck.create_line(*pts, fill=COR[f], width=2, tags="ov")
        for p in self.ancoras:
            self.ck.create_oval(p["k"] * sx - 3, p["y"] * sy - 3,
                                p["k"] * sx + 3, p["y"] * sy + 3,
                                fill=COR_CAMADA[p.get("camada", "A")], outline="", tags="ov")
        self.ck.create_line(self.i * sx, 0, self.i * sx, self.HK,
                            fill="#fff", width=2, tags="ov")
        if self.faixa:
            a, b = self.faixa[0] * sx, (self.faixa[1] + 1) * sx
            self.ck.create_rectangle(a, self.HK - 12, b, self.HK, fill="#00ff6a", outline="", tags="ov")
            for x in (a, b):
                self.ck.create_line(x, 0, x, self.HK, fill="#00ff6a", width=2, tags="ov")

    def _pa(self) -> int:
        """Passo de reprodução. Campo vazio ou inválido (no meio da digitação) vale 1."""
        try:
            return max(1, int(self.passo.get()))
        except (tk.TclError, ValueError):
            return 1

    def _estado(self):
        """Status curto. A linha de avisos (lb2) fica só para avisos: antes ela era
        reescrita a cada redesenho e o aviso de pontos ajustados sumia na hora."""
        self._verificar_alteracao()
        self._agendar_graficos()
        pa = self._pa()
        self.lb.config(
            text=f"{len(self.ancoras)} pontos · passo {pa} = {self.n.dt * pa:.1f} s"
                 + ("  · recalcular" if self.sujo and self.av_aberto else ""))

    # -------------------------------------------------------- graficos #
    def alternar_graficos(self):
        """Abre ou fecha a janela de gráficos, ao lado da janela de marcação."""
        if self.jan_graficos is not None:
            self.jan_graficos.destroy()
            self.jan_graficos, self.painel, self._sig_graficos = None, None, None
            return
        import graficos_marcacao as GM      # matplotlib só carrega se os gráficos abrirem
        R = self.raiz
        R.update_idletasks()
        x = min(R.winfo_rootx() + R.winfo_width() + 6, max(R.winfo_screenwidth() - LARG_GRAFICOS, 0))
        alt = max(R.winfo_screenheight() - 80, 500)
        self.jan_graficos = tk.Toplevel(R)
        self.jan_graficos.title("gráficos da marcação")
        self.jan_graficos.geometry(f"{LARG_GRAFICOS}x{alt}+{x}+30")
        self.jan_graficos.transient(R)
        self.jan_graficos.protocol("WM_DELETE_WINDOW", self.alternar_graficos)
        self.painel = GM.Painel(self.jan_graficos, LARG_GRAFICOS, alt, self.faixa)
        R.after(200, R.focus_force)
        self._agendar_graficos()

    def _agendar_graficos(self):
        """Refaz os gráficos pouco depois de cada alteração, sem travar o clique."""
        if self.painel is None or self._graficos_pendente:
            return
        if self._sig_graficos != (self._assinatura(), self.camada):
            self._graficos_pendente = True
            self.raiz.after(250, self._atualizar_graficos)

    def _atualizar_graficos(self):
        self._graficos_pendente = False
        if self.painel is None:
            return
        self._sig_graficos = (self._assinatura(), self.camada)
        try:
            self.painel.atualizar(self.ancoras, self.camada)
        except Exception as erro:     # um gráfico nunca derruba a marcação
            self.lb2.config(text=f"gráficos: {erro}")

    # ------------------------------------------------------ navegacao #
    def ir(self, d, abs_=False):
        self.i = int(np.clip(d if abs_ else self.i + d, 0, self.n.N - 1))
        self._pintar()

    def _arrastar(self, v):
        if self._mexendo:
            return
        self.i = int(float(v)); self._pintar()

    def _pintar_fantasmas(self, i):
        """Marcações de outros quadros, dos dois grupos, para comparar a regressão."""
        if not self.fantasma:
            return
        for c, (cor_ant, cor_prox) in FANT_CAMADA.items():
            pontos = [p for p in self.ancoras if p.get("camada", "A") == c]
            for f in FACES:
                ks = sorted({p["k"] for p in pontos if p["face"] == f and p["k"] != i})
                if self.fantasma == 2:
                    for p in pontos:
                        if p["face"] == f and p["k"] != i:
                            x, y = self.cx(p["x"]), self.cy(p["y"])
                            self.cq.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5,
                                                outline=cor_prox)
                ant = [k for k in ks if k < i][-1:]
                prox = [k for k in ks if k > i][:1]
                for kk, cor in ((ant, cor_ant), (prox, cor_prox)):
                    if not kk:
                        continue
                    pts = sorted((p["x"], p["y"]) for p in pontos
                                 if p["face"] == f and p["k"] == kk[0])
                    xy = [v for x, y in pts for v in (self.cx(x), self.cy(y))]
                    if len(xy) >= 4:
                        self.cq.create_line(*xy, fill=cor, width=1, dash=(2, 4))
                    for x, y in pts:
                        X, Y = self.cx(x), self.cy(y)
                        self.cq.create_oval(X - 2, Y - 2, X + 2, Y + 2, outline=cor)

    def _clique_kim(self, e):
        self.i = int(np.clip(e.x / (self.W / self.n.N), 0, self.n.N - 1))
        self._pintar()

    def proxima_sugestao(self):
        """Vai para a próxima face/quadro ainda não rotulada."""
        total = len(self.sugestoes)
        for _ in range(total):
            s = self.sugestoes[self.ind_sugestao % total]
            self.ind_sugestao = (self.ind_sugestao + 1) % total
            feito = any(p["face"] == s["face"] and p["k"] == s["k"]
                        for p in self._da_camada())
            if not feito and s["k"] not in self.ignora[s["face"]]:
                self.sugestao_atual = s
                self.i = s["k"]
                feitos = sum(any(p["face"] == q["face"] and p["k"] == q["k"]
                                 for p in self._da_camada()) for q in self.sugestoes)
                self.lb_sug.config(
                    foreground=COR[s["face"]],
                    text=(f"Marque a face de {s['face'].upper()}   ·   {s['rotulo']}   ·   "
                          f"{s['t_real_s']:.1f} s marcados pela câmera   ·   pedidos feitos: {feitos} de {total}")
                    if s.get("rotulo") else
                         (f"face {s['face']} · alvo ≈{s['alvo_video_s']:.0f}s do vídeo "
                          f"· escolhido {s['t_video_s']:.1f}s = {s['t_real_s']:.1f}s reais "
                          f"· incerteza {s['incerteza']:.2f}"))
                self._pintar()
                return
        self.sugestao_atual = None
        self.lb_sug.config(text=f"todos os {total} pedidos já foram marcados ou ignorados. SALVE o perfil.")

    def proxima_inicial(self):
        """Pede apenas as extremidades que faltam no perfil pré-ignição."""
        cobertura = self.n.cobertura_inicial(self.ancoras)
        faltam = []
        for face in FACES:
            c = cobertura[face]
            if c["xmin"] is None or c["xmin"] > c["alvo_min"] + 20:
                faltam.append((face, c["alvo_min"], "esquerda"))
            if c["xmax"] is None or c["xmax"] < c["alvo_max"] - 20:
                faltam.append((face, c["alvo_max"], "direita"))
        # pula as posições que já ganharam um clique próximo nesta sessão
        for face, x, lado in faltam:
            if not any(p["face"] == face and p["k"] < self.n.i_ign and
                       abs(p["x"] - x) <= 25 for p in self.ancoras):
                self.sugestao_atual = dict(face=face, k=0, x=x, inicial=True)
                self.i = 0
                self.lb_sug.config(
                    text=f"estado inicial · face {face} · extremidade {lado} "
                         f"(x≈{x:.0f}px). Clique na borda visível.")
                self._pintar()
                return
        self.sugestao_atual = None
        self.lb_sug.config(text="estado inicial coberto nas duas extremidades e nas duas faces")

    def tocar(self):
        self.tocando = not self.tocando
        self.b_play.config(text="❚❚ pausar" if self.tocando else "▶  tocar")
        if self.tocando:
            self._passo()

    def _passo(self):
        if not self.tocando:
            return
        self.i = (self.i + self.vel.get() * self._pa()) % self.n.N
        self._pintar()
        self.raiz.after(60, self._passo)

    # --------------------------------------------------------- edicao #
    def _clique_esq(self, e):
        if self.k_min is not None and self.i < self.k_min:
            self.lb2.config(text=f"antes de {self.k_min_s:g} s após a ignição não se marca "
                                 "(chama ainda não consolidada)")
            return
        x, y = self.px(e.x), self.py(e.y)
        f = "cima" if y < self.n.yc_med else "baixo"
        p = dict(k=self.i, face=f, x=x, y=y, camada=self.camada,
                 j=int(np.argmin(np.abs(self.n.xc - x))))
        # Clicar praticamente EM CIMA de um ponto do mesmo quadro o corrige; pontos
        # próximos mas distintos convivem, para dar detalhe às curvas (pedido do
        # usuário, 18/10/2026; antes qualquer ponto a menos de 60 px era trocado).
        mesma = self._da_camada()
        trocados = [q for q in mesma if q["face"] == f and q["k"] == self.i
                    and np.hypot(q["x"] - x, q["y"] - y) <= RAIO_TROCA]
        restantes = [q for q in mesma if q not in trocados]
        # a marcação mais nova manda: pontos antigos incompatíveis se ajustam a ela
        # Só recusa por conflito que ESTE clique cria: um conflito antigo entre
        # outros pontos não pode travar toda marcação nova (achado em 22/10/2026).
        ja_havia = Counter(self.n.erros_ancoras(restantes))
        movidos = self.n.ajustar_ao_novo(restantes, p)
        novos = list((Counter(self.n.erros_ancoras(restantes + [p])) - ja_havia).elements())
        if novos:
            for q, y_velho in movidos:
                q["y"] = y_velho
            # aviso na linha de avisos, sem caixa modal: no macOS a caixa às vezes
            # abria ATRÁS da janela e o programa parecia travado
            self.raiz.bell()
            self.lb2.config(text="ponto NÃO aplicado (nada foi alterado): " + novos[0])
            return
        for q in trocados:
            self.ancoras.remove(q)
        self.ancoras.append(p)
        self.pilha.append(("add", p, movidos, trocados))
        avisos = []
        if trocados:
            avisos.append(f"{len(trocados)} ponto(s) deste quadro substituído(s)")
        if movidos:
            qs = sorted({int(q["k"]) for q, _ in movidos})
            avisos.append(f"{len(movidos)} ponto(s) de outros quadros ajustado(s) à nova "
                          f"marcação (quadros {qs[:6]}{'…' if len(qs) > 6 else ''}); control+Z desfaz")
        if avisos:
            self.lb2.config(text=" · ".join(avisos))
        self.sujo = True; self._refazer_ref(); self._pintar()

    def _clique_dir(self, e):
        x, y = self.px(e.x), self.py(e.y)
        f = "cima" if y < self.n.yc_med else "baixo"
        cand = [n for n, p in enumerate(self.ancoras)
                if p["k"] == self.i and p["face"] == f and p.get("camada", "A") == self.camada]
        if not cand:
            return
        n = min(cand, key=lambda n: abs(self.ancoras[n]["x"] - x))
        self.pilha.append(("del", self.ancoras.pop(n)))
        self.sujo = True; self._refazer_ref(); self._pintar()

    def _mudou_rigidez(self):
        self.n.par.obediencia = float(self.rig.get()) / 100.0
        if self.ancoras:
            self.sujo = True
        self._estado()

    def ignorar(self):
        novo = self.i not in self.ignora["cima"]
        for f in FACES:
            self.ignora[f].add(self.i) if novo else self.ignora[f].discard(self.i)
        self.pilha.append(("ign", self.i, novo))
        self.sujo = True; self._pintar()

    def ignorar_flares(self):
        """
        Manda para o descarte os quadros com estouro de luz, achados sozinhos.

        Eles ja entram no calculo com peso baixo; este botao e para quando voce
        quiser tira-los de vez. O criterio nao usa rotulo: desvio do brilho
        medio e da fracao de pixels saturados em relacao a mediana movel.
        """
        novos = [k for k in np.flatnonzero(self.n.flare).tolist()
                 if k not in self.ignora["cima"]]
        if not novos:
            messagebox.showinfo("flares", "nenhum flare novo para ignorar")
            return
        for f in FACES:
            self.ignora[f].update(novos)
        self.pilha.append(("trecho", novos))
        self.sujo = True; self._pintar()
        messagebox.showinfo("flares", f"{len(novos)} quadro(s) com estouro de "
                                      "luz marcados para ignorar")

    def trecho(self, fim):
        if not fim:
            self.ini_trecho = self.i
            self.lb2.config(text=f"início do trecho em t={self.n.t[self.i]:.0f}s")
            return
        if self.ini_trecho is None:
            messagebox.showinfo("trecho", "marque o início primeiro")
            return
        a, b = sorted((self.ini_trecho, self.i)); self.ini_trecho = None
        novos = [k for k in range(a, b + 1) if k not in self.ignora["cima"]]
        for f in FACES:
            self.ignora[f].update(novos)
        self.pilha.append(("trecho", novos))
        self.sujo = True; self._pintar()

    def limpar_quadro(self):
        sai = [p for p in self._da_camada() if p["k"] == self.i]
        fica = [p for p in self.ancoras if not any(p is q for q in sai)]
        if sai:
            self.pilha.append(("lote", sai))
            self.ancoras = fica; self.sujo = True
            self._refazer_ref(); self._pintar()

    def desfazer(self):
        if not self.pilha:
            return
        op = self.pilha.pop()
        if op[0] == "add":
            self.ancoras.remove(op[1])
            for ponto, y_velho in (op[2] if len(op) > 2 else []):
                ponto["y"] = y_velho
            self.ancoras += (op[3] if len(op) > 3 else [])
        elif op[0] == "del":
            self.ancoras.append(op[1])
        elif op[0] == "lote":
            self.ancoras += op[1]
        elif op[0] == "ign":
            for f in FACES:
                self.ignora[f].discard(op[1]) if op[2] else self.ignora[f].add(op[1])
        else:
            for f in FACES:
                self.ignora[f].difference_update(op[1])
        self.sujo = True; self._refazer_ref(); self._pintar()

    def zerar(self):
        if not messagebox.askyesno("voltar ao automático",
                                   "descarta os pontos e os trechos ignorados?"):
            return
        self.ancoras.clear(); self.pilha.clear()
        for f in FACES:
            self.ignora[f].clear()
            self.n.Y[f] = self.n.Y_ini[f].copy()
        self.cols_atual = self.n.cols; self.sujo = False
        self._refazer_ref(); self._pintar()

    # --------------------------------------------------------- perfis #
    def _nome_livre(self, base: str) -> str:
        nome, n = base, 2
        while nome in self.perfis or (self.pasta_perfis / id_perfil(nome)).exists():
            nome, n = f"{base}-{n}", n + 1
        return nome

    def novo_perfil(self, nome: str | None = None):
        """Começa um perfil VAZIO (pedido do usuário, 21/10/2026).

        Oferece salvar o atual, pede o nome (sugere sessao-AAAA-MM-DD), limpa as
        marcações e grava o perfil novo já vazio, para ele aparecer na lista e o
        SALVAR passar a gravar nele. Os perfis antigos ficam intactos.
        """
        if not self._modo_sessao():
            messagebox.showinfo("novo perfil", "disponível na sessão de marcação (--sessao)")
            return
        if self.alterado and (self.ancoras or any(self.ignora[f] for f in FACES)):
            resp = messagebox.askyesnocancel(
                "novo perfil", "Há marcações não salvas neste perfil.\n\n"
                               "Salvar antes de começar um novo?")
            if resp is None or (resp and not self._salvar_sessao()):
                return
        if nome is None:
            nome = simpledialog.askstring(
                "novo perfil", "Nome do perfil novo (começa sem nenhuma marcação):",
                initialvalue=self._nome_livre("sessao-" + time.strftime("%Y-%m-%d")),
                parent=self.raiz)
            if not nome or not nome.strip():
                return
        nome = nome.strip()
        if nome in self.perfis or (self.pasta_perfis / id_perfil(nome)).exists():
            messagebox.showerror("novo perfil", f'Já existe um perfil "{nome}". Escolha outro nome.')
            return
        self.ancoras, self.ignora = [], {f: set() for f in FACES}
        self.pilha.clear()
        self.n.Y = {f: self.n.Y_ini[f].copy() for f in FACES}
        self.cols_atual = self.n.cols
        self.sujo = False
        self.sugestao_atual, self.ind_sugestao = None, 0
        if self.k_min is not None:
            self.i = self.k_min
        self._refazer_ref()
        self.nome_sessao = nome
        self._salvar_sessao()
        self.lb2.config(text=f'perfil novo "{nome}" criado, sem marcações. '
                             "Aperte PRÓXIMO PEDIDO para começar.")
        self._pintar()

    @property
    def pasta_perfis(self) -> Path:
        return Path(self.a.out) / "perfis"

    def _atualizar_perfis(self, selecionar: str | None = None):
        """Atualiza a lista sem confiar no nome da pasta como nome humano."""
        perfis: dict[str, Path] = {}
        # na sessão, procura também em resultados/perfis, onde as versões
        # anteriores do programa salvavam quando abertas sem --out
        pastas = [self.pasta_perfis]
        antiga = RAIZ / "resultados/perfis"
        if self._modo_sessao() and antiga.resolve() != self.pasta_perfis.resolve():
            pastas.append(antiga)
        for raiz_p in pastas:
            if not raiz_p.exists():
                continue
            for pasta in raiz_p.iterdir():
                meta = pasta / "perfil.json"
                guia = pasta / "guia.csv"
                npz = pasta / "interfaces_revisado.npz"
                if not (pasta.is_dir() and meta.exists() and guia.exists() and npz.exists()):
                    continue
                try:
                    nome = str(json.loads(meta.read_text(encoding="utf-8"))["nome"]).strip()
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if nome:
                    perfis.setdefault(nome, pasta)

        # Compatibilidade com o salvamento único das versões anteriores.
        ativo = Path(self.a.out)
        if (not self._modo_sessao() and (ativo / "guia.csv").exists()
                and (ativo / "interfaces_revisado.npz").exists()):
            perfis.setdefault("Último salvamento (sem nome)", ativo)

        self.perfis = dict(sorted(perfis.items(), key=lambda x: x[0].casefold()))
        self.cb_perfil.configure(values=list(self.perfis))
        if selecionar is not None:
            self.perfil.set(selecionar)
        elif not self.perfil.get() and self.perfis:
            self.perfil.set(next(iter(self.perfis)))

    def _ler_guia(self, caminho: Path):
        ancoras: list[dict] = []
        ignora = {f: set() for f in FACES}
        with caminho.open(newline="", encoding="utf-8") as fh:
            for linha in csv.DictReader(fh):
                face = linha.get("face", "")
                if face not in FACES:
                    continue
                k = int(linha["k"])
                if not 0 <= k < self.n.N:
                    raise ValueError(f"quadro {k} fora do ensaio")
                if linha.get("tipo") == "ancora":
                    x = float(linha["x_px"]); y = float(linha["y_roi_px"])
                    ancoras.append(dict(k=k, face=face, x=x, y=y, camada=linha.get("camada") or "A",
                                        j=int(np.argmin(np.abs(self.n.xc - x)))))
                elif linha.get("tipo") == "ignorar":
                    ignora[face].add(k)
        return ancoras, ignora

    def abrir_perfil(self):
        nome = self.perfil.get().strip()
        pasta = self.perfis.get(nome)
        if pasta is None:
            messagebox.showinfo("abrir perfil", "selecione um perfil já salvo")
            return
        if self.sujo and not messagebox.askyesno(
                "abrir perfil", "há alterações não salvas. descartá-las e abrir o perfil?"):
            return
        try:
            ancoras, ignora = self._ler_guia(pasta / "guia.csv")
            with np.load(pasta / "interfaces_revisado.npz", allow_pickle=False) as dados:
                trajetorias = {f: np.asarray(dados[CH[f]], dtype=float).copy()
                               for f in FACES}
            if any(trajetorias[f].shape != self.n.Y_ini[f].shape for f in FACES):
                raise ValueError("o perfil pertence a outra geometria ou versão do ensaio")
            meta = {}
            if (pasta / "perfil.json").exists():
                meta = json.loads((pasta / "perfil.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("erro ao abrir perfil", str(exc))
            return

        self.ancoras, self.ignora = ancoras, ignora
        self.n.Y = trajetorias
        self.pilha.clear(); self.cols_atual = self.n.cols
        if "obediencia" in meta:
            self.rig.set(100 * float(meta["obediencia"]))
            self.n.par.obediencia = float(meta["obediencia"])
        self.sujo = False
        if self._modo_sessao():
            self.nome_sessao = nome
            self._assinatura_salva = self._assinatura()
            self._titulo()
        self._refazer_ref(); self._pintar()

        # O perfil aberto vira também a seleção ativa usada pelo pipeline.
        if pasta != Path(self.a.out):
            shutil.copy2(pasta / "guia.csv", Path(self.a.out) / "guia.csv")
            shutil.copy2(pasta / "interfaces_revisado.npz",
                         Path(self.a.out) / "interfaces_revisado.npz")
        if self._modo_sessao():
            self.lb2.config(text=f'perfil "{nome}" aberto  ·  {len(ancoras)} pontos carregados')
        else:
            messagebox.showinfo("perfil aberto", f"{nome}\n\n{len(ancoras)} ponto(s) carregado(s)")

    # ----------------------------------------------------- recalculo #
    def recalcular(self, completo, depois=None):
        if self.calculando:
            return
        self.calculando = True
        self.tocando = False; self.b_play.config(text="▶  tocar")
        for b in (self.b_calc, self.b_salvar, self.b_abrir, self.b_novo):
            b.state(["disabled"])
        self.pb.pack(side="left", padx=10)
        self.pb.config(value=0, maximum=100)
        # sempre TODAS as colunas: a regularizacao espacial precisa delas.
        # Com subconjunto, o ajuste rigido em x fica sem redundancia para
        # rejeitar bloco errado, que era justamente o problema a resolver.
        cols = None

        def tarefa():
            try:
                Y, usadas = self.n.recalcular(
                    self.ancoras, self.ignora, cols=cols,
                    progresso=lambda a, b: setattr(self, "prog", (a, b)))
                self.fila.put(("ok", Y, usadas))
            except Exception as exc:                       # nao mata a janela
                self.fila.put(("erro", exc, None))

        threading.Thread(target=tarefa, daemon=True).start()
        self._vigiar(depois)

    def _vigiar(self, depois):
        try:
            tipo, a, b = self.fila.get_nowait()
        except queue.Empty:
            f, t = self.prog
            self.pb.config(value=100 * f / max(t, 1))
            self.lb2.config(text=f"calculando… {f}/{t} colunas")
            self.raiz.after(60, lambda: self._vigiar(depois))
            return
        self.calculando = False
        self.pb.pack_forget()
        for x in (self.b_calc, self.b_salvar, self.b_abrir, self.b_novo):
            x.state(["!disabled"])
        if tipo == "erro":
            messagebox.showerror("erro no recálculo", str(a))
            return
        self.n.Y = a; self.cols_atual = b; self.sujo = False
        self._pintar()
        if depois:
            depois()

    def salvar(self):
        if self._modo_sessao():
            return self._salvar_sessao()
        incompletas = [f for f, c in self.n.cobertura_inicial(self.ancoras).items()
                       if not c["completa"]]
        if incompletas and not messagebox.askyesno(
                "estado inicial incompleto",
                "Faltam âncoras nas extremidades do estado inicial em: "
                + ", ".join(incompletas) + ".\n\n"
                "A extrapolação será plana até a borda. Salvar mesmo assim?"):
            return
        inicial = self.perfil.get().strip()
        if inicial == "Último salvamento (sem nome)":
            inicial = ""
        nome = simpledialog.askstring(
            "salvar perfil", "Nome deste conjunto de marcações:",
            initialvalue=inicial, parent=self.raiz)
        if nome is None:
            return
        nome = nome.strip()
        if not nome:
            messagebox.showerror("salvar perfil", "digite um nome para o perfil")
            return
        pasta = self.pasta_perfis / id_perfil(nome)
        if pasta.exists() and not messagebox.askyesno(
                "substituir perfil", f'Já existe o perfil "{nome}". Substituir?'):
            return
        self.nome_para_salvar = nome
        self.recalcular(True, depois=self._gravar)

    def _gravar(self):
        nome = self.nome_para_salvar
        if not nome:
            return
        pasta = self.pasta_perfis / id_perfil(nome)
        pasta.mkdir(parents=True, exist_ok=True)
        g, npz = self.n.salvar(self.ancoras, self.ignora, out=pasta)
        meta = dict(nome=nome, formato=1,
                    obediencia=float(self.n.par.obediencia),
                    tol=float(self.n.par.tol), dmax=int(self.n.par.dmax),
                    lam=float(self.n.par.lam),
                    fator_tempo_real=float(self.n.fator_tempo_real),
                    flares_detectados=int(np.count_nonzero(self.n.flare)),
                    flares_ignorados=int(np.count_nonzero(
                        self.n.flare & np.isin(np.arange(self.n.N),
                                               list(self.ignora["cima"])))),
                    origem_interfaces=str(Path(self.a.interfaces).resolve()))
        (pasta / "perfil.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Mantém os nomes canônicos: todo o restante do TG continua usando o
        # último perfil salvo/aberto sem precisar conhecer o catálogo.
        shutil.copy2(g, Path(self.a.out) / "guia.csv")
        shutil.copy2(npz, Path(self.a.out) / "interfaces_revisado.npz")
        self.nome_para_salvar = None
        self._atualizar_perfis(selecionar=nome)
        messagebox.showinfo(
            "perfil salvo",
            f'Perfil "{nome}" salvo.\n\n{len(self.ancoras)} ponto(s) e '
            f"{len(self.ignora['cima'])} quadro(s) ignorado(s).\n\n"
            f"Arquivos em:\n{pasta}\n\n"
            "para o jogo completo de resultados:\n\n"
            "python codigo/taxa_regressao.py \\\n"
            "  --interfaces resultados/interfaces_revisado.npz \\\n"
            "  --sufixo _revisado")

    def sair(self):
        if self._modo_sessao():
            if self.alterado:
                resp = messagebox.askyesnocancel(
                    "sair", "Há marcações não salvas.\n\nSalvar antes de sair?")
                if resp is None:
                    return
                if resp and not self._salvar_sessao():
                    return
            self.raiz.destroy()
            return
        if self.sujo and not messagebox.askyesno(
                "sair", "há marcações sem recalcular. sair mesmo?"):
            return
        self.raiz.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfaces", default=str(RAIZ / "resultados/interfaces.npz"))
    ap.add_argument("--x", type=int, nargs=2, default=[380, 1860],
                    help="faixa axial mostrada; inclui toda a medição x=400..1840")
    ap.add_argument("--escala", type=float, default=0.8)
    ap.add_argument("--passo", type=int, default=10,
                    help="quantas amostras cada avanço pula (1 amostra = 8 "
                         "quadros de câmera = 0,27 s)")
    ap.add_argument("--alt-kim", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--obediencia", type=float, default=1.0,
                    help="0 = suas marcas sao sugestao, 1 = sao a resposta")
    ap.add_argument("--dmax", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=str(RAIZ / "resultados"),
                    help="pasta da guia e da saída humana; não sobrescreve a entrada")
    ap.add_argument("--sessao", help="JSON da sessão de marcação (lista de quadros e faces)")
    ap.add_argument("--inicio-medida-s", type=float,
                    default=(E.AMOSTRA_INICIO_MEDIDA - E.AMOSTRA_IGNICAO) * E.CAM,
                    help="segundos após a ignição a partir dos quais vale a regra de não "
                         "recuar (antes: transiente, a borda pode subir). Padrão: a amostra "
                         "de início da medida da configuração")
    ap.add_argument("--grao-inteiro", action="store_true",
                    help="linha e recálculo no grão inteiro (todas as colunas dentro de --x), "
                         "não só nas colunas 18 a 90 do detector antigo")
    ap.add_argument("--sem-linhas", action="store_true",
                    help="abre com as linhas automáticas desligadas (marcação às cegas)")
    ap.add_argument("--graficos", action="store_true",
                    help="abre os gráficos ao lado; a imagem encolhe para caberem na tela")
    ap.add_argument("--carregar-a", metavar="PERFIL",
                    help="traz as marcações desse perfil para o grupo A, sem alterar o perfil")
    ap.add_argument("--novo-perfil", metavar="NOME",
                    help="abre direto num perfil novo e vazio com este nome (sessão)")
    ap.add_argument("--abrir-perfil",
                    help="nome exato de um perfil salvo para carregar ao iniciar")
    a = ap.parse_args()
    if a.sessao and Path(a.out).resolve() == (RAIZ / "resultados").resolve():
        # sessão de marcação: perfis junto das marcações humanas antigas
        a.out = str(RAIZ / "resultados/revisao_humana")
    a.out = Path(a.out).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    raiz = tk.Tk()
    # queda nativa (Tk/macOS) também deixa rastro em arquivo
    log_nativo = open(Path(a.out) / "erros_interface.log", "a", encoding="utf-8")
    faulthandler.enable(file=log_nativo, all_threads=True)
    app = App(raiz, a)
    raiz.report_callback_exception = app._erro_callback
    if a.graficos:
        raiz.after(300, app.alternar_graficos)
    if a.abrir_perfil:
        if a.abrir_perfil not in app.perfis:
            nomes = ", ".join(app.perfis) or "nenhum"
            raise SystemExit(f"perfil não encontrado: {a.abrir_perfil!r}; disponíveis: {nomes}")
        app.perfil.set(a.abrir_perfil)
        raiz.after(150, app.abrir_perfil)
        if a.sessao:
            raiz.after(450, app._oferecer_recuperacao_perfil)
    elif a.novo_perfil and a.novo_perfil in app.perfis:
        # O mesmo comando reaproveitado do histórico não pode abrir uma janela
        # VAZIA sobre um perfil que já tem marcações (foi o que confundiu o
        # usuário depois do crash de 21/10): abre o perfil para continuar.
        app.perfil.set(a.novo_perfil)
        raiz.after(150, app.abrir_perfil)
        raiz.after(400, lambda: app.lb2.config(
            text=f'o perfil "{a.novo_perfil}" já existia e foi aberto para continuar. '
                 "Para começar outro do zero, use o botão novo (⌘N)."))
    elif a.novo_perfil:
        raiz.after(150, lambda: app.novo_perfil(a.novo_perfil))
    if a.carregar_a:
        raiz.after(350, lambda: app.carregar_grupo_a(a.carregar_a))
    raiz.mainloop()


if __name__ == "__main__":
    main()
