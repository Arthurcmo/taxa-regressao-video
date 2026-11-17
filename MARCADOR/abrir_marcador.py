"""Janela de entrada do marcador.

Escolhe a sessão (a lista de quadros a marcar) e o perfil (onde as marcações
ficam salvas) e abre o programa de marcação com as opções de sempre. Aberto
pelo arquivo "Abrir marcador.command", ao lado deste.

    python abrir_marcador.py                  # janela
    python abrir_marcador.py --mostrar-comando SESSAO PERFIL [novo]
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CASO = Path(__file__).resolve().parent.parent / "caso_polietileno_2026-08-22"
PERFIS = CASO / "resultados/revisao_humana/perfis"
PYTHON = next((p for p in (CASO / ".venv/bin/python", CASO.parent / ".venv/bin/python") if p.exists()),
              Path(sys.executable))
SEM_SESSAO = "nenhuma (navegar pelo vídeo inteiro)"
NOVO = "(novo perfil vazio)"
NENHUM = "nenhuma"


def sessoes() -> dict[str, str]:
    """Pastas do caso com um quadros.json: nome mostrado -> caminho relativo ao caso."""
    out = {}
    for q in sorted(CASO.glob("sessao_*/quadros.json")):
        out[q.parent.name] = str(q.relative_to(CASO))
    return out


def perfis() -> dict[str, dict]:
    """Perfis salvos: nome -> metadados (perfil.json)."""
    out = {}
    for meta in sorted(PERFIS.glob("*/perfil.json")):
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        out[m.get("nome", meta.parent.name)] = m
    return out


def comando(sessao: str | None, perfil: str, novo: bool, linhas: bool, graficos: bool = True,
            carregar: str | None = None) -> list[str]:
    cmd = [str(PYTHON), "codigo/interface.py"]
    if sessao:
        cmd += ["--sessao", sessao]
    if not linhas:
        cmd += ["--sem-linhas"]
    cmd += ["--grao-inteiro", "--x", "40", "1890", "--escala", "0.75"]
    if graficos:
        cmd += ["--graficos"]
    if perfil:
        cmd += ["--novo-perfil" if novo else "--abrir-perfil", perfil]
    if carregar:
        cmd += ["--carregar-a", carregar]
    return cmd


class Janela:
    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        raiz.title("Abrir marcador — polietileno")
        self.ses = sessoes()
        self.per = perfis()
        f = ttk.Frame(raiz, padding=14)
        f.grid(sticky="nsew")
        ttk.Label(f, text="1. Sessão (lista de quadros a marcar)", font=("Helvetica", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.v_ses = tk.StringVar(value=next(iter(self.ses), SEM_SESSAO))
        self.cb_ses = ttk.Combobox(f, textvariable=self.v_ses, values=list(self.ses) + [SEM_SESSAO], width=48, state="readonly")
        self.cb_ses.grid(row=1, column=0, sticky="we", pady=(2, 0))
        self.cb_ses.bind("<<ComboboxSelected>>", lambda e: self._mostrar_sessao())
        ttk.Button(f, text="outra pasta…", command=self._escolher_pasta).grid(row=1, column=1, padx=(6, 0))
        self.l_ses = ttk.Label(f, text="", wraplength=520, foreground="#555555", justify="left")
        self.l_ses.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 10))

        ttk.Label(f, text="2. Perfil (onde as marcações ficam salvas)", font=("Helvetica", 13, "bold")).grid(row=3, column=0, sticky="w")
        self.v_per = tk.StringVar(value=NOVO)
        self.cb_per = ttk.Combobox(f, textvariable=self.v_per, values=[NOVO] + list(self.per), width=48, state="readonly")
        self.cb_per.grid(row=4, column=0, sticky="we", pady=(2, 0))
        self.cb_per.bind("<<ComboboxSelected>>", lambda e: self._mostrar_perfil())
        ttk.Label(f, text="nome do perfil novo:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.v_nome = tk.StringVar(value="sessao-" + datetime.date.today().isoformat())
        self.e_nome = ttk.Entry(f, textvariable=self.v_nome, width=50)
        self.e_nome.grid(row=6, column=0, sticky="we")
        self.l_per = ttk.Label(f, text="", wraplength=520, foreground="#555555", justify="left")
        self.l_per.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 10))

        ttk.Label(f, text="3. Marcações já feitas para carregar no grupo A", font=("Helvetica", 13, "bold")).grid(row=8, column=0, sticky="w")
        self.v_car = tk.StringVar(value=NENHUM)
        self.cb_car = ttk.Combobox(f, textvariable=self.v_car, values=[NENHUM] + list(self.per), width=48, state="readonly")
        self.cb_car.grid(row=9, column=0, sticky="we", pady=(2, 0))
        ttk.Label(f, text="Só com perfil novo. As marcações entram no grupo A do perfil novo; o perfil de origem não muda.",
                  wraplength=520, foreground="#555555", justify="left").grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 10))

        self.v_linhas = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="mostrar as linhas automáticas (deixe desligado para marcar às cegas)",
                        variable=self.v_linhas).grid(row=11, column=0, columnspan=2, sticky="w")
        self.v_graficos = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="gráficos ao lado (a imagem fica menor para caber na tela)",
                        variable=self.v_graficos).grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="ABRIR MARCADOR", command=self._abrir).grid(row=13, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Label(f, text="O tempo mostrado é o tempo marcado pela câmera.",
                  foreground="#555555").grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._mostrar_sessao()
        self._mostrar_perfil()

    def _sessao_atual(self) -> str | None:
        v = self.v_ses.get()
        return None if v == SEM_SESSAO else self.ses.get(v, v)

    def _mostrar_sessao(self):
        s = self._sessao_atual()
        if not s:
            self.l_ses.configure(text="Sem lista de pedidos: você anda pelo vídeo livremente.")
            return
        try:
            d = json.loads((CASO / s).read_text())
        except (OSError, ValueError) as e:
            self.l_ses.configure(text=f"não consegui ler {s}: {e}")
            return
        n = len(d.get("quadros", []))
        self.l_ses.configure(text=f"{n} pedidos. {d.get('objetivo', '')}\n{d.get('regra', '')}".strip())

    def _escolher_pasta(self):
        p = filedialog.askdirectory(title="Pasta da sessão (com quadros.json)", initialdir=str(CASO))
        if not p:
            return
        q = Path(p) / "quadros.json"
        if not q.exists():
            messagebox.showerror("Sessão", "Essa pasta não tem um quadros.json.")
            return
        try:
            rel = str(q.resolve().relative_to(CASO))
        except ValueError:
            rel = str(q.resolve())
        self.ses[Path(p).name] = rel
        self.cb_ses.configure(values=list(self.ses) + [SEM_SESSAO])
        self.v_ses.set(Path(p).name)
        self._mostrar_sessao()

    def _mostrar_perfil(self):
        nome = self.v_per.get()
        self.cb_car.configure(state="readonly" if nome == NOVO else "disabled")
        if nome != NOVO:
            self.v_car.set(NENHUM)
        if nome == NOVO:
            self.e_nome.configure(state="normal")
            self.l_per.configure(text="Começa vazio. Recomendado para remarcar sem ver as marcações antigas.")
            return
        self.e_nome.configure(state="disabled")
        m = self.per[nome]
        s = m.get("sessao")
        txt = f"{m.get('pontos', '?')} pontos, salvo em {m.get('salvo_em', '?')}."
        if s:
            # um perfil de sessão só abre com a mesma sessão em que foi salvo
            achado = next((k for k, v in self.ses.items() if v == s), None)
            if achado:
                self.v_ses.set(achado)
                self._mostrar_sessao()
            txt += f" Sessão dele: {s}."
        if m.get("nota"):
            txt += f" {m['nota']}"
        if nome.startswith("sessao-2026-10-21"):
            txt += "  ATENÇÃO: este é o perfil original do treino; um clique novo empurra pontos antigos."
        self.l_per.configure(text=txt)

    def _abrir(self):
        nome = self.v_per.get()
        novo = nome == NOVO
        if novo:
            nome = self.v_nome.get().strip()
            if not nome:
                messagebox.showerror("Perfil", "Dê um nome ao perfil novo.")
                return
            if nome in self.per:
                messagebox.showerror("Perfil", f"Já existe um perfil chamado {nome!r}. Escolha outro nome ou abra esse na lista.")
                return
        carregar = None if self.v_car.get() == NENHUM else self.v_car.get()
        cmd = comando(self._sessao_atual(), nome, novo, self.v_linhas.get(), self.v_graficos.get(), carregar)
        print("abrindo:", " ".join(cmd), flush=True)
        subprocess.Popen(cmd, cwd=str(CASO))
        self.raiz.after(500, self.raiz.destroy)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--mostrar-comando":
        sessao, perfil = sys.argv[2], sys.argv[3]
        novo = len(sys.argv) > 4 and sys.argv[4] == "novo"
        print(" ".join(comando(sessoes().get(sessao, sessao), perfil, novo, False)))
        return
    if not PYTHON.exists():
        print(f"não achei o Python do caso em {PYTHON}")
        sys.exit(1)
    raiz = tk.Tk()
    Janela(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
