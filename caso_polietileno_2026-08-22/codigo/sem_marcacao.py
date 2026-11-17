"""Linha sem nenhuma marcação humana, para servir de base de comparação (08/11/2026).

Pedido do autor: uma rede que aprenda só com a visão (o detector clássico) e com a
fórmula (a recompensa de forma), sem ver nenhum ponto marcado por ele, para comparar com
as linhas que aprenderam das marcações.

Etapa 1, SM1 (o par da S1): codificador ResNet18 da ImageNet, mesma rede, perda,
         otimizador, aumento de dados e 1200 passos da S1. Rótulos: a posição do detector
         clássico em cada coluna, com peso pela confiança da própria imagem (escore do
         vale): w = min(1, c/0,6), e w = 0 se c < 0,3. Quadros: uma amostra a cada três
         de 79 a 767, fora dos blocos de teste B1 e B2 e da margem de 15 amostras (os
         mesmos quadros que a S1 também não viu), e fora dos quadros estourados.
Etapa 2, SMF (o par da forma): parte da SM1 e roda recompensa_oculta.py com a
         recompensa de forma (peso 5, seção 40 a 80 mm), a faixa de 4 px em torno do
         clássico (no lugar das marcações), a âncora de 6 px em torno da SM1 (no lugar da
         2b), coerência no tempo e recuo, 300 passos.

Nenhuma etapa lê marcações; elas só entram depois, na avaliação, como teste
independente em todos os quadros marcados.

    PYTHONPATH=codigo .venv/bin/python -W ignore codigo/sem_marcacao.py rotulos
    PYTHONPATH=codigo .venv/bin/python -W ignore codigo/sem_marcacao.py treinar --seed 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import cv2
import numpy as np

import experimento_p2_v2 as P2
import recompensa_lei as R
from experimento_s_sessao import BLOCOS, MARGEM

DS = R.OUT / "sem_marcacao"
FACES = ("baixo", "cima")
CHAVE = {"baixo": ("y_inf", "c_inf"), "cima": ("y_sup", "c_sup")}
C_MIN, C_CHEIO = 0.3, 0.6


def classico():
    return np.load(P2.E1 / "classico.npz")


def rotulos_classico(face: str) -> tuple[np.ndarray, np.ndarray]:
    cl = classico()
    ky, kc = CHAVE[face]
    y = cl[ky].astype(np.float32)
    c = cl[kc].astype(np.float32)
    w = np.clip(c / C_CHEIO, 0, 1)
    w[c < C_MIN] = 0.0
    return y, w.astype(np.float32)


def quadros() -> list[int]:
    cl = classico()
    proib = {k for a, z in BLOCOS.values() for k in range(a - MARGEM, z + MARGEM + 1)}
    return [k for k in range(78, 767, 3) if k not in proib and not cl["flare"][k]]


def rotulos(_a=None) -> None:
    img = DS / "dataset/img"
    img.mkdir(parents=True, exist_ok=True)
    cubo = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    ks = quadros()
    for k in ks:
        p = img / f"{k:06d}.png"
        if not p.exists():
            cv2.imwrite(str(p), np.ascontiguousarray(cubo[k]))
    res = dict(quadros=ks, c_min=C_MIN, c_cheio=C_CHEIO)
    for f in FACES:
        y, w = rotulos_classico(f)
        np.savez_compressed(DS / f"alvos_{f}.npz", alvo=y[:, None], peso=w[:, None])
        res[f] = dict(colunas_com_peso=int((w[ks] > 0).sum()), peso_medio=float(w[ks].mean()))
    (DS / "rotulos.json").write_text(json.dumps(res, indent=1))
    print(f"{len(ks)} quadros; " + "; ".join(f"{f}: {res[f]['colunas_com_peso']} colunas com peso, "
                                            f"peso médio {res[f]['peso_medio']:.2f}" for f in FACES), flush=True)


def treinar_face(face: str, seed: int, run, passos=1200, lote=8, lr=3e-4, sigma=2.0) -> None:
    """Igual a experimento_s_sessao.treinar (S1), trocando os rótulos."""
    import torch
    from torch.utils.data import DataLoader
    from chamacera.surfdata import SurfDataset
    from chamacera.surfnet import surface_loss

    z = np.load(DS / f"alvos_{face}.npz")
    tr = [k for k in quadros() if z["peso"][k].sum() > 0]
    torch.manual_seed(seed); np.random.seed(seed)
    ds = SurfDataset(str(DS / "dataset"), z["alvo"], z["peso"], [f"{k:06d}" for k in tr],
                     P2.TAMANHO["r18"], True, 3, seed, com_mascara=False)
    dl = DataLoader(ds, lote, shuffle=True, num_workers=0, drop_last=False)
    dev = P2.dispositivo()
    net = P2.construir("r18").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.1)
    print(f"[{run.name} {face}] {len(tr)} quadros | ImageNet | {dev}", flush=True)
    passo, t0, hist = 0, time.time(), []
    net.train()
    while passo < passos:
        for x, y, w, _ in dl:
            loss = surface_loss(net(x.to(dev))[0], y.to(dev), w.to(dev), sigma=sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); passo += 1
            if passo % 100 == 0:
                hist.append(dict(passo=passo, perda=float(loss.detach()), s=time.time() - t0))
            if passo % 400 == 0:
                print(f"  passo {passo:5d} perda {float(loss.detach()):7.3f} ({time.time() - t0:.0f}s)", flush=True)
            if passo >= passos:
                break
    torch.save(net.state_dict(), run / f"{face}.pt")
    (run / f"hist_{face}.json").write_text(json.dumps(hist))


def treinar(a) -> None:
    run = R.OUT / "runs" / f"sm1_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "saida.npz").exists():
        for f in FACES:
            treinar_face(f, a.seed, run)
        (run / "config.json").write_text(json.dumps(dict(nome=run.name, inicio="ImageNet", rotulos="detector clássico",
                                                         c_min=C_MIN, c_cheio=C_CHEIO, quadros=len(quadros()),
                                                         passos=1200, lote=8, lr=3e-4, sigma=2.0, seed=a.seed,
                                                         marcas=False), indent=1))
        R.inferir(run)
    cmd = [sys.executable, "-W", "ignore", str(R.RAIZ / "codigo/recompensa_oculta.py"), f"smf_s{a.seed}",
           "--recompensa", "forma", "--w", "5", "--inicio", f"run:sm1_s{a.seed}", "--ref", f"sm1_s{a.seed}",
           "--rotulos", "classico", "--seed", str(a.seed)]
    if not (R.OUT / f"runs/smf_s{a.seed}/saida.npz").exists():
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["rotulos", "treinar"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"rotulos": rotulos, "treinar": treinar}[a.etapa](a)
