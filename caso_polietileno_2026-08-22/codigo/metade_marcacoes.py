"""Rede treinada com só parte dos quadros marcados (16/11/2026).

Pergunta do autor: se ele marcar poucos quadros, a rede marca o resto? Mesma receita
da S1 (ResNet18 da ImageNet, 1200 passos, uma rede por face, mesmas marcações da
sessão de 21/10), mas cada rede vê só uma fração dos quadros marcados e é medida nos
quadros que NÃO viu, espalhados pela queima inteira.

Grupos (quadros marcados da face com k >= 179, em ordem de tempo):
  A   metade: quadros de índice par (0, 2, 4, ...); teste nos de índice ímpar;
  B   a outra metade: índice ímpar; teste nos pares (dobra a evidência);
  Q   um quarto: índices 0, 4, 8, ...; teste nos outros três quartos.

    # treino (Python com segmentation_models_pytorch):
    PYTHONPATH=codigo $PY codigo/metade_marcacoes.py treinar --grupo A --face baixo --seed 0
    # avaliação (Python do caso, com BASE_TEMPO=real):
    BASE_TEMPO=real PYTHONPATH=codigo .venv/bin/python codigo/metade_marcacoes.py avaliar
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import experimento_s_sessao as E

OUTM = E.RAIZ.parent / "experimento_metade_marcacoes_2026-11-16"
GRUPOS = ("A", "B", "Q")
SEMENTES = (0, 1, 2)


def quadros(face: str) -> list[int]:
    return [k for k in sorted(E.marcas(face)) if k >= E.K_MIN]


def divisao(grupo: str, face: str) -> tuple[list[int], list[int]]:
    ks = quadros(face)
    tr = {"A": ks[0::2], "B": ks[1::2], "Q": ks[0::4]}[grupo]
    return tr, [k for k in ks if k not in tr]


def alvos(face: str):
    alvo = np.zeros((963, 1, 93), np.float32)
    peso = np.zeros_like(alvo)
    for k, p in E.marcas(face).items():
        alvo[k, 0], peso[k, 0] = E.alvo_por_coluna(p)
    return alvo, peso


def preparar() -> None:
    import cv2
    img = OUTM / "dataset/img"
    img.mkdir(parents=True, exist_ok=True)
    cubo = np.load(E.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    for face in E.FACES:
        for k in quadros(face):
            p = img / f"{k:06d}.png"
            if not p.exists():
                cv2.imwrite(str(p), cubo[k])
    div = {g: {f: dict(zip(("treino", "teste"), divisao(g, f))) for f in E.FACES} for g in GRUPOS}
    (OUTM / "divisao.json").write_text(json.dumps(div, indent=1))


def treinar(a) -> None:
    import torch
    from torch.utils.data import DataLoader
    from chamacera.surfdata import SurfDataset
    from chamacera.surfnet import surface_loss
    import experimento_p2_v2 as P2

    preparar()
    nome = f"{a.grupo}_{a.face}_s{a.seed}"
    run = OUTM / "runs" / nome
    if (run / "saida.npz").exists():
        print(f"[{nome}] já existe, pulando", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    alvo, peso = alvos(a.face)
    tr, te = divisao(a.grupo, a.face)
    tr = [k for k in tr if peso[k].sum() > 0]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ds = SurfDataset(str(OUTM / "dataset"), alvo, peso, [f"{k:06d}" for k in tr],
                     P2.TAMANHO["r18"], True, 3, a.seed, com_mascara=False)
    dl = DataLoader(ds, 8, shuffle=True, num_workers=0, drop_last=False)
    dev = P2.dispositivo()
    net = P2.construir("r18").to(dev)
    lr = 3e-4
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.passos, pct_start=0.1)
    print(f"[{nome}] {len(tr)} quadros de treino, {len(te)} de teste | ImageNet | lr {lr} | {dev}", flush=True)
    passo, t0, hist = 0, time.time(), []
    net.train()
    while passo < a.passos:
        for x, y, w, _ in dl:
            loss = surface_loss(net(x.to(dev))[0], y.to(dev), w.to(dev), sigma=2.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); passo += 1
            if passo % 100 == 0:
                hist.append(dict(passo=passo, perda=float(loss.detach()), s=time.time() - t0))
            if passo % 400 == 0:
                print(f"  passo {passo:5d} perda {float(loss.detach()):7.3f} ({time.time()-t0:.0f}s)", flush=True)
            if passo >= a.passos:
                break
    torch.save(net.state_dict(), run / "best.pt")
    P2.save_json(run / "hist.json", hist)
    P2.save_json(run / "config.json", dict(nome=nome, grupo=a.grupo, face=a.face, seed=a.seed,
                                           quadros_treino=tr, quadros_teste=te, inicio="ImageNet",
                                           passos=a.passos, lote=8, lr=lr, sigma=2.0, modelo="r18",
                                           selecao="último passo"))
    E.inferir(run, a.face)


# ----------------------------------------------------------------------------- avaliação
SENT = {"baixo": -1, "cima": +1}          # positivo = para o lado da chama


def marcas_col(face: str) -> np.ndarray:
    A = np.full((963, 93), np.nan)
    for k, p in E.marcas(face).items():
        y, w = E.alvo_por_coluna(p)
        A[k, w > 0] = y[w > 0]
    return A


def erro(Y, A, ks, face, cols):
    e = np.concatenate([(Y[k, cols] - A[k, cols])[np.isfinite(A[k, cols]) & np.isfinite(Y[k, cols])] for k in ks])
    return dict(media=float(np.abs(e).mean()), mediana=float(np.median(np.abs(e))),
                vies=float(np.mean(SENT[face] * e)), ate4=float(np.mean(np.abs(e) <= 4)),
                p95=float(np.percentile(np.abs(e), 95)), n=int(e.size))


def avaliar(_a=None) -> None:
    import faixas_grao as FG
    import recompensa_lei as R
    import experimento_p2_v2 as P2
    import taxa_real_cap4 as TR
    import avaliar_linhas as AL

    div = json.loads((OUTM / "divisao.json").read_text())
    A = {f: marcas_col(f) for f in E.FACES}
    regioes = {"grão todo": np.arange(93), "40 a 120 mm": FG.faixa_cols(40, 120)}
    cl = np.load(P2.E1 / "classico.npz")
    ref = {"clássico (sem treino)": {0: dict(baixo=cl["y_inf"].astype(float), cima=cl["y_sup"].astype(float))},
           "SM1 v4 (nenhuma marcação)": {s: dict(zip(E.FACES, AL.run(f"sm1v4_s{s}"))) for s in SEMENTES},
           "2b (viu todos os quadros)": {s: dict(zip(E.FACES, AL.run("nenhum_d4" + ("" if s == 0 else f"_s{s}"))))
                                         for s in SEMENTES}}
    res = {"divisao": div, "grupos": {}, "referencias": {}}
    for g in GRUPOS:
        res["grupos"][g] = {}
        for s in SEMENTES:
            r = {}
            for f in E.FACES:
                z = np.load(OUTM / f"runs/{g}_{f}_s{s}/saida.npz")
                te, tr = div[g][f]["teste"], div[g][f]["treino"]
                for saida in ("bruto", "viterbi"):
                    for rg, cols in regioes.items():
                        r[f"{f}|{saida}|{rg}|teste"] = erro(z[saida].astype(float), A[f], te, f, cols)
                        r[f"{f}|{saida}|{rg}|treino"] = erro(z[saida].astype(float), A[f], tr, f, cols)
            Yb = np.load(OUTM / f"runs/{g}_baixo_s{s}/saida.npz")["bruto"].astype(float)
            Yc = np.load(OUTM / f"runs/{g}_cima_s{s}/saida.npz")["bruto"].astype(float)
            m = TR.medir(Yb, Yc, ns=(10,))
            r["taxa"] = dict(analise=m["analise"], ensaio=m["ensaio"], intervalos=m["janelas"]["10"])
            res["grupos"][g][str(s)] = r
            print(g, s, "ok", flush=True)
    # referências: nos mesmos quadros de teste de cada grupo
    for nome, sem in ref.items():
        res["referencias"][nome] = {}
        for g in GRUPOS:
            for s, Y in sem.items():
                for f in E.FACES:
                    for rg, cols in regioes.items():
                        res["referencias"][nome][f"{g}|{s}|{f}|{rg}"] = erro(Y[f], A[f], div[g][f]["teste"], f, cols)
    man = TR.medir(*AL.manual(), ns=(10,))
    res["marcacoes_taxa"] = dict(analise=man["analise"], intervalos=man["janelas"]["10"])
    (OUTM / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("gravado", OUTM / "resultado.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["preparar", "treinar", "avaliar"])
    ap.add_argument("--grupo", choices=GRUPOS, default="A")
    ap.add_argument("--face", choices=E.FACES, default="baixo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passos", type=int, default=1200)
    a = ap.parse_args()
    {"preparar": lambda a: preparar(), "treinar": treinar, "avaliar": avaliar}[a.etapa](a)
