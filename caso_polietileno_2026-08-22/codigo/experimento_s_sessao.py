"""S1 e S2: redes treinadas com a sessão de marcação do autor (28/10/2026).

Alvos: só os pontos do perfil `sessao-2026-10-21` (4.449 pontos, 43 quadros, duas
faces, grão inteiro), congelados em `experimento_s_sessao_2026-10-28/marcacoes/`.
Os perfis antigos ficam de fora: foram marcados antes de a convenção da superfície
(fuligem ignorada) ser fixada em 11/10.

Divisão: quadros dentro dos blocos B1 (k 300 a 360) e B2 (k 690 a 750) são TESTE;
quadros a até 15 amostras desses blocos não entram em nada; o resto treina.

Duas formas de começar, como no Capítulo 3 do TG:
  S1  codificador ResNet18 do ImageNet;
  S2  ajuste fino da R4 (que viu rótulos híbridos em quadros de B2: em B2 só S1 e
      o clássico são teste limpo).
Uma rede por face (a R4 tem uma saída só). Passo fixo, sem escolher checkpoint.

Decodificação: o mesmo Viterbi em três passadas do clássico, para a face pedida,
mas só a partir de k = 179 (45 s após a ignição). Antes disso a borda pode subir
(transiente da ignição) e a regra de não recuar não vale.

    python codigo/experimento_s_sessao.py rotulos
    python codigo/experimento_s_sessao.py treinar --variante S1 --face baixo
    python codigo/experimento_s_sessao.py avaliar
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import ensaio as E
import experimento_p2_v2 as P2

RAIZ = P2.RAIZ
OUT = RAIZ / "experimento_s_sessao_2026-10-28"
GUIA = E.MARCACOES
X0_ROI = E.ROI[2]
K_IGN = E.AMOSTRA_IGNICAO
K_INICIO = E.AMOSTRA_INICIO_MEDIDA
K_MIN = K_INICIO - 1
# Base de tempo: BASE_TEMPO=real usa o tempo marcado pela câmera vezes o fator medido
# por eventos de vazão (relogio_e_corte.py); BASE_TEMPO=marcador usa o tempo marcado pela
# câmera. Todos os scripts leem DT daqui.
BASE_TEMPO = os.environ.get("BASE_TEMPO", "marcador")
S_REAL = E.FATOR_TEMPO_REAL
DT = {"marcador": E.CAM, "real": E.CAM * S_REAL}[BASE_TEMPO]
K_FIM_CHAMA = E.AMOSTRA_FIM_CHAMA
T_INICIO = (K_INICIO - K_IGN) * DT
BLOCOS = dict(E.BLOCOS_TESTE)
MARGEM = 15
FACES = ("baixo", "cima")
XC = 20 * (np.arange(93) + 0.5)
MARGEM_LINHAS = 14


def marcas(face: str) -> dict[int, np.ndarray]:
    """Pontos do autor por quadro, em pixel da ROI, ordenados em x."""
    pts: dict[int, list] = {}
    for r in csv.DictReader(GUIA.open()):
        if r["tipo"] == "ancora" and r["face"] == face:
            pts.setdefault(int(r["k"]), []).append((float(r["x_px"]) - X0_ROI, float(r["y_roi_px"])))
    return {k: np.array(sorted(v)) for k, v in sorted(pts.items())}


def divisao() -> dict[str, list[int]]:
    ks = sorted(set(marcas("baixo")) | set(marcas("cima")))
    teste = {b: [k for k in ks if a <= k <= z] for b, (a, z) in BLOCOS.items()}
    proibido = {k for a, z in BLOCOS.values() for k in range(a - MARGEM, z + MARGEM + 1)}
    tst = {k for v in teste.values() for k in v}
    return dict(B1=teste["B1"], B2=teste["B2"],
                margem=[k for k in ks if k in proibido and k not in tst],
                treino=[k for k in ks if k not in proibido and k >= K_MIN])


def alvo_por_coluna(p: np.ndarray, raio: float = 10.0, vao: float = 30.0):
    """Altura nos centros das 93 colunas: mediana dos pontos a até `raio` px; onde
    não há ponto, interpolação só entre pontos a até `vao` px (não inventa trecho
    que o autor deixou sem marcar)."""
    y = np.zeros(93, np.float32)
    w = np.zeros(93, np.float32)
    for j, xc in enumerate(XC):
        perto = np.abs(p[:, 0] - xc) <= raio
        if perto.any():
            y[j], w[j] = np.median(p[perto, 1]), 1.0
            continue
        esq, dir_ = p[p[:, 0] < xc], p[p[:, 0] > xc]
        if len(esq) and len(dir_) and xc - esq[-1, 0] <= vao and dir_[0, 0] - xc <= vao:
            y[j], w[j] = np.interp(xc, [esq[-1, 0], dir_[0, 0]], [esq[-1, 1], dir_[0, 1]]), 1.0
    return y, w


def rotulos(_a=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = divisao()
    import hashlib
    resumo = dict(divisao=d, guia_sha256=hashlib.sha256(GUIA.read_bytes()).hexdigest())
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    img = OUT / "dataset/img"
    img.mkdir(parents=True, exist_ok=True)
    for face in FACES:
        m = marcas(face)
        alvo = np.zeros((963, 1, 93), np.float32)
        peso = np.zeros_like(alvo)
        for k, p in m.items():
            alvo[k, 0], peso[k, 0] = alvo_por_coluna(p)
        np.savez_compressed(OUT / f"alvos_{face}.npz", alvo=alvo, peso=peso)
        resumo[face] = dict(quadros=len(m), pontos=int(sum(len(v) for v in m.values())),
                            quadros_treino=[k for k in d["treino"] if k in m],
                            colunas_com_alvo_treino=int(sum(peso[k].sum() for k in d["treino"] if k in m)))
    for k in d["treino"] + d["B1"] + d["B2"]:
        p = img / f"{k:06d}.png"
        if not p.exists():
            cv2.imwrite(str(p), cubo[k])
    P2.save_json(OUT / "rotulos.json", resumo)
    print(json.dumps({k: (v if k == "divisao" else {kk: vv for kk, vv in v.items() if kk != "quadros_treino"})
                      for k, v in resumo.items() if k != "guia_sha256"}, indent=1, ensure_ascii=False))


def treinar(a) -> None:
    import torch
    from torch.utils.data import DataLoader
    from chamacera.surfdata import SurfDataset
    from chamacera.surfnet import surface_loss

    nome = f"{a.variante}_{a.face}_s{a.seed}"
    run = OUT / "runs" / nome
    run.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / f"alvos_{a.face}.npz")
    tr = [k for k in divisao()["treino"] if z["peso"][k].sum() > 0]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ds = SurfDataset(str(OUT / "dataset"), z["alvo"], z["peso"], [f"{k:06d}" for k in tr],
                     P2.TAMANHO["r18"], True, 3, a.seed, com_mascara=False)
    dl = DataLoader(ds, a.lote, shuffle=True, num_workers=0, drop_last=False)
    dev = P2.dispositivo()
    net = P2.construir("r18").to(dev)
    inicio = "ImageNet"
    if a.variante == "S2":
        inicio = str((P2.RUNS / "R4_s0/best.pt").relative_to(RAIZ))
        net.load_state_dict(torch.load(P2.RUNS / "R4_s0/best.pt", map_location=dev, weights_only=True))
    lr = a.lr or (1e-4 if a.variante == "S2" else 3e-4)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.passos, pct_start=0.1)
    print(f"[{nome}] {len(tr)} quadros de treino | início {inicio} | lr {lr} | {dev}", flush=True)
    passo, t0, hist = 0, time.time(), []
    net.train()
    while passo < a.passos:
        for x, y, w, _ in dl:
            loss = surface_loss(net(x.to(dev))[0], y.to(dev), w.to(dev), sigma=a.sigma)
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
    P2.save_json(run / "config.json", dict(nome=nome, variante=a.variante, face=a.face, seed=a.seed,
                                           quadros_treino=tr, inicio=inicio, passos=a.passos, lote=a.lote,
                                           lr=lr, sigma=a.sigma, modelo="r18", size=list(P2.TAMANHO["r18"]),
                                           selecao="último passo"))
    inferir(run, a.face)


def proibido(face: str, H: int, c0: float) -> np.ndarray:
    p = np.zeros(H, bool)
    c = int(round(c0))
    if face == "baixo":
        p[:c] = True; p[H - MARGEM_LINHAS:] = True
    else:
        p[c:] = True; p[:MARGEM_LINHAS] = True
    return p


def decodificar(S, yc0, flare, face, k0=K_MIN, dmax=2, lam=0.5, raio=40.0, peso=1.5,
                raio2=12.0, peso2=2.5, grau=4, jan_x=11) -> np.ndarray:
    """As três passadas do detector clássico para a face pedida, só a partir de k0."""
    from detectar_interfaces import (ajuste_robusto_x, colunas_confiaveis,
                                     corredor_no_escore, refinar, suavizar_x, viterbi)
    N, H, nb = S.shape
    mono = +1 if face == "baixo" else -1
    Ss, fl = S[k0:], flare[k0:]
    pq = np.where(fl, 0.2, 1.0)[:, None]
    Sc = {j: Ss[:, :, j] * pq for j in range(nb)}
    pr = {j: proibido(face, H, yc0[j]) for j in range(nb)}
    Y = np.zeros((len(Ss), nb))
    for j in range(nb):
        Y[:, j] = viterbi(Sc[j], dmax, lam, pr[j], mono)
    for etapa, (ref_f, r_, p_) in enumerate((
            (lambda Y: ajuste_robusto_x(Y, grau, usar=colunas_confiaveis(Y)), raio, peso),
            (lambda Y: suavizar_x(Y, jan_x), raio2, peso2))):
        ref = ref_f(Y)
        for j in range(nb):
            S2 = corredor_no_escore(Sc[j], ref[:, j], r_, p_)
            y = viterbi(S2, dmax, lam, pr[j], mono)
            Y[:, j] = refinar(S2, y, mono=mono) if etapa else y
    out = np.full((N, nb), np.nan)
    out[k0:] = Y
    return out


def inferir(run: Path, face: str) -> None:
    import torch
    import torch.nn.functional as F
    from chamacera.surfnet import soft_argmax_col

    dev = P2.dispositivo()
    net = P2.construir("r18").to(dev)
    net.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    net.eval()
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    N, H, nb = cubo.shape[:3]
    oh, ow = P2.TAMANHO["r18"]
    cl = np.load(P2.E1 / "classico.npz")
    permitido = np.stack([~proibido(face, H, c) for c in cl["yc0"]], -1)
    bruto = np.zeros((N, nb), np.float32)
    S = np.zeros((N, H, nb), np.float32)
    xs = np.linspace(0, nb - 1, ow)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, 16):
            ks = range(i, min(N, i + 16))
            x = np.stack([cv2.resize(cubo[k], (ow, oh)).astype(np.float32) / 255 for k in ks])
            lg = net(torch.from_numpy(x).permute(0, 3, 1, 2).to(dev))[0]
            y, _ = soft_argmax_col(lg)
            p = F.interpolate(F.softmax(lg, dim=2), size=(H, nb), mode="bilinear",
                              align_corners=True)[:, 0].cpu().numpy() * permitido
            S[i:i + len(ks)] = 1.5 * p / np.maximum(p.max(axis=1, keepdims=True), 1e-12)
            for r, k in enumerate(ks):
                bruto[k] = np.interp(np.arange(nb), xs, y[r, 0].cpu().numpy() * H / oh)
    t_inf = time.time() - t0
    t0 = time.time()
    vit = decodificar(S, cl["yc0"], cl["flare"], face)
    np.savez_compressed(run / "saida.npz", bruto=bruto, viterbi=vit)
    P2.save_json(run / "tempo.json", dict(inferencia_s=t_inf, decodificacao_s=time.time() - t0))
    print(f"[{run.name}] inferência {t_inf:.1f} s, decodificação {time.time()-t0:.1f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["rotulos", "treinar"])
    ap.add_argument("--variante", choices=["S1", "S2"], default="S1")
    ap.add_argument("--face", choices=FACES, default="baixo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passos", type=int, default=1200)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=2.0)
    a = ap.parse_args()
    globals()[a.etapa](a)
