"""Retreino de todas as famílias com penalidades físicas (08/11/2026).

Pedido do autor: punir a linha que sai do grão (para dentro da porta ou para fora da
área do grão) e melhorar o início, retreinando tudo nas três sementes.

Penalidades novas, sem marcação (relu ao quadrado, em pixels, peso 0,05 como as outras):
  parede   a face não pode entrar mais de 10 px na porta original. A porta original é a
           reta ajustada às bordas do quadro médio antes da ignição (amostras 1 a 40);
           10 px de folga cobrem o inchaço logo depois da ignição (as marcações de 94
           ficam 4,7 px acima da parede);
  moldura  a face não pode chegar a menos de 14 px da borda da imagem;
  recuo    entre k e k+15, a face de baixo não sobe e a de cima não desce mais de 3 px
           (o mesmo limiar do critério de aceitação).
Nas etapas que partem da ImageNet (S1v2, SM1v2) as penalidades usam 4 pares de quadros
sorteados por passo, sem aumento de dados, fora dos blocos de teste. Nas etapas de ajuste
fino (2bv2, formav2, SMFv2), recompensa_oculta.py ganhou --lam-parede.

Rótulos novos para o início (iguais nas duas famílias): 11 quadros antes da ignição
(amostras 1 a 41, uma a cada 4) com a porta original como rótulo. A S1v2 usa também as
marcações da camada B (amostras 79, 94 e 226), que a S1 não usava. A coluna 0 (x de 0 a
20 px, antes do início do grão) fica fora de todas as perdas.

Famílias, por semente s (todas com as penalidades):
  s1v2_s    ImageNet + marcações (sessão e camada B) + porta original      (par da S1)
  2bv2_s    s1v2_s + faixa de 4 px nas marcações, 400 passos               (par da 2b)
  formav2_s 2bv2_s + recompensa de forma, âncora na 2bv2, 300 passos      (par da forma)
  sm1v2_s   ImageNet + detector clássico + porta original                  (par da SM1)
  smfv2_s   sm1v2_s + recompensa de forma, faixa no clássico, 300 passos   (par da SMF)

    PYTHONPATH=codigo <python com smp> -W ignore codigo/treino_v2.py paredes
    PYTHONPATH=codigo <python com smp> -W ignore codigo/treino_v2.py marcas --seed 0
    PYTHONPATH=codigo <python com smp> -W ignore codigo/treino_v2.py classico --seed 0
"""
from __future__ import annotations

import ensaio as E
import argparse
import json
import subprocess
import sys
import time

import cv2
import numpy as np

import experimento_p2_v2 as P2
import recompensa_lei as R
import sem_marcacao as SM
from experimento_s_sessao import BLOCOS, MARGEM

OUT = R.OUT / "treino_v2"
PAREDES = R.OUT / "paredes_iniciais.npz"
FACES = ("baixo", "cima")
FOLGA_PAREDE, MOLDURA, PASSO_RECUO, TOL_RECUO = 10.0, 14.0, 15, 3.0
LAM_FIS = 0.05
PRE = list(range(0, 41, 4))
PROIB = {k for a, z in BLOCOS.values() for k in range(a - MARGEM, z + MARGEM + 1)}
COLS = np.arange(1, 93)


def paredes(_a=None) -> dict:
    """Retas da porta original (y da face de cima e da de baixo em cada coluna)."""
    if PAREDES.exists():
        z = np.load(PAREDES)
        return {"cima": z["sup"], "baixo": z["inf"]}
    from scipy.ndimage import gaussian_filter1d
    cubo = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    yc = np.load(P2.E1 / "classico.npz")["yc0"]
    G = np.stack([np.asarray(cubo[k], np.float32).mean(2) for k in range(0, 40)]).mean(0)
    I = gaussian_filter1d(G, 1.5, axis=0)
    d = np.zeros_like(I); d[3:-3] = I[6:] - I[:-6]
    sup, inf = np.zeros(93), np.zeros(93)
    for j in range(93):
        c = int(round(yc[j]))
        a, b = np.arange(c - 100, c), np.arange(c, c + 100)
        sup[j], inf[j] = a[np.argmin(d[a, j])], b[np.argmax(d[b, j])]

    def reta(y):
        cols = np.arange(2, 93); ok = np.ones(len(cols), bool)
        for _ in range(5):
            p = np.polyfit(cols[ok], y[cols][ok], 1)
            r = y[cols] - np.polyval(p, cols)
            s = 1.4826 * np.median(np.abs(r[ok])) + 1e-6
            ok = np.abs(r) < 3 * max(s, 1.5)
        return np.polyval(p, np.arange(93))
    W = {"cima": reta(sup), "baixo": reta(inf)}
    np.savez(PAREDES, sup=W["cima"], inf=W["baixo"], sup_bruto=sup, inf_bruto=inf)
    return W


def rotulos(familia: str) -> dict:
    """alvo e peso (963, 1, 93) por face e a lista de quadros de treino."""
    W = paredes()
    out, ks_all = {}, set()
    for f in FACES:
        alvo = np.zeros((963, 1, 93), np.float32)
        peso = np.zeros_like(alvo)
        if familia == "marcas":
            A = R.autor(f)
            for k in range(963):
                ok = np.isfinite(A[k])
                if k in PROIB or ok.sum() <= 10:
                    continue
                alvo[k, 0, ok] = A[k, ok]; peso[k, 0, ok] = 1.0
        elif familia in ("v3", "v4"):
            z = np.load(R.OUT / f"sem_marcacao_{familia}" / f"alvos_{f}.npz")
            for k in SM.quadros():
                alvo[k, 0], peso[k, 0] = z["y"][k], z["peso"][k]
        else:
            y, w = SM.rotulos_classico(f)
            for k in SM.quadros():
                alvo[k, 0], peso[k, 0] = y[k], w[k]
        for k in PRE:
            alvo[k, 0], peso[k, 0] = W[f], 1.0
        peso[:, 0, 0] = 0.0                               # coluna 0: antes do início do grão
        out[f] = (alvo, peso)
        ks_all |= {k for k in range(963) if peso[k].sum() > 0}
    return dict(rotulos=out, quadros=sorted(ks_all))


def preparar_imagens(ks) -> None:
    img = OUT / "dataset/img"
    img.mkdir(parents=True, exist_ok=True)
    cubo = np.load(R.RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    for k in ks:
        p = img / f"{k:06d}.png"
        if not p.exists():
            cv2.imwrite(str(p), np.ascontiguousarray(cubo[k]))


def treinar_face(face: str, seed: int, run, alvo, peso, passos=1200, lote=8, lr=3e-4, sigma=2.0,
                 fuligem: bool = False) -> None:
    """Receita da S1 (experimento_s_sessao.treinar) + penalidades físicas em quadros sem rótulo."""
    import torch
    from torch.utils.data import DataLoader
    from chamacera.surfdata import SurfDataset
    from chamacera.surfnet import surface_loss

    tr = [k for k in range(963) if peso[k].sum() > 0]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed + 1000)
    ds = SurfDataset(str(OUT / "dataset"), alvo, peso, [f"{k:06d}" for k in tr], P2.TAMANHO["r18"], True, 3, seed,
                     com_mascara=False)
    dl = DataLoader(ds, lote, shuffle=True, num_workers=0, drop_last=False)
    cubo, dev, entrada, altura, _ = R._ferramentas()
    net = P2.construir("r18").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.1)
    W = torch.tensor(paredes()[face], dtype=torch.float32, device=dev)
    if fuligem:                                          # v4: fuligem parada e linha parada pagam
        from scipy.ndimage import gaussian_filter1d as _g1
        import faixas_grao as FG
        PAR = _g1(np.load(R.OUT / "fuligem_estatica" / "parado.npy").astype(np.float32) / 255.0, 4.0, axis=1)
        m40 = torch.tensor(FG.X_MM[COLS] >= 40, device=dev)
        K_CORTE, K_FIM, MOV_MIN = int(round(E.AMOSTRA_IGNICAO + E.JANELA_ANALISE_S[0] / (E.CAM * E.FATOR_TEMPO_REAL))), int(round(E.AMOSTRA_IGNICAO + E.JANELA_ANALISE_S[1] / (E.CAM * E.FATOR_TEMPO_REAL))), 0.6   # janela de análise; px em PASSO_RECUO amostras

        def no_mapa(Pk, yy):
            i0 = yy.detach().floor().clamp(0, Pk.shape[1] - 2).long()
            w_ = (yy - i0).clamp(0, 1)
            return (1 - w_) * Pk.gather(1, i0[:, None, :])[:, 0] + w_ * Pk.gather(1, (i0 + 1)[:, None, :])[:, 0]
    Hh = cubo.shape[1]
    cols = torch.tensor(COLS, device=dev)
    livres = [k for k in range(E.AMOSTRA_IGNICAO, 767 - PASSO_RECUO) if k not in PROIB and k + PASSO_RECUO not in PROIB]
    s = 1.0 if face == "baixo" else -1.0                 # sentido do grão em y
    print(f"[{run.name} {face}] {len(tr)} quadros com rótulo | ImageNet | {dev}", flush=True)
    passo, t0, hist = 0, time.time(), []
    net.train()
    while passo < passos:
        for x, y, w, _ in dl:
            sup = surface_loss(net(x.to(dev))[0], y.to(dev), w.to(dev), sigma=sigma)
            k0 = rng.choice(livres, 4, replace=False)
            yy = altura(net, entrada(list(k0) + list(k0 + PASSO_RECUO)))[:, cols]
            p_par = torch.relu(s * (W[cols] - yy) - FOLGA_PAREDE) ** 2          # entrar na porta original
            p_mol = torch.relu(yy - (Hh - MOLDURA)) ** 2 if face == "baixo" else torch.relu(MOLDURA - yy) ** 2
            p_rec = torch.relu(-s * (yy[4:] - yy[:4]) - TOL_RECUO) ** 2
            fis = p_par.mean() + p_mol.mean() + p_rec.mean()
            if fuligem:
                kk = np.r_[k0, k0 + PASSO_RECUO]
                Pk = torch.tensor(PAR[kk][:, :, COLS], dtype=torch.float32, device=dev)
                p_fu = no_mapa(Pk, yy).mean()
                dentro = torch.tensor((k0 >= K_CORTE) & (k0 + PASSO_RECUO <= K_FIM), device=dev)[:, None] & m40[None, :]
                mv = s * (yy[4:] - yy[:4])
                p_mv = (torch.relu(MOV_MIN - mv) ** 2)[dentro]
                p_mv = p_mv.mean() if p_mv.numel() else torch.zeros((), device=dev)
                fis = fis + 20.0 * p_fu + 20.0 * p_mv
            loss = sup + LAM_FIS * fis
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); passo += 1
            if passo % 100 == 0:
                hist.append(dict(passo=passo, perda=float(sup.detach()), parede=float(p_par.mean()),
                                 moldura=float(p_mol.mean()), recuo=float(p_rec.mean()), s=time.time() - t0))
            if passo % 400 == 0:
                print(f"  passo {passo:5d} perda {float(sup.detach()):7.3f} parede {float(p_par.mean()):7.2f} "
                      f"moldura {float(p_mol.mean()):6.2f} recuo {float(p_rec.mean()):6.2f} ({time.time() - t0:.0f}s)", flush=True)
            if passo >= passos:
                break
    torch.save(net.state_dict(), run / f"{face}.pt")
    (run / f"hist_{face}.json").write_text(json.dumps(hist))


def oculta(nome, *args) -> None:
    if (R.OUT / f"runs/{nome}/saida.npz").exists():
        return
    cmd = [sys.executable, "-W", "ignore", str(R.RAIZ / "codigo/recompensa_oculta.py"), nome, *map(str, args),
           "--lam-parede", str(LAM_FIS)]
    subprocess.run(cmd, check=True)


def familia(a) -> None:
    pre = {"marcas": "s1v2", "classico": "sm1v2", "v3": "sm1v3", "v4": "sm1v4"}[a.etapa]
    run = R.OUT / "runs" / f"{pre}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "saida.npz").exists():
        d = rotulos(a.etapa)
        preparar_imagens(d["quadros"])
        for f in FACES:
            if (run / f"{f}.pt").exists():               # face já treinada numa execução interrompida
                print(f"[{run.name} {f}] já treinada, reaproveitada", flush=True)
                continue
            treinar_face(f, a.seed, run, *d["rotulos"][f], fuligem=(a.etapa == "v4"))
        (run / "config.json").write_text(json.dumps(dict(
            nome=run.name, inicio="ImageNet", rotulos=a.etapa, pre_ignicao=PRE, folga_parede=FOLGA_PAREDE,
            moldura=MOLDURA, passo_recuo=PASSO_RECUO, tol_recuo=TOL_RECUO, lam_fis=LAM_FIS, passos=1200, lote=8,
            lr=3e-4, sigma=2.0, seed=a.seed, quadros=len(d["quadros"])), indent=1))
        R.inferir(run)
    s = a.seed
    if a.etapa == "marcas":
        oculta(f"2bv2_s{s}", "--recompensa", "nada", "--inicio", f"run:s1v2_s{s}", "--ref", f"s1v2_s{s}",
               "--lam-ancora", "0", "--passos", "400", "--seed", s)
        oculta(f"formav2_s{s}", "--recompensa", "forma", "--w", "5", "--inicio", f"run:2bv2_s{s}", "--ref", f"2bv2_s{s}",
               "--seed", s)
    elif a.etapa == "v4":
        oculta(f"smfv4_s{s}", "--recompensa", "forma", "--w", "5", "--inicio", f"run:sm1v4_s{s}", "--ref", f"sm1v4_s{s}",
               "--rotulos", "v4", "--lam-parado", "1.0", "--lam-mov", "1.0", "--seed", s)
    elif a.etapa == "v3":
        oculta(f"smfv3_s{s}", "--recompensa", "forma", "--w", "5", "--inicio", f"run:sm1v3_s{s}", "--ref", f"sm1v3_s{s}",
               "--rotulos", "v3", "--seed", s)
    else:
        oculta(f"smfv2_s{s}", "--recompensa", "forma", "--w", "5", "--inicio", f"run:sm1v2_s{s}", "--ref", f"sm1v2_s{s}",
               "--rotulos", "classico", "--seed", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["paredes", "marcas", "classico", "v3", "v4"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.etapa == "paredes":
        W = paredes()
        print({f: W[f][[0, 46, 92]].round(1).tolist() for f in W})
    else:
        familia(a)
