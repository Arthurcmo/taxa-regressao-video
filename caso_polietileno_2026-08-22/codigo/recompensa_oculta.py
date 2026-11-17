"""Treino com recompensa escondida numa seção do grão, com travas.

A rede desenha a linha inteira (93 colunas, as duas faces), sem saber de seção nenhuma. Em
cada passo sorteamos uma grade de janelas (início entre 0 e 6,8 s depois do começo da janela
de análise, janelas de 15, 20 ou 25 s, até o salto de vazão; tempo da câmera vezes o fator de
tempo) e calculamos, só nas colunas da seção (40 a 80 mm por padrão), a taxa e o G_ox de cada
janela pelos extremos. A recompensa sai dali:
  forma  os pontos seguem uma reta em log-log com n entre 0,5 e 0,85 (sem impor o nível);
  nada   sem recompensa no treino (a recompensa só escolhe depois: "recebe no final").
Travas na perda:
  faixa    relu(|y - marca| - d)^2 nos quadros marcados (desligada em "sem marcas");
  ancora   relu(|y - y_2b| - D)^2 em quadros sorteados de 79 a 767 (não passear);
  tempo    a linha num quadro da grade não pode destoar da média de k-3 e k+3 (> 1,5 px);
  recuo    na grade, a face de baixo não sobe e a de cima não desce mais de 1 px.
Grade sorteada a cada passo: a rede não tem quadro fixo onde trapacear.

    PYTHONPATH=codigo <python com smp> codigo/recompensa_oculta.py NOME [opções]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ensaio as E
import faixas_grao as FG
import recompensa_lei as R
from experimento_s_sessao import K_IGN
from taxas_sessao import L_MM, MM

FACES = ("baixo", "cima")
COLS = FG.faixa_cols(40, 80)
CAM = E.CAM
RHO = E.DENSIDADE_OXIDANTE
K0, K1 = 78, 766
_PL = None


class Tempo:
    """Igual a relogio_e_corte.Tempo (sem importar o módulo, que puxa xlrd)."""

    def __init__(self, s, T0):
        self.s, self.T0 = s, T0
        self.ign = T0 + s * K_IGN * CAM

    def banco(self, k):
        return self.T0 + self.s * CAM * np.asarray(k, float)

    def k(self, t_banco):
        return (np.asarray(t_banco, float) - self.T0) / (self.s * CAM)


def contexto():
    d = json.loads((R.OUT / "relogio_e_corte/relogio_e_corte.json").read_text())
    return Tempo(d["fator"]["s"], d["fator"]["T0"]), float(E.JANELA_ANALISE_S[0]), float(d["fim_banco"])


def linha(nome):
    z = np.load(R.OUT / f"runs/{nome}/saida.npz")
    return z["baixo"].astype(float), z["cima"].astype(float)


def mdot(tm, k0, k1):
    global _PL
    if _PL is None:
        _PL = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    a, b = tm.banco(k0), tm.banco(k1)
    return float(np.nanmean(_PL[(_PL[:, 0] >= a) & (_PL[:, 0] <= b), 1])) * RHO * 1e-3


def pesos_iniciais(inicio, s):
    if inicio.startswith("run:"):                # rede de outra rodada (ex.: sem marcação)
        return {f: R.OUT / f"runs/{inicio[4:]}/{f}.pt" for f in FACES}
    if inicio == "2b":
        nome = "nenhum_d4" + ("" if s == 0 else f"_s{s}")
        return {f: R.OUT / f"runs/{nome}/{f}.pt" for f in FACES}
    return {f: R.S1_RUNS / f"S1_{f}_s{s}/best.pt" for f in FACES}


def treinar(a) -> None:
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cubo, dev, entrada, altura, _ = R._ferramentas()
    tm, corte, fim = contexto()
    run = R.OUT / "runs" / a.nome
    run.mkdir(parents=True, exist_ok=True)
    nets = {f: R._rede(f, dev, p) for f, p in pesos_iniciais(a.inicio, a.semente_inicio).items()}
    for f in FACES:
        nets[f].train()
        for m in nets[f].modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
    ref = linha(a.ref)
    Yref = {"baixo": torch.tensor(ref[0], dtype=torch.float32, device=dev),
            "cima": torch.tensor(ref[1], dtype=torch.float32, device=dev)}
    if a.rotulos == "classico":                  # faixa em torno do detector clássico, sem marcação
        import sem_marcacao as SM
        Y = {}
        for f in FACES:
            yc, wc = SM.rotulos_classico(f)
            Y[f] = np.where(wc > 0, yc, np.nan)
        ok = set(SM.quadros())
        km = {f: [k for k in range(963) if k in ok and np.isfinite(Y[f][k]).sum() > 10] for f in FACES}
    elif a.rotulos in ("v3", "v4"):              # faixa em torno dos rótulos v3/v4 (borda do lado do grão), sem marcação
        import sem_marcacao as SM
        Y = {}
        for f in FACES:
            z = np.load(R.OUT / f"sem_marcacao_{a.rotulos}" / f"alvos_{f}.npz")
            Y[f] = np.where(z["peso"] > 0, z["y"], np.nan)
        ok = set(SM.quadros())
        km = {f: [k for k in range(963) if k in ok and np.isfinite(Y[f][k]).sum() > 10] for f in FACES}
    else:
        Y = {f: R.autor(f) for f in FACES}
        km = {f: [k for k in range(963) if np.isfinite(Y[f][k]).sum() > 10] for f in FACES}
    grao = torch.tensor(R.GRAO, device=dev)
    Hh = cubo.shape[1]
    if a.lam_parede > 0:
        _w = np.load(R.OUT / "paredes_iniciais.npz")
        W0 = {"baixo": torch.tensor(_w["inf"], dtype=torch.float32, device=dev),
              "cima": torch.tensor(_w["sup"], dtype=torch.float32, device=dev)}
    if a.lam_parado > 0:
        from scipy.ndimage import gaussian_filter1d as _g1
        PAR = _g1(np.load(R.OUT / "fuligem_estatica" / "parado.npy").astype(np.float32) / 255.0, 4.0, axis=1)
    K_CORTE, K_FIM, MOV_MIN = int(round(E.AMOSTRA_IGNICAO + E.JANELA_ANALISE_S[0] / (E.CAM * E.FATOR_TEMPO_REAL))), int(round(E.AMOSTRA_IGNICAO + E.JANELA_ANALISE_S[1] / (E.CAM * E.FATOR_TEMPO_REAL))), 0.6 / 15   # janela de análise; px por amostra

    def no_mapa(Pk, yy):
        i0 = yy.detach().floor().clamp(0, Pk.shape[1] - 2).long()
        w_ = (yy - i0).clamp(0, 1)
        return (1 - w_) * Pk.gather(1, i0[:, None, :])[:, 0] + w_ * Pk.gather(1, (i0 + 1)[:, None, :])[:, 0]
    cols = torch.tensor(FG.faixa_cols(*a.secao), device=dev)
    cols_ini = torch.tensor(FG.faixa_cols(10, 40), device=dev)
    cols_resto = torch.tensor(FG.faixa_cols(40, 120), device=dev)
    lam = {"faixa": a.lam_faixa, "ancora": a.lam_ancora}
    LIM = {"faixa": a.lim_faixa, "ancora": a.lim_ancora}
    params = [p for f in FACES for p in nets[f].parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    def faixa(y, marca):
        ok = torch.isfinite(marca)
        return (torch.relu((y - torch.nan_to_num(marca)).abs() - a.d) ** 2)[ok]

    hist, t_ini = [], time.time()
    for passo in range(1, a.passos + 1):
        # grade de janelas sorteada
        t0 = corte + rng.uniform(0, 6.8)
        dur = float(rng.choice([15.0, 20.0, 25.0]))
        n = max(3, int(round((fim - tm.ign - t0) / dur)))
        ks = np.unique(np.round(tm.k(np.linspace(tm.ign + t0, fim, n + 1))).astype(int))
        dts = torch.tensor(np.diff(ks) * CAM * tm.s, dtype=torch.float32, device=dev)
        mds = torch.tensor([mdot(tm, ks[i], ks[i + 1]) for i in range(len(ks) - 1)], dtype=torch.float32, device=dev)
        viz = rng.choice(len(ks), size=min(2, len(ks)), replace=False)
        kviz = [int(ks[i]) + o for i in viz for o in (-3, 3)]
        kanc = list(rng.integers(K0, K1 + 1, size=4))
        y, perdas = {}, {"faixa": [], "ancora": [], "tempo": [], "recuo": [], "parede": [], "parado": [], "mov": []}
        for f in FACES:
            kmar = list(rng.choice(km[f], 4, replace=False)) if a.marcas else []
            todos = list(ks) + kviz + kanc + kmar
            out = altura(nets[f], entrada(todos))
            nk, nv, na = len(ks), len(kviz), len(kanc)
            yg, yv, ya, ym = out[:nk], out[nk:nk + nv], out[nk + nv:nk + nv + na], out[nk + nv + na:]
            y[f] = yg
            if a.marcas:
                perdas["faixa"].append(faixa(ym, torch.tensor(Y[f][kmar], dtype=torch.float32, device=dev)))
            perdas["ancora"].append((torch.relu((ya - Yref[f][kanc]).abs() - a.D) ** 2)[:, grao].flatten())
            meio = 0.5 * (yv[0::2] + yv[1::2])
            perdas["tempo"].append((torch.relu((yg[viz] - meio).abs() - 1.5) ** 2)[:, grao].flatten())
            dy = yg[1:] - yg[:-1]
            s = 1.0 if f == "baixo" else -1.0          # baixo: y cresce; cima: y diminui
            perdas["recuo"].append((torch.relu(-s * dy - 1.0) ** 2)[:, grao].flatten())
            if a.lam_parado > 0:                     # linha sobre fuligem parada
                kk = list(ks) + kanc
                Pk = torch.tensor(PAR[kk], dtype=torch.float32, device=dev)
                perdas["parado"].append(no_mapa(Pk, torch.cat([yg, ya]))[:, grao].flatten())
            if a.lam_mov > 0:                        # a superfície recua sempre: linha parada paga
                okk = [i for i in range(len(ks) - 1) if ks[i] >= K_CORTE and ks[i + 1] <= K_FIM]
                if okk:
                    dk = torch.tensor([ks[i + 1] - ks[i] for i in okk], dtype=torch.float32, device=dev)[:, None]
                    mv = s * dy[okk][:, cols_resto]
                    perdas["mov"].append((torch.relu(MOV_MIN * dk - mv) ** 2).flatten())
            if a.lam_parede > 0:                     # porta original (folga de 10 px) e moldura (14 px)
                yy = torch.cat([yg, ya])[:, 1:]
                pp = torch.relu(s * (W0[f][1:] - yy) - 10.0) ** 2
                pm = torch.relu(yy - (Hh - 14.0)) ** 2 if f == "baixo" else torch.relu(14.0 - yy) ** 2
                perdas["parede"].append((pp + pm).flatten())
        H = (y["baixo"] - y["cima"])[:, cols].mean(1) * MM
        r = (H[1:] - H[:-1]) / (2 * dts)
        G = mds / (L_MM * 1e-3 * 0.5 * (H[1:] + H[:-1]).clamp_min(1.0) * 1e-3)
        lr_, lg = torch.log(r.clamp_min(1e-4)), torch.log(G)
        xm, ym_ = lg - lg.mean(), lr_ - lr_.mean()
        nhat = (xm * ym_).sum() / (xm ** 2).sum().clamp_min(1e-6)
        p_forma = ((ym_ - nhat * xm) ** 2).mean() + torch.relu(0.5 - nhat) ** 2 + torch.relu(nhat - 0.85) ** 2
        p_rec = {"forma": p_forma, "nada": torch.zeros((), device=dev)}[a.recompensa]
        # entrada mais rápida: taxa de 10-40 mm / taxa de 40-120 mm >= razao_inicio (recirculação)
        def taxa(cc):
            Hc = (y["baixo"] - y["cima"])[:, cc].mean(1) * MM
            return ((Hc[1:] - Hc[:-1]) / (2 * dts)).clamp_min(1e-4)
        q_ini = torch.log(taxa(cols_ini)) - torch.log(taxa(cols_resto))
        p_ini = (torch.relu(np.log(a.razao_inicio) - q_ini) ** 2).mean()
        P = {k: (torch.cat(v).mean() if v else torch.zeros((), device=dev)) for k, v in perdas.items()}
        perda = (a.w * p_rec + a.bonus_inicio * p_ini + lam["faixa"] * P["faixa"] + lam["ancora"] * P["ancora"]
                 + a.lam_tempo * P["tempo"] + a.lam_recuo * P["recuo"] + a.lam_parede * P["parede"]
                 + a.lam_parado * P["parado"] + a.lam_mov * P["mov"])
        if a.adaptativo:                 # pesos das travas sobem quando a trava é violada (Lagrange)
            for k in lam:
                lam[k] = float(np.clip(lam[k] * np.exp(0.05 * (float(P[k]) / LIM[k] - 1)), 0.005, 50.0))
        opt.zero_grad(); perda.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if passo % 25 == 0 or passo == 1:
            item = dict(passo=passo, recompensa=float(p_rec), forma=float(p_forma), n=float(nhat),
                        inicio=float(torch.exp(q_ini).median()),
                        **{k: float(v) for k, v in P.items()}, lam_faixa=lam["faixa"], lam_ancora=lam["ancora"],
                        s=round(time.time() - t_ini))
            hist.append(item)
            print(f"[{a.nome}] {passo:4d} rec {item['recompensa']:.3f} (forma {item['forma']:.3f} "
                  f"n {item['n']:.2f}) faixa {item['faixa']:.2f} âncora {item['ancora']:.2f} tempo {item['tempo']:.2f} "
                  f"recuo {item['recuo']:.2f} | entrada/resto {item['inicio']:.2f} | λ {lam['faixa']:.3f}/{lam['ancora']:.3f} "
                  f"({item['s']} s)", flush=True)
    for f in FACES:
        torch.save(nets[f].state_dict(), run / f"{f}.pt")
    (run / "hist.json").write_text(json.dumps(hist, indent=1))
    (run / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    R.inferir(run, {f: run / f"{f}.pt" for f in FACES})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nome")
    ap.add_argument("--recompensa", default="forma", choices=["forma", "nada"])
    ap.add_argument("--w", type=float, default=5.0)
    ap.add_argument("--inicio", default="2b", help="2b, S1 ou run:NOME")
    ap.add_argument("--ref", default="nenhum_d4", help="rodada da âncora (padrão: 2b semente 0)")
    ap.add_argument("--rotulos", default="marcas", choices=["marcas", "classico", "v3", "v4"])
    ap.add_argument("--semente-inicio", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=float, default=4.0)
    ap.add_argument("--D", type=float, default=6.0)
    ap.add_argument("--lam-faixa", type=float, default=0.05)
    ap.add_argument("--lam-ancora", type=float, default=0.05)
    ap.add_argument("--lam-tempo", type=float, default=0.05)
    ap.add_argument("--lam-recuo", type=float, default=0.05)
    ap.add_argument("--sem-marcas", dest="marcas", action="store_false")
    ap.add_argument("--lam-parede", type=float, default=0.0, help="porta original e moldura (0 = desligado)")
    ap.add_argument("--lam-parado", type=float, default=0.0, help="linha sobre fuligem parada (0 = desligado)")
    ap.add_argument("--lam-mov", type=float, default=0.0, help="linha parada na janela de análise, 40 a 120 mm (0 = desligado)")
    ap.add_argument("--passos", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--secao", type=float, nargs=2, default=[40.0, 80.0])
    ap.add_argument("--bonus-inicio", type=float, default=0.0)
    ap.add_argument("--razao-inicio", type=float, default=1.2)
    ap.add_argument("--adaptativo", action="store_true")
    ap.add_argument("--lim-faixa", type=float, default=0.5)
    ap.add_argument("--lim-ancora", type=float, default=0.5)
    treinar(ap.parse_args())
