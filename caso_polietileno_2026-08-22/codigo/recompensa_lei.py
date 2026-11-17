"""Ajuste fino da rede S1 com as marcações manuais (a linha 2b) e contas comuns de taxa.

Treino: as duas redes S1 (uma por face) são ajustadas juntas com uma perda que pune a
linha que sai de uma faixa de ±d px em torno das marcações manuais (fora dela a linha
estaria na chama ou dentro do sólido). O alvo "nenhum" usa só essa faixa, e é a 2b. O
alvo "livre" soma uma recompensa que pede que a taxa siga uma lei de potência com a e n
aprendidos durante o treino (n entre 0,5 e 0,8), sem nenhum valor imposto de fora.

Este módulo também tem as contas comuns: marcações manuais (sessão principal mais a camada
B, que tem o quadro 78), vazão, trechos e a trajetória que uma lei a·G^n prevê para a porta.

    PYTHONPATH=codigo python codigo/recompensa_lei.py s1
    PYTHONPATH=codigo python codigo/recompensa_lei.py treinar nenhum 4 400
    PYTHONPATH=codigo python codigo/recompensa_lei.py lei
"""
from __future__ import annotations

import csv
import json
import sys

import cv2
import numpy as np

import ensaio as E
import lei_potencia as RG
import registro_vazao as V
from experimento_s_sessao import DT, K_IGN, RAIZ, alvo_por_coluna, marcas
from taxas_sessao import L_MM, MM

RHO = V.RHO
OUT = RAIZ.parent / "experimento_recompensa_lei_2026-11-06"
PERFIL_B = RAIZ / "resultados/revisao_humana/perfis/sessao-2026-11-02-2e7d0cbb/guia.csv"
# Pontas dos trechos: quadros marcados nas duas faces (a de cima tem 180 no lugar de 179).
PONTAS = (78, 179, 296, 405, 525, 655)
PONTAS_CIMA = {179: 180}
GRAO = np.arange(8, 90)


def t(k):
    return (np.asarray(k, float) - K_IGN) * DT


def marcas_b(face: str) -> dict[int, np.ndarray]:
    pts: dict[int, list] = {}
    for r in csv.DictReader(PERFIL_B.open()):
        if r["tipo"] == "ancora" and r["face"] == face and r.get("camada") == "B":
            pts.setdefault(int(r["k"]), []).append((float(r["x_px"]) - 30, float(r["y_roi_px"])))
    return {k: np.array(sorted(v)) for k, v in pts.items()}


def autor(face: str) -> np.ndarray:
    """(963, 93) com a altura marcada por coluna; NaN onde não há marca."""
    Y = np.full((963, 93), np.nan)
    for fonte in (marcas(face), marcas_b(face)):
        for k, p in fonte.items():
            y, w = alvo_por_coluna(p)
            Y[k, w > 0] = y[w > 0]
    return Y


def ponta(face: str, k: int) -> int:
    return PONTAS_CIMA.get(k, k) if face == "cima" else k


_PLAN = None


def mdot(k0: float, k1: float) -> float:
    """Vazão média de oxidante (kg/s) entre as amostras k0 e k1."""
    global _PLAN
    if _PLAN is None:
        _PLAN = np.loadtxt(E.VAZAO_CSV, delimiter=",", skiprows=1)
    a, b = V.t_planilha(k0), V.t_planilha(k1)
    s = (_PLAN[:, 0] >= a) & (_PLAN[:, 0] <= b)
    return float(_PLAN[s, 1].mean()) * RHO * 1e-3


def G_ox(md: float, H_mm: float) -> float:
    return md / (L_MM * 1e-3 * H_mm * 1e-3)


def lei(G, a, n):
    return a * np.asarray(G, float) ** n


def trajetoria_lei(H0_mm: float, k0: int, k1: int, a: float, n: float, passo=1):
    """Integra dH/dt = 2 a (mdot/(L H))^n de k0 a k1 (Euler, uma amostra por passo)."""
    ks = np.arange(k0, k1 + 1, passo)
    H = np.empty(len(ks))
    H[0] = H0_mm
    for i in range(1, len(ks)):
        md = mdot(ks[i - 1], ks[i])
        H[i] = H[i - 1] + 2 * lei(G_ox(md, H[i - 1]), a, n) * (ks[i] - ks[i - 1]) * DT
    return ks, H


def porta_por_quadro(Yb, Yc, ks, cols=GRAO, ref=None):
    """Altura média da porta (mm) em cada quadro. Com `ref` (k de referência), usa a
    abertura em relação a ref coluna a coluna, para não depender de quais colunas
    estão marcadas em cada quadro."""
    out = []
    for k in ks:
        kb, kc = k, ponta("cima", k) if k in PONTAS_CIMA else k
        h = (Yb[kb, cols] - Yc[kc, cols]) * MM
        if ref is None:
            out.append(np.nanmean(h))
        else:
            h0 = (Yb[ref, cols] - Yc[ref, cols]) * MM
            ok = np.isfinite(h) & np.isfinite(h0)
            out.append(np.nan if ok.sum() < 5 else float(np.nanmean(h0) + (h[ok] - h0[ok]).mean()))
    return np.array(out)


def mascaras(ks=PONTAS):
    """Colunas do grão marcadas pelo autor nas duas faces no quadro k e no quadro 78
    (base), uma linha por k, e a máscara da base."""
    Yb, Yc = autor("baixo"), autor("cima")
    grao = np.isin(np.arange(93), GRAO)
    base = grao & np.isfinite(Yb[78]) & np.isfinite(Yc[78])
    C = np.stack([base & np.isfinite(Yb[k]) & np.isfinite(Yc[ponta("cima", k) if k in PONTAS_CIMA else k]) for k in ks])
    return C, base


def porta_mascarada(Yb, Yc, ks=PONTAS) -> np.ndarray:
    """Porta média (mm) em cada k nas colunas que o autor marcou ali: altura da porta
    no quadro 78 (todas as colunas marcadas) mais a abertura média desde 78 nas
    colunas marcadas em k. Igual para todos os métodos, inclusive as marcações."""
    C, base = mascaras(ks)
    h78 = Yb[78] - Yc[78]
    out = []
    for c, k in zip(C, ks):
        hk = Yb[k] - Yc[ponta("cima", k) if k in PONTAS_CIMA else k]
        out.append((np.mean(h78[base]) + np.mean((hk - h78)[c])) * MM)
    return np.array(out)


def trechos(Hk: dict[int, float], pontas=PONTAS) -> list[dict]:
    """Taxa (extremos da porta) e G_ox = ṁ_ox / (L H médio) de cada trecho entre pontas consecutivas."""
    res = []
    for a, b in zip(pontas[:-1], pontas[1:]):
        tq = (b - a) * DT
        r = (Hk[b] - Hk[a]) / (2 * tq)
        G = G_ox(mdot(a, b), 0.5 * (Hk[a] + Hk[b]))
        res.append(dict(k=[a, b], t=[float(t(a)), float(t(b))], r=float(r), G=float(G)))
    return res


def quadros_ambas(Yb, Yc, kmin=78, kmax=700):
    kb = {k for k in range(963) if np.isfinite(Yb[k]).sum() > 10}
    kc = {k for k in range(963) if np.isfinite(Yc[k]).sum() > 10}
    ks = sorted(k for k in kb if (k in kc or (k == 179 and 180 in kc)) and kmin <= k <= kmax)
    return ks


def ajustar_trajetoria(Hk, ks, H, k0, nmin, nmax) -> dict:
    """Melhor a e n (n entre nmin e nmax) para a trajetória da lei, partindo da porta
    marcada em k0, contra a porta marcada nos quadros seguintes."""
    sel = [i for i, k in enumerate(ks) if k >= k0]
    kk = [ks[i] for i in sel]
    melhor = None
    for n in np.arange(nmin, nmax + 1e-9, 0.05):
        for a in np.geomspace(0.001, 0.2, 140):
            ksl, Hl = trajetoria_lei(Hk[k0], k0, max(ks), a, n, passo=3)
            e = np.interp(kk, ksl, Hl) - H[sel]
            s = float(np.sqrt(np.mean(e ** 2)))
            if melhor is None or s < melhor[0]:
                melhor = (s, a, n, e)
    s, a, n, e = melhor
    return dict(k0=k0, a=float(a), n=float(n), rms_px_face=s / MM / 2, max_px_face=float(np.abs(e).max() / MM / 2),
                desvio_px_face=(e / MM / 2).tolist())


def etapa_lei() -> dict:
    """Taxa das marcações manuais por trecho e a lei a·G^n que melhor reproduz a porta
    marcada, integrada a partir de um quadro inicial."""
    Yb, Yc = autor("baixo"), autor("cima")
    ks = quadros_ambas(Yb, Yc)
    H = porta_por_quadro(Yb, Yc, ks, ref=78)
    ok = np.isfinite(H)
    ks, H = [k for k, o in zip(ks, ok) if o], H[ok]
    Hk = dict(zip(ks, H))
    res = dict(quadros=ks, H_autor=H.tolist(), trechos_autor=trechos(Hk))
    print("Marcações manuais, trechos:")
    for d in res["trechos_autor"]:
        print(f"  amostras {d['k'][0]+1}-{d['k'][1]+1} ({d['t'][0]:.0f}-{d['t'][1]:.0f} s): r {d['r']:.4f} G {d['G']:.2f}")
    res["ajustes"] = {}
    for nome, k0, nmin, nmax in (("livre, desde 79", 78, -1.0, 1.2), ("n de 0,5 a 0,8, desde 79", 78, 0.5, 0.8),
                                 ("livre, desde 180", 179, -1.0, 1.2), ("n de 0,5 a 0,8, desde 180", 179, 0.5, 0.8)):
        res["ajustes"][nome] = ajustar_trajetoria(Hk, ks, H, k0, nmin, nmax)
        d = res["ajustes"][nome]
        print(f"{nome:28s}: a {d['a']:.4f} n {d['n']:+.2f}, rms {d['rms_px_face']:.1f} px/face, maior {d['max_px_face']:.1f}")
    return res


FACES = ("baixo", "cima")
S1_RUNS = RAIZ / "experimento_s_sessao_2026-10-28/runs"


def _rede(face: str, dev, pesos=None):
    import torch
    import experimento_p2_v2 as P2
    net = P2.construir("r18").to(dev)
    net.load_state_dict(torch.load(pesos or S1_RUNS / f"S1_{face}_s0/best.pt", map_location=dev, weights_only=True))
    return net


def _ferramentas():
    import torch
    import experimento_p2_v2 as P2
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    N, Hh, nb = cubo.shape[:3]
    oh, ow = P2.TAMANHO["r18"]
    xs = np.linspace(0, nb - 1, ow)
    Mx = np.stack([np.interp(np.arange(nb), xs, np.eye(ow)[i]) for i in range(ow)])   # (ow, nb)
    dev = P2.dispositivo()

    def entrada(ks):
        x = np.stack([cv2.resize(cubo[k], (ow, oh)).astype(np.float32) / 255 for k in ks])
        return torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)

    Mt = torch.tensor(Mx, dtype=torch.float32, device=dev)

    def altura(net, x):
        """y em pixel da ROI nas 93 colunas, (B, 93)."""
        from chamacera.surfnet import soft_argmax_col
        y, _ = soft_argmax_col(net(x)[0])
        return (y[:, 0] * Hh / oh) @ Mt

    def altura_img(net, net0, x, tau=4.0, eps=1e-3):
        """Como `altura`, e mais o custo de imagem: -log p0(y) por coluna, com p0 a
        distribuição da S1 original congelada, alargada (temperatura tau) e com um
        piso uniforme eps, para que sair do pico custe mas não seja proibido.
        Devolve (y em px da ROI (B, 93), custo médio por quadro (B,))."""
        import torch.nn.functional as F
        from chamacera.surfnet import soft_argmax_col
        lg = net(x)[0]
        y, _ = soft_argmax_col(lg)
        y = y[:, 0]                                                      # (B, ow), linhas da grade da rede
        with torch.no_grad():
            p0 = F.softmax(net0(x)[0][:, 0] / tau, dim=1) * (1 - eps) + eps / oh
            lp0 = p0.log()                                               # (B, oh, ow)
        i0 = y.detach().floor().clamp(0, oh - 2).long()
        w = (y - i0).clamp(0, 1)
        a_ = lp0.gather(1, i0[:, None, :])[:, 0]
        b_ = lp0.gather(1, (i0 + 1)[:, None, :])[:, 0]
        custo = -((1 - w) * a_ + w * b_)
        return (y * Hh / oh) @ Mt, custo.mean(1)

    return cubo, dev, entrada, altura, altura_img


def treinar(alvo: str, d: float, passos: int = 400, lr: float = 1e-4, lam: float = 0.05, seed: int = 0,
            lam_img: float = 0.045) -> None:
    """Ajusta as duas redes S1 juntas. Perda = lei + lam * faixa.

    lei:   só no alvo "livre": em cada trecho entre as PONTAS, (r - r_lei)^2 / r_lei^2,
           com r e G_ox calculados da porta média que as redes dão nas pontas; a e n são
           aprendidos, com n preso entre 0,5 e 0,8 (só exige que a taxa cresça com o
           fluxo). No alvo "nenhum" (a 2b) a lei não entra;
    faixa: em todo quadro marcado, relu(|y - marca| - d)^2 por coluna marcada: a
           linha pode andar d px em volta da marca sem custo; além disso paga.
    alvo com sufixo "img" ("livreimg"): a rede pode sair da marcação para
           corrigi-la. A faixa vira só o limite do plausível (d grande) e entra o
           custo de imagem lam_img * (-log p0(y)) em todas as colunas, com p0 a S1
           original alargada: a linha só sai da marca para onde a visão ainda vê
           superfície. lam_img = 0,045 faz 10 px fora do pico (~2 nats) custar o
           mesmo que errar a lei em 30 % em todos os trechos.
    alvos aceitos: "nenhum", "livre" e "livreimg".
    BatchNorm congelada (só 15 quadros por passo)."""
    img = alvo.endswith("img")
    lei_alvo = alvo[:-3] if img else alvo
    if lei_alvo not in ("nenhum", "livre"):
        raise ValueError(f"alvo {alvo!r}: use nenhum, livre ou livreimg")
    import torch
    import cv2 as _cv2  # noqa: F401
    torch.manual_seed(seed); np.random.seed(seed)
    cubo, dev, entrada, altura, altura_img = _ferramentas()
    nome = f"{alvo}_d{d:g}"
    if lam != 0.05:
        nome += f"_lf{lam:g}"
    if img and lam_img != 0.045:
        nome += f"_li{lam_img:g}"
    if seed:
        nome += f"_s{seed}"
    run = OUT / "runs" / nome
    run.mkdir(parents=True, exist_ok=True)
    pesos0 = {f: S1_RUNS / f"S1_{f}_s{seed}/best.pt" for f in FACES}     # a S1 da mesma semente
    nets = {f: _rede(f, dev, pesos0[f]) for f in FACES}
    nets0 = {f: _rede(f, dev, pesos0[f]).eval() for f in FACES} if img else {}
    for f in nets0:
        for p_ in nets0[f].parameters():
            p_.requires_grad_(False)
    Y = {f: autor(f) for f in FACES}
    km = {f: [k for k in range(963) if np.isfinite(Y[f][k]).sum() > 10] for f in FACES}
    Xp = {f: entrada([ponta(f, k) for k in PONTAS]) for f in FACES}
    Mp = {f: torch.tensor(Y[f][[ponta(f, k) for k in PONTAS]], dtype=torch.float32, device=dev) for f in FACES}
    dts = torch.tensor([(b - a) * DT for a, b in zip(PONTAS[:-1], PONTAS[1:])], dtype=torch.float32, device=dev)
    mds = torch.tensor([mdot(a, b) for a, b in zip(PONTAS[:-1], PONTAS[1:])], dtype=torch.float32, device=dev)
    C, base = mascaras(PONTAS)
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    bt = torch.tensor(base, dtype=torch.float32, device=dev)
    la = torch.tensor(float(np.log(0.03)), dtype=torch.float32, device=dev, requires_grad=True)
    zn = torch.tensor(0.0, dtype=torch.float32, device=dev, requires_grad=True)
    params = [p for f in FACES for p in nets[f].parameters()]
    opt = torch.optim.AdamW([{"params": params, "lr": lr}] + ([{"params": [la, zn], "lr": 2e-2}] if lei_alvo == "livre" else []),
                            weight_decay=1e-4)
    for f in FACES:
        nets[f].train()
        for m in nets[f].modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def faixa(y, marca):
        ok = torch.isfinite(marca)
        e = (y - torch.nan_to_num(marca)).abs()
        return (torch.relu(e - d) ** 2)[ok]

    hist = []
    t0 = __import__("time").time()
    for passo in range(1, passos + 1):
        custo_img = []
        if img:
            yp = {}
            for f in FACES:
                yp[f], c_ = altura_img(nets[f], nets0[f], Xp[f])
                custo_img.append(c_)
        else:
            yp = {f: altura(nets[f], Xp[f]) for f in FACES}
        h = yp["baixo"] - yp["cima"]                                            # (pontas, 93) px
        H = ((h[0] * bt).sum() / bt.sum() + ((h - h[0]) * Ct).sum(1) / Ct.sum(1)) * MM   # só colunas marcadas
        r = (H[1:] - H[:-1]) / (2 * dts)
        G = mds / (L_MM * 1e-3 * 0.5 * (H[1:] + H[:-1]) * 1e-3)
        a_, n_ = la.exp(), 0.5 + 0.3 * torch.sigmoid(zn)
        rl = a_ * G ** n_
        perda_lei = (((r - rl) / rl) ** 2).mean()
        if lei_alvo == "nenhum":      # a 2b: só a faixa em volta das marcações, sem lei
            perda_lei = perda_lei.detach() * 0
        pen = [faixa(yp[f], Mp[f]) for f in FACES]
        for f in FACES:
            ks = list(np.random.choice(km[f], 8, replace=False))
            if img:
                y_, c_ = altura_img(nets[f], nets0[f], entrada(ks))
                custo_img.append(c_)
            else:
                y_ = altura(nets[f], entrada(ks))
            pen.append(faixa(y_, torch.tensor(Y[f][ks], dtype=torch.float32, device=dev)))
        perda_faixa = torch.cat(pen).mean()
        perda_img = torch.cat(custo_img).mean() if img else torch.zeros((), device=dev)
        perda = perda_lei + lam * perda_faixa + lam_img * perda_img
        opt.zero_grad(); perda.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if passo % 25 == 0 or passo == 1:
            item = dict(passo=passo, perda_lei=float(perda_lei), perda_faixa=float(perda_faixa), perda_img=float(perda_img),
                        r=[round(float(v), 4) for v in r], G=[round(float(v), 2) for v in G],
                        razao=[round(float(v), 2) for v in (r / rl)], a=float(a_), n=float(n_),
                        s=round(__import__("time").time() - t0, 1))
            hist.append(item)
            print(f"[{nome}] passo {passo:4d} lei {item['perda_lei']:.3f} faixa {item['perda_faixa']:7.2f} "
                  f"img {item['perda_img']:.2f} razao {item['razao']} a {item['a']:.4f} n {item['n']:.2f} ({item['s']:.0f} s)", flush=True)
    for f in FACES:
        torch.save(nets[f].state_dict(), run / f"{f}.pt")
    (run / "hist.json").write_text(json.dumps(hist, indent=1))
    (run / "config.json").write_text(json.dumps(dict(alvo=alvo, d_px=d, passos=passos, lr=lr, lam=lam, seed=seed,
                                                     lam_img=lam_img if img else None,
                                                     pontas=PONTAS, colunas=GRAO.tolist(), inicio=f"S1_*_s{seed}"), indent=1))
    inferir(run)


def inferir(run, pesos: dict | None = None) -> None:
    """Saída quadro a quadro (sem Viterbi) das duas redes em todos os quadros."""
    import torch
    cubo, dev, entrada, altura, altura_img = _ferramentas()
    out = {}
    for f in FACES:
        net = _rede(f, dev, (pesos or {}).get(f, run / f"{f}.pt"))
        net.eval()
        Yf = np.zeros((cubo.shape[0], 93), np.float32)
        with torch.no_grad():
            for i in range(0, cubo.shape[0], 16):
                ks = list(range(i, min(cubo.shape[0], i + 16)))
                Yf[ks] = altura(net, entrada(ks)).cpu().numpy()
        out[f] = Yf
    np.savez_compressed(run / "saida.npz", **out)
    print(f"[{run.name}] saída gravada", flush=True)


if __name__ == "__main__":
    import argparse
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["lei", "treinar", "s1"])
    ap.add_argument("alvo", nargs="?")
    ap.add_argument("d", nargs="?", type=float)
    ap.add_argument("passos", nargs="?", type=int, default=400)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--lam-img", type=float, default=0.045)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.etapa == "lei":
        (OUT / "lei_contra_marcacoes.json").write_text(json.dumps(etapa_lei(), indent=1, ensure_ascii=False))
    elif a.etapa == "treinar":
        treinar(a.alvo, a.d, a.passos, lam=a.lam, lam_img=a.lam_img, seed=a.seed)
    else:
        inferir(OUT / ("runs/S1" + (f"_s{a.seed}" if a.seed else "")),
                {f: S1_RUNS / f"S1_{f}_s{a.seed}/best.pt" for f in FACES})
