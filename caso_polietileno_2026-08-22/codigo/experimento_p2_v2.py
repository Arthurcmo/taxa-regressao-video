"""Prioridade 2 refeita: ablação controlada do aprendizado profundo.

O experimento de 09/10 (`experimento_prioridades12.py`) concluiu que a rede
perdia para o detector clássico (29,6 px contra 5,0 px). A auditoria de 10/10
achou duas causas que não são do método:

1. Os rótulos de treino (`rotulos_humanos_face_inferior_v1`) estão atrasados em
   relação à superfície. Contra as 24 marcações manuais do usuário dentro do
   bloco de teste, eles erram 9,8 px com viés de -8 px, e a referência de teste
   erra 2,6 px. A rede aprendeu o atraso.
2. O treino não convergiu: 16 quadros, lote 4, 60 épocas, cerca de 240 passos;
   o erro de validação oscilava entre 22 e 66 px nas últimas épocas.

Aqui cada causa é isolada, uma mudança por rodada, e todas as redes recebem o
MESMO decodificador temporal do detector clássico (Viterbi monótono em três
passadas), trocando só o mapa de evidência. Assim a comparação final é
"filtro de vale contra rede", com o resto da cadeia igual.

O bloco de teste (k=180..480), o embargo (k=150..510), a referência de teste,
as colunas (18..90) e a escala são os do protocolo congelado de 09/10.

    python codigo/experimento_p2_v2.py rotulos
    python codigo/experimento_p2_v2.py treinar --nome R3_s0 --rotulos hibrido \
        --quadros denso --modelo surfnet --seed 0
    python codigo/experimento_p2_v2.py inferir --nome R3_s0
    python codigo/experimento_p2_v2.py avaliar
"""
from __future__ import annotations

import ensaio as E
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CODE = RAIZ.parent / "chamacera_2026-08-16/codigo"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(RAIZ / "codigo"))

E1 = RAIZ / "experimento_prioridades12_2026-10-09"       # protocolo congelado
OUT = RAIZ / "experimento_p2_v2_2026-10-10"
ROT = OUT / "rotulos"
RUNS = OUT / "runs"

COLS = np.arange(18, 91)
EMBARGO = (150, 510)
IGNICAO_K = E.AMOSTRA_IGNICAO
FIM_INF_K = 880             # face de baixo legível até aqui (~235 s de vídeo)
VAL = [90, 660, 810]        # mesma validação do experimento de 09/10
QUADROS_CODEX = [0, 75, 105, 120, 135, 540, 570, 600, 630, 690, 720, 750,
                 780, 840, 870, 900]
TAMANHO = {"surfnet": (592, 96), "r18": (576, 96)}


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def proibido_baixo(H: int, c0: float, margem: int = 14) -> np.ndarray:
    p = np.zeros(H, bool)
    p[:int(round(c0))] = True
    p[H - margem:] = True
    return p


# --------------------------------------------------------------------------- #
# Rótulos
# --------------------------------------------------------------------------- #
def rotulos(_a=None) -> None:
    """Monta os dois conjuntos de alvo da face inferior.

    `legado`: o mesmo de 09/10 (revisão assistida antiga).
    `hibrido`: detector clássico entre a ignição e o embargo, onde ele concorda
    com a referência de teste (viés +1,8 px em k=180); trajetória ancorada nas
    marcações manuais do usuário FORA do bloco depois do embargo, onde ela
    corrige o clássico preso na fuligem da cabeça do grão; e a parede marcada
    pelo usuário antes da ignição. Nenhuma marcação de dentro do embargo entra.
    """
    ROT.mkdir(parents=True, exist_ok=True)
    guia = RAIZ / "resultados/revisao_humana/perfis/marcacoes-manuais-corrigidas-aadc04d2/guia.csv"
    linhas = list(csv.DictReader(guia.open()))
    fora = [r for r in linhas if not EMBARGO[0] <= int(r["k"]) <= EMBARGO[1]]
    g = ROT / "guia_fora_do_embargo.csv"
    with g.open("w") as f:
        w = csv.DictWriter(f, fieldnames=linhas[0].keys())
        w.writeheader(); w.writerows(fora)
    anc = ROT / "interfaces_revisado.npz"
    if not anc.exists():
        import subprocess
        subprocess.run([sys.executable, str(RAIZ / "codigo/reajustar.py"), "--guia", str(g),
                        "--out", str(ROT)], check=True, env={**os.environ, "MPLBACKEND": "Agg"})
    yanc = np.load(anc)["y_inf"]
    cl = np.load(E1 / "classico.npz")
    leg = np.load(RAIZ / "resultados/rotulos_humanos_face_inferior_v1.npz")

    N, nb = yanc.shape
    hib = np.zeros((N, 1, nb), np.float32)
    ph = np.zeros_like(hib)
    origem = np.full(N, "", object)
    for k in range(N):
        if EMBARGO[0] <= k <= EMBARGO[1] or k > FIM_INF_K + 20:
            continue
        if IGNICAO_K <= k < EMBARGO[0]:
            hib[k, 0] = cl["y_inf"][k]; origem[k] = "classico"
        else:
            hib[k, 0] = yanc[k]; origem[k] = "ancorado_usuario"
        ph[k, 0, COLS] = 1.0
    lg = np.zeros_like(hib); pl = np.zeros_like(hib)
    for i, k in enumerate(leg["k"]):
        if EMBARGO[0] <= k <= EMBARGO[1]:
            continue
        lg[k, 0] = leg["y"][i, 1]; pl[k, 0, COLS] = leg["peso"][i, 1, COLS]
    assert ph[EMBARGO[0]:EMBARGO[1] + 1].sum() == 0 and pl[EMBARGO[0]:EMBARGO[1] + 1].sum() == 0
    np.savez_compressed(ROT / "alvos.npz", hibrido=hib, peso_hibrido=ph,
                        legado=lg, peso_legado=pl, origem_hibrido=origem.astype(str))
    rows = [r for r in linhas if r["face"] == "baixo"]
    save_json(ROT / "rotulos.json", dict(
        guia_origem=str(guia.relative_to(RAIZ)),
        marcacoes_usadas=sorted({(r["face"], int(r["k"])) for r in fora}),
        marcacoes_excluidas_baixo=sorted({int(r["k"]) for r in rows
                                          if EMBARGO[0] <= int(r["k"]) <= EMBARGO[1]}),
        reajuste="codigo/reajustar.py com tol=1, obediencia=1, dmax=2, lam=0,5 sobre resultados/interfaces.npz",
        hibrido=dict(pre_ignicao="trajetória ancorada (parede marcada pelo usuário em k=0)",
                     ignicao_ate_embargo="detector clássico (classico.npz de 09/10)",
                     depois_do_embargo="trajetória ancorada nas marcações do usuário fora do bloco",
                     colunas=[int(COLS[0]), int(COLS[-1])], ultimo_k=FIM_INF_K + 20)))
    print("rótulos gravados em", ROT)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def construir(modelo: str, p_drop: float = 0.2):
    import torch
    import torch.nn as nn
    if modelo == "surfnet":
        from chamacera.surfnet import SurfaceNet
        return SurfaceNet(in_ch=3, n_surf=1, base=24, depth=4, p_drop=p_drop, seg_head=False)
    if modelo == "r18":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import segmentation_models_pytorch as smp

        class UNetR18(nn.Module):
            """U-Net com codificador ResNet18 pré-treinado no ImageNet (timm).

            A saída tem um canal por superfície e passa pela mesma softmax por
            coluna da SurfaceNet; só o corpo convolucional muda.
            """

            def __init__(self):
                super().__init__()
                self.net = smp.Unet("tu-resnet18", encoder_weights="imagenet", classes=1)
                self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("dp", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x):                       # x: BGR em [0, 1]
                return self.net((x.flip(1) - self.mu) / self.dp), None

        return UNetR18()
    raise ValueError(modelo)


def dispositivo():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #
def quadros_treino(tipo: str) -> list[int]:
    if tipo == "codex":
        return list(QUADROS_CODEX)
    perto_val = {v + d for v in VAL for d in range(-3, 4)}
    ks = list(range(0, IGNICAO_K, 8)) + list(range(IGNICAO_K, EMBARGO[0], 2)) + \
        list(range(EMBARGO[1] + 2, FIM_INF_K + 1, 2))
    return [k for k in ks if k not in perto_val]


def treinar(a) -> None:
    import torch
    from torch.utils.data import DataLoader
    from chamacera.surfdata import SurfDataset
    from chamacera.surfnet import soft_argmax_col, surface_loss

    run = RUNS / a.nome
    run.mkdir(parents=True, exist_ok=True)
    z = np.load(ROT / "alvos.npz")
    alvo, peso = z[a.rotulos], z[f"peso_{a.rotulos}"]
    tr = quadros_treino(a.quadros)
    tr = [k for k in tr if peso[k].sum() > 0]
    assert not set(tr) & set(range(EMBARGO[0], EMBARGO[1] + 1)) and not set(tr) & set(VAL)
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    img = OUT / "dataset/img"
    img.mkdir(parents=True, exist_ok=True)
    for k in tr + VAL:
        p = img / f"{k:06d}.png"
        if not p.exists():
            assert cv2.imwrite(str(p), cubo[k])
    size = TAMANHO[a.modelo]
    H = cubo.shape[1]
    esc = H / size[0]
    ds_tr = SurfDataset(str(OUT / "dataset"), alvo, peso, [f"{k:06d}" for k in tr], size,
                        True, 3, a.seed, com_mascara=False)
    ds_va = SurfDataset(str(OUT / "dataset"), alvo, peso, [f"{k:06d}" for k in VAL], size,
                        False, 3, a.seed, com_mascara=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dl_tr = DataLoader(ds_tr, a.lote, shuffle=True, num_workers=0, drop_last=len(ds_tr) > a.lote)
    dl_va = DataLoader(ds_va, 1, shuffle=False)
    dev = dispositivo()
    net = construir(a.modelo).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.passos,
                                              pct_start=0.1)

    def validar():
        net.eval(); e = []
        with torch.no_grad():
            for x, y, w, _ in dl_va:
                yh, _ = soft_argmax_col(net(x.to(dev))[0])
                m = w.numpy() > 0.2
                e.append((np.abs(yh.cpu().numpy() - y.numpy()) * esc)[m])
        net.train()
        e = np.concatenate(e)
        return float(np.median(e)), float(np.mean(e))

    print(f"[{a.nome}] {len(tr)} treino / {len(VAL)} val | rótulos {a.rotulos} | "
          f"{a.modelo} {sum(p.numel() for p in net.parameters())/1e6:.2f} M | {dev}", flush=True)
    hist, melhor, passo, t0 = [], float("inf"), 0, time.time()
    net.train()
    while passo < a.passos:
        for x, y, w, _ in dl_tr:
            logits, _ = net(x.to(dev))
            loss = surface_loss(logits, y.to(dev), w.to(dev), sigma=a.sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); passo += 1
            if passo % a.cada == 0 or passo == a.passos:
                med, mean = validar()
                hist.append(dict(passo=passo, perda=float(loss.detach()), val_med_px=med,
                                 val_media_px=mean, s=time.time() - t0))
                if passo % (a.cada * 4) == 0 or passo == a.passos:
                    print(f"  passo {passo:5d} perda {float(loss.detach()):8.3f} | val mediana "
                          f"{med:6.2f} px média {mean:6.2f} ({time.time()-t0:.0f}s)", flush=True)
                if med < melhor:
                    melhor = med
                    torch.save(net.state_dict(), run / "best.pt")
                    save_json(run / "best.json", hist[-1])
            if passo >= a.passos:
                break
    save_json(run / "hist.json", hist)
    save_json(run / "config.json", dict(nome=a.nome, rotulos=a.rotulos, quadros=a.quadros,
                                        modelo=a.modelo, seed=a.seed, passos=a.passos,
                                        lote=a.lote, lr=a.lr, sigma=a.sigma, size=list(size),
                                        dispositivo=dev, train_ids=tr, val_ids=VAL,
                                        checkpoint_inicial=None,
                                        pre_treino=("ImageNet (timm resnet18.a1_in1k)"
                                                    if a.modelo == "r18" else None)))
    print(f"[{a.nome}] melhor val mediana {melhor:.2f} px", flush=True)


# --------------------------------------------------------------------------- #
# Inferência e decodificação
# --------------------------------------------------------------------------- #
def decodificar(S: np.ndarray, yc0: np.ndarray, flare: np.ndarray, dmax: int = 2,
                lam: float = 0.5, raio: float = 40.0, peso: float = 1.5,
                raio2: float = 12.0, peso2: float = 2.5, grau: int = 4,
                jan_x: int = 11) -> np.ndarray:
    """As três passadas de `detectar_interfaces.detectar` para a face de baixo,
    com o mapa de evidência S (N, H, nb) como entrada. Mesmos parâmetros."""
    from detectar_interfaces import (ajuste_robusto_x, colunas_confiaveis,
                                     corredor_no_escore, refinar, suavizar_x, viterbi)
    N, H, nb = S.shape
    pq = np.where(flare, 0.2, 1.0)[:, None]
    Sc = {j: S[:, :, j] * pq for j in range(nb)}
    pr = {j: proibido_baixo(H, yc0[j]) for j in range(nb)}
    Y = np.zeros((N, nb))
    for j in range(nb):
        Y[:, j] = viterbi(Sc[j], dmax, lam, pr[j], +1)
    for etapa, (ref_f, r_, p_) in enumerate((
            (lambda Y: ajuste_robusto_x(Y, grau, usar=colunas_confiaveis(Y)), raio, peso),
            (lambda Y: suavizar_x(Y, jan_x), raio2, peso2))):
        ref = ref_f(Y)
        for j in range(nb):
            S2 = corredor_no_escore(Sc[j], ref[:, j], r_, p_)
            y = viterbi(S2, dmax, lam, pr[j], +1)
            Y[:, j] = refinar(S2, y, mono=+1) if etapa else y
    return Y


def escore_classico(cubo) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    from detectar_interfaces import escore
    G = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in cubo]).astype(np.float32)
    return np.stack([escore(gaussian_filter1d(G[:, :, j], 2.0, axis=1))
                     for j in range(G.shape[2])], -1)


def conferir_decodificador(_a=None) -> None:
    """A réplica tem de devolver a saída do clássico de 09/10 com o escore dele."""
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy")
    cl = np.load(E1 / "classico.npz")
    Y = decodificar(escore_classico(cubo), cl["yc0"], cl["flare"])
    d = np.abs(Y - cl["y_inf"])
    print(f"réplica do decodificador: |dif| máx {d.max():.4f} px, média {d.mean():.5f} px")
    save_json(OUT / "conferencia_decodificador.json",
              dict(dif_max_px=float(d.max()), dif_media_px=float(d.mean())))


def inferir(a) -> None:
    import torch
    import torch.nn.functional as F
    from chamacera.surfnet import soft_argmax_col

    if a.nome == "R0_codex":
        run, mod, size = E1 / "run_bloco/model_surf", "surfnet", (592, 96)
        dst = RUNS / "R0_codex"; dst.mkdir(parents=True, exist_ok=True)
    else:
        run = dst = RUNS / a.nome
        cfg = json.loads((run / "config.json").read_text())
        mod, size = cfg["modelo"], tuple(cfg["size"])
    dev = dispositivo()
    net = construir(mod).to(dev)
    net.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    net.eval()
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    N, H, nb = cubo.shape[:3]
    oh, ow = size
    cl = np.load(E1 / "classico.npz")
    # Região geometricamente permitida para a face de baixo, a MESMA que o
    # decodificador do clássico usa. A evidência é normalizada só dentro dela:
    # a rede põe um pico estreito nas últimas linhas da imagem (efeito de
    # borda, fora de qualquer alvo de treino), e normalizar pelo máximo da
    # coluna inteira reduzia a superfície verdadeira a ~0,07 do pico. Com
    # evidência tão fraca, a penalidade por passo vence e o Viterbi fica
    # parado (medido: taxa zero em R0 e R1 na primeira execução).
    permitido = np.stack([~proibido_baixo(H, c) for c in cl["yc0"]], -1)   # (H, nb)
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
    viterbi_y = decodificar(S, cl["yc0"], cl["flare"])
    # Segunda forma de evidência: só a posição quadro a quadro da rede, como
    # uma gaussiana de 6 px (a meia largura do vale fino do clássico). Não
    # depende de a distribuição estar concentrada na superfície: redes
    # treinadas com poucos quadros acertam a média com uma distribuição quase
    # plana mais um acúmulo nas últimas linhas, e aí a forma (1) não informa.
    yy = np.arange(H, dtype=np.float32)[None, :, None]
    S_pos = 1.5 * np.exp(-0.5 * ((yy - bruto[:, None, :]) / 6.0) ** 2) * permitido
    del S
    viterbi_pos = decodificar(S_pos, cl["yc0"], cl["flare"])
    np.savez_compressed(dst / "saida.npz", bruto=bruto, viterbi=viterbi_y,
                        viterbi_pos=viterbi_pos)
    save_json(dst / "tempo.json", dict(inferencia_s=t_inf, decodificacao_s=time.time() - t0,
                                       n_quadros=int(N), dispositivo=dev))
    print(f"[{a.nome}] inferência {t_inf:.1f} s, decodificação {time.time()-t0:.1f} s", flush=True)


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
DESCR = {
    "R0_codex": "SurfaceNet de 09/10 (rótulos antigos, 16 quadros, 240 passos)",
    "R1": "rótulos antigos, 16 quadros, treino completo",
    "R2": "rótulos corrigidos, 16 quadros",
    "R3": "rótulos corrigidos, quadros densos, SurfaceNet do zero",
    "R4": "rótulos corrigidos, quadros densos, U-Net ResNet18 ImageNet",
}


def avaliar(_a=None) -> None:
    from experimento_prioridades12 import rate
    p = json.loads((E1 / "protocolo.json").read_text())
    ref = np.load(E1 / "referencia.npz")
    ks, cols, yref = ref["k"], ref["cols"], ref["y"]
    t = ks * p["dq"] / p["fps"]
    esc = p["scale_mm_px"]
    rr = rate(t, yref, esc)
    # Marcações manuais do usuário dentro do bloco: referência humana direta.
    # As ferramentas de marcação gravam x em pixel do QUADRO; as colunas e a
    # referência de 09/10 estão em pixel da ROI, que começa em x0 = roi[2].
    x0 = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())["roi"][2]
    xall = 20 * (np.arange(93) + 0.5)

    def ler_marcas(csv_path):
        return [(int(r["k"]), float(r["x_px"]) - x0, float(r["y_roi_px"]))
                for r in csv.DictReader(Path(csv_path).open())
                if r["face"] == "baixo" and EMBARGO[0] <= int(r["k"]) <= EMBARGO[1]]

    hum = ler_marcas(RAIZ / "resultados/revisao_humana/perfis/"
                     "marcacoes-manuais-corrigidas-aadc04d2/guia.csv")
    cego_csv = RAIZ / "referencia_humana_cega/rotulos_cegos.csv"
    cego = ler_marcas(cego_csv) if cego_csv.exists() else []

    def erro_marcas(Yfull, marcas):
        if not marcas:
            return float("nan"), float("nan")
        e = np.array([np.interp(x, xall, Yfull[k]) - yy for k, x, yy in marcas])
        return float(np.abs(e).mean()), float(e.mean())

    def metr(Yfull):
        y = Yfull[ks][:, cols]
        e = y - yref
        r = rate(t, y, esc)
        mh, vh = erro_marcas(Yfull, hum)
        mc, vc = erro_marcas(Yfull, cego)
        return dict(mae_px=float(np.abs(e).mean()), p95_px=float(np.percentile(np.abs(e), 95)),
                    vies_px=float(e.mean()), taxa_mm_s=r["rate_ols_mm_s"],
                    taxa_extremos_mm_s=r["rate_endpoints_mm_s"],
                    erro_taxa_pct=100 * (r["rate_ols_mm_s"] - rr["rate_ols_mm_s"]) / rr["rate_ols_mm_s"],
                    mae_usuario_px=mh, vies_usuario_px=vh,
                    mae_cego_px=mc, vies_cego_px=vc)

    cl = np.load(E1 / "classico.npz")
    anc = np.load(ROT / "interfaces_revisado.npz")["y_inf"]
    linhas = {"classico": dict(metodo="Detector clássico (vale + Viterbi)", **metr(cl["y_inf"])),
              "semi_automatico": dict(metodo="Clássico ancorado em marcações do usuário fora do bloco",
                                      **metr(anc))}
    for d in sorted(RUNS.iterdir()):
        if not (d / "saida.npz").exists():
            continue
        z = np.load(d / "saida.npz")
        base = d.name.split("_s")[0] if d.name != "R0_codex" else "R0_codex"
        for modo in ("bruto", "viterbi", "viterbi_pos"):
            if modo not in z.files:
                continue
            linhas[f"{d.name}/{modo}"] = dict(metodo=f"{DESCR.get(base, base)} | {modo}",
                                              grupo=base, modo=modo, **metr(z[modo]))
    # média e desvio entre sementes
    grupos = {}
    for nome, m in linhas.items():
        if "grupo" in m:
            grupos.setdefault((m["grupo"], m["modo"]), []).append(m)
    resumo = {}
    for (g, modo), ms in grupos.items():
        resumo[f"{g}/{modo}"] = dict(n_sementes=len(ms), **{
            c: [float(np.mean([m[c] for m in ms])), float(np.std([m[c] for m in ms]))]
            for c in ("mae_px", "p95_px", "vies_px", "taxa_mm_s", "erro_taxa_pct", "mae_usuario_px")})
    save_json(OUT / "metricas.json", dict(referencia=dict(taxa_mm_s=rr["rate_ols_mm_s"],
                                                          taxa_extremos_mm_s=rr["rate_endpoints_mm_s"],
                                                          n_quadros=len(ks), n_colunas=len(cols),
                                                          n_marcas_usuario_no_bloco=len(hum)),
                                          por_rodada=linhas, por_grupo=resumo))
    campos = ["mae_px", "p95_px", "vies_px", "taxa_mm_s", "erro_taxa_pct", "mae_usuario_px", "vies_usuario_px"]
    with (OUT / "comparacao.csv").open("w") as f:
        w = csv.writer(f); w.writerow(["rodada", "metodo"] + campos)
        for nome, m in linhas.items():
            w.writerow([nome, m["metodo"]] + [f"{m[c]:.5g}" for c in campos])
    print(f"referência: {rr['rate_ols_mm_s']:.5f} mm/s")
    print(f"{'rodada':22s}{'MAE':>7s}{'P95':>7s}{'viés':>7s}{'taxa':>9s}{'Δtaxa%':>8s}{'MAEusu':>8s}")
    for nome, m in linhas.items():
        print(f"{nome:22s}{m['mae_px']:7.2f}{m['p95_px']:7.1f}{m['vies_px']:7.2f}"
              f"{m['taxa_mm_s']:9.5f}{m['erro_taxa_pct']:8.1f}{m['mae_usuario_px']:8.2f}")
    print("\npor grupo (média ± desvio entre sementes):")
    for g, m in sorted(resumo.items()):
        print(f"  {g:14s} n={m['n_sementes']} MAE {m['mae_px'][0]:.2f}±{m['mae_px'][1]:.2f} "
              f"Δtaxa {m['erro_taxa_pct'][0]:+.1f}±{m['erro_taxa_pct'][1]:.1f}% "
              f"MAEusu {m['mae_usuario_px'][0]:.2f}")
    figura(linhas, t, yref, ks, cols)


def auditar(_a=None) -> None:
    """Quem está certo entre as referências: compara cada uma com as marcações
    manuais do usuário dentro do bloco (e com as cegas, quando existirem).

    A referência de 09/10 e os rótulos legados só existem em alguns quadros;
    entre eles, interpolação linear no tempo (a superfície anda ~1 px/s).
    """
    x0 = json.loads((RAIZ / "_tmp/cubo_meta.json").read_text())["roi"][2]
    xall = 20 * (np.arange(93) + 0.5)
    ref = np.load(E1 / "referencia.npz")
    rk, ry, xr = ref["k"], ref["y"], 20 * (ref["cols"] + 0.5)
    leg = np.load(RAIZ / "resultados/rotulos_humanos_face_inferior_v1.npz")
    cl = np.load(E1 / "classico.npz")["y_inf"]
    anc = np.load(ROT / "interfaces_revisado.npz")["y_inf"]

    def no_tempo(ks_, Y, xs, k, x):
        return float(np.interp(k, ks_, [np.interp(x, xs, row) for row in Y]))

    fontes = {
        "referencia_0910": lambda k, x: (no_tempo(rk, ry, xr, k, x)
                                         if rk[0] <= k <= rk[-1] else np.nan),
        "rotulos_legados": lambda k, x: no_tempo(leg["k"], leg["y"][:, 1], xall, k, x),
        "classico": lambda k, x: float(np.interp(x, xall, cl[k])),
        "semi_automatico": lambda k, x: float(np.interp(x, xall, anc[k])),
    }
    saida = {}
    for nome_m, arq in (("marcacoes_interface",
                         RAIZ / "resultados/revisao_humana/perfis/marcacoes-manuais-corrigidas-aadc04d2/guia.csv"),
                        ("marcacoes_cegas", RAIZ / "referencia_humana_cega/rotulos_cegos.csv")):
        if not arq.exists():
            continue
        pts = [(int(r["k"]), float(r["x_px"]) - x0, float(r["y_roi_px"]))
               for r in csv.DictReader(arq.open())
               if r["face"] == "baixo" and EMBARGO[0] <= int(r["k"]) <= EMBARGO[1]]
        saida[nome_m] = {"n_pontos": len(pts), "quadros": sorted({p[0] for p in pts})}
        for nome_f, f in fontes.items():
            e = np.array([f(k, x) - y for k, x, y in pts])
            e = e[np.isfinite(e)]
            saida[nome_m][nome_f] = dict(n=int(len(e)), mae_px=float(np.abs(e).mean()),
                                         vies_px=float(e.mean()),
                                         mediana_abs_px=float(np.median(np.abs(e))))
    save_json(OUT / "auditoria_referencias.json", saida)
    for nome_m, d in saida.items():
        print(f"{nome_m}: {d['n_pontos']} pontos em k={d['quadros']}")
        for nome_f in fontes:
            m = d[nome_f]
            print(f"   {nome_f:18s} n={m['n']:3d}  MAE {m['mae_px']:5.2f}  viés {m['vies_px']:+6.2f}")


def figura(linhas, t, yref, ks, cols) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Paleta categórica validada (validate_palette.js, modo claro): azul, laranja,
    # verde-água nas ordens 1-3; cinza neutro para os métodos sem rede. O
    # verde-água fica abaixo de 3:1 de contraste, então toda barra leva o valor.
    TXT, TXT2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
    MODOS = [("bruto", "Quadro a quadro", "#2a78d6"),
             ("viterbi", "Viterbi, evidência = distribuição", "#eb6834"),
             ("viterbi_pos", "Viterbi, evidência = posição", "#1baf7a")]
    grupos = [("classico", "Clássico (vale + Viterbi)"),
              ("semi_automatico", "Clássico ancorado nas\nmarcações fora do bloco"),
              ("R0_codex", "R0: rede de 09/10"),
              ("R1", "R1: rótulos antigos,\n16 quadros"),
              ("R2", "R2: rótulos corrigidos,\n16 quadros"),
              ("R3", "R3: corrigidos, 235 quadros,\nSurfaceNet do zero"),
              ("R4", "R4: corrigidos, 235 quadros,\nResNet18 ImageNet")]
    porg = {}
    for nome, m in linhas.items():
        if "grupo" in m:
            porg.setdefault((m["grupo"], m["modo"]), []).append(m)
    metricas = [("mae_px", "Erro médio do contorno\ncontra a referência de 09/10 (px)", 40),
                ("mae_usuario_px", "Erro médio contra as 24 marcações\nmanuais do usuário (px)", 40),
                ("erro_taxa_pct", "Erro da taxa frente à\nreferência de 09/10 (%)", 50)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 8.2), layout="constrained", sharey=True)
    h = 0.26
    for ax, (chave, titulo, corte) in zip(axs, metricas):
        for gi, (g, rotulo) in enumerate(grupos):
            y0 = len(grupos) - 1 - gi
            if g in linhas:
                serie = [(linhas[g][chave], 0.0, TXT2, 0.0)]
            else:
                serie = []
                for mi, (modo, _, cor) in enumerate(MODOS):
                    ms = porg.get((g, modo))
                    if ms:
                        v = [m[chave] for m in ms]
                        serie.append((float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0,
                                      cor, (1 - mi) * h))
            for v, sd, cor, dy in serie:
                vv = float(np.clip(v, -corte, corte))
                ax.barh(y0 + dy, vv, height=h * 0.86, color=cor, edgecolor="white", linewidth=1)
                txt = f"{v:+.1f}" if chave == "erro_taxa_pct" else f"{v:.1f}"
                lado = 1 if vv >= 0 else -1
                if abs(v) > corte:          # barra cortada: valor dentro dela, sem desvio
                    ax.text(vv - lado * corte * 0.03, y0 + dy, txt + " »", va="center",
                            ha="right" if lado > 0 else "left", fontsize=7.5,
                            color="white", fontweight="bold")
                    continue
                if abs(vv) + sd > corte:    # desvio maior que a escala: só no rótulo
                    txt, sd = f"{txt} ± {sd:.0f}", 0.0
                if sd > 0:
                    ax.errorbar(vv, y0 + dy, xerr=sd, color=TXT, lw=1, capsize=2)
                ax.text(vv + lado * (sd + corte * 0.02), y0 + dy, txt, va="center",
                        ha="left" if lado > 0 else "right", fontsize=7.5, color=TXT)
        ax.set_title(titulo, fontsize=10, color=TXT)
        ax.set_xlim((-corte * 1.15, corte * 1.15) if chave == "erro_taxa_pct" else (0, corte * 1.3))
        ax.axvline(0, color=TXT2, lw=0.8)
        ax.grid(axis="x", color=GRADE, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=TXT2, labelsize=8.5)
    axs[0].set_yticks(range(len(grupos)), [r for _, r in grupos][::-1], fontsize=8.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=TXT2, label="Método sem rede")] +
               [Patch(color=c, label=r) for _, r, c in MODOS],
               loc="outside upper center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Face inferior, bloco k=180 a 480 (80 s), 11 quadros x 73 colunas. "
                 "Barras: média de 3 sementes (R3, R4) com desvio; demais, 1 semente. » = valor fora da escala.",
                 fontsize=10, color=TXT2, y=-0.01)
    fig.savefig(OUT / "comparacao.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Série no tempo e contornos: a deriva dos rótulos antigos e a fuligem da cabeça.
    cl = np.load(E1 / "classico.npz")["y_inf"]
    escolha = [("R4: rótulos corrigidos, ResNet18, + Viterbi", RUNS / "R4_s0/saida.npz", "#2a78d6"),
               ("Detector clássico", cl, "#eb6834"),
               ("R2: rótulos corrigidos, 16 quadros", RUNS / "R2_s0/saida.npz", "#1baf7a"),
               ("R1: rótulos antigos, 16 quadros", RUNS / "R1_s0/saida.npz", "#eda100")]
    tq = (ks - IGNICAO_K) * E.CAM
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), layout="constrained",
                           gridspec_kw=dict(height_ratios=[1.2, 1]))
    ax[0].plot(tq, yref.mean(1), "ko-", lw=2.2, label="Referência de 09/10")
    curvas = {}
    for nome, fonte, c in escolha:
        if isinstance(fonte, Path):
            if not fonte.exists():
                continue
            z = np.load(fonte)
            Y = z["viterbi"] if "R4" in nome else z["bruto"]
        else:
            Y = fonte
        serie = Y[ks][:, cols].mean(1)
        ax[0].plot(tq, serie, "o-", color=c, lw=2, ms=4, label=nome)
        curvas[nome] = (Y, c, serie[-1])
    # rótulos diretos na ponta, afastados para não se sobreporem
    pontas = sorted(((v[2], n.split(":")[0].replace("Detector clássico", "Clássico"))
                     for n, v in curvas.items()), reverse=True)
    ultimo = None
    for yv, tag in pontas:
        yl = yv if ultimo is None else min(yv, ultimo - 4.5)
        ax[0].annotate(tag, (tq[-1], yv), xytext=(tq[-1] + 1.5, yl), va="center",
                       fontsize=8, color="#0b0b0b")
        ultimo = yl
    ax[0].set_xlim(tq[0] - 2, tq[-1] + 9)
    ax[0].set_xlabel("Tempo desde a ignição (s)")
    ax[0].set_ylabel("Posição média da face inferior (px)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    cubo = np.load(RAIZ / "_tmp/cubo_bgr.npy", mmap_mode="r")
    k = int(ks[-1]); i = len(ks) - 1
    im = cv2.resize(np.ascontiguousarray(cubo[k]), (1860, cubo.shape[1]))
    ax[1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    xc = 20 * (cols + 0.5)
    ax[1].plot(xc, yref[i], "w-", lw=2.4, label="Referência de 09/10 (branco)",
               path_effects=[__import__("matplotlib.patheffects").patheffects.withStroke(
                   linewidth=3.6, foreground="#0b0b0b")])
    for nome, (Y, c, _) in curvas.items():
        if "R1" in nome or "R2" in nome:
            continue
        ax[1].plot(xc, Y[k, cols], "-", color=c, lw=1.8, label=nome)
    ax[1].set_xlim(300, 1860); ax[1].set_ylim(593, 300)
    ax[1].set_title(f"k={k}, {tq[-1]:.0f} s após a ignição", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "series_e_contornos.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["rotulos", "treinar", "inferir", "avaliar", "auditar", "conferir_decodificador"])
    ap.add_argument("--nome")
    ap.add_argument("--rotulos", choices=["hibrido", "legado"], default="hibrido")
    ap.add_argument("--quadros", choices=["codex", "denso"], default="denso")
    ap.add_argument("--modelo", choices=["surfnet", "r18"], default="surfnet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passos", type=int, default=1200)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--cada", type=int, default=50)
    a = ap.parse_args()
    globals()[a.etapa](a)
