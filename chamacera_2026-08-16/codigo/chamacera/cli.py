"""
Interface de linha de comando.

Fluxo completo (uma volta do laço):

    chamacera bootstrap  ensaio.mp4 -o run1   # rótulos-semente, 0 cliques
    chamacera teach      run1                 # opcional: refina com SAM
    chamacera train      run1                 # destila no aluno U-Net
    chamacera infer      run1 --all           # roda no vídeo inteiro + incerteza
    chamacera select     run1 -k 20           # escolhe os frames que ensinam
    #  ... revisar run1/review/mask/*.png à mão (minutos) ...
    chamacera apply-review run1
    chamacera train      run1 --resume        # volta 2, agora com dado humano
    chamacera rate       run1                 # r-dot, métricas de chama, gráficos
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from . import CERA, CHAMA
from .io_video import Calib, VideoSource, detect_burn_window, sample_indices


def _paths(run):
    return dict(run=run, ds=os.path.join(run, "dataset"),
                model=os.path.join(run, "model"),
                review=os.path.join(run, "review"),
                out=os.path.join(run, "out"))


def _load_src(run):
    st = json.load(open(os.path.join(run, "state.json")))
    return VideoSource(st["video"], Calib(**st["calib"])), st


def _save_ds(ds_dir, src, indices, masks, confs):
    for sub in ("img", "mask", "conf"):
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)
    for k, i in enumerate(indices):
        fid = f"{i:06d}"
        cv2.imwrite(os.path.join(ds_dir, "img", f"{fid}.png"), src.frame(i))
        cv2.imwrite(os.path.join(ds_dir, "mask", f"{fid}.png"), masks[k])
        cv2.imwrite(os.path.join(ds_dir, "conf", f"{fid}.png"),
                    (np.clip(confs[k], 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
def cmd_bootstrap(a):
    from .bootstrap import bootstrap_sequence
    p = _paths(a.run)
    os.makedirs(a.run, exist_ok=True)

    calib = Calib(fps=a.fps, mm_per_px=a.mm_per_px,
                  roi=tuple(a.roi) if a.roi else None, nome=os.path.basename(a.video))
    src = VideoSource(a.video, calib)
    if a.auto_window:
        t0, t1 = detect_burn_window(src)
        src.calib.t0_frame, src.calib.t1_frame = t0, t1
        print(f"[janela] ignição no frame {t0}, fim em {t1} "
              f"({(t1-t0)/src.calib.fps:.2f} s de queima)")

    rng = list(src.burn_range)
    idx = sample_indices(len(src), a.n, rng[0], rng[-1] + 1)
    print(f"[bootstrap] {len(idx)} frames de {len(rng)} | ROI {src.shape}")

    masks, confs, S = bootstrap_sequence(
        src, idx, win=a.win, K=a.K, lam=a.lam, flame_level=a.flame_level,
        perfil=a.perfil, downsample=a.downsample, band=tuple(a.band),
        enforce_monotonic=not a.no_monotonic)
    _save_ds(p["ds"], src, idx, masks, confs)
    np.save(os.path.join(a.run, "surface_bootstrap.npy"), S)

    json.dump(dict(video=os.path.abspath(a.video),
                   calib={k: v for k, v in src.calib.__dict__.items()},
                   indices=[int(i) for i in idx], stage="bootstrap"),
              open(os.path.join(a.run, "state.json"), "w"), indent=2)

    fr = {c: float((masks == c).mean()) for c in (CHAMA, CERA)}
    print(f"[bootstrap] fração de pixels — chama {fr[CHAMA]:.3f} cera {fr[CERA]:.3f}")
    print(f"[bootstrap] dataset em {p['ds']}  (img/ mask/ conf/)")
    print("           inspecione run/review/overlay depois de `select`, "
          "ou abra mask/*.png sobre img/*.png")


def cmd_teach(a):
    from .teacher import SamTeacher, anchors_from_bootstrap
    p = _paths(a.run)
    src, st = _load_src(a.run)
    idx = st["indices"]
    masks = np.stack([cv2.imread(os.path.join(p["ds"], "mask", f"{i:06d}.png"), 0)
                      for i in idx])
    confs = np.stack([cv2.imread(os.path.join(p["ds"], "conf", f"{i:06d}.png"), 0)
                      for i in idx]).astype(np.float32) / 255.0

    t = SamTeacher(backend=a.backend, ckpt=a.ckpt)
    if not t.available:
        print("[teach] nenhum backend SAM disponível — seguindo só com o bootstrap.")
        print(t.info.strip())
        return
    print(f"[teach] backend = {t.backend} ({t.device})")

    prompts = anchors_from_bootstrap(masks, confs, a.anchors)
    new = t.propagate_sam2(src, idx, prompts)

    inter = sum(((new == c) & (masks == c)).sum() for c in (CHAMA, CERA))
    union = sum(((new == c) | (masks == c)).sum() for c in (CHAMA, CERA))
    print(f"[teach] concordância SAM vs bootstrap: IoU = {inter/max(union,1):.4f}")

    np.save(os.path.join(a.run, "masks_teacher.npy"), new)
    if a.overwrite:
        _save_ds(p["ds"], src, idx, new, np.clip(confs + 0.15, 0, 1))
        print("[teach] pseudo-rótulos substituídos pelos do professor")
    else:
        print("[teach] salvo em masks_teacher.npy (use --overwrite para adotar)")


def cmd_train(a):
    from .train import train
    p = _paths(a.run)
    best, _ = train(p["ds"], p["model"], epochs=a.epochs, batch=a.batch,
                    size=tuple(a.size), lr=a.lr, base=a.base, depth=a.depth,
                    p_drop=a.dropout, seed=a.seed, num_workers=a.workers,
                    device=a.device,
                    resume=os.path.join(p["model"], "best.pt") if a.resume else None)
    print(f"[train] mIoU de validação = {best:.4f}")


def cmd_infer(a):
    from .predict import load_model, predict_frames
    from .surface import surface_from_masks
    p = _paths(a.run)
    src, st = _load_src(a.run)
    os.makedirs(p["out"], exist_ok=True)

    rng = list(src.burn_range)
    idx = (list(range(rng[0], rng[-1] + 1, a.stride)) if a.all else st["indices"])
    net, dev, cfg = load_model(os.path.join(p["model"], "best.pt"), a.device)
    print(f"[infer] {len(idx)} frames | {dev} | T={a.mc} passagens MC")

    masks, ent, epi = predict_frames(
        net, dev, (src.frame(i) for i in idx), T=a.mc,
        size=tuple(cfg.get("size", (256, 512))), batch=a.batch)

    np.save(os.path.join(p["out"], "masks.npy"), masks)
    np.save(os.path.join(p["out"], "epi.npy"), epi.astype(np.float16))
    np.save(os.path.join(p["out"], "surface.npy"), surface_from_masks(masks))
    json.dump([int(i) for i in idx], open(os.path.join(p["out"], "indices.json"), "w"))
    print(f"[infer] incerteza epistêmica média = {epi.mean():.4f} "
          f"(p95 por frame = {np.quantile(epi.reshape(len(epi),-1).mean(1),0.95):.4f})")
    print(f"[infer] salvo em {p['out']}")


def cmd_select(a):
    from .active import acquisition, descriptors, export_review, select_diverse
    from .theory import PowerLawPrior
    p = _paths(a.run)
    src, _ = _load_src(a.run)
    idx = json.load(open(os.path.join(p["out"], "indices.json")))
    masks = np.load(os.path.join(p["out"], "masks.npy"))
    epi = np.load(os.path.join(p["out"], "epi.npy")).astype(np.float32)
    S = np.load(os.path.join(p["out"], "surface.npy"))

    tm = None
    tp = os.path.join(a.run, "masks_teacher.npy")
    if os.path.exists(tp) and len(np.load(tp)) == len(masks):
        tm = np.load(tp)

    teoria = {}
    if a.theory:
        d = json.load(open(a.theory))
        try:
            prior = PowerLawPrior(a=float(d["a_mm_s_por_Gn"]), n=float(d["n"]),
                                  sigma_rel=float(d["sigma_rel"]))
            G = np.asarray(d["G_ox_kg_m2_s"], float)
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit("arquivo --theory inválido; veja theory.example.json") from e
        if G.ndim == 0:
            G = np.full(len(idx), float(G))
        if G.shape != (len(idx),):
            raise SystemExit("G_ox_kg_m2_s deve ser escalar ou ter uma entrada por frame inferido")
        teoria = dict(fps=src.calib.fps, mm_per_px=src.calib.mm_per_px,
                      theory_rate=prior.rate_mm_s(G), theory_sigma=prior.sigma_mm_s(G))
        print("[select] prior teórico ativo: restrição fraca para PRIORIZAR revisão; "
              "não é usado como rótulo nem altera a máscara")

    parts = acquisition(epi=epi, masks=masks, surfaces=S, teacher_masks=tm, **teoria)
    frames = [src.frame(i) for i in idx]
    chosen = select_diverse(parts["total"], descriptors(frames), a.k)
    itens = export_review(p["review"], frames, masks, idx, parts, chosen, S)

    print(f"[select] {len(itens)} frames para revisão -> {p['review']}")
    print(f"[select] violação física: {parts['fisica'].mean()*100:.2f}% das colunas em média, "
          f"pior frame {parts['fisica'].max()*100:.1f}% (superfície subiu = erro certo)")
    print("   frame   score  motivo dominante")
    for i in itens[:a.k]:
        print(f"  {i.frame:6d}  {i.score:+6.2f}  {i.motivo}")
    print(f"\n  Corrija {p['review']}/mask/*.png (0=fundo 1=chama 2=cera) "
          f"e rode: chamacera apply-review {a.run}")


def cmd_apply_review(a):
    from .active import apply_review
    p = _paths(a.run)
    n = apply_review(p["review"], p["ds"])
    print(f"[apply-review] {n} máscaras revisadas incorporadas com confiança 1.0")
    print(f"[apply-review] agora rode: chamacera train {a.run} --resume")


def cmd_rate(a):
    from .flame import flame_metrics
    from .surface import rdot_by_linear_fit, regression_rate, smooth_surface
    p = _paths(a.run)
    src, st = _load_src(a.run)
    os.makedirs(p["out"], exist_ok=True)
    cal = src.calib

    mp = os.path.join(p["out"], "masks.npy")
    if os.path.exists(mp):
        masks = np.load(mp)
        idx = json.load(open(os.path.join(p["out"], "indices.json")))
        S = np.load(os.path.join(p["out"], "surface.npy"))
        fonte = "aluno"
    else:
        idx = st["indices"]
        masks = np.stack([cv2.imread(os.path.join(p["ds"], "mask", f"{i:06d}.png"), 0)
                          for i in idx])
        S = np.load(os.path.join(a.run, "surface_bootstrap.npy"))
        fonte = "bootstrap"

    fps_eff = cal.fps / max(1, (idx[1] - idx[0]) if len(idx) > 1 else 1)
    rr = regression_rate(S, fps_eff, cal.mm_per_px, win_t=a.win_t, win_x=a.win_x)
    lin = rdot_by_linear_fit(S, fps_eff, cal.mm_per_px)
    fm = flame_metrics(masks, smooth_surface(S), fps_eff, cal.mm_per_px)

    print(f"[rate] fonte das máscaras: {fonte} | {len(idx)} frames @ {fps_eff:.1f} Hz efetivos")
    print(f"[rate] r-dot (Savitzky-Golay) = {rr.rdot_bar:.4f} mm/s "
          f"(dispersão axial {rr.rdot_std:.4f})")
    print(f"[rate] r-dot (ajuste linear)  = {lin['rdot_bar']:.4f} "
          f"± {lin['se_bar']:.4f} mm/s  (R² = {lin['r2']:.4f})")
    print(f"[rate] recuo total = {rr.total_mm:.3f} mm")
    print(f"[rate] chama: altura média {np.nanmean(fm.h_bar_mm):.2f} mm | "
          f"flicker {fm.flicker_hz:.1f} Hz ({fm.flicker_power*100:.0f}% da potência)")
    if fm.flicker_hz > 0.40 * fps_eff:
        print(f"  ATENÇÃO: flicker perto de Nyquist ({fps_eff/2:.0f} Hz). Refaça o "
              f"`infer --stride 1` — subamostrar aliasa a frequência da chama.\n"
              f"  (r-dot não sofre: é banda-baixa. Só a análise espectral exige "
              f"a taxa cheia da câmera.)")

    res = dict(fonte=fonte, n_frames=len(idx), fps_efetivo=fps_eff,
               mm_per_px=cal.mm_per_px,
               rdot_savgol_mm_s=rr.rdot_bar, rdot_disp_axial_mm_s=rr.rdot_std,
               rdot_linfit_mm_s=lin["rdot_bar"], rdot_linfit_se=lin["se_bar"],
               r2=lin["r2"], recuo_total_mm=rr.total_mm,
               altura_chama_mm=float(np.nanmean(fm.h_bar_mm)),
               standoff_mm=float(np.nanmean(fm.standoff_mm)),
               flicker_hz=fm.flicker_hz, flicker_power=fm.flicker_power)
    json.dump(res, open(os.path.join(p["out"], "resultados.json"), "w"), indent=2)

    np.savetxt(os.path.join(p["out"], "rdot_por_x.csv"),
               np.column_stack([rr.x_mm, rr.rdot_x, lin["rdot_x"], lin["se_x"]]),
               delimiter=",", header="x_mm,rdot_savgol_mm_s,rdot_linfit_mm_s,se_mm_s",
               comments="")
    np.savetxt(os.path.join(p["out"], "rdot_por_t.csv"),
               np.column_stack([rr.t, rr.rdot_t, fm.area_mm2[:len(rr.t)],
                                fm.h_bar_mm[:len(rr.t)]]),
               delimiter=",", header="t_s,rdot_mm_s,area_chama_mm2,altura_chama_mm",
               comments="")

    if a.plots:
        _plots(p["out"], rr, lin, fm, cal)
    print(f"[rate] resultados e CSVs em {p['out']}")


def _plots(out, rr, lin, fm, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax[0, 0].plot(rr.x_mm, rr.rdot_x, lw=1.6, label="Savitzky-Golay")
    ax[0, 0].plot(rr.x_mm, lin["rdot_x"], lw=1.2, ls="--", label="ajuste linear")
    ax[0, 0].fill_between(rr.x_mm, lin["rdot_x"] - 2 * lin["se_x"],
                          lin["rdot_x"] + 2 * lin["se_x"], alpha=0.2, lw=0)
    ax[0, 0].set(xlabel="x [mm]", ylabel=r"$\dot r$ [mm/s]",
                 title="Taxa de regressão axial")
    ax[0, 0].legend(frameon=False, fontsize=8)

    ax[0, 1].plot(rr.t, rr.rdot_t, lw=1.4)
    ax[0, 1].set(xlabel="t [s]", ylabel=r"$\dot r$ [mm/s]",
                 title="Taxa de regressão no tempo")

    im = ax[1, 0].pcolormesh(rr.x_mm, rr.t, rr.y_mm, shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax[1, 0], label="recuo [mm]")
    ax[1, 0].set(xlabel="x [mm]", ylabel="t [s]", title="Recuo da superfície")

    ax[1, 1].plot(rr.t[:len(fm.area_mm2)], fm.area_mm2[:len(rr.t)], lw=1.0)
    ax[1, 1].set(xlabel="t [s]", ylabel="área da chama [mm²]",
                 title=f"Chama — flicker dominante {fm.flicker_hz:.1f} Hz")
    for a_ in ax.ravel():
        a_.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(out, "resultados.png"), dpi=160)
    plt.close(fig)


def cmd_eval(a):
    """Valida contra verdade-terreno (ensaio sintético ou frames revisados)."""
    from .metrics import boundary_f1, seg_scores, surface_error
    p = _paths(a.run)
    _, st = _load_src(a.run)
    mp = os.path.join(p["out"], "masks.npy")
    if os.path.exists(mp):
        masks = np.load(mp)
        idx = json.load(open(os.path.join(p["out"], "indices.json")))
        S = np.load(os.path.join(p["out"], "surface.npy"))
    else:
        idx = st["indices"]
        masks = np.stack([cv2.imread(os.path.join(p["ds"], "mask", f"{i:06d}.png"), 0)
                          for i in idx])
        S = np.load(os.path.join(a.run, "surface_bootstrap.npy"))

    gt = np.stack([cv2.imread(os.path.join(a.gt, f"{i:05d}.png"), 0) for i in idx])
    sc = seg_scores(masks, gt)
    bf = float(np.mean([boundary_f1(masks[n], gt[n], CERA, a.tol)
                        for n in range(len(masks))]))
    print(f"acurácia {sc['acc']:.4f} | mIoU {sc['miou']:.4f} | "
          f"IoU chama {sc['iou_chama']:.4f} | IoU cera {sc['iou_cera']:.4f}")
    print(f"BF-score da interface (tol {a.tol} px) = {bf:.4f}")
    if a.gt_surface and os.path.exists(a.gt_surface):
        gs = np.load(a.gt_surface)[idx]
        print("superfície:", {k: round(v, 3) for k, v in surface_error(S, gs).items()})


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser("chamacera", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="rótulos-semente sem supervisão")
    b.add_argument("video"); b.add_argument("-o", "--run", required=True)
    b.add_argument("-n", type=int, default=120, help="frames a rotular")
    b.add_argument("--fps", type=float, default=240.0)
    b.add_argument("--mm-per-px", type=float, default=1.0)
    b.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    b.add_argument("--win", type=int, default=9, help="janela temporal (frames)")
    b.add_argument("--K", type=int, default=9, help="apoio do filtro casado")
    b.add_argument("--lam", type=float, default=0.15, help="rigidez da interface")
    b.add_argument("--flame-level", type=float, default=0.50,
                   help="isolinha da borda da chama (fração do pico)")
    b.add_argument("--perfil", default="fuligem", choices=["fuligem", "quartzo"],
                   help="perfil do ensaio; usar o errado zera a segmentação")
    b.add_argument("--downsample", type=int, default=1,
                   help="fator de redução axial da DP da interface")
    b.add_argument("--band", type=float, nargs=2, default=[0.15, 0.98],
                   metavar=("LO", "HI"), help="faixa vertical de busca da interface")
    b.add_argument("--no-monotonic", action="store_true",
                   help="não impor recessão monotônica (deixa a superfície crua)")
    b.add_argument("--auto-window", action="store_true", default=True)
    b.set_defaults(f=cmd_bootstrap)

    t = sub.add_parser("teach", help="refina/propaga com SAM (opcional)")
    t.add_argument("run"); t.add_argument("--backend", default="auto")
    t.add_argument("--ckpt"); t.add_argument("--anchors", type=int, default=4)
    t.add_argument("--overwrite", action="store_true")
    t.set_defaults(f=cmd_teach)

    tr = sub.add_parser("train", help="treina o aluno U-Net")
    tr.add_argument("run"); tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--size", type=int, nargs=2, default=[256, 512])
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--base", type=int, default=24)
    tr.add_argument("--depth", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.20)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--device", help="cpu, cuda, mps; padrão = cuda se houver")
    tr.add_argument("--resume", action="store_true")
    tr.set_defaults(f=cmd_train)

    i = sub.add_parser("infer", help="inferência + incerteza")
    i.add_argument("run"); i.add_argument("--all", action="store_true")
    i.add_argument("--stride", type=int, default=1)
    i.add_argument("--mc", type=int, default=12, help="passagens Monte-Carlo")
    i.add_argument("--batch", type=int, default=2); i.add_argument("--device")
    i.set_defaults(f=cmd_infer)

    s = sub.add_parser("select", help="seleção ativa de frames p/ revisão")
    s.add_argument("run"); s.add_argument("-k", type=int, default=20)
    s.add_argument("--theory", help="JSON explícito com lei r=aG^n e G_ox; opcional")
    s.set_defaults(f=cmd_select)

    r = sub.add_parser("apply-review", help="incorpora correções humanas")
    r.add_argument("run"); r.set_defaults(f=cmd_apply_review)

    q = sub.add_parser("rate", help="taxa de regressão + métricas de chama")
    q.add_argument("run"); q.add_argument("--win-t", type=int, default=21)
    q.add_argument("--win-x", type=int, default=15)
    q.add_argument("--plots", action="store_true", default=True)
    q.set_defaults(f=cmd_rate)

    e = sub.add_parser("eval", help="valida contra verdade-terreno")
    e.add_argument("run"); e.add_argument("--gt", required=True)
    e.add_argument("--gt-surface"); e.add_argument("--tol", type=int, default=2)
    e.set_defaults(f=cmd_eval)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
