"""Treino do aluno sobre pseudo-rótulos (+ correções humanas quando existirem)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SlabDataset, class_weights
from .metrics import confusion, from_confusion
from .models import UNet, confidence_weighted_ce, dice_loss


def split_ids(ids, val_frac=0.2, seed=0, contiguous=True):
    """
    Divisão treino/validação. `contiguous=True` separa por *bloco temporal*, não
    aleatoriamente: frames vizinhos de câmera rápida são quase idênticos, e uma
    divisão aleatória vaza informação e infla a métrica de validação. Este é o
    erro metodológico mais comum em trabalhos que usam vídeo de combustão.
    """
    ids = list(ids)
    n_val = max(1, int(round(val_frac * len(ids))))
    if contiguous:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, max(1, len(ids) - n_val)))
        val = ids[start:start + n_val]
        return [i for i in ids if i not in set(val)], val
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    val = [ids[i] for i in perm[:n_val]]
    return [ids[i] for i in perm[n_val:]], val


def train(root: str, out: str, epochs: int = 60, batch: int = 4,
          size=(256, 512), lr: float = 3e-4, base: int = 24, depth: int = 4,
          p_drop: float = 0.20, val_frac: float = 0.2, seed: int = 0,
          device: str | None = None, num_workers: int = 2,
          w_dice: float = 0.5, log_every: int = 5, resume: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)

    all_ids = SlabDataset(root, size, train=False, seed=seed).ids
    tr_ids, va_ids = split_ids(all_ids, val_frac, seed)
    ds_tr = SlabDataset(root, size, True, seed, tr_ids)
    ds_va = SlabDataset(root, size, False, seed, va_ids)
    dl_tr = DataLoader(ds_tr, batch, shuffle=True, num_workers=num_workers,
                       drop_last=len(ds_tr) > batch)
    dl_va = DataLoader(ds_va, 1, shuffle=False, num_workers=0)

    cw = class_weights(root, tr_ids).to(device)
    net = UNet(3, base=base, depth=depth, p_drop=p_drop).to(device)
    if resume and os.path.exists(resume):
        net.load_state_dict(torch.load(resume, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    hist, best = [], -1.0
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[treino] {len(tr_ids)} treino / {len(va_ids)} val | {n_par/1e6:.2f} M par | {device}")
    print(f"[treino] pesos de classe = {cw.cpu().numpy().round(2)}")

    for ep in range(1, epochs + 1):
        net.train(); tl, nb, t0 = 0.0, 0, time.time()
        for x, y, c in dl_tr:
            x, y, c = x.to(device), y.to(device), c.to(device)
            logits = net(x)
            loss = confidence_weighted_ce(logits, y, c, cw) + w_dice * dice_loss(logits, y, c)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl += loss.detach().item(); nb += 1
        sched.step()

        net.eval(); cm = np.zeros((3, 3), np.int64)
        with torch.no_grad():
            for x, y, _ in dl_va:
                pred = net(x.to(device)).argmax(1).cpu().numpy()
                cm += confusion(pred, y.numpy())
        sc = from_confusion(cm)
        rec = dict(epoch=ep, loss=tl / max(nb, 1), lr=sched.get_last_lr()[0],
                   secs=time.time() - t0, **sc)
        hist.append(rec)
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(f"  ep {ep:3d} loss {rec['loss']:.4f} | mIoU {sc['miou']:.4f} "
                  f"chama {sc['iou_chama']:.4f} cera {sc['iou_cera']:.4f} "
                  f"({rec['secs']:.1f}s)")
        if sc["miou"] > best:
            best = sc["miou"]
            torch.save(net.state_dict(), os.path.join(out, "best.pt"))
            with open(os.path.join(out, "best.json"), "w") as f:
                json.dump(rec, f, indent=2)

    torch.save(net.state_dict(), os.path.join(out, "last.pt"))
    with open(os.path.join(out, "hist.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(root=root, size=list(size), base=base, depth=depth,
                       p_drop=p_drop, epochs=epochs, lr=lr, seed=seed,
                       train_ids=tr_ids, val_ids=va_ids), f, indent=2)
    print(f"[treino] melhor mIoU = {best:.4f} -> {out}/best.pt")
    return best, hist
