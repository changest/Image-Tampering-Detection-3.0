# -*- coding: utf-8 -*-
"""COVERAGE 域适应实验：40 对训练 / 10 对选模 / 50 对测试。

先零样本评测，再从 last_ema 微调若干轮，逐轮报告测试集 AUC / mIoU。
用法：
  python -m fbfn.adapt_coverage --epochs 15 --lr 1e-5
"""
import argparse
import csv
import os

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from .dataset import imread_unicode
from .precompute import compute_ela, compute_entropy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COV = os.path.join(ROOT, "data", "COVERAGE")


def build_manifest(split_ids, out):
    rows = []
    for i in split_ids:
        for tag, lab, typ in (("", 0, 0), ("t", 1, 2)):
            p = os.path.join(COV, "image", f"{i}{tag}.tif")
            if not os.path.exists(p):
                continue
            mask = os.path.join(COV, "mask", f"{i}forged.tif") if lab else ""
            if mask and not os.path.exists(mask):
                mask = ""
            rows.append((p, lab, typ, mask, 1))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "type", "mask", "has_mask"])
        w.writerows(rows)
    return len(rows)


class CovDataset(Dataset):
    def __init__(self, manifest, size=448, train=True):
        self.rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = imread_unicode(r["path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        rgb = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        ela = compute_ela(rgb)
        ent = compute_entropy(rgb)
        if ent.shape[-1] != 31:
            ent = np.stack([cv2.resize(ent[c], (31, 31), interpolation=cv2.INTER_LINEAR)
                            for c in range(ent.shape[0])])
        mask = np.zeros((1, self.size, self.size), np.float32)
        if r["mask"]:
            m = imread_unicode(r["mask"], cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)[None]
        if self.train and np.random.rand() < 0.5:
            rgb = rgb[:, :, ::-1]; ela = ela[:, :, ::-1]; ent = ent[:, :, ::-1]; mask = mask[:, :, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(np.ascontiguousarray(ent)),
                torch.from_numpy(np.ascontiguousarray(ela)),
                torch.tensor(float(r["label"])),
                torch.tensor(int(r["type"]), dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(mask)))


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad()
    ys, zs, P, M = [], [], [], []
    for x, ent, ela, y, t, mask in loader:
        x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
        with amp:
            out = model(x, ela=ela, entropy=ent)
        zs.append(out["final"].float().cpu().numpy())
        ys.append(y.numpy())
        pm = torch.sigmoid(out["final_map"].float())[:, 0].cpu().numpy()
        P.append(pm); M.append(mask.numpy()[:, 0])
    ys = np.concatenate(ys); zs = np.concatenate(zs)
    probs = 1 / (1 + np.exp(-zs))
    auc = roc_auc_score(ys, probs)
    pred = (probs >= 0.5).astype(int)
    P = np.concatenate(P); M = np.concatenate(M)
    best = 0.0
    for th in np.arange(0.2, 0.86, 0.05):
        pm = P > th
        tp = (pm * M).sum(); fp = (pm * (1 - M)).sum(); fn = ((1 - pm) * M).sum()
        best = max(best, tp / (tp + fp + fn + 1e-7))
    return float(auc), float(accuracy_score(ys, pred)), float(f1_score(ys, pred)), float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "last_ema.pth"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--amp", choices=["fp32","bf16"], default="bf16")
    ap.add_argument("--freeze", action="store_true", help="冻结分支，仅训练融合与头")
    ap.add_argument("--out", default=os.path.join(HERE, "runs_type", "cov_adapted.pth"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_m = os.path.join(HERE, "_cov_train.csv")
    va_m = os.path.join(HERE, "_cov_val.csv")
    te_m = os.path.join(HERE, "_cov_test.csv")
    print("train", build_manifest(range(1, 41), tr_m),
          "val", build_manifest(range(41, 51), va_m),
          "test", build_manifest(range(51, 101), te_m))

    tr = CovDataset(tr_m, train=True); va = CovDataset(va_m, train=False); te = CovDataset(te_m, train=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=2)
    el = DataLoader(te, batch_size=4, shuffle=False, num_workers=2)

    from .type_model_large import FBFNTypeLarge
    model = FBFNTypeLarge(pretrained=False, img_size=448).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    if args.freeze:
        for n, p in model.named_parameters():
            if not (n.startswith("fusion") or n.startswith("type_trunk")
                    or n.startswith("type_fc") or n.startswith("bin_fc")):
                p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    print("zero-shot test:", evaluate(model, el, dev), flush=True)
    best = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        opt.zero_grad()
        for i, (x, ent, ela, y, t, mask) in enumerate(tl):
            x, ent, ela, y, t, mask = x.to(dev), ent.to(dev), ela.to(dev), y.to(dev), t.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, ela=ela, entropy=ent)
                loss = (0.5 * F.binary_cross_entropy_with_logits(out["final"].float(), y)
                        + 0.5 * ce(out["type"].float(), t)
                        + F.binary_cross_entropy_with_logits(out["final_map"].float(), mask)
                        + dice_loss(out["final_map"].float(), mask))
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(tl):
                opt.step(); opt.zero_grad()
            run += loss.item()
        m_va = evaluate(model, vl, dev)
        m_te = evaluate(model, el, dev)
        print(f"ep{ep} loss={run/len(tl):.3f} val AUC/F1/mIoU={m_va} test AUC/F1/mIoU={m_te}", flush=True)
        if m_va[0] > best:
            best = m_va[0]
            torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
