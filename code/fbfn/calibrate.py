# -*- coding: utf-8 -*-
"""在真实 CASIA 验证集上做二分类阈值搜索与温度缩放校准。

AUC 与温度无关（单调变换），本脚本用于：
  1. 搜索使二分类准确率/F1 最优的判定阈值（默认 0.5 往往不是最优）
  2. 拟合温度 T 改善概率校准（ECE），便于对外输出置信度
  3. 输出类型分类与分割的配套指标

用法：
  python -m fbfn.calibrate --ckpt fbfn/runs_type/ep20_ema.pth --model large --img_size 448
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def ece(scores, labels, bins=15):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(scores, edges[1:-1])
    total = len(scores)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()
        out += (m.sum() / total) * abs(conf - acc)
    return out


def fit_temperature(logits, labels):
    """LBFGS 最小化 BCE，返回温度 T。"""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(z / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 20.0))


def best_threshold(probs, labels):
    best = (0.5, -1, 0)
    for th in np.linspace(0.05, 0.95, 91):
        pred = (probs >= th).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best[1]:
            best = (float(th), float(acc), float(f1_score(labels, pred)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "ep20_ema.pth"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "type_manifest_val.csv"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--model", choices=["small", "large"], default="large")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--casia_only", action="store_true", default=True)
    ap.add_argument("--out", default="", help="可选：保存 logits 的 npz 路径")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TypeManifestDataset(
        args.manifest, args.img_size, False,
        os.path.join(ROOT, "data", "tigas", "cache256"),
        os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")] \
        if args.casia_only else list(range(len(ds)))
    sub = Subset(ds, idx)
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    if args.model == "large":
        from .type_model_large import FBFNTypeLarge
        model = FBFNTypeLarge(pretrained=False, img_size=args.img_size).to(dev)
    else:
        from .type_model import FBFNType
        model = FBFNType(pretrained=False, img_size=args.img_size).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    ys, zs, yt, pt, zt = [], [], [], [], []
    tp = fp = fn = 0.0
    with torch.no_grad():
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            mask, hm = mask.to(dev), hm.to(dev)
            out = model(x, ela=ela, entropy=ent)
            zs.append(out["final"].float().cpu().numpy())
            zt.append(out["type"].float().cpu().numpy())
            pt.append(out["type"].argmax(1).cpu().numpy())
            ys.append(y.numpy()); yt.append(t.numpy())
            pm = (torch.sigmoid(out["final_map"]) > 0.5).float()
            sel = hm.bool()
            if sel.any():
                p_, m_ = pm[sel], mask[sel]
                tp += (p_ * m_).sum().item()
                fp += (p_ * (1 - m_)).sum().item()
                fn += ((1 - p_) * m_).sum().item()

    ys = np.concatenate(ys).astype(int)
    zs = np.concatenate(zs)
    zt = np.concatenate(zt)
    yt = np.concatenate(yt); pt = np.concatenate(pt)

    probs = 1 / (1 + np.exp(-zs))
    auc = roc_auc_score(ys, probs)
    ece0 = ece(probs, ys)
    th, acc_th, f1_th = best_threshold(probs, ys)
    acc05 = accuracy_score(ys, (probs >= 0.5).astype(int))
    f1_05 = f1_score(ys, (probs >= 0.5).astype(int))

    T = fit_temperature(zs, ys)
    probs_t = 1 / (1 + np.exp(-zs / T))
    ece1 = ece(probs_t, ys)
    th_t, acc_th_t, f1_th_t = best_threshold(probs_t, ys)

    print(f"checkpoint: {args.ckpt}")
    print(f"samples={len(ys)} real={int((ys == 0).sum())} fake={int((ys == 1).sum())}")
    print(f"AUC = {auc:.4f}  (阈值/温度不改变 AUC)")
    print(f"[原始概率] ECE={ece0:.4f}  thr=0.50 acc={acc05:.4f} F1={f1_05:.4f}")
    print(f"[最优阈值] thr={th:.2f} acc={acc_th:.4f} F1={f1_th:.4f}")
    print(f"[温度缩放] T={T:.3f}  ECE={ece1:.4f}")
    print(f"[温度+阈值] thr={th_t:.2f} acc={acc_th_t:.4f} F1={f1_th_t:.4f}")
    present = sorted(set(yt.tolist()))
    wrong_extra = int(np.isin(pt, [c for c in range(4) if c not in present]).sum())
    print(f"类型准确率={accuracy_score(yt, pt):.4f} "
          f"macro-F1(真实存在的类)={f1_score(yt, pt, average='macro', labels=present):.4f} "
          f"误判为未出现类={wrong_extra}")
    print(f"分割 mIoU={tp / (tp + fp + fn + 1e-7):.4f} F1={2 * tp / (2 * tp + fp + fn + 1e-7):.4f}")

    if args.out:
        np.savez(args.out, z=zs, zt=zt, y=ys, type_true=yt, type_pred=pt)
        print(f"saved logits to {args.out}")


if __name__ == "__main__":
    main()
