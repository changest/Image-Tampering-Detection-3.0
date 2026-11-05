# -*- coding: utf-8 -*-
"""收集 CASIA 验证集上的二分类分数，做集成与 max-pool 融合分析。

每个检查点保存：二分类 logit、final_map 的 max logit、top-k 均值、标签。
分析：
  1. 单模型 AUC
  2. 多模型 logit 集成
  3. max-pool 分数单独 AUC
  4. 在验证集前半调 alpha、后半报告（避免自评自测）
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def collect(ckpt, ds, idx, dev, bs=24):
    from .type_model_large import FBFNTypeLarge
    m = FBFNTypeLarge(pretrained=False, img_size=448).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    loader = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=8)
    z, zmax, ztop, y = [], [], [], []
    with torch.no_grad():
        for x, ent, ela, yy, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            out = m(x, ela=ela, entropy=ent)
            z.append(out["final"].float().cpu().numpy())
            fm = out["final_map"].float().flatten(1)
            zmax.append(fm.max(1).values.cpu().numpy())
            k = max(1, fm.shape[1] // 100)
            ztop.append(fm.topk(k, dim=1).values.mean(1).cpu().numpy())
            y.append(yy.numpy())
    return np.concatenate(z), np.concatenate(zmax), np.concatenate(ztop), np.concatenate(y).astype(int)


def zs(a):
    return (a - a.mean()) / (a.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        os.path.join(HERE, "runs_type", "ep20_ema.pth"),
        os.path.join(HERE, "runs_type", "ep30_raw.pth"),
        os.path.join(HERE, "runs_type", "last_ema.pth")])
    args = ap.parse_args()

    dev = torch.device("cuda")
    ds = TypeManifestDataset(os.path.join(HERE, "type_manifest_val.csv"), 448, False,
                             os.path.join(ROOT, "data", "tigas", "cache256"),
                             os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]

    names, Z, ZM, ZT = [], [], [], []
    y = None
    for p in args.ckpts:
        z, zmax, ztop, yy = collect(p, ds, idx, dev)
        names.append(os.path.splitext(os.path.basename(p))[0])
        Z.append(z); ZM.append(zmax); ZT.append(ztop)
        y = yy
        print(f"{names[-1]}: AUC={roc_auc_score(y, z):.4f}  "
              f"maxpool AUC={roc_auc_score(y, zmax):.4f}  "
              f"top1% AUC={roc_auc_score(y, ztop):.4f}", flush=True)

    Z = np.stack(Z); ZM = np.stack(ZM); ZT = np.stack(ZT)
    n = len(y)
    tune = np.arange(n) % 2 == 0      # 偶数索引调参
    test = ~tune                      # 奇数索引报告

    print("\n[集成]")
    ens = Z.mean(0)
    print(f"  全部: {roc_auc_score(y, ens):.4f}   测试半集: {roc_auc_score(y[test], ens[test]):.4f}")

    # max-pool 与二分类 logit 融合（调参只在 tune 半集）
    print("\n[bin + maxpool 融合]  alpha 在偶数索引上选，报告奇数索引")
    stat = ZM.mean(0)                 # 三个模型的 max-pool 平均
    zb, zsx = zs(Z.mean(0)), zs(stat)
    best = None
    for a in np.linspace(0, 1, 21):
        auc = roc_auc_score(y[tune], (a * zb + (1 - a) * zsx)[tune])
        if best is None or auc > best[1]:
            best = (float(a), float(auc))
    a = best[0]
    fused = a * zb + (1 - a) * zsx
    print(f"  最佳 alpha={a:.2f}  (tune {best[1]:.4f})")
    print(f"  测试半集融合 AUC: {roc_auc_score(y[test], fused[test]):.4f}")
    print(f"  测试半集集成 AUC: {roc_auc_score(y[test], ens[test]):.4f}")
    print(f"  测试半集最佳单模型: ", end="")
    singles = [(nm, roc_auc_score(y[test], z[test])) for nm, z in zip(names, Z)]
    print(", ".join(f"{nm}={v:.4f}" for nm, v in singles))

    out = os.path.join(HERE, "runs_type", "casia_scores.npz")
    np.savez(out, y=y, Z=Z, ZM=ZM, ZT=ZT, names=np.array(names))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
