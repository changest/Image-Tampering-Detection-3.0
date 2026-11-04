# -*- coding: utf-8 -*-
"""分辨率级联评估：低分辨率（252）快速判 + 低置信升级 448。

这是"查询路由"在单骨干架构下的可行形式：不同输入分辨率对应不同算力。
在 CASIA 验证集上比较：
  - AUC(252)
  - AUC(448)
  - 级联：|p252-0.5| < tau 时用 448，否则用 252
延迟使用实测或参数指定的 t252/t448。

用法：
  python -m fbfn.eval_res_cascade --ckpt fbfn/runs_type/last_ema.pth --lo 252 --hi 448
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load(ckpt, hi, dev):
    from .type_model_large import FBFNTypeLarge
    m = FBFNTypeLarge(pretrained=False, img_size=hi).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "last_ema.pth"))
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--lo", type=int, default=252)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--t_lo", type=float, default=0.0, help="低分辨率单图延迟(ms)，0=实测")
    ap.add_argument("--t_hi", type=float, default=0.0, help="高分辨率单图延迟(ms)，0=实测")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(args.ckpt, args.img_size, dev)

    ds = TypeManifestDataset(os.path.join(HERE, "type_manifest_val.csv"),
                             args.img_size, False,
                             os.path.join(ROOT, "data", "tigas", "cache256"),
                             os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    y_all, plo, phi = [], [], []
    with torch.no_grad():
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            out_hi = model(x, ela=ela, entropy=ent)
            phi.append(torch.sigmoid(out_hi["final"]).float().cpu().numpy())

            xl = F.interpolate(x, size=(args.lo, args.lo), mode="bilinear", align_corners=False)
            elal = F.interpolate(ela, size=(args.lo, args.lo), mode="bilinear", align_corners=False)
            out_lo = model(xl, ela=elal, entropy=ent)
            plo.append(torch.sigmoid(out_lo["final"]).float().cpu().numpy())
            y_all.append(y.numpy())

    y = np.concatenate(y_all).astype(int)
    plo = np.concatenate(plo)
    phi = np.concatenate(phi)
    auc_lo = roc_auc_score(y, plo)
    auc_hi = roc_auc_score(y, phi)
    print(f"样本 {len(y)}  AUC@ {args.lo} = {auc_lo:.4f}   AUC@ {args.img_size} = {auc_hi:.4f}", flush=True)

    # 实测延迟（batch=1，预热后 30 次均值）
    if args.t_lo <= 0 or args.t_hi <= 0:
        xb = next(iter(DataLoader(Subset(ds, idx[:1]), batch_size=1)))
        x0, ent0, ela0 = xb[0].to(dev), xb[1].to(dev), xb[2].to(dev)
        xl0 = F.interpolate(x0, size=(args.lo, args.lo), mode="bilinear", align_corners=False)
        elal0 = F.interpolate(ela0, size=(args.lo, args.lo), mode="bilinear", align_corners=False)
        for _ in range(5):
            model(xl0, ela=elal0, entropy=ent0); model(x0, ela=ela0, entropy=ent0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            model(xl0, ela=elal0, entropy=ent0)
        torch.cuda.synchronize(); t_lo = (time.perf_counter() - t0) / 30 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            model(x0, ela=ela0, entropy=ent0)
        torch.cuda.synchronize(); t_hi = (time.perf_counter() - t0) / 30 * 1000
    else:
        t_lo, t_hi = args.t_lo, args.t_hi
    print(f"延迟实测: {args.lo}px={t_lo:.1f}ms  {args.img_size}px={t_hi:.1f}ms", flush=True)

    print(f"\n{'tau':>5} {'rho':>6} {'级联AUC':>9} {'平均延迟(ms)':>12}")
    for tau in np.round(np.arange(0.0, 0.5, 0.05), 2):
        up = np.abs(plo - 0.5) < tau
        z = np.where(up, phi, plo)
        rho = float(up.mean())
        print(f"{tau:5.2f} {rho:6.3f} {roc_auc_score(y, z):9.4f} {t_lo + rho * (t_hi - t_lo):12.1f}")


if __name__ == "__main__":
    main()
