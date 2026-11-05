# -*- coding: utf-8 -*-
"""分割掩码阈值扫描 + 水平翻转 TTA。

用法：
  python -m fbfn.eval_seg_threshold --ckpt fbfn/runs_type/ep30_raw.pth
  python -m fbfn.eval_seg_threshold --ckpt fbfn/runs_type/ep30_raw.pth --tta
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .type_dataset import TypeManifestDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "ep30_raw.pth"))
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--amp", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    dev = torch.device("cuda")
    from .type_model_large import FBFNTypeLarge
    m = FBFNTypeLarge(pretrained=False, img_size=args.img_size).to(dev)
    m.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    m.eval()

    ds = TypeManifestDataset(os.path.join(HERE, "type_manifest_val.csv"), args.img_size, False,
                             os.path.join(ROOT, "data", "tigas", "cache256"),
                             os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" else torch.no_grad()

    probs, masks = [], []
    with torch.no_grad(), amp:
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            p = torch.sigmoid(m(x, ela=ela, entropy=ent)["final_map"].float())
            if args.tta:
                xf = torch.flip(x, dims=[3])
                ef = torch.flip(ela, dims=[3])
                pf = torch.flip(torch.sigmoid(m(xf, ela=ef, entropy=ent)["final_map"].float()),
                                dims=[3])
                p = 0.5 * (p + pf)
            probs.append(p.cpu().numpy()[:, 0])
            masks.append(mask.numpy()[:, 0])

    P = np.concatenate(probs); M = np.concatenate(masks)
    # 有掩码样本：使用 mask 非空或 hm；这里用 hm 不可得，改为按数据顺序过滤
    # 重新读取 hm 标记
    hm_all = []
    for i in idx:
        hm_all.append(ds.rows[i][4])
    hm_all = np.array(hm_all) > 0
    P, M = P[hm_all], M[hm_all]
    print(f"ckpt={os.path.basename(args.ckpt)} tta={args.tta} samples={len(P)}")

    best = None
    for th in np.arange(0.20, 0.86, 0.05):
        pm = P > th
        tp = (pm * M).sum(); fp = (pm * (1 - M)).sum(); fn = ((1 - pm) * M).sum()
        miou = tp / (tp + fp + fn + 1e-7)
        if best is None or miou > best[1]:
            best = (float(th), float(miou))
        print(f"  th={th:.2f}  mIoU={miou:.4f}")
    print(f"best th={best[0]:.2f}  mIoU={best[1]:.4f}")


if __name__ == "__main__":
    main()
