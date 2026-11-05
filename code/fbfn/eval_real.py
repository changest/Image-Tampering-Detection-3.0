# -*- coding: utf-8 -*-
"""在纯真实 CASIA 验证集上评测（排除合成数据与 TIGAS）。"""
import argparse
import contextlib
import os

import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

from .type_dataset import TypeManifestDataset
from .type_model import FBFNType

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# max-pool 融合的标准化常数（在 CASIA 验证集偶数索引上拟合，alpha=0.75）
BIN_MEAN, BIN_STD = -1.0719, 4.4558
MAX_MEAN, MAX_STD = 1.8947, 6.2276


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "best_type.pth"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "type_manifest_val.csv"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--model", choices=["small", "large"], default="small")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--fusion", action="store_true",
                    help="报告二分类 logit 与 final_map max-pool 的融合 AUC")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--binhead", default="", help="专用二分类头 pkl（fit_binary_head 输出）")
    args = ap.parse_args()
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TypeManifestDataset(
        args.manifest, args.img_size, False,
        os.path.join(ROOT, "data", "tigas", "cache256"),
        os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]
    sub = Subset(ds, idx)
    print("CASIA 验证样本:", len(sub), flush=True)
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    if args.model == "large":
        from .type_model_large import FBFNTypeLarge
        model = FBFNTypeLarge(pretrained=False, img_size=args.img_size).to(dev)
    else:
        model = FBFNType(pretrained=False, img_size=args.img_size).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    ys, ps, yt, pt = [], [], [], []
    zlist, mlist, featlist = [], [], []
    tp = fp = fn = 0.0
    amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    with torch.no_grad(), amp:
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(dev), ent.to(dev), ela.to(dev)
            mask, hm = mask.to(dev), hm.to(dev)
            out = model(x, ela=ela, entropy=ent)
            ps.append(torch.sigmoid(out["final"].float()).cpu().numpy())
            zlist.append(out["final"].float().cpu().numpy())
            mlist.append(out["final_map"].float().flatten(1).max(1).values.cpu().numpy())
            fm = out["final_map"].float().flatten(1)
            k = max(1, fm.shape[1] // 100)
            featlist.append(np.concatenate([
                out["final"].float().cpu().numpy()[:, None],
                fm.max(1).values.cpu().numpy()[:, None],
                fm.topk(k, dim=1).values.mean(1).cpu().numpy()[:, None],
                fm.mean(1).cpu().numpy()[:, None],
                fm.std(1).cpu().numpy()[:, None],
                out["type"].float().cpu().numpy()], axis=1))
            pt.append(out["type"].argmax(1).cpu().numpy())
            ys.append(y.numpy()); yt.append(t.numpy())
            pm = (torch.sigmoid(out["final_map"].float()) > 0.5).float()
            sel = hm.bool()
            if sel.any():
                p_, m_ = pm[sel], mask[sel]
                tp += (p_ * m_).sum().item()
                fp += (p_ * (1 - m_)).sum().item()
                fn += ((1 - p_) * m_).sum().item()

    ys = np.concatenate(ys); ps = np.concatenate(ps)
    yt = np.concatenate(yt); pt = np.concatenate(pt)
    labels = sorted(set(yt.tolist()))
    print(f"样本 {len(ys)} | 真 {int((ys == 0).sum())} 假 {int((ys == 1).sum())}")
    print(f"二分类 AUC        = {roc_auc_score(ys, ps):.4f}")
    if args.fusion:
        zz = np.concatenate(zlist); mm = np.concatenate(mlist)
        zb = (zz - BIN_MEAN) / BIN_STD
        zm = (mm - MAX_MEAN) / MAX_STD
        fused = args.alpha * zb + (1 - args.alpha) * zm
        print(f"融合 AUC（bin+maxpool, alpha={args.alpha:.2f}） = {roc_auc_score(ys, fused):.4f}")
    print(f"类型准确率        = {(pt == yt).mean():.4f}")
    print(f"类型 macro-F1     = {f1_score(yt, pt, average='macro', labels=labels):.4f}")
    print("混淆矩阵(行=真值, 列=预测; 0真 1拼接 2复制):\n",
          confusion_matrix(yt, pt, labels=labels))
    print(f"分割 mIoU         = {tp / (tp + fp + fn + 1e-7):.4f}")
    print(f"分割 F1           = {2 * tp / (2 * tp + fp + fn + 1e-7):.4f}")
    print(classification_report(yt, pt, labels=labels, digits=4))


if __name__ == "__main__":
    main()
