# -*- coding: utf-8 -*-
"""真实 CASIA 验证集上的鲁棒性评测：JPEG 压缩 / 缩放 / 模糊 / 二次压缩。

扰动施加在统一缩放后的 448×448 RGB 上，并重新计算 ELA 与熵图，
其余流程与 eval_real 一致；掩码不做几何变换（缩放后回到原尺寸，坐标一致）。

用法：
  python -m fbfn.robust_eval --ckpt fbfn/runs_type/ep30_raw.pth --model large --img_size 448
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, f1_score

from .type_dataset import TypeManifestDataset
from .precompute import compute_ela, compute_entropy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _to_rgb(x):
    """x: (3,H,W) tensor [0,1] -> (H,W,3) uint8 RGB"""
    a = (x.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return a


def _from_rgb(a):
    rgb = a.astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


def perturb(img, mode, rng=None):
    """img: (H,W,3) uint8 RGB -> (H,W,3) uint8"""
    h, w = img.shape[:2]
    bgr = img[:, :, ::-1]
    if mode.startswith("jpeg"):
        q = int(mode[4:])
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    if mode.startswith("resize"):
        s = float(mode[6:]) / 100.0
        small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return up[:, :, ::-1]
    if mode.startswith("blur"):
        k = int(mode[4:]) | 1
        return cv2.GaussianBlur(bgr, (k, k), 0)[:, :, ::-1]
    if mode == "screenshot":
        small = cv2.resize(bgr, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        up = cv2.resize(dec, (w, h), interpolation=cv2.INTER_LINEAR)
        return up[:, :, ::-1]
    if mode == "clean":
        return img
    raise ValueError(mode)


class PerturbedDataset(Dataset):
    def __init__(self, base, mode, img_size):
        self.base = base
        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, ent, ela, y, t, mask, hm = self.base[idx]
        img = _to_rgb(x)
        img = perturb(img, self.mode)
        rgb = _from_rgb(img)
        ela = compute_ela(rgb)
        ent = compute_entropy(rgb)
        if ent.shape[-1] != 31:
            ent = np.stack([cv2.resize(ent[c], (31, 31), interpolation=cv2.INTER_LINEAR)
                            for c in range(ent.shape[0])])
        return (torch.from_numpy(np.ascontiguousarray(rgb)).float(),
                torch.from_numpy(np.ascontiguousarray(ent)),
                torch.from_numpy(np.ascontiguousarray(ela)), y, t, mask, hm)


MODES = ["clean", "jpeg90", "jpeg70", "jpeg50", "resize75", "resize50", "blur5", "screenshot"]


def evaluate(model, loader, device):
    model.eval()
    ys, ps, yt, pt = [], [], [], []
    tp = fp = fn = 0.0
    with torch.no_grad():
        for x, ent, ela, y, t, mask, hm in loader:
            x, ent, ela = x.to(device), ent.to(device), ela.to(device)
            mask, hm = mask.to(device), hm.to(device)
            out = model(x, ela=ela, entropy=ent)
            ps.append(torch.sigmoid(out["final"]).cpu().numpy())
            pt.append(out["type"].argmax(1).cpu().numpy())
            ys.append(y.numpy()); yt.append(t.numpy())
            pm = (torch.sigmoid(out["final_map"]) > 0.5).float()
            sel = hm.bool()
            if sel.any():
                p_, m_ = pm[sel], mask[sel]
                tp += (p_ * m_).sum().item()
                fp += (p_ * (1 - m_)).sum().item()
                fn += ((1 - p_) * m_).sum().item()
    ys = np.concatenate(ys).astype(int); ps = np.concatenate(ps)
    yt = np.concatenate(yt); pt = np.concatenate(pt)
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    labels = sorted(set(yt.tolist()))
    f1 = f1_score(yt, pt, average="macro", labels=labels)
    acc = float((pt == yt).mean())
    cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(yt, pt):
        cm[a, b] += 1
    copy_recall = cm[2, 2] / max(1, cm[2].sum())
    miou = tp / (tp + fp + fn + 1e-7)
    return auc, acc, f1, miou, copy_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs_type", "ep30_raw.pth"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "type_manifest_val.csv"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--model", choices=["small", "large"], default="large")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TypeManifestDataset(
        args.manifest, args.img_size, False,
        os.path.join(ROOT, "data", "tigas", "cache256"),
        os.path.join(ROOT, "data", "cache_type"))
    idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]
    sub = Subset(ds, idx)

    if args.model == "large":
        from .type_model_large import FBFNTypeLarge
        model = FBFNTypeLarge(pretrained=False, img_size=args.img_size).to(dev)
    else:
        from .type_model import FBFNType
        model = FBFNType(pretrained=False, img_size=args.img_size).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    print(f"checkpoint={args.ckpt} samples={len(sub)}", flush=True)
    print(f"{'mode':10s} {'AUC':>7s} {'typeAcc':>8s} {'typeF1':>7s} {'mIoU':>7s} {'copyRec':>8s}", flush=True)
    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        pds = PerturbedDataset(sub, mode, args.img_size)
        loader = DataLoader(pds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        auc, acc, f1, miou, cr = evaluate(model, loader, dev)
        print(f"{mode:10s} {auc:7.4f} {acc:8.4f} {f1:7.4f} {miou:7.4f} {cr:8.4f}", flush=True)


if __name__ == "__main__":
    main()
