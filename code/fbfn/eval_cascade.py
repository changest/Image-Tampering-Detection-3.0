"""级联推理评估：系统一（噪声+颜色）快速筛选 + 低置信升级全模型。

两种系统一分数来源：
1. 提供 --sys1_ckpt：加载独立训练的系统一模型（drop cnn/vit），
   用其融合 logit 作为快速分数（严谨口径）。
2. 未提供 --sys1_ckpt：回退为 full 模型内部分支 logit 平均
   s1 = sigmoid((ela + color) / 2)，作为无需额外训练的近似。

级联规则：若 |s1 - 0.5| >= tau（系统一高置信），直接采纳 s1 决策；
否则升级到全模型，采纳 final 决策。
扫描 tau，输出 (升级比例 rho, 级联 AUC, 平均延迟)。

延迟口径沿用 profiling.py：系统一 3 ms，全模型 30 ms，
级联平均延迟 = 3 + rho * (30 - 3) ms。

用法：
    python -m fbfn.eval_cascade --gen val
    python -m fbfn.eval_cascade --gen val --sys1_ckpt <研究项目>/data/tigas/runs_sys1/best_cls.pth
    python -m fbfn.eval_cascade --gen ADM --sys1_ckpt ...
"""
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .cls_model import FBFNClassifier
from .cls_dataset import ImageClsDataset

# TIGAS 数据根目录：<研究项目>/data/tigas（随项目一起移动，不再硬编码盘符）
DST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "tigas",
)
CACHE = os.path.join(DST, "cache256")
T_SYS1 = 3.0   # ms
T_FULL = 30.0  # ms


def _dirs(gen):
    if gen.lower() in ("val", "sd14"):
        return os.path.join(DST, "val", "real"), os.path.join(DST, "val", "fake")
    return os.path.join(DST, "crossval", gen, "real"), os.path.join(DST, "crossval", gen, "fake")


def _default_ckpt():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_cls", "best_cls.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="val", help="val/sd14 或跨生成器目录名（ADM/Glide/...）")
    ap.add_argument("--ckpt", default=None, help="full 模型权重，默认 runs_cls/best_cls.pth")
    ap.add_argument("--sys1_ckpt", default=None, help="独立系统一权重（可选，drop cnn/vit 训练）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or _default_ckpt()

    full = FBFNClassifier(pretrained=False, img_size=256).to(device)
    full.load_state_dict(torch.load(ckpt, map_location=device))
    full.eval()

    sys1 = None
    if args.sys1_ckpt:
        sys1 = FBFNClassifier(pretrained=False, img_size=256,
                              drop_branches=("cnn", "vit")).to(device)
        sys1.load_state_dict(torch.load(args.sys1_ckpt, map_location=device))
        sys1.eval()

    real_dir, fake_dir = _dirs(args.gen)
    ds = ImageClsDataset(real_dir, fake_dir, img_size=256, train=False, cache_dir=CACHE)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    s_full, s1, y = [], [], []
    with torch.no_grad():
        for imgs, ent, ela, lab in loader:
            imgs, ent, ela = imgs.to(device), ent.to(device), ela.to(device)
            out = full(imgs, ela=ela, entropy=ent)
            s_full.append(torch.sigmoid(out["final"]).cpu().numpy())
            if sys1 is not None:
                s1.append(torch.sigmoid(sys1(imgs, ela=ela, entropy=ent)["final"]).cpu().numpy())
            else:
                s1.append(torch.sigmoid(0.5 * (out["ela"] + out["color"])).cpu().numpy())
            y.append(lab.numpy())

    s_full = np.concatenate(s_full)
    s1 = np.concatenate(s1)
    y = np.concatenate(y)

    src = "独立系统一模型" if sys1 is not None else "full 分支 logit 平均"
    print(f"[{args.gen}] 样本 {len(y)}（真 {int((y == 0).sum())} / 假 {int((y == 1).sum())}）")
    print(f"  系统一来源: {src}")
    print(f"  全模型 AUC        = {roc_auc_score(y, s_full) * 100:.2f}%")
    print(f"  系统一(噪声+颜色) AUC = {roc_auc_score(y, s1) * 100:.2f}%")
    print(f"  {'tau':>5} {'升级比例rho':>10} {'级联AUC':>10} {'平均延迟(ms)':>12}")

    margin = np.abs(s1 - 0.5)
    for tau in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        upgrade = margin < tau
        rho = float(upgrade.mean())
        z = np.where(upgrade, s_full, s1)
        auc = roc_auc_score(y, z)
        latency = T_SYS1 + rho * (T_FULL - T_SYS1)
        print(f"  {tau:5.2f} {rho:10.3f} {auc * 100:10.2f} {latency:12.1f}")


if __name__ == "__main__":
    main()
