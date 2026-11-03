"""评估重训后的 w-o-cnn-rerun / w-o-vit-rerun checkpoint，并与原始结果对比。

目的：检验 §4.4 中语义分支（CNN/ViT）在 GLIDE 上的单次运行方差是否为偶发。
原始值：w-o-cnn GLIDE AUC=39.27%、w-o-vit GLIDE AUC=92.91%（跨库方向相反的反常）。

只做推理，数据与 cache256 均已就绪。

运行：python -m fbfn.run_rerun_eval
"""
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .cls_model import FBFNClassifier
from .cls_dataset import ImageClsDataset

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = os.path.dirname(os.path.abspath(__file__))
_TIGAS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "tigas")
CROSS = os.path.join(_TIGAS, "crossval")
CACHE = os.path.join(_TIGAS, "cache256")
IMG = 256

GENS = ["ADM", "Glide", "biggan", "Midjourney", "VQDM", "sd15_1", "sd15_2", "wuk"]

CONFIGS = [
    ("w-o-cnn", os.path.join(HERE, "runs_ablation", "w-o-cnn", "best_cls.pth"), ["cnn"]),
    ("w-o-cnn-rerun", os.path.join(HERE, "runs_ablation", "w-o-cnn-rerun", "best_cls.pth"), ["cnn"]),
    ("w-o-vit", os.path.join(HERE, "runs_ablation", "w-o-vit", "best_cls.pth"), ["vit"]),
    ("w-o-vit-rerun", os.path.join(HERE, "runs_ablation", "w-o-vit-rerun", "best_cls.pth"), ["vit"]),
]


def eval_gen(model, device, gen):
    ds = ImageClsDataset(os.path.join(CROSS, gen, "real"), os.path.join(CROSS, gen, "fake"),
                         img_size=IMG, train=False, cache_dir=CACHE)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, ent, ela, lab in loader:
            imgs = imgs.to(device)
            out = model(imgs, ela=ela.to(device), entropy=ent.to(device))
            scores.append(torch.sigmoid(out["final"]).cpu().numpy())
            labels.append(lab.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    acc = float(((scores > 0.5).astype(float) == labels).mean())
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    return acc, auc, ap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, ckpt, drop in CONFIGS:
        if not os.path.exists(ckpt):
            print(f"跳过 {name}：未找到 {ckpt}", flush=True)
            continue
        print(f"\n===== {name} (drop={drop}) =====", flush=True)
        model = FBFNClassifier(pretrained=False, img_size=IMG, drop_branches=drop).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        results[name] = {}
        for gen in GENS:
            acc, auc, ap = eval_gen(model, device, gen)
            results[name][gen] = {"acc": acc, "auc": auc, "ap": ap}
            print(f"  {gen:12s} Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f}",
                  flush=True)

    out = os.path.join(HERE, "runs_ablation", "rerun_cross_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== GLIDE AUC 对比（原 vs 重训）=====", flush=True)
    for key in ["w-o-cnn", "w-o-vit"]:
        orig = results.get(key, {}).get("Glide", {}).get("auc")
        rerun = results.get(key + "-rerun", {}).get("Glide", {}).get("auc")
        if orig is not None and rerun is not None:
            print(f"  {key:12s} 原={orig * 100:6.2f}%  重训={rerun * 100:6.2f}%  "
                  f"Δ={ (rerun - orig) * 100:+6.2f}%", flush=True)

    print("\n===== 全部跨库 AUC 对比（%），仅列出差异关键项 =====", flush=True)
    for gen in GENS:
        cells = []
        for key in ["w-o-cnn", "w-o-cnn-rerun", "w-o-vit", "w-o-vit-rerun"]:
            v = results.get(key, {}).get(gen, {}).get("auc")
            cells.append(f"{v * 100:6.2f}" if v is not None else "   -- ")
        print(f"  {gen:12s} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
