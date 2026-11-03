"""跨生成器消融：用 full / w-o-noise / w-o-color 三个 checkpoint 在 8 个跨库生成器上评估。

目的：sd14 同分布下 noise/color 分支贡献近乎为零（w-o-noise AUC 99.96、w-o-color 99.85
均不低于 full 99.93），本脚本证明它们在跨生成器泛化上的价值——去掉后跨库 AUC 应明显下降。

只做推理，不下载不预计算（数据与 cache256 均已就绪）。

运行：python -m fbfn.run_cross_ablation
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
    ("full", os.path.join(HERE, "runs_cls", "best_cls.pth"), []),
    ("w-o-cnn", os.path.join(HERE, "runs_ablation", "w-o-cnn", "best_cls.pth"), ["cnn"]),
    ("w-o-vit", os.path.join(HERE, "runs_ablation", "w-o-vit", "best_cls.pth"), ["vit"]),
    ("w-o-noise", os.path.join(HERE, "runs_ablation", "w-o-noise", "best_cls.pth"), ["noise"]),
    ("w-o-color", os.path.join(HERE, "runs_ablation", "w-o-color", "best_cls.pth"), ["color"]),
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

    out = os.path.join(HERE, "runs_ablation", "cross_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== 跨库 AUC 对比（全部 5 配置）=====", flush=True)
    names = [name for name, _, _ in CONFIGS]
    print(f"{'gen':12s}" + "".join(f"{n:>11s}" for n in names), flush=True)
    for gen in GENS:
        vals = "".join(f"{results[n][gen]['auc'] * 100:>10.2f}%" for n in names)
        print(f"{gen:12s}{vals}", flush=True)


if __name__ == "__main__":
    main()
