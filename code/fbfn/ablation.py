"""消融：分别评测 FBFN 四个分支与融合结果在 CASIA val 上的性能。

逐图平均协议（与 experiments/scripts/compute_metrics.py 一致，阈值 0.5）。
用于判断「四分支融合」相对单分支是否带来增益——论文核心卖点。
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .model import FBFN
from .dataset import CASIADataset


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(cfg.out_dir, "best.pth")
    model = FBFN(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    val_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="val",
                          img_size=cfg.img_size, val_ratio=cfg.val_ratio,
                          seed=cfg.seed, train=False)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    heads = ["final", "cnn", "vit", "ela", "color"]
    acc = {h: {"tp": [], "fp": [], "fn": []} for h in heads}

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            for h in heads:
                pred = (torch.sigmoid(out[h]) > 0.5).float()
                for b in range(imgs.size(0)):
                    p, m = pred[b], masks[b]
                    acc[h]["tp"].append((p * m).sum().item())
                    acc[h]["fp"].append((p * (1 - m)).sum().item())
                    acc[h]["fn"].append(((1 - p) * m).sum().item())

    print(f"CASIA val 消融（逐图平均，阈值 0.5，样本 {len(val_ds)}）：")
    print(f"{'分支':<8}{'mIoU':>9}{'F1':>9}{'Precision':>11}{'Recall':>9}")
    for h in heads:
        tp = np.array(acc[h]["tp"]); fp = np.array(acc[h]["fp"]); fn = np.array(acc[h]["fn"])
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        miou = tp / (tp + fp + fn + 1e-7)
        print(f"{h:<8}{miou.mean() * 100:>8.2f}{f1.mean() * 100:>8.2f}"
              f"{prec.mean() * 100:>10.2f}{rec.mean() * 100:>8.2f}")


if __name__ == "__main__":
    main()
