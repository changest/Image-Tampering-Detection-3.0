"""在 CASIA 验证集上评估 FBFN：mIoU / F1 / Precision / Recall / Pixel AUC。"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import Config
from .model import FBFN
from .dataset import CASIADataset


def pixel_auc(probs, labels, budget=10_000_000, seed=0):
    p = np.asarray(probs).ravel()
    l = np.asarray(labels).ravel()
    if p.size > budget:
        rng = np.random.RandomState(seed)
        idx = rng.choice(p.size, budget, replace=False)
        p, l = p[idx], l[idx]
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    return float(roc_auc_score(l, p))


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0.0
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            prob = torch.sigmoid(out["final"])
            pred = (prob > 0.5).float()
            tp += (pred * masks).sum().item()
            fp += (pred * (1 - masks)).sum().item()
            fn += ((1 - pred) * masks).sum().item()
            probs_all.append(prob.cpu().numpy())
            labels_all.append(masks.cpu().numpy())

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    miou = tp / (tp + fp + fn + 1e-7)
    auc = pixel_auc(np.concatenate(probs_all), np.concatenate(labels_all))
    return {
        "mIoU": miou * 100,
        "F1": f1 * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "AUC": auc * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="best.pth 路径，默认 runs/best.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt or os.path.join(cfg.out_dir, "best.pth")
    model = FBFN(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    val_ds = CASIADataset(cfg.casia_root, cfg.gt_root, split="val",
                          img_size=cfg.img_size, val_ratio=cfg.val_ratio,
                          seed=cfg.seed, train=False)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    print(f"验证样本: {len(val_ds)}")
    metrics = evaluate(model, loader, device)
    print("CASIA 验证集指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
