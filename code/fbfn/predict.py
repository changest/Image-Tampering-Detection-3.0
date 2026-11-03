"""FBFN 推理：对目录内图像输出篡改概率图（0-255 PNG，命名与输入 stem 一致）。

用于跨数据集评测，与 experiments/scripts/compute_metrics.py 对接。
输入固定 resize 到 384x384（论文训练/推理尺寸），输出还原到原始分辨率。
"""
import os
import argparse

import numpy as np
import cv2
import torch

from .config import Config
from .model import FBFN


def predict_dir(model, data_root, save_dir, img_size=384, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    imgs = sorted(f for f in os.listdir(data_root) if f.lower().endswith(exts))
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(imgs):
            path = os.path.join(data_root, name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[跳过] {path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_h, ori_w = img.shape[:2]
            x = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            x = x.to(device)
            out = model(x)
            prob = torch.sigmoid(out["final"])[0, 0].cpu().numpy()  # (384,384)
            if prob.shape != (ori_h, ori_w):
                prob = cv2.resize(prob, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            out_png = (prob * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(name)[0] + ".png"), out_png)
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(imgs)}")
    print(f"完成，预测图保存至 {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(cfg.out_dir, "best.pth")
    model = FBFN(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    predict_dir(model, args.data_root, args.save_dir, img_size=cfg.img_size, device=device)


if __name__ == "__main__":
    main()
