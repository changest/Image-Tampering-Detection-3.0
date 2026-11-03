"""ResNet-50 baseline 训练：sd14 子集（与 FBFN 同协议：256 输入、BCE、15 epoch、flip 增强）。

输出 baselines/resnet50/best_resnet50.pth（按 val AUC 选优）。
"""
import os
import time
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG = 256


class ClsDataset(Dataset):
    def __init__(self, real_dir, fake_dir, train=True):
        self.train = train
        self.items = []
        for d, lab in ((real_dir, 0), (fake_dir, 1)):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    self.items.append((os.path.join(d, f), lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__((i + 1) % len(self.items))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        if self.train and random.random() < 0.5:
            img = cv2.flip(img, 1)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (img - mean) / std
        return img, torch.tensor(float(lab))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default=r"E:\datasets\tigas\train\real")
    ap.add_argument("--fake_dir", default=r"E:\datasets\tigas\train\fake")
    ap.add_argument("--val_real_dir", default=r"E:\datasets\tigas\val\real")
    ap.add_argument("--val_fake_dir", default=r"E:\datasets\tigas\val\fake")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "resnet50")
    os.makedirs(out_dir, exist_ok=True)

    train_ds = ClsDataset(args.real_dir, args.fake_dir, train=True)
    val_ds = ClsDataset(args.val_real_dir, args.val_fake_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"训练 {len(train_ds)} 张，验证 {len(val_ds)} 张", flush=True)

    model = timm.create_model("resnet50", pretrained=True, num_classes=1)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for imgs, lab in train_loader:
            imgs, lab = imgs.to(device), lab.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
        sched.step()

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for imgs, lab in val_loader:
                logits = model(imgs.to(device)).squeeze(1)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(lab.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        acc = float(((scores > 0.5).astype(float) == labels).mean())
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        print(f"Epoch {epoch + 1}/{args.epochs} | train {run_loss / len(train_loader):.4f} | "
              f"val Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f} | {time.time() - t0:.0f}s",
              flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_resnet50.pth"))

    print(f"最佳 val AUC={best_auc * 100:.2f}，保存 best_resnet50.pth", flush=True)


if __name__ == "__main__":
    main()
