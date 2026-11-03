"""三个对比基线在 GenImage 协议下的图像级评估（与 FBFN 对齐，输出 Acc/AUC/AP）。

方法：
  - cnndetection: ResNet50 + Blur/JPEG 增强（ProGAN 训练，zero-shot）
  - univfd: CLIP ViT-L/14 + fc 线性层（zero-shot）
  - resnet50: GenImage 风格 ResNet-50 baseline（sd14 训练，需先 train_resnet50.py 出权重）

测试集：sd14 val + crossval 8 生成器（ADM/Glide/biggan/Midjourney/VQDM/wuk/sd15_1/sd15_2）。

用法：
    python -m baselines.eval_baselines --methods cnndetection univfd resnet50
"""
import os
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score, average_precision_score

import open_clip

HERE = os.path.dirname(os.path.abspath(__file__))
DST = r"E:\datasets\tigas"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GENS = ["sd14", "ADM", "Glide", "biggan", "Midjourney", "VQDM", "wuk", "sd15_1", "sd15_2"]


def _test_dirs(gen):
    if gen == "sd14":
        return os.path.join(DST, "val", "real"), os.path.join(DST, "val", "fake")
    return (os.path.join(DST, "crossval", gen, "real"),
            os.path.join(DST, "crossval", gen, "fake"))


class ImgDir(Dataset):
    """遍历 real/fake 目录，输出 (PIL, label)。"""

    def __init__(self, real_dir, fake_dir, transform):
        self.transform = transform
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
        img = self.transform(Image.fromarray(img))
        return img, lab


# ---------------- 方法定义 ----------------
def load_cnndetection(device):
    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, 1)
    sd = torch.load(os.path.join(HERE, "cnndetection", "weights", "blur_jpg_prob0.5.pth"),
                    map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd)
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf


def load_univfd(device):
    clip_model, _, pre = open_clip.create_model_and_transforms(
        "ViT-L/14", pretrained="openai", force_quick_gelu=True)
    clip_model.to(device).eval()
    fc = nn.Linear(768, 1)
    sd = torch.load(os.path.join(HERE, "univfd", "pretrained_weights", "fc_weights.pth"),
                    map_location="cpu", weights_only=False)
    fc.weight.data = sd["weight"].clone()
    fc.bias.data = sd["bias"].clone()
    fc.to(device).eval()
    return (clip_model, fc), pre


def load_resnet50(device):
    import timm
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)
    sd = torch.load(os.path.join(HERE, "resnet50", "best_resnet50.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf


def _forward(method, model, imgs, device):
    """返回 (B,) 的 logits（未 sigmoid）。"""
    with torch.no_grad():
        if method == "univfd":
            clip_model, fc = model
            feats = clip_model.encode_image(imgs.to(device)).float()
            return fc(feats).squeeze(1)
        else:
            return model(imgs.to(device)).squeeze(1)


def eval_method(method, model, transform, device):
    print(f"\n=== {method} ===", flush=True)
    results = []
    for gen in GENS:
        real_dir, fake_dir = _test_dirs(gen)
        ds = ImgDir(real_dir, fake_dir, transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        scores, labels = [], []
        for imgs, lab in loader:
            logits = _forward(method, model, imgs, device)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(lab.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        acc = float(((scores > 0.5).astype(float) == labels).mean())
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        print(f"  {gen:10s} 真 {int((labels == 0).sum())}/假 {int((labels == 1).sum())} | "
              f"Acc={acc * 100:.2f} AUC={auc * 100:.2f} AP={ap * 100:.2f}", flush=True)
        results.append((gen, acc, auc, ap))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["cnndetection", "univfd", "resnet50"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {"cnndetection": load_cnndetection, "univfd": load_univfd, "resnet50": load_resnet50}
    for m in args.methods:
        model, tf = loaders[m](device)
        eval_method(m, model, tf, device)


if __name__ == "__main__":
    main()
