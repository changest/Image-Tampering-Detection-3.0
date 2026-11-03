"""生成 FBFN 四分支 evidence map 可视化图（支撑论文「可解释性」卖点）。

对一张真图 + 一张假图，分别提取 CNN / ViT / Noise / Color 四个分支的
像素级证据图（sigmoid 后，值高=该分支认为该区域是伪造），再加上融合后的
final 证据图，排成 2 行（真/假）× 6 列（原图 + 4 分支 + final）的 panel。

运行：python paper/make_evidence_figure.py
输出：paper/figures/evidence_map.svg + .png
"""
import os
import sys

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fbfn.cls_model import FBFNClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CKPT = r"c:\Users\change\Desktop\tphoto\fbfn\runs_cls\best_cls.pth"
VAL_REAL = r"E:\datasets\tigas\val\real"
VAL_FAKE = r"E:\datasets\tigas\val\fake"
CACHE = r"E:\datasets\tigas\cache256"
IMG = 256


def load_cache(path):
    sub = os.path.basename(os.path.dirname(path))
    split = os.path.basename(os.path.dirname(os.path.dirname(path)))
    stem = os.path.splitext(os.path.basename(path))[0]
    npz = os.path.join(CACHE, split, sub, stem + ".npz")
    d = np.load(npz)
    return d["ent"].astype(np.float32), d["ela"].astype(np.float32)


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    return img


def extract_maps(model, path, device):
    img_np = load_img(path)
    ent, ela = load_cache(path)
    x = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    ent_t = torch.from_numpy(ent).unsqueeze(0)
    ela_t = torch.from_numpy(ela).unsqueeze(0)
    x, ent_t, ela_t = x.to(device), ent_t.to(device), ela_t.to(device)

    with torch.no_grad():
        cnn = torch.sigmoid(model.cnn_branch(x))[0, 0].cpu().numpy()
        vit = torch.sigmoid(model.vit_branch(x))[0, 0].cpu().numpy()
        noise = torch.sigmoid(model.noise_branch(x, ela=ela_t))[0, 0].cpu().numpy()
        color = torch.sigmoid(model.color_branch(x, entropy=ent_t))[0, 0].cpu().numpy()
        feats = [model.cnn_branch(x), model.vit_branch(x),
                 model.noise_branch(x, ela=ela_t), model.color_branch(x, entropy=ent_t)]
        final, _ = model.fusion(feats)
        final = torch.sigmoid(final)[0, 0].cpu().numpy()

    # ViT/Noise/Color 分支内部已插值到 256，但仍保险 resize 到 256 对齐
    def r(m):
        return cv2.resize(m, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
    return {
        "input": img_np,
        "CNN": r(cnn), "ViT": r(vit), "Noise": r(noise),
        "Color": r(color), "Fused": r(final),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FBFNClassifier(pretrained=False, img_size=IMG).to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
    model.eval()

    real_p = os.path.join(VAL_REAL, sorted(os.listdir(VAL_REAL))[0])
    fake_p = os.path.join(VAL_FAKE, sorted(os.listdir(VAL_FAKE))[0])
    real_maps = extract_maps(model, real_p, device)
    fake_maps = extract_maps(model, fake_p, device)
    # 顺便打印图像级得分，确认方向正确
    print(f"real image 得分: {real_maps['Fused'].mean():.3f} (应偏低)")
    print(f"fake image 得分: {fake_maps['Fused'].mean():.3f} (应偏高)")

    labels = ["Input", "CNN", "ViT", "Noise", "Color", "Fused"]
    fig, axes = plt.subplots(2, 6, figsize=(13.5, 4.6))
    for col, lab in enumerate(labels):
        for row, (maps, row_name) in enumerate(((real_maps, "Real"), (fake_maps, "Fake"))):
            ax = axes[row, col]
            if lab == "Input":
                ax.imshow(maps["input"])
            else:
                ax.imshow(maps[lab], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_name, fontsize=10)
            if row == 0:
                ax.set_title(lab, fontsize=10)

    # 右侧 colorbar
    cbar_ax = fig.add_axes([0.92, 0.20, 0.012, 0.6])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel("Forgery probability", fontsize=8)

    plt.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.08, wspace=0.08, hspace=0.08)
    for ext in ("svg", "png"):
        out = os.path.join(FIG_DIR, f"evidence_map.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"保存 {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
