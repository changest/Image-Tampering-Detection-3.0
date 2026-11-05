# -*- coding: utf-8 -*-
"""生成定性分割结果图：输入 / GT / 预测热图 / 叠加（3 个样例）。"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from fbfn.type_dataset import TypeManifestDataset  # noqa: E402
from fbfn.type_model_large import FBFNTypeLarge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "code", "fbfn", "runs_type", "ep30_raw.pth")

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7})

ds = TypeManifestDataset(os.path.join(ROOT, "code", "fbfn", "type_manifest_val.csv"),
                         448, False,
                         os.path.join(ROOT, "data", "tigas", "cache256"),
                         os.path.join(ROOT, "data", "cache_type"))
# 选拼接 2 张、复制移动 1 张（有掩码）
idx = [i for i, r in enumerate(ds.rows) if "CASIA" in r[0].replace("\\", "/")]
picks = []
for typ, n in ((1, 2), (2, 1)):
    c = 0
    for i in idx:
        r = ds.rows[i]
        if r[2] == typ and r[3] and os.path.exists(r[3]):
            picks.append(i); c += 1
            if c == n:
                break

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = FBFNTypeLarge(pretrained=False, img_size=448).to(dev)
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
m.eval()

fig, axes = plt.subplots(len(picks), 4, figsize=(7.2, 1.85 * len(picks)), dpi=300)
titles = ["Input", "Ground truth", "Prediction", "Overlay"]
for j, t in enumerate(titles):
    axes[0, j].set_title(t, fontsize=8, pad=3)

with torch.no_grad():
    for row, i in enumerate(picks):
        path, label, typ, mask_path, has_mask = ds.rows[i]
        x, ent, ela, y, t, mask, hm = ds[i]
        out = m(x[None].to(dev), ela=ela[None].to(dev), entropy=ent[None].to(dev))
        heat = torch.sigmoid(out["final_map"].float())[0, 0].cpu().numpy()
        img = x.permute(1, 2, 0).numpy()
        gt = mask[0].numpy()
        colored = cv2.applyColorMap((heat * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(img * 0.55 + colored * 0.45, 0, 1)
        axes[row, 0].imshow(img)
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(heat, cmap="magma", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay)
        name = "splice" if typ == 1 else "copy-move"
        axes[row, 0].set_ylabel(f"{name}", fontsize=7)
        for j in range(4):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            for s in axes[row, j].spines.values():
                s.set_linewidth(0.4); s.set_color("#888888")

fig.tight_layout(pad=0.3)
out_png = os.path.join(HERE, "figures", "fig3_qualitative.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures", "fig3_qualitative_300dpi.tiff"), dpi=300,
            bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("saved", out_png, picks)
