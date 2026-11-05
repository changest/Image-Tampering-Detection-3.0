# -*- coding: utf-8 -*-
"""生成 JVCIR 投稿用 v3 架构图（300 dpi）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, w, h, text, fc="#f2f2f2", ec="#444444", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                linewidth=0.8, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.8, color="#555555",
                                 shrinkA=0, shrinkB=0))


# input
box(0.2, 3.0, 2.0, 2.4, "Input RGB\n448 × 448", fc="#ffffff", bold=True, fs=9)

# branches
by = [6.55, 4.95, 3.35, 1.75]
labels = [
    ("CNN branch\nResNet-34 + UNet++", "#dbe8f7"),
    ("ViT branch\nDINOv2 ViT-L/14", "#d7efd9"),
    ("Noise branch\nELA + SRM", "#f8d9d9"),
    ("Color branch\nLab + local entropy", "#fbe3cc"),
]
for y, (txt, c) in zip(by, labels):
    box(2.8, y, 3.9, 1.1, txt, fc=c, fs=8.5)
    arrow(2.22, 4.2, 2.78, y + 0.55)
    box(7.1, y + 0.05, 1.75, 1.0, "Evidence\nmap (1 ch)", fc="#fff8d6", fs=8)
    arrow(6.72, y + 0.55, 7.08, y + 0.55)

# fusion
box(9.3, 3.25, 2.7, 2.4, "Dual attention\nfusion\n(channel + spatial)", fc="#e4ddf2",
    bold=True, fs=8.5)
for y in by:
    arrow(8.87, y + 0.55, 9.28, 4.45)

# fused map
box(12.5, 4.0, 1.55, 1.7, "Fused\nmap", fc="#fff2b3", fs=8.5)
arrow(12.02, 4.45, 12.48, 4.45)

# shared trunk
box(14.5, 3.9, 1.65, 1.9, "Shared\n128-d\ntrunk", fc="#eeeeee", bold=True, fs=8.5)
arrow(14.07, 4.45, 14.48, 4.45)

# heads
box(16.65, 6.3, 2.5, 1.15, "Authenticity head\n(BCE)", fc="#e8e8e8", fs=8)
box(16.65, 4.45, 2.5, 1.15, "Type head\n(4 classes)", fc="#e8e8e8", fs=8)
box(16.65, 2.55, 2.5, 1.15, "Localization head\n(BCE + Dice)", fc="#e8e8e8", fs=8)
for yc in (6.87, 5.02, 3.12):
    arrow(16.17, 4.85, 16.63, yc)

# output labels
ax.text(17.9, 7.62, "real / fake", ha="center", fontsize=7.5, style="italic")
ax.text(17.9, 5.78, "real / splice /\ncopy-move / AI", ha="center", fontsize=7.5, style="italic")
ax.text(17.9, 3.88, "pixel mask", ha="center", fontsize=7.5, style="italic")

ax.text(0.2, 0.45,
        "All four branches produce pixel-level evidence; the fused map is used for localization,\n"
        "while a shared 128-d trunk feeds the authenticity and type heads. Trained with BCE + Dice + weighted CE.",
        fontsize=7.5, color="#333333")

fig.tight_layout(pad=0.2)
os.makedirs(OUT, exist_ok=True)
png = os.path.join(OUT, "fig1_architecture_v3.png")
tif = os.path.join(OUT, "fig1_architecture_v3_300dpi.tiff")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(tif, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("saved", png)
