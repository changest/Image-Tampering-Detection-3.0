# -*- coding: utf-8 -*-
"""Fig. 1 (Nature-style schematic): FBFN v3 architecture, 180 mm width, editable text."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
GREEN = "#8BCF8B"
RED = "#B64342"
TEAL = "#42949E"
GRAY = "#767676"
DARK = "#272727"
LIGHT = "#E9EEF5"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

# 180 mm x 82 mm canvas
fig = plt.figure(figsize=(180 / 25.4, 82 / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 180)
ax.set_ylim(0, 82)
ax.axis("off")

# evidence sheet crops
EV = Image.open(os.path.join(HERE, "..", "paper", "figures", "evidence_map.png")).convert("RGB")
W, H = EV.size
COLS = {  # (x0, x1) fractions
    "input": (0.020, 0.165),
    "cnn": (0.176, 0.320),
    "vit": (0.326, 0.470),
    "noise": (0.476, 0.620),
    "color": (0.626, 0.770),
    "fused": (0.776, 0.920),
}
ROWS = {"real": (0.115, 0.485), "fake": (0.545, 0.925)}


def crop(name, row="fake"):
    x0, x1 = COLS[name]; y0, y1 = ROWS[row]
    return EV.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))


def box(x, y, w, h, ec=BLUE, fc="white", lw=0.6, rad=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rad}",
                                linewidth=lw, edgecolor=ec, facecolor=fc))


def label(x, y, s, fs=7, color=DARK, weight="normal", ha="left", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va)


def arrow(x1, y1, x2, y2, color=GRAY, lw=0.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=5.5, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


# panel label
label(2, 78.5, "a", fs=8, weight="bold")

# input
ax.imshow(crop("input"), extent=(3, 21, 34, 58), aspect="auto", zorder=1)
ax.add_patch(Rectangle((3, 34), 18, 24, fill=False, ec=DARK, lw=0.6, zorder=2))
label(12, 30.5, "Input", fs=7, ha="center")
label(12, 27.5, "448 × 448 × 3", fs=6.2, color=GRAY, ha="center")

# branches
rows = [
    ("CNN", "ResNet-34 + UNet++", "cnn", BLUE),
    ("ViT", "DINOv2 ViT-L/14", "vit", TEAL),
    ("Noise", "ELA + SRM", "noise", RED),
    ("Color", "Lab + local entropy", "color", "#B08A3E"),
]
ys = [55, 44.5, 34, 23.5]
for (name, sub, key, color), y in zip(rows, ys):
    ax.add_patch(Rectangle((31, y), 1.1, 9, fc=color, ec="none"))
    label(34, y + 6.3, name, fs=7, weight="bold")
    label(34, y + 3.2, sub, fs=6.2, color=GRAY)
    ax.imshow(crop(key), extent=(65, 77, y + 0.7, y + 8.3), aspect="auto", zorder=1)
    ax.add_patch(Rectangle((65, y + 0.7), 12, 7.6, fill=False, ec=GRAY, lw=0.4, zorder=2))
    arrow(21.5, 46, 30.8, y + 4.5, lw=0.5)
    arrow(77.3, y + 4.5, 82, 41, lw=0.5)

label(71, 20.5, "Evidence maps (1 ch)", fs=6.2, color=GRAY, ha="center")

# fusion
box(82, 29, 25, 24, ec=BLUE, fc=LIGHT, lw=0.7)
label(94.5, 47.5, "Dual-attention", fs=7, weight="bold", ha="center")
label(94.5, 44.3, "fusion", fs=7, weight="bold", ha="center")
label(94.5, 38.5, "channel + spatial", fs=6.2, color=GRAY, ha="center")
label(94.5, 33.5, "4 × 448 × 448", fs=6.2, color=GRAY, ha="center")

# fused map
ax.imshow(crop("fused"), extent=(110, 121, 35, 47), aspect="auto", zorder=1)
ax.add_patch(Rectangle((110, 35), 11, 12, fill=False, ec=GRAY, lw=0.4, zorder=2))
label(115.5, 32, "Fused map", fs=6.2, color=GRAY, ha="center")
arrow(107.2, 41, 109.8, 41)

# trunk
box(124, 35, 16, 12, ec=GRAY, fc="#F4F4F4", lw=0.6)
label(132, 43.5, "128-d shared", fs=7, weight="bold", ha="center")
label(132, 40.2, "trunk", fs=7, weight="bold", ha="center")
arrow(121.2, 41, 123.8, 41)

# heads
heads = [
    ("Authenticity", "real / fake", "BCE", 55),
    ("Type", "real / splice / copy-move / AI", "weighted CE", 38.5),
    ("Localization", "pixel mask", "BCE + Dice", 22),
]
for name, sub, loss, y in heads:
    box(143, y, 33, 10.5, ec=BLUE, fc="white", lw=0.6)
    label(145, y + 7.2, name, fs=7, weight="bold")
    label(145, y + 3.9, sub, fs=6.0, color=GRAY)
    label(145, y + 1.5, loss, fs=5.6, color=GRAY)
    arrow(140.2, 41, 142.8, y + 5.25, lw=0.5)

# loss note
label(90, 12, r"$\mathcal{L} = \mathcal{L}_{seg} + 0.5\,\mathcal{L}_{bin} + 0.5\,\mathcal{L}_{type}$",
      fs=8.6, ha="center")
label(90, 7.5, "Pixel-level supervision: BCE + Dice;  type: class-weighted cross-entropy",
      fs=6.2, color=GRAY, ha="center")

os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "fig1_architecture_v3")
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.02)
fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight", pad_inches=0.02,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("saved", base)
