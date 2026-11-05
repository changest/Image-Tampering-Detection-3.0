# -*- coding: utf-8 -*-
"""JVCIR Fig.1：Keynote 信息图风格的 FBFN v3 架构图。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

NAVY = "#16294A"
BLUE = "#2E5B9A"
LIGHT = "#E8F0F8"
PALE = "#F7FAFD"
BEIGE = "#F6EBD0"
GREEN = "#4E8A3E"
RED = "#B02418"
GRAY = "#555555"

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(11.6, 7.2), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")


def rbox(x, y, w, h, fc, ec=BLUE, lw=1.0, rad=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rad}",
                                linewidth=lw, edgecolor=ec, facecolor=fc))


def txt(x, y, s, fs=9, color=NAVY, style="normal", ha="left", va="center",
        weight="normal", family="DejaVu Serif", ls=1.15):
    ax.text(x, y, s, fontsize=fs, color=color, style=style, ha=ha, va=va,
            fontweight=weight, family=family, linespacing=ls)


def arrow(x1, y1, x2, y2, color=NAVY, lw=1.4, ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def cross(x, y, s=0.4, c=RED, lw=3):
    ax.plot([x - s, x + s], [y - s, y + s], color=c, lw=lw, solid_capstyle="round")
    ax.plot([x - s, x + s], [y + s, y - s], color=c, lw=lw, solid_capstyle="round")


def check(x, y, s=0.4, c=GREEN, lw=3):
    ax.plot([x - s, x - 0.1 * s, x + s], [y, y - 0.7 * s, y + 0.7 * s],
            color=c, lw=lw, solid_capstyle="round", solid_joinstyle="round")


# ---------------- top narrative ----------------
txt(0.35, 12.25, "The forensic gap — why type and region matter", fs=14, style="italic")
txt(12.5, 12.28,
    "Fraud images are recompressed and resized in social media;\n"
    "a scalar real/fake score cannot say what changed or where.",
    fs=8.6, style="italic", color=GRAY)

# contrast row (left)
rbox(0.4, 9.95, 3.9, 1.55, PALE)
txt(2.35, 11.05, "real / fake only", fs=10, ha="center", color=GRAY)
txt(2.35, 10.55, "where? what type?\ncan I trust it after JPEG?", fs=8.0, ha="center", color=GRAY)
cross(3.75, 11.15, s=0.32)

arrow(4.45, 10.7, 5.15, 10.7, color=RED, lw=2.0, ms=14)

rbox(5.3, 9.95, 4.25, 1.55, "#EAF3E6", ec=GREEN)
txt(7.42, 11.05, "type + region + confidence", fs=10, ha="center", color=GREEN)
txt(7.42, 10.55, "one backbone, three outputs,\ncalibrated probability", fs=8.0, ha="center", color=GRAY)
check(9.85, 11.18, s=0.30)

# evidence strip inset (top right)
ev = Image.open(os.path.join(HERE, "..", "paper", "figures", "evidence_map.png")).convert("RGB")
w, h = ev.size
panel = w // 5
strip_w = 1.05
for i in range(5):
    im = ev.crop((i * panel, 0, (i + 1) * panel, h))
    ax.imshow(im, extent=(10.55 + i * strip_w, 11.4 + i * strip_w, 9.95, 11.55),
              aspect="auto", zorder=2)
    ax.add_patch(Rectangle((10.55 + i * strip_w, 9.95), strip_w - 0.05, 1.6,
                           fill=False, ec=BLUE, lw=0.6, zorder=3))
txt(13.3, 9.55, "branch evidence: real (top) vs. tampered (bottom)", fs=7.2, ha="center", color=GRAY)

# ---------------- pipeline ----------------
txt(0.35, 9.15, "Evidence branches", fs=12, style="italic", color=BLUE)
txt(0.35, 8.85, "four physically interpretable streams", fs=8.0, style="italic", color=GRAY)

branches = [
    ("CNN", "ResNet-34 + UNet++", LIGHT),
    ("ViT", "DINOv2 ViT-L/14", "#E6F2E4"),
    ("Noise", "ELA + SRM filters", "#F8E2E0"),
    ("Color", "Lab + local entropy", BEIGE),
]
by = [7.75, 6.8, 5.85, 4.9]
for i, ((name, sub, c), y) in enumerate(zip(branches, by)):
    rbox(0.4, y, 3.0, 0.78, c)
    txt(0.58, y + 0.51, name, fs=9.2, weight="bold")
    txt(0.58, y + 0.22, sub, fs=7.4, color=GRAY)
    # evidence thumbnail
    im = ev.crop((i * panel, 0, (i + 1) * panel, h))
    ax.imshow(im, extent=(3.6, 4.5, y + 0.06, y + 0.72), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((3.6, y + 0.06), 0.9, 0.66, fill=False, ec=BLUE, lw=0.6, zorder=3))
    arrow(3.42, y + 0.39, 3.58, y + 0.39, lw=0.8, ms=6)
    arrow(4.52, y + 0.39, 4.95, y + 0.39, lw=0.8, ms=6)

# fusion
rbox(5.0, 5.65, 2.55, 2.15, "#E6E2F2", ec="#5B4B8A")
txt(6.27, 7.05, "Dual attention\nfusion", fs=10, ha="center", style="italic", color="#3B2F63")
txt(6.27, 6.3, "channel + spatial", fs=8.0, ha="center", color=GRAY)
arrow(4.56, 6.72, 4.98, 6.72, lw=0.9, ms=8)

# trunk
rbox(7.9, 5.85, 1.7, 1.75, "#EDEDED", ec=GRAY)
txt(8.75, 6.95, "Shared\n128-d\ntrunk", fs=9.0, ha="center", weight="bold")
arrow(7.57, 6.72, 7.87, 6.72, lw=1.1)

# outputs
outs = [
    ("Authenticity", "real / fake", "BCE", 8.2, LIGHT),
    ("Type", "real / splice / copy-move / AI", "weighted CE", 6.55, "#E6F2E4"),
    ("Localization", "pixel mask", "BCE + Dice", 4.9, BEIGE),
]
for name, sub, loss, y, c in outs:
    rbox(9.95, y, 3.3, 1.2, c)
    txt(10.12, y + 0.83, name, fs=9.2, weight="bold")
    txt(10.12, y + 0.5, sub, fs=7.4, color=GRAY)
    txt(10.12, y + 0.21, loss, fs=7.0, style="italic", color=BLUE)
    arrow(9.62, 6.72, 9.92, y + 0.6, lw=0.8, ms=7)

# ---------------- robustness mini chart ----------------
txt(0.35, 4.25, "Robust after distribution", fs=12, style="italic", color=BLUE)
txt(0.35, 3.95, "AUC on real CASIA, before → after augmentation", fs=8.0, style="italic", color=GRAY)
conds = [("clean", 0.950, 0.950), ("JPEG q90", 0.850, 0.888), ("resize", 0.783, 0.933), ("blur", 0.685, 0.872)]
base = 2.35
for i, (name, before, after) in enumerate(conds):
    x = 0.55 + i * 1.25
    ax.add_patch(Rectangle((x + 0.12, base), 0.28, (before - 0.6) * 1.4, fc="#C9D6E8", ec="none"))
    ax.add_patch(Rectangle((x + 0.44, base), 0.28, (after - 0.6) * 1.4, fc=BLUE, ec="none"))
    txt(x + 0.42, base - 0.16, name, fs=7.0, ha="center", color=GRAY)
    txt(x + 0.24, base + (before - 0.6) * 1.4 + 0.09, f"{before:.2f}", fs=6.0, ha="center", color=GRAY)
    txt(x + 0.64, base + (after - 0.6) * 1.4 + 0.09, f"{after:.2f}", fs=6.0, ha="center", color=BLUE)
ax.plot([0.5, 5.5], [base, base], color=GRAY, lw=0.8)

# ---------------- metric cards ----------------
txt(6.0, 4.25, "What the model delivers", fs=12, style="italic", color=BLUE)
txt(6.0, 3.95, "same protocol, real-only validation", fs=8.0, style="italic", color=GRAY)
cards = [("AUC", "0.969", "vs. TruFor 0.946"), ("Type F1", "0.849", "4-class macro"),
         ("Localization", "0.603", "mIoU (TTA)"), ("Latency", "81 ms", "A40, bf16")]
for i, (lab, val, sub) in enumerate(cards):
    x = 6.0 + i * 2.2
    rbox(x, 2.35, 2.0, 1.2, LIGHT)
    txt(x + 1.0, 3.28, lab, fs=7.2, ha="center", color=GRAY, style="italic")
    txt(x + 1.0, 2.98, val, fs=13, ha="center", color=NAVY, weight="bold")
    txt(x + 1.0, 2.6, sub, fs=6.6, ha="center", color=GRAY)

# ---------------- bottom banner ----------------
rbox(0.35, 1.05, 19.3, 0.62, "#F9E5E3", ec=RED, lw=1.4)
txt(10.0, 1.36, "A tampering triage tool — not a substitute for forensic examination.",
    fs=10.5, ha="center", color=RED, weight="bold")

txt(10.0, 0.45,
    "Interpretable four-branch evidence fusion for image tampering detection, type classification and localization",
    fs=12, ha="center", style="italic", color=NAVY)

fig.tight_layout(pad=0.1)
os.makedirs(OUT, exist_ok=True)
png = os.path.join(OUT, "fig1_architecture_v3.png")
tif = os.path.join(OUT, "fig1_architecture_v3_300dpi.tiff")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(tif, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("saved", png)
