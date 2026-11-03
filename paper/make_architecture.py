"""生成 FBFN 架构图（Figure 1）：四分支 + 双注意力融合。

运行：python paper/make_architecture.py
输出：paper/figures/architecture.png + .pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_BRANCH = {"CNN": "#4C72B0", "ViT": "#55A868", "Noise": "#C44E52", "Color": "#DD8452"}
C_IO = "#E0E0E0"
C_FUSION = "#8172B2"
C_MAP = "#F0E68C"


def box(ax, x, y, w, h, text, fc, ec="black", fs=9, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                        linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # ---- 输入 ----
    box(ax, 0.005, 0.40, 0.115, 0.20, "Input RGB\n(256×256)", C_IO, fs=10)

    # ---- 四个分支（竖排）----
    branches = [
        ("CNN", "CNN Branch\nUNet++ / ResNet-34"),
        ("ViT", "ViT Branch\nSwin-T + SegFormer"),
        ("Noise", "Noise Branch\nELA + SRM filters"),
        ("Color", "Color Branch\nLab + local entropy"),
    ]
    bx, bw, bh = 0.165, 0.185, 0.155
    ys = [0.815, 0.565, 0.315, 0.065]
    for (key, label), by in zip(branches, ys):
        box(ax, bx, by, bw, bh, label, C_BRANCH[key], fs=9)

    # ---- 证据图（每个分支右侧）----
    mx, mw, mh = 0.385, 0.075, 0.14
    for by in ys:
        box(ax, mx, by + 0.01, mw, mh, "Evidence\nmap", C_MAP, fs=8)

    # ---- 融合 ----
    box(ax, 0.505, 0.365, 0.185, 0.27,
        "Dual Attention\nFusion\n(channel + spatial)", C_FUSION, fs=9.5)

    # ---- 输出 ----
    box(ax, 0.725, 0.42, 0.115, 0.16, "Fused\nmap", C_MAP, fs=9)
    box(ax, 0.865, 0.42, 0.13, 0.16, "Global avg\npooling", C_IO, fs=8.5)

    # ---- 箭头 ----
    for by in ys:
        arrow(ax, 0.120, 0.50, bx - 0.005, by + bh / 2)      # input -> branch
        arrow(ax, bx + bw, by + bh / 2, mx - 0.005, by + 0.08)  # branch -> map
        arrow(ax, mx + mw, by + 0.08, 0.505, 0.50)           # map -> fusion
    arrow(ax, 0.690, 0.50, 0.725, 0.50)                       # fusion -> fused
    arrow(ax, 0.840, 0.50, 0.865, 0.50)                       # fused -> GAP

    # ---- 输出标签 ----
    ax.text(0.93, 0.30, "Image-level score\n(real / fake)", ha="center", va="center",
            fontsize=9.5, style="italic")
    arrow(ax, 0.93, 0.42, 0.93, 0.345, lw=1.2)

    # ---- 图例说明：分支颜色 ----
    legend_y = 0.02
    ax.text(0.02, legend_y + 0.06, "Semantic branches", fontsize=8, color="#4C72B0", weight="bold")
    ax.text(0.02, legend_y, "Artifact branches (physically interpretable evidence)",
            fontsize=8, color="#C44E52", weight="bold")

    for ext in ("png", "pdf"):
        out = os.path.join(FIG_DIR, f"architecture.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white" if ext == "png" else "auto")
        print(f"保存 {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
