# -*- coding: utf-8 -*-
"""生成答辩用图表（训练轨迹、鲁棒性前后对比）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "assets", "figures")

FONT = "/c/Windows/Fonts/msyh.ttc"
if os.path.exists("C:/Windows/Fonts/msyh.ttc"):
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

INK = "#1F2A37"
MUTED = "#6B7280"
ACCENT = "#0E7C86"
ORANGE = "#D97706"


def chart_train():
    ep = [5, 10, 14, 20, 30]
    f1 = [0.7179, 0.8019, 0.8251, 0.8492, 0.8394]
    miou = [0.3323, 0.4830, 0.5437, 0.5614, 0.5836]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
    for ax, vals, name, color, y0 in (
            (axes[0], f1, "类型 macro-F1", ACCENT, 0.65),
            (axes[1], miou, "分割 mIoU", ORANGE, 0.30)):
        ax.plot(ep, vals, marker="o", ms=4.5, lw=2.2, color=color)
        ax.set_ylim(y0, max(vals) + 0.06)
        ax.set_xlabel("训练轮次", fontsize=10, color=MUTED)
        ax.set_ylabel(name, fontsize=10, color=MUTED)
        ax.set_xticks(ep)
        ax.grid(axis="y", color="#E5E7EB", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D1D5DB")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.annotate(f"{vals[-1]:.3f}", (ep[-1], vals[-1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=10, color=color, fontweight="bold")
    fig.tight_layout(pad=1.2)
    p = os.path.join(OUT, "chart_train.png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    print(p)


def chart_robust():
    conds = ["原始", "JPEG q90", "JPEG q70", "缩放 50%", "模糊 5×5", "截图模拟"]
    before = [0.9586, 0.8497, 0.7597, 0.7826, 0.6853, 0.6448]
    after = [0.9502, 0.8881, 0.8671, 0.9331, 0.8716, 0.7812]
    y = list(range(len(conds)))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    for yi, b, a in zip(y, before, after):
        ax.plot([b, a], [yi, yi], color="#D1D5DB", lw=3, zorder=1)
        ax.scatter([b], [yi], s=48, color="#9CA3AF", zorder=2)
        ax.scatter([a], [yi], s=58, color=ACCENT, zorder=3)
        ax.text(a + 0.006, yi, f"{a:.3f}", va="center", fontsize=9.5, color=ACCENT)
        ax.text(b - 0.006, yi, f"{b:.3f}", va="center", ha="right", fontsize=9, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=10, color=INK)
    ax.set_xlim(0.60, 1.00)
    ax.set_xlabel("二分类 AUC", fontsize=10, color=MUTED)
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=48, color="#9CA3AF", label="增强前 (ep30_raw)")
    ax.scatter([], [], s=58, color=ACCENT, label="增强后 (last_ema)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, "chart_robust.png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    chart_train()
    chart_robust()
