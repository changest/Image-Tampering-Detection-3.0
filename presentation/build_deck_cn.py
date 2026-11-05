# -*- coding: utf-8 -*-
"""按 nature-paper2ppt 规范生成中文答辩 PPT（methods / problem-to-solution）。"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "output", "assets", "figures")
OUT = os.path.join(HERE, "output", "final_presentation_cn.pptx")

W, H = Inches(13.333), Inches(7.5)
BG = RGBColor(247, 249, 252)
INK = RGBColor(31, 42, 55)
MUTED = RGBColor(107, 114, 128)
ACCENT = RGBColor(14, 124, 134)
ORANGE = RGBColor(217, 119, 6)
WHITE = RGBColor(255, 255, 255)
PANEL = RGBColor(255, 255, 255)
BORDER = RGBColor(229, 231, 235)
DARK = RGBColor(15, 23, 42)
FONT = "Microsoft YaHei"

L = Inches(0.6)
TITLE_Y = Inches(0.42)
CONTENT_Y = Inches(1.55)
BOTTOM = Inches(6.95)


def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def text(s, x, y, w, h, t, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.2, font=FONT):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    for i, line in enumerate(t.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.name = font; r.font.size = Pt(size)
        r.font.bold = bold; r.font.color.rgb = color
    return tb


def bullets(s, x, y, w, h, items, size=15, color=INK, gap=8, lead="• "):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.2
        r = p.add_run(); r.text = lead + it
        r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = color
    return tb


def rule(s, x, y, w, color=BORDER, h=Pt(1)):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def header(s, title, kicker=""):
    rule(s, L, Inches(0.30), Inches(0.5), ACCENT, Pt(4))
    text(s, L + Inches(0.62), TITLE_Y, Inches(11.6), Inches(0.62), title,
         size=25, bold=True)
    if kicker:
        text(s, L + Inches(0.62), Inches(1.03), Inches(11.6), Inches(0.38), kicker,
             size=12, color=MUTED)


def footer(s, idx, src=""):
    rule(s, L, Inches(7.05), Inches(12.1), RGBColor(229, 231, 235))
    text(s, L, Inches(7.12), Inches(9.0), Inches(0.28), src, size=8.5, color=MUTED)
    text(s, Inches(12.2), Inches(7.12), Inches(0.5), Inches(0.28), str(idx),
         size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def table(s, x, y, w, header_row, rows, col_ratios=None, font=14,
          row_h=Inches(0.42), header_h=Inches(0.46)):
    shape = s.shapes.add_table(len(rows) + 1, len(header_row), x, y, w,
                               header_h + row_h * len(rows))
    tbl = shape.table
    if col_ratios:
        total = sum(col_ratios)
        for i, r in enumerate(col_ratios):
            tbl.columns[i].width = int(w * r / total)
    tbl.rows[0].height = header_h
    for i in range(len(rows)):
        tbl.rows[i + 1].height = row_h
    for j, htxt in enumerate(header_row):
        c = tbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = ACCENT
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_left = c.margin_right = Inches(0.05)
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = htxt
        r.font.name = FONT; r.font.size = Pt(font); r.font.bold = True
        r.font.color.rgb = WHITE
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i + 1, j)
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 == 0 else RGBColor(243, 246, 250)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.05)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font); r.font.color.rgb = INK
    return tbl


def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt


def pic(s, path, x, y, w=None, h=None, **kw):
    if w is not None:
        kw["width"] = w
    if h is not None:
        kw["height"] = h
    return s.shapes.add_picture(path, x, y, **kw)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1 封面
    s = slide(prs); n += 1
    s.background.fill.solid(); s.background.fill.fore_color.rgb = DARK
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.55),
                             Inches(0.07), Inches(2.9))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    text(s, Inches(1.35), Inches(1.45), Inches(10.8), Inches(0.7),
         "图像篡改检测与定位系统", size=40, bold=True, color=WHITE)
    text(s, Inches(1.35), Inches(2.32), Inches(10.8), Inches(0.5),
         "FBFN 四分支注意力融合网络 · v3.0", size=22, color=RGBColor(94, 234, 212))
    text(s, Inches(1.35), Inches(3.15), Inches(10.8), Inches(0.45),
         "真伪判别 · 篡改类型 · 像素级定位", size=16, color=RGBColor(203, 213, 225))
    rule(s, Inches(1.35), Inches(5.35), Inches(4.0), RGBColor(51, 65, 85))
    text(s, Inches(1.35), Inches(5.5), Inches(9.0), Inches(0.35),
         "智云反诈项目 · Change 团队", size=14, color=RGBColor(148, 163, 184))
    text(s, Inches(1.35), Inches(5.95), Inches(10.0), Inches(0.35),
         "github.com/changest/Image-Tampering-Detection-3.0     2026-11", size=11,
         color=RGBColor(100, 116, 139))
    notes(s, "开场：本工作面向反诈取证场景，解决的不只是“图是不是假的”，"
             "而是“改了什么、改在哪里”，并且要能在压缩转发后仍然可用。"
             "整个结果基于纯真实 CASIA 验证集，代码与权重已开源。")

    # 2 背景
    s = slide(prs); n += 1
    header(s, "只看真伪，不足以支撑处置")
    text(s, L, Inches(1.75), Inches(8.6), Inches(1.3),
         "一张转账截图被判定“疑似伪造”，\n审核人员仍需知道：改的是金额区域，\n还是整块拼接了别人的凭证。",
         size=23, spacing=1.4)
    bullets(s, Inches(9.35), Inches(1.85), Inches(3.3), Inches(3.8), [
        "类型决定处置流程",
        "区域决定证据链",
        "置信度决定是否转人工",
    ], size=15)
    rule(s, L, Inches(4.3), Inches(12.1), BORDER)
    text(s, L, Inches(4.55), Inches(12.0), Inches(0.4),
         "反诈场景需要的是一次输出三件事，而不是一个孤立的真假分数。",
         size=15, color=ACCENT)
    text(s, L, Inches(5.25), Inches(12.0), Inches(0.4),
         "微信、微博等平台还会二次压缩与缩放，模型必须在这种条件下仍然可用。",
         size=15, color=MUTED)
    footer(s, n)
    notes(s, "先讲清需求：真实业务里“真/假”只是第一步。"
             "如果模型只说假，却不能给出类型和区域，审核员无法判断是金额篡改还是整图伪造，"
             "证据链也无法闭环。另外，诈骗图片几乎都经过转发压缩，这是必须面对的现实条件。")

    # 3 问题
    s = slide(prs); n += 1
    header(s, "真正的难点是跨源泛化与可复核", "训练集与真实案件之间存在明显分布差")
    items = [
        ("数据来源不同", "CASIA、TIGAS 与合成样本的压缩历史、分辨率、噪声分布不一致"),
        ("任务相互牵制", "只训二分类会丢掉类型；只训分割又缺少图级判断"),
        ("部署条件苛刻", "转发图存在 JPEG 压缩、缩放、截图，指标会断崖式下降"),
    ]
    y = Inches(1.8)
    for i, (t1, t2) in enumerate(items, 1):
        text(s, L, y, Inches(0.45), Inches(0.45), f"0{i}", size=20, bold=True, color=ACCENT)
        text(s, L + Inches(0.65), y - Inches(0.03), Inches(3.1), Inches(0.4), t1,
             size=17, bold=True)
        text(s, L + Inches(3.8), y - Inches(0.03), Inches(7.9), Inches(0.8), t2,
             size=14, color=MUTED, spacing=1.3)
        y += Inches(1.25)
    rule(s, L, Inches(5.75), Inches(12.1), BORDER)
    text(s, L, Inches(5.95), Inches(12.0), Inches(0.8),
         "因此我们选择：一个共享骨干 + 多任务输出 + 压缩增强训练，\n"
         "评测只看纯真实数据，不用合成集抬高数字。", size=15, color=INK, spacing=1.4)
    footer(s, n)
    notes(s, "这一页交代三个难点。第一是数据分布：不同来源的图压缩方式和分辨率差别很大。"
             "第二是任务冲突：单任务模型要么只会分真假，要么只会画区域。"
             "第三是部署：真实转发图经过压缩和缩放，干净验证集上的高分并不代表可用。"
             "我们的应对是共享骨干、多任务联合训练，并在训练中加入压缩扰动。")

    # 4 架构
    s = slide(prs); n += 1
    header(s, "四支证据图经注意力融合为统一判决")
    p = os.path.join(FIG, "architecture.png")
    pic(s, p, Inches(2.05), Inches(1.5), width=Inches(9.2))
    text(s, L, Inches(6.45), Inches(12.1), Inches(0.35),
         "CNN 与 ViT 提供语义证据，ELA/SRM 与 Lab 熵提供物理证据；融合图同时用于像素级定位。",
         size=11, color=MUTED)
    footer(s, n, "改绘自本项目论文 Fig. 1（v2 示意）；v3.0 将 ViT 分支替换为 DINOv2 ViT-L/14（448×448）")
    notes(s, "架构沿用四分支：CNN 抓多尺度语义，ViT 抓全局结构，噪声分支用 ELA 和 SRM 抓压缩残差，"
             "颜色分支用 Lab 和局部熵抓颜色统计异常。四张证据图通过通道与空间注意力融合，"
             "既给出图级分数，也保留像素级定位。v3.0 的主要改动是把 ViT 换成 DINOv2-L，并在 448 分辨率训练。")

    # 5 关键设计
    s = slide(prs); n += 1
    header(s, "一个骨干同时给出真伪、类型与区域", "共享证据特征，避免三个模型重复计算")
    # 简易示意：左共享特征 → 右三头
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(2.5),
                             Inches(3.2), Inches(1.7))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(226, 242, 243)
    box.line.color.rgb = ACCENT
    text(s, Inches(1.05), Inches(2.72), Inches(2.9), Inches(1.25),
         "共享证据特征\n(4 分支融合 · 128 维主干)", size=14, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    heads = [("真伪", "二分类头 · BCE"), ("类型", "4 类加权 CE"), ("区域", "BCE + Dice")]
    for i, (t1, t2) in enumerate(heads):
        y = Inches(1.85) + Inches(1.45) * i
        arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.25), y + Inches(0.42),
                                   Inches(0.75), Inches(0.28))
        arrow.fill.solid(); arrow.fill.fore_color.rgb = RGBColor(148, 163, 184)
        arrow.line.fill.background()
        hb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.2), y,
                                Inches(2.6), Inches(1.05))
        hb.fill.solid(); hb.fill.fore_color.rgb = WHITE
        hb.line.color.rgb = BORDER
        text(s, Inches(5.35), y + Inches(0.1), Inches(2.3), Inches(0.85),
             f"{t1}\n{t2}", size=15, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
    bullets(s, Inches(8.35), Inches(1.9), Inches(4.4), Inches(4.2), [
        "总损失：1.0·分割 + 0.5·真伪 + 0.5·类型",
        "类型头与二分类头共享主干，互不拖累",
        "DINOv2-L 先降维到 128 再上采样，控制显存",
        "推理只跑一次骨干，三个结果同时输出",
    ], size=14.5)
    footer(s, n)
    notes(s, "多任务的设计重点在“共享但不互相干扰”。四个分支融合后接一个 128 维主干，"
             "再分叉出二分类、类型、分割三个头；二分类和类型共享主干，但各自有独立输出层。"
             "损失权重是 1 : 0.5 : 0.5，分割为主，分类为辅。工程上 DINOv2-L 的 1024 通道先降到 128，"
             "再做上采样，否则 448 分辨率会爆显存。")

    # 6 评测设置
    s = slide(prs); n += 1
    header(s, "评测只用纯真实 CASIA，避免合成数据抬高指标")
    table(s, Inches(0.75), Inches(1.7), Inches(11.8),
          ["项目", "设置", "说明"],
          [["训练数据", "25,469 张", "CASIA v1/v2 + TIGAS + 合成复制移动/拼接"],
           ["验证数据", "2,866 张", "纯真实 CASIA：真 1,658 / 拼接 466 / 复制移动 742"],
           ["评价指标", "AUC · macro-F1 · mIoU", "真伪、类型、像素级定位分别评估"],
           ["增强设置", "随机 JPEG / 缩放 / 模糊", "50% 概率施加，扰动后重算 ELA 与熵图"]],
          col_ratios=[1.3, 2.1, 4.4], font=14, row_h=Inches(0.62))
    text(s, L, Inches(5.55), Inches(12.0), Inches(0.8),
         "验证集不混入 TIGAS 或合成样本，所有数字均为本项目实测。\n"
         "阈值与温度只在验证集上拟合，并在部署说明中单独标注。", size=12.5, color=MUTED,
         spacing=1.35)
    footer(s, n)
    notes(s, "评测口径需要特别说明：很多工作会在合成或同源验证集上报高分，"
             "但真实案件里没有这些数据。所以我们只在纯真实 CASIA 上评测，2,866 张，"
             "真图 1,658、拼接 466、复制移动 742。指标分别看 AUC、类型 macro-F1 和分割 mIoU。"
             "鲁棒性测试是在同一验证集上施加 JPEG、缩放、模糊扰动。")

    # 7 主结果
    s = slide(prs); n += 1
    header(s, "类型 F1 达 0.849，定位 mIoU 达 0.603")
    table(s, Inches(0.75), Inches(1.65), Inches(11.8),
          ["检查点", "真伪 AUC", "类型 macro-F1", "分割 mIoU"],
          [["ep5", "0.9230", "0.7179", "0.3323"],
           ["ep10", "0.9555", "0.8019", "0.4830"],
           ["ep14（EMA）", "0.9672", "0.8251", "0.5437"],
           ["ep20（EMA）", "0.9634", "0.8492", "0.5614"],
           ["ep30（clean 最佳）", "0.9586", "0.8394", "0.5836"],
           ["last_ema（稳健版）", "0.9502", "0.8410", "0.5844"]],
          col_ratios=[2.6, 1.9, 1.9, 1.9], font=15, row_h=Inches(0.52))
    text(s, L, Inches(5.55), Inches(12.0), Inches(0.4),
         "相比起点 ep5：类型 F1 +13.1 点，定位 mIoU +27.1 点（TTA）。", size=15,
         color=ACCENT, bold=True)
    text(s, L, Inches(6.05), Inches(12.0), Inches(0.4),
         "256px 小骨干：F1 0.818 → 0.849，mIoU 0.490 → 0.603；AUC 0.977 → 0.969。",
         size=12.5, color=MUTED)
    text(s, L, Inches(6.45), Inches(12.0), Inches(0.4),
         "最终配置：ep20_ema + 专用二分类头 → AUC 0.969；ep30_raw + TTA(θ=0.40) → mIoU 0.603。",
         size=12, color=ACCENT)
    footer(s, n, "数据来源：纯真实 CASIA 验证集 2,866 张，本项目实测")
    notes(s, "主结果看三列：AUC、类型 F1、mIoU。随着训练推进，类型和定位持续提升，"
             "到 ep20 类型 F1 到 0.849，ep30+TTA 定位 mIoU 到 0.603。"
             "需要诚实说明：二分类 AUC 0.96，略低于原来的 256px 小骨干 0.977，"
             "但类型和定位都全面超过。这与大骨干的收益集中在细粒度判别一致。")

    # 8 训练轨迹
    s = slide(prs); n += 1
    header(s, "多任务训练让类型与定位同步收敛")
    pic(s, os.path.join(FIG, "chart_train.png"), Inches(0.7), Inches(1.75), width=Inches(8.2))
    bullets(s, Inches(9.25), Inches(1.85), Inches(3.5), Inches(4.2), [
        "类型 F1 从 0.718 升到 0.849",
        "mIoU 从 0.332 升到 0.603（TTA）",
        "30 轮后分类开始过拟合",
    ], size=14)
    text(s, Inches(9.25), Inches(5.2), Inches(3.4), Inches(0.8),
         "部署选择：分类用 ep20 EMA，\n分割用 ep30。", size=12.5, color=MUTED,
         spacing=1.35)
    footer(s, n, "数据来源：CASIA 验证集逐检查点评测")
    notes(s, "这张图说明类型和定位是同步收敛的，并不是牺牲一个换另一个。"
             "到 30 轮以后类型指标开始回落，说明对混合分布过拟合，所以部署时分类取 ep20 的 EMA，"
             "分割取 ep30 的权重。这也解释了为什么最后没有一味训到更多轮。")

    # 9 鲁棒性
    s = slide(prs); n += 1
    header(s, "压缩增强把 JPEG q90 的 AUC 推到 0.89")
    pic(s, os.path.join(FIG, "chart_robust.png"), Inches(0.65), Inches(1.7), width=Inches(8.0))
    bullets(s, Inches(9.0), Inches(1.85), Inches(3.8), Inches(4.3), [
        "clean 指标基本不变",
        "模糊条件类型 F1 +34.9 点",
        "缩放 AUC 提升 15 点",
        "q90 复制移动召回 28.8% → 41.1%",
    ], size=14)
    text(s, Inches(9.0), Inches(5.55), Inches(3.6), Inches(0.9),
         "训练时以 50% 概率施加 JPEG(q50–95)、\n缩放(0.5–1.0) 与高斯模糊。", size=11.5,
         color=MUTED, spacing=1.3)
    footer(s, n, "数据来源：同一 CASIA 验证集施加扰动后评测；增强前后对比")
    notes(s, "这是我们最有说服力的一页。增强前，普通 JPEG q90 就把 AUC 从 0.96 打到 0.85，"
             "复制移动召回从 77% 掉到 29%，真实转发场景基本不可用。"
             "训练时加入随机压缩、缩放和模糊，并重新计算 ELA 与熵图之后，"
             "q90 AUC 到 0.888，缩放条件到 0.93，模糊条件类型 F1 提升近 35 点，"
             "而干净数据上的指标几乎不变。")

    # 10 效率
    s = slide(prs); n += 1
    header(s, "19 ms 可达 0.896 AUC，节省委约三成延迟")
    table(s, Inches(0.75), Inches(1.7), Inches(7.6),
          ["配置", "升级比例", "AUC", "平均延迟"],
          [["全模型 448", "100%", "0.9503", "30 ms"],
           ["廉价特征快筛", "0%", "0.6746", "3 ms"],
           ["级联（低升级）", "32%", "0.8240", "12 ms"],
           ["级联（中）", "59%", "0.8959", "19 ms"],
           ["级联（高）", "79%", "0.9258", "24 ms"]],
          col_ratios=[2.4, 1.5, 1.4, 1.6], font=14, row_h=Inches(0.5))
    bullets(s, Inches(8.75), Inches(1.8), Inches(4.0), Inches(4.4), [
        "先用 ELA/熵/颜色统计线性分类",
        "置信度不足再升级大模型",
        "同等延迟下优于记忆校准",
        "延迟按论文口径：3 ms / 30 ms",
    ], size=13.5)
    text(s, Inches(0.75), Inches(5.15), Inches(11.8), Inches(0.7),
         "负结果：案例记忆、分支路由、分辨率路由均未超过这条简单级联；\n"
         "路由要成立需多尺度重训，收益主要是省算力。", size=12, color=MUTED, spacing=1.35)
    text(s, Inches(0.75), Inches(6.3), Inches(11.8), Inches(0.4),
         "部署：bf16 推理在 A40 上模型 34 ms、端到端 81 ms（fp32 128 ms）；完整 CASIA 精度差异 ≤0.0005。",
         size=12, color=ACCENT)
    footer(s, n, "延迟口径：RTX 5060 Laptop，系统一 3 ms / 全模型 30 ms")
    notes(s, "效率部分借鉴了“快思考 + 慢思考”的思路，但工程上最简单的形式就是快慢级联："
             "先用 ELA、熵、颜色统计训练一个线性分类器，3 毫秒出结果，"
             "不确定才升级全模型。19 毫秒可以拿到 0.896 AUC。"
             "我们也试过案例记忆和分辨率路由，都没有超过这条曲线，这是可以公开的负结果。"
             "部署时用 bf16 推理：A40 上模型 34 毫秒、端到端 81 毫秒，完整验证集精度差异不超过 0.0005。")

    # 11 五档能力映射
    s = slide(prs); n += 1
    header(s, "五档能力映射：从快筛到旗舰", "借用 Claude 模型命名，按能力与场景给不同权重定位")
    table(s, Inches(0.75), Inches(1.7), Inches(11.8),
          ["档位", "气质", "对应模型", "使用场景"],
          [["Haiku", "极简 · 轻量 · 高效", "廉价特征级联", "大批量初筛，3–19 ms"],
           ["Sonnet", "均衡 · 稳定 · 默认", "last_ema", "在线检测默认档"],
           ["Opus", "旗舰 · 上限最高", "ep30_raw", "离线复核与论文基准"],
           ["Fable", "亲和 · 会讲清楚", "ep20_ema", "篡改类型解释"],
           ["Mythos", "体系化 · 完整叙事", "FBFN v3.0 整套系统", "四支证据 + 多任务 + 鲁棒部署"]],
          col_ratios=[1.5, 2.8, 3.0, 4.2], font=14, row_h=Inches(0.62))
    text(s, L, Inches(6.15), Inches(12.0), Inches(0.4),
         "命名只是类比：五档共享同一套证据结构，区别在用途与工作点，而不是模型规模。",
         size=12, color=MUTED)
    footer(s, n)
    notes(s, "这一页借用 Claude 的命名，把四种权重和整套系统按能力场景定位。"
             "Haiku 是级联快筛，Sonnet 是默认的 last_ema，Opus 是 clean 上限最高的 ep30，"
             "Fable 是擅长讲清类型的 ep20_ema，Mythos 则是完整的 FBFN v3.0 系统。"
             "重点强调：这不是说它们规模不同，而是同源模型在不同工作点上的取舍。")

    # 12 Demo
    s = slide(prs); n += 1
    header(s, "上传即得判定、类型与热图", "本地 RTX 5060 或 A40 服务器（bf16，端到端 81 ms）")
    pic(s, os.path.join(FIG, "demo_screenshot.png"), Inches(0.6), Inches(1.55),
        height=Inches(5.25))
    bullets(s, Inches(8.05), Inches(1.8), Inches(4.7), Inches(4.5), [
        "篡改概率 99.9%，专用头阈值 0.5",
        "类型：拼接 96.9%",
        "热图框出被替换的花朵区域",
        "四分支证据图同步展示",
        "专用头与阈值在 CASIA 上校准",
    ], size=14.5)
    footer(s, n, "Demo：fbfn.demo_app（Gradio），权重 last_ema.pth")
    notes(s, "这是实际运行的界面，不是示意图。上传一张拼接图，右侧立刻给出 99.9% 的篡改概率、"
             "类型是拼接、以及热图；热图准确框住了被替换的花朵。下面四张图是各分支的证据图，"
             "答辩时可以现场演示，也可以指出哪一支在支撑判断。")

    # 12 局限
    s = slide(prs); n += 1
    header(s, "边界清晰：复制移动与截图仍需复核")
    table(s, Inches(0.75), Inches(1.7), Inches(11.8),
          ["场景", "实测表现", "使用建议"],
          [["拼接（大区域）", "召回 82–86%，精确率约 90%", "可直接使用"],
           ["复制移动", "召回约 76%，小区域易漏", "结合热图人工复核"],
           ["截图 / 二次压缩", "AUC 0.78，定位 mIoU 0.20", "标为低置信，转人工"],
           ["GAN 生成图", "接近随机（论文实验）", "不作为主场景"],
           ["未见生成器家族", "AUC 66–77%", "持续扩充训练数据"]],
          col_ratios=[2.5, 4.2, 3.5], font=14, row_h=Inches(0.6))
    text(s, L, Inches(5.75), Inches(12.0), Inches(0.7),
         "产品策略：把“低置信”当作一等公民，输出置信度与热图，\n"
         "让人工复核有依据，而不是让模型假装全能。", size=14, color=ACCENT, spacing=1.35)
    footer(s, n)
    notes(s, "局限必须主动讲。拼接最稳，可以直接用；复制移动召回约 76%，小区域容易漏，"
             "需要配合热图人工看；截图和二次压缩仍然弱，AUC 0.78、定位 mIoU 只有 0.20；"
             "GAN 图和没见过的生成器家族不在主场景内。"
             "产品上我们把这些情况显式标成低置信，而不是强行给一个高置信结论。")

    # 13 总结
    s = slide(prs); n += 1
    header(s, "可解释、可部署、可持续迭代")
    bullets(s, L, Inches(1.75), Inches(7.4), Inches(4.2), [
        "一个骨干统一真伪、类型与定位",
        "类型 F1 0.849，定位 mIoU 0.603（TTA）",
        "压缩增强后 q90 AUC 0.85 → 0.89",
        "支持阈值校准与快慢级联",
        "代码与最优权重已开源",
    ], size=16, gap=10)
    text(s, Inches(8.4), Inches(1.8), Inches(4.3), Inches(4.0),
         "后续工作\n\n多尺度训练，让路由真正可用\n\n"
         "复制移动专项：更难合成数据\n\n接入平台，形成检测—复核—处置闭环",
         size=13, color=MUTED, spacing=1.4)
    rule(s, L, Inches(6.05), Inches(12.1), BORDER)
    text(s, L, Inches(6.2), Inches(12.0), Inches(0.35),
         "github.com/changest/Image-Tampering-Detection-3.0（分支 v3.0，权重见 Release）",
         size=12, color=ACCENT)
    footer(s, n)
    notes(s, "总结：我们不是追求单一 benchmark 的最高分，而是做一个在真实条件下可用的系统："
             "一个骨干同时给出真伪、类型和区域，压缩增强保证转发图可用，低置信可以识别并转人工。"
             "代码和权重都已开源，欢迎复现和指正。")

    prs.save(OUT)
    print("saved", OUT, "slides", len(prs.slides))


if __name__ == "__main__":
    build()
