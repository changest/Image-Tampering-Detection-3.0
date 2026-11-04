# -*- coding: utf-8 -*-
"""生成 v3.0 答辩 PPT（16:9，深色科技风）。"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "FBFN_v3.0_答辩.pptx")

W, H = Inches(13.333), Inches(7.5)
BG = RGBColor(11, 18, 32)
PANEL = RGBColor(20, 30, 50)
ACCENT = RGBColor(0, 200, 255)
ACCENT2 = RGBColor(255, 140, 60)
TEXT = RGBColor(232, 238, 248)
MUTED = RGBColor(150, 165, 190)
FONT = "Microsoft YaHei"


def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def add_rect(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(shape, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def textbox(s, x, y, w, h, text, size=18, color=TEXT, bold=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.name = FONT; r.font.size = Pt(size)
        r.font.bold = bold; r.font.color.rgb = color
    return tb


def bullets(s, x, y, w, h, items, size=17, color=TEXT, gap=6):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.2
        r = p.add_run(); r.text = "▸  " + it
        r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = color
    return tb


def header(s, title, subtitle=None):
    add_rect(s, 0, 0, W, Inches(0.09), ACCENT, shape=MSO_SHAPE.RECTANGLE)
    textbox(s, Inches(0.6), Inches(0.35), Inches(11), Inches(0.7), title,
            size=28, bold=True)
    if subtitle:
        textbox(s, Inches(0.62), Inches(0.95), Inches(11.5), Inches(0.45), subtitle,
                size=14, color=MUTED)


def footer(s, idx):
    textbox(s, Inches(0.6), Inches(7.05), Inches(8), Inches(0.3),
            "FBFN v3.0 · 图像篡改检测", size=11, color=MUTED)
    textbox(s, Inches(12.2), Inches(7.05), Inches(0.7), Inches(0.3),
            str(idx), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def table(s, x, y, w, h, header_row, rows, col_widths=None, font=15):
    shape = s.shapes.add_table(len(rows) + 1, len(header_row), x, y, w, h)
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw
    for j, htxt in enumerate(header_row):
        c = tbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0, 90, 130)
        c.margin_left = c.margin_right = Inches(0.06)
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = htxt
        r.font.name = FONT; r.font.size = Pt(font); r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i + 1, j)
            c.fill.solid()
            c.fill.fore_color.rgb = PANEL if i % 2 == 0 else RGBColor(16, 25, 42)
            c.margin_left = c.margin_right = Inches(0.06)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font); r.font.color.rgb = TEXT
    return tbl


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    idx = 0

    # 1 封面
    s = new_slide(prs); idx += 1
    add_rect(s, 0, 0, W, H, BG, shape=MSO_SHAPE.RECTANGLE)
    add_rect(s, Inches(0.8), Inches(1.7), Inches(0.09), Inches(2.2), ACCENT,
             shape=MSO_SHAPE.RECTANGLE)
    textbox(s, Inches(1.15), Inches(1.6), Inches(10.8), Inches(0.6),
            "图像篡改检测与定位系统", size=42, bold=True)
    textbox(s, Inches(1.15), Inches(2.45), Inches(10.8), Inches(0.6),
            "FBFN 四分支注意力融合网络 · v3.0", size=24, color=ACCENT)
    textbox(s, Inches(1.15), Inches(3.35), Inches(10.8), Inches(0.5),
            "真伪判别 · 篡改类型分类 · 像素级定位 · 鲁棒部署", size=18, color=TEXT)
    textbox(s, Inches(1.15), Inches(5.4), Inches(10), Inches(0.4),
            "智云反诈项目 · Change 团队", size=16, color=MUTED)
    textbox(s, Inches(1.15), Inches(5.85), Inches(10), Inches(0.4),
            "github.com/changest/Image-Tampering-Detection-3.0   |   2026-11", size=13, color=MUTED)

    # 2 问题
    s = new_slide(prs); idx += 1
    header(s, "问题：反诈场景中的图像真实性", "为什么需要篡改检测，而不只是“真假”二分类")
    bullets(s, Inches(0.7), Inches(1.7), Inches(5.8), Inches(4.6), [
        "诈骗案件中的转账截图、聊天记录、证件照、事故照片经常被拼接或复制移动",
        "只看“真/假”无法指导处置，需要知道：改了什么类型、改在哪个区域",
        "真实网络图片经过微信/微博转发，普遍存在 JPEG 压缩与缩放",
        "人工审核成本高，需要可解释、可复核的自动筛查",
    ])
    add_rect(s, Inches(6.9), Inches(1.75), Inches(5.7), Inches(4.4), PANEL)
    textbox(s, Inches(7.25), Inches(1.95), Inches(5.1), Inches(0.5),
            "本项目目标", size=20, bold=True, color=ACCENT)
    bullets(s, Inches(7.2), Inches(2.55), Inches(5.15), Inches(3.4), [
        "输入一张图，输出三件事：",
        "① 真伪概率与判定",
        "② 篡改类型（拼接 / 复制移动 / AI 生成）",
        "③ 像素级热图，标出可疑区域",
        "并保证在压缩、缩放后仍然可用",
    ], size=16)
    footer(s, idx)

    # 3 方法
    s = new_slide(prs); idx += 1
    header(s, "方法：四分支证据融合", "语义证据 + 物理证据，每支输出可检查的证据图")
    pic = os.path.join(FIG, "architecture.png")
    if os.path.exists(pic):
        s.shapes.add_picture(pic, Inches(0.6), Inches(1.75), width=Inches(6.6))
        textbox(s, Inches(0.6), Inches(5.35), Inches(6.7), Inches(0.6),
                "架构示意（v2）：v3.0 将 ViT 分支替换为 DINOv2 ViT-L/14（448×448），并新增类型/二分类头",
                size=11, color=MUTED)
    bullets(s, Inches(7.6), Inches(1.7), Inches(5.2), Inches(4.8), [
        "CNN 分支：ResNet-34 + UNet++，多尺度语义",
        "ViT 分支：DINOv2 ViT-L/14（448×448），全局语义",
        "噪声分支：ELA + SRM，压缩/残差伪影（约 0.04M 参数）",
        "颜色分支：Lab + 局部熵，颜色统计异常",
        "双重注意力融合：通道 + 空间，输出像素级证据图",
    ], size=16)
    footer(s, idx)

    # 4 多任务
    s = new_slide(prs); idx += 1
    header(s, "多任务学习：一个骨干，三个输出", "共享证据特征，类型与定位互相促进")
    table(s, Inches(0.7), Inches(1.75), Inches(6.2), Inches(3.2),
          ["损失项", "作用", "权重"],
          [["L_bin（BCE）", "图级真/假", "0.5"],
           ["L_type（加权 CE）", "真/拼接/复制/AI 四分类", "0.5"],
           ["L_seg（BCE+Dice）", "像素级篡改区域", "1.0"]], font=15)
    bullets(s, Inches(7.3), Inches(1.75), Inches(5.4), Inches(4.6), [
        "从分类模型升级为统一多任务模型，推理只跑一次骨干",
        "类型头与二分类头共享 128 维主干，避免互相拖累",
        "训练数据：CASIA v1/v2 + TIGAS + 合成复制移动/拼接，共 25,469 张",
        "DINOv2-L 448px；先 1×1 降维到 128 再上采样，避免显存溢出",
    ], size=16)
    add_rect(s, Inches(0.7), Inches(5.25), Inches(12), Inches(1.5), PANEL)
    textbox(s, Inches(1.0), Inches(5.45), Inches(11.4), Inches(0.5),
            "评测口径", size=16, bold=True, color=ACCENT)
    textbox(s, Inches(1.0), Inches(5.85), Inches(11.4), Inches(0.8),
            "纯真实 CASIA 验证集 2,866 张（真 1,658 / 拼接 466 / 复制移动 742），不混入 TIGAS 与合成数据",
            size=15)
    footer(s, idx)

    # 5 主结果
    s = new_slide(prs); idx += 1
    header(s, "主结果：类型与定位达到可用水平", "真实 CASIA 验证集；AUC 与阈值/温度无关")
    table(s, Inches(0.7), Inches(1.7), Inches(8.6), Inches(4.3),
          ["检查点", "真伪 AUC", "类型 F1", "分割 mIoU"],
          [["ep5", "0.9230", "0.7179", "0.3323"],
           ["ep10", "0.9555", "0.8019", "0.4830"],
           ["ep14 (EMA)", "0.9672", "0.8251", "—"],
           ["ep20 (EMA)", "0.9634", "0.8492", "0.5614"],
           ["ep30 (clean最佳)", "0.9586", "0.8394", "0.5836"],
           ["last_ema (稳健版)", "0.9502", "0.8410", "0.5844"]],
          col_widths=[Inches(2.5), Inches(2.0), Inches(1.9), Inches(2.2)], font=15)
    add_rect(s, Inches(9.7), Inches(1.75), Inches(3.0), Inches(4.2), PANEL)
    textbox(s, Inches(9.95), Inches(1.95), Inches(2.6), Inches(0.5),
            "关键数字", size=18, bold=True, color=ACCENT)
    bullets(s, Inches(9.9), Inches(2.5), Inches(2.7), Inches(3.2), [
        "类型 F1\n0.718 → 0.849",
        "分割 mIoU\n0.332 → 0.584",
        "旧小骨干对比\nF1 0.818 / mIoU 0.490",
    ], size=15)
    footer(s, idx)

    # 6 鲁棒性
    s = new_slide(prs); idx += 1
    header(s, "鲁棒性：压缩后仍可用", "训练时随机 JPEG / 缩放 / 模糊增强（50% 概率）")
    table(s, Inches(0.7), Inches(1.65), Inches(9.2), Inches(4.5),
          ["条件", "AUC 前→后", "类型 F1 前→后", "mIoU 前→后"],
          [["原始", "0.9586 → 0.9502", "0.8386 → 0.8410", "0.5836 → 0.5844"],
           ["JPEG q90", "0.8497 → 0.8881", "0.6435 → 0.7282", "0.4094 → 0.4895"],
           ["JPEG q70", "0.7597 → 0.8671", "0.6287 → 0.7234", "0.3002 → 0.4046"],
           ["缩放 50%", "0.7826 → 0.9331", "0.5800 → 0.7948", "0.4507 → 0.4925"],
           ["模糊 5×5", "0.6853 → 0.8716", "0.3795 → 0.7283", "0.3526 → 0.4480"],
           ["截图模拟", "0.6448 → 0.7812", "0.4590 → 0.5712", "0.1499 → 0.2023"]],
          col_widths=[Inches(1.9), Inches(2.6), Inches(2.5), Inches(2.4)], font=14)
    add_rect(s, Inches(10.2), Inches(1.7), Inches(2.6), Inches(4.4), PANEL)
    textbox(s, Inches(10.45), Inches(1.9), Inches(2.2), Inches(0.5),
            "要点", size=18, bold=True, color=ACCENT)
    bullets(s, Inches(10.4), Inches(2.45), Inches(2.3), Inches(3.4), [
        "clean 基本无损",
        "模糊 F1 +34.9 点",
        "缩放 AUC +15 点",
        "q90 COPY 召回\n28.8% → 41.1%",
    ], size=14)
    footer(s, idx)

    # 7 效率
    s = new_slide(prs); idx += 1
    header(s, "效率：快慢级联", "Jev 系统一/系统二思路的工程实现")
    table(s, Inches(0.7), Inches(1.75), Inches(7.6), Inches(3.6),
          ["配置", "升级比例", "AUC", "延迟"],
          [["全模型 448", "100%", "0.9503", "30 ms"],
           ["廉价特征快速判", "0%", "0.6746", "3 ms"],
           ["级联（中）", "59%", "0.8959", "19 ms"],
           ["级联（低升级）", "32%", "0.8240", "12 ms"],
           ["级联（高）", "79%", "0.9258", "24 ms"]],
          col_widths=[Inches(2.4), Inches(1.6), Inches(1.6), Inches(1.6)], font=15)
    bullets(s, Inches(8.7), Inches(1.8), Inches(4.0), Inches(4.4), [
        "先用 ELA/熵/颜色统计的线性分类器快速筛查",
        "置信度不足才升级全模型",
        "19 ms 可达 0.896 AUC，节省约 37% 延迟",
        "负结果：案例记忆、分支路由、分辨率路由均未超过该级联",
    ], size=15)
    footer(s, idx)

    # 8 Demo
    s = new_slide(prs); idx += 1
    header(s, "演示：上传即出结果", "本地 RTX 5060 实时推理（last_ema，448px）")
    shot = os.path.join(FIG, "demo_screenshot.png")
    if os.path.exists(shot):
        s.shapes.add_picture(shot, Inches(0.55), Inches(1.55), height=Inches(5.2))
    bullets(s, Inches(9.6), Inches(1.8), Inches(3.2), Inches(4.4), [
        "篡改概率 96.1%",
        "类型：拼接 96.8%",
        "热图框出被替换的花朵",
        "四分支证据图可解释",
        "阈值 0.42 / T=1.849 校准",
    ], size=14)
    footer(s, idx)

    # 9 局限
    s = new_slide(prs); idx += 1
    header(s, "诚实的局限", "哪些场景仍需人工复核")
    table(s, Inches(0.7), Inches(1.7), Inches(11.8), Inches(3.9),
          ["场景", "表现", "建议"],
          [["拼接（大区域）", "召回约 82–86%，精确率约 90%", "可直接用"],
           ["复制移动", "召回约 76%，小区域易漏", "结合热图人工复核"],
           ["截图/二次压缩", "AUC 0.78，定位 mIoU 0.20", "标低置信，转人工"],
           ["GAN 生成图", "接近随机（论文实验）", "不作为主场景"],
           ["未见过的生成器家族", "AUC 66–77%", "需持续扩充训练数据"]],
          col_widths=[Inches(3.2), Inches(4.6), Inches(4.0)], font=14)
    textbox(s, Inches(0.7), Inches(5.9), Inches(11.8), Inches(0.8),
            "设计取舍：优先保证“可用 + 可解释 + 低置信可识别”，而不是追求单一 benchmark 的最高分。",
            size=15, color=ACCENT)
    footer(s, idx)

    # 10 结论
    s = new_slide(prs); idx += 1
    header(s, "结论与展望", "")
    bullets(s, Inches(0.8), Inches(1.7), Inches(6.0), Inches(4.8), [
        "四分支证据融合支持真伪、类型、定位三任务统一",
        "类型 F1 0.849、定位 mIoU 0.584，超过原小骨干",
        "鲁棒微调后压缩/缩放场景大幅提升，clean 基本无损",
        "工程上支持阈值/温度校准与快慢级联",
        "代码与最优权重已开源",
    ], size=17)
    add_rect(s, Inches(7.2), Inches(1.75), Inches(5.5), Inches(4.3), PANEL)
    textbox(s, Inches(7.5), Inches(1.95), Inches(5.0), Inches(0.5),
            "后续工作", size=18, bold=True, color=ACCENT)
    bullets(s, Inches(7.45), Inches(2.5), Inches(4.9), Inches(3.3), [
        "多尺度训练，让低分辨率路由真正可用",
        "复制移动专项：更难合成数据 + 对比学习",
        "扩充真实转发图与截图数据",
        "接入平台，形成“检测 → 复核 → 处置”闭环",
    ], size=15)
    textbox(s, Inches(0.8), Inches(6.0), Inches(11.8), Inches(0.5),
            "github.com/changest/Image-Tampering-Detection-3.0 （分支 v3.0，权重见 Release）",
            size=14, color=MUTED)
    footer(s, idx)

    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
