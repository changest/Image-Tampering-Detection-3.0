# -*- coding: utf-8 -*-
"""基于 guizang-ppt-skill 模板生成瑞士风单文件 HTML 演示（IKB 主题）。"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.expanduser(r"~/.claude/skills/guizang-ppt-skill")
TPL = os.path.join(SKILL, "assets", "template-swiss.html")
OUT = os.path.join(HERE, "index.html")

TOTAL = 10
SANS = "font-family:var(--sans),var(--sans-zh)"


def chrome(n, left, right=None):
    right = right or f"{n:02d} / {TOTAL:02d}"
    return (f'<div class="chrome-min"><div class="l">{left}</div>'
            f'<div class="r">{right}</div></div>')


def head(meta, title, size="min(6.4vw,11vh)"):
    return (f'<div data-anim="head" style="display:flex;flex-direction:column;gap:1.4vh">'
            f'<div class="t-meta">{meta}</div>'
            f'<h2 style="{SANS};font-weight:200;font-size:{size};line-height:.98;'
            f'letter-spacing:-.035em">{title}</h2></div>')


S = []

# ---------- 01 封面 ----------
S.append(f'''
<section class="slide accent" data-layout="SWISS-COVER-ASCII" data-animate="hero" data-slide-id="cover">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    {chrome(1, "FBFN · 图像篡改检测 · v3.0", "SS · 2026.11 · 01 / 10")}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.6vh">
      <div data-anim="kicker" class="t-meta" style="color:rgba(var(--accent-on-rgb),.78);letter-spacing:.22em">IMAGE FORENSICS · DEFENSE</div>
      <h1 data-anim="title" style="{SANS};align-self:center;font-weight:200;font-size:min(9.6vw,17vh);line-height:.96;letter-spacing:-.03em;color:var(--accent-on)">图像篡改<br/>检测与定位</h1>
      <div data-anim="bottom" style="display:grid;grid-template-rows:auto auto;gap:1.6vh;border-top:1px solid rgba(var(--accent-on-rgb),.22);padding-top:2vh">
        <div data-anim="lead" class="lead" style="max-width:54ch;color:rgba(var(--accent-on-rgb),.86);font-weight:300">一个骨干给出真伪、类型与区域，压缩转发后仍可用。</div>
        <div style="display:flex;justify-content:space-between;align-items:end">
          <div class="t-meta" style="color:rgba(var(--accent-on-rgb),.6)">智云反诈项目 · Change 团队 · 2026-11</div>
          <div class="t-meta" style="color:rgba(var(--accent-on-rgb),.6)">→ swipe / arrow keys</div>
        </div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 02 问题 ----------
S.append(f'''
<section class="slide split" data-layout="S03" data-animate="split-statement" data-slide-id="problem">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;justify-content:space-between">
        {chrome(2, "PROBLEM")}
        <div data-anim="stmt" style="{SANS};font-weight:200;font-size:min(6.8vw,12vh);line-height:1.02;letter-spacing:-.04em">
          只看真伪，<br/>不够支撑<br/>处置。
        </div>
        <div class="t-meta" style="color:var(--text-helper)">02 / 10 · WHY TYPE AND REGION MATTER</div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 4.4vh;background:var(--grey-1);justify-content:center">
        <div data-anim="copy" style="display:flex;flex-direction:column;gap:3vh;max-width:44ch">
          <span class="t-cat accent">反诈审核的真实需求</span>
          <p class="lead" style="font-weight:300">一张转账截图被判定“疑似伪造”之后，审核员还要回答三个问题。</p>
          <div style="display:flex;flex-direction:column;gap:1.6vh;border-top:1px solid var(--line)">
            <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding-top:1.6vh"><span class="t-meta">01</span><p class="t-body">改的是金额区域，还是整块拼接了别人的凭证？</p></div>
            <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding-top:1.6vh;border-top:1px solid var(--line)"><span class="t-meta">02</span><p class="t-body">这是拼接、复制移动，还是 AI 生成？</p></div>
            <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;padding-top:1.6vh;border-top:1px solid var(--line)"><span class="t-meta">03</span><p class="t-body">图经过微信转发，模型还能不能信？</p></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 03 三个难点 ----------
S.append(f'''
<section class="slide light" data-layout="S05" data-animate="stack-build" data-slide-id="frictions">
  <div class="canvas-card">
    {chrome(3, "FRICTIONS")}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:4vh">
      {head("THREE FRICTIONS", "难点不在模型大小，<br/>而在分布与部署")}
      <div class="stack-row">
        <div class="stack-block b-ink" data-anim="block">
          <div class="layer-nb">FRICTION 01</div>
          <div class="layer-ttl">来源不一致</div>
          <div class="layer-desc">CASIA、TIGAS 与合成样本的压缩历史、分辨率、噪声分布不同，模型容易学到数据集的捷径。</div>
          <div class="layer-tag">DATA DISTRIBUTION</div>
        </div>
        <div class="stack-block b-grey" data-anim="block">
          <div class="layer-nb">FRICTION 02</div>
          <div class="layer-ttl">任务相互牵制</div>
          <div class="layer-desc">只训二分类会丢掉类型信息；只训分割又缺少图级判断。三个任务需要共享同一套证据。</div>
          <div class="layer-tag">MULTI-TASK</div>
        </div>
        <div class="stack-block b-accent" data-anim="block">
          <div class="layer-nb">FRICTION 03</div>
          <div class="layer-ttl">部署条件苛刻</div>
          <div class="layer-desc">真实案件图片普遍经过 JPEG 压缩、缩放甚至截图，干净验证集上的高分并不等于可用。</div>
          <div class="layer-tag">ROBUSTNESS</div>
        </div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 04 架构 ----------
S.append(f'''
<section class="slide light" data-layout="S22" data-animate="image-hero" data-slide-id="architecture">
  <div class="canvas-card" style="padding:0;display:flex;flex-direction:column;overflow:hidden">
    <div data-anim="img" style="position:relative;flex:0 0 58%;overflow:hidden;background:var(--grey-1)">
      <img src="images/04-architecture.png" alt="FBFN 四分支架构" loading="eager" data-image-slot="s22-hero-21x9"
           style="position:absolute;left:16%;top:0;width:84%;height:100%;object-fit:contain;object-position:center center;background:var(--paper)">
      <div class="chrome-min" style="position:absolute;top:0;left:0;right:0;color:var(--text-primary);padding:5.6vh 5vw 0">
        <div class="l">ARCHITECTURE · V2</div>
        <div class="r">04 / 10</div>
      </div>
      <div data-anim="title-block" style="position:absolute;left:2.6vw;bottom:10vh;background:var(--paper);padding:2vh 1.4vw;max-width:15vw">
        <div style="{SANS};font-weight:200;font-size:min(2.8vw,5vh);line-height:1;letter-spacing:-.035em">四支证据<br/>一次判决</div>
      </div>
    </div>
    <div data-anim="kpi" class="image-hero-body" style="padding-top:4.2vh">
      <div style="max-width:50ch;{SANS};font-size:max(15px,1.15vw);line-height:1.55;font-weight:300">
        CNN 与 ViT 提供语义证据，ELA/SRM 与 Lab 熵提供物理证据；融合图同时给出图级分数和像素级定位。v3.0 将 ViT 分支替换为 DINOv2 ViT-L/14（448×448）。
      </div>
      <div class="image-hero-stats" style="gap:4vw">
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">BRANCHES</div><div style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.6vh);line-height:.95;letter-spacing:-.04em">4</div><p class="body-sm">语义 + 物理证据</p></div>
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">INPUT</div><div style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.6vh);line-height:.95;letter-spacing:-.04em">448</div><p class="body-sm">DINOv2 patch 对齐</p></div>
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">OUTPUTS</div><div style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.6vh);line-height:.95;letter-spacing:-.04em;color:var(--accent-display)">3</div><p class="body-sm">真伪 · 类型 · 区域</p></div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 05 主结果 KPI 塔 ----------
S.append(f'''
<section class="slide light" data-layout="S06" data-animate="measure-up" data-slide-id="results">
  <div class="canvas-card">
    {chrome(5, "RESULTS")}
    <div style="flex:1;padding:0 0 7vh 0;display:grid;grid-template-columns:5fr 7fr;gap:3vw;align-items:stretch">
      <div style="display:flex;flex-direction:column;gap:2.4vh;justify-content:center">
        <div class="t-meta">PURE CASIA VALIDATION · 2,866</div>
        <h2 style="{SANS};font-weight:200;font-size:min(6.2vw,10.6vh);line-height:.96;letter-spacing:-.035em">类型与定位<br/>达到可用线</h2>
        <p class="lead" style="font-weight:300;max-width:38ch">相比 ep5：类型 F1 +13.1，定位 mIoU +25.2。</p>
        <div class="t-help" style="color:var(--text-helper)">256px 小骨干：F1 0.818→0.849，mIoU 0.490→0.584；AUC 略低。</div>
      </div>
      <div class="bar-towers" style="margin-top:0">
        <div class="bar-tower" data-anim="tower"><div class="cap"><span class="t-meta">TYPE</span></div><div class="body-block h-3 b-accent"><div class="lbl">MACRO-F1</div><div class="nb">0.849</div><div class="sub">四类加权，ep20 EMA</div></div></div>
        <div class="bar-tower" data-anim="tower"><div class="cap"><span class="t-meta">SEG</span></div><div class="body-block h-2"><div class="lbl">MIOU</div><div class="nb">0.584</div><div class="sub">像素级定位，ep30</div></div></div>
        <div class="bar-tower" data-anim="tower"><div class="cap"><span class="t-meta">AUC</span></div><div class="body-block h-3"><div class="lbl">BINARY</div><div class="nb">0.963</div><div class="sub">真伪判别，ep20 EMA</div></div></div>
        <div class="bar-tower" data-anim="tower"><div class="cap"><span class="t-meta">COPY</span></div><div class="body-block h-2"><div class="lbl">RECALL</div><div class="nb">0.765</div><div class="sub">复制移动，最弱项</div></div></div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 06 训练轨迹 ----------
nodes = [
    ("EP 05", "0.718", "类型 F1 起点"),
    ("EP 10", "0.802", "多任务开始收敛"),
    ("EP 14", "0.825", "EMA 首次反超"),
    ("EP 20", "0.849", "分类最佳"),
    ("EP 30", "0.839", "分割最佳 0.584"),
]
timeline = "".join(
    f'''<div class="tl-node" data-anim="node"><div class="tl-axis"><span class="dot"></span></div>
  <span class="yr">{a}</span><span class="multi">{b}</span>
  <p class="desc">{c}</p></div>''' for a, b, c in nodes)
S.append(f'''
<section class="slide light" data-layout="S02" data-animate="timeline-walk" data-slide-id="training" style="--tl-pad:1.2vh">
  <style>#deck .slide[data-slide-id="training"] .tl-node{{padding:1.2vh 0}} #deck .slide[data-slide-id="training"] .timeline-v{{margin-top:2vh}}</style>
  <div class="canvas-card">
    {chrome(6, "TRAINING")}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:3vh">
      {head("TYPE F1 OVER EPOCHS", "类型与定位同步收敛")}
      <div style="display:grid;grid-template-columns:8fr 4fr;gap:4vw;align-items:start">
        <div class="timeline-v">{timeline}</div>
        <div style="display:flex;flex-direction:column;gap:2.2vh;border-top:2px solid var(--accent);padding-top:2vh">
          <div class="t-meta">READING</div>
          <p class="t-body" style="font-weight:300">类型与定位同步提升，说明共享证据能同时服务分类与定位，而不是此消彼长。</p>
          <div style="border-top:1px solid var(--line);padding-top:1.6vh"><div class="t-meta">OVERSHOOT</div><p class="t-body">30 轮后类型指标回落，混合分布上开始过拟合。</p></div>
          <div style="border-top:1px solid var(--line);padding-top:1.6vh"><div class="t-meta">DEPLOY</div><p class="t-body">分类取 ep20 EMA，分割取 ep30。</p></div>
        </div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 07 鲁棒性 h-bar ----------
bars = [
    ("原始", 0.9502, "87%", ""),
    ("JPEG q90", 0.8881, "71%", "accent"),
    ("JPEG q70", 0.8671, "65%", ""),
    ("缩放 50%", 0.9331, "82%", ""),
    ("模糊 5×5", 0.8716, "66%", ""),
    ("截图模拟", 0.7812, "42%", ""),
]
rows = "".join(
    f'<div class="row-lbl">{n}</div>'
    f'<div class="row-track"><div class="row-fill {c}" style="width:{w}"></div></div>'
    f'<div class="row-val">{v:.3f}</div>' for n, v, w, c in bars)
S.append(f'''
<section class="slide light" data-layout="S07" data-animate="bar-grow" data-slide-id="robustness">
  <div class="canvas-card">
    {chrome(7, "ROBUSTNESS")}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.4vh">
      {head("AUC AFTER JPEG / RESIZE / BLUR", "压缩增强后，q90 的 AUC 到 0.89")}
      <div style="display:grid;grid-template-columns:7fr 4fr;gap:4vw;align-items:center">
        <div class="h-bar-chart">{rows}</div>
        <div style="display:flex;flex-direction:column;gap:1.8vh">
          <div style="border-top:2px solid var(--accent);padding-top:1.4vh"><div class="t-meta">CLEAN</div><div class="t-body">原始指标基本不变：F1 0.839→0.841，mIoU 0.584→0.584</div></div>
          <div style="border-top:1px solid var(--line);padding-top:1.4vh"><div class="t-meta">GAIN</div><div class="t-body">模糊条件类型 F1 +34.9 点；缩放 AUC +15 点</div></div>
          <div style="border-top:1px solid var(--line);padding-top:1.4vh"><div class="t-meta">COPY RECALL</div><div class="t-body">q90 从 28.8% 提升到 41.1%，仍是弱项</div></div>
        </div>
      </div>
      <div class="t-helper" style="color:var(--text-helper)">训练时以 50% 概率施加 JPEG(q50–95)、缩放(0.5–1.0) 与高斯模糊，并重算 ELA 与熵图。数据：同一 CASIA 验证集。</div>
    </div>
  </div>
</section>''')

# ---------- 08 Demo ----------
S.append(f'''
<section class="slide light" data-layout="S22" data-animate="image-hero" data-slide-id="demo">
  <div class="canvas-card" style="padding:0;display:flex;flex-direction:column;overflow:hidden">
    <div class="chrome-min" style="padding:5.6vh 5vw 0;flex:0 0 auto">
      <div class="l">DEMO · LOCAL RTX 5060</div>
      <div class="r">08 / 10</div>
    </div>
    <div data-anim="img" style="position:relative;flex:0 0 52%;overflow:hidden;background:var(--paper);margin-top:1.6vh">
      <img src="images/08-demo.png" alt="Demo 界面：上传图片得到判定、类型与热图" loading="eager" data-image-slot="s22-hero-21x9"
           style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center center;background:var(--paper)">
    </div>
    <div data-anim="kpi" class="image-hero-body" style="padding:3vh 5vw 7vh;flex:0 0 auto">
      <div style="max-width:50ch;{SANS};font-size:max(15px,1.1vw);line-height:1.55;font-weight:300">
        Gradio 单页应用，本机 RTX 5060 实时推理；上传后同时给出真伪概率、类型分布、热图与四分支证据图。
      </div>
      <div class="image-hero-stats" style="gap:4vw">
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">TAMPER PROB</div><div style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.6vh);line-height:.95;letter-spacing:-.04em;color:var(--accent-display)">96.1%</div><p class="body-sm">阈值 0.42 校准</p></div>
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">SPLICE</div><div style="font-family:var(--sans);font-weight:200;font-size:min(4.6vw,7.6vh);line-height:.95;letter-spacing:-.04em">96.8%</div><p class="body-sm">类型预测</p></div>
        <div style="display:flex;flex-direction:column;gap:.6vh"><div style="height:1px;background:var(--ink)"></div><div class="t-meta">MODEL</div><div style="font-family:var(--sans);font-weight:200;font-size:min(2.6vw,4.6vh);line-height:1;letter-spacing:-.02em">last_ema</div><p class="body-sm">448 px · 鲁棒版</p></div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 09 局限 ----------
S.append(f'''
<section class="slide light" data-layout="S04" data-animate="grid-reveal" data-slide-id="limits">
  <div class="canvas-card">
    {chrome(9, "BOUNDARIES")}
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.6vh">
      {head("WHERE IT STILL FAILS", "边界清晰，低置信交给人工")}
      <div class="sub-grid-3-2" style="padding-bottom:6vh">
        <div class="sub-card accent" data-anim="card"><span class="nb-corner">01</span><div class="ttl">拼接 · 可直接用</div><div class="desc">召回 82–86%，精确率约 90%，大区域拼接最稳。</div></div>
        <div class="sub-card" data-anim="card"><span class="nb-corner">02</span><div class="ttl">复制移动 · 需复核</div><div class="desc">召回约 76%，小区域和相似纹理容易漏。</div></div>
        <div class="sub-card" data-anim="card"><span class="nb-corner">03</span><div class="ttl">截图 · 低置信</div><div class="desc">AUC 0.78、定位 mIoU 0.20，建议转人工。</div></div>
        <div class="sub-card" data-anim="card"><span class="nb-corner">04</span><div class="ttl">GAN · 非主场景</div><div class="desc">BigGAN 接近随机，模型主要针对扩散生成。</div></div>
        <div class="sub-card" data-anim="card"><span class="nb-corner">05</span><div class="ttl">新生成器 · 待补数据</div><div class="desc">未见家族 AUC 66–77%，需持续扩充训练集。</div></div>
        <div class="sub-card ink" data-anim="card"><span class="nb-corner" style="color:rgba(var(--paper-rgb),.6)">06</span><div class="ttl">产品策略</div><div class="desc">把“低置信”当作一等公民：输出置信度与热图，让人工复核有依据。</div></div>
      </div>
    </div>
  </div>
</section>''')

# ---------- 10 收尾 ----------
S.append(f'''
<section class="slide split" data-layout="S10" data-animate="split-statement" data-slide-id="closing">
  <div class="canvas-card">
    <div class="split-half">
      <div class="half b-accent" style="padding:5.6vh 3.6vw 9vh;justify-content:space-between;position:relative;overflow:hidden">
        <canvas class="ascii-bg" aria-hidden="true"></canvas>
        <div class="chrome-min" style="margin-bottom:0;position:relative;z-index:1"><div class="l">10 / 10</div><div class="r">CLOSING</div></div>
        <div data-anim="manifesto" style="display:flex;flex-direction:column;gap:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(var(--accent-on-rgb),.78);letter-spacing:.22em">MANIFESTO</div>
          <h2 style="{SANS};font-size:min(7.4vw,13vh);line-height:.96;letter-spacing:-.03em;font-weight:200;color:var(--accent-on)">可解释，<br/>可部署。</h2>
          <div style="font-size:max(min(14px,1.944vh),1vw);line-height:1.6;color:rgba(var(--accent-on-rgb),.82);font-weight:400;max-width:34ch">一个骨干统一三个任务；压缩增强保证真实转发图可用；低置信可以识别并转人工。</div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:end;border-top:1px solid rgba(var(--accent-on-rgb),.22);padding-top:2vh;position:relative;z-index:1">
          <div class="t-meta" style="color:rgba(var(--accent-on-rgb),.62)">Change 团队 · 2026-11</div>
          <div class="t-meta" style="color:rgba(var(--accent-on-rgb),.62)">→ END</div>
        </div>
      </div>
      <div class="half" style="padding:5.6vh 3.6vw 9vh;justify-content:space-between">
        <div class="chrome-min"><div class="l">TAKEAWAYS</div><div class="r">03 RULES</div></div>
        <div data-anim="rules" style="display:flex;flex-direction:column;gap:0">
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.6vh 0;border-top:1px solid var(--line)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(4.4vw,7.8vh);line-height:.9">01</div>
            <div><h3 style="{SANS};font-weight:400;font-size:min(1.8vw,3.2vh);line-height:1.2;margin-bottom:1vh">统一三任务</h3><p class="t-body">共享证据特征，一次推理给出真伪、类型与区域。</p></div>
          </div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.6vh 0;border-top:1px solid var(--line)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(4.4vw,7.8vh);line-height:.9">02</div>
            <div><h3 style="{SANS};font-weight:400;font-size:min(1.8vw,3.2vh);line-height:1.2;margin-bottom:1vh">压缩后仍可用</h3><p class="t-body">增强训练让 q90 AUC 从 0.85 到 0.89，clean 基本无损。</p></div>
          </div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:2vw;align-items:start;padding:2.6vh 0;border-top:1px solid var(--line);border-bottom:2px solid var(--accent)">
            <div style="font-family:var(--sans);font-weight:200;font-size:min(4.4vw,7.8vh);line-height:.9;color:var(--accent-ink)">03</div>
            <div><h3 style="{SANS};font-weight:400;font-size:min(1.8vw,3.2vh);line-height:1.2;margin-bottom:1vh;color:var(--accent-ink)">低置信可识别</h3><p class="t-body">阈值与温度校准，输出可复核的置信度与热图。</p></div>
          </div>
        </div>
        <div class="t-meta" style="color:var(--text-helper);text-align:right">github.com/changest/Image-Tampering-Detection-3.0</div>
      </div>
    </div>
  </div>
</section>''')

NOTES = '''const SPEAKER_NOTES = [
  {id:'cover', title:'图像篡改检测与定位', section:'开场', minutes:0.8, purpose:'建立主题与预期', talk:['反诈场景不只需要真假，还需要类型与区域','本工作用一套骨干完成三件事','强调所有数字来自纯真实 CASIA 验证集'], transition:'进入问题页', cue:'停顿，确认观众屏已同步'},
  {id:'problem', title:'只看真伪不够', section:'问题', minutes:1.2, purpose:'说明业务需求', talk:['用转账截图举例：假只是第一步','审核员需要知道改了什么、改在哪里','转发压缩是现实条件'], transition:'引出三个难点', cue:'指向右侧三条需求'},
  {id:'frictions', title:'三个难点', section:'问题', minutes:1.5, purpose:'定义技术挑战', talk:['数据分布不一致','任务之间相互牵制','部署条件苛刻'], transition:'进入方法', cue:''},
  {id:'architecture', title:'四支证据一次判决', section:'方法', minutes:2.0, purpose:'讲清架构', talk:['四个分支分别负责什么','注意力融合如何给出像素级证据','v3 换成 DINOv2-L 448，并新增类型头'], transition:'进入结果', cue:'指出图上的 Evidence map'},
  {id:'results', title:'主结果', section:'结果', minutes:2.0, purpose:'给出核心数字', talk:['类型 F1 0.849，mIoU 0.584','相比起点提升 13 和 25 个点','AUC 略低于小骨干，解释取舍'], transition:'进入训练过程', cue:''},
  {id:'training', title:'同步收敛', section:'结果', minutes:1.5, purpose:'说明训练动态', talk:['类型和定位同步提升','30 轮后分类过拟合','部署时分类取 ep20，分割取 ep30'], transition:'进入鲁棒性', cue:''},
  {id:'robustness', title:'压缩后仍可用', section:'结果', minutes:2.5, purpose:'展示最有说服力的增强', talk:['增强前 q90 掉到 0.85，复制召回 29%','增强后 q90 到 0.888，模糊 F1 +35 点','clean 指标几乎不变'], transition:'进入 Demo', cue:'强调这是真实转发条件'},
  {id:'demo', title:'Demo', section:'演示', minutes:2.0, purpose:'现场展示', talk:['上传拼接图','概率 96.1%，类型拼接 96.8%','热图框住被替换的花朵'], transition:'进入局限', cue:'可现场操作一遍'},
  {id:'limits', title:'边界', section:'讨论', minutes:1.5, purpose:'诚实交代边界', talk:['拼接最稳，复制移动需复核','截图与二次压缩仍弱','低置信显式输出，转人工'], transition:'进入收尾', cue:''},
  {id:'closing', title:'收尾', section:'收束', minutes:1.0, purpose:'收束并给出行动点', talk:['一个骨干统一三任务','压缩增强保证可用','代码与权重已开源'], transition:'致谢', cue:'停顿 2 秒'}
];'''


FIX = """
<style>
[data-theme="ikb"]{--accent:#002FA7;--accent-ink:#002FA7;--accent-bright:#5B7BFF;--accent-deep:#001d6b;--accent-display:#002FA7;--accent-soft:#d5deef;--accent-rgb:0,47,167;--accent-on:#ffffff;--accent-on-rgb:255,255,255;--accent-on-text:#ffffff;--accent-on-text-rgb:255,255,255}
</style>
</head>"""

html = open(TPL, encoding="utf-8").read()
html = html.replace('data-theme="lemon"', 'data-theme="ikb"')
html = re.sub(r'data-theme="[^"]+"', 'data-theme="ikb"', html, count=1)

start = html.find("<!-- SLIDES_HERE")
end = html.find("</div>\n\n<div id=\"nav\">", start)
if start < 0 or end < 0:
    raise SystemExit("找不到模板插入区")
deck = "\n\n".join(S) + "\n\n"
html = html[:start] + "<!-- SLIDES: guizang swiss · generated -->\n" + deck + html[end:]

# 替换演讲备注数组
nstart = html.find("const SPEAKER_NOTES = [")
nend = html.find("];", nstart)
html = html[:nstart] + NOTES + html[nend + 2:]

html = html.replace("</head>", FIX, 1)
open(OUT, "w", encoding="utf-8").write(html)
print("saved", OUT, len(html))
