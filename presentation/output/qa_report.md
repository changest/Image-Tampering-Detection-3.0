# QA 报告 — v3.0 答辩 PPT

## 交付物

- `output/final_presentation_cn.pptx`：13 页，16:9，含演讲者备注（13/13）
- `paper/FBFN_v3.0_答辩.pptx`：同一文件副本，便于直接查找
- 素材：`output/assets/figures/`（4 张）
- 渲染预览：`output/rendered/`（用于人工检查，非交付必需）

## 内容结构（methods / problem-to-solution）

1 封面 · 2 背景（只看真伪不够）· 3 难点（跨源/任务/部署）· 4 架构总览 ·
5 多任务设计 · 6 评测设置 · 7 主结果 · 8 训练轨迹 · 9 鲁棒性 ·
10 效率级联 · 11 Demo · 12 局限 · 13 总结

所有数字来自本项目实测日志（`results/`），未虚构。

## 质量检查

- **程序化审计**：`scripts/audit_pptx_quality.py` → `output/pptx_audit.md`
  - Slides 13；Embedded media 4；Notes 13
  - **high=0，medium=0，low=14**（均为 2–7 pt 的 near-miss 对齐提示，不影响阅读）
- **文本溢出**：通过 PowerPoint 渲染 1600×900 逐页目检，无裁切、无越界
- **图表可读性**：训练轨迹与鲁棒性对比图在投影尺寸下可读；表格字号 14–15 pt
- **图文一致性**：架构图为 v2 示意，已在图下标注 v3.0 的 ViT 替换与新增头
- **去模板化**：13 页中未使用重复的"三卡片"版式；背景、难点、架构、结果、图表、
  Demo、局限、总结分别采用不同构图

## 自审发现与修正

| 严重度 | 页面 | 问题 | 处理 |
|---|---|---|---|
| medium | 2 | 下半页留白偏多 | 上移分隔线与结论句，放大主陈述 |
| medium | 10 | 负结果说明离表格过远 | 上移至表格下方 |
| low | 3 | 编号与标题左缘 2.2 pt 偏差 | 保留（有意层级缩进） |
| low | 5 | 箭头与输出框顶缘 3.6–7.2 pt 偏差 | 保留（视觉上居中） |

## 已知限制

- 架构图沿用论文 v2 插图，v3 改动以文字标注；如需 100% 对应可重绘。
- Demo 截图为实际运行界面，包含 Gradio 页脚空白，未裁切。
- 表格中 ep5–ep30 为不同检查点，不是同一模型的不同运行。

## 复现

```
python presentation/make_charts.py
python presentation/build_deck_cn.py
python ~/.claude/skills/nature-skills/skills/nature-paper2ppt/scripts/audit_pptx_quality.py \
  presentation/output/final_presentation_cn.pptx --report presentation/output/pptx_audit.md
```
