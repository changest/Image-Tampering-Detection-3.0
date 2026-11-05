# 投稿图文件

## Fig. 1（Nature 风格示意图）
- `fig1_architecture_v3.pdf` — 矢量 PDF，文字可编辑（180 mm 宽）
- `fig1_architecture_v3.svg` — 矢量 SVG，文字保持 `<text>`
- `fig1_architecture_v3.tiff` — 600 dpi TIFF（LZW）
- `fig1_architecture_v3.png` — 预览

生成脚本：`make_fig_arch_nature.py`
- 白底、Arial 7 pt、蓝/青/红/金四支分支色
- 证据缩略图取自 `paper/figures/evidence_map.png`
- 已通过 `validate_figure.py`（18 pass / 0 fail）与 `audit_pdf_text.py`（最小字号 5.6 pt）

## Fig. 2（证据图）
- `fig2_evidence_maps_300dpi.tiff`
- `fig2_evidence_maps.png`

说明：旧版 `architecture_300dpi.tiff` 为 v2 示意图（Swin-T、256），仅备查；投稿用 Fig. 1 上方文件。
