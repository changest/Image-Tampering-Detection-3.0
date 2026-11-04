# 网页版演示（guizang-ppt-skill · 瑞士国际主义 / IKB）

## 打开方式

直接用浏览器打开 `index.html`（单文件，无需服务器；图片在同级 `images/`）。

- `← / →`、滚轮、触屏翻页；`ESC` 总览；`P` 演讲者模式；`B` 低功耗静态模式
- PDF 版：`FBFN_v3.0_swiss.pdf`（10 页，1280×720 渲染后嵌入）

## 内容（10 页）

1. 封面（IKB + ASCII 呼吸场）
2. 问题：只看真伪不够
3. 三个难点（S05 三层色块）
4. 架构（S22 图 + 三项 KPI）
5. 主结果（S06 KPI 塔）
6. 训练轨迹（S02 纵向时间轴）
7. 鲁棒性（S07 横向条形图）
8. Demo（S22 界面截图）
9. 边界（S04 六格）
10. 收尾（S10 左宣言 + 右三条 takeaway）

## 校验

```bash
node ~/.claude/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs index.html
# → Swiss deck validation passed: 10 slide(s)
```

- 每页有 `data-layout`；图片绑定 `data-image-slot="s22-hero-21x9"`
- 演讲者备注已写入 `SPEAKER_NOTES`（10 页）
- 渲染检查：`rendered/slide-01..10.png`（1600×900）
- 已知处理：模板内 IKB token 存在重复定义（第二段误写为柠檬黄），已在页面末尾追加 IKB 覆盖样式

## 重建

```bash
python build_web.py
```
