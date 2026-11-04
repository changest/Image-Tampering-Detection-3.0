# 素材清单

| 文件 | 内容 | 来源 | 用于 |
|---|---|---|---|
| architecture.png | 四分支网络结构示意 | 论文 Fig.1（v2） | 第 4 页 |
| chart_train.png | 类型 F1 与 mIoU 随训练轮次变化 | 本项目 `results/eval_snapshots_c.log` 等实测数据，matplotlib 绘制 | 第 8 页 |
| chart_robust.png | 六种扰动下 AUC 增强前后对比 | 本项目 `results/robustness.md` 实测数据，matplotlib 绘制 | 第 9 页 |
| demo_screenshot.png | Demo 实际运行界面（拼接样例，96.1%） | 本地 Gradio 演示截图（last_ema） | 第 11 页 |

说明：
- 图表均由本项目评测数据重绘，未改动原始数值。
- architecture.png 保留原有面板标签；v3.0 的改动（DINOv2-L 448、新增类型/二分类头）在图下以文字说明。
