# JVCIR 投稿清单

## 文件
- [ ] `manuscript_jvcir.md` → 转成 Elsevier Word/PDF（需补作者信息、单位、邮箱）
- [ ] `highlights.txt`（5 条，每条 ≤85 字符）
- [x] `cover_letter.md`（补通讯作者信息）
- [ ] 图 1–2（architecture / evidence）单独高清文件（≥300 dpi，TIFF/EPS）
- [ ] 表 1–10 可编辑
- [ ] 参考文献 [1]–[16] 核对格式

## 内容
- [x] 标题页：作者、单位、通讯作者、ORCID（如有）
- [ ] 摘要 ≤250 词（当前约 300 词，需压缩）
- [ ] 关键词 5–6 个
- [x] Declaration of competing interest
- [x] CRediT author statement
- [ ] Data availability statement（代码/权重链接 + 数据可索取）
- [ ] 伦理声明（公开数据集，无需伦理审批）

## 风险点（审稿人大概率会问）
- [ ] COVERAGE 跨库 AUC 0.55：已在 4.8 与局限中主动说明，准备好"下一步用相似物体复制移动数据重训"的答复
- [ ] 与 TruFor 的定位差距：已注明 mIoU 0.603 vs 0.665，强调多任务与效率
- [ ] 训练集包含 CASIA：已在同协议对比中注明可能的数据重叠
- [ ] 类型分类无公开 benchmark：说明为本文自定义任务，给出混叠矩阵与每类指标

## 投稿系统
- [ ] Editorial Manager 账号
- [ ] 推荐审稿人 3–5 位（图像取证方向，避开合作者）
- [ ] 确认 APC：JVCIR 为混合期刊，非 OA 不收费
