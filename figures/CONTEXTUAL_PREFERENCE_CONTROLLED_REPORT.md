# Norma 67D 上下文偏好学习：公开图像上的受控半合成实验

## 结论边界

这是一项 **controlled semi-synthetic preference-learning test**，不是用户研究。图像来自 Wikimedia Commons；偏好由三套提前声明的类别效用函数产生，没有真人做 A/B 选择。它能验证“67D 小型适配器是否会从反馈中学习并迁移到未参与反馈的图片”，不能证明真人满意度、总体泛化，也不能称为 DPO、SFT、LoRA 或 OpenCLIP 微调。

## 模型与模拟器

冻结 Multilingual OpenCLIP 得到单位图像向量 \(z_i\in\mathbb{R}^{512}\) 和中文查询向量 \(q\)。固定 signed-Hadamard 投影 \(P\in\mathbb{R}^{32\times512}\) 构造：

\[
\phi(i,q)=[Pz_i,\;Pz_i\odot Pq,\;z_i^\top q,\;r_i,\;m_i]\in\mathbb{R}^{67}.
\]

偏好适配器的分数是：

\[
s_\theta(i,q)=z_i^\top q+\theta^\top\phi(i,q),\qquad
p(i\succ j)=\sigma(s_\theta(i,q)-s_\theta(j,q)).
\]

训练直接复用生产代码中的高斯先验、全批量 damped Newton + Armijo MAP 和 Laplace 协方差。受控用户的潜在效用为：

\[
u^*(i)=z_i^\top q+\beta_{\text{profile}}(c_i),
\qquad
p^*(i\succ j)=\sigma((u^*(i)-u^*(j))/0.55).
\]

三套 \(\beta\) 分别偏好旅行建筑、城市夜景和山地旅行。每个无序图片对的 Bernoulli 随机数由 `seed/profile/pair` 的哈希固定；不同采样方法查询同一对图片时得到同一个模拟反馈，避免把反馈噪声差异误算成方法收益。

## 数据和协议

- 70 张公开 Wikimedia 图片：旅行建筑 22、城市夜景 27、山地旅行 21。
- 每个 seed 按类别分层为 42 张反馈训练图和 28 张测试图，文件名交集严格为 0。
- 这里仅保证偏好适配器阶段的 exact-file disjoint；没有按摄影师、场景或语义近重复去重，也无法保证 OpenCLIP 预训练没见过公开图。
- 10 个 split/choice seed；每个 seed 内先平均三套固定模拟 profile。
- 反馈预算为 0、10、30、60。
- 方法为零反馈 OpenCLIP cosine、随机未查询图片对 + contextual、Bayesian predictive-entropy 图片对 + contextual。
- held-out 指标在 28 张测试图的全部 378 个无序对上计算：模拟概率与模型概率的期望 log loss，以及潜在效用顺序准确率。
- set regret 在“测试集精确选 6 张、每个代理类别最多 3 张”的硬约束下计算，并除以 6。
- 区间为：三套 profile 在 seed 内平均后，对 10 个 seed mean 做 10,000 次非参数 percentile bootstrap。重叠的数据划分意味着它们是敏感性重复，不是 10 名独立用户。

## 60 次反馈的主结果

| 方法 | Pair log loss ↓ | Pair accuracy ↑ | Set regret/photo ↓ |
|---|---:|---:|---:|
| Cosine（零反馈） | 0.694 [0.693, 0.694] | 0.606 [0.596, 0.616] | 0.381 [0.371, 0.388] |
| Contextual + random pairs | 0.630 [0.623, 0.636] | 0.694 [0.677, 0.710] | 0.058 [0.040, 0.079] |
| Contextual + predictive entropy | 0.628 [0.619, 0.636] | 0.698 [0.681, 0.716] | 0.052 [0.038, 0.069] |

相对零反馈基线，随机图片对在 60 次反馈后将 log loss 降低 9.21%，排序准确率提高 8.78 个百分点，set regret/photo 降低 84.80%；对应的配对 seed-bootstrap 改善区间分别为 `[0.0575, 0.0703]`、`[0.0723, 0.1049]` 和 `[0.3017, 0.3422]`，均不跨 0。

熵采样相对零反馈基线将 log loss 降低 9.50%，排序准确率提高 9.14 个百分点，set regret/photo 降低 86.29%；改善区间分别为 `[0.0574, 0.0748]`、`[0.0705, 0.1154]` 和 `[0.3125, 0.3425]`，均不跨 0。

但是，熵采样相对随机采样的三项改善区间都跨 0：log loss `[-0.0062, 0.0113]`、accuracy `[-0.0149, 0.0216]`、regret `[-0.0203, 0.0334]`。因此本实验只支持“contextual 学习优于 zero-feedback cosine”，不支持“熵采样优于随机采样”。

## 发现的问题

熵采样并非从第一步起就更好：10 次反馈时，其 held-out accuracy 为 0.592，低于零反馈的 0.606 和随机采样的 0.620；到 30 次反馈才升到 0.680。当前最大化预测熵的策略在高方差先验下容易优先选择不确定但未必具有集合决策价值的图片对。这正是后续需要把 production PDRR 与随机/熵采样放进同一协议比较的原因，不能从本图推断 PDRR 的效果。

## 复现

模型权重必须已存在于本地缓存，脚本强制 Hugging Face 离线模式：

```powershell
python figures/benchmark_contextual_preference_simulation.py `
  --album .norma/demo-album-eval `
  --cache-dir .norma/data/models `
  --output figures/contextual_preference_controlled_20260828.json `
  --device cpu --batch-size 8 `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --bootstrap-resamples 10000

python figures/gen_fig3_contextual_preference_learning.py `
  --input figures/contextual_preference_controlled_20260828.json
```

复现脚本现在会在推理前验证 `--cache-dir/openclip` 下两个固定模型缓存目录，并拒绝 `--reuse-embeddings-from` 与 `--output` 指向同一文件（含可解析的同一文件别名）。公开 provenance 的 `path_base` 固定为仓库根，所有路径使用 POSIX 仓库相对形式；loader 会拒绝盘符、绝对路径、反斜杠和 `..` 逃逸。加入防呆后的完整离线重跑中，70 图 + 1 个中文 query 的 OpenCLIP 阶段为 `56.74553 s`，模拟与统计阶段为 `13.29519 s`；两项科研防呆回归测试均通过。

图生成器固定 `SOURCE_DATE_EPOCH=0`，避免 PDF 创建时间破坏字节级复现；对同一 JSON 连续生成两次时，PDF、PNG、结果表和 LaTeX include 的 SHA-256 均逐字节一致。

核心证据文件：

- `contextual_preference_controlled_20260828.json`（当前 LF 规范化 SHA-256：`16bfdde5c61fc6dca02d19676a441fd37b265effb9bd0631dc4947ad5bb2cdbc`）：公开来源、许可、文件哈希、512D 向量、67D 特征、潜在效用、全部 split、3,600 条反馈轨迹、checkpoint、原始 seed mean、bootstrap 和配对差异。路径脱敏与换行规范化只改 provenance；排除 provenance 后的科学载荷 SHA-256 在迁移前后均为 `4d145b917e9d4462a72f5eee6735ebe87a819e96ca3ead206b4e323ed41b2984`。
- `fig3_contextual_preference_learning.pdf/png`：可复现的三联学习曲线。
- `TABLE_contextual_preference_controlled.tex`：60 次反馈结果表。
- `contextual_preference_latex_include.tex`：自包含图注。

## 一段话总结

在 70 张公开 Wikimedia 图片、10 个分层文件级隔离划分和三套明确类别效用模拟器上，冻结 Multilingual OpenCLIP 后训练的 67D Bayesian contextual residual adapter 能随 60 次 A/B 反馈降低未参与反馈图片上的 pair log loss，并在这些敏感性重复中稳定减小硬约束选片的受控 regret；这提供了“实现确实会学习”的可复现实验证据，但反馈不是人类产生、类别标签是搜索词代理、划分不是场景级隔离，且 predictive entropy 对随机采样没有可靠优势，因此结果只能作为工程与算法集成验证，不能外推为真实用户效果。
