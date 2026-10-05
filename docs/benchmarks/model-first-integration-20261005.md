# 本地检索＋云端分析：工程验证记录

日期：2026-10-05。工作基线 `58bc08c2bf9feaaecd0dc418b1d292075ea3bc20`；验证记录生成时尚未提交/推送，之后的发布状态以 Git 记录为准。此处是功能与安全回归记录，不是模型质量 benchmark。

## 实际执行

| 命令 / 检查 | 结果 |
| --- | --- |
| `python -m pytest ai/tests -q` | 421 passed，1 warning，155.43 秒 |
| 云端 provider / cloud integration / RAG HTTP 专项 | 120 passed，1 warning，22.64 秒（全套子集） |
| record-only 模式专项 | 10 passed（全套子集） |
| `pnpm build`，包含 `vue-tsc --noEmit` | 通过，15 modules transformed |
| `python -m ruff check ai` | 通过 |
| `python -m ruff format --check ai` | 82 files already formatted |
| `git diff --check` | 通过 |
| 两份前后链路图 | Mermaid CLI 11.16.0 渲染，PNG 已目视检查，Markdown 与源码一致 |

唯一 warning 是已存在的 Starlette/httpx 弃用提示。本轮没有真实外部模型网络请求；HTTP 传输由测试替身提供，输入图片由测试构造。测试数增加不是模型效果提升。

## 行为对比

- 以前有效 preferred 反馈可以拟合 67D/7D 两条小模型；现在默认 record-only，测试断言 train、legacy update、activate 调用均为 0。
- 以前兼容历史偏好权重可参与排序；现在默认四条消费链 Search/Selection/Replacement/RAG 均忽略它们，原数据保留。
- 以前生成需要本地 Qwen；现在默认可配置云端接口，本地 Qwen 保留为显式选项。云端实际可用性尚未用真实供应商验证。
- 云端网页调用发送本地 Top-3；API 上限 6 图，最长边 1280px，剥除 EXIF，禁止重定向和自动重试。
- 新增安全回归覆盖 JSON 转义密钥回显、证据 ID 越界、字节/像素预算、协议错误与正文滴流读取预算。DNS/TLS/协议行解析仍不是硬性端到端 deadline，详见配置文档。

## 不能由这些测试推出的结论

没有新增检索准确率、视觉事实正确率、真实偏好胜率、云端费用或端到端延迟数据。也没有生成新标注集、运行 LoRA/SFT/DPO。下一阶段须配置真实视觉服务，在固定且来源透明的标注集上比较原模型、prompt/检索改进与可选后训练。

配置见 [cloud-analysis.md](../cloud-analysis.md)，算法与标注计划见 [model-first.md](../model-first.md)。本机 Obsidian 的 `Projects/Norma/Dev`、`FIX`、`Model-First` 已同步前后图、根因、修复与限制。
