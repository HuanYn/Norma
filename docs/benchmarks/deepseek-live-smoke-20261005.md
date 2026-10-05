# DeepSeek 真实连接与单图 smoke

时间：2026-10-05 04:13:14 UTC（北京时间 12:13:14）。用户明确授权使用所提供的凭据；凭据不写入此记录或源代码。

## 范围与输入

- 地址：`https://api.deepseek.com`；模型：`deepseek-flash`。
- 先 GET `/models`，成功返回 Flash 与 Pro；随后仅发起 **1 次**视觉生成请求，无自动重试。
- 图片：Pillow 生成的 320×180 白底几何图，左红色正方形、右蓝色圆形。没有读取或上传私人相册。
- 提问只要求描述颜色/形状/左右关系，不在问题、文件名或 caption 中透露答案。
- 参数：`thinking.type=disabled`，`response_format.type=json_object`，输出上限 384 tokens。网页常规预设仍为 1024。
- 图片 SHA-256：`c72b61ba72bfc70539913c3a8a9868fcdd9758fac81e5c26f5e02df097bd8379`。

## 观察结果

模型返回：**“图片中左侧是一个红色正方形，右侧是一个蓝色圆形。”** claim `c1` 引用 `synthetic-01`。经现有 `generate_grounded` 严格格式、图片 ID 与 provenance 校验通过，主代理目视图像后确认这条描述与合成图一致。

| 观测项 | 实际结果 |
| --- | --- |
| 模型列表请求耗时 | 2253 ms |
| 视觉调用及引用校验耗时 | 582 ms |
| smoke 脚本总耗时 | 2853 ms |
| 视觉请求数 | 1 |
| 私人图片上传数 | 0 |
| 之后启动本机网页 | `/health` 为 ok、VLM configured=true；首页 HTTP 200 |
| 默认偏好训练 | 关闭，record-only |

以上为单次网络与处理观测，无重复实验、无误差区间，不外推为真实照片准确率或稳定性能。脚本使用固定合成证据，不包含 OpenCLIP 检索；没有验证完整本地检索 → 多图云端分析链路。当前 runtime 不保留 usage 账单字段，本轮没有查账，不报告费用或 token 使用量。

## 证据与复现

脚本：`scripts/check_deepseek.py`，从进程环境读取 Key。执行会真实调用服务，可能计费。`--serve-on-success` 仅在 smoke 成功后启动本机网站，失败不会偷偷切换模型。

本机原始产物：`.norma/deepseek-smoke/result.json` 与 `synthetic-shapes.png`（Git 忽略）。Obsidian 稳定副本位于 `Projects/Norma/assets/deepseek-live-20261005/`，Dev 中独立嵌入图片。仅包含合成图、结果与哈希，不包含凭据。

尚缺：真实相册端到端联调、固定评测集、多图与复杂问题、重复延迟/成本测量。首次连通没有同口径修复前真实失败样本，不能编造“优化前后准确率提升”。

脚本回归：新增 9 项隔离测试覆盖缺失/无效凭据、模型未列出、HTTP 错误脱敏、provider 错误与合成图成功；加上启动器共 19 项通过，全部是假凭据、假网络，不增加真实服务调用。Ruff 与 diff 检查通过。
