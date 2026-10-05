# 本地检索＋云端看图分析

Norma 先在本机用预训练 OpenCLIP 检索照片，再调用配置的云端视觉模型分析少量候选图片。使用 `POST /chat/completions` 的 OpenAI-compatible 非流式协议；具体服务必须支持图片输入与文本 JSON 回答。兼容协议不保证所有供应商/模型的参数都相同。

## 配置

### DeepSeek：在本机隐藏输入 Key 启动

2026-10-05 核对官方文档：`deepseek-flash` 支持图片输入，Base URL 为 `https://api.deepseek.com`。不是所有 DeepSeek 型号都能看图，请不要把纯文本型号用于这里。[官方视觉说明](https://api-docs.deepseek.com/guides/vision/)

如果 Key 曾贴到聊天、截图或公开页面，建议在服务商控制台撤销并重新生成。程序不会替你撤销或强制轮换，也可以使用你明确确认继续使用的有效 Key；不要再把 Key 发回聊天。停止旧的 Norma 服务后，在本机 PowerShell 中执行：

```powershell
cd E:\Norma
python scripts/start_deepseek.py
```

按提示粘贴 **API Key**，输入不会回显；按回车启动网站。脚本只将 Key 放入本次后端子进程的环境，不写文件、不修改持久系统环境，也不把 Key 放进启动命令参数。无法隐藏输入时会拒绝启动，不降级成明文输入。它不能保证进程内存被安全擦除，也不能保护已被他人控制的电脑。

启动预设：`record-only`、本机回环地址、`deepseek-flash`、JSON 输出、关闭思考、最大输出 1024 tokens。启动本身不测试 Key，不发起云端推理；只有网页点击“云端看图分析”才发送候选图片并可能计费。Key 正确性、余额和实际模型效果仍需真实调用验证。

显式关闭思考是为了先验证短结构化回答，不是证明效果更好。DeepSeek 当前默认开启思考，仅填 Base URL 与模型名不会自动关闭它；JSON 模式仍需现有 claims/citations 校验，不能替代事实核对。[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)

如手动配置，可设置 `NORMA_VLM_THINKING_MODE=disabled` 与 `NORMA_VLM_JSON_RESPONSE_FORMAT=1`。默认 `provider-default` 不传 `thinking` 字段，JSON 开关默认关闭，避免影响其他兼容服务。这些请求选项进入 provider 指纹，切换选项会改变运行身份；不代表云端权重已固定。

验证分为两个阶段：先完成模拟专项 143 项通过（21.81 秒，1 个已有弃用 warning）；随后按用户明确授权，真实 `/models` 鉴权和一次合成图视觉请求成功，严格引用校验通过。真实测试没有上传私人照片，也未执行完整 OpenCLIP 检索链路，不能作为相册效果 benchmark。详见 [真实 smoke 记录](benchmarks/deepseek-live-smoke-20261005.md)。密钥仅在运行进程中使用，不写入 Obsidian 或 Git。

### 其他兼容服务或手动配置

在启动 Python 的同一个 PowerShell 终端设置环境变量：

```powershell
$env:NORMA_PREFERENCE_MODE = "record-only"
$env:NORMA_VLM_PROVIDER = "openai-compatible"
$env:NORMA_VLM_BASE_URL = "https://YOUR_PROVIDER/compatible-mode/v1"
$env:NORMA_VLM_MODEL = "YOUR_VISION_MODEL"
$env:NORMA_VLM_API_KEY = "YOUR_SECRET_KEY"
$env:NORMA_VLM_TIMEOUT_SECONDS = "60"
python -m ai web
```

把示例地址、模型名、Key 替换为供应商控制台中同一区域/工作空间的配置。Key 留在本机；不要提交到 Git 或复制到公开笔记。项目不会自动读取 `.env`；`.env.example` 是配置说明。修改环境变量后重启后端。

- 百炼/通义千问：从控制台复制当前 OpenAI-compatible Base URL 与支持图片的模型 ID。[官方接入文档](https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai)
- DeepSeek：使用上面的本机隐藏输入启动方式，或按控制台和官方视觉文档配置。
- OpenAI：Base URL 为 `https://api.openai.com/v1`，选择支持 Chat Completions 图像输入与这里所用参数的模型。[官方图像输入文档](https://developers.openai.com/api/docs/guides/images-vision)
- 其他兼容服务：使用对应 Base URL 和视觉模型 ID；仅文本模型不能看图。

配置缺失只影响云端回答，不影响文件夹导入、本地人脸/质量分析与本地检索。健康状态中的“已配置”仅表示配置项存在，不能证明 Key 有效或远端模型可用。

## 网页使用

1. 打开相册，点击“语义索引”。
2. 在 AI Selection 输入问题，例如“哪些建筑照片适合作为旅行相册封面？请结合图片解释”。
3. 点击“云端看图分析”。按钮固定请求检索前 3 张候选；引用图片会显示在每条说明下。

后端接口允许 `top_k` 为 1–6；网页的普通 Search 仍只使用本地检索，不自动调用云端。关闭或切换网页不保证取消已经发出的远端请求；接口不自动重试，以免失败后产生额外调用。

超时参数默认 60 秒：连接使用底层网络超时，响应体每轮读取共享剩余时间，避免每收到一块正文就重新计算完整超时。DNS、TLS/响应头，以及 chunked 传输的长度行/尾部解析仍受平台与网络库行为影响，不承诺任何异常远端行为下整个 API 都严格在 60 秒内返回；本地检索时间也不在这个远端读取预算内。

```powershell
$albumId = "YOUR_ALBUM_ID"
$body = @{
  query = "比较这几张照片的构图，并指出适合作封面的照片"
  top_k = 3
  user_id = "local"
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8765/albums/$albumId/rag" `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## 数据与校验

本地保留原图、目录索引、embedding、人脸描述符、偏好反馈和运行审计。云端接收本次问题、候选 ID/相关证据描述与 Top-K 的压缩 JPEG；图像长边最多 1280px，并去除 EXIF。文件名等显示元数据可能包含在提示中，完整本地路径经过脱敏。不要把“只发候选图”理解成照片内容完全不离开本机。

请求仍然使用原有证据 bytes SHA、固定候选快照、claim/citation 校验和服务端构造的 provenance。模型不能返回任意相册外图片 ID。但引用正确不等于回答语义一定正确，仍需单独标注与评测。

云端 provider 指纹记录配置与调用协议；远程权重由供应商控制，不能沿用本地 pinned 权重的逐字节复现承诺。固定模型版本 ID 比滚动 `latest` 更适合对照实验。

## 本地模式

需要使用原有本地 Qwen 时：

```powershell
$env:NORMA_VLM_PROVIDER = "local"
python -m ai web
```

安装与本地推理限制见 [grounded-multimodal-rag.md](grounded-multimodal-rag.md)。云端模式不会加载本地 Qwen 权重。

## 验证状态

通用适配最初使用模拟 HTTP 验证。后续 DeepSeek 已完成一次真实合成图 smoke，调用与校验耗时 582 ms，包含模型列表获取的脚本总耗时 2853 ms；仅一次小图观测，不是稳定延迟或完整相册性能。账单未读取，不能报告实际费用。真实相册、多图及独立效果评测仍未完成。历史本地 Qwen smoke 与半合成偏好实验不能被改称为云端实验。
