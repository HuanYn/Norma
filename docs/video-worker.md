# 3090 图生视频 Worker：首个可运行闭环

此模块是三天 Demo 的单图、单镜头基础设施，不是完整剪辑器。模型是 **Wan2.2-TI2V-5B**，不训练。代码已具备真实推理调用；单元测试使用明确标记的替身，不代表已经生成真实视频或测得 3090 性能。服务器烟测的结果应另外记录。

## 架构与边界

本机 Norma 只上传用户确认的单张图片和提示词，通过 SSH 本地转发访问服务器的 `127.0.0.1:8766`。Worker 一次执行一个任务，返回任务 ID；本机轮询进度、取消或下载 MP4。Worker 不接收图片路径、任意 URL、模型路径或 shell 命令。模型路径由启动服务器的操作者设置。

上传前客户端移除 EXIF 并将长边压到 2048；服务端再次解码为干净 RGB 像素。原始图片不改动。生成会按输出比例居中裁剪，应在网页上传确认界面展示裁剪预览。图像和提示词保留在服务器的私有任务目录中用于复现，不自动删除，需由操作者明确归档。

## 独立环境

不把视频依赖混装到本地选片环境，不升级其他用户的服务器环境。服务器使用 Python 3.11+，先安装与服务器驱动兼容的 CUDA 版 PyTorch（`torch>=2.4,<3`），再安装：

```bash
python -m pip install -r ai/video/requirements.txt
```

固定 `diffusers==0.35.1`，其 `WanImageToVideoPipeline` 源码已包含 5B 的 `expand_timesteps` 图像条件分支。其他依赖首次成功后保存环境锁定清单，再评估升级。没有要求额外编译 FlashAttention。

官方 checkpoint 为 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`。将整个 Diffusers snapshot 下载到操作者批准的模型目录，记录解析后的 commit hash。不要误用原生 Wan 仓库格式的权重目录。

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir /approved/workdir/models/wan2.2-ti2v-5b-diffusers
```

模型下载是显式部署步骤，不会因访问健康检查或导入模块而发生。Worker 默认 `local_files_only=True`；只有启动参数 `--allow-model-downloads` 才允许 Hugging Face 下载。

## 启动

将以下路径换为已批准的隔离工作目录；`CUDA_VISIBLE_DEVICES` 必须指向已确认空闲的 GPU，不得抢占其他任务：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/video_worker.py \
  --host 127.0.0.1 --port 8766 \
  --data-dir /approved/workdir/video-jobs \
  --model-path /approved/workdir/models/wan2.2-ti2v-5b-diffusers
```

仅允许此数据目录运行 **一个进程、一个 Uvicorn worker**，不要开启 reload 或多个副本。默认整模型 CPU 卸载＋FP32 VAE 非分块解码，Transformer 使用 BF16。如果实际 OOM，先减少帧数/面积；仍不够再用 `--offload sequential`，后者可能明显更慢。不要把官方 4090 的运行时间当作 3090 的实测性能。

### 已验证的依赖兼容性修正

Diffusers 0.35.1 的 Wan VAE 分块编码在 `patchify` 之前提前进入 tiled 分支；TI2V-5B 的 `patch_size=2` 需要把 RGB 的 3 通道变成 12 通道，分块分支遗漏了此步骤。因此超过默认 256 像素阈值就会遇到通道不匹配。分块解码也不能直接视为适配此架构。

已在服务器实际安装的 torch 2.8.0+cu128 / diffusers 0.35.1 / transformers 4.57.6 环境，以真正的微型随机权重 `AutoencoderKLWan`（672,608 参数）做纯 CPU 复现：开启分块，实际卷积报“需要 12 通道但收到 3 通道”；关闭分块，同一输入成功输出 `(1,4,1,2,2)`。这不是完整模型生成性能实验，也不能替代完整视频验收。

当前最小修复是明确关闭 VAE tiling，保持模型、dtype 与其余生成参数不变，不修改第三方源码。非分块增加 VAE 峰值内存风险，所以先验证 640×384、49 帧；不得把此修复推断成 720P 必然适配 24GB。之后只有在独立升级测试确认 5B 的编码/解码、尺寸与时序都正确后，才重新启用分块。

本机建立 SSH tunnel 后，应用配置使用 `http://127.0.0.1:8766`。可通过环境变量 `NORMA_VIDEO_WORKER_TOKEN` 设置至少 24 位不含空格的 ASCII token；不写进 Git、命令行参数或前端。非 loopback 绑定必须有 token，但推荐始终使用 SSH tunnel，而不是开放公网 HTTP。

共享服务器优先使用 `--create-token-file /approved/workdir/worker-token`：首次启动独占创建随机 token，POSIX 权限 0600，不输出内容；后续改用 `--token-file /approved/workdir/worker-token`。已有文件不会覆盖，组/其他人可读或不属于当前用户的文件会被拒绝。不要同时设置环境变量和 token 文件。通过已有加密 SSH/SFTP 会话将 token 提供给本地后端，不能发送到网页或提交版本库。

## API

所有端点在配置 token 时要求 `Authorization: Bearer <token>`。

| 接口 | 用途 |
|---|---|
| `GET /health` | 进程存活与是否加载模型；不代表 CUDA/权重/推理验证成功 |
| `POST /v1/video/jobs` | 提交经确认的单张 JPEG/PNG，返回 202 与随机任务 ID |
| `GET /v1/video/jobs/{id}` | 查询状态、配置、输入哈希、阶段、耗时与结果 |
| `POST /v1/video/jobs/{id}/cancel` | 排队任务立即取消；运行任务在下一安全检查点取消 |
| `GET /v1/video/jobs/{id}/artifact` | 仅在 completed 时下载真实生成的 MP4 |

提交 JSON：

```json
{
  "image_base64": "<JPEG/PNG base64, no data URL prefix>",
  "upload_confirmed": true,
  "prompt": "A slow camera push in. Preserve the subject and lighting.",
  "width": 640,
  "height": 384,
  "num_frames": 49,
  "num_inference_steps": 20,
  "guidance_scale": 5.0,
  "seed": 42
}
```

49 帧以原生 24 fps 编码约 **2.04 秒**，是降低成本的通路预览，不是已验收的高质量模式。640×384 也是试验性低分辨率。确认成功后再对照 81 帧（3.375 秒）或 121 帧（5.04 秒）、更高分辨率和 30/50 步。只改变帧率来声称“5 秒生成”是不允许的。

宽高必须是 32 的倍数，总面积不超过 1280×704，帧数满足 `4k+1`，最多 121 帧和 50 步；上传最多 8 MiB、20 MP，HTTP 请求上限 12 MiB。最多 4 个未完成任务和 200 个保留任务；达到容量返回错误，不偷偷清理原有结果。

进度字段 `step_percent` **只表示去噪步骤比例**，不是整个任务完成比例。排队、模型加载、编码、解码与导出返回 `null`；界面应显示阶段与不定进度，不伪造缓慢爬升的百分比。任务状态为 queued/running/completed/failed/cancelled/interrupted。首次模型加载可能慢，加载阶段不能立即打断；GPU 步骤间会检查取消。重启后未完成任务标记 interrupted，必须明确重新提交，不重复消耗算力。

## 本机集成

```python
from ai.video.client import VideoWorkerClient
from ai.video.models import VideoOptions

client = VideoWorkerClient("http://127.0.0.1:8766", token=operator_token)
job = client.submit(
    selected_image_bytes,
    VideoOptions(prompt="Slow cinematic camera push in"),
    upload_confirmed=True,
)
status = client.status(job["id"])
```

应用必须从已登记且属于当前相册的照片 ID 读取图片，不能把网页任意文件路径直接读入。Worker 地址来自服务器配置，不接受浏览器自定义，避免 SSRF。网页确认不是由模型自动勾选；没有确认时客户端和服务器均拒绝上传。client 拒绝非 loopback 明文 HTTP、代理和重定向；下载也不覆盖已存在文件。

## 最小验收与优化顺序

1. 无模型启动成功、健康检查不下载；错误 GPU/缺权重明确失败，不生成占位视频。
2. 一张合法样图完成 49 帧预览，记录输入哈希、完整提示词、种子、维度、步数、模型 revision、库版本、GPU 峰值与耗时，检查 MP4 可解码和实际帧数。
3. 相同输入/种子对比 20/30/50 步与 49/81/121 帧；每次只改变一个参数。保留失败尝试，不能只挑最好的一次。
4. 再加入自然语言生成运动提示、风格偏好、3–5 个镜头拼接；视频质量评估同时看主体保持、运动与闪烁，不能靠静止视频获得“稳定”高分。
5. 首个 Demo 不做 Alpha、DPO、视频 LoRA 或批量并行；这些是后续、有独立预算和评测门槛的增强项。

## 依据与许可证

- [Wan2.2 官方项目](https://github.com/Wan-Video/Wan2.2)：5B 同时支持 T2V/I2V，官方卸载配置面向至少 24GB 显存；其时间与显存数字不自动适用于此 Diffusers/3090 组合。
- [官方 Diffusers checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)：代码/权重许可证需在下载时归档，当前标为 Apache-2.0。
- [Diffusers v0.35.1 I2V 源码](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/wan/pipeline_wan_i2v.py)：本实现明确使用 I2V 类与 5B 条件分支，不把模型卡中的纯文本 WanPipeline 示例误当作图生视频。

局限：进程内单队列，取消不是硬实时，未配置跨进程锁、用户租户隔离或自动磁盘回收，因此仅部署为私有单用户服务。HTTP 单元测试与 mock 推理测试不证明生成质量或服务器速度；必须以实际烟测为准。
