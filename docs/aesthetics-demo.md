# 三天 Demo：学习型质量与美学评分

状态（2026-10-05）：后端模块、真实CPU单图推理和8图网页分析已验证，原始计时见 [实际报告](benchmarks/demo-sprint-20261005.md)。这不是独立质量准确率评测，不能用测试替身分数或单元测试数量冒充实验效果。

## 首版范围

首版暂时把原计划的 SigLIP Aesthetic Predictor V2.5 换成 **MUSIQ-AVA**，与 **MUSIQ-KonIQ10k** 复用同一推理框架，减少三天 Demo 的依赖、部署和调试成本。不是把一个质量分数重新命名为美学分数：两者使用不同的预训练权重、训练目标和输出头。

| 输出 | 权重/训练目标 | 展示含义 |
|---|---|---|
| `technical_quality` | MUSIQ-KonIQ10k，单值回归头 | 通用感知质量，原始量纲约 0–100，越高越好；不是精确的模糊/曝光问题诊断 |
| `aesthetic_quality` | MUSIQ-AVA，10 档分布头的期望评分 | 通用美学，1–10；不代表当前用户的个人喜好 |

两项原始值分开返回，不裁剪技术分、不合成为未经校准的百分制总分，也不覆盖原来的规则评分。模型分数不自动触发删除、拒绝或训练。个性化和选片排序的接入需要独立验证。

## 权重与环境

- 固定 `pyiqa==0.1.16`。该版本的架构加载是惰性的，MUSIQ 路线不需要 bitsandbytes、Qwen 或另一个视觉骨干。
- 不建议在现有环境直接安装整个 pyiqa 全依赖集合：它会包含其它评测模型需要的包。可先在已具备兼容 PyTorch/torchvision 的专用环境安装 `pyiqa==0.1.16 --no-deps`，再验证缺失依赖；这不是所有 pyiqa 模型的完整环境。
- MUSIQ 路线的直接导入依赖包括 PyTorch、torchvision、NumPy、OpenCV、Pillow、PyYAML、huggingface-hub、requests、tqdm、SciPy。真实运行的环境清单需要一起留档。
- 导入模块、打开状态接口都不加载模型、不联网下载。只有显式发起分析任务才会加载已经存在的本地权重。
- 模型状态中的 `weights_present_unverified` 只表示文件和包版本存在；并不表示哈希已通过、CUDA 可用或推理已成功。

权重来源固定为 `chaofengc/IQA-PyTorch-Weights`，修订版 `0df2df423c65f6a64209309695f3845727431027`。放在配置指定目录，例如 `.norma/data/models/musiq/`：

| 文件 | SHA-256 |
|---|---|
| `musiq_koniq_ckpt-e95806b9.pth` | `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa` |
| `musiq_ava_ckpt-e8d3f067.pth` | `e8d3f0671965dfe4301a1f7a4aedc8f49be8fed33b82dd5d116a9eb3ce2bd16b` |

这些是上游 LFS 清单的完整哈希。运行时先校验，再让上游加载 pickle 权重；不允许同名但不同内容的 checkpoint 冒用模型身份。当前上游加载器使用 `weights_only=False`，因此固定可信来源与完整哈希尤其重要。

官方资料：[pyiqa v0.1.16](https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.16)、[MUSIQ 实现](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.16/pyiqa/archs/musiq_arch.py)、[权重清单](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/tree/0df2df423c65f6a64209309695f3845727431027)。

许可边界：需分别核对 [pyiqa 许可](https://github.com/chaofengc/IQA-PyTorch#-license)、[权重模型卡](https://huggingface.co/chaofengc/IQA-PyTorch-Weights) 与原始数据/模型条款；当前路线按非商业研究演示审查，不把“能下载”当成商业授权。

## 接入契约

应用启动且核心数据库初始化之后：

```python
from ai.aesthetics import AestheticsService, PyiqaMusiqProvider
from ai.aesthetics.api import create_aesthetics_router
from ai.aesthetics.jobs import AestheticsJobManager

provider = PyiqaMusiqProvider(settings.model_cache_dir / "musiq", device="cpu")
service = AestheticsService(database, provider)
aesthetics_jobs = AestheticsJobManager(service)
aesthetics_jobs.start()
# 此 include_router 在应用构建时调用一次，回调返回当前服务实例。
app.include_router(create_aesthetics_router(lambda: service, lambda: aesthetics_jobs))
# 在应用关闭阶段调用 aesthetics_jobs.shutdown()
```

使用有兼容 CUDA 环境的显卡推理时明确指定 `device="cuda:0"`。不自动把 CPU 配置变成 GPU，也不自动切换回规则算法。

| 接口 | 作用 |
|---|---|
| `GET /aesthetics/status` | 包版本、权重文件存在性、是否加载；不是实时性能探针 |
| `POST /albums/{album_id}/aesthetics/jobs`，请求体 `{"force": false}` | 只分析已索引相册；立即返回任务，后台逐图计算 |
| `GET /aesthetics/jobs/{job_id}` | 任务状态与真实已完成图片比例；`progress` 在 0–1 之间 |
| `POST /aesthetics/jobs/{job_id}/cancel` | 请求取消；在两张图片之间生效，当前 GPU 调用不强行中断 |
| `GET /albums/{album_id}/aesthetics` | 读取当前模型、当前图片内容对应的缓存；不加载模型 |

单个工作进程使用一条评分队列，同一相册不能重复排队，最多八个在途任务。进程重启把遗留任务标记为中断，需要明确重试；之前完成的照片缓存仍可复用。任务复用核心 `jobs` 表，类型为 `analyze_aesthetics`。评分使用独立的 `aesthetics_scores_v1` 表，初始化幂等，不抢占核心数据库版本号。

## 实现与边界

1. 先检查照片属于指定已索引相册，再核对大小、mtime 和完整内容 SHA-256。即使大小和时间戳被恢复，图片内容变化也不能复用旧分数。
2. 缓存身份包括 checkpoint 完整哈希、pyiqa 版本、模型输入预处理版本、最大边长、设备和精度。重新配置模型或预处理会得到不同缓存。
3. 显式模型推理使用 EXIF 方向校正、RGB、最长边最多 1024 像素、不放大小图；原文件不修改。该下采样是 Demo 的资源上限，不宣称等价于原分辨率基准测评。
4. 本地加载 AVA checkpoint 时必须显式指定 `num_class=10`。否则上游根据本地路径跳过自动推断，默认单值头会与权重形状不匹配。
5. 推理后再次核对源内容与当前数据库行，漂移则拒绝写入。缓存损坏会重算；部分完成可续跑；非有限分数不进入数据库。
6. 取消是图片间的协作式取消，不保证在模型加载或单次 GPU 运算中立即停止。
7. 读取全部缓存会完整校验源文件，巨大相册存在磁盘 I/O 成本。Demo 限制最多 5000 张，先用小相册验收，后续可增加分页与后台完整性审计；不能为了速度直接信任旧哈希。

## 验证与后续优化

离线测试覆盖依赖可选、固定权重完整性、AVA 头配置、缓存重用/损坏、源文件漂移、取消续跑、队列去重、接口输入校验与重启恢复。测试替身只出现在测试文件，不作为线上回退。

真实模型验收需另记录：同一组输入图、环境版本、冷启动时间、单图推理时间、总耗时、显存峰值及两类原始分数。不得把这里的单元测试数当成美学提升率。

三天 Demo 之后再做：① 固定公开人工评分集，比较规则/MUSIQ/V2.5 的 SRCC、PLCC；② 按场景分层分析失败样本；③ 校准选片中的质量软约束，保持人物数量等硬约束；④ 在独立用户偏好数据上评估个性化。用同一个模型既排序又打分的提升不能单独证明实际审美提升。
