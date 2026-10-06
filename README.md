# Norma

**从一组照片，到可探索的瞬间。**

Norma 是一个本地优先的多模态照片工作台。导入旅行相册后，完成美学评估、相近构图筛选、自然语言选片，再将喜欢的照片转成可通过 WASD 和方向键逐步探索的视频。使用浏览器操作，Python 提供后端服务。

## 功能

- **导入与相册管理**：读取本地 JPG/JPEG 目录，生成缩略图，保存相册与处理记录。
- **美学筛选**：MUSIQ-AVA 评估美学，MUSIQ-KonIQ10k 评估技术质量；调整门槛后复用已有评分重新筛选。
- **相似只留最佳**：结合感知哈希、OpenCLIP 和空间匹配，折叠相近构图并展示保留原因，原图始终保存在原目录。
- **自然语言选片**：支持指定数量、人物/自然风景配额和必留照片，例如“挑4张雪山照片”“选6张照片，1张人物5张风景”。
- **个性化重排**：使用已有偏好案例调整排序，可选择助手代理记录、本人反馈或关闭偏好。
- **图片问答**：检索相关照片，调用配置的视觉语言模型回答问题并关联照片引用。
- **双图交互探索**：两张照片各自建立 LingBot 会话，支持 WASD 平移、方向键转向、继续生成、历史回放和视频下载。
- **任务与进度**：按需启动分析，显示阶段和处理数量，支持取消与已完成缓存复用。

## 快速开始

本地照片流程使用 CPU；视频生成使用单独部署的 NVIDIA GPU 服务。推荐 Python 3.11～3.13；修改前端时使用 Node.js 22 和 pnpm 11。

以下命令在 Windows PowerShell 中执行：

~~~powershell
git clone https://github.com/HuanYn/Norma.git
cd Norma

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python -m pip install pyiqa==0.1.16 --no-deps

python scripts/install_demo_models.py
python scripts/start_demo.py
~~~

打开 **[http://127.0.0.1:8767](http://127.0.0.1:8767)**。

仓库包含构建好的网页。模型准备命令下载固定版本的 OpenCLIP 与两套 MUSIQ 权重，并核验完整性；准备完成后可复用本地缓存。详细步骤见 [最小复现](docs/reproduce.md)。

Linux/macOS 创建环境后，将激活命令换成：

~~~bash
source .venv/bin/activate
~~~

### 网页使用顺序

1. 输入照片目录的绝对路径，例如 D:\Photos\Trip，点击 **导入照片**。
2. 点击 **美学评估 · 相似只留最佳**，等待质量评估和语义索引完成。
3. 查看保留照片；点击 **查看折叠** 可查看相似图与处理原因。修改门槛后点击 **重新筛选**。
4. 输入要求，点击 **模型选片**。未写数量时默认最多选9张。
5. 需要指定照片时点击 **设为必留**；需要覆盖两个人物时，分别指定 **人物A代表照** 和 **人物B代表照**。
6. 选择偏好模式；新相册可先使用 **不使用偏好**，积累反馈后再启用记忆。
7. 勾选两张照片，确认发送到自己的视频服务器，点击 **生成两张交互视频**。

示例提示词：

~~~text
挑4张雪山照片
挑一组适合发旅游朋友圈的照片
挑出最有代表性的6张照片，1张人物5张风景
挑出最有代表性的9张照片，两个不同的人最少分别有一张
~~~

最后一种要求需先指定两张人物代表照。数量、类别配额或必留条件冲突时，页面会提示调整要求。

### 可选：文字大模型解析

准备本地 Qwen3-VL-2B-Instruct：

~~~powershell
python scripts/install_qwen3vl_model.py
~~~

在网页展开 **文字大模型解析** 并启用。此选片步骤使用模型解析文字要求，图片相关性由本地 OpenCLIP 计算。云端解析与图片问答配置见 [云端模型配置](docs/cloud-analysis.md)。

### 视频探索

视频工作进程部署在 Linux GPU 服务器上；每个活跃场景使用一个独立工作进程和GPU。双图同时探索使用两个工作进程。本地网页通过 SSH 转发连接。

- **W / S**：前进、后退。
- **A / D**：向左、向右平移。
- **↑ / ↓ / ← / →**：转动视角。
- 点击一张视频卡片后，键盘操作作用于该卡片。
- 每次提交一个方向，等待该段生成完成后再继续。
- 7步累计约10.31秒，14步约20.81秒；卡片显示当前会话支持的步数。
- 全屏播放保留方向控件，视频可下载保存。

完整环境、权重、令牌、双工作进程与SSH命令见 [视频复现与部署](docs/reproduce.md#4-可选部署交互视频)。

## 最小复现

准备环境与模型后，用一张带来源和校验信息的公开照片跑通后端：

~~~powershell
python scripts/download_public_smoke_image.py
python scripts/reproduce_demo.py
~~~

脚本使用独立数据库执行：

~~~text
公开照片 → 质量与OpenCLIP索引 → 双MUSIQ评估 → 相似筛选 → 文字选片
~~~

完成后输出 passed 和 .norma/reproduce-*/result.json，其中保存处理结果、模型信息、选片数量和原图完整性检查。

使用自己的相册：

~~~powershell
python scripts/reproduce_demo.py --album "D:\Photos\Trip" --prompt "挑4张雪山照片" --min-aesthetic 5 --min-technical 40
~~~

更多验收步骤和故障处理见 [最小复现文档](docs/reproduce.md)。

## 技术实现

| 环节 | 实现 |
| --- | --- |
| 页面与API | Vue 3、TypeScript、FastAPI |
| 数据与任务 | SQLite、本地文件缓存、后台任务与进度 |
| 图文检索 | 多语言 OpenCLIP，512维归一化向量 |
| 技术/美学质量 | MUSIQ-KonIQ10k、MUSIQ-AVA |
| 相近构图 | pHash/dHash、OpenCLIP、SIFT/RANSAC |
| 文字解析 | 有界条件解析、可选 Qwen3-VL |
| 整组选片 | 相关性/质量/偏好融合、CP-SAT数量与配额约束 |
| 偏好记忆 | 查询相关案例检索、胜败向量对比、受限分数调整 |
| 图片问答 | 图像检索、视觉语言模型、引用与证据记录 |
| 视频探索 | LingBot-World-v2 1.3B、有状态动作续生、因果VAE增量解码 |

## 项目结构

~~~text
ai/
  aesthetics/       美学与技术质量
  index/            索引、图文编码、基础质量
  selection/        解析、去重、约束求解
  preferences/      偏好记录、案例记忆
  rag/              图片检索与问答
  exploration/      动作控制、视频会话、增量解码
  tests/            后端测试
  web_dist/         构建后的网页
src/components/     照片工作流、视频与方向控件
scripts/            模型准备、启动、复现、评测工具
docs/               使用方法、实现说明、开发记录
requirements-demo.txt          本地照片环境
requirements-world.txt         Linux GPU视频环境
requirements-world-service.txt 视频服务依赖
~~~

## 开发与测试

~~~powershell
corepack enable
pnpm install --frozen-lockfile
pnpm build
python -m pytest ai/tests -q
~~~

pnpm build 将网页构建到 ai/web_dist/。开发时可分别运行 python -m ai web 与 pnpm dev，默认后端端口8765、前端端口1420。

## 配置与数据

默认运行数据保存在 .norma/，模型缓存为 .norma/data/models/，Demo数据库为 .norma/demo-data/。照片、模型、数据库、密钥和运行报告由本机保存。

常用配置：

| 变量 | 用途 |
| --- | --- |
| NORMA_MODEL_CACHE_DIR | 模型缓存根目录 |
| NORMA_EMBEDDING_DEVICE | OpenCLIP设备：cpu / cuda / auto |
| NORMA_AESTHETICS_DEVICE | MUSIQ设备：cpu / cuda / cuda:0 |
| NORMA_VLM_PROVIDER | local / openai-compatible |
| NORMA_VIDEO_TOKEN_FILE | 私有视频服务令牌文件 |

Web默认监听127.0.0.1，供本机浏览器访问；远端视频服务使用令牌及SSH转发。模型和样例照片各自遵循上游许可，LingBot适配与部署条款见 [NOTICE](ai/exploration/NOTICE.md)。

## 文档

- [最小复现与视频部署](docs/reproduce.md)
- [当前Demo使用说明](docs/demo-quickstart.md)
- [项目技术与面试讲解](docs/interview-project-guide-20261006.md)
- [相近构图筛选实现](docs/fix-visual-curation-20261006.md)
- [数量、类别配额与整组选片](docs/fix-collection-selection-20261006.md)
- [偏好案例记忆](docs/preference-memory-demo.md)
- [云端模型配置](docs/cloud-analysis.md)
- [开发与优化记录](docs/gap-closure-20261005.md)
