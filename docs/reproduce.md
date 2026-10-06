# 最小复现与部署

本文从新克隆仓库开始，分为本地照片流程、可选文字解析、远端交互视频三部分。所有命令从仓库根目录执行。

## 1. 安装本地照片环境

本地模型使用CPU，建议至少16GB内存，并预留模型、依赖和照片缓存空间。使用Python 3.11～3.13的独立环境。仓库已包含构建好的网页，运行时可以直接使用。

Windows PowerShell：

~~~powershell
git clone https://github.com/HuanYn/Norma.git
cd Norma
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python -m pip install pyiqa==0.1.16 --no-deps
python scripts/install_demo_models.py
~~~

Linux/macOS激活环境使用 source .venv/bin/activate。PyTorch安装若需平台专用wheel，先安装对应torch/torchvision版本，再安装其余依赖。

requirements-demo.txt固定照片链路的主要依赖；pyiqa使用单独的--no-deps安装命令，仅配合本清单中MUSIQ需要的依赖。可选的其他pyiqa模型需准备各自依赖。

模型准备包含：

| 组件 | 固定版本与保存位置 |
| --- | --- |
| OpenCLIP | laion5b_s13b_b90k；.norma/data/models/openclip/ |
| MUSIQ-KonIQ10k | musiq_koniq_ckpt-e95806b9.pth |
| MUSIQ-AVA | musiq_ava_ckpt-e8d3f067.pth |
| 两套MUSIQ目录 | .norma/data/models/musiq/ |

OpenCLIP模型和tokenizer修订版记录在 ai/index/openclip_xlm_roberta_manifest.json；MUSIQ文件SHA和仓库修订版记录在 ai/aesthetics/provider.py。下载后执行完整哈希检查。

已有模型可只做离线校验：

~~~powershell
python scripts/install_demo_models.py --offline
~~~

自定义缓存时，对安装和运行使用同一个目录：

~~~powershell
python scripts/install_demo_models.py --cache-dir "D:\NormaModels"
$env:NORMA_MODEL_CACHE_DIR = "D:\NormaModels"
~~~

## 2. 自动复现照片链路

下载带固定内容哈希和归属信息的公开JPEG：

~~~powershell
python scripts/download_public_smoke_image.py
python scripts/reproduce_demo.py
~~~

默认样例来自Wikimedia Commons：Gothic-architecture-banner.jpg，作者Traveler100，CC BY-SA 3.0。脚本同时生成归属信息；上游返回内容变化时校验会停止，可按下述命令改用自己的JPG/JPEG照片。

自动复现会：

1. 创建新的 .norma/reproduce-随机ID/ 目录和独立SQLite数据库。
2. 通过与网页相同的API路由运行基础质量、相似分析和OpenCLIP编码。
3. 调用两套真实MUSIQ模型并读取评分。
4. 进行相近构图筛选，再执行“选1张建筑照片”。
5. 检查可行性、实际张数和原图SHA一致性。
6. 保存 result.json，成功时终端显示 passed。

该入口使用进程内HTTP测试客户端，实际模型在CPU运行。它独立配置数据、模型设备和偏好开关，无需先启动网页服务或提供云端密钥。首次模型加载需要等待；默认每个后台准备阶段等待上限1800秒，取消会在模型处理检查点生效。

默认门槛为美学1、技术0，便于单张样例完成链路。对实际相册可指定产品常用门槛：

~~~powershell
python scripts/reproduce_demo.py --album "D:\Photos\Trip" --prompt "挑4张雪山照片" --min-aesthetic 5 --min-technical 40
~~~

可以指定结果目录，每次使用一个新目录：

~~~powershell
python scripts/reproduce_demo.py --output .norma/reproduce-first-run
~~~

增加公开演示照片：

~~~powershell
python scripts/download_demo_album.py --count 24 --output .norma/public-album
python scripts/reproduce_demo.py --album .norma/public-album --prompt "选4张旅行照片"
~~~

该相册下载器会保存每张图片的来源与许可；搜索下载的具体图片随上游内容变化。

### 网页操作

~~~powershell
python scripts/start_demo.py
~~~

打开 http://127.0.0.1:8767/：

1. 导入同一个相册目录。
2. 点击“美学评估 · 相似只留最佳”。
3. 输入“挑4张雪山照片”或其他要求后点“模型选片”。
4. 根据需要指定必留照片与人物A/B代表照。
5. 在结果中查看选片原因，或调整门槛重新筛选。

自动复现与网页默认使用不同数据库，因此第一次在网页打开相册时需要进行网页自己的准备任务。要查看复现库中的相册，可显式启动：

~~~powershell
python scripts/start_demo.py --data-dir .norma/reproduce-first-run/data
~~~

有本地预览构建时，启动器优先使用 .norma/demo-web-dist；否则使用仓库中的 ai/web_dist。指定 --web-dist ai/web_dist 可显式使用本次仓库构建。

## 3. 可选文字解析与图片问答

准备本地模型：

~~~powershell
python scripts/install_qwen3vl_model.py
~~~

在Demo页面勾选“文字大模型解析”。模型为Qwen3-VL-2B-Instruct，Demo启动器默认采用本地解析配置。文字解析负责整理查询语义，后端执行数量、配额和必留条件。

图片问答使用独立的检索与视觉语言模型接口。云端地址、模型名称、私有密钥和授权方式见 [cloud-analysis.md](cloud-analysis.md)。

偏好可以选择“仅本人反馈”“助手审美”或“不使用偏好”。全新数据库先通过网页积累反馈；历史代理案例的来源与导入说明见 [proxy-preferences.md](proxy-preferences.md)。

## 4. 可选部署交互视频

### 4.1 准备Linux GPU环境

以下以Linux x86_64、Python3.11、RTX3090 24GB为参考。远端显卡需有兼容CUDA12.8 wheel的驱动。每个worker绑定一张空闲GPU；双图同时探索需要两张可用GPU。只有一张卡时，启动8769工作进程，在 http://127.0.0.1:8767/demo/world/ 上传一张照片并逐张生成。

在远端单独克隆仓库、创建环境：

~~~bash
git clone https://github.com/HuanYn/Norma.git
cd Norma
python3.11 -m venv world-env
source world-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-world.txt -r requirements-world-service.txt
mkdir -p .norma
git clone https://github.com/Robbyant/lingbot-world-v2.git .norma/lingbot-world-v2
git -C .norma/lingbot-world-v2 checkout --detach 1895d300d8ac936401689b26389f51cbd36530eb
python scripts/download_world_models.py --destination .norma/world-models --acknowledge-noncommercial
~~~

模型与共享资产约17.44GiB，下载器检查可用空间并保存逐文件清单。LingBot上游及适配遵循CC BY-NC-SA 4.0，部署前阅读 [NOTICE](../ai/exploration/NOTICE.md)。

requirements-world.txt固定GPU依赖，与本地照片环境分开使用。远端从仓库根目录直接运行脚本，使用这份独立依赖清单即可。

### 4.2 生成私有令牌

在远端仓库根目录执行一次，文件必须尚不存在：

~~~bash
python -c "from pathlib import Path; from scripts.video_worker import create_private_token; _ = create_private_token(Path('.norma/video-worker.token'))"
~~~

本地创建 .norma 目录后，通过自己的SSH连接复制令牌。替换USER、HOST和REMOTE_NORMA_DIR为实际配置：

~~~powershell
New-Item -ItemType Directory -Force .norma
scp USER@HOST:REMOTE_NORMA_DIR/.norma/video-worker.token .norma/video-worker.token
~~~

本地与远端使用同一个令牌文件。保存在私有目录，并保持远端文件仅属主可读写。

### 4.3 启动两个worker

先查看GPU UUID：

~~~bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
~~~

在两个远端终端分别激活world-env，替换GPU_UUID_1与GPU_UUID_2：

~~~bash
CUDA_VISIBLE_DEVICES=GPU_UUID_1 python scripts/world_worker.py \
  --source .norma/lingbot-world-v2 --models .norma/world-models \
  --data-dir .norma/world-slot-1 --token-file .norma/video-worker.token \
  --port 8769 --decode-mode prefix --max-steps 7
~~~

~~~bash
CUDA_VISIBLE_DEVICES=GPU_UUID_2 python scripts/world_worker.py \
  --source .norma/lingbot-world-v2 --models .norma/world-models \
  --data-dir .norma/world-slot-2 --token-file .norma/video-worker.token \
  --port 8770 --decode-mode stream --max-steps 14
~~~

GPU UUID的实际值以GPU-开头。启动器检查当前占用，选择空闲GPU即可。两张卡的会话独立；第一张采用7步前缀参考解码，第二张采用14步增量解码。

### 4.4 本地连接与操作

本地保持SSH转发终端开启：

~~~powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 8769:127.0.0.1:8769 -L 8770:127.0.0.1:8770 USER@HOST
~~~

另一终端启动本地网页：

~~~powershell
python scripts/start_demo.py
~~~

在选片结果中勾选两张照片，确认上传到自己的服务器后生成。等待初始化完成，点击当前卡片，通过WASD和方向键继续探索。生成期间方向操作暂时锁定；会话达到步数上限后可新建。

初始化包含模型加载和图像条件编码，参考运行约3分钟；后续每个动作继续生成一段。累计7步165帧、10.3125秒，14步333帧、20.8125秒，输出以16fps播放。

结束时先关闭网页中的会话，再在各自远端worker终端按Ctrl+C停止服务。已保存的视频和权重可继续保留，下一次使用时再启动服务。

## 5. 前端开发与测试

~~~powershell
corepack enable
pnpm install --frozen-lockfile
pnpm build
python -m pytest ai/tests -q
~~~

开发模式：终端一运行 python -m ai web，终端二运行 pnpm dev；打开 http://127.0.0.1:1420/，API代理到8765。

网页启动后可运行只读首页检查：

~~~powershell
pnpm exec playwright install chromium
node scripts/check_release_ui.cjs http://127.0.0.1:8767 .norma/release-ui-first
~~~

使用已安装的Edge时，可设置 NORMA_BROWSER_CHANNEL=msedge 后运行该脚本。输出目录保存网页截图与result.json。其他带trip/world日期样例的历史检查脚本使用各自开发数据，通用复现使用本页的reproduce_demo.py和check_release_ui.cjs。

只读检查工具：

~~~powershell
python scripts/demo_doctor.py
~~~

检查会列出依赖、模型文件、网页和视频连接状态。使用照片流程时，视频或可选解析项可以在准备相应功能时再配置。

## 6. 常见问题

| 现象 | 处理 |
| --- | --- |
| OpenCLIP或MUSIQ权重缺失 | 运行install_demo_models.py，并核对运行时模型缓存目录 |
| 公开样例下载失败或内容校验失败 | 检查网络，或用--album指定有使用权的JPG/JPEG目录 |
| 找不到网页构建 | pnpm install --frozen-lockfile后运行pnpm build，启动时指定--web-dist ai/web_dist |
| PowerShell禁止环境激活 | 直接用 .venv\\Scripts\\python.exe 执行各条Python命令 |
| 无法选够指定数量 | 查看折叠原因，调整质量门槛、类别配额或数量 |
| 人物覆盖要求缺少代表照 | 先在照片卡片指定人物A与B代表照 |
| 视频无法连接 | 检查两个worker、SSH转发、本地和远端的令牌是否一致 |
| worker提示GPU忙 | 使用空闲GPU UUID，等待已有任务结束 |
| 关闭浏览器后GPU仍占用 | 关闭对应会话，演示结束后停止Norma worker |
| 同一路径换图后缓存过期 | 重新运行准备与评估，生成匹配新图片的缓存 |

## 7. 发布验证记录

本次发布验证记录见 [release-20261006.md](release-20261006.md)。后端测试、前端构建和CPU真实模型复现分别保存结果，便于检查环境与执行范围。
