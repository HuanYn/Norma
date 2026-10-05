# Wan 5B VAE 分块兼容性：真实依赖 CPU 复现

记录时间：2026-10-05 15:02:02 +08:00。结构化配置、源码哈希及边界见同目录 `wan-vae-tiling-cpu-reproduction-20261005.json`。

## 结论与证据范围

服务器实际安装的 Diffusers 0.35.1 中，`AutoencoderKLWan._encode` 第 1148–1149 行先进入分块分支，第 1152–1153 行的 `patchify` 未执行；`tiled_encode` 第 1313 行将 3 通道 RGB 直接交给需要 12 通道的编码器。官方 TI2V-5B 的 VAE 配置确实为 `patch_size=2`、`in_channels=12`。同版本非分块解码还包含 `unpatchify`，不能仅补编码后就认为分块路径完整兼容。

下面是实际库类的 CPU 实验，不是替身。为避免另起完整模型和占用显卡，只构造 672,608 参数的微型随机权重 VAE，保留触发故障所需的通道与 patch 配置。开启分块报错，关闭分块成功。完整第一任务没有保留错误堆栈，因此不能将本实验的堆栈冒充该历史任务的堆栈。

## 实际执行代码

以下代码通过现有 SSH 会话传给隔离环境的 Python 标准输入执行；未保存到服务器、未载入 checkpoint、未下载任何模型、未使用 GPU。

```python
import time
import traceback
import torch
import diffusers
import transformers
from diffusers import AutoencoderKLWan

torch.set_num_threads(1)
print('versions', torch.__version__, diffusers.__version__, transformers.__version__)
print('CPU-only random tiny AutoencoderKLWan, no checkpoint loaded')
vae = AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=4,
    dim_mult=[1, 2, 4, 4], num_res_blocks=1, in_channels=12, out_channels=12,
    is_residual=True, patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16,
    latents_mean=[0.0]*4, latents_std=[1.0]*4).to(device='cpu', dtype=torch.float32)
print('parameter_count', sum(p.numel() for p in vae.parameters()))
x = torch.zeros(1, 3, 1, 32, 32)
vae.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16,
                  tile_sample_stride_height=16, tile_sample_stride_width=16)
for tiling in (True, False):
    if not tiling: vae.disable_tiling()
    started = time.monotonic()
    print('use_tiling', tiling, flush=True)
    try:
        with torch.no_grad():
            result = vae.encode(x).latent_dist.mode()
        print('encoded_shape', tuple(result.shape))
    except Exception as exc:
        print('exception', type(exc).__name__, str(exc), flush=True)
        traceback.print_exc(limit=8)
    print('elapsed_seconds', round(time.monotonic()-started, 4), flush=True)
print('No model weights downloaded, no server files modified, no GPU used.')
```

## 原始合并输出

以下为终端实际返回的 stdout/stderr 合并内容，保留交错顺序；错误实验被显式捕获，所以进程最后正常退出。

```text
versions 2.8.0+cu128 0.35.1 4.57.6
CPU-only random tiny AutoencoderKLWan, no checkpoint loaded
parameter_count 672608
use_tiling True
exception RuntimeError Given groups=1, weight of size [8, 12, 3, 3, 3], expected input[1, 3, 3, 18, 18] to have 12 channels, but got 3 channels instead
Traceback (most recent call last):
  File "<stdin>", line 25, in <module>
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/diffusers/utils/accelerate_utils.py", line 46, in wrapper
    return method(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py", line 1191, in encode
    h = self._encode(x)
        ^^^^^^^^^^^^^^^
elapsed_seconds 0.006
use_tiling False
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py", line 1149, in _encode
    return self.tiled_encode(x)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py", line 1313, in tiled_encode
    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yinhuan/Norma-demo/env/lib/python3.11/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py", line 593, in forward
    x = self.conv_in(x, feat_cache[idx])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Given groups=1, weight of size [8, 12, 3, 3, 3], expected input[1, 3, 3, 18, 18] to have 12 channels, but got 3 channels instead
encoded_shape (1, 4, 1, 2, 2)
elapsed_seconds 0.0454
No model weights downloaded, no server files modified, no GPU used.
```

## 最小修复及资源边界

Norma 的 Worker 改为 `pipe.vae.disable_tiling()`，其余权重、FP32 VAE、BF16 Transformer、CPU offload 和生成配置保持不变。未升级依赖、未修改供应商源码。结果元数据增加 `vae_tiling: false`。

关闭分块可能提高 VAE 编解码峰值显存，当前仅建议先复测 640×384、49 帧，不保证 720P 或 121 帧一定适配 24GB。若 OOM，先减小输出面积/帧数或明确启用更慢的 sequential offload，不能静默更改实验参数。完整视频是否成功、耗时和画质，需要另一个独立的真实推理记录。

回归测试增加真实依赖的微型 CPU 用例：仅在 `diffusers==0.35.1` 环境运行，并验证分块失败、非分块成功。本机目标测试结果为 41 passed、2 skipped；其中该真实依赖测试因本机版本不同而跳过，另外一项 POSIX 权限检查因 Windows 跳过。服务器上的上述独立 CPU 实验已实际执行成功，不能将本机 skipped 写成 passed。

## 相关候选原因的排除证据

- 服务器 checkpoint 的 Transformer 配置 `image_dim=null`；实际 `pipeline_wan_i2v.py` 第 678 行只在 image_dim 非空时调用 CLIP 图像编码器，因此 `image_encoder=None / image_processor=None` 符合 5B 分支。
- 图片预处理第 693 行显式转 FP32，latent 条件随后按 `self.vae.dtype` 转换；本次 CPU 复现也使用 FP32，因此无需以改 dtype 作为此次修复。
- `expand_timesteps=True` 是 5B 的帧条件分支，非此次通道错配的根因。
- 本实验没有运行完整 UMT5，因此不宣称已证明 Transformers 的所有路径兼容；目前未见它导致本次确定性通道故障的证据。

来源：[Diffusers v0.35.1 官方 VAE 实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py)、[官方 5B checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)。上述行号来自服务器安装文件，已附 SHA-256，网页渲染行号不作为替代证据。
