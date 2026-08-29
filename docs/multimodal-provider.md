# Multilingual OpenCLIP provider

Norma uses `openclip-multilingual` as the default learned image/text embedding
backend. `lightweight-semantic-v1` remains a zero-download handcrafted baseline
that must be selected explicitly; it is never a silent fallback. The default
provider uses LAION's multilingual XLM-R + ViT-B/32 OpenCLIP model and
produces normalized 512-dimensional vectors for image and text retrieval.

## Install and enable

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
$env:NORMA_OPENCLIP_CACHE = (New-Item -ItemType Directory -Force ".norma/data/models/openclip").FullName
@'
import os
from huggingface_hub import snapshot_download

cache = os.environ["NORMA_OPENCLIP_CACHE"]
snapshot_download(
    repo_id="laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    revision="506d40eb551f4801a1c27fc20a31c7b8f590deda",
    allow_patterns=["open_clip_pytorch_model.bin"],
    cache_dir=cache,
)
snapshot_download(
    repo_id="xlm-roberta-base",
    revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    allow_patterns=[
        "config.json",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    cache_dir=cache,
)
'@ | python -
python -m ai providers
python -m ai --pretty warmup
python -m ai web
```

`NORMA_EMBEDDING_PROVIDER=openclip-multilingual` is optional because it is the
default. If the dependency stack or model cache is unavailable, semantic
indexing returns an explicit error instead of producing handcrafted vectors.
Use the baseline only when that change is intentional:

```powershell
$env:NORMA_EMBEDDING_PROVIDER = "lightweight"
python -m ai web
```

Norma inference never downloads or resolves a mutable Hub tag. The explicit
setup command above downloads two immutable revisions into
`.norma/data/models/openclip`; startup/warmup fails closed if that exact local
snapshot is absent, incomplete, contains an unpinned runtime asset, or fails a
full SHA-256 check. With `NORMA_MODEL_CACHE_DIR=D:\NormaModels`, use
`D:\NormaModels\openclip` as the download cache instead. The existing
single-machine public-data run used about 1.38 GB of model cache.
This is an observed cache size, not a guaranteed download size across package
revisions. Override runtime settings when needed:

```powershell
$env:NORMA_MODEL_CACHE_DIR = "D:\NormaModels"
$env:NORMA_EMBEDDING_DEVICE = "auto" # auto, cpu, or cuda
$env:NORMA_EMBEDDING_BATCH_SIZE = "16"
```

Set `HF_TOKEN` only for the separate pinned download command if required.
During inference Norma forces both Hugging Face Hub and Transformers offline.
The XLM-R model config and tokenizer are loaded only from the verified local
snapshot, so reload does not depend on a separate user-level cache.

## Background warmup

Inspect state without loading the model, then trigger an idempotent background
load:

```text
GET  /providers/embedding/status
POST /providers/embedding/warmup
```

The POST returns HTTP 202 with `loading` or `ready`; poll the status endpoint for
`ready` or `failed`. Repeated requests while loading share the same attempt.
Set `NORMA_PREWARM_EMBEDDING=1` to submit this warmup automatically when the
website starts. The Python CLI command `python -m ai --pretty warmup` performs a
synchronous probe for scripts that need the model ready before continuing.

The model is loaded lazily on the first embedding request, not during API
startup. Image inference is batched. Norma validates vector shape, finite
values, and normalization before committing a complete album cache. A failed
model load returns HTTP 503 instead of silently changing providers.

The exact model is
[`laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k),
whose model card lists an MIT license. OpenCLIP's official
[`PRETRAINED.md`](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md)
describes it as the first multilingual OpenCLIP trained on LAION-5B.

## Provider contract and cache safety

Inspect available providers without loading the model:

```powershell
python -m ai --pretty providers
```

The website/API equivalent is `GET /providers/embedding`. It reports the active
provider, its versioned cache identity, dimensions, availability, and install
requirements.

SQLite schema v4 introduced `embedding_provider`, and schema v5 added the
indexed source fingerprint used for every vector; schema v13 additionally
binds vectors to source-content SHA-256. Search, selection, and replacement
reject missing, stale, or
mixed-provider caches. Switching among
lightweight, raw OpenCLIP, and the legacy bridge therefore requires running
`embed` or `prepare` again for the album; existing vectors are never compared
across incompatible spaces.

The canonical v3 provider identity embeds cryptographic digests of the pinned
manifest, curated directly output-affecting Python package versions, query
contract, resolved `cpu`/`cuda` backend, and the native numeric-threading
contract.
Raw Chinese and English
queries go directly to the XLM-R text tower, so a phrase such as “红色裙子的女孩
在海边” is not reduced to a few dictionary concepts. The former bounded
Chinese-to-English rewrite is available only as
`NORMA_EMBEDDING_PROVIDER=openclip-legacy-bridge`, with a distinct query-contract
digest, for an auditable ablation. All old raw-v2, bridge-v1, and generic-v1
caches are deliberately stale after this upgrade; rerun **语义索引** once.
CPU-produced vectors are never reused by a CUDA query (or vice versa). The
fingerprint guarantees the same model/runtime/backend contract, not bitwise
identity across different physical CPUs or GPUs.

### Pinned v3 local smoke

Download the licensed, content-pinned Wikimedia fixture and run one real local
image/text inference smoke after the model snapshots have been installed:

```powershell
python scripts/download_public_smoke_image.py
python figures/benchmark_openclip_pinned_identity_smoke.py `
  --cache-root .norma/data/models `
  --image .norma/public-smoke/gothic-architecture-banner.jpg `
  --output figures/openclip_pinned_v3_smoke_20260829.json
```

The downloader verifies the fixture's byte size and SHA-256 and writes adjacent
CC BY-SA 3.0 attribution. The smoke records the full provider fingerprint,
runtime versions, vector hashes, latency, and peak RSS. It is an identity and
inference check, not a retrieval-quality benchmark.

## Raw-v2 CPU smoke test (2026-08-28)

On an Intel Core Ultra 7 255HX with 31.43 GiB RAM, the completed raw-v2 cache
occupied 1.377 GiB, excluding an abandoned partial download. A successful
recovery warmup after that partial cache took 276.544 seconds wall-clock
(272.621 seconds reported inside the provider); because it reused partial data,
this is not a clean full-download measurement. Two fresh, network-disabled
processes loaded the cache and encoded their first Chinese text in 41.032 and
48.303 seconds. Four subsequent text encodes took 0.084--0.125 seconds, and one
public LFW image encode took 0.308 seconds.

The smoke test confirmed finite, unit-normalized 512-dimensional vectors and
verified that the Unicode Chinese query was passed through unchanged. The
Chinese phrase and its English paraphrase had cosine similarity 0.86794; this is
only a cross-language sanity check, not a retrieval-accuracy result. Raw
observations, environment metadata, a reproducible plotting script, and the
reviewed log-scale point plot are in
[`figures/openclip_raw_v2_runtime_20260828.json`](../figures/openclip_raw_v2_runtime_20260828.json)
and [`figures/fig1_openclip_raw_v2_runtime.pdf`](../figures/fig1_openclip_raw_v2_runtime.pdf).

On the tested machine, the Hugging Face Xet transport left an incomplete cache
during the first attempt. Retrying the warmup with
`HF_HUB_DISABLE_XET=1` completed successfully, after which fresh processes
loaded with both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Use this only
as a troubleshooting override when Xet transfer makes no progress; it is not a
model or retrieval setting.

The historical 81-image CPU run measured 51.93 seconds for one cold embedding
process and 5.50 seconds for a warm full re-embedding, with about 1.38 GB in the
model cache. Those are single-run integration measurements for the previous
legacy query protocol. Linear extrapolation gave roughly 15 minutes for 1,407
photos, but startup, image dimensions, batch size, and CPU contention can
change the result; it is a planning estimate, not a raw-v2 benchmark claim.

## Native dependency compatibility

`open-clip-torch` imports PyTorch, torchvision, and in some environments
torchaudio through Transformers. Install mutually compatible builds from the
same PyTorch release/channel, especially when choosing CPU versus CUDA wheels.
Norma does not bundle those native wheels or model weights.

On Windows, Anaconda NumPy/MKL and the PyPI Torch wheel can carry different
copies of `libiomp5md.dll`. Loading both copies terminates the interpreter with
OpenMP Error #15 and can otherwise produce incorrect numeric results. Norma
therefore selects `MKL_THREADING_LAYER=SEQUENTIAL` in the package entry point,
before importing NumPy, OpenCV, or Torch. NumPy continues to use MKL kernels via
`mkl_sequential`, while Torch owns the only Intel OpenMP runtime. OpenCLIP
loading fails closed if an explicit non-sequential layer is configured or if a
foreign Intel OpenMP DLL was already initialized before Norma was imported.

Do not set `KMP_DUPLICATE_LIB_OK`: it suppresses the runtime check without making
the two runtimes safe, so Norma rejects truthy values. If another host program
performed threaded NumPy/MKL work before importing `ai`, restart that process
and import Norma first. The diagnosis, DLL identities, correctness regression,
and representative microbenchmarks are recorded in
[`benchmarks/windows-numeric-runtime.md`](benchmarks/windows-numeric-runtime.md).
