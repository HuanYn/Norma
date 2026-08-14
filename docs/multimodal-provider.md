# Multilingual OpenCLIP provider

Norma keeps `lightweight-semantic-v1` as the zero-download default and offers
`openclip-multilingual` as an optional, real image/text embedding backend. The
optional provider uses LAION's multilingual XLM-R + ViT-B/32 OpenCLIP model and
produces normalized 512-dimensional vectors for image and text retrieval.

## Install and enable

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
$env:NORMA_EMBEDDING_PROVIDER = "openclip-multilingual"
python -m ai providers
python -m ai web
```

The first inference downloads model files to `.norma/data/models` by default.
The tested cache occupied about 1.38 GB. Override runtime settings when needed:

```powershell
$env:NORMA_MODEL_CACHE_DIR = "D:\NormaModels"
$env:NORMA_EMBEDDING_DEVICE = "auto" # auto, cpu, or cuda
$env:NORMA_EMBEDDING_BATCH_SIZE = "16"
```

Set `HF_TOKEN` only if the Hugging Face download path requires authentication.
After a successful download, an already populated cache can be used with
`HF_HUB_OFFLINE=1`. Norma points both OpenCLIP downloads and OpenCLIP's implicit
Transformers/XLM-R lookups at the configured Norma model cache, so offline
reload does not depend on a separate user-level Hugging Face cache.

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

SQLite schema v4 records `embedding_provider` for every photo. Search,
selection, and replacement reject missing or mixed-provider caches. Switching
between lightweight and OpenCLIP therefore requires running `embed` or
`prepare` again for the album; existing vectors are treated as stale and are
never compared across incompatible spaces.

For the currently supported bounded Chinese concepts, Norma converts the
recognized concept into an auditable English CLIP prompt before inference.
Unrecognized Chinese text remains unchanged rather than being guessed. This
bridge complements the multilingual text tower and preserves the selection
parser's visible unsupported-concept behavior.

## Native dependency compatibility

`open-clip-torch` imports PyTorch, torchvision, and in some environments
torchaudio through Transformers. Install mutually compatible builds from the
same PyTorch release/channel, especially when choosing CPU versus CUDA wheels.
Norma does not bundle those native wheels or model weights.
