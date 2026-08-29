# Multilingual OpenCLIP end-to-end check

This historical check used 81 public Wikimedia Commons images in an ignored
local fixture, with attribution retained in `ATTRIBUTION.json`. It validated
the original OpenCLIP integration and gives directional retrieval evidence; it
is not a curated relevance benchmark and is not a measurement of the current
raw multilingual-query provider.

## Environment and pipeline

- provider: `openclip-xlm-roberta-base-vit-b-32-laion5b-v1` (now retained as
  the explicit `openclip-legacy-bridge` ablation)
- device: PyTorch CPU
- vector dimension: 512
- indexed: 81 images
- rejected suggestions: 4
- similarity groups: 7
- indexing: 7.79 seconds
- first cold embedding process: 51.93 seconds
- another cold process: 44.68 seconds
- warm re-embedding of all 81 images: 5.50 seconds
- pipeline errors: 0
- downloaded model cache: about 1.38 GB
- cached model reload with `HF_HUB_OFFLINE=1`: passed

Every sampled image and text vector was finite and unit-normalized. A real
Uvicorn process also reported the active provider and returned versioned search
results.

## Coarse Precision@10 comparison

Labels were derived from the fixture's original Wikimedia search attribution,
so they are noisy and biased toward the download queries.

| English query | Lightweight | OpenCLIP |
| --- | ---: | ---: |
| travel architecture | 0.60 | 0.60 |
| city night photography | 0.90 | 0.80 |
| mountain travel landscape | 0.50 | 1.00 |

Using the bounded Chinese concept bridge, this legacy provider scored 0.50 for
`旅行建筑`, 0.60 for `城市夜景摄影`, and 0.90 for `山地旅行风景`. Those Chinese
figures must not be attributed to the current
`openclip-xlm-roberta-base-vit-b-32-laion5b-raw-v2` provider, which sends the
original multilingual query directly to XLM-R and requires a fresh benchmark.

Qualitative open-vocabulary checks placed St. Patrick's Cathedral first for
`church cathedral`, both transit maps at ranks 1 and 3 for `map diagram`, and
Budapest Chain Bridge at rank 2 for `bridge`. The last query also had a false
positive at rank 1.

## Interpretation

OpenCLIP is not uniformly better on this small fixture: the lightweight color
and luminance descriptor was stronger for the biased night query. OpenCLIP's
clear gains were mountain/landscape retrieval and concepts outside the bounded
lightweight vocabulary. A larger hand-labelled multilingual benchmark is still
needed before claiming general quality superiority. Treat the timing and cache
figures above as engineering baselines for the same model family, and rerun the
retrieval metrics before comparing the current raw-v2 query path with either
the handcrafted or legacy-bridge ablation.
