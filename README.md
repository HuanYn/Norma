# Norma

Norma is a local-first photo intelligence website. A Python service reads a
local JPG/JPEG folder, creates disposable thumbnails and indexes, and serves a
Vue web interface in the browser. Original photos are never moved, deleted, or
uploaded.

The current MVP supports:

- fast recursive album cataloging with metadata and local thumbnails;
- on-demand quality/similarity analysis, semantic retrieval indexes, and conservative people grouping;
- bilingual natural-language selection with explicit hard constraints;
- OR-Tools CP-SAT optimization, auditable reasons, and locked replacement;
- a 67-dimensional Bayesian contextual pairwise-preference adapter over frozen
  OpenCLIP features, with immutable feedback/model history;
- CAPU-PDRR-MC active pair acquisition for asking decision-relevant preference
  questions under the current collection constraints;
- grounded multimodal RAG: learned OpenCLIP retrieval, pinned local Qwen3-VL
  claims/citations, and server-owned answer/provenance;
- one local website and one SQLite database, with no desktop runtime required;
- persistent album/history APIs and queued background analysis for large folders;
- default multilingual OpenCLIP retrieval with provider-versioned cache safety;
- an explicit zero-download handcrafted baseline for diagnostics and ablation;
- incremental indexing and resumable embedding that reuse unchanged photos;
- persistent human relevance labels and auditable Precision/Recall/nDCG/MRR reports;
- local YuNet/SFace people analysis that reuses unchanged face/no-face results;
- dry-run-first derived-cache cleanup and background embedding-model warmup;
- persisted maintenance audits and conservative disk-budget enforcement.

## Run the website

Requirements: Python 3.11+ and Node.js/pnpm for the one-time frontend build.

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
pnpm install
pnpm build
python -m ai web
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Enter an absolute local
folder path such as `D:\Photos\Trip`, then select **Open local folder**. Opening
an album only catalogs its JPG/JPEG files, reads basic metadata, and creates
local thumbnails; it does not automatically run quality, embedding, or face
analysis. The browser and API share the same local origin, and the photo folder
stays on the same machine.

The service binds to loopback (`127.0.0.1`) by default and has no authentication.
Keep that default for normal use; do not expose Norma to an untrusted network.

After the album opens, start only the modules you need from the three buttons:

- **质量与相似** computes quality signals, suggested exclusions, and perceptual-hash groups in one combined image pass;
- **语义索引** creates the image embeddings required for semantic text/image retrieval;
- **人脸分组** uses OpenCV YuNet and SFace to detect, align, describe,
  and conservatively group faces on this computer.

Each module runs as a persistent background job with a real percentage and
processed-photo count. It can be cancelled cooperatively, and the active job is
restored after a browser refresh. Previews are paged 300 at a time.

### Local face models

The default face provider is OpenCV YuNet 2023mar plus SFace. On the first
**人脸分组** run, Norma downloads the two pinned ONNX files (about 37 MB
combined), verifies their fixed SHA-256 digests, and atomically places them in
`.norma/data/models/opencv/` by default. Later runs use only that local cache.
The download contains model weights only: source photos, face crops, and
descriptors are never uploaded.

YuNet detects on a preview whose longest side is at most 1600 pixels with a
score threshold of 0.8. SFace then uses YuNet's five landmarks with
`alignCrop`, produces a 128-dimensional descriptor, and Norma L2-normalizes it
before clustering. Grouping first forms strict high-confidence seeds, then
rejoins pose-fragmented seeds only when they are mutual best prototype matches
and pass centroid, mean, and strongest-pair gates. Faces from one photo remain
a hard cannot-link throughout both passes. This is the versioned experimental
default for a personal organizer, not a production biometric guarantee. The
provider fingerprint includes
both model SHA prefixes, the alignment revision, and the clustering-policy
revision. Consequently, an
album produced by the old Haar/DCT provider or another model revision is shown
as needing a new people run instead of being treated as complete.

The OpenCV Zoo [YuNet model and files are MIT-licensed](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE),
while the [SFace model and files are Apache-2.0-licensed](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE).
If false merges are more costly than split groups, keep YuNet/SFace but disable
the experimental prototype pass with:

```powershell
$env:NORMA_FACE_PROVIDER = "opencv-yunet-sface-strict"
python -m ai web
```

If the model download is unavailable, the legacy Haar/DCT implementation can
be selected explicitly as a lower-quality fallback requiring no extra download:

```powershell
$env:NORMA_FACE_PROVIDER = "opencv-haar"
python -m ai web
```

After the frontend and model dependencies have been installed once, normal use
only needs:

```powershell
python -m ai web
```

Multilingual OpenCLIP is the default semantic provider. Confirm that its model
stack is available before clicking **语义索引**:

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
python -m ai --pretty providers
python -m ai web
```

The model is not bundled with Norma, and inference never resolves a mutable
Hub tag. Run the pinned setup command in
[docs/multimodal-provider.md](docs/multimodal-provider.md#install-and-enable)
once to download the exact model and tokenizer revisions; warmup then verifies
their complete SHA-256 manifest and runs offline. A missing or modified snapshot
is reported explicitly, and Norma never silently substitutes handcrafted
vectors. For a zero-download diagnostic or ablation, opt in deliberately:

```powershell
$env:NORMA_EMBEDDING_PROVIDER = "lightweight"
python -m ai web
```

See
[docs/multimodal-provider.md](docs/multimodal-provider.md) for cache, device,
offline, provider-switching, and the reproducible raw-v2 CPU smoke test. Its
runtime figure is generated from checked-in JSON observations; it is an
engineering measurement, not a retrieval-accuracy claim.

### Learned preference and grounded RAG

Create a semantic selection, then use **A/B preference** in the website to
record which photo you prefer. With the default 512D OpenCLIP provider, each
comparison trains a versioned 67D Bayesian contextual posterior. Subsequent
search, selection, and replacement requests use
`OpenCLIP cosine + learned residual`; when no compatible feedback exists, the
score is exactly OpenCLIP cosine. Exact-count, minimum-quality, reject, and
similarity-group limits remain hard constraints and are never learned away.

The backend also exposes CAPU-PDRR-MC active pair suggestions. It chooses a
comparison that is expected to reduce posterior decision regret for the current
constrained collection, rather than simply showing an arbitrary pair. The
end-to-end PowerShell workflow and one-shot feedback contract are documented in
[docs/web.md](docs/web.md#learned-preference-and-active-pair-questions).

Grounded RAG is currently a backend endpoint. It retrieves at most six originals
with OpenCLIP, sends only those in-memory byte snapshots to a pinned local
`Qwen/Qwen3-VL-2B-Instruct` runtime, accepts only structured claims/citations,
and lets the server construct the answer and provenance. Install and call it via
[docs/grounded-multimodal-rag.md](docs/grounded-multimodal-rag.md). The endpoint
checks citation and provenance integrity, but does **not** verify that every
claim is semantically entailed by its cited pixels.

The explicit provisioning command downloads the fixed Qwen revision, verifies
all 11 assets against the checked-in SHA-256 manifest, and publishes the local
directory only after the complete snapshot passes:

```powershell
python scripts/install_qwen3vl_model.py
```

No API or website request downloads generation weights.

Use a different state directory or port when needed:

```powershell
python -m ai --data-dir D:\NormaData web --port 8879
```

## Develop the web interface

Run the Python API and Vite development server in separate terminals:

```powershell
python -m ai serve
pnpm dev
```

Open [http://127.0.0.1:1420](http://127.0.0.1:1420). Vite proxies the local API
and media routes to port 8765. Production assets are generated under
`ai/web_dist/` and are served by FastAPI.

## Direct Python commands

The CLI is useful for automation and diagnostics, but the supported end-user
interface is the website.

```powershell
python -m ai --pretty prepare "D:\Photos\Trip"
python -m ai --pretty albums
python -m ai --pretty search ALBUM_ID "夜景 blue city"
python -m ai --pretty select ALBUM_ID "选 12 张夜景，质量至少 45，相似组最多 1 张"
```

The installed `norma ...` command is equivalent to `python -m ai ...`. See
[docs/python-cli.md](docs/python-cli.md) for every command and
[docs/web.md](docs/web.md) for the browser workflow.

## Public demo photos

If no local album is available, the live-search downloader can assemble a
licensed Wikimedia Commons demo album. Its results depend on the current
Commons search response, so it is a convenient demo input—not a frozen
experiment fixture. Image files stay untracked and `ATTRIBUTION.json` records
their source, creator, and license.

```powershell
python scripts/download_demo_album.py --count 72 --output .norma/demo-album
python scripts/build_demo_eval_album.py
```

The controlled preference experiments use a separate fixed 72-image manifest
and verifier; see `fixtures/contextual_preference_wikimedia_20260814.json` and
`scripts/download_contextual_preference_fixture.py`. The manifest is the
historical scientific input contract, while `download_demo_album.py` remains
exploratory. Wikimedia thumbnail responses are mutable: the verifier refuses
upstream byte drift instead of silently changing a completed experiment, so an
exact rerun still requires the pinned local files or a future licensed,
content-addressed archive.

For people-pipeline validation:

```powershell
python scripts/download_demo_album.py --count 30 `
  --output .norma/demo-portraits `
  --search "portrait face photograph" `
  --search "headshot portrait photography"
python scripts/build_demo_people_eval_album.py
```

For the small OpenCLIP/Qwen smoke commands documented below, download the
content-pinned CC BY-SA fixture separately:

```powershell
python scripts/download_public_smoke_image.py
```

## Validation and design notes

```powershell
python -m pytest ai/tests -q
pnpm test:ui
pnpm build
```

Architecture and evidence:

- [Architecture](docs/architecture.md)
- [Retrieval benchmark](docs/benchmarks/retrieval-e2e.md)
- [Legacy Haar/DCT people lifecycle smoke](docs/benchmarks/people-e2e.md)
- [Selection benchmark](docs/benchmarks/selection-e2e.md)
- [Preference/replacement benchmark](docs/benchmarks/preference-replacement-e2e.md)
- [Direct Python benchmark](docs/benchmarks/python-cli-e2e.md)
- [Backend library lifecycle](docs/backend-library-lifecycle.md)
- [Library lifecycle benchmark](docs/benchmarks/library-lifecycle-e2e.md)
- [Multilingual OpenCLIP provider](docs/multimodal-provider.md)
- [OpenCLIP public-data benchmark](docs/benchmarks/openclip-e2e.md)
- [Raw multilingual OpenCLIP proxy evaluation](docs/benchmarks/openclip-raw-v2-proxy-20260828.md)
- [67D contextual preference controlled report](figures/CONTEXTUAL_PREFERENCE_CONTROLLED_REPORT.md)
- [CAPU-PDRR-MC controlled acquisition report](figures/PDRR_ACQUISITION_CONTROLLED_REPORT.md)
- [Grounded multimodal RAG and local Qwen3-VL usage](docs/grounded-multimodal-rag.md)
- [Incremental prepare benchmark](docs/benchmarks/incremental-prepare-e2e.md)
- [Retrieval evaluation workflow](docs/retrieval-evaluation.md)
- [Retrieval evaluation benchmark](docs/benchmarks/retrieval-evaluation-e2e.md)
- [Incremental people benchmark](docs/benchmarks/incremental-people-e2e.md)
- [Cache maintenance and warmup](docs/cache-maintenance.md)
- [Maintenance benchmark](docs/benchmarks/cache-maintenance-e2e.md)
- [Maintenance audit benchmark](docs/benchmarks/maintenance-audit-e2e.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

The default semantic provider is frozen multilingual OpenCLIP. Raw Chinese and
English query text goes directly to its multilingual text tower. The
deterministic 16-dimensional provider remains an explicit CPU baseline, not a
silent fallback. A legacy Chinese-keyword-to-English bridge is retained only
as an explicit ablation provider.
The default face pipeline now uses the local YuNet/SFace models and two-stage,
constrained deterministic grouping. Its groups are organizational suggestions,
not claims of biometric identity.

The frozen OpenCLIP/Qwen3-VL inference stack plus the small Bayesian preference
adapter is not DPO, SFT, LoRA, or end-to-end multimodal fine-tuning. The checked-in
preference experiments support contextual learning over zero-feedback cosine and
an exploratory low-feedback-budget advantage for CAPU-PDRR-MC in their stated
semi-synthetic public-image protocol; they do not establish universal or
real-user superiority.
