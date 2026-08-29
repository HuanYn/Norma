# Norma local website

## Production-style local run

Build the Vue assets once, then let Python serve the complete website:

```powershell
python -m pip install -e ".[dev,selection,multimodal]"
pnpm install
pnpm build
python -m ai web
```

Open `http://127.0.0.1:8765`. The web assets, JSON API, thumbnails, and face
crops use the same local origin. The default host is loopback-only and the
service has no authentication; do not set `NORMA_HOST` to a non-loopback
interface on an untrusted network.

## Album workflow

1. Paste an absolute JPG/JPEG folder path in **Library**.
2. Select **Open local folder**. This first job only catalogs files, reads basic
   metadata, and creates disposable thumbnails. It does not automatically run
   quality scoring, semantic embeddings, or face detection, so the photo grid
   becomes usable without waiting for every analysis module.
3. Start the analysis modules you actually need from the compact button row:
   **质量与相似** computes quality signals and perceptual hashes together from
   the same decoded image; **语义索引** creates retrieval embeddings; **人脸分组**
   runs local YuNet detection, five-point SFace alignment/description, and
   conservative grouping.
4. Review the photo grid. Suggested exclusions and similarity groups appear
   after **质量与相似** completes.
5. Open **AI Selection** after the required modules are ready. Semantic queries
   such as `night blue` require **语义索引**; constrained requests such as
   `选 12 张夜景，质量至少 45` also require **质量与相似**.
6. For a selection, inspect per-photo reasons, replace one item, or use
   **A/B preference** to mark one photo preferred to another. Under the
   default OpenCLIP provider, this records an immutable 67D contextual event and
   trains a new versioned Bayesian posterior. Create a new search/selection or
   replacement request to observe the current posterior's score.

The folder import and each analysis button create a SQLite-backed background
job. The active button shows a real 0–100% progress bar and processed-photo
count. **Cancel** stops cooperatively after the current photo or model batch; it
never interrupts an original file mid-read. Job identity and progress survive a
browser refresh, and the UI reconnects to the active job. Norma runs one heavy
album job at a time so concurrent modules do not compete for the local machine.
The first 300 previews load after the base import, and later pages load on
demand.

Runtime state defaults to `.norma/data`. Override it before the `web` command:

```powershell
python -m ai --data-dir D:\NormaData web --port 8879
```

## Learned preference and active pair questions

The current browser comparison flow collects valid contextual pairwise
feedback from selected photos. The CAPU-PDRR-MC question selector is additionally
available through HTTP so the backend can choose a pair that is expected to
reduce decision regret under the current exact-count and similarity-group
constraints.

First create a feasible semantic selection with the default 512D OpenCLIP
provider and a complete **语义索引**:

```powershell
$albumId = "YOUR_ALBUM_ID"
$selection = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/selections" `
  -ContentType "application/json" `
  -Body (@{
    album_id = $albumId
    prompt = "选 6 张城市夜景，相似组最多 2 张"
    user_id = "local"
  } | ConvertTo-Json)
```

Ask for the production default (`B=64`, shortlist 16) pair:

```powershell
$suggestion = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/selections/$($selection.selection_id)/preference-pairs/suggest" `
  -ContentType "application/json" `
  -Body '{}'

$suggestion.left
$suggestion.right
```

If the left photo is preferred, consume that exact displayed suggestion once:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/feedback/pairwise" `
  -ContentType "application/json" `
  -Body (@{
    album_id = $albumId
    selection_id = $selection.selection_id
    suggestion_id = $suggestion.suggestion_id
    preferred_photo_id = $suggestion.left.photo_id
    rejected_photo_id = $suggestion.right.photo_id
    user_id = "local"
    choice = "preferred"
  } | ConvertTo-Json)
```

Swap `preferred_photo_id` and `rejected_photo_id` when the right photo wins.
`tie`, `skip`, and `both_bad` are accepted as immutable audit choices but do not
train the binary posterior. A suggestion is bound to its displayed provider,
selection, candidate/source/feature snapshot, and model-at-display; reusing it
returns HTTP 409. Previously shown pairs are excluded by default even after the
posterior version changes. `exhaustive=true` evaluates every eligible pair but
still uses Monte Carlo posterior integration and can be expensive.

With no compatible trained event, semantic ranking is exactly OpenCLIP cosine.
After feedback, search, semantic selection, replacement, and grounded RAG
retrieval load one compatible posterior and use
`cosine + posterior_mean · 67D_features`. Hard quality/reject/count/group
constraints are unchanged.

The controlled acquisition evidence is intentionally limited: the checked-in
[CAPU-PDRR-MC report](../figures/PDRR_ACQUISITION_CONTROLLED_REPORT.md) found an
exploratory advantage over random and predictive entropy at a budget of 10
queries, while all corresponding intervals crossed zero at 30 and 60 queries.
It is a semi-synthetic public-image experiment, not a real-user study.

## Grounded multimodal RAG endpoint

The current frontend does not wrap the RAG endpoint. After **语义索引** and the
pinned local Qwen3-VL snapshot are ready, call
`POST /albums/{album_id}/rag` directly. The complete install, request, response,
and failure contract is in
[grounded-multimodal-rag.md](grounded-multimodal-rag.md).

RAG is serialized per Python process and bounded to Top-K 1–6, 128 MiB of
encoded evidence, 64 megapixels per image, and 96 megapixels total. The local
Qwen runtime creates patch-aligned model-input derivatives under a deterministic
3,840-visual-token preflight budget, then verifies the processor's actual
`image_grid_thw` count against a 4,096-token hard limit. A violation returns
`413`; the original evidence bytes and SHA-256 identity are not changed by this
resize. These are admission/decoding safeguards, not claims about semantic
answer correctness.
The response explicitly reports `validation_level="citation-referential-only"`
and `semantic_entailment_verified=false`.

## Face model download, privacy, and fallback

The default `opencv-yunet-sface` provider uses YuNet 2023mar with a maximum
detection-preview side of 1600 pixels and score threshold 0.8. SFace consumes
YuNet's five landmarks through OpenCV `alignCrop`; its 128-dimensional output is
L2-normalized before clustering. Grouping uses strict high-confidence seeds plus
a guarded prototype-attachment pass for pose-fragmented groups. The second pass
requires mutual-best candidates and separate centroid, mean, and strongest-pair
evidence; the same-photo cannot-link applies throughout. This prototype pass is
a versioned experimental organizer policy, not a biometric guarantee. To keep
YuNet/SFace while prioritizing split groups over any extra merge risk, start with
`NORMA_FACE_PROVIDER=opencv-yunet-sface-strict`.

The first click on **人脸分组** downloads the pinned YuNet and SFace ONNX files,
about 37 MB combined. Each file must match its fixed SHA-256 before an atomic
rename makes it available in the local model cache. By default the files live
under `.norma/data/models/opencv/` and subsequent runs do not download them
again. Only public model files are fetched: Norma never uploads source photos,
thumbnails, face crops, or descriptors.

The OpenCV Zoo [YuNet files use the MIT license](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE),
and [SFace files use Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE).
For an offline run without those model files, select the legacy lower-quality
Haar/DCT fallback explicitly before starting the service:

```powershell
$env:NORMA_FACE_PROVIDER = "opencv-haar"
python -m ai web
```

People results are provider-versioned. The website compares the album's saved
people provider with the worker's active provider; results from Haar/DCT or an
older model/alignment/clustering revision show **重新运行** and are not loaded as
ready groups.

## Frontend development

```powershell
# terminal 1
python -m ai serve

# terminal 2
pnpm dev
```

Open `http://127.0.0.1:1420`. Vite proxies `/health`, `/albums`, `/selections`,
`/jobs`, `/feedback`, `/preferences`, `/providers`, `/evaluation`,
`/maintenance`, and `/media` to FastAPI at port 8765.

The backend also exposes a persistent catalog and background preparation API;
see [backend-library-lifecycle.md](backend-library-lifecycle.md).

The default semantic backend is multilingual OpenCLIP. After its immutable,
pinned local snapshot has been installed as documented in
[`multimodal-provider.md`](multimodal-provider.md), on the first
**语义索引** run the website shows a separate model-loading state while the
verified public model files are loaded locally; only after warmup succeeds
does the per-photo embedding progress begin. Existing vectors from another
provider are shown as stale and require a new semantic-index run. Use
`NORMA_EMBEDDING_PROVIDER=lightweight` only for the explicit zero-download
handcrafted baseline. See [multimodal-provider.md](multimodal-provider.md).
The provider endpoint reports capabilities without forcing the model into
memory, while its status endpoint is authoritative for load success.

The relevance-labeling and metric APIs are available for a future evaluation
view and for direct browser/API tooling. See
[retrieval-evaluation.md](retrieval-evaluation.md) for the endpoint workflow and
label semantics.

Run the checks before rebuilding committed assets:

```powershell
pnpm test:ui
pnpm build
python -m pytest ai/tests -q
```
