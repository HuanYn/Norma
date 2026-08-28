# Norma local website

## Production-style local run

Build the Vue assets once, then let Python serve the complete website:

```powershell
python -m pip install -e ".[dev,selection]"
pnpm install
pnpm build
python -m ai web
```

Open `http://127.0.0.1:8765`. The web assets, JSON API, thumbnails, and face
crops use the same local origin.

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
6. For a selection, inspect per-photo reasons, replace one item, or mark one
   photo preferred and another less preferred.

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

The default lightweight embedding backend requires no model download. To use
the optional multilingual OpenCLIP backend, follow
[multimodal-provider.md](multimodal-provider.md). The provider endpoint exposes
capabilities without forcing the model into memory.

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
