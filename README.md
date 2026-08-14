# Norma

Norma is a local-first **Personal Multimodal Photo Agent** built as a portfolio
demo for multimodal selection, retrieval, preference learning, video, and
interactive world generation.

The current implementation includes the Milestone 0 foundation, the Milestone
1 CPU library pipeline, Milestone 2 retrieval/people baselines, and the first
Milestone 3 structured selection path: recursive
JPG/JPEG scan, local thumbnails, quality signals, perceptual-hash similarity
groups, cached 16-dimensional semantic descriptors, and local text/image
retrieval. Source photos are always read-only.

## Development

Requirements:

- Python 3.11+
- Node.js 20+ and pnpm
- Rust stable and the Windows prerequisites required by Tauri

```powershell
python -m pip install -e ".[dev]"
pnpm install
pnpm test:ai
pnpm build:ui
pnpm dev
```

Run the worker without the desktop shell:

```powershell
python -m uvicorn ai.app:app --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/health`.

Index a folder directly through the local API:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/albums/index `
  -ContentType application/json `
  -Body '{"folder":"C:\\path\\to\\jpg-album"}'
```

Use the returned `album_id` to build the semantic cache and search it:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/albums/ALBUM_ID/embed

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/albums/search `
  -ContentType application/json `
  -Body '{"album_id":"ALBUM_ID","query":"夜景 blue city","limit":20}'
```

The default `lightweight-semantic-v1` provider is a deterministic CPU baseline,
not a pretrained vision-language model. It keeps the provider/cache/API
boundary testable while larger model dependencies remain optional.

Build conservative local people groups for an indexed album:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/albums/ALBUM_ID/people/index
```

The default `opencv-haar-dct-v1` provider is a face-detection and regression
baseline, not a biometric identity system. It intentionally uses a high
similarity threshold to reduce false merges.

Create an auditable collection selection from natural-language hard constraints:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/selections `
  -ContentType application/json `
  -Body '{"album_id":"ALBUM_ID","prompt":"选 12 张夜景，质量至少 45，相似组最多 1 张"}'
```

Install `.[selection]` to use OR-Tools CP-SAT. Without it, Norma uses a
deterministic optimizer that is exact for the current cardinality and
per-similarity-group capacity constraints.

If no local album is available, download a reproducible 72-image Wikimedia
Commons demo album. Image files remain untracked; `ATTRIBUTION.json` records
the source page, creator and per-file license metadata.

```powershell
python scripts/download_demo_album.py --count 72 --output .norma/demo-album
```

For deterministic reject/similarity validation, create a separate evaluation
copy with exact duplicates and controlled degraded derivatives. The downloaded
source album is not modified.

```powershell
python scripts/build_demo_eval_album.py
```

For people-pipeline validation, download a separate public portrait fixture and
create an evaluation copy with four controlled exact duplicates:

```powershell
python scripts/download_demo_album.py --count 30 `
  --output .norma/demo-portraits `
  --search "portrait face photograph" `
  --search "headshot portrait photography" `
  --search "people portrait travel"
python scripts/build_demo_people_eval_album.py
```

See [docs/architecture.md](docs/architecture.md) for process and data
boundaries. Third-party attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Public-fixture results are recorded in
[docs/benchmarks/retrieval-e2e.md](docs/benchmarks/retrieval-e2e.md).
People-pipeline evidence is in
[docs/benchmarks/people-e2e.md](docs/benchmarks/people-e2e.md).
Structured-selection evidence is in
[docs/benchmarks/selection-e2e.md](docs/benchmarks/selection-e2e.md).
Preference/replacement evidence is in
[docs/benchmarks/preference-replacement-e2e.md](docs/benchmarks/preference-replacement-e2e.md).

Record a local pairwise preference and request a locked-set replacement:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/feedback/pairwise `
  -ContentType application/json `
  -Body '{"album_id":"ALBUM_ID","preferred_photo_id":"PHOTO_A","rejected_photo_id":"PHOTO_B","selection_id":"SELECTION_ID"}'

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/selections/SELECTION_ID/replace `
  -ContentType application/json `
  -Body '{"remove_photo_id":"PHOTO_ID"}'
```

Preferences remain local in SQLite. Each selected photo exposes semantic,
quality, hard-constraint, and learned-preference evidence rather than a generic
AI explanation.
