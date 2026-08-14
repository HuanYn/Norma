# Norma

Norma is a local-first **Personal Multimodal Photo Agent** built as a portfolio
demo for multimodal selection, retrieval, preference learning, video, and
interactive world generation.

The current implementation includes the Milestone 0 foundation, the Milestone
1 CPU library pipeline, and the first Milestone 2 retrieval baseline: recursive
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

See [docs/architecture.md](docs/architecture.md) for process and data
boundaries. Third-party attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Public-fixture results are recorded in
[docs/benchmarks/retrieval-e2e.md](docs/benchmarks/retrieval-e2e.md).
