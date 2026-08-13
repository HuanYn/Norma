# Norma

Norma is a local-first **Personal Multimodal Photo Agent** built as a portfolio
demo for multimodal selection, retrieval, preference learning, video, and
interactive world generation.

The current implementation is focused on Milestone 0: the Vue/Tauri desktop
shell starts a single FastAPI worker, verifies it over loopback HTTP, and
initializes a maintainable SQLite schema. Source photos are always read-only.

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

See [docs/architecture.md](docs/architecture.md) for process and data
boundaries. Third-party attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

