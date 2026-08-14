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
2. Select **Open local folder**. Norma indexes the folder read-only, builds the
   semantic cache, and runs conservative face grouping.
3. Review normal photos and the folded AI-suggested exclusions.
4. Open **AI Selection** and enter either a semantic query such as `night blue`
   or a constrained request such as `选 12 张夜景，质量至少 45`.
5. For a selection, inspect per-photo reasons, replace one item, or mark one
   photo preferred and another less preferred.

Runtime state defaults to `.norma/data`. Override it before the `web` command:

```powershell
python -m ai --data-dir D:\NormaData web --port 8879
```

## Frontend development

```powershell
# terminal 1
python -m ai serve

# terminal 2
pnpm dev
```

Open `http://127.0.0.1:1420`. Vite proxies `/health`, `/albums`, `/selections`,
`/jobs`, `/feedback`, `/preferences`, `/providers`, and `/media` to FastAPI at
port 8765.

The backend also exposes a persistent catalog and background preparation API;
see [backend-library-lifecycle.md](backend-library-lifecycle.md).

The default lightweight embedding backend requires no model download. To use
the optional multilingual OpenCLIP backend, follow
[multimodal-provider.md](multimodal-provider.md). The provider endpoint exposes
capabilities without forcing the model into memory.

Run the checks before rebuilding committed assets:

```powershell
pnpm test:ui
pnpm build
python -m pytest ai/tests -q
```
