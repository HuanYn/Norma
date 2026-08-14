# Norma

Norma is a local-first photo intelligence website. A Python service reads a
local JPG/JPEG folder, creates disposable thumbnails and indexes, and serves a
Vue web interface in the browser. Original photos are never moved, deleted, or
uploaded.

The current MVP supports:

- recursive album indexing, quality signals, suggested rejects, and similar-photo groups;
- local semantic text/image retrieval and conservative people grouping;
- bilingual natural-language selection with explicit hard constraints;
- OR-Tools CP-SAT optimization, auditable reasons, locked replacement, and pairwise preference learning;
- one local website and one SQLite database, with no desktop runtime required.

## Run the website

Requirements: Python 3.11+ and Node.js/pnpm for the one-time frontend build.

```powershell
python -m pip install -e ".[dev,selection]"
pnpm install
pnpm build
python -m ai web
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Enter an absolute local
folder path such as `D:\Photos\Trip`, then let Norma prepare the album. The
browser and API share the same local origin; the photo folder stays on the same
machine.

After the frontend has been built once, normal use only needs:

```powershell
python -m ai web
```

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

If no local album is available, download a reproducible Wikimedia Commons
fixture. Image files stay untracked and `ATTRIBUTION.json` records their source,
creator, and license.

```powershell
python scripts/download_demo_album.py --count 72 --output .norma/demo-album
python scripts/build_demo_eval_album.py
```

For people-pipeline validation:

```powershell
python scripts/download_demo_album.py --count 30 `
  --output .norma/demo-portraits `
  --search "portrait face photograph" `
  --search "headshot portrait photography"
python scripts/build_demo_people_eval_album.py
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
- [People benchmark](docs/benchmarks/people-e2e.md)
- [Selection benchmark](docs/benchmarks/selection-e2e.md)
- [Preference/replacement benchmark](docs/benchmarks/preference-replacement-e2e.md)
- [Direct Python benchmark](docs/benchmarks/python-cli-e2e.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

The default semantic and face providers are deterministic CPU integration
baselines, not claims of CLIP-level open-vocabulary retrieval or biometric
identity. Their provider boundaries are designed for later model upgrades.
