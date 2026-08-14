# Python CLI E2E — 2026-08-14

This check executes domain services directly through `python -m ai`; no HTTP
server, Tauri process, Node.js, or Rust toolchain is involved.

## Public fixture preparation

```powershell
python -m ai --data-dir .norma\cli-e2e --pretty `
  prepare .norma\demo-album-eval --skip-people
```

| Metric | Result |
| --- | ---: |
| Input JPEGs | 81 |
| Suggested rejects | 4 |
| Similarity groups | 7 |
| Index duration | 10,003 ms |
| Embedded photos | 81 |
| Embedding duration | 6,765 ms |
| Errors | 0 |

Pillow emitted the previously documented truncated EXIF/TIFF metadata warning
for one decodable public JPEG. The file still indexed successfully.

## Direct search

`search ... "night dark city" --limit 5` returned valid JSON. The top three
source files correspond to New York City at night HDR, Wellington City Night,
and a Sydney/Darling Harbour city image. A controlled dark reject also appeared
in raw retrieval, as expected: search exposes similarity, while collection
selection owns reject constraints.

## Direct selection

`select ... "Select 5 photos of night, quality at least 45, maximum 1 per
similarity group"` returned:

- `feasible=true`;
- 5/5 selected photos from 77 eligible candidates;
- `ortools-cp-sat / optimal`;
- minimum selected quality 48.635;
- 920 ms service duration;
- per-photo semantic, quality, and constraint reasons.

## Automated gates

The CLI test suite builds a temporary three-JPEG album and exercises `prepare`,
`albums`, `photos`, `search`, and `select` in one Python process. It also checks
that domain failures return UTF-8 JSON on stderr with a non-zero exit code.
